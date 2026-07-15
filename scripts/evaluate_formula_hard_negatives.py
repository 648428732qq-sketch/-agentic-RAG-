from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


for thread_env in (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
from scripts.run_chatmed_gold_eval import install_local_qdrant_grpc_stub_if_blocked  # noqa: E402

from core.syndrome_retriever import SyndromeRetriever, should_use_structured_answer  # noqa: E402
from core.symptom_query_translator import translate_symptom_query  # noqa: E402
from core.syndrome_terms import clean_text, unique  # noqa: E402


DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_CASES = ROOT / "tests" / "evals" / "formula_hard_negative_broad" / "cases.jsonl"
DEFAULT_PREDICTIONS = ROOT / "datasets" / "structured" / "formula_hard_negative_broad_predictions.jsonl"
DEFAULT_REPORT = ROOT / "datasets" / "structured" / "formula_hard_negative_broad_report.json"

MANDATORY_FORMULAS = ("麻黄汤", "桂枝汤", "小青龙汤", "苏子降气汤")
CASE_STYLES = (
    "full_signature_topk",
    "missing_required_clarify",
    "forbidden_conflict",
    "shared_terms_clarify",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            rows.append(json.loads(raw_line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _list_values(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if isinstance(value, list):
        return [clean_text(str(item)) for item in value if clean_text(str(item))]
    if value:
        return [clean_text(str(value))]
    return []


def _required_groups(payload: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    for raw_group in payload.get("required_symptom_groups", []) or []:
        if isinstance(raw_group, list):
            group = [clean_text(str(term)) for term in raw_group if clean_text(str(term))]
        else:
            group = [clean_text(str(raw_group))]
        group = [term for term in unique(group) if 1 <= len(term) <= 16]
        if group:
            groups.append(group)
    return groups


def _flatten_required_groups(payload: dict[str, Any]) -> list[str]:
    return unique(term for group in _required_groups(payload) for term in group)


def _rankable_terms(values: list[str], *, max_len: int = 16, min_len: int = 1) -> list[str]:
    selected: list[str] = []
    for value in values:
        value = clean_text(str(value))
        if not value:
            continue
        if len(value) < min_len:
            continue
        if len(value) > max_len:
            continue
        if value in selected:
            continue
        selected.append(value)
    return selected


def formula_richness(payload: dict[str, Any]) -> int:
    return (
        len(_required_groups(payload)) * 5
        + len(_list_values(payload, "differential_keys")) * 2
        + len(_list_values(payload, "diagnostic_keys"))
        + len(_list_values(payload, "forbidden_terms")) * 3
    )


def formula_signature_terms(payload: dict[str, Any]) -> set[str]:
    terms: list[str] = []
    terms.extend(_flatten_required_groups(payload))
    terms.extend(_list_values(payload, "differential_keys"))
    terms.extend(_list_values(payload, "diagnostic_keys"))
    terms.extend(_list_values(payload, "ancient_symptoms"))
    return {term for term in _rankable_terms(terms, max_len=16) if term != payload.get("formula")}


def pick_best_formula_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_formula: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("source_type") != "formula_syndrome":
            continue
        formula = clean_text(str(row.get("formula", "")))
        if not formula:
            continue
        if not _required_groups(row):
            continue
        current = best_by_formula.get(formula)
        if current is None or formula_richness(row) > formula_richness(current):
            best_by_formula[formula] = row
    return list(best_by_formula.values())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def attach_nearest_neighbors(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    signatures = {entry["formula"]: formula_signature_terms(entry) for entry in entries}
    neighbors: dict[str, dict[str, Any]] = {}
    for entry in entries:
        formula = entry["formula"]
        ranked: list[dict[str, Any]] = []
        for other in entries:
            other_formula = other["formula"]
            if other_formula == formula:
                continue
            shared = sorted(signatures[formula] & signatures[other_formula], key=lambda item: (-len(item), item))
            if not shared:
                continue
            ranked.append(
                {
                    "formula": other_formula,
                    "similarity": round(_jaccard(signatures[formula], signatures[other_formula]), 4),
                    "shared_terms": shared[:8],
                }
            )
        ranked.sort(key=lambda item: (item["similarity"], len(item["shared_terms"]), item["formula"]), reverse=True)
        if ranked:
            neighbors[formula] = ranked[0]
    return neighbors


def select_formula_entries(entries: list[dict[str, Any]], max_formulas: int) -> list[dict[str, Any]]:
    neighbors = attach_nearest_neighbors(entries)
    ranked = sorted(
        entries,
        key=lambda entry: (
            entry["formula"] in MANDATORY_FORMULAS,
            bool(neighbors.get(entry["formula"])),
            len((neighbors.get(entry["formula"]) or {}).get("shared_terms", [])),
            formula_richness(entry),
            entry["formula"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for formula in MANDATORY_FORMULAS:
        for entry in entries:
            if entry["formula"] == formula and formula not in seen:
                selected.append(entry)
                seen.add(formula)
                break
    for entry in ranked:
        if len(selected) >= max_formulas:
            break
        formula = entry["formula"]
        if formula in seen:
            continue
        selected.append(entry)
        seen.add(formula)
    return selected


def _pick_one_from_group(group: list[str], *, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    candidates = [term for term in group if term not in excluded] or group
    return sorted(candidates, key=lambda term: (len(term) == 1, len(term), term))[0]


def _term_overlaps_any(term: str, terms: set[str]) -> bool:
    term = clean_text(term)
    if not term:
        return False
    return any(other and (term == other or term in other or other in term) for other in terms)


def _candidate_terms_for_payload(payload: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(_flatten_required_groups(payload))
    for key in ("diagnostic_keys", "ancient_symptoms", "modern_symptoms", "pathogenesis", "differential_keys"):
        terms.extend(_list_values(payload, key))
    return _rankable_terms(unique(terms), max_len=16)


def _terms_leak_into_targets(
    query_terms: list[str],
    target_terms: list[str],
    payload: dict[str, Any],
) -> bool:
    targets = set(_rankable_terms(target_terms, max_len=16))
    if not targets:
        return False
    if any(_term_overlaps_any(term, targets) for term in query_terms):
        return True
    translated = translate_symptom_query(
        _join_terms(query_terms),
        candidate_terms=_candidate_terms_for_payload(payload),
    )
    positive_terms = set(translated.get("canonical_terms", [])) | set(translated.get("primary_canonical_terms", []))
    return bool(positive_terms & targets)


def _omitted_group_leaks_into_terms(
    query_terms: list[str],
    omitted_group: list[str],
    payload: dict[str, Any],
) -> bool:
    return _terms_leak_into_targets(query_terms, omitted_group, payload)


def _pick_query_terms(payload: dict[str, Any], *, omit_group_index: int | None = None, limit: int = 7) -> list[str]:
    terms: list[str] = []
    groups = _required_groups(payload)
    forbidden_terms = set(_rankable_terms(_list_values(payload, "forbidden_terms"), max_len=16))
    omitted_terms: set[str] = set()
    if omit_group_index is not None and 0 <= omit_group_index < len(groups):
        omitted_terms = set(groups[omit_group_index])
    excluded_extras = forbidden_terms | omitted_terms | {str(payload.get("formula", ""))}
    for index, group in enumerate(groups):
        if omit_group_index is not None and index == omit_group_index:
            continue
        picked = _pick_one_from_group(group, excluded=forbidden_terms | omitted_terms)
        if picked and not _term_overlaps_any(picked, omitted_terms):
            terms.append(picked)
    positive_support = []
    for key in ("diagnostic_keys", "ancient_symptoms", "modern_symptoms", "pathogenesis"):
        positive_support.extend(_list_values(payload, key))
    for term in _rankable_terms(positive_support, max_len=12, min_len=2):
        if term not in terms and term not in excluded_extras and not _term_overlaps_any(term, omitted_terms):
            terms.append(term)
        if len(terms) >= limit:
            break
    return unique(terms)[:limit]


def _pick_missing_required_case(
    payload: dict[str, Any],
    required: list[list[str]],
    *,
    limit: int = 6,
) -> tuple[int, list[str]] | None:
    ranked_indexes = sorted(range(len(required)), key=lambda idx: (len(required[idx]), idx), reverse=True)
    forbidden_terms = _rankable_terms(_list_values(payload, "forbidden_terms"), max_len=16)
    for omit_index in ranked_indexes:
        missing_terms = _pick_query_terms(payload, omit_group_index=omit_index, limit=limit)
        if not missing_terms:
            continue
        if _omitted_group_leaks_into_terms(missing_terms, required[omit_index], payload):
            continue
        if _terms_leak_into_targets(missing_terms, forbidden_terms, payload):
            continue
        return omit_index, missing_terms
    return None


def _join_terms(terms: list[str]) -> str:
    return "、".join(term for term in terms if term)


def _first_forbidden_term(payload: dict[str, Any], positive_terms: list[str]) -> str:
    positive_set = set(positive_terms)
    candidates = [
        term
        for term in _rankable_terms(_list_values(payload, "forbidden_terms"), max_len=12)
        if term not in positive_set
    ]
    candidates.sort(key=lambda term: (term.startswith("有"), len(term), term))
    for term in candidates:
        if term not in positive_set:
            return term
    return ""


def generate_cases(entries: list[dict[str, Any]], *, max_formulas: int) -> list[dict[str, Any]]:
    selected = select_formula_entries(entries, max_formulas=max_formulas)
    neighbors = attach_nearest_neighbors(selected)
    cases: list[dict[str, Any]] = []
    case_index = 1
    for entry in selected:
        formula = entry["formula"]
        required = _required_groups(entry)
        if len(required) < 2:
            continue
        full_terms = _pick_query_terms(entry, limit=7)
        if len(full_terms) >= 2:
            cases.append(
                {
                    "id": f"formula_hn_{case_index:04d}",
                    "style": "full_signature_topk",
                    "query": f"我现在{_join_terms(full_terms)}，从本地方证库看最接近哪类方证？",
                    "expected_formula": formula,
                    "expected_terms": full_terms,
                    "source_entry_id": entry.get("entry_id", ""),
                    "source_title": entry.get("title", ""),
                    "nearest_neighbor": neighbors.get(formula, {}),
                }
            )
            case_index += 1

        missing_case = _pick_missing_required_case(entry, required, limit=6)
        if missing_case is not None:
            omit_index, missing_terms = missing_case
            cases.append(
                {
                    "id": f"formula_hn_{case_index:04d}",
                    "style": "missing_required_clarify",
                    "query": f"只有{_join_terms(missing_terms)}这些表现，信息够不够直接判断用哪类方证？",
                    "expected_formula": formula,
                    "expected_terms": missing_terms,
                    "omitted_required_group": required[omit_index],
                    "source_entry_id": entry.get("entry_id", ""),
                    "source_title": entry.get("title", ""),
                    "nearest_neighbor": neighbors.get(formula, {}),
                }
            )
            case_index += 1

        forbidden = _first_forbidden_term(entry, full_terms)
        if forbidden and len(full_terms) >= 2:
            conflict_terms = full_terms[:5] + [forbidden]
            cases.append(
                {
                    "id": f"formula_hn_{case_index:04d}",
                    "style": "forbidden_conflict",
                    "query": f"我有{_join_terms(full_terms[:5])}，但是也出现{forbidden}，还能直接按这个方证处理吗？",
                    "expected_formula": formula,
                    "expected_terms": conflict_terms,
                    "forbidden_term": forbidden,
                    "source_entry_id": entry.get("entry_id", ""),
                    "source_title": entry.get("title", ""),
                    "nearest_neighbor": neighbors.get(formula, {}),
                }
            )
            case_index += 1

        neighbor = neighbors.get(formula)
        shared_terms = [term for term in (neighbor or {}).get("shared_terms", []) if term not in {formula}]
        if neighbor and len(shared_terms) >= 2:
            terms = shared_terms[:4]
            cases.append(
                {
                    "id": f"formula_hn_{case_index:04d}",
                    "style": "shared_terms_clarify",
                    "query": f"我就说得清楚{_join_terms(terms)}，这能不能直接锁定一个方证？",
                    "expected_formula": formula,
                    "expected_terms": terms,
                    "contrast_formula": neighbor["formula"],
                    "shared_terms": terms,
                    "source_entry_id": entry.get("entry_id", ""),
                    "source_title": entry.get("title", ""),
                    "nearest_neighbor": neighbor,
                }
            )
            case_index += 1
    return cases


def _top_formulas(matches: list[dict[str, Any]]) -> list[str]:
    return [str(match.get("payload", {}).get("formula", "")) for match in matches]


def _target_matches(matches: list[dict[str, Any]], formula: str) -> list[dict[str, Any]]:
    return [match for match in matches if match.get("payload", {}).get("formula") == formula]


def evaluate_case(retriever: SyndromeRetriever, case: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = retriever.search(case["query"], limit=top_k, candidate_limit=max(top_k * 10, 80))
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    matches = result.get("matches", [])
    formulas = _top_formulas(matches)
    expected_formula = str(case.get("expected_formula", ""))
    target_matches = _target_matches(matches, expected_formula)
    top_formula = formulas[0] if formulas else ""
    decision = result.get("decision", {})
    status = str(decision.get("status", ""))
    gate = should_use_structured_answer(result)
    target_topk = bool(expected_formula and expected_formula in formulas)
    target_top1 = bool(expected_formula and top_formula == expected_formula)
    target_missing_required = any(
        bool(match.get("missing_required_symptom_groups")) for match in target_matches
    )
    target_forbidden_conflict = any(
        case.get("forbidden_term") in match.get("forbidden_conflicts", []) for match in target_matches
    )
    unsafe_target_grounded = gate and status == "grounded_answer" and target_top1
    style = case["style"]
    checks: dict[str, bool] = {
        "route_ok": not result.get("retrieval_debug", {}).get("errors"),
    }
    if style == "full_signature_topk":
        checks.update(
            {
                "target_in_topk": target_topk,
                "not_no_match": status != "no_match",
            }
        )
    elif style == "missing_required_clarify":
        checks.update(
            {
                "target_in_topk": target_topk,
                "not_grounded": status != "grounded_answer" and not gate,
                "missing_required_detected": target_missing_required
                or "missing_required_symptom_groups" in decision.get("reasons", []),
            }
        )
    elif style == "forbidden_conflict":
        checks.update(
            {
                "no_unsafe_target_grounded": not unsafe_target_grounded,
                "target_not_top1": not target_top1,
                "target_filtered_or_conflicted": (not target_topk) or target_forbidden_conflict,
            }
        )
    elif style == "shared_terms_clarify":
        checks.update(
            {
                "not_grounded": status != "grounded_answer" and not gate,
                "has_competing_context": bool(case.get("contrast_formula"))
                and (
                    case.get("contrast_formula") in formulas
                    or target_topk
                    or "competing_indistinguishable_matches" in decision.get("reasons", [])
                    or status in {"clarify", "no_match"}
                ),
            }
        )
    else:
        raise ValueError(f"unknown case style: {style}")

    return {
        "id": case["id"],
        "style": style,
        "query": case["query"],
        "expected_formula": expected_formula,
        "contrast_formula": case.get("contrast_formula", ""),
        "ok": all(checks.values()),
        "checks": checks,
        "latency_ms": latency_ms,
        "decision": decision,
        "gate": gate,
        "top_formula": top_formula,
        "target_rank": formulas.index(expected_formula) + 1 if target_topk else None,
        "top_k_formulas": formulas,
        "canonical_terms": result.get("query", {}).get("canonical_terms", []),
        "candidate_terms": result.get("query", {}).get("candidate_terms", []),
        "negative_terms": result.get("query", {}).get("negative_terms", []),
        "expected_terms": case.get("expected_terms", []),
        "omitted_required_group": case.get("omitted_required_group", []),
        "forbidden_term": case.get("forbidden_term", ""),
        "nearest_neighbor": case.get("nearest_neighbor", {}),
        "matches": [
            {
                "formula": match.get("payload", {}).get("formula", ""),
                "title": match.get("payload", {}).get("title", ""),
                "source_type": match.get("payload", {}).get("source_type", ""),
                "matched_terms": match.get("matched_terms", []),
                "missing_required_symptom_groups": match.get("missing_required_symptom_groups", []),
                "forbidden_conflicts": match.get("forbidden_conflicts", []),
                "required_group_coverage": match.get("required_group_coverage", 0.0),
                "differential_coverage": match.get("differential_coverage", 0.0),
                "query_coverage": match.get("query_coverage", 0.0),
                "canonical_match_count": match.get("canonical_match_count", 0),
            }
            for match in matches
        ],
        "retrieval_debug": result.get("retrieval_debug", {}),
    }


def summarize(
    results: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    formula_count: int,
    mode: str = "local_payload_hard_rank_no_rerank",
) -> dict[str, Any]:
    by_style: dict[str, dict[str, Any]] = {}
    for style in CASE_STYLES:
        subset = [result for result in results if result["style"] == style]
        if not subset:
            continue
        check_names = sorted({name for result in subset for name in result["checks"]})
        by_style[style] = {
            "count": len(subset),
            "passed": sum(1 for result in subset if result["ok"]),
            "pass_rate": round(sum(1 for result in subset if result["ok"]) / len(subset), 4),
            "checks": {
                name: round(sum(1 for result in subset if result["checks"].get(name)) / len(subset), 4)
                for name in check_names
            },
        }
    thresholds = {
        "full_signature_topk.target_in_topk": 0.9,
        "missing_required_clarify.not_grounded": 0.95,
        "forbidden_conflict.no_unsafe_target_grounded": 1.0,
        "shared_terms_clarify.not_grounded": 0.95,
    }
    threshold_results: dict[str, float] = {}
    for key in thresholds:
        style, check = key.split(".", 1)
        threshold_results[key] = by_style.get(style, {}).get("checks", {}).get(check, 0.0)
    failed_thresholds = {
        key: {"actual": threshold_results[key], "threshold": threshold}
        for key, threshold in thresholds.items()
        if threshold_results[key] < threshold
    }
    latencies = sorted(result["latency_ms"] for result in results)
    formulas = sorted({case.get("expected_formula", "") for case in cases if case.get("expected_formula")})
    top_failure_reasons = Counter()
    for result in results:
        if result["ok"]:
            continue
        for name, ok in result["checks"].items():
            if not ok:
                top_failure_reasons[f"{result['style']}.{name}"] += 1
    rerank_debug = [
        result.get("retrieval_debug", {}).get("rerank", {})
        for result in results
    ]
    rerank_used_count = sum(1 for item in rerank_debug if item.get("rerank_used"))
    rerank_errors = Counter(
        str(item.get("error"))
        for item in rerank_debug
        if item.get("error")
    )
    return {
        "ok": not failed_thresholds,
        "mode": mode,
        "case_count": len(results),
        "passed": sum(1 for result in results if result["ok"]),
        "pass_rate": round(sum(1 for result in results if result["ok"]) / len(results), 4) if results else 0.0,
        "formula_count": formula_count,
        "tested_formula_count": len(formulas),
        "tested_formulas_sample": formulas[:40],
        "by_style": by_style,
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "failed_thresholds": failed_thresholds,
        "failure_counts": dict(top_failure_reasons.most_common(20)),
        "rerank": {
            "used_count": rerank_used_count,
            "usage_rate": round(rerank_used_count / len(results), 4) if results else 0.0,
            "errors": dict(rerank_errors),
        },
        "latency_ms": {
            "average": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "failures": [result for result in results if not result["ok"]][:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run broad similar-formula hard-negative retrieval checks with local payload ranking."
    )
    parser.add_argument("--dictionary", default=str(DEFAULT_DICTIONARY))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--max-formulas", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--with-rerank", action="store_true")
    parser.add_argument("--rerank-candidates", type=int, default=8)
    parser.add_argument("--rerank-device", default="cpu")
    parser.add_argument("--rerank-max-length", type=int, default=256)
    parser.add_argument("--rerank-mode", choices=("evidence_first", "score_first"), default="evidence_first")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dump-stacks-after", type=int, default=0)
    args = parser.parse_args()

    if args.dump_stacks_after > 0:
        faulthandler.dump_traceback_later(args.dump_stacks_after, repeat=True)

    install_local_qdrant_grpc_stub_if_blocked()
    config.ENABLE_LLM_SYMPTOM_TRANSLATOR = False
    config.ENABLE_SYNDROME_RERANK = bool(args.with_rerank)
    config.SYNDROME_RERANK_CANDIDATES = max(2, args.rerank_candidates)
    config.SYNDROME_RERANK_DEVICE = args.rerank_device
    config.SYNDROME_RERANK_MAX_LENGTH = max(64, args.rerank_max_length)
    config.SYNDROME_RERANK_MODE = args.rerank_mode

    predictions_path = Path(args.predictions)
    report_path = Path(args.report)
    if args.with_rerank:
        if predictions_path == DEFAULT_PREDICTIONS:
            predictions_path = DEFAULT_PREDICTIONS.with_name("formula_hard_negative_broad_rerank_predictions.jsonl")
        if report_path == DEFAULT_REPORT:
            report_path = DEFAULT_REPORT.with_name("formula_hard_negative_broad_rerank_report.json")

    dictionary_path = Path(args.dictionary)
    rows = read_jsonl(dictionary_path)
    formula_entries = pick_best_formula_entries(rows)
    cases = generate_cases(formula_entries, max_formulas=max(1, args.max_formulas))
    if not cases:
        raise RuntimeError(f"no hard-negative cases generated from {dictionary_path}")
    write_jsonl(Path(args.cases), cases)

    retriever = SyndromeRetriever()
    results = [evaluate_case(retriever, case, top_k=max(1, args.top_k)) for case in cases]
    write_jsonl(predictions_path, results)

    mode = "local_payload_hard_rank_with_rerank" if args.with_rerank else "local_payload_hard_rank_no_rerank"
    report = summarize(results, cases, formula_count=len(formula_entries), mode=mode)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_error and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
