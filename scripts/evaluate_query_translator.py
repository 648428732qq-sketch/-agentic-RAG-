from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
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

from scripts.run_chatmed_gold_eval import install_local_qdrant_grpc_stub_if_blocked  # noqa: E402

from core.syndrome_retriever import SyndromeRetriever, should_use_structured_answer  # noqa: E402


DEFAULT_CASES = ROOT / "tests" / "evals" / "query_translator_cases.jsonl"
DEFAULT_REPORT = ROOT / "datasets" / "structured" / "query_translator_eval.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            cases.append(json.loads(raw_line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return cases


def top_text(payload: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "title",
        "syndrome_name",
        "formula",
        "intervention_name",
        "herb_name",
        "source_book",
        "evidence",
    ):
        if payload.get(key):
            values.append(str(payload[key]))
    for key in ("ancient_symptoms", "modern_symptoms", "diagnostic_keys", "theory_terms", "acupuncture_terms"):
        values.extend(str(value) for value in payload.get(key, []) if value)
    return " ".join(values)


def _term_groups(case: dict[str, Any]) -> list[set[str]]:
    if case.get("expected_term_groups"):
        return [set(group) for group in case["expected_term_groups"]]
    return [{term} for term in case.get("expected_terms", [])]


def _groups_covered(groups: list[set[str]], actual: set[str]) -> bool:
    return all(bool(group & actual) for group in groups)


def evaluate_case(retriever: SyndromeRetriever, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = retriever.search(case["query"], limit=5)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    query_info = result.get("query", {})
    matches = result.get("matches", [])
    payloads = [match.get("payload", {}) for match in matches]
    top_payload = payloads[0] if payloads else {}
    canonical_terms = set(query_info.get("canonical_terms", []))
    negative_terms = set(query_info.get("negative_terms", []))
    candidate_terms = set(query_info.get("candidate_terms", []))

    expected_groups = _term_groups(case)
    expected_negative = set(case.get("expected_negative_terms", []))
    forbidden = set(case.get("forbidden_terms", []))
    term_ok = _groups_covered(expected_groups, canonical_terms)
    candidate_ok = _groups_covered(expected_groups, candidate_terms | canonical_terms)
    candidate_ok = candidate_ok and expected_negative.issubset(candidate_terms | negative_terms)
    negative_ok = expected_negative.issubset(negative_terms)
    hallucination_ok = not (forbidden & canonical_terms)
    needs_more_info_ok = (
        "expected_needs_more_info" not in case
        or bool(query_info.get("needs_more_info")) is bool(case["expected_needs_more_info"])
    )

    retrieval_ok = True
    if case.get("expected_formula"):
        retrieval_ok = retrieval_ok and top_payload.get("formula") == case["expected_formula"]
    if case.get("expected_formula_in_top_k"):
        retrieval_ok = retrieval_ok and any(
            payload.get("formula") == case["expected_formula_in_top_k"]
            for payload in payloads
        )
    if case.get("expected_any_formula_in_top_k"):
        accepted_formulas = set(case["expected_any_formula_in_top_k"])
        retrieval_ok = retrieval_ok and any(payload.get("formula") in accepted_formulas for payload in payloads)
    if case.get("expected_source_type"):
        retrieval_ok = retrieval_ok and top_payload.get("source_type") == case["expected_source_type"]
    if case.get("expected_source_type_in_top_k"):
        retrieval_ok = retrieval_ok and any(
            payload.get("source_type") == case["expected_source_type_in_top_k"]
            for payload in payloads
        )
    if case.get("expected_intervention_text"):
        retrieval_ok = retrieval_ok and any(
            case["expected_intervention_text"] in top_text(payload)
            for payload in payloads
        )
    gate = should_use_structured_answer(result)
    gate_ok = "expected_gate" not in case or gate is bool(case["expected_gate"])
    decision_ok = (
        "expected_decision" not in case
        or result.get("decision", {}).get("status") == case["expected_decision"]
    )

    checks = {
        "candidate_coverage": candidate_ok,
        "term_recall": term_ok,
        "negative_terms": negative_ok,
        "no_forbidden_positive": hallucination_ok,
        "needs_more_info": needs_more_info_ok,
        "retrieval": retrieval_ok,
        "gate": gate_ok,
        "decision": decision_ok,
    }
    return {
        "id": case["id"],
        "query": case["query"],
        "ok": all(checks.values()),
        "checks": checks,
        "latency_ms": latency_ms,
        "translation_method": query_info.get("translation_method", ""),
        "canonical_terms": sorted(canonical_terms),
        "candidate_terms": sorted(candidate_terms),
        "negative_terms": sorted(negative_terms),
        "evidence_mappings": query_info.get("evidence_mappings", []),
        "unknown_phrases": query_info.get("unknown_phrases", []),
        "translation_errors": query_info.get("translation_errors", []),
        "gate": gate,
        "decision": result.get("decision", {}),
        "top": {
            "title": top_payload.get("title", ""),
            "source_type": top_payload.get("source_type", ""),
            "formula": top_payload.get("formula", ""),
            "intervention_name": top_payload.get("intervention_name", ""),
            "source_book": top_payload.get("source_book", ""),
        },
        "top_k": [
            {
                "title": payload.get("title", ""),
                "source_type": payload.get("source_type", ""),
                "formula": payload.get("formula", ""),
                "intervention_name": payload.get("intervention_name", ""),
                "canonical_match_count": match.get("canonical_match_count", 0),
                "matched_terms": match.get("matched_terms", []),
                "diagnostic_coverage": match.get("diagnostic_coverage", 0.0),
            }
            for match, payload in zip(matches, payloads)
        ],
        "retrieval_debug": result.get("retrieval_debug", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 Query Translator 的未见表达、否定和检索泛化能力")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--with-llm", action="store_true", help="调用当前配置的 LLM；默认只测本地降级链")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--case-id", action="append", default=None, help="只运行指定 case id，可重复传入")
    parser.add_argument("--dump-stacks-after", type=int, default=0, help="调试阻塞时定时输出线程栈")
    args = parser.parse_args()
    install_local_qdrant_grpc_stub_if_blocked()
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    if args.dump_stacks_after > 0:
        faulthandler.dump_traceback_later(args.dump_stacks_after, repeat=True)

    retriever = SyndromeRetriever()
    if args.with_llm:
        from core.llm_factory import create_query_translator_client

        retriever.set_llm(create_query_translator_client())

    cases = load_cases(Path(args.cases))
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
        missing = selected - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown case ids: {sorted(missing)}")
    results = [evaluate_case(retriever, case) for case in cases]
    latencies = sorted(item["latency_ms"] for item in results)
    check_names = sorted({name for item in results for name in item["checks"]})
    metrics = {
        name: round(sum(1 for item in results if item["checks"][name]) / len(results), 4)
        for name in check_names
    }
    report = {
        "ok": all(item["ok"] for item in results),
        "mode": "llm" if args.with_llm else "local_fallback",
        "case_count": len(results),
        "passed": sum(1 for item in results if item["ok"]),
        "metrics": metrics,
        "latency_ms": {
            "average": round(sum(latencies) / len(latencies), 2),
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "max": max(latencies),
        },
        "cases": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_error and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
