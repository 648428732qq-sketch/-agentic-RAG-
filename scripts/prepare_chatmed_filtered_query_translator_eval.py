from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILTERED_DIR = ROOT / "datasets" / "structured" / "chatmed_candidates" / "filtered"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "evals" / "query_translator_chatmed_filtered"

NEGATED_SWEAT_TERMS = {"出汗", "有汗", "汗出"}
NEGATED_TO_PRESENT_GROUPS = {
    "出汗": ["无汗"],
    "有汗": ["无汗"],
    "汗出": ["无汗"],
}

NON_DIAGNOSTIC_TERMS = {
    "饮食",
    "治疗",
    "中药",
    "方剂",
    "药物",
    "推荐",
    "建议",
    "症状",
    "疾病",
    "患者",
}
INVALID_NEGATION_TERMS = {
    "有力",  # “没有力气”应作为乏力/无力类正向表征，不是“有力”的可靠负例。
    "欲食",  # “不欲食”是古籍中的复合正向表征。
    "想喝水",  # 这是“欲饮水/口渴”的口语别名，不应作为规范负例。
}

UNCERTAIN_TERM_PATTERNS = (
    r"是不是{term}",
    r"是否{term}",
    r"能否{term}",
    r"有无{term}",
    r"有没有{term}",
    r"是{term}吗",
)

NEGATION_PREFIX_RE = re.compile(r"(?:没有|没|并不|不|未|无|勿)")


def _contains_uncertain_term(query: str, term: str) -> bool:
    escaped = re.escape(term)
    return any(re.search(pattern.format(term=escaped), query) for pattern in UNCERTAIN_TERM_PATTERNS)


def _is_explicit_negative_query(query: str, term: str) -> bool:
    if _contains_uncertain_term(query, term):
        return False
    escaped = re.escape(term)
    return bool(
        re.search(rf"(?:没有|没|并不|不|未|无|勿)[^，。；;、！？!?]{{0,8}}{escaped}", query)
        or re.search(rf"{escaped}[^，。；;、！？!?]{{0,4}}(?:没有|没|无|未)", query)
    )


def _is_negated_sweat_query(query: str) -> bool:
    return bool(
        re.search(
            r"(?:没有|没|无|未|不)[^，。；;、！？!?]{0,8}(?:汗|出汗|汗出)|(?:汗|汗出|出汗)[^，。；;、！？!?]{0,8}(?:没有|没|无|未|不出来|出不来)",
            query,
        )
    )


def _select_positive_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    term = str(row["term"])
    for sample in row.get("sample_queries", []):
        query = str(sample.get("query", ""))
        if term in query and not _contains_uncertain_term(query, term):
            return sample
    return None


def _select_negative_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    term = str(row["term"])
    if term in INVALID_NEGATION_TERMS:
        return None
    if term.startswith("有") and term not in NEGATED_SWEAT_TERMS:
        return None
    for sample in row.get("sample_queries", []):
        query = str(sample.get("query", ""))
        if term in NEGATED_SWEAT_TERMS and _is_negated_sweat_query(query):
            return sample
        if _is_explicit_negative_query(query, term):
            return sample
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def unique_cases(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["query"])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def diagnostic_cases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    cases: list[dict[str, Any]] = []
    for row in rows:
        term = str(row["term"])
        if term in NON_DIAGNOSTIC_TERMS:
            continue
        if row.get("priority") not in {"high", "medium"}:
            continue
        if "generic_symptom" in set(row.get("risk_flags", [])):
            continue
        sample = _select_positive_sample(row)
        if not sample:
            continue
        cases.append(
            {
                "id": f"chatmed_filtered_diag_{len(cases) + 1:04d}",
                "query": sample["query"],
                "source_queue": "diagnostic_term_review_queue",
                "candidate_term": term,
                "priority": row.get("priority", ""),
                "expected_term_groups": [[term]],
                "expected_negative_terms": [],
                "forbidden_terms": [],
                "review_status": "auto_generated_from_filtered_candidate_unverified",
            }
        )
        if len(cases) >= limit:
            break
    return cases


def pathogenesis_cases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    cases: list[dict[str, Any]] = []
    for row in rows:
        if row.get("priority") not in {"high", "medium"}:
            continue
        sample = _select_positive_sample(row)
        if not sample:
            continue
        cases.append(
            {
                "id": f"chatmed_filtered_pathogenesis_{len(cases) + 1:04d}",
                "query": sample["query"],
                "source_queue": "pathogenesis_query_review_queue",
                "candidate_term": row["term"],
                "priority": row.get("priority", ""),
                "expected_term_groups": [[row["term"]]],
                "expected_negative_terms": [],
                "forbidden_terms": [],
                "review_status": "auto_generated_from_filtered_candidate_unverified",
            }
        )
        if len(cases) >= limit:
            break
    return cases


def negation_cases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    cases: list[dict[str, Any]] = []
    for row in rows:
        sample = _select_negative_sample(row)
        if not sample:
            continue
        term = str(row["term"])
        case: dict[str, Any] = {
            "id": f"chatmed_filtered_negation_{len(cases) + 1:04d}",
            "query": sample["query"],
            "source_queue": "negation_forbidden_review_queue",
            "candidate_term": term,
            "priority": row.get("priority", ""),
            "expected_term_groups": [],
            "expected_negative_terms": [term],
            "forbidden_terms": [term],
            "review_status": "auto_generated_from_filtered_candidate_unverified",
        }
        if term in NEGATED_SWEAT_TERMS:
            case["expected_term_groups"] = [NEGATED_TO_PRESENT_GROUPS[term]]
            case["expected_negative_terms"] = []
            case["forbidden_terms"] = ["汗出", "出汗", "有汗"]
        cases.append(case)
        if len(cases) >= limit:
            break
    return cases


def source_hint_cases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    cases: list[dict[str, Any]] = []
    for row in rows:
        if row.get("priority") != "high":
            continue
        samples = row.get("sample_queries", [])
        if not samples:
            continue
        source_type = row.get("source_type", "")
        if not source_type:
            continue
        sample = samples[0]
        cases.append(
            {
                "id": f"chatmed_filtered_source_{len(cases) + 1:04d}",
                "query": sample["query"],
                "source_queue": "source_hint_review_queue",
                "candidate_term": row["term"],
                "priority": row.get("priority", ""),
                "expected_term_groups": [[row["term"]]],
                "expected_negative_terms": [],
                "forbidden_terms": [],
                "expected_source_type_in_top_k": source_type,
                "review_status": "auto_generated_from_filtered_candidate_unverified",
            }
        )
        if len(cases) >= limit:
            break
    return cases


def build_cases(
    *,
    filtered_dir: Path,
    diagnostic_limit: int,
    pathogenesis_limit: int,
    negation_limit: int,
    source_hint_limit: int,
) -> list[dict[str, Any]]:
    diagnostic = read_jsonl(filtered_dir / "diagnostic_term_review_queue.jsonl")
    pathogenesis = read_jsonl(filtered_dir / "pathogenesis_query_review_queue.jsonl")
    negation = read_jsonl(filtered_dir / "negation_forbidden_review_queue.jsonl")
    source_hint = read_jsonl(filtered_dir / "source_hint_review_queue.jsonl")
    return unique_cases(
        [
            *diagnostic_cases(diagnostic, diagnostic_limit),
            *pathogenesis_cases(pathogenesis, pathogenesis_limit),
            *negation_cases(negation, negation_limit),
            *source_hint_cases(source_hint, source_hint_limit),
        ]
    )


def write_manifest(output_dir: Path, cases: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["source_queue"]] = counts.get(case["source_queue"], 0) + 1
    manifest = {
        "name": "query_translator_chatmed_filtered",
        "source": str(args.filtered_dir),
        "case_count": len(cases),
        "by_source_queue": counts,
        "review_status": "auto_generated_from_filtered_candidate_unverified",
        "safety_note": (
            "This set tests local query understanding against filtered ChatMed query candidates. "
            "It excludes source-hint knowledge prompts by default and does not use ChatMed responses as gold medical answers."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 ChatMed filtered review queue 生成 Query Translator 增量测试集")
    parser.add_argument("--filtered-dir", type=Path, default=DEFAULT_FILTERED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--diagnostic-limit", type=int, default=50)
    parser.add_argument("--pathogenesis-limit", type=int, default=15)
    parser.add_argument("--negation-limit", type=int, default=20)
    parser.add_argument(
        "--source-hint-limit",
        type=int,
        default=0,
        help="默认不把本草/方剂/理论来源提示混入 Query Translator 临床症状评测。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(
        filtered_dir=args.filtered_dir,
        diagnostic_limit=args.diagnostic_limit,
        pathogenesis_limit=args.pathogenesis_limit,
        negation_limit=args.negation_limit,
        source_hint_limit=args.source_hint_limit,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    written = write_jsonl(cases_path, cases)
    write_manifest(output_dir, cases, args)
    print(json.dumps({"cases": str(cases_path), "written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
