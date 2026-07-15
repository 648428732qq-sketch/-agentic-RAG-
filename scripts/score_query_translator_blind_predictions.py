from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "datasets" / "structured" / "query_translator_mixed_300_predictions.jsonl"
DEFAULT_GOLD = ROOT / "tests" / "evals" / "query_translator_mixed_300" / "private" / "gold_keys.jsonl"
DEFAULT_REPORT = ROOT / "datasets" / "structured" / "query_translator_mixed_300_blind_report.json"


EXPECTED_TERM_CONFLICTS = {
    "无汗": {"汗出", "出汗", "有汗", "自汗", "多汗", "汗大出"},
    "汗出": {"无汗"},
    "出汗": {"无汗"},
    "有汗": {"无汗"},
    "自汗": {"无汗"},
    "多汗": {"无汗"},
    "不渴": {"口渴", "口大渴", "渴欲得水", "渴欲饮水"},
    "口渴": {"不渴"},
    "口大渴": {"不渴"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def groups_covered(groups: list[list[str]], actual: set[str]) -> bool:
    return all(bool(set(group) & actual) for group in groups)


def top_text(item: dict[str, Any]) -> str:
    return " ".join(str(value) for value in item.values() if value and not isinstance(value, list))


def find_expected_positive_conflicts(
    expected_groups: list[list[str]],
    canonical: set[str],
) -> list[str]:
    expected_terms = {term for group in expected_groups for term in group}
    conflicts = {
        conflicting
        for expected in expected_terms
        for conflicting in EXPECTED_TERM_CONFLICTS.get(expected, set())
        if conflicting in canonical and conflicting not in expected_terms
    }
    return sorted(conflicts)


def score_case(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    canonical = set(prediction.get("canonical_terms", []))
    candidates = set(prediction.get("candidate_terms", []))
    negatives = set(prediction.get("negative_terms", []))
    expected_groups = gold.get("expected_term_groups", [])
    expected_negatives = set(gold.get("expected_negative_terms", []))
    forbidden = set(gold.get("forbidden_terms", []))
    top_k = prediction.get("top_k", [])
    top = top_k[0] if top_k else {}
    positive_conflicts = find_expected_positive_conflicts(expected_groups, canonical)

    retrieval_ok = True
    if gold.get("expected_formula"):
        retrieval_ok &= top.get("formula") == gold["expected_formula"]
    if gold.get("expected_formula_in_top_k"):
        retrieval_ok &= any(item.get("formula") == gold["expected_formula_in_top_k"] for item in top_k)
    if gold.get("expected_any_formula_in_top_k"):
        accepted = set(gold["expected_any_formula_in_top_k"])
        retrieval_ok &= any(item.get("formula") in accepted for item in top_k)
    if gold.get("expected_source_type"):
        retrieval_ok &= top.get("source_type") == gold["expected_source_type"]
    if gold.get("expected_source_type_in_top_k"):
        retrieval_ok &= any(item.get("source_type") == gold["expected_source_type_in_top_k"] for item in top_k)
    if gold.get("expected_intervention_text"):
        retrieval_ok &= any(gold["expected_intervention_text"] in top_text(item) for item in top_k)

    expected_decision = gold.get("expected_decision")
    decision_status = prediction.get("decision", {}).get("status")
    checks = {
        "candidate_coverage": groups_covered(expected_groups, candidates | canonical)
        and expected_negatives.issubset(candidates | negatives),
        "term_recall": groups_covered(expected_groups, canonical),
        "negative_terms": expected_negatives.issubset(negatives),
        "no_forbidden_positive": not bool(forbidden & canonical),
        "no_expected_positive_conflict": not positive_conflicts,
        "needs_more_info": (
            "expected_needs_more_info" not in gold
            or bool(prediction.get("needs_more_info")) is bool(gold["expected_needs_more_info"])
        ),
        "retrieval": bool(retrieval_ok),
        "gate": "expected_gate" not in gold or bool(prediction.get("gate")) is bool(gold["expected_gate"]),
        "decision": expected_decision is None or decision_status == expected_decision,
        "must_clarify": not gold.get("must_clarify") or decision_status == "clarify",
    }
    return {
        "id": gold["id"],
        "style": gold.get("generation_style", "unknown"),
        "ok": all(checks.values()),
        "checks": checks,
        "canonical_terms": sorted(canonical),
        "negative_terms": sorted(negatives),
        "positive_conflicts": positive_conflicts,
        "decision": decision_status,
        "latency_ms": prediction.get("latency_ms", 0),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    check_names = sorted({name for case in cases for name in case["checks"]})
    metrics = {
        name: round(sum(case["checks"][name] for case in cases) / len(cases), 4)
        for name in check_names
    }
    styles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        styles[case["style"]].append(case)
    by_style = {
        style: {
            "count": len(rows),
            "passed": sum(row["ok"] for row in rows),
            "pass_rate": round(sum(row["ok"] for row in rows) / len(rows), 4),
            "term_recall": round(sum(row["checks"]["term_recall"] for row in rows) / len(rows), 4),
            "decision": round(sum(row["checks"]["decision"] for row in rows) / len(rows), 4),
        }
        for style, rows in sorted(styles.items())
    }
    return {
        "ok": all(case["ok"] for case in cases),
        "case_count": len(cases),
        "passed": sum(case["ok"] for case in cases),
        "metrics": metrics,
        "by_style": by_style,
        "average_latency_ms": round(sum(float(case["latency_ms"]) for case in cases) / len(cases), 2),
        "failure_counts": dict(Counter(name for case in cases for name, passed in case["checks"].items() if not passed)),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="完全离线地用私有金标准评分公开问题预测")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = {record["id"]: record for record in read_jsonl(args.predictions)}
    gold = read_jsonl(args.gold)
    missing = [record["id"] for record in gold if record["id"] not in predictions]
    if missing:
        raise ValueError(f"缺少预测: {missing[:10]}，共{len(missing)}条")
    report = summarize([score_case(predictions[record["id"]], record) for record in gold])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
