from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from core.evidence_gate import evaluate_evidence_gate, intervention_identity  # noqa: E402
from core.hybrid_retrieval import payload_contains_term  # noqa: E402
from core.syndrome_retriever import local_rank_key  # noqa: E402


DEFAULT_NEGATIVES = ROOT / "datasets" / "external" / "supervision" / "hard_negatives.jsonl"
DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "reports" / "local_hard_negative_ranking_report.json"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def list_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if value else []


def required_groups(payload: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    for value in payload.get("required_symptom_groups") or []:
        group = list_terms(value)
        if group:
            groups.append(group)
    return groups


def build_match(payload: dict[str, Any], query_terms: list[str]) -> dict[str, Any]:
    matched_terms = [term for term in query_terms if payload_contains_term(payload, term)]
    groups = required_groups(payload)
    matched_groups = [group for group in groups if set(group) & set(query_terms)]
    missing_groups = [group for group in groups if group not in matched_groups]
    diagnostic = list_terms(payload.get("diagnostic_keys") or payload.get("ancient_symptoms"))
    differential = list_terms(payload.get("differential_keys"))
    matched_diagnostic = [term for term in query_terms if term in diagnostic]
    matched_differential = [term for term in query_terms if term in differential]
    forbidden = set(list_terms(payload.get("forbidden_terms")))
    coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0
    required_coverage = len(matched_groups) / len(groups) if groups else 0.0
    return {
        "payload": payload,
        "matched_terms": matched_terms,
        "primary_matched_terms": matched_terms,
        "canonical_match_count": len(matched_terms),
        "primary_canonical_match_count": len(matched_terms),
        "query_coverage": round(coverage, 4),
        "matched_required_symptom_groups": matched_groups,
        "missing_required_symptom_groups": missing_groups,
        "required_group_match_count": len(matched_groups),
        "required_group_coverage": round(required_coverage, 4),
        "matched_diagnostic_terms": matched_diagnostic,
        "diagnostic_coverage": round(len(matched_diagnostic) / len(diagnostic), 4) if diagnostic else 0.0,
        "matched_differential_terms": matched_differential,
        "differential_coverage": round(len(matched_differential) / len(differential), 4) if differential else 0.0,
        "specificity_score": sum(len(term) for term in matched_terms),
        "negative_conflicts": [],
        "forbidden_conflicts": [term for term in query_terms if term in forbidden],
        "exact_match_count": 0,
        "overlap_score": len(matched_terms),
        "route_count": 1,
        "rrf_score": 0.0,
        "score": 0.0,
    }


def evaluate_cases(
    hard_negatives: Iterable[dict[str, Any]],
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    interventions: set[str] = set()
    for negative in hard_negatives:
        anchor = entries.get(str(negative.get("anchor_entry_id")))
        candidate = entries.get(str(negative.get("candidate_entry_id")))
        if not anchor or not candidate:
            counts["missing_payload"] += 1
            continue
        query_terms = [str(term) for term in negative.get("query_terms") or [] if str(term)]
        query_info = {
            "original_query": " ".join(query_terms),
            "query_intent": "clinical_symptom",
            "canonical_terms": query_terms,
            "primary_canonical_terms": query_terms,
            "negative_terms": [],
            "needs_more_info": False,
        }
        matches = [build_match(anchor, query_terms), build_match(candidate, query_terms)]
        matches.sort(key=lambda item: local_rank_key(query_info, item), reverse=True)
        decision = evaluate_evidence_gate(query_info, matches)
        top_id = str(matches[0]["payload"].get("entry_id", ""))
        expected = str(negative.get("expected_decision", "rank_anchor"))
        if expected == "rank_anchor":
            passed = top_id == str(negative.get("anchor_entry_id"))
            counts["distinguishable_cases"] += 1
            counts["distinguishable_passed"] += int(passed)
        else:
            passed = decision.get("status") == "clarify"
            counts["clarify_cases"] += 1
            counts["clarify_passed"] += int(passed)
        counts["cases"] += 1
        counts["passed"] += int(passed)
        source_type = str(anchor.get("source_type", "unknown"))
        source_types[source_type] += 1
        interventions.update((intervention_identity(anchor), intervention_identity(candidate)))
        cases.append(
            {
                "negative_id": negative.get("negative_id"),
                "source_type": source_type,
                "query_terms": query_terms,
                "expected_decision": expected,
                "expected_entry_id": negative.get("anchor_entry_id"),
                "top_entry_id": top_id,
                "gate_status": decision.get("status"),
                "gate_reasons": decision.get("reasons", []),
                "passed": passed,
            }
        )

    distinguishable_rate = counts["distinguishable_passed"] / counts["distinguishable_cases"] if counts["distinguishable_cases"] else 0.0
    clarify_rate = counts["clarify_passed"] / counts["clarify_cases"] if counts["clarify_cases"] else 1.0
    overall_rate = counts["passed"] / counts["cases"] if counts["cases"] else 0.0
    ready = (
        counts["cases"] >= 100
        and counts["clarify_cases"] >= 20
        and len(source_types) >= 3
        and len(interventions) >= 50
        and distinguishable_rate >= 0.85
        and clarify_rate >= 0.98
    )
    return {
        "ok": ready,
        "metrics": {
            "case_count": counts["cases"],
            "pass_rate": round(overall_rate, 6),
            "distinguishable_top1": round(distinguishable_rate, 6),
            "must_clarify_accuracy": round(clarify_rate, 6),
            "source_type_count": len(source_types),
            "intervention_count": len(interventions),
        },
        "counts": dict(sorted(counts.items())),
        "by_source_type": dict(source_types.most_common()),
        "readiness_thresholds": {
            "case_count": 100,
            "clarify_cases": 20,
            "source_type_count": 3,
            "intervention_count": 50,
            "distinguishable_top1": 0.85,
            "must_clarify_accuracy": 0.98,
        },
        "failures": [case for case in cases if not case["passed"]],
    }


def run_evaluation(hard_negative_path: Path, dictionary_path: Path, output_path: Path) -> dict[str, Any]:
    entries = {str(entry.get("entry_id")): entry for entry in iter_jsonl(dictionary_path)}
    result = evaluate_cases(iter_jsonl(hard_negative_path), entries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate payload-first hard ranking before neural rerank")
    parser.add_argument("--hard-negatives", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(args.hard_negatives, args.dictionary, args.output)
    print(json.dumps({"ok": result["ok"], **result["metrics"]}, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
