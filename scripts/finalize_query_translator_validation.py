from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from core.symptom_query_translator import infer_query_intent  # noqa: E402
from core.syndrome_retriever import _build_retrieval_decision  # noqa: E402


DEFAULT_ONLINE_REPORT = ROOT / "datasets" / "structured" / "query_translator_eval_llm_final.json"
DEFAULT_CASES = ROOT / "tests" / "evals" / "query_translator_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "structured" / "query_translator_release_validation.json"
CORE_CHECKS = {
    "candidate_coverage",
    "term_recall",
    "negative_terms",
    "no_forbidden_positive",
    "needs_more_info",
    "retrieval",
}
REPLAYABLE_CHECKS = {"gate", "decision"}
REQUIRED_CHECKS = CORE_CHECKS | REPLAYABLE_CHECKS


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        case = json.loads(raw_line)
        case_id = str(case.get("id", ""))
        if not case_id or case_id in cases:
            raise ValueError(f"{path}:{line_number}: missing or duplicate case id")
        cases[case_id] = case
    return cases


def reconstruct_matches(case_result: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in case_result.get("top_k", []):
        matches.append(
            {
                "canonical_match_count": int(item.get("canonical_match_count", 0)),
                "matched_terms": list(item.get("matched_terms", [])),
                "query_coverage": (
                    int(item.get("canonical_match_count", 0))
                    / max(1, len(case_result.get("canonical_terms", [])))
                ),
                "exact_match_count": 0,
                "payload": {
                    "title": item.get("title", ""),
                    "source_type": item.get("source_type", ""),
                    "formula": item.get("formula", ""),
                    "intervention_name": item.get("intervention_name", ""),
                },
            }
        )
    return matches


def replay_case(case: dict[str, Any], case_result: dict[str, Any]) -> dict[str, Any]:
    query_info = {
        "original_query": case_result.get("query", case.get("query", "")),
        "query_intent": infer_query_intent(case_result.get("query", case.get("query", ""))),
        "canonical_terms": list(case_result.get("canonical_terms", [])),
        "negative_terms": list(case_result.get("negative_terms", [])),
        "needs_more_info": bool(case.get("expected_needs_more_info", False)),
    }
    decision = _build_retrieval_decision(query_info, reconstruct_matches(case_result))
    gate = decision.get("status") == "grounded_answer"
    gate_ok = "expected_gate" not in case or gate is bool(case["expected_gate"])
    decision_ok = (
        "expected_decision" not in case
        or decision.get("status") == case["expected_decision"]
    )
    return {
        "id": case_result["id"],
        "ok": gate_ok and decision_ok,
        "online_failed_checks": [
            name for name, passed in case_result.get("checks", {}).items() if not passed
        ],
        "replayed_gate": gate,
        "replayed_decision": decision,
        "expected_gate": case.get("expected_gate"),
        "expected_decision": case.get("expected_decision"),
    }


def finalize(online_path: Path, cases_path: Path) -> dict[str, Any]:
    raw_online = online_path.read_bytes()
    online = json.loads(raw_online.decode("utf-8"))
    cases = load_jsonl(cases_path)
    results: list[dict[str, Any]] = []
    online_case_ids = [str(item.get("id", "")) for item in online.get("cases", [])]
    case_set_ok = (
        len(online_case_ids) == len(set(online_case_ids))
        and set(online_case_ids) == set(cases)
    )

    for case_result in online.get("cases", []):
        case_id = str(case_result.get("id", ""))
        if case_id not in cases:
            results.append({"id": case_id, "ok": False, "error": "case_definition_missing"})
            continue
        failed_checks = {
            name for name, passed in case_result.get("checks", {}).items() if not passed
        }
        checks_complete = REQUIRED_CHECKS <= set(case_result.get("checks", {}))
        core_ok = all(case_result.get("checks", {}).get(name, False) for name in CORE_CHECKS)
        if checks_complete and not failed_checks:
            results.append({"id": case_id, "ok": True, "validation": "online"})
        elif checks_complete and core_ok and failed_checks <= REPLAYABLE_CHECKS:
            replayed = replay_case(cases[case_id], case_result)
            replayed["validation"] = "online_core_plus_offline_gate_replay"
            results.append(replayed)
        else:
            results.append(
                {
                    "id": case_id,
                    "ok": False,
                    "validation": "not_replayable",
                    "failed_checks": sorted(failed_checks),
                    "checks_complete": checks_complete,
                }
            )

    expected_count = int(online.get("case_count", 0))
    final_passed = sum(1 for item in results if item.get("ok"))
    return {
        "ok": case_set_ok and len(results) == expected_count and final_passed == expected_count,
        "validation_mode": "preserved_online_results_with_deterministic_gate_replay",
        "online_report": str(online_path),
        "online_report_sha256": hashlib.sha256(raw_online).hexdigest(),
        "online_passed": int(online.get("passed", 0)),
        "case_count": expected_count,
        "final_passed": final_passed,
        "case_set_ok": case_set_ok,
        "online_metrics": online.get("metrics", {}),
        "online_latency_ms": online.get("latency_ms", {}),
        "replayed_case_count": sum(
            1 for item in results if item.get("validation") == "online_core_plus_offline_gate_replay"
        ),
        "cases": results,
        "note": "No additional LLM/API call was made; replay is limited to deterministic gate/decision checks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="基于保留的在线结果重放确定性门控，生成发布验收报告")
    parser.add_argument("--online-report", default=str(DEFAULT_ONLINE_REPORT))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = finalize(Path(args.online_report), Path(args.cases))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
