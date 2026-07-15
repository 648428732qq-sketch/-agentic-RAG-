from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path[:0] = [str(ROOT), str(PROJECT)]

from scripts.evaluate_grounded_rag_end_to_end import build_in_memory_qdrant  # noqa: E402
from scripts.run_chatmed_gold_eval import install_local_qdrant_grpc_stub_if_blocked  # noqa: E402


DEFAULT_CASES = ROOT / "tests" / "evals" / "required_group_hard_negative_v1" / "cases.jsonl"
DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary_effective.jsonl"
DEFAULT_PREDICTIONS = ROOT / "datasets" / "structured" / "required_group_hard_negative_predictions.jsonl"
DEFAULT_REPORT = ROOT / "datasets" / "structured" / "required_group_hard_negative_report.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def evaluate_case(retriever: Any, case: dict[str, Any], top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = retriever.search(case["query"], limit=top_k, candidate_limit=max(80, top_k * 10))
    matches = result.get("matches", [])
    decision = result.get("decision", {})
    formulas = [str(match.get("payload", {}).get("formula", "")) for match in matches]
    entry_ids = [str(match.get("payload", {}).get("entry_id", "")) for match in matches]
    grounded = decision.get("status") == "grounded_answer"
    expected_gate = bool(case["expected_gate"])
    reason_details = decision.get("reason_details", [])
    checks = {
        "route_ok": not result.get("retrieval_debug", {}).get("errors", []),
        "gate_matches_expected": grounded == expected_gate,
        "top_payload_conflict_free": not matches
        or (not matches[0].get("negative_conflicts") and not matches[0].get("forbidden_conflicts")),
    }
    if expected_gate:
        checks.update(
            {
                "target_in_top_k": case["expected_entry_id"] in entry_ids,
                "grounded": grounded,
            }
        )
    else:
        checks.update(
            {
                "safe_rejection": not grounded,
                "reason_present": bool(reason_details)
                and decision.get("response_policy") in {"clarify", "refuse"},
            }
        )
    return {
        "id": case["id"],
        "style": case["style"],
        "query": case["query"],
        "expected_formula": case["expected_formula"],
        "expected_entry_id": case["expected_entry_id"],
        "expected_gate": expected_gate,
        "actual_gate": grounded,
        "omitted_required_group": case.get("omitted_required_group", []),
        "checks": checks,
        "decision": decision,
        "canonical_terms": result.get("query", {}).get("canonical_terms", []),
        "primary_canonical_terms": result.get("query", {}).get("primary_canonical_terms", []),
        "query_intent": result.get("query", {}).get("query_intent", "unknown"),
        "target_rank": entry_ids.index(case["expected_entry_id"]) + 1 if case["expected_entry_id"] in entry_ids else None,
        "top_k_formulas": formulas,
        "top_match": {
            "formula": str(matches[0].get("payload", {}).get("formula", "")) if matches else "",
            "entry_id": str(matches[0].get("payload", {}).get("entry_id", "")) if matches else "",
            "matched_terms": matches[0].get("matched_terms", []) if matches else [],
            "matched_required_symptom_groups": matches[0].get("matched_required_symptom_groups", []) if matches else [],
            "missing_required_symptom_groups": matches[0].get("missing_required_symptom_groups", []) if matches else [],
            "query_coverage": matches[0].get("query_coverage", 0.0) if matches else 0.0,
            "required_group_coverage": matches[0].get("required_group_coverage", 0.0) if matches else 0.0,
        },
        "retrieval_debug": result.get("retrieval_debug", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [row for row in rows if row["style"] == "full_required_signature"]
    missing = [row for row in rows if row["style"] == "missing_one_required_group"]

    def rate(subset: list[dict[str, Any]], check: str) -> float:
        return round(sum(bool(row["checks"].get(check)) for row in subset) / max(len(subset), 1), 4)

    missing_reasons = Counter(
        detail.get("code", "")
        for row in missing
        for detail in row.get("decision", {}).get("reason_details", [])
        if detail.get("code")
    )
    full_reasons = Counter(
        detail.get("code", "")
        for row in full
        if not row["checks"].get("grounded")
        for detail in row.get("decision", {}).get("reason_details", [])
        if detail.get("code")
    )
    metrics = {
        "full_target_recall_at_k": rate(full, "target_in_top_k"),
        "full_grounded_rate": rate(full, "grounded"),
        "missing_group_safe_rejection_rate": rate(missing, "safe_rejection"),
        "missing_group_reason_coverage": rate(missing, "reason_present"),
        "route_success": rate(rows, "route_ok"),
        "top_payload_conflict_free": rate(rows, "top_payload_conflict_free"),
    }
    thresholds = {
        "full_target_recall_at_k": 0.8,
        "full_grounded_rate": 0.95,
        "missing_group_safe_rejection_rate": 1.0,
        "missing_group_reason_coverage": 1.0,
        "route_success": 1.0,
        "top_payload_conflict_free": 1.0,
    }
    failures = {
        key: {"actual": metrics[key], "required": threshold}
        for key, threshold in thresholds.items()
        if metrics[key] < threshold
    }
    return {
        "ok": not failures,
        "case_count": len(rows),
        "full_signature_count": len(full),
        "missing_one_group_count": len(missing),
        "metrics": metrics,
        "thresholds": thresholds,
        "failed_thresholds": failures,
        "missing_group_rejection_reason_counts": dict(missing_reasons.most_common()),
        "full_signature_rejection_reason_counts": dict(full_reasons.most_common()),
        "failures": [row for row in rows if not all(row["checks"].values())][:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate evidence-traced one-group-at-a-time hard negatives")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--enable-biencoder", action="store_true")
    parser.add_argument("--biencoder-device", default="cpu")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    install_local_qdrant_grpc_stub_if_blocked()
    import config
    from core.syndrome_retriever import SyndromeRetriever

    config.ENABLE_LLM_SYMPTOM_TRANSLATOR = False
    config.ENABLE_SYNDROME_RERANK = False
    config.ENABLE_QUERY_TRANSLATOR_BIENCODER = bool(args.enable_biencoder)
    config.QUERY_TRANSLATOR_BIENCODER_DEVICE = args.biencoder_device
    client, embedding, indexed_points = build_in_memory_qdrant(args.dictionary, config)
    atexit.register(client.close)
    retriever = SyndromeRetriever(client=client, embedding=embedding)
    cases = read_jsonl(args.cases)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        row = evaluate_case(retriever, case, max(1, args.top_k))
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] {row['id']} gate={row['actual_gate']} target_rank={row['target_rank']}",
            flush=True,
        )
    write_jsonl(args.predictions, rows)
    report = summarize(rows)
    report["configuration"] = {
        "dictionary": str(args.dictionary.resolve()),
        "indexed_points": indexed_points,
        "biencoder_enabled": bool(args.enable_biencoder),
        "llm_enabled": False,
        "rerank_enabled": False,
        "top_k": max(1, args.top_k),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, ensure_ascii=False, indent=2))
    if args.fail_on_error and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
