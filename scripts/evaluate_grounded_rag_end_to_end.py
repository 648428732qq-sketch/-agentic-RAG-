from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path[:0] = [str(ROOT), str(PROJECT)]

from scripts.run_chatmed_gold_eval import install_local_qdrant_grpc_stub_if_blocked  # noqa: E402


DEFAULT_EVAL = ROOT / "tests" / "evals" / "query_translator_v3_nondialect_100"
DEFAULT_QUESTIONS = DEFAULT_EVAL / "questions_mixed.jsonl"
DEFAULT_GOLD = DEFAULT_EVAL / "private" / "gold_keys.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "structured" / "grounded_rag_e2e_v3_predictions.jsonl"
DEFAULT_REPORT = ROOT / "datasets" / "structured" / "grounded_rag_e2e_v3_report.json"
DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile)))
    return round(ordered[index], 2)


def build_in_memory_qdrant(dictionary_path: Path, config: Any) -> tuple[Any, Any, int]:
    from langchain_huggingface import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    payloads = read_jsonl(dictionary_path)
    if not payloads:
        raise ValueError(f"empty syndrome dictionary: {dictionary_path}")
    entry_ids = [str(payload.get("entry_id", "")) for payload in payloads]
    if any(not entry_id for entry_id in entry_ids) or len(set(entry_ids)) != len(entry_ids):
        raise ValueError("syndrome dictionary contains empty or duplicate entry_id")
    model_kwargs = {"local_files_only": config.EMBEDDING_LOCAL_FILES_ONLY}
    if config.EMBEDDING_DEVICE and config.EMBEDDING_DEVICE != "auto":
        model_kwargs["device"] = config.EMBEDDING_DEVICE
    embedding = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL, model_kwargs=model_kwargs)
    vector_size = len(embedding.embed_query("test"))
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=config.SYNDROME_COLLECTION,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )
    batch_size = 128
    for start in range(0, len(payloads), batch_size):
        batch = payloads[start : start + batch_size]
        vectors = embedding.embed_documents([str(payload.get("search_text", "")) for payload in batch])
        client.upsert(
            collection_name=config.SYNDROME_COLLECTION,
            points=[
                qmodels.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(payload["entry_id"]))),
                    vector=vector,
                    payload=payload,
                )
                for payload, vector in zip(batch, vectors)
            ],
            wait=True,
        )
        print(f"[qdrant-bootstrap] {min(start + len(batch), len(payloads))}/{len(payloads)}", flush=True)
    count = client.count(collection_name=config.SYNDROME_COLLECTION, exact=True).count
    if count != len(payloads):
        client.close()
        raise RuntimeError(f"in-memory Qdrant count mismatch: expected {len(payloads)}, got {count}")
    return client, embedding, count


def evaluate_case(retriever: Any, question: dict[str, Any], gold: dict[str, Any], top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = retriever.search(str(question["query"]), limit=top_k, candidate_limit=max(80, top_k * 10))
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    query_info = result.get("query", {})
    decision = result.get("decision", {})
    matches = result.get("matches", [])
    terms = set(query_info.get("canonical_terms", []))
    expected_groups = [set(group) for group in gold.get("expected_term_groups", []) if group]
    covered_groups = [bool(group & terms) for group in expected_groups]
    entry_ids = [str(match.get("payload", {}).get("entry_id", "")) for match in matches]
    source_types = [str(match.get("payload", {}).get("source_type", "")) for match in matches]
    expected_entry = str(gold.get("expected_entry_id_in_top_k", ""))
    expected_source = str(gold.get("expected_source_type_in_top_k", ""))
    expected_gate = bool(gold.get("expected_gate"))
    actual_gate = decision.get("status") == "grounded_answer"
    reason_details = decision.get("reason_details", [])
    all_candidates_conflict_free = all(
        not match.get("negative_conflicts") and not match.get("forbidden_conflicts")
        for match in matches
    )
    top_payload_conflict_free = not matches or (
        not matches[0].get("negative_conflicts") and not matches[0].get("forbidden_conflicts")
    )
    route_errors = result.get("retrieval_debug", {}).get("errors", [])
    checks = {
        "all_term_groups_recalled": bool(covered_groups) and all(covered_groups),
        "expected_entry_in_top_k": bool(expected_entry) and expected_entry in entry_ids,
        "expected_source_in_top_k": bool(expected_source) and expected_source in source_types,
        "gate_matches_gold": actual_gate == expected_gate,
        "top_payload_conflict_free": top_payload_conflict_free,
        "all_candidates_conflict_free": all_candidates_conflict_free,
        "route_ok": not route_errors,
    }
    if not expected_gate:
        checks.update(
            {
                "safe_rejection": not actual_gate,
                "rejection_reason_present": not actual_gate
                and bool(reason_details)
                and bool(decision.get("rejection", {}).get("primary_reason")),
                "response_policy_not_answer": decision.get("response_policy") in {"clarify", "refuse"},
            }
        )
    return {
        "id": question["id"],
        "query": question["query"],
        "style": gold.get("generation_style", ""),
        "expected_gate": expected_gate,
        "actual_gate": actual_gate,
        "checks": checks,
        "canonical_terms": sorted(terms),
        "expected_term_groups": [sorted(group) for group in expected_groups],
        "covered_term_groups": covered_groups,
        "expected_entry_id": expected_entry,
        "expected_entry_rank": entry_ids.index(expected_entry) + 1 if expected_entry in entry_ids else None,
        "expected_source_type": expected_source,
        "safety_critical_rejection": not expected_gate and expected_source == "formula_syndrome",
        "decision": decision,
        "rejection_reason_codes": [str(item.get("code", "")) for item in reason_details],
        "rejection_reason_messages": [str(item.get("message", "")) for item in reason_details],
        "matches": [
            {
                "entry_id": match.get("payload", {}).get("entry_id", ""),
                "title": match.get("payload", {}).get("title", ""),
                "source_type": match.get("payload", {}).get("source_type", ""),
                "formula": match.get("payload", {}).get("formula", ""),
                "matched_terms": match.get("matched_terms", []),
                "query_coverage": match.get("query_coverage", 0.0),
                "required_group_coverage": match.get("required_group_coverage", 0.0),
                "negative_conflicts": match.get("negative_conflicts", []),
                "forbidden_conflicts": match.get("forbidden_conflicts", []),
            }
            for match in matches
        ],
        "retrieval_debug": result.get("retrieval_debug", {}),
        "latency_ms": latency_ms,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        matches = row.get("matches", [])
        actual_gate = bool(row.get("actual_gate"))
        expected_gate = bool(row.get("expected_gate"))
        decision = row.get("decision", {})
        reason_details = decision.get("reason_details", [])
        checks = row.setdefault("checks", {})
        checks["top_payload_conflict_free"] = not matches or (
            not matches[0].get("negative_conflicts") and not matches[0].get("forbidden_conflicts")
        )
        checks["all_candidates_conflict_free"] = all(
            not match.get("negative_conflicts") and not match.get("forbidden_conflicts")
            for match in matches
        )
        if not expected_gate:
            checks["safe_rejection"] = not actual_gate
            checks["rejection_reason_present"] = not actual_gate and bool(reason_details) and bool(
                decision.get("rejection", {}).get("primary_reason")
            )
            checks["response_policy_not_answer"] = decision.get("response_policy") in {"clarify", "refuse"}
        row["safety_critical_rejection"] = (
            not expected_gate and row.get("expected_source_type") == "formula_syndrome"
        )
    case_count = len(rows)
    total_groups = sum(len(row["covered_term_groups"]) for row in rows)
    covered_groups = sum(sum(row["covered_term_groups"]) for row in rows)
    expected_rejections = [row for row in rows if not row["expected_gate"]]
    actual_rejections = [row for row in rows if not row["actual_gate"]]
    critical_rejections = [row for row in rows if row.get("safety_critical_rejection")]

    def rate(check: str, subset: list[dict[str, Any]] | None = None) -> float:
        selected = subset if subset is not None else rows
        return round(sum(bool(row["checks"].get(check)) for row in selected) / max(len(selected), 1), 4)

    by_style: dict[str, Any] = {}
    for style in sorted({str(row.get("style", "")) for row in rows}):
        selected = [row for row in rows if row.get("style") == style]
        by_style[style] = {
            "count": len(selected),
            "term_group_recall": round(
                sum(sum(row["covered_term_groups"]) for row in selected)
                / max(sum(len(row["covered_term_groups"]) for row in selected), 1),
                4,
            ),
            "entry_recall_at_k": rate("expected_entry_in_top_k", selected),
            "gate_accuracy": rate("gate_matches_gold", selected),
        }
    reason_counts = Counter(
        code for row in actual_rejections for code in row.get("rejection_reason_codes", []) if code
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    metrics = {
        "term_group_recall": round(covered_groups / max(total_groups, 1), 4),
        "case_all_terms_recalled": rate("all_term_groups_recalled"),
        "entry_recall_at_k": rate("expected_entry_in_top_k"),
        "source_type_recall_at_k": rate("expected_source_in_top_k"),
        "synthetic_gold_gate_accuracy": rate("gate_matches_gold"),
        "top_payload_conflict_free": rate("top_payload_conflict_free"),
        "all_candidates_conflict_free": rate("all_candidates_conflict_free"),
        "route_success": rate("route_ok"),
        "all_hard_negative_safe_rejection_rate": rate("safe_rejection", expected_rejections),
        "clinical_formula_safe_rejection_rate": rate("safe_rejection", critical_rejections),
        "expected_rejection_with_reason_rate": rate("rejection_reason_present", expected_rejections),
        "actual_rejection_reason_coverage": round(
            sum(
                bool(row.get("decision", {}).get("reason_details"))
                and row.get("decision", {}).get("response_policy") in {"clarify", "refuse"}
                for row in actual_rejections
            )
            / max(len(actual_rejections), 1),
            4,
        ),
        "unsafe_answer_rate": round(
            sum(row["actual_gate"] for row in expected_rejections) / max(len(expected_rejections), 1), 4
        ),
    }
    thresholds = {
        "top_payload_conflict_free": 1.0,
        "route_success": 1.0,
        "clinical_formula_safe_rejection_rate": 1.0,
        "actual_rejection_reason_coverage": 1.0,
        "entry_recall_at_k": 0.8,
    }
    failures = {
        key: {"actual": metrics[key], "required": required}
        for key, required in thresholds.items()
        if metrics[key] < required
    }
    return {
        "ok": not failures,
        "scope": "qdrant_payload_filter_evidence_gate_rejection_e2e",
        "case_count": case_count,
        "expected_answer_count": sum(row["expected_gate"] for row in rows),
        "expected_rejection_count": len(expected_rejections),
        "safety_critical_rejection_count": len(critical_rejections),
        "actual_rejection_count": len(actual_rejections),
        "metrics": metrics,
        "thresholds": thresholds,
        "failed_thresholds": failures,
        "rejection_reason_counts": dict(reason_counts.most_common()),
        "by_style": by_style,
        "latency_ms": {
            "average": round(sum(latencies) / max(len(latencies), 1), 2),
            "p95": percentile(latencies, 0.95),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "failures": [row for row in rows if not all(row["checks"].values())][:30],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qdrant, payload filters, evidence gate and rejection reasons")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--enable-biencoder", action="store_true")
    parser.add_argument("--biencoder-model", type=Path)
    parser.add_argument("--biencoder-catalog", type=Path)
    parser.add_argument("--biencoder-device", default="cpu")
    parser.add_argument(
        "--bootstrap-dictionary",
        type=Path,
        help="Build an isolated in-memory Qdrant collection from this frozen JSONL before evaluation",
    )
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument(
        "--rescore-predictions",
        type=Path,
        help="Recompute report metrics from an existing predictions JSONL without running retrieval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rescore_predictions:
        rows = read_jsonl(args.rescore_predictions)
        report = summarize(rows)
        report["configuration"] = {"rescored_from": str(args.rescore_predictions.resolve())}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "failures"}, ensure_ascii=False, indent=2))
        if args.fail_on_error and not report["ok"]:
            raise SystemExit(1)
        return
    install_local_qdrant_grpc_stub_if_blocked()
    import config
    from core.syndrome_retriever import SyndromeRetriever

    config.ENABLE_LLM_SYMPTOM_TRANSLATOR = False
    config.ENABLE_SYNDROME_RERANK = False
    config.ENABLE_QUERY_TRANSLATOR_BIENCODER = bool(args.enable_biencoder)
    config.QUERY_TRANSLATOR_BIENCODER_DEVICE = args.biencoder_device
    if args.biencoder_model:
        config.QUERY_TRANSLATOR_BIENCODER_MODEL = str(args.biencoder_model.resolve())
    if args.biencoder_catalog:
        config.QUERY_TRANSLATOR_BIENCODER_CATALOG = str(args.biencoder_catalog.resolve())

    questions = read_jsonl(args.questions)
    if any(set(row) != {"id", "query"} for row in questions):
        raise ValueError("public questions may only contain id and query")
    gold_by_id = {str(row["id"]): row for row in read_jsonl(args.gold)}
    missing = [str(row["id"]) for row in questions if str(row["id"]) not in gold_by_id]
    if missing:
        raise ValueError(f"missing gold rows: {missing[:5]}")

    runtime_client = None
    runtime_embedding = None
    indexed_points = None
    if args.bootstrap_dictionary:
        runtime_client, runtime_embedding, indexed_points = build_in_memory_qdrant(
            args.bootstrap_dictionary,
            config,
        )
        atexit.register(runtime_client.close)
    retriever = SyndromeRetriever(client=runtime_client, embedding=runtime_embedding)
    atexit.register(retriever.close)
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        row = evaluate_case(retriever, question, gold_by_id[str(question["id"])], max(1, args.top_k))
        rows.append(row)
        print(
            f"[{index}/{len(questions)}] id={row['id']} gate={row['actual_gate']} "
            f"entry_rank={row['expected_entry_rank']} reasons={','.join(row['rejection_reason_codes'])}",
            flush=True,
        )
    write_jsonl(args.predictions, rows)
    report = summarize(rows)
    report["configuration"] = {
        "biencoder_enabled": bool(args.enable_biencoder),
        "biencoder_model": str(config.QUERY_TRANSLATOR_BIENCODER_MODEL),
        "biencoder_catalog": str(config.QUERY_TRANSLATOR_BIENCODER_CATALOG),
        "biencoder_device": config.QUERY_TRANSLATOR_BIENCODER_DEVICE,
        "llm_translator_enabled": False,
        "rerank_enabled": False,
        "top_k": max(1, args.top_k),
        "qdrant_mode": "in_memory_frozen_dictionary" if args.bootstrap_dictionary else "configured_runtime",
        "bootstrap_dictionary": str(args.bootstrap_dictionary.resolve()) if args.bootstrap_dictionary else "",
        "indexed_points": indexed_points,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, ensure_ascii=False, indent=2))
    if args.fail_on_error and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
