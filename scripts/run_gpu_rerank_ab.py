from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))


DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "structured" / "rerank_gpu_ab"
DEFAULT_RERANK_MODEL = ROOT / "models" / "bge-reranker-v2-m3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                rows.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def merge_questions_gold(
    questions: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    question_ids = [str(row["id"]) for row in questions]
    missing = [case_id for case_id in question_ids if case_id not in gold_by_id]
    extras = sorted(set(gold_by_id) - set(question_ids))
    if missing or extras:
        raise ValueError(f"question/gold id mismatch: missing={missing[:5]}, extras={extras[:5]}")
    return [{**gold_by_id[str(question["id"])], **question} for question in questions]


def ranking_metrics(results: list[dict[str, Any]], *, style: str = "full_signature_topk") -> dict[str, Any]:
    selected = [result for result in results if result.get("style") == style]
    ranks = [result.get("target_rank") for result in selected]
    count = len(ranks)
    if not count:
        return {"count": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_8": 0.0, "mrr_at_8": 0.0, "ndcg_at_8": 0.0}
    return {
        "count": count,
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / count, 4),
        "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / count, 4),
        "recall_at_8": round(sum(rank is not None and rank <= 8 for rank in ranks) / count, 4),
        "mrr_at_8": round(
            sum((1.0 / rank) if rank is not None and rank <= 8 else 0.0 for rank in ranks) / count,
            4,
        ),
        "ndcg_at_8": round(
            sum((1.0 / math.log2(rank + 1)) if rank is not None and rank <= 8 else 0.0 for rank in ranks)
            / count,
            4,
        ),
    }


def compare_runs(
    baseline: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(item["id"]): item for item in baseline}
    if set(baseline_by_id) != {str(item["id"]) for item in reranked}:
        raise ValueError("baseline and rerank result ids differ")
    return {
        "pass_gained": sum((not baseline_by_id[str(item["id"])]["ok"]) and item["ok"] for item in reranked),
        "pass_lost": sum(baseline_by_id[str(item["id"])]["ok"] and (not item["ok"]) for item in reranked),
        "top1_changed": sum(
            baseline_by_id[str(item["id"])].get("top_formula") != item.get("top_formula") for item in reranked
        ),
        "target_rank_improved": sum(
            (item.get("target_rank") or 999)
            < (baseline_by_id[str(item["id"])].get("target_rank") or 999)
            for item in reranked
        ),
        "target_rank_worsened": sum(
            (item.get("target_rank") or 999)
            > (baseline_by_id[str(item["id"])].get("target_rank") or 999)
            for item in reranked
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible in-memory GPU baseline/rerank A/B evaluation.")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rerank-model", type=Path, default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-formulas", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rerank-candidates", type=int, default=8)
    parser.add_argument("--rerank-max-length", type=int, default=256)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--rerank-mode", choices=("evidence_first", "score_first"), default="evidence_first")
    return parser.parse_args()


def load_cases(args: argparse.Namespace, dictionary_rows: list[dict[str, Any]], evaluator: Any) -> list[dict[str, Any]]:
    if bool(args.questions) != bool(args.gold):
        raise ValueError("--questions and --gold must be supplied together")
    if args.questions and args.gold:
        return merge_questions_gold(read_jsonl(args.questions), read_jsonl(args.gold))
    if args.cases:
        return read_jsonl(args.cases)
    formula_entries = evaluator.pick_best_formula_entries(dictionary_rows)
    return evaluator.generate_cases(formula_entries, max_formulas=max(1, args.max_formulas))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dictionary.exists():
        raise FileNotFoundError(args.dictionary)
    if not (args.rerank_model / "model.safetensors").exists():
        raise FileNotFoundError(args.rerank_model / "model.safetensors")

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import config
    from core.syndrome_retriever import SyndromeRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    import scripts.evaluate_formula_hard_negatives as evaluator

    config.ENABLE_LLM_SYMPTOM_TRANSLATOR = False
    config.DENSE_MODEL = args.embedding_model
    config.EMBEDDING_DEVICE = args.device
    config.EMBEDDING_LOCAL_FILES_ONLY = True
    config.SYNDROME_RERANK_MODEL = str(args.rerank_model.resolve())
    config.SYNDROME_RERANK_CANDIDATES = max(2, args.rerank_candidates)
    config.SYNDROME_RERANK_DEVICE = args.device
    config.SYNDROME_RERANK_MAX_LENGTH = max(64, args.rerank_max_length)
    config.SYNDROME_RERANK_BATCH_SIZE = max(1, args.rerank_batch_size)
    config.SYNDROME_RERANK_MODE = args.rerank_mode
    config.SYNDROME_RERANK_LOCAL_FILES_ONLY = True

    dictionary_rows = read_jsonl(args.dictionary)
    formula_entries = evaluator.pick_best_formula_entries(dictionary_rows)
    cases = load_cases(args, dictionary_rows, evaluator)
    if not cases:
        raise RuntimeError("no evaluation cases")
    write_jsonl(args.output_dir / "cases.jsonl", cases)
    print(f"STAGE=cases entries={len(dictionary_rows)} formulas={len(formula_entries)} cases={len(cases)}", flush=True)

    embedding = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": args.device, "local_files_only": True},
    )
    probe = embedding.embed_query("test")
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=config.SYNDROME_COLLECTION,
        vectors_config=qmodels.VectorParams(size=len(probe), distance=qmodels.Distance.COSINE),
    )
    print("STAGE=embedding_documents", flush=True)
    vectors = embedding.embed_documents([str(row.get("search_text", "")) for row in dictionary_rows])
    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(row.get("entry_id", index)))),
            vector=vector,
            payload=row,
        )
        for index, (row, vector) in enumerate(zip(dictionary_rows, vectors))
    ]
    client.upsert(collection_name=config.SYNDROME_COLLECTION, points=points, wait=True)
    print(f"STAGE=index_ready points={client.count(config.SYNDROME_COLLECTION, exact=True).count}", flush=True)

    baseline_retriever = SyndromeRetriever(client=client, embedding=embedding)
    rerank_retriever = SyndromeRetriever(client=client, embedding=embedding)
    config.ENABLE_SYNDROME_RERANK = True
    scores, preload_debug = rerank_retriever._ensure_reranker().score(
        "warmup",
        [{"payload": {"title": "warmup", "evidence": "warmup"}}],
    )
    if not scores or not preload_debug.get("rerank_used"):
        raise RuntimeError(f"reranker preload failed: {preload_debug}")

    # Warm both retrieval paths before measuring. Alternate execution order per case
    # to reduce systematic hot-cache bias in the latency comparison.
    warmup_case = cases[0]
    config.ENABLE_SYNDROME_RERANK = False
    evaluator.evaluate_case(baseline_retriever, warmup_case, top_k=max(1, args.top_k))
    config.ENABLE_SYNDROME_RERANK = True
    evaluator.evaluate_case(rerank_retriever, warmup_case, top_k=max(1, args.top_k))

    baseline_results: list[dict[str, Any]] = []
    rerank_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        if index % 2:
            config.ENABLE_SYNDROME_RERANK = False
            baseline_result = evaluator.evaluate_case(baseline_retriever, case, top_k=max(1, args.top_k))
            config.ENABLE_SYNDROME_RERANK = True
            rerank_result = evaluator.evaluate_case(rerank_retriever, case, top_k=max(1, args.top_k))
        else:
            config.ENABLE_SYNDROME_RERANK = True
            rerank_result = evaluator.evaluate_case(rerank_retriever, case, top_k=max(1, args.top_k))
            config.ENABLE_SYNDROME_RERANK = False
            baseline_result = evaluator.evaluate_case(baseline_retriever, case, top_k=max(1, args.top_k))
        baseline_results.append(baseline_result)
        rerank_results.append(rerank_result)
        if index % 25 == 0 or index == len(cases):
            print(f"STAGE=evaluate done={index}/{len(cases)}", flush=True)
    wall_clock_seconds = round(time.perf_counter() - started, 2)

    baseline_report = evaluator.summarize(
        baseline_results,
        cases,
        formula_count=len(formula_entries),
        mode="gpu_in_memory_payload_hard_rank_no_rerank",
    )
    rerank_report = evaluator.summarize(
        rerank_results,
        cases,
        formula_count=len(formula_entries),
        mode="gpu_in_memory_payload_hard_rank_with_bge_reranker_v2_m3",
    )
    comparison = compare_runs(baseline_results, rerank_results)
    combined = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
        },
        "config": {
            "dictionary": str(args.dictionary.resolve()),
            "rerank_model": str(args.rerank_model.resolve()),
            "embedding_model": args.embedding_model,
            "top_k": args.top_k,
            "rerank_candidates": args.rerank_candidates,
            "rerank_max_length": args.rerank_max_length,
            "rerank_batch_size": args.rerank_batch_size,
            "rerank_mode": args.rerank_mode,
        },
        "wall_clock_seconds": wall_clock_seconds,
        "baseline": baseline_report,
        "rerank": rerank_report,
        "ranking_metrics_full_signature": {
            "baseline": ranking_metrics(baseline_results),
            "rerank": ranking_metrics(rerank_results),
        },
        "comparison": comparison,
        "latency_delta_ms": {
            "average": round(rerank_report["latency_ms"]["average"] - baseline_report["latency_ms"]["average"], 2),
            "p95": round(rerank_report["latency_ms"]["p95"] - baseline_report["latency_ms"]["p95"], 2),
        },
    }
    write_jsonl(args.output_dir / "baseline_predictions.jsonl", baseline_results)
    write_jsonl(args.output_dir / "rerank_predictions.jsonl", rerank_results)
    (args.output_dir / "report.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "report": str((args.output_dir / "report.json").resolve()),
        "baseline_pass_rate": baseline_report["pass_rate"],
        "rerank_pass_rate": rerank_report["pass_rate"],
        "comparison": comparison,
        "baseline_latency_ms": baseline_report["latency_ms"],
        "rerank_latency_ms": rerank_report["latency_ms"],
    }, ensure_ascii=False), flush=True)
    client.close()


if __name__ == "__main__":
    main()
