from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from core.evidence_gate import evaluate_evidence_gate  # noqa: E402
from core.syndrome_reranker import CrossEncoderReranker, build_rerank_query, payload_to_rerank_text  # noqa: E402
from core.syndrome_retriever import local_rank_key, rerank_evidence_key  # noqa: E402
from scripts.evaluate_local_hard_negative_ranking import build_match  # noqa: E402


DEFAULT_NEGATIVES = ROOT / "datasets" / "external" / "supervision" / "hard_negatives.jsonl"
DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_MODEL = ROOT / "models" / "bge-reranker-v2-m3"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "reports" / "neural_rerank_hard_negative_ab.json"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def prepare_cases(
    negatives_path: Path,
    dictionary_path: Path,
    limit: int = 0,
) -> list[dict[str, Any]]:
    entries = {str(entry.get("entry_id")): entry for entry in iter_jsonl(dictionary_path)}
    cases: list[dict[str, Any]] = []
    for negative in iter_jsonl(negatives_path):
        anchor = entries.get(str(negative.get("anchor_entry_id")))
        candidate = entries.get(str(negative.get("candidate_entry_id")))
        if not anchor or not candidate:
            continue
        query_terms = [str(term) for term in negative.get("query_terms") or [] if str(term)]
        query_info = {
            "original_query": " ".join(query_terms),
            "query_intent": "clinical_symptom",
            "canonical_terms": query_terms,
            "primary_canonical_terms": query_terms,
            "colloquial_terms": [],
            "negative_terms": [],
            "pathogenesis_hints": [],
            "needs_more_info": False,
        }
        matches = [build_match(anchor, query_terms), build_match(candidate, query_terms)]
        matches.sort(key=lambda item: local_rank_key(query_info, item), reverse=True)
        cases.append(
            {
                "negative": negative,
                "query_info": query_info,
                "baseline_matches": matches,
            }
        )
        if limit and len(cases) >= limit:
            break
    return cases


def case_passed(case: dict[str, Any], matches: list[dict[str, Any]]) -> bool:
    expected = str(case["negative"].get("expected_decision", "rank_anchor"))
    if expected == "clarify":
        return evaluate_evidence_gate(case["query_info"], matches).get("status") == "clarify"
    top_id = str(matches[0].get("payload", {}).get("entry_id", ""))
    return top_id == str(case["negative"].get("anchor_entry_id", ""))


def evaluate_scored_cases(cases: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    if len(scores) != len(cases) * 2:
        raise ValueError(f"expected {len(cases) * 2} scores, received {len(scores)}")
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        baseline = case["baseline_matches"]
        baseline_pass = case_passed(case, baseline)
        reranked: list[dict[str, Any]] = []
        for match_index, match in enumerate(baseline):
            updated = dict(match)
            updated["rerank_score"] = float(scores[case_index * 2 + match_index])
            reranked.append(updated)
        reranked.sort(
            key=lambda item: (
                rerank_evidence_key(case["query_info"], item),
                item["rerank_score"],
                local_rank_key(case["query_info"], item),
            ),
            reverse=True,
        )
        rerank_pass = case_passed(case, reranked)
        counts["cases"] += 1
        counts["baseline_passed"] += int(baseline_pass)
        counts["rerank_passed"] += int(rerank_pass)
        if baseline_pass and not rerank_pass:
            counts["worsened"] += 1
        elif not baseline_pass and rerank_pass:
            counts["improved"] += 1
        else:
            counts["unchanged"] += 1
        baseline_evidence = rerank_evidence_key(case["query_info"], baseline[0])
        rerank_evidence = rerank_evidence_key(case["query_info"], reranked[0])
        if rerank_evidence < baseline_evidence:
            counts["evidence_boundary_violations"] += 1
        expected = str(case["negative"].get("expected_decision", "rank_anchor"))
        if expected == "clarify":
            counts["clarify_cases"] += 1
            counts["baseline_clarify_passed"] += int(baseline_pass)
            counts["rerank_clarify_passed"] += int(rerank_pass)
        else:
            counts["rank_cases"] += 1
            counts["baseline_rank_passed"] += int(baseline_pass)
            counts["rerank_rank_passed"] += int(rerank_pass)
        if not rerank_pass and len(failures) < 200:
            failures.append(
                {
                    "negative_id": case["negative"].get("negative_id"),
                    "expected_decision": expected,
                    "query_terms": case["negative"].get("query_terms", []),
                    "baseline_top": baseline[0].get("payload", {}).get("entry_id"),
                    "rerank_top": reranked[0].get("payload", {}).get("entry_id"),
                    "rerank_scores": [item["rerank_score"] for item in reranked],
                    "rerank_gate": evaluate_evidence_gate(case["query_info"], reranked),
                }
            )

    def rate(numerator: str, denominator: str) -> float:
        return counts[numerator] / counts[denominator] if counts[denominator] else 0.0

    metrics = {
        "case_count": counts["cases"],
        "baseline_pass_rate": round(rate("baseline_passed", "cases"), 6),
        "rerank_pass_rate": round(rate("rerank_passed", "cases"), 6),
        "baseline_rank_top1": round(rate("baseline_rank_passed", "rank_cases"), 6),
        "rerank_rank_top1": round(rate("rerank_rank_passed", "rank_cases"), 6),
        "baseline_clarify_accuracy": round(rate("baseline_clarify_passed", "clarify_cases"), 6),
        "rerank_clarify_accuracy": round(rate("rerank_clarify_passed", "clarify_cases"), 6),
    }
    safe = (
        counts["cases"] > 0
        and counts["evidence_boundary_violations"] == 0
        and counts["worsened"] == 0
        and metrics["rerank_clarify_accuracy"] >= 0.98
    )
    return {
        "ok": safe,
        "metrics": metrics,
        "counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def run_ab(
    negatives_path: Path,
    dictionary_path: Path,
    model_path: Path,
    output_path: Path,
    *,
    device: str,
    batch_size: int,
    max_length: int,
    limit: int,
) -> dict[str, Any]:
    cases = prepare_cases(negatives_path, dictionary_path, limit=limit)
    reranker = CrossEncoderReranker(
        str(model_path),
        device=device,
        local_files_only=True,
        max_length=max_length,
        batch_size=batch_size,
        trust_remote_code=False,
    )
    model = reranker._ensure_model()
    all_scores: list[float] = []
    batch_latencies: list[float] = []
    cases_per_batch = max(1, batch_size // 2)
    started = time.perf_counter()
    for offset in range(0, len(cases), cases_per_batch):
        case_batch = cases[offset : offset + cases_per_batch]
        pairs: list[tuple[str, str]] = []
        for case in case_batch:
            query_text = build_rerank_query(case["query_info"])
            pairs.extend(
                (query_text, payload_to_rerank_text(match.get("payload", {})))
                for match in case["baseline_matches"]
            )
        batch_started = time.perf_counter()
        raw_scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.perf_counter() - batch_started) * 1000
        batch_latencies.extend([elapsed_ms / len(case_batch)] * len(case_batch))
        all_scores.extend(float(getattr(score, "item", lambda: score)()) for score in raw_scores)
    total_seconds = time.perf_counter() - started
    result = evaluate_scored_cases(cases, all_scores)
    gpu_memory_mb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    result.update(
        {
            "model": str(model_path),
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "batch_size": batch_size,
            "max_length": max_length,
            "runtime": {
                "total_seconds": round(total_seconds, 3),
                "average_latency_ms": round(statistics.mean(batch_latencies), 3) if batch_latencies else 0.0,
                "p95_latency_ms": round(percentile(batch_latencies, 0.95), 3),
                "peak_cuda_memory_mb": round(gpu_memory_mb, 3),
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU A/B evaluation for evidence-first neural rerank")
    parser.add_argument("--hard-negatives", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ab(
        args.hard_negatives,
        args.dictionary,
        args.model,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
    )
    print(json.dumps({"ok": result["ok"], **result["metrics"], **result["runtime"]}, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
