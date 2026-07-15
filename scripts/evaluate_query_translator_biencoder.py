from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from core.query_translator_biencoder import (  # noqa: E402
    build_term_catalog,
    collapse_catalog_scores,
    cosine_scores,
    expected_groups_from_frozen,
    expected_terms_from_supervision,
    iter_jsonl,
    load_catalog,
    score_predictions,
)


DEFAULT_LABELS = ROOT / "datasets" / "structured" / "query_translator_evidence_label_pool.jsonl"
DEFAULT_DEV = ROOT / "datasets" / "external" / "splits" / "dev" / "query_term_pairs.jsonl"
DEFAULT_EXCLUDED_STYLES = {"regional_dialect"}


def load_eval_cases(path: Path, kind: str, excluded_styles: set[str] | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    excluded_styles = excluded_styles or set()
    for index, record in enumerate(iter_jsonl(path), start=1):
        style = str(record.get("generation_style", ""))
        if style in excluded_styles:
            continue
        if kind == "supervision":
            groups = [{term} for term in sorted(expected_terms_from_supervision(record))]
        else:
            groups = expected_groups_from_frozen(record)
        if not groups:
            continue
        cases.append(
            {
                "id": str(record.get("id") or record.get("query_id") or index),
                "query": str(record.get("query", "")).strip(),
                "style": style,
                "expected_groups": [sorted(group) for group in groups if group],
            }
        )
    return cases


def evaluate(
    model_path: Path,
    cases_path: Path,
    kind: str,
    catalog_path: Path | None,
    labels_path: Path,
    output_path: Path,
    device: str,
    batch_size: int,
    max_top_k: int,
    excluded_styles: set[str] | None = None,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    model.max_seq_length = 160
    catalog = load_catalog(catalog_path) if catalog_path else build_term_catalog(labels_path)
    cases = load_eval_cases(cases_path, kind, excluded_styles)
    catalog_embeddings = model.encode(
        [item.search_text for item in catalog],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        [case["query"] for case in cases],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    matrix = cosine_scores(query_embeddings, catalog_embeddings)
    for case, scores in zip(cases, matrix):
        case["predictions"] = collapse_catalog_scores(catalog, scores, max_top_k)
    metrics = score_predictions(cases, (1, 3, 5, 10, max_top_k))
    by_style: dict[str, dict[str, Any]] = {}
    for style in sorted({str(case.get("style", "")) for case in cases if case.get("style")}):
        style_cases = [case for case in cases if case.get("style") == style]
        by_style[style] = score_predictions(style_cases, (5, 10, max_top_k))
    report = {
        "report_version": 1,
        "evaluation_kind": kind,
        "model_path": str(model_path.resolve()),
        "cases_path": str(cases_path.resolve()),
        "catalog_path": str(catalog_path.resolve()) if catalog_path else str(labels_path.resolve()),
        "catalog_item_count": len(catalog),
        "catalog_term_count": len({item.canonical_term for item in catalog}),
        "excluded_styles": sorted(excluded_styles or []),
        "metrics": metrics,
        "by_style": by_style,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the offline Query Translator bi-encoder")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--kind", choices=("supervision", "frozen"), default="supervision")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-top-k", type=int, default=20)
    parser.add_argument(
        "--include-dialect",
        action="store_true",
        help="Include archived regional_dialect cases; excluded by current project policy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        args.model,
        args.cases,
        args.kind,
        args.catalog,
        args.labels,
        args.output,
        args.device,
        args.batch_size,
        args.max_top_k,
        set() if args.include_dialect else DEFAULT_EXCLUDED_STYLES,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
