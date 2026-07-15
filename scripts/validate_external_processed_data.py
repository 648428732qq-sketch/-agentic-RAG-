from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    from scripts.prepare_external_query_data import DEDUPE_DROP_PATTERN
except ModuleNotFoundError:  # Direct execution sets scripts/ as sys.path[0].
    from prepare_external_query_data import DEDUPE_DROP_PATTERN


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = ROOT / "datasets" / "external" / "processed"
DEFAULT_CLEANING_REPORT = ROOT / "datasets" / "external" / "reports" / "data_cleaning_report.json"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "reports" / "processed_data_validation.json"

HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RESIDUAL_PII_PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)"),
    "phone": re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
}
QUESTION_FORBIDDEN_KEYS = {"answer", "answers", "options", "response", "completion"}
VALIDATION_DATASETS = {"mtcmb", "tcm_ladder"}
DEVELOPMENT_DATASETS = {"cblue_mirror_unverified", "huatuo26m_lite"}


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield line_number, value


def add_error(errors: list[dict[str, Any]], error: dict[str, Any], limit: int) -> None:
    if len(errors) < limit:
        errors.append(error)


def validate_processed_data(
    processed_root: Path,
    cleaning_report_path: Path,
    output_path: Path,
    max_errors: int = 100,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    record_ids: set[str] = set()
    text_keys: set[str] = set()
    validation_question_ids: set[str] = set()
    label_question_ids: set[str] = set()

    question_root = processed_root / "questions"
    for path in sorted(question_root.rglob("*.jsonl")):
        partition = path.parent.name
        for line_number, record in iter_jsonl(path):
            counts["questions"] += 1
            counts[f"questions_{partition}"] += 1
            dataset = str(record.get("source_dataset", ""))
            expected_role = "validation_only" if partition == "validation" else "development_only"
            if record.get("split_role") != expected_role:
                add_error(errors, {"path": str(path), "line": line_number, "error": "split_role_mismatch"}, max_errors)
            allowed = VALIDATION_DATASETS if partition == "validation" else DEVELOPMENT_DATASETS
            if dataset not in allowed:
                add_error(errors, {"path": str(path), "line": line_number, "error": "dataset_partition_mismatch"}, max_errors)
            forbidden = sorted(QUESTION_FORBIDDEN_KEYS & set(record))
            if forbidden:
                add_error(
                    errors,
                    {"path": str(path), "line": line_number, "error": "answer_leakage", "keys": forbidden},
                    max_errors,
                )
            record_id = str(record.get("record_id", ""))
            if not record_id or record_id in record_ids:
                add_error(errors, {"path": str(path), "line": line_number, "error": "duplicate_or_missing_record_id"}, max_errors)
            record_ids.add(record_id)
            if partition == "validation":
                validation_question_ids.add(record_id)
            source_hash = str(record.get("source_hash", ""))
            if not HASH_PATTERN.fullmatch(source_hash):
                add_error(errors, {"path": str(path), "line": line_number, "error": "invalid_source_hash"}, max_errors)
            text = str(record.get("text", ""))
            key = DEDUPE_DROP_PATTERN.sub("", text.casefold())
            if not key or key in text_keys:
                add_error(errors, {"path": str(path), "line": line_number, "error": "cross_partition_exact_duplicate"}, max_errors)
            text_keys.add(key)
            for pii_name, pattern in RESIDUAL_PII_PATTERNS.items():
                if pattern.search(text):
                    counts[f"residual_pii_{pii_name}"] += 1
                    add_error(
                        errors,
                        {"path": str(path), "line": line_number, "error": f"residual_pii_{pii_name}"},
                        max_errors,
                    )

    pair_path = processed_root / "pairs" / "cblue_semantic_pairs.jsonl"
    if pair_path.exists():
        pair_ids: set[str] = set()
        for line_number, record in iter_jsonl(pair_path):
            counts["semantic_pairs"] += 1
            if QUESTION_FORBIDDEN_KEYS & set(record):
                add_error(errors, {"path": str(pair_path), "line": line_number, "error": "pair_answer_leakage"}, max_errors)
            pair_id = str(record.get("pair_id", ""))
            if not pair_id or pair_id in pair_ids:
                add_error(errors, {"path": str(pair_path), "line": line_number, "error": "duplicate_or_missing_pair_id"}, max_errors)
            pair_ids.add(pair_id)
            if record.get("source_dataset") != "cblue_mirror_unverified":
                add_error(errors, {"path": str(pair_path), "line": line_number, "error": "invalid_pair_dataset"}, max_errors)

    label_root = processed_root / "validation_labels"
    for path in sorted(label_root.glob("*.jsonl")):
        for line_number, record in iter_jsonl(path):
            counts["validation_labels"] += 1
            if record.get("source_dataset") not in VALIDATION_DATASETS:
                add_error(errors, {"path": str(path), "line": line_number, "error": "development_label_leakage"}, max_errors)
            question_record_id = str(record.get("question_record_id", ""))
            label_question_ids.add(question_record_id)
            if question_record_id not in validation_question_ids:
                add_error(errors, {"path": str(path), "line": line_number, "error": "orphan_validation_label"}, max_errors)

    missing_labels = validation_question_ids - label_question_ids
    if missing_labels:
        add_error(
            errors,
            {"error": "validation_questions_without_labels", "count": len(missing_labels), "examples": sorted(missing_labels)[:10]},
            max_errors,
        )

    expected = json.loads(cleaning_report_path.read_text(encoding="utf-8"))
    expected_counts = expected.get("counts", {})
    for actual_name, expected_name in (
        ("questions", "kept_questions"),
        ("semantic_pairs", "kept_pairs"),
        ("validation_labels", "kept_validation_labels"),
    ):
        if counts[actual_name] != int(expected_counts.get(expected_name, -1)):
            add_error(
                errors,
                {
                    "error": "cleaning_report_count_mismatch",
                    "field": actual_name,
                    "actual": counts[actual_name],
                    "expected": expected_counts.get(expected_name),
                },
                max_errors,
            )

    result = {
        "ok": not errors,
        "counts": dict(sorted(counts.items())),
        "unique_record_ids": len(record_ids),
        "unique_text_keys": len(text_keys),
        "validation_question_ids": len(validation_question_ids),
        "validation_label_ids": len(label_question_ids),
        "error_count": len(errors),
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate isolated external Query Translator data")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--cleaning-report", type=Path, default=DEFAULT_CLEANING_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-errors", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_processed_data(args.processed_root, args.cleaning_report, args.output, args.max_errors)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
