from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_external_processed_data import validate_processed_data


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


class ValidateExternalProcessedDataTests(unittest.TestCase):
    def test_valid_isolated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            source_hash = "a" * 64
            write_jsonl(
                processed / "questions" / "development" / "huatuo26m_lite.jsonl",
                [{
                    "record_id": "dev1",
                    "text": "最近总是头疼",
                    "source_dataset": "huatuo26m_lite",
                    "source_hash": source_hash,
                    "split_role": "development_only",
                }],
            )
            write_jsonl(
                processed / "questions" / "validation" / "mtcmb.jsonl",
                [{
                    "record_id": "val1",
                    "text": "恶寒发热怎么辨证",
                    "source_dataset": "mtcmb",
                    "source_hash": source_hash,
                    "split_role": "validation_only",
                }],
            )
            write_jsonl(
                processed / "validation_labels" / "mtcmb.jsonl",
                [{"question_record_id": "val1", "source_dataset": "mtcmb", "answer": "A"}],
            )
            report = root / "cleaning.json"
            report.write_text(
                json.dumps({"counts": {"kept_questions": 2, "kept_pairs": 0, "kept_validation_labels": 1}}),
                encoding="utf-8",
            )
            result = validate_processed_data(processed, report, root / "validation.json")
            self.assertTrue(result["ok"])

    def test_detects_answer_and_cross_partition_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            source_hash = "b" * 64
            write_jsonl(
                processed / "questions" / "development" / "huatuo26m_lite.jsonl",
                [{
                    "record_id": "dev1",
                    "text": "头疼发热",
                    "answer": "泄漏",
                    "source_dataset": "huatuo26m_lite",
                    "source_hash": source_hash,
                    "split_role": "development_only",
                }],
            )
            write_jsonl(
                processed / "questions" / "validation" / "mtcmb.jsonl",
                [{
                    "record_id": "val1",
                    "text": "头疼、发热",
                    "source_dataset": "mtcmb",
                    "source_hash": source_hash,
                    "split_role": "validation_only",
                }],
            )
            write_jsonl(
                processed / "validation_labels" / "mtcmb.jsonl",
                [{"question_record_id": "val1", "source_dataset": "mtcmb", "answer": "A"}],
            )
            report = root / "cleaning.json"
            report.write_text(
                json.dumps({"counts": {"kept_questions": 2, "kept_pairs": 0, "kept_validation_labels": 1}}),
                encoding="utf-8",
            )
            result = validate_processed_data(processed, report, root / "validation.json")
            self.assertFalse(result["ok"])
            error_names = {error["error"] for error in result["errors"]}
            self.assertIn("answer_leakage", error_names)
            self.assertIn("cross_partition_exact_duplicate", error_names)


if __name__ == "__main__":
    unittest.main()
