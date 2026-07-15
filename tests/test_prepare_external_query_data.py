from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_external_query_data import NearDuplicateIndex, clean_text, process_datasets


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


class PrepareExternalQueryDataTests(unittest.TestCase):
    def test_clean_text_masks_pii_and_html(self) -> None:
        text, stats = clean_text("<b>我叫张三</b>，电话 13800138000，a@example.com\x00")
        self.assertEqual(text, "[NAME]，电话 [PHONE]，[EMAIL]")
        self.assertEqual(stats["html_removed"], 1)
        self.assertEqual(stats["pii_name_masked"], 1)
        self.assertEqual(stats["pii_phone_masked"], 1)
        self.assertEqual(stats["pii_email_masked"], 1)
        self.assertEqual(stats["control_removed"], 1)

    def test_near_duplicate_index(self) -> None:
        index = NearDuplicateIndex(threshold=0.80)
        self.assertIsNone(index.find_or_add("我今天头痛发热并且怕冷"))
        self.assertEqual(index.find_or_add("我今天头痛、发热并且怕冷"), 0)
        self.assertIsNone(index.find_or_add("我最近总是腹泻"))

    def test_processing_separates_answers_and_deduplicates_validation_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            huatuo = raw / "huatuo26m_lite" / "data.jsonl"
            mtcmb = raw / "mtcmb" / "test.jsonl"
            write_jsonl(
                huatuo,
                [
                    {"id": 1, "question": "头疼发热怎么办", "answer": "不得导出", "label": "内科"},
                    {"id": 2, "question": "我叫李四，电话13900139000，肚子痛", "answer": "不得导出"},
                ],
            )
            write_jsonl(
                mtcmb,
                [{"id": 9, "question": "头疼发热怎么办", "answer": "仅验证", "source": "fixture"}],
            )
            from scripts.audit_external_datasets import sha256_file

            manifest = {
                "dataset_metadata": {
                    "huatuo26m_lite": {"license": "Apache-2.0", "usage": "question_text_only"},
                    "mtcmb": {"license": "CC-BY-4.0", "usage": "validation_only"},
                },
                "files": [
                    {"path": "huatuo26m_lite/data.jsonl", "sha256": sha256_file(huatuo), "parse_status": "ok"},
                    {"path": "mtcmb/test.jsonl", "sha256": sha256_file(mtcmb), "parse_status": "ok"},
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "processed"
            report_path = root / "report.json"

            report = process_datasets(raw, manifest_path, output, report_path)

            validation_lines = (output / "questions" / "validation" / "mtcmb.jsonl").read_text(encoding="utf-8").splitlines()
            development_lines = (output / "questions" / "development" / "huatuo26m_lite.jsonl").read_text(encoding="utf-8").splitlines()
            label_lines = (output / "validation_labels" / "mtcmb.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(validation_lines), 1)
            self.assertEqual(len(development_lines), 1)
            self.assertNotIn("answer", development_lines[0])
            self.assertNotIn("不得导出", development_lines[0])
            self.assertIn("[NAME]", development_lines[0])
            self.assertIn("[PHONE]", development_lines[0])
            self.assertIn("仅验证", label_lines[0])
            self.assertEqual(report["counts"]["dropped_exact_duplicate"], 1)

    def test_cblue_pairs_survive_question_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            cblue = raw / "cblue_mirror_unverified" / "KUAKE-QQR_train.json"
            cblue.parent.mkdir(parents=True, exist_ok=True)
            cblue.write_text(
                json.dumps(
                    [
                        {"id": "1", "query1": "总是头疼怎么办", "query2": "头痛如何处理", "label": "1"},
                        {"id": "2", "query1": "总是头疼怎么办", "query2": "发热如何处理", "label": "0"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            from scripts.audit_external_datasets import sha256_file

            manifest = {
                "dataset_metadata": {
                    "cblue_mirror_unverified": {"license": "unverified", "usage": "development_only_no_redistribution"}
                },
                "files": [
                    {
                        "path": "cblue_mirror_unverified/KUAKE-QQR_train.json",
                        "sha256": sha256_file(cblue),
                        "parse_status": "ok",
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = process_datasets(raw, manifest_path, root / "processed", root / "report.json")

            pairs = (root / "processed" / "pairs" / "cblue_semantic_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(pairs), 2)
            self.assertEqual(report["counts"]["kept_pairs"], 2)
            self.assertEqual(report["counts"]["dropped_exact_duplicate"], 1)

    def test_managed_outputs_are_truncated_when_source_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            output = root / "processed"
            stale = output / "pairs" / "cblue_semantic_pairs.jsonl"
            stale.parent.mkdir(parents=True)
            stale.write_text('{"stale":true}\n', encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"dataset_metadata": {}, "files": []}), encoding="utf-8")

            process_datasets(raw, manifest_path, output, root / "report.json")

            self.assertEqual(stale.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
