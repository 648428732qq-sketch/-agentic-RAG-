from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_external_datasets import build_manifest, inspect_json_records


class ExternalDatasetAuditTests(unittest.TestCase):
    def test_schema_counts_fields_and_types(self) -> None:
        result = inspect_json_records(
            [
                {"query": "头痛", "label": 1},
                {"query": "腹痛", "label": "positive"},
            ]
        )

        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["fields"]["query"]["types"], {"str": 2})
        self.assertEqual(result["fields"]["label"]["types"], {"int": 1, "str": 1})

    def test_manifest_hashes_and_parses_json_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "sample"
            dataset.mkdir()
            (dataset / "rows.json").write_text(
                json.dumps([{"query": "头痛"}], ensure_ascii=False), encoding="utf-8"
            )
            (dataset / "rows.jsonl").write_text(
                json.dumps({"query": "腹痛"}, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            manifest = build_manifest(root)

        self.assertEqual(manifest["file_count"], 2)
        self.assertEqual(manifest["parse_error_count"], 0)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))
        self.assertEqual({item["record_count"] for item in manifest["files"]}, {1})


if __name__ == "__main__":
    unittest.main()
