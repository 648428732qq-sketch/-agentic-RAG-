from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_runtime_environment import inspect_utf8_file, is_utf8_encoding


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_utf8_encoding_variants(self) -> None:
        self.assertTrue(is_utf8_encoding("utf-8"))
        self.assertTrue(is_utf8_encoding("UTF_8"))
        self.assertTrue(is_utf8_encoding("utf-8-sig"))

    def test_non_utf8_encoding_is_rejected(self) -> None:
        self.assertFalse(is_utf8_encoding("cp936"))
        self.assertFalse(is_utf8_encoding(None))

    def test_utf8_file_inspection_accepts_chinese(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "中医.txt"
            path.write_text("恶寒、无汗、身痛", encoding="utf-8")
            result = inspect_utf8_file(path)

        self.assertTrue(result["ok"])
        self.assertFalse(result["bom"])

    def test_utf8_file_inspection_rejects_invalid_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.txt"
            path.write_bytes(b"\xff\xfe\x00")
            result = inspect_utf8_file(path)

        self.assertFalse(result["ok"])
        self.assertIn("UnicodeDecodeError", result["error"])


if __name__ == "__main__":
    unittest.main()
