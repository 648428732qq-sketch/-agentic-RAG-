from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_query_translator_data_pipeline import ROOT, pipeline_steps, project_venv_ok


class RunQueryTranslatorDataPipelineTests(unittest.TestCase):
    def test_steps_are_ordered_and_use_current_python(self) -> None:
        names = [name for name, _ in pipeline_steps()]
        self.assertEqual(
            names,
            [
                "raw_manifest",
                "clean_external_data",
                "validate_processed_data",
                "audit_payload_evidence",
                "build_supervision",
                "split_supervision",
                "evaluate_local_hard_negatives",
                "data_pipeline_tests",
            ],
        )

    def test_project_venv_boundary(self) -> None:
        self.assertTrue(project_venv_ok(ROOT / ".venv" / "Scripts" / "python.exe"))
        self.assertTrue(project_venv_ok(ROOT / ".venv-linux" / "bin" / "python"))
        self.assertFalse(project_venv_ok(Path("C:/Python/python.exe")))


if __name__ == "__main__":
    unittest.main()
