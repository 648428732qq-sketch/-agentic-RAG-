from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_syndrome_payload_evidence import audit_entry, run_audit


def entry(**overrides):
    value = {
        "entry_id": "formula::test",
        "source_type": "formula_syndrome",
        "source_book": "方剂大全",
        "source_file": "formula.md",
        "review_status": "rule_extracted",
        "confidence": 0.65,
        "evidence": "恶寒发热，无汗而喘，脉浮紧。",
        "raw_text": "",
        "ancient_symptoms": ["恶寒", "发热", "无汗", "喘", "脉浮紧"],
        "diagnostic_keys": ["恶寒", "无汗", "喘", "脉"],
        "differential_keys": ["无汗", "脉浮紧"],
        "required_symptom_groups": [["恶寒"], ["无汗"], ["喘"]],
        "forbidden_terms": ["汗出"],
        "formula": "测试方",
    }
    value.update(overrides)
    return value


class AuditSyndromePayloadEvidenceTests(unittest.TestCase):
    def test_formula_with_grounded_required_groups_is_high(self) -> None:
        result = audit_entry(entry())
        self.assertEqual(result["tier"], "high")
        self.assertIn("脉", result["broad_terms"])
        self.assertNotIn("脉", result["supported_terms"])

    def test_conflicting_or_noisy_entry_is_quarantined(self) -> None:
        result = audit_entry(entry(forbidden_terms=["无汗"], evidence="元素。恶寒无汗而喘"))
        self.assertEqual(result["tier"], "quarantine")
        self.assertIn("required_forbidden_conflict", result["severe_reasons"])
        self.assertIn("known_source_noise", result["severe_reasons"])

    def test_medium_terms_are_candidates_but_not_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps(entry(source_type="classical_clause", confidence=0.52), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            report = run_audit(
                input_path,
                root / "verified.jsonl",
                root / "quarantine.jsonl",
                root / "labels.jsonl",
                root / "report.json",
            )
            labels = [json.loads(line) for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["tier_counts"], {"medium": 1})
            self.assertTrue(labels)
            self.assertTrue(all(not label["eligible_as_gold_label"] for label in labels))

    def test_intervention_and_differential_only_terms_are_not_gold_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            value = entry(
                formula="测试方",
                evidence="测试方用于恶寒无汗，汗出者不宜。",
                ancient_symptoms=["恶寒", "无汗"],
                diagnostic_keys=["恶寒", "无汗"],
                differential_keys=["汗出"],
                required_symptom_groups=[["恶寒"], ["无汗"]],
            )
            input_path = root / "input.jsonl"
            input_path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
            run_audit(
                input_path,
                root / "verified.jsonl",
                root / "quarantine.jsonl",
                root / "labels.jsonl",
                root / "report.json",
            )
            labels = {row["canonical_term"]: row for row in map(json.loads, (root / "labels.jsonl").read_text(encoding="utf-8").splitlines())}
            self.assertTrue(labels["恶寒"]["eligible_as_gold_label"])
            self.assertFalse(labels["汗出"]["eligible_as_gold_label"])
            self.assertFalse(labels["测试方"]["eligible_as_gold_label"])


if __name__ == "__main__":
    unittest.main()
