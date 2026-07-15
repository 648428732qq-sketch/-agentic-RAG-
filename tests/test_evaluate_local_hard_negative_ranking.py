from __future__ import annotations

import unittest

from scripts.evaluate_local_hard_negative_ranking import evaluate_cases


def payload(entry_id, formula, symptoms, required, confidence=0.65):
    return {
        "entry_id": entry_id,
        "source_type": "formula_syndrome",
        "formula": formula,
        "intervention_name": formula,
        "confidence": confidence,
        "diagnostic_keys": symptoms,
        "ancient_symptoms": symptoms,
        "differential_keys": symptoms,
        "required_symptom_groups": required,
        "forbidden_terms": [],
    }


class EvaluateLocalHardNegativeRankingTests(unittest.TestCase):
    def test_distinguishing_term_ranks_anchor(self) -> None:
        entries = {
            "a": payload("a", "甲方", ["恶寒", "无汗", "喘"], [["恶寒"], ["无汗"]]),
            "b": payload("b", "乙方", ["恶寒", "汗出"], [["恶寒"], ["汗出"]]),
        }
        cases = [{
            "negative_id": "n1",
            "anchor_entry_id": "a",
            "candidate_entry_id": "b",
            "query_terms": ["恶寒", "无汗", "喘"],
            "expected_decision": "rank_anchor",
        }]
        result = evaluate_cases(cases, entries)
        self.assertEqual(result["metrics"]["distinguishable_top1"], 1.0)

    def test_indistinguishable_low_confidence_case_clarifies(self) -> None:
        entries = {
            "a": payload("a", "甲方", ["腹痛", "呕吐"], [], confidence=0.52),
            "b": payload("b", "乙方", ["腹痛", "呕吐"], [], confidence=0.52),
        }
        cases = [{
            "negative_id": "n2",
            "anchor_entry_id": "a",
            "candidate_entry_id": "b",
            "query_terms": ["腹痛", "呕吐"],
            "expected_decision": "clarify",
        }]
        result = evaluate_cases(cases, entries)
        self.assertEqual(result["metrics"]["must_clarify_accuracy"], 1.0)
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
