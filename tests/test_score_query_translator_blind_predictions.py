from __future__ import annotations

import unittest

from scripts.score_query_translator_blind_predictions import score_case, summarize


class BlindPredictionScoringTests(unittest.TestCase):
    def test_perfect_local_prediction_scores_all_checks(self) -> None:
        prediction = {
            "id": "case_1",
            "canonical_terms": ["恶寒", "无汗"],
            "candidate_terms": ["恶寒", "无汗", "喘"],
            "negative_terms": ["喘"],
            "needs_more_info": True,
            "gate": False,
            "decision": {"status": "clarify"},
            "top_k": [{"formula": "测试方", "source_type": "formula_syndrome"}],
            "latency_ms": 100,
        }
        gold = {
            "id": "case_1",
            "generation_style": "hard_negative_clarify",
            "expected_term_groups": [["恶寒"], ["无汗"]],
            "expected_negative_terms": ["喘"],
            "forbidden_terms": [],
            "expected_formula_in_top_k": "测试方",
            "expected_source_type_in_top_k": "formula_syndrome",
            "expected_needs_more_info": True,
            "expected_gate": False,
            "expected_decision": "clarify",
            "must_clarify": True,
        }

        case = score_case(prediction, gold)
        report = summarize([case])

        self.assertTrue(case["ok"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["metrics"]["term_recall"], 1.0)

    def test_expected_absent_sweat_rejects_positive_sweat_conflict(self) -> None:
        prediction = {
            "id": "case_conflict",
            "canonical_terms": ["恶寒", "无汗", "汗出"],
            "candidate_terms": ["恶寒", "无汗", "汗出"],
            "negative_terms": [],
            "needs_more_info": False,
            "gate": True,
            "decision": {"status": "grounded_answer"},
            "top_k": [{"formula": "测试方", "source_type": "formula_syndrome"}],
            "latency_ms": 100,
        }
        gold = {
            "id": "case_conflict",
            "generation_style": "metaphor_incomplete",
            "expected_term_groups": [["恶寒"], ["无汗"]],
            "expected_negative_terms": [],
            "forbidden_terms": [],
            "expected_formula_in_top_k": "测试方",
            "expected_gate": True,
            "expected_decision": "grounded_answer",
            "must_clarify": False,
        }

        case = score_case(prediction, gold)

        self.assertFalse(case["ok"])
        self.assertFalse(case["checks"]["no_expected_positive_conflict"])
        self.assertEqual(case["positive_conflicts"], ["汗出"])


if __name__ == "__main__":
    unittest.main()
