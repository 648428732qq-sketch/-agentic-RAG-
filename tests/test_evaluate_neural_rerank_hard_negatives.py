from __future__ import annotations

import unittest

from scripts.evaluate_neural_rerank_hard_negatives import evaluate_scored_cases
from scripts.evaluate_local_hard_negative_ranking import build_match


def payload(entry_id, formula, symptoms, required):
    return {
        "entry_id": entry_id,
        "source_type": "formula_syndrome",
        "formula": formula,
        "confidence": 0.65,
        "diagnostic_keys": symptoms,
        "ancient_symptoms": symptoms,
        "differential_keys": symptoms,
        "required_symptom_groups": required,
        "forbidden_terms": [],
    }


def case(expected="rank_anchor"):
    query_terms = ["恶寒", "无汗", "喘"]
    query_info = {
        "original_query": " ".join(query_terms),
        "query_intent": "clinical_symptom",
        "canonical_terms": query_terms,
        "primary_canonical_terms": query_terms,
        "negative_terms": [],
        "needs_more_info": False,
    }
    anchor = build_match(payload("a", "甲方", query_terms, [["恶寒"], ["无汗"]]), query_terms)
    candidate = build_match(payload("b", "乙方", ["恶寒", "喘"], [["恶寒"], ["头痛"]]), query_terms)
    return {
        "negative": {"negative_id": "n1", "anchor_entry_id": "a", "expected_decision": expected, "query_terms": query_terms},
        "query_info": query_info,
        "baseline_matches": [anchor, candidate],
    }


class EvaluateNeuralRerankHardNegativesTests(unittest.TestCase):
    def test_evidence_first_does_not_cross_stronger_payload_rank(self) -> None:
        result = evaluate_scored_cases([case()], [0.1, 0.9])
        self.assertEqual(result["counts"].get("worsened", 0), 0)
        self.assertEqual(result["counts"].get("evidence_boundary_violations", 0), 0)

    def test_invalid_score_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_scored_cases([case()], [0.1])


if __name__ == "__main__":
    unittest.main()
