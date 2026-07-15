from __future__ import annotations

import unittest

from project.core.evidence_gate import decision_result, evaluate_evidence_gate, payload_rank_signature


def match(
    *,
    source_type="formula_syndrome",
    intervention_type="formula",
    confidence=0.65,
    required=1.0,
    count=3,
    coverage=1.0,
    missing=None,
    terms=None,
    identity="测试干预",
):
    return {
        "payload": {
            "entry_id": f"{source_type}::{identity}",
            "source_type": source_type,
            "intervention_type": intervention_type,
            "intervention_name": identity,
            "confidence": confidence,
            "review_status": "rule_extracted",
        },
        "canonical_match_count": count,
        "primary_canonical_match_count": count,
        "query_coverage": coverage,
        "required_group_coverage": required,
        "matched_required_symptom_groups": [["症状甲"], ["症状乙"]] if required else [],
        "missing_required_symptom_groups": missing or [],
        "matched_terms": terms or ["症状甲", "症状乙", "症状丙"],
        "negative_conflicts": [],
        "forbidden_conflicts": [],
        "exact_match_count": 0,
    }


class EvidenceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["症状甲", "症状乙", "症状丙"],
            "needs_more_info": False,
        }

    def test_complete_payload_is_grounded_without_source_type_special_case(self) -> None:
        for source_type, intervention_type in (
            ("formula_syndrome", "formula"),
            ("classical_acupuncture", "acupuncture"),
            ("herb_indication", "herb"),
        ):
            result = evaluate_evidence_gate(self.query, [match(source_type=source_type, intervention_type=intervention_type)])
            self.assertEqual(result["status"], "grounded_answer")

    def test_missing_required_group_or_low_confidence_clarifies(self) -> None:
        missing = evaluate_evidence_gate(self.query, [match(required=0.5, missing=[["症状乙"]])])
        low_confidence = evaluate_evidence_gate(self.query, [match(confidence=0.52)])
        self.assertIn("missing_required_symptom_groups", missing["reasons"])
        self.assertIn("payload_evidence_confidence_below_threshold", low_confidence["reasons"])

    def test_indistinguishable_competitor_clarifies(self) -> None:
        first = match(identity="甲方", terms=["症状甲", "症状乙"])
        second = match(identity="乙方", terms=["症状甲", "症状乙"], count=2)
        result = evaluate_evidence_gate(self.query, [first, second])
        self.assertIn("competing_indistinguishable_matches", result["reasons"])

    def test_rank_signature_is_payload_driven(self) -> None:
        self.assertGreater(payload_rank_signature(self.query, match(source_type="unknown_new_type")), 0)
        self.assertGreater(payload_rank_signature(self.query, match(required=0.5, missing=[["症状乙"]])), 0)

    def test_rejection_contains_stable_machine_code_and_user_reason(self) -> None:
        result = evaluate_evidence_gate(self.query, [])

        self.assertEqual(result["response_policy"], "clarify")
        self.assertTrue(result["rejection"]["required"])
        self.assertEqual(result["rejection"]["primary_reason"], "no_grounded_match")
        self.assertIn("Qdrant", result["reason_details"][0]["message"])

    def test_retrieval_failure_reason_is_safe_for_display(self) -> None:
        result = decision_result("no_match", ["retrieval_failure:TimeoutError"])

        self.assertEqual(result["response_policy"], "refuse")
        self.assertIn("TimeoutError", result["reason_details"][0]["message"])


if __name__ == "__main__":
    unittest.main()
