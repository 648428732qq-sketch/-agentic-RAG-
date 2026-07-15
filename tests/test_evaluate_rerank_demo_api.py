from __future__ import annotations

import unittest

from scripts.evaluate_rerank_demo_api import evaluate_output, parse_summary


class RerankDemoApiTest(unittest.TestCase):
    def test_parse_summary(self) -> None:
        summary = """### 对比结果
- 基线：582.0 ms；status=clarify; gate=False; reasons=missing_required_symptom_groups
- Rerank：725.5 ms；status=grounded_answer; gate=True; reasons=-
"""
        parsed = parse_summary(summary)
        self.assertEqual(parsed["baseline"]["status"], "clarify")
        self.assertFalse(parsed["baseline"]["gate"])
        self.assertEqual(parsed["rerank"]["latency_ms"], 725.5)
        self.assertEqual(parsed["rerank"]["reasons"], [])

    def test_missing_required_case(self) -> None:
        case = {
            "id": "x",
            "style": "missing_required_clarify",
            "expected_formula": "麻黄汤",
        }
        rows = [[1, "麻黄汤", "formula_syndrome", 0.9, "恶寒", '[["喘"]]', "-"]]
        state = {
            "latency_ms": 10.0,
            "status": "clarify",
            "gate": False,
            "reasons": ["missing_required_symptom_groups"],
        }
        result = evaluate_output(case, rows, state)
        self.assertTrue(result["ok"])
        self.assertEqual(result["target_rank"], 1)


if __name__ == "__main__":
    unittest.main()
