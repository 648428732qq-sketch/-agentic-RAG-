from __future__ import annotations

import unittest

from scripts.evaluate_formula_hard_negatives import (
    _terms_leak_into_targets,
    generate_cases,
    pick_best_formula_entries,
)


def make_formula(
    formula: str,
    required: list[list[str]],
    *,
    diagnostic: list[str] | None = None,
    forbidden: list[str] | None = None,
    differential: list[str] | None = None,
) -> dict:
    return {
        "entry_id": f"formula::{formula}",
        "title": f"{formula}证",
        "source_type": "formula_syndrome",
        "formula": formula,
        "required_symptom_groups": required,
        "diagnostic_keys": diagnostic or [term for group in required for term in group],
        "ancient_symptoms": [term for group in required for term in group],
        "modern_symptoms": [],
        "pathogenesis": [],
        "forbidden_terms": forbidden or [],
        "differential_keys": differential or [],
    }


class FormulaHardNegativeEvalTests(unittest.TestCase):
    def test_generate_cases_covers_more_than_the_mandatory_four_formulas(self) -> None:
        rows = [
            make_formula("麻黄汤", [["恶寒"], ["无汗"], ["喘"]], forbidden=["汗出"]),
            make_formula("桂枝汤", [["恶风"], ["汗出"]], forbidden=["无汗"]),
            make_formula("小青龙汤", [["恶寒"], ["无汗"], ["咳喘"], ["痰多"]], forbidden=["汗出"]),
            make_formula("苏子降气汤", [["咳喘"], ["痰多"], ["胸膈满闷"]], forbidden=["阴虚"]),
            make_formula("银翘散", [["发热"], ["微恶风"], ["咽痛"]], forbidden=["恶寒重"]),
            make_formula("理中丸", [["呕吐"], ["下利"], ["不渴"]], forbidden=["湿热"]),
        ]

        entries = pick_best_formula_entries(rows)
        cases = generate_cases(entries, max_formulas=6)
        tested_formulas = {case["expected_formula"] for case in cases}

        self.assertGreater(len(tested_formulas), 4)
        self.assertIn("银翘散", tested_formulas)
        self.assertIn("理中丸", tested_formulas)

    def test_full_signature_terms_do_not_include_forbidden_differential_terms(self) -> None:
        rows = [
            make_formula(
                "桂枝汤",
                [["恶风"], ["汗出"]],
                diagnostic=["恶风", "汗出", "脉浮缓"],
                forbidden=["无汗"],
                differential=["汗出", "无汗", "营卫不和"],
            )
        ]

        entries = pick_best_formula_entries(rows)
        cases = generate_cases(entries, max_formulas=1)
        full_case = next(case for case in cases if case["style"] == "full_signature_topk")

        self.assertIn("汗出", full_case["expected_terms"])
        self.assertNotIn("无汗", full_case["expected_terms"])
        self.assertNotIn("无汗", full_case["query"])

    def test_missing_required_cases_do_not_leak_overlapping_omitted_terms(self) -> None:
        rows = [
            make_formula(
                "overlap_formula",
                [["喘", "喘咳", "咳喘"], ["咳嗽", "喘咳", "咳喘"], ["痰多"]],
                diagnostic=["咳喘", "咳嗽", "痰多", "胸闷"],
            )
        ]

        entries = pick_best_formula_entries(rows)
        cases = generate_cases(entries, max_formulas=1)
        missing_case = next(case for case in cases if case["style"] == "missing_required_clarify")

        for omitted in missing_case["omitted_required_group"]:
            self.assertNotIn(omitted, missing_case["query"])
            self.assertNotIn(omitted, missing_case["expected_terms"])

    def test_missing_required_cases_do_not_leak_support_term_substrings(self) -> None:
        rows = [
            make_formula(
                "fever_formula",
                [["发热", "身热", "大热", "壮热"], ["汗出"], ["口渴"]],
                diagnostic=["壮热面赤", "汗大出", "口大渴", "烦躁"],
            )
        ]

        entries = pick_best_formula_entries(rows)
        cases = generate_cases(entries, max_formulas=1)
        missing_case = next(case for case in cases if case["style"] == "missing_required_clarify")

        self.assertEqual(["发热", "身热", "大热", "壮热"], missing_case["omitted_required_group"])
        self.assertNotIn("壮热面赤", missing_case["expected_terms"])
        self.assertNotIn("壮热面赤", missing_case["query"])

    def test_missing_required_cases_do_not_leak_forbidden_equivalents(self) -> None:
        rows = [
            make_formula(
                "dry_mouth_formula",
                [["不渴"], ["舌淡"], ["脉沉"]],
                diagnostic=["口干", "舌淡", "脉沉"],
                forbidden=["口渴"],
            )
        ]

        entries = pick_best_formula_entries(rows)
        cases = generate_cases(entries, max_formulas=1)
        missing_case = next(case for case in cases if case["style"] == "missing_required_clarify")

        self.assertFalse(
            _terms_leak_into_targets(
                missing_case["expected_terms"],
                entries[0]["forbidden_terms"],
                entries[0],
            )
        )


if __name__ == "__main__":
    unittest.main()
