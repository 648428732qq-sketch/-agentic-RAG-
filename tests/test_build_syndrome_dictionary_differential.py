from __future__ import annotations

import unittest

from scripts.build_syndrome_dictionary import FormulaHerb, SyndromeEntry, sanitize_entries


class BuildSyndromeDictionaryDifferentialTests(unittest.TestCase):
    def test_formula_entry_derives_required_forbidden_and_clarify_fields(self) -> None:
        entry = SyndromeEntry(
            entry_id="formula::mahuang_tang",
            title="\u9ebb\u9ec4\u6c64\u8bc1",
            source_type="formula_syndrome",
            formula="\u9ebb\u9ec4\u6c64",
            syndrome_name="\u5916\u611f\u98ce\u5bd2\u8868\u5b9e\u8bc1",
            ancient_symptoms=[
                "\u6076\u5bd2",
                "\u53d1\u70ed",
                "\u5934\u75db",
                "\u8eab\u75bc",
                "\u65e0\u6c57",
                "\u5598",
            ],
            modern_symptoms=[
                "\u6015\u51b7",
                "\u53d1\u70e7",
                "\u6d51\u8eab\u9178\u75db",
                "\u4e0d\u51fa\u6c57",
                "\u61cb\u6c14\u5598",
            ],
            diagnostic_keys=["\u6076\u5bd2", "\u65e0\u6c57", "\u5598"],
            pathogenesis=[
                "\u5916\u611f\u98ce\u5bd2",
                "\u5bd2\u90aa\u675f\u8868",
                "\u80ba\u6c14\u5931\u5ba3",
            ],
            formula_composition=[
                FormulaHerb(name="\u9ebb\u9ec4", original_dose="\u4e09\u4e24\uff089g\uff09"),
            ],
            indications="\u5916\u611f\u98ce\u5bd2\u8868\u5b9e\u8bc1\u3002\u6076\u5bd2\u53d1\u70ed\uff0c\u5934\u8eab\u75bc\u75db\uff0c\u65e0\u6c57\u800c\u5598\u3002",
            contraindications="\u8868\u865a\u81ea\u6c57\u8005\u5fcc\u7528\u3002",
            evidence="\u4f24\u5bd2\u8bba\u9ebb\u9ec4\u6c64\u6761\u6587",
        )

        sanitized = sanitize_entries([entry])[0]

        self.assertEqual(sanitized.formula_composition[0].original_dose, "\u4e09\u4e24")
        flattened_required = {term for group in sanitized.required_symptom_groups for term in group}
        self.assertIn("\u6076\u5bd2", flattened_required)
        self.assertIn("\u65e0\u6c57", flattened_required)
        self.assertIn("\u5598", flattened_required)
        self.assertIn("\u6c57\u51fa", sanitized.forbidden_terms)
        self.assertIn("\u81ea\u6c57", sanitized.forbidden_terms)
        self.assertNotIn("\u65e0\u6c57", sanitized.forbidden_terms)
        self.assertIn("\u8868\u5b9e", sanitized.differential_keys)
        self.assertIn("\u662f\u5426\u51fa\u6c57", sanitized.must_clarify_fields)
        self.assertIn("\u5fc5\u9700\u75c7\u72b6\u7ec4", sanitized.search_text)
        self.assertIn("\u6392\u9664\u75c7\u72b6", sanitized.search_text)

    def test_required_groups_remove_mutually_exclusive_thirst_terms(self) -> None:
        entry = SyndromeEntry(
            entry_id="formula::sanren_tang",
            title="三仁汤证",
            source_type="formula_syndrome",
            formula="三仁汤",
            syndrome_name="湿温初起",
            ancient_symptoms=["身热", "口干", "不渴", "苔白"],
            modern_symptoms=["身体发热", "口干", "不口渴"],
            diagnostic_keys=["身热", "口干", "不渴", "苔白"],
            required_symptom_groups=[
                ["发热", "身热"],
                ["口渴", "口大渴", "烦渴引饮", "口干"],
                ["不渴"],
            ],
            evidence="身热不渴，口干，苔白。",
        )

        sanitized = sanitize_entries([entry])[0]
        flattened_required = {term for group in sanitized.required_symptom_groups for term in group}

        self.assertIn("口干", flattened_required)
        self.assertIn("不渴", flattened_required)
        self.assertNotIn("口渴", flattened_required)
        self.assertIn("口渴", sanitized.forbidden_terms)


if __name__ == "__main__":
    unittest.main()
