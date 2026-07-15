from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_query_translator_supervision import (
    build_local_hard_negatives,
    build_query_mappings,
    classify_cblue_pair,
    load_label_pool,
    should_generate_clarify_pair,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


class BuildQueryTranslatorSupervisionTests(unittest.TestCase):
    def test_exact_high_mapping_and_negative_polarity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels = root / "labels.jsonl"
            write_jsonl(
                labels,
                [
                    {"canonical_term": "恶寒", "entry_id": "f1", "source_type": "formula_syndrome", "source_book": "方剂大全", "evidence_tier": "high", "eligible_as_gold_label": True},
                    {"canonical_term": "寸口", "entry_id": "t1", "source_type": "classical_theory", "source_book": "难经", "evidence_tier": "medium", "eligible_as_gold_label": False},
                ],
            )
            write_jsonl(
                root / "processed" / "questions" / "development" / "huatuo26m_lite.jsonl",
                [
                    {"record_id": "q1", "text": "我没有恶寒但头痛", "source_dataset": "huatuo26m_lite", "source_id": "1", "source_hash": "a" * 64},
                    {"record_id": "q2", "text": "为什么独取寸口", "source_dataset": "huatuo26m_lite", "source_id": "2", "source_hash": "a" * 64},
                ],
            )
            high, medium = load_label_pool(labels)
            mapped, rejected, counts = build_query_mappings(root / "processed", high, medium)
            self.assertEqual(len(mapped), 1)
            self.assertEqual(mapped[0]["mappings"][0]["polarity"], "absent")
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["reason"], "medium_evidence_only")
            self.assertEqual(counts["high_confidence_mappings"], 1)

    def test_cblue_relation_policy(self) -> None:
        self.assertEqual(classify_cblue_pair({"source_file": "CHIP-STS_train.json", "label": "1"})[1], True)
        self.assertEqual(classify_cblue_pair({"source_file": "KUAKE-QQR_train.json", "label": "0"})[1], False)
        self.assertEqual(classify_cblue_pair({"source_file": "KUAKE-QQR_train.json", "label": "1"})[1], None)
        self.assertEqual(classify_cblue_pair({"source_file": "KUAKE-QTR_train.json", "label": "3"})[1], True)

    def test_hard_negatives_cover_general_same_type_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dictionary = root / "dictionary.jsonl"
            write_jsonl(
                dictionary,
                [
                    {"entry_id": "f1", "source_type": "formula_syndrome", "formula": "甲方"},
                    {"entry_id": "f2", "source_type": "formula_syndrome", "formula": "乙方"},
                    {"entry_id": "h1", "source_type": "herb_indication", "herb_name": "甲药"},
                    {"entry_id": "h2", "source_type": "herb_indication", "herb_name": "乙药"},
                ],
            )
            high = {
                "恶寒": {"entry_ids": {"f1", "f2"}},
                "无汗": {"entry_ids": {"f1"}},
                "汗出": {"entry_ids": {"f2"}},
                "目痛": {"entry_ids": {"h1", "h2"}},
                "头痛": {"entry_ids": {"h1"}},
                "眩晕": {"entry_ids": {"h2"}},
            }
            negatives = build_local_hard_negatives(dictionary, high, {})
            self.assertEqual(len(negatives), 4)
            self.assertEqual({item["source_type"] for item in negatives}, {"formula_syndrome", "herb_indication"})
            self.assertEqual(sum(item["expected_decision"] == "clarify" for item in negatives), 2)

    def test_medium_entry_is_not_promoted_by_shared_high_term(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dictionary = root / "dictionary.jsonl"
            write_jsonl(
                dictionary,
                [
                    {"entry_id": "high", "source_type": "formula_syndrome", "formula": "甲方"},
                    {"entry_id": "medium", "source_type": "formula_syndrome", "formula": "乙方"},
                ],
            )
            high = {"恶寒": {"entry_ids": {"high"}}, "头痛": {"entry_ids": {"high"}}}
            medium = {"头痛": {"entry_ids": {"medium"}}, "腹痛": {"entry_ids": {"medium"}}}

            negatives = build_local_hard_negatives(dictionary, high, medium)

            self.assertEqual(len(negatives), 2)
            tiers = {negatives[0]["anchor_entry_id"]: negatives[0]["anchor_evidence_tier"], negatives[0]["candidate_entry_id"]: negatives[0]["candidate_evidence_tier"]}
            self.assertEqual(tiers["high"], "high")
            self.assertEqual(tiers["medium"], "medium")

    def test_intervention_name_role_is_excluded_from_hard_negative_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dictionary = root / "dictionary.jsonl"
            write_jsonl(
                dictionary,
                [
                    {"entry_id": "a", "source_type": "formula_syndrome", "formula": "甲方"},
                    {"entry_id": "b", "source_type": "formula_syndrome", "formula": "乙方"},
                ],
            )
            labels = {
                "腹痛": {"entry_ids": {"a", "b"}, "term_roles": {"diagnostic_key"}},
                "呕吐": {"entry_ids": {"a"}, "term_roles": {"required_symptom"}},
                "腹泻": {"entry_ids": {"b"}, "term_roles": {"required_symptom"}},
                "甲方": {"entry_ids": {"a"}, "term_roles": {"formula_name"}},
                "乙方": {"entry_ids": {"b"}, "term_roles": {"formula_name"}},
            }

            negatives = build_local_hard_negatives(dictionary, labels, {})

            self.assertEqual(len(negatives), 2)
            self.assertTrue(all("甲方" not in item["query_terms"] for item in negatives))
            self.assertTrue(all("乙方" not in item["query_terms"] for item in negatives))

    def test_clarify_pair_requires_comparable_payload_evidence(self) -> None:
        anchor = {"required_symptom_groups": [["恶寒"], ["无汗"]], "forbidden_terms": [], "confidence": 0.65}
        comparable = {"required_symptom_groups": [["恶寒"], ["无汗"]], "forbidden_terms": [], "confidence": 0.65}
        incomplete = {"required_symptom_groups": [["腹痛"], ["呕吐"]], "forbidden_terms": [], "confidence": 0.65}
        forbidden = {"required_symptom_groups": [["恶寒"]], "forbidden_terms": ["无汗"], "confidence": 0.65}
        shared = {"恶寒", "无汗"}

        self.assertTrue(should_generate_clarify_pair(anchor, comparable, shared))
        self.assertFalse(should_generate_clarify_pair(anchor, incomplete, shared))
        self.assertFalse(should_generate_clarify_pair(anchor, forbidden, shared))


if __name__ == "__main__":
    unittest.main()
