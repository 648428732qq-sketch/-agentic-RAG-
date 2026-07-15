from __future__ import annotations

import unittest

from scripts.split_query_translator_supervision import split_mapping_records, split_semantic_pairs


class SplitQueryTranslatorSupervisionTests(unittest.TestCase):
    def test_canonical_target_terms_do_not_leak(self) -> None:
        records = [
            {"query_id": "q1", "mappings": [{"canonical_term": "恶寒", "evidence_entry_ids": ["f1"]}]},
            {"query_id": "q2", "mappings": [{"canonical_term": "恶寒", "evidence_entry_ids": ["f2"]}]},
            {"query_id": "q3", "mappings": [{"canonical_term": "腹痛", "evidence_entry_ids": ["f2"]}]},
            {"query_id": "q4", "mappings": [{"canonical_term": "眩晕", "evidence_entry_ids": ["f3"]}]},
        ]
        splits, report = split_mapping_records(records, dev_fraction=0.5, seed="test")
        self.assertFalse(report["term_overlap"])
        q1_split = "train" if any(record["query_id"] == "q1" for record in splits["train"]) else "dev"
        q2_split = "train" if any(record["query_id"] == "q2" for record in splits["train"]) else "dev"
        self.assertEqual(q1_split, q2_split)

    def test_record_with_targets_in_both_splits_is_excluded(self) -> None:
        terms = [f"症状{index}" for index in range(100)]
        records = [
            {"query_id": f"q{index}", "mappings": [{"canonical_term": term, "evidence_entry_ids": [f"e{index}"]}]}
            for index, term in enumerate(terms)
        ]
        splits, _ = split_mapping_records(records, dev_fraction=0.5, seed="test")
        train_term = splits["train"][0]["mappings"][0]["canonical_term"]
        dev_term = splits["dev"][0]["mappings"][0]["canonical_term"]
        mixed = {
            "query_id": "mixed",
            "mappings": [
                {"canonical_term": train_term, "evidence_entry_ids": ["x"]},
                {"canonical_term": dev_term, "evidence_entry_ids": ["y"]},
            ],
        }
        outputs, _ = split_mapping_records(records + [mixed], dev_fraction=0.5, seed="test")
        self.assertTrue(any(record["query_id"] == "mixed" for record in outputs["excluded"]))

    def test_shared_pair_text_stays_in_one_split(self) -> None:
        records = [
            {"pair_id": "p1", "text_a": "头痛怎么办", "text_b": "脑袋疼咋办"},
            {"pair_id": "p2", "text_a": "脑袋疼咋办", "text_b": "头疼怎么处理"},
            {"pair_id": "p3", "text_a": "腹痛", "text_b": "肚子疼"},
        ]
        splits, report = split_semantic_pairs(records, dev_fraction=0.5, seed="test")
        self.assertEqual(report["text_overlap_count"], 0)
        p1_split = "train" if any(record["pair_id"] == "p1" for record in splits["train"]) else "dev"
        p2_split = "train" if any(record["pair_id"] == "p2" for record in splits["train"]) else "dev"
        self.assertEqual(p1_split, p2_split)


if __name__ == "__main__":
    unittest.main()
