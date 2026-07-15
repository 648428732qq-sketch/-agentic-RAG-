from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.filter_chatmed_candidate_terms import FILTERED_ORIGIN, FILTERED_STATUS, filter_candidates


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def syndrome_entry(**overrides: object) -> dict:
    entry = {
        "entry_id": "formula::测试方",
        "title": "测试方证",
        "source_type": "formula_syndrome",
        "formula": "测试方",
        "herb_name": "",
        "diagnostic_keys": ["恶寒", "发热", "汗出"],
        "ancient_symptoms": [],
        "modern_symptoms": ["怕冷", "发烧", "出汗"],
        "symptom_aliases": ["冷得明显", "身上发热", "冒汗"],
        "pathogenesis": ["外感风寒"],
    }
    entry.update(overrides)
    return entry


class FilterChatMedCandidateTermsTests(unittest.TestCase):
    def test_filters_high_frequency_diagnostic_and_negation_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chatmed = root / "chatmed.jsonl"
            dictionary = root / "syndrome.jsonl"
            output = root / "out"
            write_jsonl(
                chatmed,
                [
                    {"query": "我怕冷发烧", "response": "隐藏答案A"},
                    {"query": "最近怕冷，还有点发烧", "response": "隐藏答案B"},
                    {"query": "怕冷，没有出汗", "response": "隐藏答案C"},
                    {"query": "请问外感风寒怎么判断？", "response": "隐藏答案D"},
                ],
            )
            write_jsonl(dictionary, [syndrome_entry()])

            summary = filter_candidates(
                chatmed_path=chatmed,
                syndrome_dictionary_path=dictionary,
                output_dir=output,
                min_query_count=2,
                max_local_entries=5,
                max_rows_per_file=100,
            )

            self.assertEqual(summary["review_status"], FILTERED_STATUS)
            self.assertEqual(summary["evidence_origin"], FILTERED_ORIGIN)

            diagnostic_text = (output / "diagnostic_term_review_queue.jsonl").read_text(encoding="utf-8")
            negation_text = (output / "negation_forbidden_review_queue.jsonl").read_text(encoding="utf-8")
            pathogenesis_text = (output / "pathogenesis_query_review_queue.jsonl").read_text(encoding="utf-8")

            self.assertIn("怕冷", diagnostic_text)
            self.assertIn('"priority": "high"', diagnostic_text)
            self.assertIn("出汗", negation_text)
            self.assertIn("外感风寒", pathogenesis_text)
            self.assertNotIn("隐藏答案", diagnostic_text + negation_text + pathogenesis_text)

    def test_broad_terms_are_not_marked_high_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chatmed = root / "chatmed.jsonl"
            dictionary = root / "syndrome.jsonl"
            output = root / "out"
            write_jsonl(
                chatmed,
                [
                    {"query": "我腹痛", "response": "隐藏答案A"},
                    {"query": "最近腹痛", "response": "隐藏答案B"},
                ],
            )
            write_jsonl(
                dictionary,
                [
                    syndrome_entry(entry_id="formula::A", title="A", diagnostic_keys=["腹痛"]),
                    syndrome_entry(entry_id="formula::B", title="B", diagnostic_keys=["腹痛"]),
                    syndrome_entry(entry_id="formula::C", title="C", diagnostic_keys=["腹痛"]),
                ],
            )

            filter_candidates(
                chatmed_path=chatmed,
                syndrome_dictionary_path=dictionary,
                output_dir=output,
                min_query_count=2,
                max_local_entries=1,
                max_rows_per_file=100,
            )

            rows = [
                json.loads(line)
                for line in (output / "diagnostic_term_review_queue.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(rows)
            self.assertTrue(all(row["priority"] != "high" for row in rows))
            self.assertTrue(any("broad_local_term" in row["risk_flags"] for row in rows))

    def test_generic_symptom_terms_are_downgraded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chatmed = root / "chatmed.jsonl"
            dictionary = root / "syndrome.jsonl"
            output = root / "out"
            write_jsonl(
                chatmed,
                [
                    {"query": "我头晕", "response": "隐藏答案A"},
                    {"query": "最近头晕", "response": "隐藏答案B"},
                    {"query": "总是头晕", "response": "隐藏答案C"},
                ],
            )
            write_jsonl(
                dictionary,
                [
                    syndrome_entry(
                        entry_id="formula::A",
                        title="A",
                        diagnostic_keys=["头晕"],
                        modern_symptoms=[],
                        symptom_aliases=[],
                    ),
                ],
            )

            filter_candidates(
                chatmed_path=chatmed,
                syndrome_dictionary_path=dictionary,
                output_dir=output,
                min_query_count=2,
                max_local_entries=8,
                max_rows_per_file=100,
            )

            rows = [
                json.loads(line)
                for line in (output / "diagnostic_term_review_queue.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(rows)
            self.assertTrue(all(row["priority"] != "high" for row in rows))
            self.assertTrue(any("generic_symptom" in row["risk_flags"] for row in rows))


if __name__ == "__main__":
    unittest.main()
