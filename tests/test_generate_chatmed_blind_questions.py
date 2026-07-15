from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_chatmed_blind_questions import (
    STYLES,
    STYLE_PROMPTS,
    build_plan,
    clean_source_question,
    export_isolated_views,
    parse_json_object,
    validate_generation,
)


def make_seed(style: str = "colloquial_dialect", behavior: str = "answer") -> dict:
    return {
        "seed_id": "seed_fuzzy_0001",
        "category_key": "fuzzy_colloquial",
        "category": "模糊口语、隐喻表达",
        "generation_style": style,
        "expected_behavior": behavior,
        "source_dataset": "ChatMed_TCM-v0.2.json",
        "source_line": 7,
        "source_question_hash": "a" * 64,
        "source_question": "我头痛还怕冷，一直不出汗，这是怎么回事？",
        "reference_answer": "需要结合其他表现辨证，不能仅凭这些症状直接决定方剂。",
        "source_answer_status": "imported_unverified",
    }


class ChatMedBlindQuestionTests(unittest.TestCase):
    def test_every_style_has_a_distinct_prompt(self) -> None:
        self.assertEqual(set(STYLES), set(STYLE_PROMPTS))
        normalized = {" ".join(prompt.split()) for prompt in STYLE_PROMPTS.values()}
        self.assertEqual(len(normalized), len(STYLE_PROMPTS))

    def test_meta_instruction_is_removed(self) -> None:
        source = "我肚子疼。要求：1. 请考虑所有症状。2. 请输出推理过程。"
        self.assertEqual(clean_source_question(source), "我肚子疼")

    def test_missing_no_sweat_is_validated(self) -> None:
        plan = build_plan(make_seed("missing_no_sweat", "clarify"), 0)
        generated = {
            "history": [],
            "query": "我这两天头疼还特别怕冷，这是咋了？",
            "only_intended_change": True,
            "reference_answer_still_applicable": False,
            "intentional_omissions": ["无汗"],
            "added_facts": [],
            "answer_leakage": False,
            "typo_pairs": [],
        }
        self.assertEqual(validate_generation(plan, generated), [])

    def test_typo_pairs_must_exist(self) -> None:
        plan = build_plan(make_seed("input_typos"), 0)
        generated = {
            "history": [],
            "query": "我头藤还怕冷，一直不出汉，这是怎么回事？",
            "only_intended_change": True,
            "reference_answer_still_applicable": True,
            "intentional_omissions": [],
            "added_facts": [],
            "answer_leakage": False,
            "typo_pairs": [
                {"correct": "头疼", "typo": "头藤"},
                {"correct": "出汗", "typo": "出汉"},
            ],
        }
        self.assertEqual(validate_generation(plan, generated), [])

    def test_multi_turn_requires_user_and_assistant_history(self) -> None:
        plan = build_plan(make_seed("multi_turn"), 0)
        generated = {
            "history": [
                {"role": "user", "content": "我这两天头疼怕冷。"},
                {"role": "assistant", "content": "出汗情况怎么样？"},
            ],
            "query": "一直都不出汗，这是咋回事？",
            "only_intended_change": True,
            "reference_answer_still_applicable": True,
            "intentional_omissions": [],
            "added_facts": [],
            "answer_leakage": False,
            "typo_pairs": [],
        }
        self.assertEqual(validate_generation(plan, generated), [])

    def test_answer_key_is_not_exported_to_public_questions(self) -> None:
        record = {
            "id": "case_1",
            "history": [],
            "query": "我头疼怕冷，这是咋回事？",
            "category_key": "fuzzy_colloquial",
            "category": "模糊口语、隐喻表达",
            "generation_style": "colloquial_dialect",
            "expected_behavior": "answer",
            "intentional_omissions": [],
            "paraphrase_family_id": "family_1",
            "source_question": "原始问题",
            "reference_answer": "隐藏参考答案",
            "source_question_hash": "a" * 64,
            "source_line": 1,
            "source_answer_status": "imported_unverified",
            "review_status": "auto_generated_unreviewed",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "private" / "master.jsonl"
            master.parent.mkdir(parents=True)
            master.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            export_isolated_views(master, root)
            public = (root / "questions" / "all_questions.jsonl").read_text(encoding="utf-8")
            private = (root / "private" / "answer_keys" / "fuzzy_colloquial.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("隐藏参考答案", public)
            self.assertIn("隐藏参考答案", private)

    def test_list_content_and_fenced_json_are_supported(self) -> None:
        value = parse_json_object([{"text": "```json\n{\"query\":\"测试问句\"}\n```"}])
        self.assertEqual(value["query"], "测试问句")


if __name__ == "__main__":
    unittest.main()
