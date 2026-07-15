from __future__ import annotations

import unittest

from scripts.run_chatmed_gold_eval import behavior_heuristic, bigram_f1, extract_final_answer


class ChatMedGoldEvalTests(unittest.TestCase):
    def test_extracts_last_plain_assistant_answer(self) -> None:
        emission = [
            {"role": "assistant", "content": "检索卡片", "metadata": {"node": "syndrome_matches"}},
            {"role": "assistant", "content": "最终回答"},
        ]
        self.assertEqual(extract_final_answer(emission), "最终回答")

    def test_bigram_f1_identical_text_is_one(self) -> None:
        self.assertEqual(bigram_f1("恶寒发热无汗", "恶寒发热无汗"), 1.0)

    def test_clarify_behavior_requires_sweat_question(self) -> None:
        record = {
            "rag_answer": "目前信息不足，请补充是否出汗，以及怕冷程度。",
            "expected_behavior": "clarify",
            "intentional_omissions": ["无汗"],
        }
        self.assertTrue(behavior_heuristic(record)["behavior_ok"])

    def test_definitive_formula_fails_clarify_behavior(self) -> None:
        record = {
            "rag_answer": "建议服用某方，目前也可以补充是否出汗。",
            "expected_behavior": "clarify",
            "intentional_omissions": ["无汗"],
        }
        self.assertFalse(behavior_heuristic(record)["behavior_ok"])


if __name__ == "__main__":
    unittest.main()
