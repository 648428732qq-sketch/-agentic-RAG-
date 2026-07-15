from __future__ import annotations

import unittest

from scripts.run_gpu_rerank_ab import compare_runs, merge_questions_gold, ranking_metrics


class GpuRerankAbHelpersTest(unittest.TestCase):
    def test_merge_questions_gold_requires_exact_ids(self) -> None:
        questions = [{"id": "a", "query": "q", "style": "full_signature_topk"}]
        gold = [{"id": "a", "expected_formula": "f"}]
        self.assertEqual(merge_questions_gold(questions, gold)[0]["expected_formula"], "f")
        with self.assertRaises(ValueError):
            merge_questions_gold(questions, [{"id": "b", "expected_formula": "f"}])

    def test_ranking_metrics(self) -> None:
        rows = [
            {"style": "full_signature_topk", "target_rank": 1},
            {"style": "full_signature_topk", "target_rank": 2},
            {"style": "full_signature_topk", "target_rank": None},
            {"style": "shared_terms_clarify", "target_rank": 1},
        ]
        metrics = ranking_metrics(rows)
        self.assertEqual(metrics["count"], 3)
        self.assertEqual(metrics["recall_at_1"], 0.3333)
        self.assertEqual(metrics["recall_at_5"], 0.6667)
        self.assertEqual(metrics["mrr_at_8"], 0.5)

    def test_compare_runs(self) -> None:
        baseline = [
            {"id": "a", "ok": False, "top_formula": "x", "target_rank": 3},
            {"id": "b", "ok": True, "top_formula": "y", "target_rank": 1},
        ]
        reranked = [
            {"id": "a", "ok": True, "top_formula": "z", "target_rank": 1},
            {"id": "b", "ok": True, "top_formula": "y", "target_rank": 2},
        ]
        comparison = compare_runs(baseline, reranked)
        self.assertEqual(comparison["pass_gained"], 1)
        self.assertEqual(comparison["pass_lost"], 0)
        self.assertEqual(comparison["top1_changed"], 1)
        self.assertEqual(comparison["target_rank_improved"], 1)
        self.assertEqual(comparison["target_rank_worsened"], 1)


if __name__ == "__main__":
    unittest.main()
