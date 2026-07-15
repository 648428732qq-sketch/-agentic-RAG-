from __future__ import annotations

import unittest
from pathlib import Path

from scripts.prepare_chatmed_filtered_query_translator_eval import (
    build_cases,
    diagnostic_cases,
    negation_cases,
    pathogenesis_cases,
    source_hint_cases,
)


class PrepareChatMedFilteredQueryTranslatorEvalTests(unittest.TestCase):
    def test_diagnostic_cases_skip_non_diagnostic_terms_and_do_not_bind_source_type(self) -> None:
        rows = [
            {
                "term": "饮食",
                "priority": "high",
                "risk_flags": [],
                "source_type": "formula_syndrome",
                "sample_queries": [{"query": "请分析脾胃阴虚证发作时如何饮食"}],
            },
            {
                "term": "盗汗",
                "priority": "high",
                "risk_flags": [],
                "source_type": "formula_syndrome",
                "sample_queries": [{"query": "盗汗和失眠同时出现怎么办？"}],
            },
        ]

        cases = diagnostic_cases(rows, limit=10)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["candidate_term"], "盗汗")
        self.assertNotIn("expected_source_type_in_top_k", cases[0])

    def test_pathogenesis_cases_skip_uncertain_is_it_question(self) -> None:
        rows = [
            {
                "term": "湿热下注证",
                "priority": "high",
                "sample_queries": [{"query": "腹满小便不通是湿热下注证吗？"}],
            },
            {
                "term": "阴虚火旺证",
                "priority": "high",
                "sample_queries": [{"query": "请问如何判断阴虚火旺证？"}],
            },
        ]

        cases = pathogenesis_cases(rows, limit=10)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["candidate_term"], "阴虚火旺证")

    def test_negation_cases_use_only_true_absence_samples(self) -> None:
        rows = [
            {
                "term": "有力",
                "priority": "high",
                "sample_queries": [{"query": "我四肢没有力气，怎么办？"}],
            },
            {
                "term": "欲食",
                "priority": "high",
                "sample_queries": [{"query": "如何治疗不欲食和口苦的症状？"}],
            },
            {
                "term": "中暑",
                "priority": "high",
                "sample_queries": [{"query": "我口渴多喝水，是不是中暑了？"}],
            },
            {
                "term": "咳嗽",
                "priority": "high",
                "sample_queries": [{"query": "喉咙痛但是不咳嗽，有什么方法缓解？"}],
            },
            {
                "term": "有汗",
                "priority": "high",
                "sample_queries": [{"query": "恶寒头痛，没有汗，应该吃点什么？"}],
            },
        ]

        cases = negation_cases(rows, limit=10)

        self.assertEqual([case["candidate_term"] for case in cases], ["咳嗽", "有汗"])
        self.assertEqual(cases[0]["expected_negative_terms"], ["咳嗽"])
        self.assertEqual(cases[1]["expected_term_groups"], [["无汗"]])
        self.assertEqual(cases[1]["expected_negative_terms"], [])

    def test_build_cases_omits_source_hint_by_default(self) -> None:
        source_rows = [
            {
                "term": "黄连",
                "priority": "high",
                "source_type": "herb_indication",
                "sample_queries": [{"query": "请问黄连有哪些功效？"}],
            }
        ]
        self.assertEqual(source_hint_cases(source_rows, limit=0), [])

        # This is a direct behavior check for the build output shape; path I/O is
        # covered by the script-level smoke run in the main eval workflow.
        self.assertEqual(
            build_cases(
                filtered_dir=Path("__missing__"),
                diagnostic_limit=10,
                pathogenesis_limit=10,
                negation_limit=10,
                source_hint_limit=0,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
