from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import config
from core.hybrid_retrieval import collect_global_candidate_terms, collect_payload_evidence_terms
from core.symptom_query_translator import (
    QueryEvidenceMapping,
    SymptomQueryTranslation,
    translate_symptom_query,
    infer_query_intent,
)
from core.syndrome_retriever import (
    SyndromeRetriever,
    _build_retrieval_decision,
    _forbidden_term_conflicts,
    _is_direct_clinical_signature,
    _is_rank_clinical_signature,
    _payload_contains_evidence_term,
    diversify_matches,
    format_syndrome_clarification,
    local_rank_key,
    merge_direct_payload_terms,
    should_refuse_ungrounded_local_query,
    source_priority,
)
from core.syndrome_reranker import CrossEncoderReranker, payload_to_rerank_text


class TemporaryConfig:
    def __init__(self, **values):
        self.values = values
        self.originals = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.originals[name] = getattr(config, name)
            setattr(config, name, value)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for name, value in self.originals.items():
            setattr(config, name, value)


class FakeStructuredLLM:
    def __init__(self, response: SymptomQueryTranslation):
        self.response = response

    def with_config(self, **kwargs):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self.response


class StrictLocalRetrievalTests(unittest.TestCase):
    def test_reranker_fake_scorer_scores_payload_text(self) -> None:
        seen_pairs = []

        def scorer(pairs):
            seen_pairs.extend(pairs)
            return [0.1, 0.9]

        reranker = CrossEncoderReranker("fake-reranker", scorer=scorer)
        scores, debug = reranker.score(
            "query text",
            [
                {"payload": {"title": "first", "diagnostic_keys": ["a"]}},
                {"payload": {"title": "second", "diagnostic_keys": ["b"]}},
            ],
        )

        self.assertEqual(scores, [0.1, 0.9])
        self.assertTrue(debug["rerank_used"])
        self.assertIn("title: first", seen_pairs[0][1])
        self.assertIn("diagnostic_keys: a", payload_to_rerank_text({"diagnostic_keys": ["a"]}))

    def test_retriever_rerank_promotes_within_same_evidence_tier(self) -> None:
        retriever = SyndromeRetriever()
        retriever._reranker = CrossEncoderReranker(
            "fake-reranker",
            scorer=lambda pairs: [0.1, 0.9],
        )
        query_info = {
            "original_query": "query",
            "query_intent": "clinical_symptom",
            "canonical_terms": ["a"],
            "primary_canonical_terms": ["a"],
        }
        base = {
            "score": 0.1,
            "overlap_score": 1,
            "rrf_score": 0.1,
            "route_count": 1,
            "exact_match_count": 0,
            "matched_terms": ["a"],
            "canonical_match_count": 1,
            "primary_canonical_match_count": 1,
            "query_coverage": 1.0,
            "matched_diagnostic_terms": ["a"],
            "diagnostic_coverage": 1.0,
            "specificity_score": 1,
            "negative_conflicts": [],
        }
        matches = [
            {**base, "payload": {"source_type": "formula_syndrome", "title": "first"}},
            {**base, "payload": {"source_type": "formula_syndrome", "title": "second"}},
        ]

        with TemporaryConfig(ENABLE_SYNDROME_RERANK=True, SYNDROME_RERANK_INTENTS="clinical_symptom"):
            debug = retriever._rerank_matches(query_info, matches)

        self.assertTrue(debug["rerank_used"])
        self.assertEqual(matches[0]["payload"]["title"], "second")
        self.assertEqual(matches[0]["pre_rerank_rank"], 2)

    def test_rerank_skips_non_clinical_intent_by_default(self) -> None:
        retriever = SyndromeRetriever()
        retriever._reranker = CrossEncoderReranker(
            "fake-reranker",
            scorer=lambda pairs: [0.1, 0.9],
        )

        with TemporaryConfig(ENABLE_SYNDROME_RERANK=True, SYNDROME_RERANK_INTENTS="clinical_symptom"):
            debug = retriever._rerank_matches(
                {"query_intent": "herb_indication", "original_query": "test"},
                [
                    {"payload": {"source_type": "formula_syndrome"}, "canonical_match_count": 1},
                    {"payload": {"source_type": "herb_indication"}, "canonical_match_count": 1},
                ],
            )

        self.assertFalse(debug["rerank_used"])
        self.assertEqual(debug["reason"], "intent_not_enabled")

    def test_herb_query_prioritizes_herb_payloads(self) -> None:
        query_info = {"query_intent": "herb_indication"}

        self.assertGreater(
            source_priority({"source_type": "herb_indication"}, query_info),
            source_priority({"source_type": "formula_syndrome"}, query_info),
        )

    def test_acupuncture_principle_rank_prefers_matching_source_type(self) -> None:
        query_info = {"query_intent": "acupuncture_principle", "original_query": "针刺补泻和迎随是什么意思"}
        base = {
            "canonical_match_count": 2,
            "matched_diagnostic_terms": [],
            "specificity_score": 4,
            "exact_match_count": 1,
            "overlap_score": 2,
            "route_count": 2,
            "rrf_score": 0.01,
            "score": 0.5,
        }
        principle = {**base, "payload": {"source_type": "classical_acupuncture_principle"}}
        theory = {
            **base,
            "specificity_score": 6,
            "overlap_score": 3,
            "payload": {"source_type": "classical_theory"},
        }

        self.assertGreater(local_rank_key(query_info, principle), local_rank_key(query_info, theory))

    def test_clinical_pulse_query_does_not_promote_theory_over_formula(self) -> None:
        query_info = {
            "query_intent": "clinical_symptom",
            "original_query": "恶寒发热脉沉下利",
            "canonical_terms": ["恶寒", "发热", "脉沉", "下利", "脉"],
        }

        self.assertGreater(
            source_priority({"source_type": "formula_syndrome"}, query_info),
            source_priority({"source_type": "classical_theory"}, query_info),
        )

    def test_rank_prefers_complete_required_groups_over_extra_broad_match(self) -> None:
        query_info = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["恶寒", "头痛", "无汗", "喘"],
        }
        base = {
            "matched_diagnostic_terms": ["恶寒", "无汗", "喘"],
            "specificity_score": 12,
            "exact_match_count": 0,
            "overlap_score": 9,
            "route_count": 3,
            "rrf_score": 0.01,
            "score": 0.7,
            "payload": {"source_type": "formula_syndrome"},
        }
        complete_signature = {
            **base,
            "canonical_match_count": 6,
            "required_group_match_count": 3,
            "required_group_coverage": 1.0,
            "query_coverage": 1.0,
        }
        broad_incomplete = {
            **base,
            "canonical_match_count": 7,
            "required_group_match_count": 3,
            "required_group_coverage": 0.6,
            "query_coverage": 1.0,
            "matched_differential_terms": ["恶寒", "头痛", "无汗", "喘"],
            "differential_coverage": 1.0,
        }

        self.assertGreater(local_rank_key(query_info, complete_signature), local_rank_key(query_info, broad_incomplete))

    def test_rank_retains_high_coverage_missing_required_candidate_for_clarification(self) -> None:
        query_info = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["无汗", "喘", "脉浮紧", "发热", "脉浮", "头身疼痛", "身疼"],
            "primary_canonical_terms": ["无汗", "喘", "脉浮紧", "发热", "脉浮", "头身疼痛", "身疼"],
            "negative_terms": [],
        }
        base = {
            "matched_diagnostic_terms": [],
            "diagnostic_coverage": 0.0,
            "matched_differential_terms": [],
            "differential_coverage": 0.0,
            "specificity_score": 8,
            "exact_match_count": 0,
            "overlap_score": 4,
            "route_count": 1,
            "rrf_score": 0.01,
            "score": 0.6,
        }
        likely_target = {
            **base,
            "payload": {"source_type": "formula_syndrome", "formula": "麻黄汤"},
            "matched_terms": ["无汗", "喘", "脉浮紧", "发热", "脉浮", "头身疼痛", "身疼"],
            "canonical_match_count": 7,
            "primary_canonical_match_count": 7,
            "query_coverage": 1.0,
            "required_group_match_count": 2,
            "required_group_coverage": 0.67,
            "missing_required_symptom_groups": [["恶寒"]],
        }
        generic_complete = {
            **base,
            "payload": {"source_type": "formula_syndrome", "formula": "泛发热喘证"},
            "matched_terms": ["喘", "发热"],
            "canonical_match_count": 2,
            "primary_canonical_match_count": 2,
            "query_coverage": 0.2857,
            "required_group_match_count": 1,
            "required_group_coverage": 1.0,
            "missing_required_symptom_groups": [],
        }

        self.assertGreater(local_rank_key(query_info, likely_target), local_rank_key(query_info, generic_complete))
        decision = _build_retrieval_decision(query_info, [likely_target, generic_complete])
        self.assertEqual(decision["status"], "clarify")
        self.assertIn("missing_required_symptom_groups", decision["reasons"])

    def test_payload_overlap_search_uses_local_term_index(self) -> None:
        target = {
            "entry_id": "formula::mahuang",
            "source_type": "formula_syndrome",
            "formula": "麻黄汤",
            "diagnostic_keys": ["无汗", "喘", "脉浮紧", "发热"],
            "required_symptom_groups": [["恶寒"], ["无汗"], ["喘"]],
        }
        generic = {
            "entry_id": "formula::generic",
            "source_type": "formula_syndrome",
            "formula": "泛喘证",
            "diagnostic_keys": ["喘"],
            "required_symptom_groups": [["喘"]],
        }
        query_info = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["无汗", "喘", "脉浮紧", "发热"],
            "primary_canonical_terms": ["无汗", "喘", "脉浮紧", "发热"],
        }
        term_index = {
            "无汗": [target],
            "喘": [target, generic],
            "脉浮紧": [target],
            "发热": [target],
        }

        hits = SyndromeRetriever._payload_overlap_search([target, generic], term_index, query_info, limit=2)

        self.assertEqual(hits[0]["payload"]["formula"], "麻黄汤")
        self.assertEqual({hit["payload"]["formula"] for hit in hits}, {"麻黄汤", "泛喘证"})

    def test_diversify_matches_prefers_unique_interventions_before_duplicates(self) -> None:
        matches = [
            {"payload": {"formula": "四逆汤", "title": "四逆汤证"}},
            {"payload": {"formula": "四逆汤", "title": "四逆汤古籍条文证"}},
            {"payload": {"formula": "麻黄细辛附子汤", "title": "麻黄细辛附子汤证"}},
            {"payload": {"formula": "四逆汤", "title": "四逆汤另一条文"}},
        ]

        diversified = diversify_matches(matches, limit=3)

        self.assertEqual(
            [match["payload"]["formula"] for match in diversified],
            ["四逆汤", "麻黄细辛附子汤", "四逆汤"],
        )

    def test_diversify_prefers_structured_payload_for_same_formula_identity(self) -> None:
        matches = [
            {
                "payload": {"source_type": "classical_clause", "formula": "麻黄汤", "title": "麻黄汤古籍条文证"},
                "query_coverage": 1.0,
                "canonical_match_count": 4,
                "primary_canonical_match_count": 4,
                "required_group_coverage": 1.0,
                "required_group_match_count": 4,
                "missing_required_symptom_groups": [],
                "matched_terms": ["恶寒", "头痛", "无汗", "喘"],
            },
            {
                "payload": {"source_type": "formula_syndrome", "formula": "参苏饮"},
                "query_coverage": 1.0,
                "canonical_match_count": 4,
                "primary_canonical_match_count": 4,
                "required_group_coverage": 0.6,
                "required_group_match_count": 3,
                "missing_required_symptom_groups": [["发热"], ["咳嗽"]],
                "matched_terms": ["恶寒", "头痛", "无汗", "喘"],
            },
            {
                "payload": {"source_type": "formula_syndrome", "formula": "麻黄汤"},
                "query_coverage": 0.75,
                "canonical_match_count": 3,
                "primary_canonical_match_count": 3,
                "required_group_coverage": 1.0,
                "required_group_match_count": 4,
                "missing_required_symptom_groups": [],
                "matched_terms": ["恶寒", "无汗", "喘"],
            },
        ]

        diversified = diversify_matches(matches, limit=2, query_info={"query_intent": "clinical_symptom"})

        self.assertEqual(diversified[0]["payload"]["formula"], "麻黄汤")
        self.assertEqual(diversified[0]["payload"]["source_type"], "formula_syndrome")

    def test_diversify_retains_high_coverage_missing_required_candidate(self) -> None:
        matches = [
            {
                "payload": {"formula": "完整方一", "source_type": "formula_syndrome"},
                "query_coverage": 0.8,
                "canonical_match_count": 4,
                "primary_canonical_match_count": 4,
                "missing_required_symptom_groups": [],
            },
            {
                "payload": {"formula": "完整方二", "source_type": "formula_syndrome"},
                "query_coverage": 0.7,
                "canonical_match_count": 3,
                "primary_canonical_match_count": 3,
                "missing_required_symptom_groups": [],
            },
            {
                "payload": {"formula": "完整方三", "source_type": "formula_syndrome"},
                "query_coverage": 0.4,
                "canonical_match_count": 3,
                "primary_canonical_match_count": 3,
                "missing_required_symptom_groups": [],
            },
            {
                "payload": {"formula": "缺项但高覆盖", "source_type": "formula_syndrome"},
                "query_coverage": 1.0,
                "canonical_match_count": 5,
                "primary_canonical_match_count": 5,
                "missing_required_symptom_groups": [["口渴"]],
            },
        ]

        diversified = diversify_matches(matches, limit=3, query_info={"query_intent": "clinical_symptom"})

        self.assertEqual(diversified[0]["payload"]["formula"], "完整方一")
        self.assertIn("缺项但高覆盖", [match["payload"]["formula"] for match in diversified])
        self.assertNotIn("完整方三", [match["payload"]["formula"] for match in diversified])

    def test_diversify_retains_missing_required_candidate_at_four_of_five_terms(self) -> None:
        matches = [
            {
                "payload": {"formula": "完整方一", "source_type": "formula_syndrome"},
                "query_coverage": 1.0,
                "canonical_match_count": 5,
                "primary_canonical_match_count": 5,
                "missing_required_symptom_groups": [],
            },
            {
                "payload": {"formula": "泛化低覆盖", "source_type": "formula_syndrome"},
                "query_coverage": 0.6,
                "canonical_match_count": 3,
                "primary_canonical_match_count": 3,
                "missing_required_symptom_groups": [],
            },
            {
                "payload": {"formula": "缺项四中五", "source_type": "formula_syndrome"},
                "query_coverage": 0.8,
                "canonical_match_count": 4,
                "primary_canonical_match_count": 4,
                "missing_required_symptom_groups": [["发热", "大热", "壮热"]],
            },
        ]

        diversified = diversify_matches(matches, limit=2, query_info={"query_intent": "clinical_symptom"})

        self.assertEqual(diversified[0]["payload"]["formula"], "完整方一")
        self.assertIn("缺项四中五", [match["payload"]["formula"] for match in diversified])
        self.assertNotIn("泛化低覆盖", [match["payload"]["formula"] for match in diversified])

    def test_rank_prefers_required_coverage_over_differential_count(self) -> None:
        query_info = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["下利", "小便不利", "浮肿", "苔白", "咳喘"],
        }
        base = {
            "matched_diagnostic_terms": ["苔白"],
            "specificity_score": 12,
            "exact_match_count": 0,
            "overlap_score": 6,
            "route_count": 2,
            "rrf_score": 0.01,
            "score": 0.6,
            "payload": {"source_type": "formula_syndrome"},
        }
        complete_required = {
            **base,
            "canonical_match_count": 5,
            "required_group_match_count": 4,
            "required_group_coverage": 1.0,
            "query_coverage": 1.0,
            "matched_differential_terms": ["小便不利"],
            "differential_coverage": 0.1,
        }
        differential_heavy = {
            **base,
            "canonical_match_count": 3,
            "required_group_match_count": 1,
            "required_group_coverage": 0.25,
            "query_coverage": 0.6,
            "matched_differential_terms": ["下利", "苔白", "小便不利"],
            "differential_coverage": 0.8,
        }

        self.assertGreater(local_rank_key(query_info, complete_required), local_rank_key(query_info, differential_heavy))

    def test_cold_fluid_orthopnea_signature_is_grounded_by_payload_evidence(self) -> None:
        query_info = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["咳嗽", "痰涎清稀", "不得平卧", "喘"],
            "negative_terms": [],
        }
        cold_fluid_match = {
            "payload": {"source_type": "formula_syndrome", "formula": "小青龙汤"},
            "matched_terms": ["咳嗽", "痰涎清稀", "不得平卧", "喘"],
        }
        generic_cough_match = {
            "payload": {"source_type": "formula_syndrome", "formula": "清气化痰丸"},
            "matched_terms": ["咳嗽", "喘"],
        }

        self.assertTrue(_is_direct_clinical_signature(query_info, cold_fluid_match))
        self.assertFalse(_is_direct_clinical_signature(query_info, generic_cough_match))

    def test_cold_fluid_signature_can_rank_even_when_exterior_groups_are_missing(self) -> None:
        query_info = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["咳嗽", "痰涎清稀", "不得平卧", "喘"],
            "negative_terms": [],
        }
        cold_fluid_match = {
            "payload": {"source_type": "formula_syndrome", "formula": "小青龙汤"},
            "matched_terms": ["咳嗽", "痰涎清稀", "不得平卧", "喘"],
            "missing_required_symptom_groups": [["恶寒"], ["无汗"]],
        }
        exterior_match = {
            "payload": {"source_type": "formula_syndrome", "formula": "麻黄汤"},
            "matched_terms": ["恶寒", "头痛", "无汗", "喘"],
            "missing_required_symptom_groups": [["身疼"]],
        }
        exterior_query = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["恶寒", "头痛", "无汗", "喘"],
            "negative_terms": [],
        }

        self.assertTrue(_is_direct_clinical_signature(query_info, cold_fluid_match))
        self.assertFalse(_is_direct_clinical_signature(exterior_query, exterior_match))

    def test_post_vomit_thirst_signature_is_rank_only(self) -> None:
        query_info = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["吐后", "渴欲得水", "口渴"],
            "negative_terms": [],
        }
        classical_match = {
            "payload": {"source_type": "classical_clause", "formula": "文蛤汤"},
            "matched_terms": ["吐后", "渴欲得水", "口渴"],
        }
        generic_thirst_match = {
            "payload": {"source_type": "formula_syndrome", "formula": "白头翁汤"},
            "matched_terms": ["口渴"],
        }

        self.assertFalse(_is_direct_clinical_signature(query_info, classical_match))
        self.assertTrue(_is_rank_clinical_signature(query_info, classical_match))
        self.assertFalse(_is_direct_clinical_signature(query_info, generic_thirst_match))
        self.assertFalse(_is_rank_clinical_signature(query_info, generic_thirst_match))

    def test_rank_promotes_cold_fluid_orthopnea_signature_over_generic_cough(self) -> None:
        query_info = {
            "query_intent": "clinical_symptom",
            "canonical_terms": ["咳嗽", "痰涎清稀", "不得平卧", "喘"],
            "primary_canonical_terms": ["咳嗽", "痰涎清稀", "不得平卧", "喘"],
            "negative_terms": [],
        }
        base = {
            "matched_diagnostic_terms": [],
            "diagnostic_coverage": 0.0,
            "matched_differential_terms": [],
            "differential_coverage": 0.0,
            "specificity_score": 8,
            "exact_match_count": 0,
            "overlap_score": 4,
            "route_count": 1,
            "rrf_score": 0.01,
            "score": 0.6,
        }
        cold_fluid_match = {
            **base,
            "payload": {"source_type": "formula_syndrome", "formula": "小青龙汤"},
            "matched_terms": ["咳嗽", "痰涎清稀", "不得平卧", "喘"],
            "canonical_match_count": 4,
            "primary_canonical_match_count": 4,
            "query_coverage": 1.0,
            "required_group_match_count": 2,
            "required_group_coverage": 0.5,
        }
        generic_cough_match = {
            **base,
            "payload": {"source_type": "formula_syndrome", "formula": "清气化痰丸"},
            "matched_terms": ["咳嗽", "喘"],
            "canonical_match_count": 2,
            "primary_canonical_match_count": 2,
            "query_coverage": 0.5,
            "required_group_match_count": 2,
            "required_group_coverage": 1.0,
        }

        self.assertGreater(local_rank_key(query_info, cold_fluid_match), local_rank_key(query_info, generic_cough_match))

    def test_forbidden_terms_detect_positive_query_conflict(self) -> None:
        conflicts = _forbidden_term_conflicts(
            {"forbidden_terms": ["汗出", "有汗"]},
            ["恶寒", "汗出"],
        )

        self.assertEqual(conflicts, ["汗出"])

    def test_forbidden_terms_do_not_create_negative_conflict(self) -> None:
        payload = {
            "diagnostic_keys": ["恶寒", "无汗"],
            "forbidden_terms": ["汗出"],
            "must_clarify_fields": ["是否出汗"],
        }

        self.assertFalse(_payload_contains_evidence_term(payload, "汗出"))
        self.assertTrue(_payload_contains_evidence_term(payload, "无汗"))

    def test_lexicalized_absence_does_not_trigger_negative_conflict(self) -> None:
        payload = {
            "diagnostic_keys": ["不渴", "苔白"],
            "modern_symptoms": ["不口渴", "没有明显口渴"],
            "required_symptom_groups": [["不渴"]],
            "forbidden_terms": ["口渴"],
        }

        self.assertFalse(_payload_contains_evidence_term(payload, "口渴"))
        self.assertTrue(_payload_contains_evidence_term(payload, "苔白"))

    def test_sweat_mutual_exclusion_preserves_explicit_contradiction(self) -> None:
        plain = translate_symptom_query(
            "我一点汗都没有",
            candidate_terms=["无汗", "汗出", "恶寒"],
        )
        contradictory = translate_symptom_query(
            "我有恶寒、无汗，但是也出现汗出",
            candidate_terms=["无汗", "汗出", "恶寒"],
        )

        self.assertIn("无汗", plain["canonical_terms"])
        self.assertNotIn("汗出", plain["canonical_terms"])
        self.assertIn("无汗", contradictory["canonical_terms"])
        self.assertIn("汗出", contradictory["canonical_terms"])

    def test_positive_sweating_surface_is_not_overridden_by_absent_candidate(self) -> None:
        result = translate_symptom_query(
            "我现在多汗、恶风、小便不利、汗出、苔白、脉浮、舌淡苔白",
            candidate_terms=["无汗", "汗出", "多汗", "恶风", "小便不利", "苔白", "脉浮", "舌淡苔白"],
        )

        self.assertIn("汗出", result["canonical_terms"])
        self.assertNotIn("无汗", result["canonical_terms"])

    def test_thirst_mutual_exclusion_preserves_explicit_contradiction(self) -> None:
        plain = translate_symptom_query(
            "我不渴",
            candidate_terms=["不渴", "口渴", "口干"],
        )
        contradictory = translate_symptom_query(
            "我有口干、不渴，但是也出现口渴",
            candidate_terms=["不渴", "口渴", "口干"],
        )

        self.assertIn("不渴", plain["canonical_terms"])
        self.assertNotIn("口渴", plain["canonical_terms"])
        self.assertIn("不渴", contradictory["canonical_terms"])
        self.assertIn("口渴", contradictory["canonical_terms"])

    def test_missing_required_symptom_groups_force_clarification_reason(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "clinical_symptom",
                "canonical_terms": ["恶寒", "头痛"],
                "primary_canonical_terms": ["恶寒", "头痛"],
            },
            [
                {
                    "payload": {"source_type": "formula_syndrome", "formula": "麻黄汤"},
                    "matched_terms": ["恶寒", "头痛"],
                    "canonical_match_count": 2,
                    "primary_canonical_match_count": 2,
                    "query_coverage": 1.0,
                    "missing_required_symptom_groups": [["无汗"], ["喘"]],
                }
            ],
        )

        self.assertEqual(decision["status"], "clarify")
        self.assertIn("missing_required_symptom_groups", decision["reasons"])

    def test_direct_clinical_signature_requires_breathing_evidence(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "clinical_symptom",
                "canonical_terms": ["恶寒", "身疼", "无汗"],
                "primary_canonical_terms": ["恶寒", "身疼", "无汗"],
            },
            [
                {
                    "payload": {"source_type": "formula_syndrome", "formula": "麻黄汤"},
                    "matched_terms": ["恶寒", "身疼", "无汗"],
                    "canonical_match_count": 3,
                    "primary_canonical_match_count": 3,
                    "query_coverage": 1.0,
                    "missing_required_symptom_groups": [["喘"]],
                }
            ],
        )

        self.assertEqual(decision["status"], "clarify")
        self.assertIn("missing_required_symptom_groups", decision["reasons"])

    def test_direct_clinical_signature_can_ground_with_breathing_evidence(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "clinical_symptom",
                "canonical_terms": ["恶寒", "身疼", "无汗", "喘"],
                "primary_canonical_terms": ["恶寒", "身疼", "无汗", "喘"],
            },
            [
                {
                    "payload": {"source_type": "formula_syndrome", "formula": "麻黄汤"},
                    "matched_terms": ["恶寒", "身疼", "无汗", "喘"],
                    "canonical_match_count": 4,
                    "primary_canonical_match_count": 4,
                    "query_coverage": 1.0,
                    "missing_required_symptom_groups": [],
                }
            ],
        )

        self.assertEqual(decision["status"], "grounded_answer")

    def test_direct_clinical_signature_accepts_headache_with_breathing_evidence(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "clinical_symptom",
                "canonical_terms": ["恶寒", "头痛", "无汗", "喘"],
                "primary_canonical_terms": ["恶寒", "头痛", "无汗", "喘"],
            },
            [
                {
                    "payload": {"source_type": "formula_syndrome", "formula": "麻黄汤"},
                    "matched_terms": ["恶寒", "头痛", "无汗", "喘"],
                    "canonical_match_count": 4,
                    "primary_canonical_match_count": 4,
                    "query_coverage": 1.0,
                    "missing_required_symptom_groups": [],
                }
            ],
        )

        self.assertEqual(decision["status"], "grounded_answer")

    def test_global_candidates_only_include_terms_present_in_payloads(self) -> None:
        payloads = [
            {
                "entry_id": "local_1",
                "diagnostic_keys": ["恶寒", "发热"],
                "modern_symptoms": ["怕冷"],
                "forbidden_terms": ["阴虚"],
            }
        ]

        terms = collect_global_candidate_terms(payloads)

        self.assertIn("恶寒", terms)
        self.assertIn("发热", terms)
        self.assertIn("阴虚", terms)
        self.assertNotIn("耳聋", terms)

    def test_uncapped_payload_evidence_terms_include_uncommon_required_signs(self) -> None:
        payloads = [
            {
                "entry_id": "formula::测试方",
                "required_symptom_groups": [["血色深红甚", "紫黑稠粘"], ["脉弦数"]],
                "diagnostic_keys": ["月经过多"],
            }
        ]

        terms = collect_payload_evidence_terms(payloads)

        self.assertIn("血色深红甚", terms)
        self.assertIn("紫黑稠粘", terms)
        self.assertIn("脉弦数", terms)
        self.assertIn("月经过多", terms)

    def test_literal_payload_terms_are_separate_from_translator_expansions(self) -> None:
        query_info = {
            "canonical_terms": ["脘腹胀痛", "疼痛", "腹痛", "腹胀满"],
            "primary_canonical_terms": ["脘腹胀痛", "疼痛", "腹痛", "腹胀满"],
            "negative_terms": [],
        }

        merged = merge_direct_payload_terms(query_info, ["脘腹胀痛", "舌苔白腻"])

        self.assertEqual(merged["literal_payload_terms"], ["脘腹胀痛", "舌苔白腻"])
        self.assertIn("舌苔白腻", merged["canonical_terms"])
        self.assertIn("腹痛", merged["canonical_terms"])

    def test_literal_payload_terms_do_not_reintroduce_explicit_negatives(self) -> None:
        query_info = {
            "canonical_terms": ["恶寒"],
            "primary_canonical_terms": ["恶寒"],
            "negative_terms": ["发热"],
        }

        merged = merge_direct_payload_terms(query_info, ["恶寒", "发热"])

        self.assertEqual(merged["literal_payload_terms"], ["恶寒"])
        self.assertNotIn("发热", merged["canonical_terms"])

    def test_patient_presentation_with_local_terms_is_clinical_intent(self) -> None:
        result = translate_symptom_query(
            "患者目前有带下色黄、其气腥秽、舌苔黄腻",
            candidate_terms=["带下色黄", "其气腥秽", "舌苔黄腻"],
        )

        self.assertEqual(result["query_intent"], "clinical_symptom")

    def test_llm_can_classify_unknown_dialect_using_local_whitelist(self) -> None:
        query = "脑壳打旋旋"
        response = SymptomQueryTranslation(
            query_intent="clinical_symptom",
            evidence_mappings=[
                QueryEvidenceMapping(
                    source_phrase=query,
                    source_start=0,
                    source_end=len(query),
                    canonical_term="头晕",
                    polarity="present",
                    confidence=0.94,
                )
            ],
            confidence=0.94,
        )

        result = translate_symptom_query(
            query,
            llm=FakeStructuredLLM(response),
            force_llm=True,
            candidate_terms=["头晕"],
        )

        self.assertEqual(result["query_intent"], "clinical_symptom")
        self.assertEqual(result["canonical_terms"], ["头晕"])
        self.assertTrue(result["llm_used"])

    def test_noisy_asr_colloquial_query_maps_to_local_terms(self) -> None:
        result = translate_symptom_query(
            "没有发热，就是一吹峰难瘦而且一直冒汉",
            candidate_terms=["恶风", "汗出", "发热"],
        )

        self.assertIn("恶风", result["canonical_terms"])
        self.assertIn("汗出", result["canonical_terms"])
        self.assertIn("发热", result["negative_terms"])
        self.assertNotIn("发热", result["canonical_terms"])

    def test_noisy_nasal_wind_query_maps_to_local_terms(self) -> None:
        result = translate_symptom_query(
            "鼻子赛住了，头也涨疼，还怕空条冷风",
            candidate_terms=["鼻塞", "头痛", "恶风", "恶寒"],
        )

        self.assertIn("鼻塞", result["canonical_terms"])
        self.assertIn("头痛", result["canonical_terms"])
        self.assertIn("恶风", result["canonical_terms"])

    def test_vision_improvement_query_maps_to_mingmu(self) -> None:
        result = translate_symptom_query(
            "古书中哪些单味药记载能让眼睛看东西更清楚",
            candidate_terms=["明目", "目痛"],
        )

        self.assertEqual(result["query_intent"], "herb_indication")
        self.assertIn("明目", result["canonical_terms"])

    def test_candidate_alias_terms_are_normalized_to_canonical_terms(self) -> None:
        result = translate_symptom_query(
            "能不能让眼睛看东西清楚点，针扎后有酸麻胀的感觉？",
            candidate_terms=["明目", "看东西清楚", "得气", "酸麻胀"],
        )

        self.assertIn("明目", result["canonical_terms"])
        self.assertIn("得气", result["canonical_terms"])

    def test_noisy_asr_clinical_terms_normalize_to_local_canonicals(self) -> None:
        result = translate_symptom_query(
            "一吹分难守还一直冒汉，完全木有客扣。",
            candidate_terms=["恶风", "汗出", "咳嗽", "发热"],
        )

        self.assertIn("恶风", result["canonical_terms"])
        self.assertIn("汗出", result["canonical_terms"])
        self.assertIn("咳嗽", result["negative_terms"])

    def test_cantonese_nasal_and_orthopnea_phrases_are_normalized(self) -> None:
        nasal = translate_symptom_query(
            "个鼻封咗，个头发涨又痛，最怕阵冷风吹入来。",
            candidate_terms=["鼻塞", "头痛", "恶风"],
        )
        orthopnea = translate_symptom_query(
            "成日咳到停唔过，啲痰似清水咁，一瞓落去就喘到喊醒。",
            candidate_terms=["咳嗽", "痰涎清稀", "不得平卧", "喘"],
        )

        self.assertIn("鼻塞", nasal["canonical_terms"])
        self.assertIn("恶风", nasal["canonical_terms"])
        self.assertIn("不得平卧", orthopnea["canonical_terms"])
        self.assertIn("痰涎清稀", orthopnea["canonical_terms"])

    def test_post_vomit_thirst_sequence_maps_to_classical_terms(self) -> None:
        result = translate_symptom_query(
            "食物刚过嘴就不由自主地往上返，等全呕出来了，嗓子眼儿干得直冒烟，拼了命想灌口水。",
            candidate_terms=["食已即吐", "食入口即吐", "胃反", "吐后", "渴欲得水", "渴欲饮水", "口渴"],
        )

        self.assertIn("食已即吐", result["canonical_terms"])
        self.assertIn("吐后", result["canonical_terms"])
        self.assertIn("口渴", result["canonical_terms"])

    def test_noisy_negated_cough_and_cold_are_absent_terms(self) -> None:
        result = translate_symptom_query(
            "我而家只系发烧，完全唔怕冻，亦都冇咳。",
            candidate_terms=["发热", "恶寒", "咳嗽"],
        )

        self.assertIn("发热", result["canonical_terms"])
        self.assertIn("恶寒", result["negative_terms"])
        self.assertIn("咳嗽", result["negative_terms"])
        self.assertNotIn("咳嗽", result["canonical_terms"])

    def test_noisy_wrist_position_maps_to_cunkou_when_pulse_context_present(self) -> None:
        result = translate_symptom_query(
            "身上那些筋络条有那么多，咋个号脉就偏去按腕壳里头那点位置嘛",
            candidate_terms=["十二经", "经络", "寸口", "脉诊"],
        )

        self.assertIn("十二经", result["canonical_terms"])
        self.assertIn("寸口", result["canonical_terms"])

    def test_food_cannot_enter_and_no_seen_sweat_are_canonicalized(self) -> None:
        food = translate_symptom_query(
            "饭菜送不进嘴，这肚子胀得跟皮球似的绷着，心口窝这儿还阵阵揪着疼。",
            candidate_terms=["食饮不下", "腹䐜胀", "腹胀满", "胃脘当心而痛", "疼痛", "腹痛"],
        )
        sweat = translate_symptom_query(
            "身上酸痛得厉害，在家待了大半天也没见出汗",
            candidate_terms=["身疼", "肢体酸楚疼痛", "无汗", "汗出"],
        )

        self.assertIn("食饮不下", food["canonical_terms"])
        self.assertIn("无汗", sweat["canonical_terms"])
        self.assertNotIn("汗出", sweat["negative_terms"])

    def test_rhetorical_negative_does_not_become_positive(self) -> None:
        result = translate_symptom_query(
            "我就觉得是发烧，可怎么会怕冷呢？绝对不会有咳嗽。",
            candidate_terms=["发热", "恶寒", "咳嗽"],
        )

        self.assertIn("发热", result["canonical_terms"])
        self.assertIn("恶寒", result["negative_terms"])
        self.assertIn("咳嗽", result["negative_terms"])
        self.assertNotIn("恶寒", result["canonical_terms"])

    def test_rhetorical_question_does_not_negate_present_fever(self) -> None:
        result = translate_symptom_query(
            "难道就单是发烧吗？我根本不怕冷，也根本没咳嗽。",
            candidate_terms=["发热", "恶寒", "咳嗽"],
        )

        self.assertIn("发热", result["canonical_terms"])
        self.assertIn("恶寒", result["negative_terms"])
        self.assertIn("咳嗽", result["negative_terms"])
        self.assertNotIn("发热", result["negative_terms"])

    def test_continuous_sweating_is_not_negated_by_buting(self) -> None:
        result = translate_symptom_query(
            "哪有什么发热，就是一吹风难受，而且汗一直没完没了地一直往外冒。",
            candidate_terms=["发热", "恶风", "汗出"],
        )

        self.assertIn("发热", result["negative_terms"])
        self.assertIn("恶风", result["canonical_terms"])
        self.assertIn("汗出", result["canonical_terms"])
        self.assertNotIn("汗出", result["negative_terms"])

    def test_buting_continuous_sweating_is_positive(self) -> None:
        result = translate_symptom_query(
            "我不发烧但一吹风就觉得难受还一直不停冒汗",
            candidate_terms=["发热", "恶风", "汗出"],
        )

        self.assertIn("发热", result["negative_terms"])
        self.assertIn("汗出", result["canonical_terms"])
        self.assertNotIn("汗出", result["negative_terms"])

    def test_can_or_not_question_does_not_negate_vision_improvement(self) -> None:
        result = translate_symptom_query(
            "这双眼看啥都显影不清，能不能给擦亮透亮了？",
            candidate_terms=["明目"],
        )

        self.assertIn("明目", result["canonical_terms"])
        self.assertNotIn("明目", result["negative_terms"])

    def test_swallowing_down_is_lexicalized_present_symptom(self) -> None:
        result = translate_symptom_query(
            "饭到嘴边怎么都咽不下去",
            candidate_terms=["食饮不下"],
        )

        self.assertIn("食饮不下", result["canonical_terms"])
        self.assertNotIn("食饮不下", result["negative_terms"])

    def test_vague_wrist_reference_is_not_enough_for_cunkou(self) -> None:
        result = translate_symptom_query(
            "全身经脉那么多，为什么偏偏看手腕那一处",
            candidate_terms=["十二经", "寸口", "脉诊"],
        )

        self.assertIn("十二经", result["canonical_terms"])
        self.assertNotIn("寸口", result["canonical_terms"])

    def test_pulse_wrist_reference_maps_to_cunkou_when_contextualized(self) -> None:
        result = translate_symptom_query(
            "全身经脉那么多，为什么号脉偏偏看手腕那一处",
            candidate_terms=["十二经", "寸口", "脉诊"],
        )

        self.assertIn("十二经", result["canonical_terms"])
        self.assertIn("寸口", result["canonical_terms"])

    def test_llm_vague_cunkou_evidence_is_rejected(self) -> None:
        query = "全身经脉那么多，为什么偏偏看手腕那一处"
        response = SymptomQueryTranslation(
            query_intent="classical_theory",
            evidence_mappings=[
                QueryEvidenceMapping(
                    source_phrase="手腕那一处",
                    source_start=query.index("手腕那一处"),
                    source_end=query.index("手腕那一处") + len("手腕那一处"),
                    canonical_term="寸口",
                    polarity="present",
                    confidence=1.0,
                )
            ],
            confidence=1.0,
        )

        result = translate_symptom_query(
            query,
            llm=FakeStructuredLLM(response),
            force_llm=True,
            candidate_terms=["十二经", "寸口"],
        )

        self.assertNotIn("寸口", result["canonical_terms"])
        self.assertTrue(any(error.startswith("rejected_vague_cunkou_evidence") for error in result["translation_errors"]))

    def test_regex_rules_extract_compound_stomach_complaint(self) -> None:
        result = translate_symptom_query(
            "饭根本吃不下去胃里面连着又胀又疼",
            candidate_terms=["食饮不下", "腹䐜胀", "腹胀满", "胃脘当心而痛"],
        )

        self.assertIn("食饮不下", result["canonical_terms"])
        self.assertTrue({"腹䐜胀", "腹胀满"} & set(result["canonical_terms"]))
        self.assertIn("胃脘当心而痛", result["canonical_terms"])

    def test_regex_rules_extract_umbilical_palpitation(self) -> None:
        result = translate_symptom_query(
            "肚脐眼下头老是扑通扑通地跳，跟敲小鼓似的。",
            candidate_terms=["脐下悸", "脐下有悸"],
        )

        self.assertTrue({"脐下悸", "脐下有悸"} & set(result["canonical_terms"]))

    def test_noisy_umbilical_palpitation_maps_to_palpitation(self) -> None:
        result = translate_symptom_query(
            "肚起下面还一挑一挑的",
            candidate_terms=["脐下悸", "脐下有悸"],
        )

        self.assertTrue({"脐下悸", "脐下有悸"} & set(result["canonical_terms"]))

    def test_running_piglet_colloquial_query_maps_to_upward_qi(self) -> None:
        result = translate_symptom_query(
            "像有股气从小肚子直往胸口窜，肚脐下面还一跳一跳",
            candidate_terms=["气上冲", "欲作奔豚", "脐下悸", "脐下有悸"],
        )

        self.assertTrue({"脐下悸", "脐下有悸"} & set(result["canonical_terms"]))
        self.assertTrue({"气上冲", "欲作奔豚"} & set(result["canonical_terms"]))

    def test_noisy_wrist_theory_can_map_to_cunkou_when_explicit(self) -> None:
        result = translate_symptom_query(
            "全身经迈那么多，为神马号脉偏偏看手碗那处",
            candidate_terms=["十二经", "寸口", "脉诊"],
        )

        self.assertIn("十二经", result["canonical_terms"])
        self.assertIn("寸口", result["canonical_terms"])

    def test_reversed_wind_dialect_maps_to_wind_aversion(self) -> None:
        result = translate_symptom_query(
            "没得烧起哈，就是遭不住打风吹，还一直淌汗。",
            candidate_terms=["恶风", "汗出", "发热"],
        )

        self.assertIn("恶风", result["canonical_terms"])
        self.assertIn("汗出", result["canonical_terms"])
        self.assertIn("发热", result["negative_terms"])

    def test_bentun_metaphor_with_action_before_chest_maps_to_upward_qi(self) -> None:
        result = translate_symptom_query(
            "倒像是憋着的那股气非要从小腹一路窜进心窝子里，肚脐正下方还跟着扑通扑通地乱跳",
            candidate_terms=["欲作奔豚", "气上冲", "脐下悸", "脐下有悸"],
        )

        self.assertTrue({"欲作奔豚", "气上冲"} & set(result["canonical_terms"]))
        self.assertTrue({"脐下悸", "脐下有悸"} & set(result["canonical_terms"]))

    def test_cold_pain_no_sweat_metaphor_maps_to_canonical_terms(self) -> None:
        result = translate_symptom_query(
            "冻得直打颤，厚棉被压身还是透骨地冷；周身酸痛得像灌了铅，连个汗星子都挤不出来。",
            candidate_terms=["恶寒", "身疼", "肢体酸楚疼痛", "头身疼痛", "无汗", "汗出"],
        )

        self.assertIn("恶寒", result["canonical_terms"])
        self.assertIn("身疼", result["canonical_terms"])
        self.assertIn("无汗", result["canonical_terms"])
        self.assertNotIn("汗出", result["canonical_terms"])

    def test_unable_to_force_out_sweat_overrides_sweat_surface_word(self) -> None:
        result = translate_symptom_query(
            "裹着大厚被子还是冷，捂得再严实也憋不出一滴汗珠子。",
            candidate_terms=["恶寒", "无汗", "汗出"],
        )

        self.assertIn("无汗", result["canonical_terms"])
        self.assertNotIn("汗出", result["canonical_terms"])

    def test_vision_fog_metaphor_maps_to_mingmu(self) -> None:
        result = translate_symptom_query(
            "老书里头可曾记着，能不能把眼皮底下那层雾水散尽，让视线落下去个个分明？",
            candidate_terms=["明目"],
        )

        self.assertIn("明目", result["canonical_terms"])
        self.assertNotIn("明目", result["negative_terms"])

    def test_continuous_seeping_sweat_is_positive_not_absent(self) -> None:
        result = translate_symptom_query(
            "这风稍微一扑腾就让人挺不得劲，身子自己还没完没了地往外渗汗，脑袋瓜子上像勒了根皮筋似的直发紧。",
            candidate_terms=["恶风", "汗出", "头痛", "无汗"],
        )

        self.assertIn("恶风", result["canonical_terms"])
        self.assertIn("汗出", result["canonical_terms"])
        self.assertIn("头痛", result["canonical_terms"])
        self.assertNotIn("无汗", result["canonical_terms"])

    def test_pulse_wrist_half_inch_reference_maps_to_cunkou(self) -> None:
        result = translate_symptom_query(
            "满身子经脉横七竖八的，把脉干嘛就专找胳膊肘到手腕那寸半截地儿较劲？",
            candidate_terms=["寸口", "十二经", "经络", "脉诊"],
        )

        self.assertIn("十二经", result["canonical_terms"])
        self.assertIn("寸口", result["canonical_terms"])

    def test_which_herb_question_does_not_negate_mingmu(self) -> None:
        result = translate_symptom_query(
            "古书里有哪些法子能明木呀，让眼紧看东西清出点",
            candidate_terms=["明目", "看东西清楚"],
        )

        self.assertIn("明目", result["canonical_terms"])
        self.assertNotIn("明目", result["negative_terms"])

    def test_even_cough_absence_is_negative_not_positive(self) -> None:
        result = translate_symptom_query(
            "难道就只是发烧吗？我确实一点都不怕冷，更是连咳嗽都没有。",
            candidate_terms=["发热", "恶寒", "咳嗽"],
        )

        self.assertIn("发热", result["canonical_terms"])
        self.assertIn("恶寒", result["negative_terms"])
        self.assertIn("咳嗽", result["negative_terms"])
        self.assertNotIn("咳嗽", result["canonical_terms"])

    def test_needle_like_headache_is_clinical_not_acupuncture_intent(self) -> None:
        result = translate_symptom_query(
            "鼻子憋得死紧连不上风，脑袋瓜子胀着跟针扎似的揪痛，冷风一吹就得躲着。",
            candidate_terms=["鼻塞", "头痛", "恶风", "针刺", "得气"],
        )

        self.assertEqual(result["query_intent"], "clinical_symptom")
        self.assertIn("鼻塞", result["canonical_terms"])
        self.assertIn("头痛", result["canonical_terms"])
        self.assertIn("恶风", result["canonical_terms"])

    def test_literal_needle_action_remains_acupuncture_intent(self) -> None:
        result = translate_symptom_query(
            "针扎后有酸麻胀的感觉，这算不算得气？",
            candidate_terms=["针刺", "得气", "酸麻胀"],
        )

        self.assertEqual(result["query_intent"], "acupuncture_principle")

    def test_breathing_negation_is_not_positive_asthma(self) -> None:
        result = translate_symptom_query(
            "汗一点都没有，但是呼吸完全不急，也不憋。",
            candidate_terms=["无汗", "喘"],
        )

        self.assertIn("无汗", result["canonical_terms"])
        self.assertIn("喘", result["negative_terms"])
        self.assertNotIn("喘", result["canonical_terms"])

    def test_exact_local_candidate_in_query_is_accepted(self) -> None:
        result = translate_symptom_query(
            "盗汗和失眠同时出现怎么办？请帮我推荐中药或者方剂",
            candidate_terms=["盗汗", "失眠"],
        )

        self.assertEqual(result["query_intent"], "clinical_symptom")
        self.assertIn("盗汗", result["canonical_terms"])
        self.assertIn("失眠", result["canonical_terms"])

    def test_pathogenesis_candidate_in_query_is_accepted(self) -> None:
        result = translate_symptom_query(
            "请问如何判断阴虚火旺证？",
            candidate_terms=["阴虚火旺证"],
        )

        self.assertEqual(result["query_intent"], "clinical_symptom")
        self.assertIn("阴虚火旺证", result["canonical_terms"])

    def test_plain_no_sweat_phrase_is_present_wuhan(self) -> None:
        result = translate_symptom_query(
            "最近身体很虚弱，但热不寒，没有汗。",
            candidate_terms=["无汗", "有汗"],
        )

        self.assertIn("无汗", result["canonical_terms"])
        self.assertNotIn("有汗", result["negative_terms"])

    def test_primary_terms_exclude_direct_colloquial_aliases(self) -> None:
        result = translate_symptom_query(
            "怕冷头疼不出汗还喘",
            candidate_terms=["恶寒", "头痛", "无汗", "喘", "怕冷", "头疼", "不出汗"],
        )

        self.assertNotIn("怕冷", result["primary_canonical_terms"])
        self.assertNotIn("头疼", result["primary_canonical_terms"])
        self.assertNotIn("不出汗", result["primary_canonical_terms"])
        self.assertEqual(result["primary_canonical_terms"], ["恶寒", "头痛", "无汗", "喘"])

    def test_negated_sweating_surface_maps_to_present_wuhan(self) -> None:
        result = translate_symptom_query(
            "我最近恶寒头痛，没有出汗。",
            candidate_terms=["无汗", "出汗", "汗出", "恶寒", "头痛"],
        )

        self.assertIn("无汗", result["canonical_terms"])
        self.assertNotIn("出汗", result["canonical_terms"])
        self.assertNotIn("汗出", result["canonical_terms"])
        self.assertNotIn("出汗", result["negative_terms"])
        self.assertNotIn("汗出", result["negative_terms"])

    def test_negated_hanchu_surface_maps_to_present_wuhan(self) -> None:
        result = translate_symptom_query(
            "我最近感觉痫，没汗出。",
            candidate_terms=["无汗", "汗出"],
        )

        self.assertIn("无汗", result["canonical_terms"])
        self.assertNotIn("汗出", result["canonical_terms"])
        self.assertNotIn("汗出", result["negative_terms"])

    def test_negated_exact_candidate_is_not_positive(self) -> None:
        result = translate_symptom_query(
            "口臭没有牙痛怎么办？",
            candidate_terms=["口臭", "牙痛"],
        )

        self.assertIn("口臭", result["canonical_terms"])
        self.assertIn("牙痛", result["negative_terms"])
        self.assertNotIn("牙痛", result["canonical_terms"])

    def test_lexicalized_negative_candidate_remains_present(self) -> None:
        result = translate_symptom_query(
            "如何治疗不欲食和口苦的症状？",
            candidate_terms=["不欲食", "欲食", "口苦"],
        )

        self.assertIn("不欲食", result["canonical_terms"])
        self.assertIn("口苦", result["canonical_terms"])
        self.assertNotIn("欲食", result["negative_terms"])

    def test_plain_negative_cough_is_absent(self) -> None:
        result = translate_symptom_query(
            "我喉咙疼痛，但是不咳嗽。",
            candidate_terms=["咳嗽", "咽喉肿痛"],
        )

        self.assertIn("咳嗽", result["negative_terms"])
        self.assertNotIn("咳嗽", result["canonical_terms"])

    def test_not_thirsty_keeps_buke_and_filters_thirst(self) -> None:
        result = translate_symptom_query(
            "咽喉肿痛，但是并不口渴。",
            candidate_terms=["咽喉肿痛", "口渴", "不渴"],
        )

        self.assertIn("不渴", result["canonical_terms"])
        self.assertIn("口渴", result["negative_terms"])
        self.assertNotIn("口渴", result["canonical_terms"])

    def test_urination_blocked_is_present_compound_not_negation(self) -> None:
        result = translate_symptom_query(
            "腹满小便不通是湿热下注证吗？",
            candidate_terms=["腹满", "小便不通", "湿热下注证"],
        )

        self.assertIn("小便不通", result["canonical_terms"])
        self.assertNotIn("湿热下注证", result["negative_terms"])

    def test_common_local_terms_are_available_without_candidate_context(self) -> None:
        result = translate_symptom_query("我经常感觉烦躁不安，伴有潮热，怎么办呢？")

        self.assertIn("烦躁", result["canonical_terms"])
        self.assertIn("潮热", result["canonical_terms"])

    def test_low_confidence_mapping_is_not_allowed_into_retrieval(self) -> None:
        query = "脑壳打旋旋"
        response = SymptomQueryTranslation(
            query_intent="clinical_symptom",
            evidence_mappings=[
                QueryEvidenceMapping(
                    source_phrase=query,
                    source_start=0,
                    source_end=len(query),
                    canonical_term="头晕",
                    polarity="present",
                    confidence=0.4,
                )
            ],
            confidence=0.4,
        )

        result = translate_symptom_query(
            query,
            llm=FakeStructuredLLM(response),
            force_llm=True,
            candidate_terms=["头晕"],
        )

        self.assertEqual(result["canonical_terms"], [])
        self.assertTrue(result["needs_more_info"])
        self.assertTrue(any(error.startswith("rejected_low_confidence") for error in result["translation_errors"]))

    def test_lexicalized_absence_is_a_present_symptom(self) -> None:
        query = "汗一点都没有"
        response = SymptomQueryTranslation(
            query_intent="clinical_symptom",
            evidence_mappings=[
                QueryEvidenceMapping(
                    source_phrase=query,
                    source_start=0,
                    source_end=len(query),
                    canonical_term="无汗",
                    polarity="absent",
                    confidence=1.0,
                )
            ],
            confidence=1.0,
        )

        result = translate_symptom_query(
            query,
            llm=FakeStructuredLLM(response),
            force_llm=True,
            candidate_terms=["无汗"],
        )

        self.assertIn("无汗", result["canonical_terms"])
        self.assertNotIn("无汗", result["negative_terms"])
        self.assertEqual(result["evidence_mappings"][0]["polarity"], "present")

    def test_same_source_phrase_counts_once_for_primary_coverage(self) -> None:
        query = "身上酸疼"
        response = SymptomQueryTranslation(
            query_intent="clinical_symptom",
            evidence_mappings=[
                QueryEvidenceMapping(
                    source_phrase=query,
                    source_start=0,
                    source_end=len(query),
                    canonical_term="身疼",
                    polarity="present",
                    confidence=1.0,
                ),
                QueryEvidenceMapping(
                    source_phrase=query,
                    source_start=0,
                    source_end=len(query),
                    canonical_term="肢体酸楚疼痛",
                    polarity="present",
                    confidence=1.0,
                ),
            ],
            confidence=1.0,
        )

        result = translate_symptom_query(
            query,
            llm=FakeStructuredLLM(response),
            force_llm=True,
            candidate_terms=["身疼", "肢体酸楚疼痛"],
        )

        self.assertIn("身疼", result["canonical_terms"])
        self.assertIn("肢体酸楚疼痛", result["canonical_terms"])
        self.assertEqual(result["primary_canonical_terms"], ["身疼"])

    def test_unknown_local_no_match_is_refused(self) -> None:
        result = {
            "query": {"query_intent": "unknown", "canonical_terms": []},
            "matches": [],
            "decision": {"status": "no_match", "reasons": ["unknown_intent"]},
        }

        self.assertTrue(should_refuse_ungrounded_local_query(result))

    def test_clinical_query_without_local_terms_is_no_match(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "clinical_symptom",
                "canonical_terms": [],
                "needs_more_info": True,
            },
            [],
        )

        self.assertEqual(decision["status"], "no_match")

    def test_strong_formula_match_is_not_forced_to_clarify_by_competition(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "clinical_symptom",
                "canonical_terms": ["恶寒", "身疼", "无汗", "喘"],
                "primary_canonical_terms": ["恶寒", "身疼", "无汗", "喘"],
            },
            [
                {
                    "payload": {"source_type": "formula_syndrome", "formula": "麻黄汤"},
                    "matched_terms": ["恶寒", "身疼", "无汗", "喘"],
                    "canonical_match_count": 4,
                    "primary_canonical_match_count": 4,
                    "query_coverage": 1.0,
                },
                {
                    "payload": {"source_type": "formula_syndrome", "formula": "竞争方"},
                    "matched_terms": ["恶寒", "身疼", "无汗", "喘"],
                    "canonical_match_count": 4,
                    "primary_canonical_match_count": 4,
                    "query_coverage": 1.0,
                },
            ],
        )

        self.assertEqual(decision["status"], "clarify")
        self.assertIn("competing_indistinguishable_matches", decision["reasons"])

    def test_non_signature_formula_match_requires_clarification(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "clinical_symptom",
                "canonical_terms": ["恶风", "汗出", "头痛"],
                "primary_canonical_terms": ["恶风", "汗出", "头痛"],
            },
            [
                {
                    "payload": {"source_type": "formula_syndrome", "formula": "桂枝汤", "confidence": 0.52},
                    "matched_terms": ["恶风", "汗出", "头痛"],
                    "canonical_match_count": 3,
                    "primary_canonical_match_count": 3,
                    "query_coverage": 1.0,
                }
            ],
        )

        self.assertEqual(decision["status"], "clarify")
        self.assertIn("payload_evidence_confidence_below_threshold", decision["reasons"])

    def test_theory_relation_with_one_side_only_requires_clarification(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "classical_theory",
                "canonical_terms": ["寸口"],
            },
            [
                {
                    "payload": {"source_type": "classical_theory", "required_symptom_groups": [["寸口"], ["十二经"]]},
                    "canonical_match_count": 1,
                    "exact_match_count": 0,
                    "missing_required_symptom_groups": [["十二经"]],
                }
            ],
        )

        self.assertEqual(decision["status"], "clarify")

    def test_pulse_theory_with_only_generic_pulse_requires_clarification(self) -> None:
        decision = _build_retrieval_decision(
            {
                "query_intent": "classical_theory",
                "canonical_terms": ["脉", "脉诊"],
            },
            [
                {
                    "payload": {
                        "source_type": "classical_theory",
                        "title": "脉诊/诊法理论",
                        "required_symptom_groups": [["具体脉位"], ["具体理论关系"]],
                    },
                    "matched_terms": ["脉", "脉诊"],
                    "canonical_match_count": 2,
                    "primary_canonical_match_count": 2,
                    "query_coverage": 1.0,
                    "missing_required_symptom_groups": [["具体脉位"], ["具体理论关系"]],
                }
            ],
        )

        self.assertEqual(decision["status"], "clarify")

    def test_personal_treatment_request_is_not_formula_knowledge(self) -> None:
        self.assertEqual(
            infer_query_intent("我得了克罗恩病，本地古籍里有什么方子"),
            "clinical_symptom",
        )
        self.assertEqual(
            infer_query_intent("麻黄汤的组成和用法是什么"),
            "formula_knowledge",
        )

    def test_clarification_does_not_expose_candidate_formula(self) -> None:
        result = {
            "query": {"query_intent": "clinical_symptom", "canonical_terms": ["头晕"]},
            "matches": [
                {
                    "payload": {"formula": "测试方", "title": "测试方证"},
                    "missing_diagnostic_terms": ["恶寒", "无汗"],
                }
            ],
            "decision": {"status": "clarify"},
        }

        answer = format_syndrome_clarification(result)

        self.assertNotIn("测试方", answer)
        self.assertIn("恶寒", answer)


if __name__ == "__main__":
    unittest.main()
