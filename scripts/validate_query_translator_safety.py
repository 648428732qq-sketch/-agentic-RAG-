from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
from core.syndrome_retriever import SyndromeRetriever, _build_retrieval_decision  # noqa: E402
from core.symptom_query_translator import (  # noqa: E402
    QueryEvidenceMapping,
    SymptomQueryTranslation,
    _invoke_with_hard_timeout,
    translate_symptom_query,
)


class FakeStructuredRunnable:
    def __init__(self, response: SymptomQueryTranslation):
        self._response = response

    def invoke(self, messages):
        return self._response


class FakeStructuredLLM:
    def __init__(self, response: SymptomQueryTranslation):
        self._response = response

    def with_config(self, **kwargs):
        return self

    def with_structured_output(self, schema=None, **kwargs):
        return FakeStructuredRunnable(self._response)


class FailingStructuredRunnable:
    def __init__(self, owner):
        self._owner = owner

    def invoke(self, messages):
        self._owner.calls += 1
        raise RuntimeError("injected translator failure")


class FailingStructuredLLM:
    def __init__(self):
        self.calls = 0

    def with_config(self, **kwargs):
        return self

    def with_structured_output(self, schema=None, **kwargs):
        return FailingStructuredRunnable(self)


def run_validation() -> dict:
    cases: list[dict] = []

    started = time.perf_counter()
    timeout_ok = False
    try:
        _invoke_with_hard_timeout(lambda: time.sleep(5), 0.1)
    except TimeoutError:
        timeout_ok = time.perf_counter() - started < 0.5
    cases.append({"name": "hard_timeout", "ok": timeout_ok})

    grounded_response = SymptomQueryTranslation(
        query_intent="clinical_symptom",
        evidence_mappings=[
            QueryEvidenceMapping(
                source_phrase="裹着被子还是冷",
                canonical_term="恶寒",
                polarity="present",
                confidence=0.95,
            ),
            QueryEvidenceMapping(
                source_phrase="裹着被子还是冷",
                canonical_term="发热",
                polarity="present",
                confidence=0.9,
            ),
            QueryEvidenceMapping(
                source_phrase="用户没有说过的片段",
                canonical_term="恶寒",
                polarity="present",
                confidence=0.9,
            ),
        ],
        body_parts=["头部"],
        pathogenesis_hints=["外感风寒"],
        confidence=0.9,
    )
    grounded = translate_symptom_query(
        "裹着被子还是冷",
        llm=FakeStructuredLLM(grounded_response),
        force_llm=True,
        candidate_terms=["恶寒", "外感风寒"],
    )
    grounded_ok = (
        grounded["canonical_terms"] == ["恶寒"]
        and "发热" not in grounded["canonical_terms"]
        and len(grounded["evidence_mappings"]) == 1
        and not grounded["body_parts"]
        and not grounded["pathogenesis_hints"]
        and any(error.startswith("rejected_outside_local_candidates") for error in grounded["translation_errors"])
        and any(error.startswith("rejected_non_evidence") for error in grounded["translation_errors"])
    )
    cases.append({"name": "grounded_whitelist_and_evidence", "ok": grounded_ok, "result": grounded})

    offset_response = SymptomQueryTranslation(
        query_intent="clinical_symptom",
        evidence_mappings=[
            QueryEvidenceMapping(
                source_phrase="恶寒",
                source_start=0,
                source_end=7,
                canonical_term="恶寒",
                polarity="present",
                confidence=0.9,
            )
        ],
    )
    offset_grounded = translate_symptom_query(
        "裹着被子还是冷",
        llm=FakeStructuredLLM(offset_response),
        force_llm=True,
        candidate_terms=["恶寒"],
    )
    cases.append(
        {
            "name": "evidence_offsets_resolved_from_query",
            "ok": (
                offset_grounded["canonical_terms"] == ["恶寒"]
                and offset_grounded["evidence_mappings"][0]["source_phrase"] == "裹着被子还是冷"
            ),
            "result": offset_grounded,
        }
    )

    intervention_response = SymptomQueryTranslation(
        query_intent="clinical_symptom",
        evidence_mappings=[
            QueryEvidenceMapping(
                source_phrase="这个情况",
                canonical_term="麻黄汤",
                polarity="present",
                confidence=0.99,
            )
        ],
    )
    intervention = translate_symptom_query(
        "这个情况",
        llm=FakeStructuredLLM(intervention_response),
        force_llm=True,
        candidate_terms=["麻黄汤", "吐后，渴欲得水而贪饮者，文蛤汤主之"],
    )
    cases.append(
        {
            "name": "intervention_candidates_blocked",
            "ok": not intervention["canonical_terms"] and not intervention["candidate_terms"],
            "result": intervention,
        }
    )

    negated = translate_symptom_query("只有发烧，完全不怕冷，也没有咳嗽")
    negation_ok = (
        negated["canonical_terms"] == ["发热"]
        and set(negated["negative_terms"]) == {"恶寒", "咳嗽"}
    )
    cases.append({"name": "deterministic_negation", "ok": negation_ok, "result": negated})

    lexicalized_absence = translate_symptom_query("怕冷头疼不出汗还喘")
    cases.append(
        {
            "name": "lexicalized_absence_not_double_negated",
            "ok": (
                "无汗" in lexicalized_absence["canonical_terms"]
                and "汗出" not in lexicalized_absence["canonical_terms"]
                and "汗出" not in lexicalized_absence["negative_terms"]
            ),
            "result": lexicalized_absence,
        }
    )

    compound_negative_response = SymptomQueryTranslation(
        query_intent="clinical_symptom",
        evidence_mappings=[
            QueryEvidenceMapping(
                source_phrase="呼吸完全不急",
                canonical_term="喘咳",
                polarity="absent",
                confidence=0.9,
            )
        ],
    )
    narrowed_negative = translate_symptom_query(
        "呼吸完全不急",
        llm=FakeStructuredLLM(compound_negative_response),
        force_llm=True,
        candidate_terms=["喘咳", "喘"],
    )
    cases.append(
        {
            "name": "negative_compound_narrowing",
            "ok": narrowed_negative["negative_terms"] == ["喘"],
            "result": narrowed_negative,
        }
    )

    clinical_query = {
        "query_intent": "clinical_symptom",
        "canonical_terms": ["食饮不下", "胃脘当心而痛", "腹䐜胀"],
        "needs_more_info": False,
    }
    clinical_intervention_match = {
        "canonical_match_count": 3,
        "query_coverage": 1.0,
        "matched_terms": clinical_query["canonical_terms"],
        "payload": {
            "source_type": "classical_acupuncture",
            "intervention_name": "三里",
            "title": "针刺条文",
        },
    }
    clinical_intervention_decision = _build_retrieval_decision(
        clinical_query,
        [clinical_intervention_match],
    )
    cases.append(
        {
            "name": "clinical_acupuncture_always_clarifies",
            "ok": (
                clinical_intervention_decision["status"] == "clarify"
                and "clinical_intervention_requires_clarification"
                in clinical_intervention_decision["reasons"]
            ),
            "decision": clinical_intervention_decision,
        }
    )

    out_of_domain = translate_symptom_query("电脑开机以后风扇很响")
    out_of_domain_ok = (
        out_of_domain["query_intent"] == "unknown"
        and not out_of_domain["canonical_terms"]
    )
    cases.append({"name": "out_of_domain_no_hallucination", "ok": out_of_domain_ok, "result": out_of_domain})

    ambiguous = translate_symptom_query("最近整个人不舒服，说不上哪里不对")
    ambiguous_ok = ambiguous["needs_more_info"] and not ambiguous["canonical_terms"]
    cases.append({"name": "ambiguous_requires_more_info", "ok": ambiguous_ok, "result": ambiguous})

    original_threshold = config.LLM_SYMPTOM_TRANSLATOR_FAILURE_THRESHOLD
    original_cooldown = config.LLM_SYMPTOM_TRANSLATOR_COOLDOWN_SECONDS
    try:
        config.LLM_SYMPTOM_TRANSLATOR_FAILURE_THRESHOLD = 2
        config.LLM_SYMPTOM_TRANSLATOR_COOLDOWN_SECONDS = 60
        failing_llm = FailingStructuredLLM()
        retriever = SyndromeRetriever()
        retriever.set_llm(failing_llm)
        first = retriever._translate("冷得厉害一", ["恶寒"], True)
        second = retriever._translate("冷得厉害二", ["恶寒"], True)
        third = retriever._translate("冷得厉害三", ["恶寒"], True)
        circuit_ok = (
            failing_llm.calls == 2
            and any(error.startswith("llm_failure:") for error in first["translation_errors"])
            and any(error.startswith("llm_failure:") for error in second["translation_errors"])
            and "llm_circuit_open" in third["translation_errors"]
            and not third["llm_used"]
        )
        cases.append(
            {
                "name": "translator_circuit_breaker",
                "ok": circuit_ok,
                "llm_calls": failing_llm.calls,
                "third_errors": third["translation_errors"],
            }
        )
    finally:
        config.LLM_SYMPTOM_TRANSLATOR_FAILURE_THRESHOLD = original_threshold
        config.LLM_SYMPTOM_TRANSLATOR_COOLDOWN_SECONDS = original_cooldown

    return {"ok": all(case["ok"] for case in cases), "cases": cases}


def main() -> None:
    report = run_validation()
    report_path = ROOT / "datasets" / "structured" / "query_translator_safety_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
