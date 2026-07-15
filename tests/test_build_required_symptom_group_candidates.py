from __future__ import annotations

from scripts.build_required_symptom_group_candidates import (
    build_candidate,
    extract_diagnostic_phrase,
    groups_from_phrase,
)


def entry(**overrides):
    value = {
        "entry_id": "formula::左金丸",
        "title": "左金丸证",
        "source_type": "formula_syndrome",
        "source_book": "方剂大全",
        "source_file": "方剂大全.md",
        "source_url": "https://example.invalid",
        "chapter": "方剂大全",
        "syndrome_name": "肝火犯胃证",
        "ancient_symptoms": ["呕吐", "胁痛", "口苦", "舌红苔黄", "脉弦数"],
        "modern_symptoms": [],
        "symptom_aliases": [],
        "diagnostic_keys": ["呕吐", "胁痛", "口苦", "舌红苔黄", "脉弦数"],
        "pathogenesis": ["肝火犯胃"],
        "required_symptom_groups": [["呕吐", "干呕", "气逆欲呕"]],
        "forbidden_terms": [],
        "differential_keys": ["呕吐", "吞酸", "胁痛", "口苦", "舌红苔黄", "脉弦数"],
        "must_clarify_fields": [],
        "intervention_type": "formula",
        "intervention_name": "",
        "treatment_method": "",
        "acupoints_or_channels": [],
        "treatment_principle": "清肝泻火，降逆止呕",
        "formula": "左金丸",
        "formula_composition": [],
        "herb_name": "",
        "herb_grade": "",
        "herb_category": "",
        "herb_aliases": [],
        "nature_flavor": [],
        "origin_habitat": "",
        "property_text": "",
        "theory_topic": "",
        "theory_question": "",
        "theory_answer": "",
        "theory_terms": [],
        "diagnostic_method": "",
        "acupuncture_principle": "",
        "acupuncture_terms": [],
        "usage_original": "",
        "functions": "",
        "indications": "肝火犯胃证。呕吐口苦，胁痛，舌红苔黄，脉弦数。",
        "formula_analysis": "",
        "modifications": "",
        "modern_applications": "",
        "contraindications": "",
        "evidence": "肝火犯胃证。呕吐口苦，胁痛，舌红苔黄，脉弦数。",
        "review_status": "rule_extracted",
        "confidence": 0.65,
        "search_text": "",
        "raw_text": "运用\n1、辨证要点：临床应用以呕吐吞酸，胁痛口苦，舌红苔黄，脉弦数为辨证要点。\n2、加减变化：略。",
        "payload_version": "syndrome_entry_v1",
    }
    value.update(overrides)
    return value


def test_extracts_last_explicit_diagnostic_phrase_with_offsets() -> None:
    raw = entry()["raw_text"]
    phrase, start, end = extract_diagnostic_phrase(raw)
    assert phrase == "呕吐吞酸，胁痛口苦，舌红苔黄，脉弦数"
    assert raw[start:end] == phrase


def test_groups_cover_main_tongue_and_pulse_axes() -> None:
    value = entry()
    phrase, _, _ = extract_diagnostic_phrase(value["raw_text"])
    groups, _ = groups_from_phrase(phrase, value)
    flat = {term for group in groups for term in group}
    assert "呕吐" in flat
    assert "胁痛" in flat
    assert "口苦" in flat
    assert "舌红苔黄" in flat
    assert "脉弦数" in flat


def test_candidate_has_evidence_offsets_and_is_eligible() -> None:
    candidate, replacement = build_candidate(entry())
    assert candidate["validation"]["auto_apply_eligible"] is True
    assert replacement is not None
    assert all(trace["source_start"] >= 0 for trace in candidate["group_traces"])
    assert "舌象" in replacement["must_clarify_fields"]
    assert "脉象" in replacement["must_clarify_fields"]


def test_ascii_source_noise_requires_manual_review() -> None:
    noisy = entry(
        evidence="局部红肿xian痛，脉数有力。",
        raw_text="运用\n1、辨证要点：临床应用以局部红肿xian痛，脉数有力为辨证要点。",
    )
    candidate, replacement = build_candidate(noisy)
    assert candidate["validation"]["auto_apply_eligible"] is False
    assert "ascii_source_noise" in candidate["validation"]["reasons"]
    assert replacement is None


def test_ascii_transliteration_is_corrected_only_with_local_evidence() -> None:
    noisy = entry(
        evidence="阳证痈疡初起。局部红肿焮痛，脉数有力。",
        differential_keys=["局部红肿焮痛", "脉数有力"],
        raw_text=(
            "主治\n阳证痈疡初起。局部红肿焮痛，脉数有力。\n"
            "运用\n1、辨证要点：临床应用以局部红肿xian痛，脉数有力为辨证要点。"
        ),
    )
    candidate, replacement = build_candidate(noisy)
    assert candidate["diagnostic_phrase"] == "局部红肿焮痛，脉数有力"
    assert candidate["evidence_backed_normalizations"]
    assert replacement is not None


def test_mouth_tongue_sore_is_not_misclassified_as_tongue_observation() -> None:
    value = entry(
        required_symptom_groups=[["口渴"]],
        diagnostic_keys=["心胸烦热", "口渴", "口舌生疮", "小便赤涩", "舌红", "脉数"],
        differential_keys=["心胸烦热", "口渴", "口舌生疮", "小便赤涩", "舌红", "脉数"],
        raw_text="运用\n1、辨证要点：临床应用以心胸烦热，口渴，口舌生疮或小便赤涩，舌红脉数为辨证要点。",
    )
    candidate, replacement = build_candidate(value)
    assert candidate["validation"]["auto_apply_eligible"] is True
    assert replacement is not None
    assert any("口舌生疮" in group for group in replacement["required_symptom_groups"])


def test_overlapping_required_terms_are_collapsed_into_one_group() -> None:
    value = entry(
        diagnostic_keys=["腹痛", "里急", "后重", "里急后重", "舌红苔黄", "脉弦数"],
        differential_keys=["腹痛", "里急", "后重", "里急后重", "舌红苔黄", "脉弦数"],
        required_symptom_groups=[],
        raw_text="运用\n1、辨证要点：临床应用以腹痛，里急后重，舌红苔黄，脉弦数为辨证要点。",
    )
    candidate, replacement = build_candidate(value)
    assert candidate["validation"]["auto_apply_eligible"] is True
    assert replacement is not None
    matching = [group for group in replacement["required_symptom_groups"] if "里急" in group or "里急后重" in group]
    assert len(matching) == 1
