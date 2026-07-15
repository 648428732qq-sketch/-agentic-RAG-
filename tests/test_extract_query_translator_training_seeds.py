from __future__ import annotations

from scripts.extract_query_translator_training_seeds import exact_signature, select_training_candidates


def candidate(entry_id: str, source_type: str, concept: str, terms: set[str]) -> dict:
    return {
        "entry_id": entry_id,
        "source_type": source_type,
        "concept_key": concept,
        "signature": terms,
        "direction": "digestive",
        "quality": 4,
    }


def test_training_selection_deduplicates_concepts_and_exact_signatures() -> None:
    candidates = [
        candidate("a", "formula_syndrome", "c1", {"腹痛", "泄泻"}),
        candidate("b", "formula_syndrome", "c1", {"头痛", "恶寒"}),
        candidate("c", "formula_syndrome", "c2", {"腹痛", "泄泻"}),
        candidate("d", "formula_syndrome", "c3", {"咳嗽", "气喘"}),
    ]
    selected = select_training_candidates(candidates, {"formula_syndrome": 2}, "test")
    assert len(selected) == 2
    assert len({item["concept_key"] for item in selected}) == 2
    assert len({exact_signature(item) for item in selected}) == 2
