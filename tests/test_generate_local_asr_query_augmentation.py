from __future__ import annotations

from scripts.generate_local_asr_query_augmentation import build_record


def test_build_record_uses_confusion_pair_and_preserves_mapping() -> None:
    seed = {
        "id": "seed-1",
        "expected_term_groups": [["恶寒"], ["头痛"]],
        "forbidden_terms": [],
        "expected_entry_id_in_top_k": "entry-1",
        "expected_source_type_in_top_k": "formula_syndrome",
    }
    record = build_record(seed, 0, {"恶寒": ["饿寒"], "头痛": ["头通"]}, {}, "salt")
    assert record is not None
    assert "饿寒" in record["query"]
    assert {item["canonical_term"] for item in record["evidence_mappings"]} == {"恶寒", "头痛"}
    assert record["generation_style"] == "asr_homophone_typos"
