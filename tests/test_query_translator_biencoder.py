from __future__ import annotations

import json
from pathlib import Path

from project.core.query_translator_biencoder import (
    TermCatalogItem,
    build_term_catalog,
    collapse_catalog_scores,
    expected_groups_from_frozen,
    fuse_prediction_lists,
    score_predictions,
)
from scripts.evaluate_query_translator_biencoder import load_eval_cases
from project.core.syndrome_retriever import merge_biencoder_predictions


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def test_catalog_excludes_quarantine_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    base = {
        "canonical_term": "无汗",
        "entry_id": "entry-1",
        "term_id": "term-1",
        "evidence": "恶寒发热，无汗而喘。",
        "term_roles": ["ancient_symptom"],
        "eligible_as_gold_label": True,
    }
    write_jsonl(
        path,
        [
            {**base, "evidence_tier": "high"},
            {**base, "evidence_tier": "high"},
            {**base, "canonical_term": "坏标签", "evidence_tier": "quarantine"},
        ],
    )
    catalog = build_term_catalog(path)
    assert len(catalog) == 1
    assert catalog[0].canonical_term == "无汗"
    assert "恶寒发热" in catalog[0].search_text


def test_collapse_catalog_scores_keeps_best_evidence_per_term() -> None:
    items = [
        TermCatalogItem("无汗", "无汗 a", entry_id="a"),
        TermCatalogItem("无汗", "无汗 b", entry_id="b"),
        TermCatalogItem("恶寒", "恶寒", entry_id="c"),
    ]
    predictions = collapse_catalog_scores(items, [0.2, 0.9, 0.8], 2)
    assert [item["canonical_term"] for item in predictions] == ["无汗", "恶寒"]
    assert predictions[0]["entry_id"] == "b"


def test_frozen_groups_and_metrics_accept_synonyms() -> None:
    groups = expected_groups_from_frozen({"expected_term_groups": [["腹泻", "下利"], ["腹痛"]]})
    cases = [
        {
            "expected_groups": [sorted(group) for group in groups],
            "predictions": [
                {"canonical_term": "下利", "score": 0.9},
                {"canonical_term": "腹痛", "score": 0.8},
            ],
        }
    ]
    metrics = score_predictions(cases, (1, 2))
    assert metrics["group_recall_at_1"] == 0.5
    assert metrics["group_recall_at_2"] == 1.0
    assert metrics["case_all_at_2"] == 1.0


def test_fuse_prediction_lists_uses_equal_weight_rrf() -> None:
    fused = fuse_prediction_lists(
        [
            [{"canonical_term": "恶寒"}, {"canonical_term": "无汗"}],
            [{"canonical_term": "无汗"}, {"canonical_term": "头痛"}],
        ],
        limit=3,
        rank_constant=60,
    )
    assert fused[0]["canonical_term"] == "无汗"
    assert {item["canonical_term"] for item in fused} == {"恶寒", "无汗", "头痛"}


def test_eval_loader_excludes_archived_dialect_style(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    write_jsonl(
        path,
        [
            {"id": "dialect", "query": "方言题", "generation_style": "regional_dialect", "expected_terms": ["恶寒"]},
            {"id": "typo", "query": "错字题", "generation_style": "asr_homophone_typos", "expected_terms": ["恶寒"]},
        ],
    )
    cases = load_eval_cases(path, "frozen", {"regional_dialect"})
    assert [case["id"] for case in cases] == ["typo"]


def test_online_merge_uses_catalog_terms_without_overriding_negatives() -> None:
    query_info = {
        "query_intent": "clinical_symptom",
        "canonical_terms": ["头痛"],
        "primary_canonical_terms": ["头痛"],
        "negative_terms": ["恶寒"],
        "needs_more_info": True,
        "translation_method": "local_alias",
        "expanded_query": "用户原话 脑袋疼",
    }
    result = merge_biencoder_predictions(
        query_info,
        [
            {"canonical_term": "恶寒", "score": 0.99},
            {"canonical_term": "无汗", "score": 0.81},
            {"canonical_term": "喘", "score": 0.40},
        ],
        min_score=0.55,
        max_terms=5,
        primary_terms=3,
    )

    assert result["canonical_terms"] == ["头痛", "无汗"]
    assert result["primary_canonical_terms"] == ["头痛", "无汗"]
    assert result["biencoder_predictions"][0]["canonical_term"] == "无汗"
    assert "biencoder_v3" in result["translation_method"]
