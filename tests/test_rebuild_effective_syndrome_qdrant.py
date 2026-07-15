from __future__ import annotations

import json

import pytest

from scripts.rebuild_effective_syndrome_qdrant import read_entries


def minimal_entry(entry_id: str) -> dict:
    return {
        "entry_id": entry_id,
        "title": "测试方证",
        "source_type": "formula_syndrome",
        "source_book": "测试",
        "source_file": "test.md",
        "formula": "测试方",
    }


def test_read_entries_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "effective.jsonl"
    row = minimal_entry("formula::测试方")
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate entry_id"):
        read_entries(path)


def test_read_entries_validates_effective_dictionary(tmp_path) -> None:
    path = tmp_path / "effective.jsonl"
    path.write_text(json.dumps(minimal_entry("formula::测试方"), ensure_ascii=False) + "\n", encoding="utf-8")

    entries = read_entries(path)

    assert len(entries) == 1
    assert entries[0].entry_id == "formula::测试方"
