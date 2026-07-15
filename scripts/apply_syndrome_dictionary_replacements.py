from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path[:0] = [str(ROOT), str(PROJECT)]

from scripts.build_syndrome_dictionary import SyndromeEntry, make_search_text  # noqa: E402


DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_REPLACEMENTS = ROOT / "datasets" / "structured" / "syndrome_dictionary_reviewed_replacements.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "structured" / "syndrome_dictionary_effective.jsonl"
DEFAULT_REPORT = ROOT / "datasets" / "structured" / "syndrome_dictionary_effective_report.json"


def read_entries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_entries(base: list[dict[str, Any]], replacements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    merged = list(base)
    positions = {str(row.get("entry_id")): index for index, row in enumerate(merged)}
    replaced = 0
    appended = 0
    for replacement in replacements:
        entry_id = str(replacement.get("entry_id", ""))
        if not entry_id:
            raise ValueError("replacement missing entry_id")
        if entry_id in positions:
            merged[positions[entry_id]] = replacement
            replaced += 1
        else:
            positions[entry_id] = len(merged)
            merged.append(replacement)
            appended += 1
    return merged, replaced, appended


def validate_and_refresh(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        model = SyndromeEntry.model_validate(row)
        if model.entry_id in seen:
            raise ValueError(f"duplicate entry_id: {model.entry_id}")
        seen.add(model.entry_id)
        model.search_text = make_search_text(model)
        validated.append(model.model_dump(mode="json"))
    return validated


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply reviewed SyndromeEntry replacements to a frozen dictionary")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--replacements", type=Path, default=DEFAULT_REPLACEMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    base = read_entries(args.dictionary)
    replacements = read_entries(args.replacements)
    merged, replaced, appended = merge_entries(base, replacements)
    validated = validate_and_refresh(merged)
    write_jsonl(args.output, validated)

    formulas = [row for row in validated if row.get("source_type") == "formula_syndrome"]
    report = {
        "base_count": len(base),
        "replacement_count": len(replacements),
        "replaced_count": replaced,
        "appended_count": appended,
        "effective_count": len(validated),
        "source_type_counts": dict(Counter(str(row.get("source_type")) for row in validated)),
        "formula_count": len(formulas),
        "formula_empty_required_groups": sum(not row.get("required_symptom_groups") for row in formulas),
        "formula_single_required_group": sum(len(row.get("required_symptom_groups") or []) == 1 for row in formulas),
        "required_groups_rule_validated": sum(
            "required_groups_rule_validated" in str(row.get("review_status", "")) for row in formulas
        ),
        "output": str(args.output.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
