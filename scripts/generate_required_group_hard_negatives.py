from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary_effective.jsonl"
DEFAULT_CANDIDATES = ROOT / "datasets" / "structured" / "syndrome_required_groups_candidates.jsonl"
DEFAULT_OUTPUT = ROOT / "tests" / "evals" / "required_group_hard_negative_v1" / "cases.jsonl"
DEFAULT_REPORT = ROOT / "tests" / "evals" / "required_group_hard_negative_v1" / "manifest.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def representative(group: list[str]) -> str:
    return next((str(term) for term in group if 1 < len(str(term)) <= 16), str(group[0]) if group else "")


def overlaps(term: str, group: list[str]) -> bool:
    return any(term == value or term in value or value in term for value in group if term and value)


def make_query(terms: list[str], index: int) -> str:
    joined = "、".join(terms)
    templates = (
        "患者目前有{terms}，仅按本地方证证据能否直接判断对应方证？",
        "现在能确认的表现只有{terms}，这些信息够不够确定方证？",
        "已知症状为{terms}，请根据本地结构化方证判断是否还需要追问。",
    )
    return templates[index % len(templates)].format(terms=joined)


def generate_cases(dictionary: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible_ids = {
        str(row.get("entry_id"))
        for row in candidate_rows
        if row.get("validation", {}).get("auto_apply_eligible")
    }
    entries = [
        row
        for row in dictionary
        if row.get("source_type") == "formula_syndrome"
        and row.get("entry_id") in eligible_ids
        and len(row.get("required_symptom_groups") or []) >= 2
    ]
    cases: list[dict[str, Any]] = []
    for entry in entries:
        groups = [[str(term) for term in group if str(term)] for group in entry["required_symptom_groups"]]
        full_terms = [representative(group) for group in groups]
        case_prefix = hashlib.sha1(str(entry["entry_id"]).encode("utf-8")).hexdigest()[:12]
        cases.append(
            {
                "id": f"rg_full_{case_prefix}",
                "style": "full_required_signature",
                "query": make_query(full_terms, 0),
                "expected_formula": entry.get("formula"),
                "expected_entry_id": entry.get("entry_id"),
                "expected_gate": True,
                "required_symptom_groups": groups,
                "present_terms": full_terms,
                "omitted_group_index": None,
                "omitted_required_group": [],
                "evidence": entry.get("evidence", ""),
                "source_file": entry.get("source_file", ""),
                "review_status": entry.get("review_status", ""),
            }
        )
        for omitted_index, omitted_group in enumerate(groups):
            present_terms = [
                representative(group)
                for index, group in enumerate(groups)
                if index != omitted_index
            ]
            if any(overlaps(term, omitted_group) for term in present_terms):
                raise ValueError(
                    f"omitted group leaked into query for {entry.get('formula')}: {omitted_group} vs {present_terms}"
                )
            cases.append(
                {
                    "id": f"rg_missing_{case_prefix}_{omitted_index + 1}",
                    "style": "missing_one_required_group",
                    "query": make_query(present_terms, omitted_index + 1),
                    "expected_formula": entry.get("formula"),
                    "expected_entry_id": entry.get("entry_id"),
                    "expected_gate": False,
                    "required_symptom_groups": groups,
                    "present_terms": present_terms,
                    "omitted_group_index": omitted_index,
                    "omitted_required_group": omitted_group,
                    "evidence": entry.get("evidence", ""),
                    "source_file": entry.get("source_file", ""),
                    "review_status": entry.get("review_status", ""),
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one-group-at-a-time hard negatives from validated payload groups")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    dictionary = read_jsonl(args.dictionary)
    candidates = read_jsonl(args.candidates)
    cases = generate_cases(dictionary, candidates)
    write_jsonl(args.output, cases)
    full = [case for case in cases if case["style"] == "full_required_signature"]
    missing = [case for case in cases if case["style"] == "missing_one_required_group"]
    report = {
        "case_count": len(cases),
        "formula_count": len(full),
        "full_signature_count": len(full),
        "missing_one_group_count": len(missing),
        "unique_omissions": len({(case["expected_entry_id"], case["omitted_group_index"]) for case in missing}),
        "source_policy": "explicit_diagnostic_phrase_v1",
        "dictionary": str(args.dictionary.resolve()),
        "cases": str(args.output.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
