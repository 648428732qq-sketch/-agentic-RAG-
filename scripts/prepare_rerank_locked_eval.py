from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from scripts.evaluate_formula_hard_negatives import (  # noqa: E402
    generate_cases,
    pick_best_formula_entries,
    read_jsonl,
    write_jsonl,
)


DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "evals" / "rerank_locked_v1"
PUBLIC_FIELDS = ("id", "style", "query")


def stable_key(entry: dict[str, Any] | str, seed: str) -> str:
    if isinstance(entry, str):
        identity = entry
    else:
        identity = str(entry.get("entry_id") or entry.get("formula") or entry.get("title"))
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def split_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    public = {key: case.get(key) for key in PUBLIC_FIELDS}
    gold = {key: value for key, value in case.items() if key not in {"query", "style"}}
    gold["id"] = case["id"]
    return public, gold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic locked synthetic rerank regression set.")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formula-count", type=int, default=60)
    parser.add_argument("--seed", default="rerank-locked-v1-20260702")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dictionary)
    formula_entries = pick_best_formula_entries(rows)
    all_cases = generate_cases(formula_entries, max_formulas=len(formula_entries))
    available_formulas = sorted(
        {str(case.get("expected_formula", "")) for case in all_cases if case.get("expected_formula")},
        key=lambda formula: stable_key(formula, args.seed),
    )
    selected_formulas = set(available_formulas[: max(1, args.formula_count)])
    cases = [case for case in all_cases if str(case.get("expected_formula", "")) in selected_formulas]
    cases = [case for case in cases if str(case.get("expected_formula", "")) not in str(case.get("query", ""))]
    if not cases:
        raise RuntimeError("no locked evaluation cases generated")

    questions: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for case in cases:
        public, gold = split_case(case)
        questions.append(public)
        gold_rows.append(gold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "questions.jsonl", questions)
    write_jsonl(private_dir / "gold_keys.jsonl", gold_rows)
    dictionary_sha256 = hashlib.sha256(args.dictionary.read_bytes()).hexdigest()
    style_counts: dict[str, int] = {}
    for case in cases:
        style = str(case.get("style", "unknown"))
        style_counts[style] = style_counts.get(style, 0) + 1
    manifest = {
        "schema_version": 1,
        "name": "rerank_locked_v1",
        "created": date.today().isoformat(),
        "seed": args.seed,
        "source": "synthetic hard-negative cases generated from the structured syndrome dictionary",
        "external_clinical_blindset": False,
        "dictionary": args.dictionary.resolve().relative_to(ROOT.resolve()).as_posix(),
        "dictionary_sha256": dictionary_sha256,
        "selected_formula_count": len(selected_formulas),
        "case_count": len(cases),
        "style_counts": style_counts,
        "questions": "questions.jsonl",
        "gold": "private/gold_keys.jsonl",
        "usage": "Lock after generation; do not tune prompts, gates, or rerank parameters against this set.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
