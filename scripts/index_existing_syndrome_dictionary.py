from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path[:0] = [str(ROOT), str(PROJECT)]

from scripts.build_syndrome_dictionary import SyndromeEntry, write_qdrant  # noqa: E402


DEFAULT_INPUT = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"


def load_entries(path: Path) -> list[SyndromeEntry]:
    entries: list[SyndromeEntry] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = SyndromeEntry.model_validate_json(line)
            if entry.entry_id in seen:
                raise ValueError(f"duplicate entry_id at {path}:{line_number}: {entry.entry_id}")
            seen.add(entry.entry_id)
            entries.append(entry)
    if not entries:
        raise ValueError(f"empty syndrome dictionary: {path}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a frozen syndrome_dictionary.jsonl without re-extracting Markdown")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    entries = load_entries(args.input)
    count = write_qdrant(entries, recreate=not args.keep_existing)
    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "input_entries": len(entries),
                "indexed_points": count,
                "source_type_counts": dict(Counter(entry.source_type for entry in entries)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
