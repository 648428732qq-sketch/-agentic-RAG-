from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract compact unsafe-answer diagnostics from grounded RAG predictions")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    selected = []
    for row in rows:
        if row.get("expected_gate") or not row.get("actual_gate"):
            continue
        selected.append(
            {
                "id": row.get("id"),
                "query": row.get("query"),
                "style": row.get("style"),
                "canonical_terms": row.get("canonical_terms", []),
                "decision": row.get("decision", {}),
                "matches": row.get("matches", [])[:3],
            }
        )
    print(json.dumps({"unsafe_count": len(selected), "cases": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
