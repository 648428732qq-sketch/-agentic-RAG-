from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.syndrome_retriever import SyndromeRetriever  # noqa: E402
from core.syndrome_terms import expand_symptom_query  # noqa: E402


_DEFAULT_RETRIEVER: SyndromeRetriever | None = None


def expand_query(query: str) -> dict[str, list[str] | str]:
    return expand_symptom_query(query)


def query_syndrome_dictionary(query: str, limit: int = 5, candidate_limit: int | None = None) -> dict:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = SyndromeRetriever()
    result = _DEFAULT_RETRIEVER.search(query, limit=limit, candidate_limit=candidate_limit)
    flat_matches: list[dict] = []
    for match in result.get("matches", []):
        flat = dict(match.get("payload", {}))
        flat.update({key: value for key, value in match.items() if key != "payload"})
        flat_matches.append(flat)
    return {**result, "matches": flat_matches}


def close_default_retriever() -> None:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is not None:
        _DEFAULT_RETRIEVER.close()
        _DEFAULT_RETRIEVER = None


def main() -> None:
    parser = argparse.ArgumentParser(description="查询结构化方证字典")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=None)
    args = parser.parse_args()
    try:
        print(json.dumps(query_syndrome_dictionary(args.query, args.limit, args.candidate_limit), ensure_ascii=True, indent=2))
    finally:
        close_default_retriever()


if __name__ == "__main__":
    main()
