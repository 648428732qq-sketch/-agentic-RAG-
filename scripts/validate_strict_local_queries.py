from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


for thread_env in (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
for path in (ROOT, PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_chatmed_gold_eval import install_local_qdrant_grpc_stub_if_blocked


DEFAULT_QUERIES = [
    "我脑壳昏戳戳的，胸口还扯起扯起的",
    "冷得我摆起，两床被子都捂不热，也没出汗",
    "我得了克罗恩病，本地古籍里有什么方子",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate strict local-only retrieval decisions.")
    parser.add_argument("--query", action="append", dest="queries", help="Query to validate; repeatable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_local_qdrant_grpc_stub_if_blocked()

    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")

    import config
    from core.llm_factory import create_query_translator_client
    from core.syndrome_retriever import SyndromeRetriever

    retriever = SyndromeRetriever()
    retriever.set_llm(create_query_translator_client())
    summaries = []
    for query in args.queries or DEFAULT_QUERIES:
        result = retriever.search(query, limit=3)
        summaries.append(
            {
                "query": query,
                "intent": result.get("query", {}).get("query_intent"),
                "translation_method": result.get("query", {}).get("translation_method"),
                "canonical_terms": result.get("query", {}).get("canonical_terms", []),
                "unknown_phrases": result.get("query", {}).get("unknown_phrases", []),
                "decision": result.get("decision", {}),
                "match_count": len(result.get("matches", [])),
                "top_entry_id": (
                    result.get("matches", [{}])[0].get("payload", {}).get("entry_id", "")
                    if result.get("matches")
                    else ""
                ),
                "llm_requested": result.get("retrieval_debug", {}).get("llm_requested", False),
                "llm_used": result.get("retrieval_debug", {}).get("llm_used", False),
                "local_candidate_terms": result.get("retrieval_debug", {}).get("candidate_term_count", 0),
                "retrieval_errors": result.get("retrieval_debug", {}).get("errors", []),
            }
        )
    print(json.dumps({"collection": config.SYNDROME_COLLECTION, "results": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
