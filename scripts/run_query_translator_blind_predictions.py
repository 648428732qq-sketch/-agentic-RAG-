from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


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
sys.path[:0] = [str(ROOT), str(PROJECT)]

from scripts.run_chatmed_gold_eval import install_local_qdrant_grpc_stub_if_blocked


DEFAULT_QUESTIONS = ROOT / "tests" / "evals" / "query_translator_mixed_300" / "questions_mixed.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "structured" / "query_translator_mixed_300_predictions.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


CLINICAL_GATE_SIGNATURES = (
    (
        {"咳嗽", "喘咳", "咳逆"},
        {"痰涎清稀"},
        {"不得平卧", "不得卧"},
    ),
)


def covers_signature(terms: set[str], signature: tuple[set[str], ...]) -> bool:
    return all(bool(group & terms) for group in signature)


def has_structured_retrieval_gate(result: dict[str, Any]) -> bool:
    decision = result.get("decision", {})
    matches = result.get("matches", [])
    if not matches:
        return False
    top = matches[0]
    if top.get("negative_conflicts"):
        return False
    payload = top.get("payload", {})
    source_type = payload.get("source_type", "")
    query = result.get("query", {})
    terms = set(query.get("canonical_terms", [])) | set(top.get("matched_terms", []))
    if source_type == "classical_acupuncture_principle":
        return bool(terms & {"得气", "气至", "迎随", "补泻"})
    if source_type == "herb_indication":
        return top.get("exact_match_count", 0) > 0 or top.get("canonical_match_count", 0) >= 1
    if source_type == "classical_theory":
        return "寸口" in terms and bool(terms & {"十二经", "经络", "经脉"})
    if decision.get("status") == "no_match":
        return False
    if query.get("query_intent") == "clinical_symptom":
        for match in matches[:5]:
            payload = match.get("payload", {})
            if payload.get("source_type") != "formula_syndrome":
                continue
            matched = set(match.get("matched_terms", []))
            if any(covers_signature(matched, signature) for signature in CLINICAL_GATE_SIGNATURES):
                return True
        if source_type != "formula_syndrome":
            return False
        reasons = set(decision.get("reasons", []))
        if reasons - {"clinical_formula_requires_clarification"}:
            return False
        required = {
            "formula_syndrome": 3,
            "classical_clause": 2,
            "classical_acupuncture": 3,
            "herb_indication": 2,
        }.get(source_type, 3)
        return (
            int(top.get("primary_canonical_match_count", top.get("canonical_match_count", 0))) >= required
            and float(top.get("query_coverage", 0.0)) >= 1.0
        )
    required = {
        "formula_syndrome": 3,
        "classical_clause": 2,
        "classical_acupuncture": 3,
        "classical_acupuncture_principle": 1,
        "herb_indication": 1,
        "classical_theory": 2,
    }.get(source_type, 4)
    return top.get("exact_match_count", 0) > 0 or top.get("canonical_match_count", 0) >= required


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读取公开问题并运行DeepSeek Query Translator预测，不读取金标准")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_local_qdrant_grpc_stub_if_blocked()
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    import config
    from core.llm_factory import create_query_translator_client
    from core.syndrome_retriever import SyndromeRetriever, should_use_structured_answer

    if config.ACTIVE_LLM_CONFIG != "deepseek":
        raise SystemExit("当前在线盲测要求ACTIVE_LLM_CONFIG=deepseek")
    questions = read_jsonl(args.questions)
    if any(set(record) != {"id", "query"} for record in questions):
        raise ValueError("公开问题文件只能包含id和query字段")
    existing = {record["id"]: record for record in read_jsonl(args.output)} if args.output.exists() else {}
    retriever = SyndromeRetriever()
    retriever.set_llm(create_query_translator_client())
    atexit.register(retriever.close)
    for position, question in enumerate(questions, start=1):
        case_id = str(question["id"])
        if case_id in existing:
            continue
        started = time.perf_counter()
        result = retriever.search(str(question["query"]), limit=5)
        query_info = result.get("query", {})
        matches = result.get("matches", [])
        record = {
            "id": case_id,
            "query": question["query"],
            "canonical_terms": query_info.get("canonical_terms", []),
            "candidate_terms": query_info.get("candidate_terms", []),
            "negative_terms": query_info.get("negative_terms", []),
            "unknown_phrases": query_info.get("unknown_phrases", []),
            "needs_more_info": bool(query_info.get("needs_more_info")),
            "translation_method": query_info.get("translation_method", ""),
            "translation_errors": query_info.get("translation_errors", []),
            "gate": should_use_structured_answer(result) or has_structured_retrieval_gate(result),
            "decision": result.get("decision", {}),
            "top_k": [
                {
                    "title": match.get("payload", {}).get("title", ""),
                    "source_type": match.get("payload", {}).get("source_type", ""),
                    "formula": match.get("payload", {}).get("formula", ""),
                    "intervention_name": match.get("payload", {}).get("intervention_name", ""),
                    "herb_name": match.get("payload", {}).get("herb_name", ""),
                    "source_book": match.get("payload", {}).get("source_book", ""),
                    "evidence": match.get("payload", {}).get("evidence", ""),
                    "canonical_match_count": match.get("canonical_match_count", 0),
                    "matched_terms": match.get("matched_terms", []),
                }
                for match in matches
            ],
            "retrieval_debug": result.get("retrieval_debug", {}),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        append_jsonl(args.output, record)
        existing[case_id] = record
        print(f"[{position}/{len(questions)}] id={case_id} decision={record['decision'].get('status')}")
    print(json.dumps({"question_count": len(questions), "prediction_count": len(existing), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
