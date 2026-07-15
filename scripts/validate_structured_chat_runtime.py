from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from core.chat_interface import ChatInterface  # noqa: E402
from core.syndrome_retriever import SyndromeRetriever  # noqa: E402


RUNTIME_CASES = [
    {
        "query": "怕冷头疼不出汗还喘",
        "expected_source_type": "formula_syndrome",
        "expected_text": "目前证据不足",
    },
    {
        "query": "眼睛痛看不清",
        "expected_source_type": "herb_indication",
        "expected_text": "目前证据不足",
    },
    {
        "query": "咳嗽喘不上气",
        "expected_text": "目前证据不足",
        "expected_has_matches": False,
    },
    {
        "query": "十二经为什么独取寸口",
        "expected_source_type": "classical_theory",
        "expected_text": "从结构化古籍理论库看",
    },
    {
        "query": "针刺补泻和迎随是什么意思",
        "expected_source_type": "classical_acupuncture_principle",
        "expected_text": "从结构化针法原则库看",
    },
]


class FailingGraph:
    def __init__(self) -> None:
        self.stream_calls = 0

    def get_state(self, config: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(next=())

    def stream(self, *args, **kwargs):
        self.stream_calls += 1
        raise AssertionError("ChatInterface fell through to old LangGraph/RAG stream")

    def update_state(self, *args, **kwargs) -> None:
        raise AssertionError("ChatInterface unexpectedly updated old graph state")


class RuntimeRagStub:
    def __init__(self) -> None:
        self.agent_graph = FailingGraph()
        self.retriever = SyndromeRetriever()

    def get_config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": "structured-runtime-validation"}}

    def search_syndromes(self, query: str, limit: int = 3) -> dict[str, Any]:
        return self.retriever.search(query, limit=limit)


class FailingRetrievalRagStub(RuntimeRagStub):
    def search_syndromes(self, query: str, limit: int = 3) -> dict[str, Any]:
        raise RuntimeError("injected retrieval failure")


def flatten_messages(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in messages)
    return str(messages)


def run_case(chat: ChatInterface, graph: FailingGraph, case: dict[str, Any]) -> dict[str, Any]:
    graph.stream_calls = 0
    emissions = list(chat.chat(case["query"], []))
    final = emissions[-1] if emissions else []
    text = flatten_messages(final)
    metadata_nodes = [
        message.get("metadata", {}).get("node")
        for message in final
        if isinstance(message, dict) and message.get("metadata")
    ] if isinstance(final, list) else []
    expected_source_type = case.get("expected_source_type", "")
    source_type_seen = not expected_source_type or expected_source_type in text
    expected_has_matches = case.get("expected_has_matches", True)
    match_metadata_ok = ("syndrome_matches" in metadata_nodes) is expected_has_matches
    old_tool_seen = "search_child_chunks" in text or "retrieve_parent_chunks" in text or "NO_RELEVANT_CHUNKS" in text
    ok = (
        bool(emissions)
        and match_metadata_ok
        and case["expected_text"] in text
        and source_type_seen
        and not old_tool_seen
        and graph.stream_calls == 0
    )
    return {
        "query": case["query"],
        "ok": ok,
        "expected_source_type": expected_source_type,
        "expected_has_matches": expected_has_matches,
        "expected_text": case["expected_text"],
        "emission_count": len(emissions),
        "stream_calls": graph.stream_calls,
        "metadata_nodes": metadata_nodes,
        "old_tool_seen": old_tool_seen,
        "source_type_seen": source_type_seen,
        "final_preview": text[:800],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证聊天入口优先返回结构化 payload，不落回旧 RAG chunk")
    parser.add_argument("--report", default=str(ROOT / "datasets" / "structured" / "structured_chat_runtime_validation.json"))
    args = parser.parse_args()

    rag = RuntimeRagStub()
    chat = ChatInterface(rag)
    results = [run_case(chat, rag.agent_graph, case) for case in RUNTIME_CASES]
    failing_rag = FailingRetrievalRagStub()
    failure_case = run_case(
        ChatInterface(failing_rag),
        failing_rag.agent_graph,
        {
            "query": "肚子疼而且发冷",
            "expected_text": "目前证据不足",
            "expected_has_matches": False,
        },
    )
    failure_case["name"] = "clinical_retrieval_failure_does_not_fall_through"
    results.append(failure_case)
    report = {
        "ok": all(item["ok"] for item in results),
        "cases": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
