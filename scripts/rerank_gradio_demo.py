from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_RUNTIME = ROOT / ".runtime"
PROJECT_RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(PROJECT_RUNTIME / "gradio_tmp"))

import gradio as gr


PROJECT = ROOT / "project"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

# Keep offline model discovery inside the repository. The remote server has no
# reliable Hugging Face access, and service restarts do not inherit shell-only
# cache settings.
PROJECT_HF_HOME = ROOT / ".cache" / "huggingface"
if PROJECT_HF_HOME.exists():
    os.environ.setdefault("HF_HOME", str(PROJECT_HF_HOME))

import config  # noqa: E402
from core.syndrome_retriever import (  # noqa: E402
    SyndromeRetriever,
    format_local_no_match,
    format_syndrome_answer,
    format_syndrome_clarification,
    should_refuse_ungrounded_local_query,
    should_request_structured_clarification,
    should_use_structured_answer,
)
from ui.css import custom_css  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402


DICTIONARY_PATH = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
RERANK_MODEL_PATH = ROOT / "models" / "bge-reranker-v2-m3"
SERVER_NAME = os.environ.get("RERANK_DEMO_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.environ.get("RERANK_DEMO_SERVER_PORT", "17860"))
TOP_K_DEFAULT = 8
USER_RESULT_LIMIT = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                rows.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def configure_runtime() -> None:
    config.ENABLE_LLM_SYMPTOM_TRANSLATOR = False
    config.EMBEDDING_DEVICE = "cuda"
    config.EMBEDDING_LOCAL_FILES_ONLY = True
    config.ENABLE_SYNDROME_RERANK = True
    config.SYNDROME_RERANK_MODEL = str(RERANK_MODEL_PATH)
    config.SYNDROME_RERANK_CANDIDATES = 8
    config.SYNDROME_RERANK_DEVICE = "cuda"
    config.SYNDROME_RERANK_MAX_LENGTH = 256
    config.SYNDROME_RERANK_BATCH_SIZE = 8
    config.SYNDROME_RERANK_MODE = "evidence_first"
    config.SYNDROME_RERANK_LOCAL_FILES_ONLY = True


def build_runtime() -> tuple[SyndromeRetriever, SyndromeRetriever, int]:
    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(DICTIONARY_PATH)
    if not (RERANK_MODEL_PATH / "model.safetensors").exists():
        raise FileNotFoundError(RERANK_MODEL_PATH / "model.safetensors")

    configure_runtime()
    print(f"[startup] HF_HOME={os.environ.get('HF_HOME', '-')}", flush=True)
    rows = read_jsonl(DICTIONARY_PATH)
    print(f"[startup] loading embedding model for {len(rows)} entries", flush=True)
    embedding = HuggingFaceEmbeddings(
        model_name=config.DENSE_MODEL,
        model_kwargs={"device": "cuda", "local_files_only": True},
    )
    vector_size = len(embedding.embed_query("test"))
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=config.SYNDROME_COLLECTION,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )
    vectors = embedding.embed_documents([str(row.get("search_text", "")) for row in rows])
    print("[startup] embedding index ready; loading reranker", flush=True)
    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(row.get("entry_id", index)))),
            vector=vector,
            payload=row,
        )
        for index, (row, vector) in enumerate(zip(rows, vectors))
    ]
    client.upsert(collection_name=config.SYNDROME_COLLECTION, points=points, wait=True)

    baseline = SyndromeRetriever(client=client, embedding=embedding)
    reranked = SyndromeRetriever(client=client, embedding=embedding)

    # Preload the 2.3 GB CrossEncoder before serving the page so the first click is predictable.
    reranker = reranked._ensure_reranker()
    reranker.score(
        "warmup",
        [{"payload": {"title": "warmup", "evidence": "warmup"}}],
    )
    print("[startup] reranker ready", flush=True)
    return baseline, reranked, len(rows)


def _decision_text(result: dict[str, Any]) -> str:
    decision = result.get("decision", {}) or {}
    status = str(decision.get("status", "unknown"))
    reasons = decision.get("reasons", []) or []
    gate = should_use_structured_answer(result)
    return f"status={status}; gate={gate}; reasons={', '.join(map(str, reasons)) or '-'}"


def _rows(result: dict[str, Any]) -> list[list[Any]]:
    output: list[list[Any]] = []
    for rank, match in enumerate(result.get("matches", []), start=1):
        payload = match.get("payload", {}) or {}
        score = match.get("rerank_score", match.get("score", 0.0))
        output.append(
            [
                rank,
                payload.get("formula") or payload.get("title") or "-",
                payload.get("source_type", "-"),
                round(float(score or 0.0), 5),
                "、".join(map(str, match.get("matched_terms", []) or [])) or "-",
                json.dumps(match.get("missing_required_symptom_groups", []) or [], ensure_ascii=False),
                "、".join(map(str, match.get("forbidden_conflicts", []) or [])) or "-",
            ]
        )
    return output


def compare(query: str, top_k: int) -> tuple[str, list[list[Any]], list[list[Any]], dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        raise gr.Error("请输入症状或方证描述。")
    top_k = max(1, min(int(top_k), 12))

    config.ENABLE_SYNDROME_RERANK = False
    baseline_started = time.perf_counter()
    baseline_result = BASELINE_RETRIEVER.search(query, limit=top_k, candidate_limit=max(80, top_k * 10))
    baseline_ms = (time.perf_counter() - baseline_started) * 1000

    config.ENABLE_SYNDROME_RERANK = True
    rerank_started = time.perf_counter()
    rerank_result = RERANK_RETRIEVER.search(query, limit=top_k, candidate_limit=max(80, top_k * 10))
    rerank_ms = (time.perf_counter() - rerank_started) * 1000

    baseline_top = _rows(baseline_result)
    rerank_top = _rows(rerank_result)
    changed = sum(
        1
        for left, right in zip(baseline_top, rerank_top)
        if left[1] != right[1]
    )
    summary = (
        f"### 对比结果\n"
        f"- 基线：{baseline_ms:.1f} ms；{_decision_text(baseline_result)}\n"
        f"- Rerank：{rerank_ms:.1f} ms；{_decision_text(rerank_result)}\n"
        f"- Top-{top_k} 位置变化：{changed}\n"
        f"- 配置：`bge-reranker-v2-m3 / evidence_first / candidates=8 / max_length=256 / A100`"
    )
    debug = {
        "query": rerank_result.get("query", {}),
        "baseline_decision": baseline_result.get("decision", {}),
        "rerank_decision": rerank_result.get("decision", {}),
        "rerank_debug": rerank_result.get("retrieval_debug", {}).get("rerank", {}),
        "baseline_latency_ms": round(baseline_ms, 2),
        "rerank_latency_ms": round(rerank_ms, 2),
    }
    return summary, baseline_top, rerank_top, debug


def _fallback_answer(result: dict[str, Any]) -> str:
    """Return one user-facing answer from the reranked, evidence-gated result."""
    if should_use_structured_answer(result):
        return format_syndrome_answer(result)
    if should_request_structured_clarification(result):
        return format_syndrome_clarification(result)
    if should_refuse_ungrounded_local_query(result):
        return format_local_no_match(result)
    if result.get("matches"):
        # Knowledge queries may have a useful top result even when the clinical
        # answer gate is intentionally conservative.
        return format_syndrome_answer(result)
    return format_local_no_match(result)


def respond(query: str, history: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
    query = (query or "").strip()
    if not query:
        raise gr.Error("请先描述症状或输入想查询的方剂、药材、古籍条文。")

    messages = list(history or [])
    messages.append({"role": "user", "content": query})
    started = time.perf_counter()
    try:
        config.ENABLE_SYNDROME_RERANK = True
        result = RERANK_RETRIEVER.search(
            query,
            limit=USER_RESULT_LIMIT,
            candidate_limit=80,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        answer = _fallback_answer(result)
        answer += (
            "\n\n---\n"
            f"*已通过本地知识库检索、语义重排和证据门控完成回答 · {latency_ms:.0f} ms*"
        )
    except Exception:
        answer = (
            "系统暂时无法完成本次检索，请稍后重试。"
            "如果问题涉及个人诊疗，请咨询具备资质的中医师。"
        )
    messages.append({"role": "assistant", "content": answer})
    return "", messages


def clear_chat() -> tuple[str, list[dict[str, Any]]]:
    return "", []


configure_runtime()
BASELINE_RETRIEVER, RERANK_RETRIEVER, ENTRY_COUNT = build_runtime()


HEADERS = ["排名", "方剂/标题", "证据类型", "分数", "命中词", "缺失必要组", "禁忌冲突"]
USER_CSS = """
    #assistant-shell { gap: 18px; }
    #capability-card { min-width: 260px; }
    #capability-card .prose { font-size: 14px; }
    #rag-chatbot { min-height: 560px; }
    #privacy-note { color: #5d584b; font-size: 13px; text-align: center; margin-top: 10px; }
    @media (max-width: 760px) {
        .gradio-container { padding: 12px !important; }
        #app-hero { padding: 20px !important; box-shadow: 5px 5px 0 #111 !important; }
        #app-hero h1 { font-size: 27px !important; }
        #rag-chatbot { min-height: 460px; }
    }
"""

with gr.Blocks(title="仲景智检 · 中医方证知识助手") as demo:
    gr.HTML(
        f"""
        <section id="app-hero">
            <span class="hero-mark">AGENTIC RAG · BGE RERANK</span>
            <h1>仲景智检 · 中医方证知识助手</h1>
            <p>描述症状或查询方剂、药材与古籍条文。系统将从 {ENTRY_COUNT} 条结构化本地证据中给出一份经过语义重排与安全门控的回答。</p>
        </section>
        """
    )

    with gr.Row(elem_id="assistant-shell"):
        with gr.Column(scale=4, elem_classes=["paper-card", "chat-card"]):
            chatbot = gr.Chatbot(
                value=[],
                height=560,
                elem_id="rag-chatbot",
                show_label=False,
                layout="bubble",
                avatar_images=(None, PROJECT / "assets" / "chatbot_avatar.png"),
                placeholder=(
                    "<strong>你好，我可以检索本地中医方证知识库。</strong><br>"
                    "请尽量说明寒热、汗出、疼痛部位、口渴、咳痰等信息。"
                ),
            )
            with gr.Row(elem_id="chat-input-row"):
                query_input = gr.Textbox(
                    placeholder="例如：恶寒、无汗、身体疼痛，并且喘咳……",
                    show_label=False,
                    container=False,
                    lines=2,
                    max_lines=5,
                    scale=7,
                    elem_id="chat-input",
                )
                send_button = gr.Button("发送", variant="primary", scale=1, elem_id="send-btn")
            with gr.Row():
                clear_button = gr.Button("清空对话", size="sm", elem_id="clear-chat-btn")
            gr.Examples(
                examples=[
                    "恶寒，无汗，身体疼痛，并且喘咳",
                    "怕风，发热，出汗，头痛",
                    "只知道咳嗽、痰多，其他情况说不清楚",
                    "口渴但又说不渴，寒热表现互相矛盾",
                ],
                inputs=query_input,
                label="试试这些问题",
            )

        with gr.Column(scale=1, elem_id="capability-card", elem_classes=["paper-card"]):
            gr.Markdown(
                """
                ### 回答机制

                1. 识别口语症状与中医表征
                2. 混合召回本地结构化证据
                3. BGE Cross-Encoder 语义重排
                4. 必要症状、禁忌与证据门控

                ### 使用提示

                - 信息不足时，系统会继续追问
                - 证据冲突时，不会强行推荐方剂
                - 回答附带古籍或数据来源

                > 本系统用于知识检索与项目演示，不代替医师诊断或处方。
                """
            )

    gr.HTML('<div id="privacy-note">本地知识库检索 · 单一最终答案 · 不展示内部候选排序</div>')

    send_event = send_button.click(
        respond,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot],
        api_name="chat",
        show_progress="minimal",
    )
    query_input.submit(
        respond,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot],
        api_name=False,
        show_progress="minimal",
    )
    clear_button.click(clear_chat, outputs=[query_input, chatbot], api_name=False)
    chatbot.clear(clear_chat, outputs=[query_input, chatbot])

    # Preserve the locked black-box benchmark endpoint without exposing its
    # baseline/rerank tables in the user interface.
    with gr.Group(visible="hidden"):
        benchmark_query = gr.Textbox()
        benchmark_top_k = gr.Number(value=TOP_K_DEFAULT, precision=0)
        benchmark_button = gr.Button("benchmark")
        benchmark_summary = gr.Markdown()
        benchmark_baseline = gr.Dataframe(headers=HEADERS)
        benchmark_rerank = gr.Dataframe(headers=HEADERS)
        benchmark_debug = gr.JSON()
    benchmark_button.click(
        compare,
        inputs=[benchmark_query, benchmark_top_k],
        outputs=[benchmark_summary, benchmark_baseline, benchmark_rerank, benchmark_debug],
        api_name="compare",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        css=custom_css + USER_CSS,
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=False,
        show_error=True,
    )
