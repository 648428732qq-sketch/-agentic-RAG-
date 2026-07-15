import json
import re
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from core.evidence_gate import decision_result
from core.syndrome_retriever import (
    format_local_no_match,
    format_syndrome_answer,
    format_syndrome_card,
    format_syndrome_clarification,
    should_request_structured_clarification,
    should_refuse_ungrounded_local_query,
    should_use_structured_answer,
)
from core.symptom_query_translator import infer_query_intent

SILENT_NODES = {"rewrite_query"}
SYSTEM_NODES = {"summarize_history", "rewrite_query"}
PUBLIC_TOOL_NAMES = {"search_child_chunks", "retrieve_parent_chunks"}
STRUCTURED_OUTPUT_TOOL_NAMES = {"QueryAnalysis"}

SYSTEM_NODE_CONFIG = {
    "rewrite_query":     {"title": "🔍 查询分析与改写"},
    "summarize_history": {"title": "📋 对话历史摘要"},
}

# --- 工具函数 ---

def stringify_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(stringify_content(item) for item in content)
    if isinstance(content, dict):
        for key in ("text", "content"):
            if key in content:
                return stringify_content(content[key])
        return str(content)
    return str(content)


def make_message(content, *, title=None, node=None):
    msg = {"role": "assistant", "content": content}
    if title or node:
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}
    return msg


def find_msg_idx(messages, node):
    return next(
        (i for i, m in enumerate(messages) if m.get("metadata", {}).get("node") == node),
        None,
    )


def parse_rewrite_json(buffer):
    match = re.search(r"\{.*\}", buffer, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def format_rewrite_content(buffer):
    data = parse_rewrite_json(buffer)
    if not data:
        return "⏳ 正在分析查询..."
    if data.get("is_clear"):
        lines = ["✅ **查询意图清晰**"]
        if data.get("questions"):
            lines += ["\n**改写后的查询:**"] + [f"- {q}" for q in data["questions"]]
    else:
        lines = ["❓ **查询意图不清晰**"]
        clarification = data.get("clarification_needed", "")
        if clarification and clarification.strip().lower() != "no":
            lines.append(f"\n需要进一步说明: *{clarification}*")
    return "\n".join(lines)

# --- 工具函数结束 ---

class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def _handle_system_node(self, chunk, node, response_messages, system_node_buffer):
        """更新（或创建）可折叠的系统节点消息，并显示澄清请求"""
        system_node_buffer[node] = system_node_buffer.get(node, "") + stringify_content(chunk.content)
        self._update_system_node_message(node, response_messages, system_node_buffer)

    def _handle_system_tool_call(self, chunk, node, response_messages, system_node_buffer):
        """处理结构化输出内部 tool call，不把它暴露成检索工具。"""
        handled = False
        for tc in getattr(chunk, "tool_calls", []) or []:
            if tc.get("name") not in STRUCTURED_OUTPUT_TOOL_NAMES:
                continue

            args = tc.get("args") or {}
            if args:
                system_node_buffer[node] = json.dumps(args, ensure_ascii=False)
                self._update_system_node_message(node, response_messages, system_node_buffer)
            handled = True
        return handled

    def _update_system_node_message(self, node, response_messages, system_node_buffer):
        buffer = system_node_buffer.get(node, "")
        title  = SYSTEM_NODE_CONFIG[node]["title"]
        content = format_rewrite_content(buffer) if node == "rewrite_query" else buffer

        idx = find_msg_idx(response_messages, node)
        if idx is None:
            response_messages.append(make_message(content, title=title, node=node))
        else:
            response_messages[idx]["content"] = content

        if node == "rewrite_query":
            self._surface_clarification(buffer, response_messages)

    def _surface_clarification(self, buffer, response_messages):
        """如果查询不清晰，添加/更新一条澄清消息"""
        data          = parse_rewrite_json(buffer) or {}
        clarification = data.get("clarification_needed", "")
        if not data.get("is_clear") and clarification.strip().lower() not in ("", "no"):
            cidx = find_msg_idx(response_messages, "clarification")
            if cidx is None:
                response_messages.append(make_message(clarification, node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification

    def _handle_tool_call(self, chunk, response_messages, active_tool_calls):
        """记录公开检索工具调用，内部结构化输出不展示。"""
        for tc in chunk.tool_calls:
            name = tc.get("name", "")
            if name not in PUBLIC_TOOL_NAMES:
                continue
            if tc.get("id") and tc["id"] not in active_tool_calls:
                active_tool_calls[tc["id"]] = {"name": name, "idx": None}

    def _handle_tool_result(self, chunk, response_messages, active_tool_calls):
        """将工具结果填入对应的可折叠消息中"""
        record = active_tool_calls.get(chunk.tool_call_id, {})
        tool_name = record.get("name") or getattr(chunk, "name", "检索结果")
        preview = str(chunk.content)[:500]
        suffix  = "\n..." if len(str(chunk.content)) > 500 else ""
        content = f"```\n{preview}{suffix}\n```"

        idx = record.get("idx")
        if idx is None:
            response_messages.append(make_message(content, title=f"🛠️ {tool_name}"))
            if chunk.tool_call_id:
                active_tool_calls[chunk.tool_call_id] = {"name": tool_name, "idx": len(response_messages) - 1}
        else:
            response_messages[idx]["content"] = content

    def _handle_llm_token(self, chunk, node, response_messages):
        """将流式 LLM token 追加到最后一条助手消息"""
        last = response_messages[-1] if response_messages else None
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            response_messages.append(make_message(""))
        response_messages[-1]["content"] += stringify_content(chunk.content)

    def chat(self, message, history):
        """流式返回 Gradio 聊天消息字典的生成器"""
        if not self.rag_system.agent_graph:
            yield "⚠️ 系统未初始化！"
            return

        user_message = message.strip()
        config        = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(config)

        try:
            response_messages  = []
            try:
                syndrome_result = self.rag_system.search_syndromes(user_message, limit=3)
            except Exception as exc:
                intent = infer_query_intent(user_message)
                syndrome_result = {
                    "query": {
                        "original_query": user_message,
                        "query_intent": intent,
                        "canonical_terms": [],
                    },
                    "matches": [],
                    "decision": decision_result(
                        "clarify" if intent == "clinical_symptom" else "no_match",
                        [f"retrieval_failure:{type(exc).__name__}"],
                    ),
                }
            syndrome_matches = syndrome_result.get("matches", [])
            if syndrome_matches:
                response_messages.append(
                    make_message(format_syndrome_card(syndrome_result), title="方证结构化匹配", node="syndrome_matches")
                )
                if should_use_structured_answer(syndrome_result):
                    response_messages.append(make_message(format_syndrome_answer(syndrome_result)))
                    yield response_messages
                    return
                yield response_messages
            if should_request_structured_clarification(syndrome_result):
                response_messages.append(make_message(format_syndrome_clarification(syndrome_result)))
                yield response_messages
                return
            if should_refuse_ungrounded_local_query(syndrome_result):
                response_messages.append(make_message(format_local_no_match(syndrome_result)))
                yield response_messages
                return

            if current_state.next:
                self.rag_system.agent_graph.update_state(config, {"messages": [HumanMessage(content=user_message)]})
                stream_input = None
            else:
                stream_input = {"messages": [HumanMessage(content=user_message)]}

            active_tool_calls  = {}
            system_node_buffer = {}

            for chunk, metadata in self.rag_system.agent_graph.stream(stream_input, config=config, stream_mode="messages"):
                node = metadata.get("langgraph_node", "")

                if node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                    self._handle_system_node(chunk, node, response_messages, system_node_buffer)

                elif node in SYSTEM_NODES and hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_system_tool_call(chunk, node, response_messages, system_node_buffer)

                elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_tool_call(chunk, response_messages, active_tool_calls)

                elif isinstance(chunk, ToolMessage):
                    self._handle_tool_result(chunk, response_messages, active_tool_calls)

                elif isinstance(chunk, AIMessageChunk) and chunk.content and node not in SILENT_NODES:
                    self._handle_llm_token(chunk, node, response_messages)

                yield response_messages

        except Exception as e:
            yield f"❌ 系统错误: {str(e)}"

    def clear_session(self):
        self.rag_system.reset_thread()
        self.rag_system.observability.flush()
