from __future__ import annotations

import json
import os
import re
from typing import Any

import config


class _BoundStructuredHTTPClient:
    def __init__(self, client: "DeepSeekStructuredHTTPClient", schema):
        self._client = client
        self._schema = schema

    def invoke(self, messages):
        return self._client.invoke_structured(messages, self._schema)


class DeepSeekStructuredHTTPClient:
    """Minimal one-shot DeepSeek client used only by Query Translator."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.model = model
        self._api_key = api_key
        self._endpoint = base_url.rstrip("/") + "/chat/completions"

    def with_config(self, **kwargs):
        return self

    def with_structured_output(self, schema=None, **kwargs):
        if schema is None:
            raise ValueError("structured schema is required")
        return _BoundStructuredHTTPClient(self, schema)

    @staticmethod
    def _message_payload(messages) -> list[dict[str, str]]:
        role_map = {"system": "system", "human": "user", "ai": "assistant"}
        payload: list[dict[str, str]] = []
        for message in messages:
            role = role_map.get(getattr(message, "type", ""), "user")
            content = getattr(message, "content", message)
            payload.append({"role": role, "content": str(content)})
        return payload

    @staticmethod
    def _parse_arguments(data: dict[str, Any]) -> dict[str, Any]:
        message = data.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments", "{}")
        else:
            arguments = message.get("content", "{}")
            arguments = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(arguments).strip())
        if isinstance(arguments, dict):
            return arguments
        return json.loads(arguments)

    def invoke_structured(self, messages, schema):
        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured")
        import httpx

        timeout_seconds = float(getattr(config, "LLM_SYMPTOM_TRANSLATOR_TIMEOUT_SECONDS", 12))
        configured_connect_timeout = float(
            getattr(config, "LLM_SYMPTOM_TRANSLATOR_CONNECT_TIMEOUT_SECONDS", 4)
        )
        connect_timeout = min(max(1.0, configured_connect_timeout), timeout_seconds)
        read_timeout = max(1.0, timeout_seconds - connect_timeout)
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        function_name = "emit_query_translation"
        request = {
            "model": self.model,
            "messages": self._message_payload(messages),
            "temperature": 0,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": "输出经过本地候选词约束的查询翻译结果",
                        "parameters": schema.model_json_schema(),
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": function_name}},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.post(self._endpoint, headers=headers, json=request)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text[:500].replace(self._api_key, "[REDACTED]")
                raise RuntimeError(f"DeepSeek structured request failed: {response.status_code} {detail}") from exc
            data = response.json()
        return schema.model_validate(self._parse_arguments(data))


def create_llm():
    """Create the configured chat model without importing the full RAG system."""
    active = config.ACTIVE_LLM_CONFIG
    cfg = config.LLM_CONFIGS.get(active)
    if not cfg:
        raise ValueError(f"不支持的 LLM 提供商: {active}")

    model = cfg["model"]
    temperature = cfg["temperature"]
    if active == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature, base_url=cfg.get("url"))
    if active == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if active == "deepseek":
        from langchain_openai import ChatOpenAI

        class DeepSeekChatOpenAI(ChatOpenAI):
            def bind_tools(self, tools, **kwargs):
                if kwargs.get("strict") is None:
                    kwargs["strict"] = False
                return super().bind_tools(tools, **kwargs)

            def with_structured_output(
                self, schema=None, *, method="json_schema", include_raw=False, strict=None, **kwargs
            ):
                return super().with_structured_output(
                    schema,
                    method="function_calling",
                    include_raw=include_raw,
                    strict=False,
                    **kwargs,
                )

        return DeepSeekChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            timeout=getattr(config, "LLM_SYMPTOM_TRANSLATOR_TIMEOUT_SECONDS", 12),
            max_retries=getattr(config, "LLM_SYMPTOM_TRANSLATOR_MAX_RETRIES", 1),
        )
    if active == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)
    if active == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    raise ValueError(f"不支持的 LLM 提供商: {active}")


def create_query_translator_client():
    """Create a bounded one-shot client for Query Translator."""
    active = config.ACTIVE_LLM_CONFIG
    cfg = config.LLM_CONFIGS.get(active)
    if not cfg:
        raise ValueError(f"不支持的 LLM 提供商: {active}")
    if active == "deepseek":
        return DeepSeekStructuredHTTPClient(
            model=cfg["model"],
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=cfg.get("url", "https://api.deepseek.com"),
        )
    return create_llm()
