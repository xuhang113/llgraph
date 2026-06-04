"""Kimi 等网关：assistant tool 消息 reasoning_content + thinking 块注入。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from llgraph.context.chat_history_repair import (
    _REASONING_PLACEHOLDER,
    _extract_reasoning_text,
    ai_message_has_tool_calls,
)

_PATCHED = False


def _formatted_assistant_has_tool_use(block: dict[str, Any]) -> bool:
    """
    判断 Anthropic 格式 assistant 是否含 tool_use（网关视为 tool call 消息）。

    @param block formatted message
    @return 是否含 tool_use
    """
    content = block.get("content")
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and item.get("type") == "tool_use":
            return True
    return False


def _extract_thinking_block_from_ai(msg: AIMessage | None) -> dict[str, Any] | None:
    """
    从 AIMessage 列表 content 提取原始 thinking 块（含 signature）。

    @param msg assistant 消息
    @return thinking 块或 None
    """
    if msg is None:
        return None
    content = getattr(msg, "content", "")
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("type", "")).lower()
        if kind not in ("thinking", "redacted_thinking"):
            continue
        thinking_text = (
            block.get("thinking")
            or block.get("reasoning")
            or block.get("text")
            or block.get("data")
        )
        if not thinking_text:
            continue
        out: dict[str, Any] = {"type": "thinking", "thinking": str(thinking_text)}
        signature = block.get("signature")
        if signature is not None:
            out["signature"] = signature
        return out
    return None


def resolve_kimi_reasoning_content(msg: AIMessage) -> str:
    """
    解析应写入网关 payload 的 reasoning_content。

    @param msg 带 tool_calls 的 assistant 消息
    @return 非空 reasoning 文本
    """
    extra = dict(getattr(msg, "additional_kwargs", None) or {})
    raw = extra.get("reasoning_content")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    meta = extra.get("llgraph")
    if isinstance(meta, dict):
        thinking = meta.get("thinking_text")
        if isinstance(thinking, str) and thinking.strip():
            return thinking.strip()

    extracted = _extract_reasoning_text(msg)
    if extracted:
        return extracted

    content = getattr(msg, "content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()

    return _REASONING_PLACEHOLDER


def _normalize_assistant_content_list(block: dict[str, Any]) -> list[dict[str, Any]]:
    """
    将 formatted assistant content 规范为 block 列表。

    @param block formatted assistant
    @return content 块列表
    """
    content = block.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    if isinstance(content, str) and content.strip():
        return [{"type": "text", "text": content.strip()}]
    return []


def _ensure_thinking_block_in_content(
    block: dict[str, Any],
    reasoning: str,
    source: AIMessage | None,
) -> None:
    """
    在 assistant content 中补 thinking 块（Kimi 网关校验需要，仅顶层 reasoning_content 不够）。

    @param block formatted assistant
    @param reasoning 思考文本
    @param source 原始 AIMessage
    """
    content = _normalize_assistant_content_list(block)
    has_thinking = any(
        isinstance(b, dict)
        and str(b.get("type", "")).lower() in ("thinking", "redacted_thinking")
        for b in content
    )
    if has_thinking:
        block["content"] = content
        return

    original = _extract_thinking_block_from_ai(source)
    if original is not None:
        content.insert(0, original)
    elif reasoning.strip():
        content.insert(0, {"type": "thinking", "thinking": reasoning.strip()})
    block["content"] = content


def _apply_kimi_tool_assistant_fields(
    block: dict[str, Any],
    reasoning: str,
    source: AIMessage | None,
) -> None:
    """
    写入 Kimi tool assistant 所需的 reasoning_content 与 content.thinking。

    @param block formatted assistant
    @param reasoning 思考文本
    @param source 原始 AIMessage
    """
    block["reasoning_content"] = reasoning.strip() or _REASONING_PLACEHOLDER
    _ensure_thinking_block_in_content(block, block["reasoning_content"], source)


def inject_reasoning_into_formatted_messages(
    source_messages: list[BaseMessage],
    formatted_messages: list[dict[str, Any]],
) -> int:
    """
    为 formatted assistant 消息补上 reasoning_content 与 thinking 块。

    @param source_messages 原始 LangChain 消息（出站修链后）
    @param formatted_messages Anthropic 格式 messages
    @return 注入条数
    """
    tool_ai_messages = [
        msg
        for msg in source_messages
        if isinstance(msg, AIMessage) and ai_message_has_tool_calls(msg)
    ]

    ai_idx = 0
    injected = 0
    for block in formatted_messages:
        if block.get("role") != "assistant":
            continue
        if not _formatted_assistant_has_tool_use(block):
            continue
        source = (
            tool_ai_messages[ai_idx]
            if ai_idx < len(tool_ai_messages)
            else None
        )
        ai_idx += 1

        existing = block.get("reasoning_content")
        if isinstance(existing, str) and existing.strip():
            reasoning = existing.strip()
        else:
            reasoning = (
                resolve_kimi_reasoning_content(source)
                if source is not None
                else _REASONING_PLACEHOLDER
            )

        _apply_kimi_tool_assistant_fields(block, reasoning, source)
        injected += 1
    return injected


def missing_reasoning_on_formatted_tool_assistants(
    formatted_messages: list[dict[str, Any]],
) -> list[int]:
    """
    列出 formatted messages 中缺 reasoning_content 的 tool assistant 下标。

    @param formatted_messages Anthropic 格式 messages
    @return 缺失的 message 索引列表
    """
    missing: list[int] = []
    for idx, block in enumerate(formatted_messages):
        if block.get("role") != "assistant":
            continue
        if not _formatted_assistant_has_tool_use(block):
            continue
        raw = block.get("reasoning_content")
        if not isinstance(raw, str) or not raw.strip():
            missing.append(idx)
            continue
        content = block.get("content")
        if not isinstance(content, list):
            missing.append(idx)
            continue
        has_thinking = any(
            isinstance(b, dict)
            and str(b.get("type", "")).lower() in ("thinking", "redacted_thinking")
            for b in content
        )
        if not has_thinking:
            missing.append(idx)
    return missing


def resolve_kimi_thinking_payload() -> dict[str, str]:
    """
    Kimi k2.6 多轮 tool 需显式声明 keep=all，并与历史 reasoning 一并回传。

    @return payload.thinking 字段
    """
    return {"type": "enabled", "keep": "all"}


def model_requires_kimi_reasoning_payload(model_id: str | None) -> bool:
    """
    是否需在 HTTP payload 注入 reasoning_content。

    @param model_id 模型 id
    @return 是否 Kimi k2 等
    """
    if not model_id:
        return False
    mid = str(model_id).strip().lower()
    return "kimi" in mid or "k2.5" in mid or "k2.6" in mid or "k2-" in mid


def patch_gateway_kimi_reasoning_payload() -> None:
    """
    包装 ChatAnthropic._get_request_payload，向 Kimi 网关写入 reasoning 相关字段。
    """
    global _PATCHED
    if _PATCHED:
        return

    from langchain_anthropic import chat_models as anthropic_chat_models

    original = anthropic_chat_models.ChatAnthropic._get_request_payload

    def _patched_get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = original(self, input_, stop=stop, **kwargs)
        model_id = getattr(self, "model", None) or payload.get("model")
        if not model_requires_kimi_reasoning_payload(str(model_id or "")):
            return payload
        formatted = payload.get("messages")
        if not isinstance(formatted, list):
            return payload
        try:
            source = self._convert_input(input_).to_messages()
        except Exception:
            source = []
        inject_reasoning_into_formatted_messages(source, formatted)
        still_missing = missing_reasoning_on_formatted_tool_assistants(formatted)
        if still_missing:
            for idx in still_missing:
                block = formatted[idx]
                if not isinstance(block, dict) or block.get("role") != "assistant":
                    continue
                _apply_kimi_tool_assistant_fields(
                    block,
                    _REASONING_PLACEHOLDER,
                    None,
                )
        if payload.get("thinking") is None:
            payload["thinking"] = resolve_kimi_thinking_payload()
        return payload

    anthropic_chat_models.ChatAnthropic._get_request_payload = _patched_get_request_payload
    _PATCHED = True
