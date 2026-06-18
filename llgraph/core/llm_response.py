"""从 LLM 响应提取用户可见正文（与 Agent _message_text 对齐，支持 thinking 块剥离）。"""

from __future__ import annotations

import ast
from typing import Any


def llm_content_text(content: Any, *, fallback_thinking: bool = False) -> str:
    """
    从 AIMessage.content 或等价结构提取展示/落盘文本。

    @param content str | list[dict] | 其它
    @param fallback_thinking 无 text 块时是否降级使用 thinking（结构化输出场景）
    @return 用户可见正文
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return _maybe_unwrap_repr_list(content, fallback_thinking=fallback_thinking)
    if isinstance(content, list):
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "thinking" and fallback_thinking:
                    thinking_parts.append(str(block.get("thinking", "")))
            elif isinstance(block, str):
                text_parts.append(block)
        text = "".join(text_parts).strip()
        if text:
            return text
        if fallback_thinking:
            thinking = "".join(thinking_parts).strip()
            if thinking:
                return thinking
        return ""
    return str(content).strip()


def llm_response_text(response: Any, *, fallback_thinking: bool = False) -> str:
    """
    从 llm.invoke 返回值提取正文。

    @param response LangChain AIMessage 或兼容对象
    @param fallback_thinking 无 text 时是否用 thinking
    @return 正文
    """
    content = getattr(response, "content", response)
    return llm_content_text(content, fallback_thinking=fallback_thinking)


def normalize_stored_llm_text(raw: Any, *, fallback_thinking: bool = False) -> str:
    """
    规范化已落盘字段（含历史 str(list) 脏数据）。

    @param raw plan_state.final_report 等
    @param fallback_thinking 是否降级 thinking
    @return 清洗后文本
    """
    if raw is None:
        return ""
    if isinstance(raw, list):
        return llm_content_text(raw, fallback_thinking=fallback_thinking)
    if isinstance(raw, str):
        return _maybe_unwrap_repr_list(raw, fallback_thinking=fallback_thinking)
    return llm_content_text(raw, fallback_thinking=fallback_thinking)


def _maybe_unwrap_repr_list(text: str, *, fallback_thinking: bool) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("[") and ("'type'" in stripped or '"type"' in stripped):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                inner = llm_content_text(parsed, fallback_thinking=fallback_thinking)
                if inner:
                    return inner
        except (ValueError, SyntaxError):
            pass
    return stripped
