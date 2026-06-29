"""Anthropic 结构化 tool_use 入站校验（禁止正文 XML/Token 降级解析）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from llgraph.adapters.inbound.kimi_native import (
    ai_message_text_content,
    content_has_kimi_tool_tokens,
)
from llgraph.adapters.inbound.kimi_native import _PLAIN_FUNCTIONS_HEAD
from llgraph.adapters.inbound.xml_tool_call import content_has_xml_tool_calls


class UnstructuredToolCallError(RuntimeError):
    """
    模型未返回 Anthropic 结构化 ``tool_use`` / ``AIMessage.tool_calls``，
    却在正文泄漏 XML、Kimi token 或 plain ``functions.*`` markup。
    """

    def __init__(
        self,
        *,
        model_id: str | None,
        detail: str,
        preview: str = "",
    ) -> None:
        self.model_id = model_id
        self.detail = detail
        self.preview = preview
        model_label = model_id or "unknown"
        hint = (
            "请确认网关返回原生 tool_use 块（LangChain tool_calls），"
            "而非正文 XML/JSON 字符串；检查 bind_tools、tool_choice 与模型 id。"
        )
        body = f"[{model_label}] {detail}。{hint}"
        if preview:
            body = f"{body}\n响应片段: {preview[:400]}"
        super().__init__(body)


def ai_message_has_structured_tool_calls(msg: AIMessage) -> bool:
    """
    是否已有 Anthropic/LangChain 结构化 tool call。

    @param msg assistant 消息
    @return 是否含 tool_calls 或 content 内 tool_use 块
    """
    if msg.tool_calls:
        return True
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in content
        )
    return False


def content_has_leaked_tool_markup(text: str) -> bool:
    """
    正文是否含非结构化的工具调用 markup（XML / Kimi token / plain functions.*）。

    @param text 合并后的 assistant 文本
    @return 是否泄漏
    """
    if not text or not text.strip():
        return False
    if content_has_kimi_tool_tokens(text):
        return True
    if content_has_xml_tool_calls(text):
        return True
    return _PLAIN_FUNCTIONS_HEAD.search(text) is not None


def validate_structured_tool_response(
    msg: AIMessage,
    model_id: str | None = None,
) -> None:
    """
    校验入站 assistant 响应：禁止正文工具 markup 替代结构化 tool_calls。

    @param msg LangChain 原始/修复后的 AIMessage
    @param model_id 当前模型 id（错误信息用）
    @raises UnstructuredToolCallError 正文含工具 markup 但无结构化 tool_calls
    """
    if ai_message_has_structured_tool_calls(msg):
        return
    invalid = list(getattr(msg, "invalid_tool_calls", None) or [])
    if invalid:
        raise UnstructuredToolCallError(
            model_id=model_id,
            detail="存在 invalid_tool_calls 碎片且未能修复为结构化 tool_calls",
            preview=str(invalid[0])[:200] if invalid else "",
        )
    text = ai_message_text_content(msg)
    if content_has_leaked_tool_markup(text):
        raise UnstructuredToolCallError(
            model_id=model_id,
            detail="模型在正文返回工具 XML/Token，但未提供结构化 tool_use/tool_calls",
            preview=text,
        )
