"""入站 AIMessage 归一化入口（StateGraph / 探测脚本共用）。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from llgraph.adapters.inbound.kimi_native import (
    _apply_cleaned_text_to_content,
    ai_message_text_content,
)
from llgraph.adapters.inbound.profile import InboundAdapterProfile, resolve_inbound_profile
from llgraph.adapters.inbound.streaming import repair_streaming_invalid_tool_calls
from llgraph.adapters.inbound.structured import (
    ai_message_has_structured_tool_calls,
    content_has_leaked_tool_markup,
    validate_structured_tool_response,
)
from llgraph.adapters.inbound.xml_tool_call import strip_inbound_tool_call_markup


def normalize_ai_response(
    msg: AIMessage,
    workspace: Path | None,
    model_id: str | None = None,
    *,
    profile: InboundAdapterProfile | None = None,
) -> AIMessage:
    """
    将 LLM 原始 AIMessage 规范为编排层可消费形态。

    仅接受 Anthropic/LangChain 结构化 ``tool_use`` / ``tool_calls``；
    正文 XML、Kimi token、plain ``functions.*`` **不再解析**，检测到即报错。
    流式 ``invalid_tool_calls`` JSON 碎片可在结构化字段内修复。

    @param msg LangChain 原始响应
    @param workspace 工作区根
    @param model_id 模型 id
    @param profile 可选，跳过 resolve
    @return 归一化后的 AIMessage
    @raises UnstructuredToolCallError 工具 markup 泄漏且无结构化 tool_calls
    """
    if not isinstance(msg, AIMessage):
        return msg

    effective = profile or resolve_inbound_profile(workspace, model_id)
    effective_model = (model_id or "").strip() or None
    out = msg

    if effective.repair_streaming_tool_calls:
        out, _ = repair_streaming_invalid_tool_calls(out)

    validate_structured_tool_response(out, effective_model)

    if ai_message_has_structured_tool_calls(out):
        text = ai_message_text_content(out)
        if content_has_leaked_tool_markup(text):
            cleaned = strip_inbound_tool_call_markup(text)
            if cleaned != text.strip():
                new_content = _apply_cleaned_text_to_content(out.content, cleaned)
                out = out.model_copy(update={"content": new_content})

    return out


def classify_tool_call_response(msg: AIMessage) -> dict[str, bool]:
    """
    探测用：分类 AIMessage 的 tool call 形态。

    @param msg assistant 消息
    @return structured / kimi_tokens / has_invalid 标志
    """
    from llgraph.adapters.inbound.kimi_native import content_has_kimi_tool_tokens
    from llgraph.adapters.inbound.xml_tool_call import content_has_xml_tool_calls

    text = ai_message_text_content(msg)
    return {
        "structured_tool_calls": ai_message_has_structured_tool_calls(msg),
        "kimi_tokens_in_content": content_has_kimi_tool_tokens(text),
        "xml_tool_calls_in_content": content_has_xml_tool_calls(text),
        "has_invalid_tool_calls": bool(getattr(msg, "invalid_tool_calls", None)),
    }
