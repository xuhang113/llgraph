"""向后兼容：tool_call 修复入口已迁至 llgraph.adapters.inbound。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, RemoveMessage

from llgraph.adapters.inbound import normalize_ai_response


def _ai_message_changed(before: AIMessage, after: AIMessage) -> bool:
    if list(before.tool_calls or []) != list(after.tool_calls or []):
        return True
    if before.content != after.content:
        return True
    invalid_before = list(getattr(before, "invalid_tool_calls", None) or [])
    invalid_after = list(getattr(after, "invalid_tool_calls", None) or [])
    return invalid_before != invalid_after


def repair_streaming_tool_calls_on_ai_message(
    msg: AIMessage,
    *,
    workspace=None,
    model_id: str | None = None,
) -> tuple[AIMessage, bool]:
    """
    归一化 AIMessage tool_calls（兼容旧调用；请优先 normalize_ai_response）。

    @param msg assistant 消息
    @param workspace 工作区根（可选，用于按模型 profile）
    @param model_id 模型 id
    @return (修复后消息, 是否改写)
    """
    normalized = normalize_ai_response(msg, workspace, model_id)
    return normalized, _ai_message_changed(msg, normalized)


def repair_tool_calls_post_model_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph post_model_hook：修复本轮 AIMessage 的 tool_calls。

    @param state ReAct 图状态
    @return 状态增量
    """
    messages = list(state.get("messages") or [])
    if not messages:
        return {}
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return {}

    repaired, changed = repair_streaming_tool_calls_on_ai_message(last)
    if not changed:
        return {}

    if getattr(last, "id", None):
        return {"messages": [RemoveMessage(id=last.id), repaired]}
    return {"messages": [repaired]}
