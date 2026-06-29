"""Plan 子图 ReAct 路由：仅 visible text 含结构化交付物时可 END。"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from llgraph.core.agent_turn import FALLBACK_INCOMPLETE_TURN
from llgraph.core.llm_response import llm_content_text


def _json_dict_from_text(text: str) -> dict[str, Any] | None:
    from llgraph.plan.nodes.planner import _extract_plan_json_raw

    raw = _extract_plan_json_raw(text)
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _planner_text_has_deliverable(text: str) -> bool:
    data = _json_dict_from_text(text)
    if not data:
        return False
    tasks = data.get("tasks")
    return isinstance(tasks, list) and len(tasks) > 0


def _worker_text_has_deliverable(text: str) -> bool:
    data = _json_dict_from_text(text)
    if not data:
        return False
    return bool(str(data.get("summary") or "").strip() or str(data.get("status") or "").strip())


def _visible_body(msg: AIMessage) -> str:
    return llm_content_text(msg.content, fallback_thinking=False).strip()


def _is_fallback_visible(text: str) -> bool:
    return text.strip() == FALLBACK_INCOMPLETE_TURN


def planner_deliverable_complete(msg: AIMessage) -> bool:
    """
    Planner：仅当 visible text（type:text）含可解析 plan JSON 时视为 turn 完成。

    thinking / llgraph.thinking_text 不算交付物，不应触发 END。

    @param msg 最后一条 AI 消息
    @return 是否可 END
    """
    visible = _visible_body(msg)
    if not visible or _is_fallback_visible(visible):
        return False
    return _planner_text_has_deliverable(visible)


def worker_deliverable_complete(msg: AIMessage) -> bool:
    """
    Worker：仅当 visible text 含 worker 结果 JSON 时视为 turn 完成。

    @param msg 最后一条 AI 消息
    @return 是否可 END
    """
    visible = _visible_body(msg)
    if not visible or _is_fallback_visible(visible):
        return False
    return _worker_text_has_deliverable(visible)


def resolve_structured_complete_fn(
    subgraph_kind: str | None,
) -> Callable[[AIMessage], bool] | None:
    """
    Plan 子图 structured END 谓词（仅 visible 交付物；Chat Agent 为 None）。

    @param subgraph_kind planner | worker | None
    @return 谓词或 None
    """
    if subgraph_kind == "planner":
        return planner_deliverable_complete
    if subgraph_kind == "worker":
        return worker_deliverable_complete
    return None


def extract_structured_deliverable_text(
    messages: list[BaseMessage] | list[Any],
    *,
    subgraph_kind: str,
) -> str:
    """
    从 checkpoint 提取 visible 结构化交付正文（与 END 谓词同语义）。

    @param messages 子图 messages
    @param subgraph_kind planner | worker
    @return 交付正文
    """
    if subgraph_kind not in ("planner", "worker"):
        return ""
    has_fn = _planner_text_has_deliverable if subgraph_kind == "planner" else _worker_text_has_deliverable
    for msg in reversed(messages or []):
        role = getattr(msg, "type", None) or getattr(msg, "role", "")
        if str(role) not in ("ai", "assistant"):
            continue
        if not isinstance(msg, AIMessage):
            continue
        visible = _visible_body(msg)
        if visible and not _is_fallback_visible(visible) and has_fn(visible):
            return visible
    return ""
