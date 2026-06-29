"""Plan 子图 messages 落盘（Worker / Planner 共用）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llgraph.plan.plan_registry import subgraph_messages_path


def _message_row(msg: Any) -> dict[str, str]:
    role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
    content = getattr(msg, "content", "")
    return {"role": str(role), "content": content}


def persist_subgraph_messages(
    workspace: Path,
    thread_id: str,
    subgraph_key: str,
    messages: list[Any],
) -> Path | None:
    """
    写入 subgraphs/{key}/messages.jsonl。

    @param subgraph_key Worker task id（w1）或 planner 目录名（planner-v1）
    @return 写入路径；无内容时 None
    """
    rows: list[str] = []
    for msg in messages or []:
        rows.append(json.dumps(_message_row(msg), ensure_ascii=False))
    if not rows:
        return None
    path = subgraph_messages_path(workspace, thread_id, subgraph_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def collect_and_persist_subgraph_messages(
    workspace: Path,
    thread_id: str,
    subgraph_key: str,
    subgraph: Any,
    sub_thread: str,
    *,
    fallback_messages: list[Any] | None = None,
) -> Path | None:
    """
    从 checkpoint 收集 messages 并落盘；collect 失败时用 fallback。

    @param subgraph_key 落盘目录名
    @param fallback_messages invoke 返回的 messages（可选）
    @return 写入路径或 None
    """
    from llgraph.plan.subgraphs.base import collect_subgraph_messages

    messages = collect_subgraph_messages(subgraph, sub_thread)
    if not messages and fallback_messages:
        messages = list(fallback_messages)
    return persist_subgraph_messages(workspace, thread_id, subgraph_key, messages)
