"""子 Agent messages 落盘（子 thread 会话目录，可供 UI /history 点入）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llgraph.session.user_storage import session_messages_path, session_thread_dir


def subagent_messages_path(workspace: Path, parent_thread_id: str, key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key.strip()) or "sub"
    return session_thread_dir(workspace, parent_thread_id) / "subagents" / safe / "messages.jsonl"


def persist_subagent_messages(
    workspace: Path,
    parent_thread_id: str,
    key: str,
    messages: list[Any],
    *,
    sub_thread: str | None = None,
) -> Path | None:
    """
    写入子会话 messages.jsonl；并在父目录留一份索引副本。

    @param key 父目录下 subagents/{key}/ 名
    @param sub_thread 子 thread_id；有则写正式会话 messages
    """
    rows: list[str] = []
    for msg in messages or []:
        role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")
        rows.append(json.dumps({"role": str(role), "content": content}, ensure_ascii=False))
    if not rows:
        return None
    text = "\n".join(rows) + "\n"
    primary: Path | None = None
    if sub_thread and sub_thread.strip():
        primary = session_messages_path(workspace, sub_thread.strip())
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_text(text, encoding="utf-8")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key.strip()) or "sub"
    mirror = session_thread_dir(workspace, parent_thread_id) / "subagents" / safe / "messages.jsonl"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(text, encoding="utf-8")
    return primary or mirror
