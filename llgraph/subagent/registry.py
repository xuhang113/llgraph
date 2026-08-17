"""父会话下的子 Agent 注册表（供会话树发现）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llgraph.session.user_storage import session_thread_dir


def subagents_registry_path(workspace: Path, parent_thread_id: str) -> Path:
    return session_thread_dir(workspace, parent_thread_id) / "subagents.json"


def load_subagent_children(workspace: Path, parent_thread_id: str) -> list[dict[str, Any]]:
    path = subagents_registry_path(workspace, parent_thread_id)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("children") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [row for row in items if isinstance(row, dict)]


def register_subagent_child(
    workspace: Path,
    parent_thread_id: str,
    child: dict[str, Any],
) -> None:
    """
    登记或更新子 Agent 元数据。

    @param child 须含 sub_thread；建议 kind/sub_id/title/status
    """
    sub_thread = str(child.get("sub_thread") or "").strip()
    if not sub_thread:
        return
    path = subagents_registry_path(workspace, parent_thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_subagent_children(workspace, parent_thread_id)
    now = time.time()
    row = {
        **child,
        "sub_thread": sub_thread,
        "updated_at": now,
    }
    if "created_at" not in row:
        row["created_at"] = now
    replaced = False
    next_rows: list[dict[str, Any]] = []
    for item in existing:
        if str(item.get("sub_thread") or "") == sub_thread:
            merged = {**item, **row}
            if "created_at" in item:
                merged["created_at"] = item["created_at"]
            next_rows.append(merged)
            replaced = True
        else:
            next_rows.append(item)
    if not replaced:
        next_rows.append(row)
    path.write_text(
        json.dumps({"children": next_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
