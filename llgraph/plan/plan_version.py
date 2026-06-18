"""Plan 模式版本 diff（Phase 3）。"""

from __future__ import annotations

from typing import Any


def diff_plan_versions(old: dict[str, Any], new: dict[str, Any]) -> str:
    """
    对比两个 plan 版本。

    @param old 旧 plan
    @param new 新 plan
    @return 可读 diff 文本
    """
    lines = [
        f"title: {old.get('title')} → {new.get('title')}",
        f"version: {old.get('version')} → {new.get('version')}",
        f"task count: {len(old.get('tasks') or [])} → {len(new.get('tasks') or [])}",
    ]
    old_ids = {str(t.get("id")) for t in (old.get("tasks") or []) if isinstance(t, dict)}
    new_ids = {str(t.get("id")) for t in (new.get("tasks") or []) if isinstance(t, dict)}
    added = new_ids - old_ids
    removed = old_ids - new_ids
    if added:
        lines.append(f"added tasks: {', '.join(sorted(added))}")
    if removed:
        lines.append(f"removed tasks: {', '.join(sorted(removed))}")
    return "\n".join(lines)
