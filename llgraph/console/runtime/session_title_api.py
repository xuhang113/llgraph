"""Web 会话标题更新。"""

from __future__ import annotations

from pathlib import Path

from llgraph.session.session_meta import get_session_title, load_session_meta, set_session_title


def update_session_display_title(
    workspace: Path,
    thread_id: str,
    title: str,
) -> tuple[bool, str, str]:
    """
    更新 Agent / Plan 侧边栏展示标题。

    Plan 会话同步更新 plan.json 中的 title（若存在）。

    @param workspace 工作区根
    @param thread_id 会话 thread_id
    @param title 新标题
    @return (成功, 提示, 规范化后的标题)
    """
    ok, msg = set_session_title(workspace, thread_id, title, source="manual")
    if not ok:
        return False, msg, ""

    normalized = get_session_title(workspace, thread_id) or title.strip()
    meta = load_session_meta(workspace, thread_id)
    is_plan = meta.get("session_kind") == "plan" or thread_id.startswith("plan-")
    if is_plan:
        plan_id = str(meta.get("plan_id") or "").strip()
        if plan_id:
            from llgraph.plan.config import resolve_plan_settings
            from llgraph.plan.plan_store import load_plan, save_plan

            settings = resolve_plan_settings(workspace)
            plan = load_plan(workspace, plan_id, plans_dir=settings.plans_dir)
            if plan:
                plan["title"] = normalized
                save_plan(workspace, plan, plans_dir=settings.plans_dir)

    return True, msg, normalized
