"""Plan 会话列举与切换。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgraph.plan.plan_phase_resolve import resolve_plan_phase
from llgraph.plan.plan_store import count_task_progress, load_plan, pick_richer_plan
from llgraph.plan.config import resolve_plan_settings
from llgraph.session.session_meta import get_session_title, resolve_session_display_title
from llgraph.session.user_storage import session_thread_dir, user_sessions_root


@dataclass(frozen=True)
class PlanTaskStub:
    """Plan 任务摘要（会话树 worker 节点）。"""

    id: str
    title: str
    status: str


@dataclass(frozen=True)
class PlanSummary:
    """Plan 会话摘要。"""

    thread_id: str
    plan_id: str
    title: str
    phase: str
    tasks_done: int
    tasks_total: int
    updated_at: str | None
    task_stubs: tuple[PlanTaskStub, ...] = ()


def _mtime_iso(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _load_plan_meta(workspace: Path, thread_id: str) -> dict[str, Any]:
    from llgraph.session.session_meta import load_session_meta

    meta = load_session_meta(workspace, thread_id)
    if meta.get("session_kind") == "plan":
        return meta
    if thread_id.startswith("plan-"):
        return meta
    return {}


def discover_plan_sessions(workspace: Path) -> list[PlanSummary]:
    """
    列举本工作区 Plan 会话。

    @param workspace 工作区根
    @return PlanSummary 列表
    """
    root = workspace.expanduser().resolve()
    settings = resolve_plan_settings(root)
    sessions_root = user_sessions_root(root)
    if not sessions_root.is_dir():
        return []

    summaries: list[PlanSummary] = []
    from llgraph.session.session_delete import is_plan_main_thread

    for child in sessions_root.iterdir():
        if not child.is_dir():
            continue
        thread_id = child.name
        if not is_plan_main_thread(thread_id):
            continue
        meta_path = child / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        plan_id = str(meta.get("plan_id") or "")
        phase = str(meta.get("phase") or "unknown")
        plan_state_path = child / "plan_state.json"
        plan_state: dict[str, Any] = {}
        if plan_state_path.is_file():
            try:
                ps = json.loads(plan_state_path.read_text(encoding="utf-8"))
                if isinstance(ps, dict):
                    plan_state = ps
                    plan_id = plan_id or str(ps.get("plan_id") or "")
                    phase = str(ps.get("phase") or phase)
            except (OSError, json.JSONDecodeError):
                pass

        title = get_session_title(root, thread_id) or resolve_session_display_title(root, thread_id)
        tasks_done, tasks_total = 0, 0
        plan: dict[str, Any] | None = None
        if plan_id:
            plan_file = load_plan(root, plan_id, plans_dir=settings.plans_dir)
            plan_inline = plan_state.get("plan") if isinstance(plan_state.get("plan"), dict) else None
            plan = pick_richer_plan(plan_file, plan_inline)
            if plan:
                title = str(plan.get("title") or title)
                tasks_done, tasks_total = count_task_progress(plan)
        if plan:
            phase = resolve_plan_phase(plan_state=plan_state, meta=meta, plan=plan)

        task_stubs: tuple[PlanTaskStub, ...] = ()
        if plan:
            raw_tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
            task_stubs = tuple(
                PlanTaskStub(
                    id=str(t.get("id") or ""),
                    title=str(t.get("title") or t.get("id") or ""),
                    status=str(t.get("status") or "pending"),
                )
                for t in raw_tasks
                if isinstance(t, dict) and t.get("id")
            )

        updated = _mtime_iso(plan_state_path) or _mtime_iso(meta_path) or _mtime_iso(child / "messages.jsonl")
        summaries.append(
            PlanSummary(
                thread_id=thread_id,
                plan_id=plan_id,
                title=title,
                phase=phase,
                tasks_done=tasks_done,
                tasks_total=tasks_total,
                updated_at=updated,
                task_stubs=task_stubs,
            )
        )

    summaries.sort(key=lambda s: s.updated_at or "", reverse=True)
    return summaries


def format_plans_list(workspace: Path, *, current_thread_id: str | None = None) -> str:
    """
    格式化 --list-plans 输出。

    @param workspace 工作区根
    @param current_thread_id 当前 thread（标记 ●）
    @return 多行文本
    """
    plans = discover_plan_sessions(workspace)
    lines = [
        "Plan 会话列表（plan-*）",
        "",
    ]
    if not plans:
        lines.append("  （无）启动: llgraph plan -C <工作区>")
        lines.append("恢复: llgraph plan -C <工作区> --thread-id plan-xxxxxxxx")
        return "\n".join(lines)

    for item in plans:
        mark = "● " if current_thread_id and item.thread_id == current_thread_id else "  "
        updated = item.updated_at[:19].replace("T", " ") if item.updated_at else "未知"
        progress = f"{item.tasks_done}/{item.tasks_total} tasks"
        title = item.title
        if len(title) > 28:
            title = title[:27] + "…"
        lines.append(
            f"{mark}{item.thread_id}  [{item.phase}]  {title}  · {progress}  ·  updated {updated}"
        )
    lines.append("")
    lines.append("恢复: llgraph plan -C <工作区> --thread-id <id>")
    lines.append("会话内: /plan switch plan-xxxxxxxx  |  /plan list")
    return "\n".join(lines)


def init_plan_session_meta(
    workspace: Path,
    thread_id: str,
    plan_id: str,
    *,
    phase: str,
    title: str = "",
) -> None:
    """
    初始化 Plan 会话 meta.json。

    @param workspace 工作区根
    @param thread_id plan thread
    @param plan_id Plan ID
    @param phase 当前 phase
    @param title 标题
    """
    from llgraph.session.session_meta import save_session_meta

    save_session_meta(
        workspace,
        thread_id,
        {
            "session_kind": "plan",
            "plan_id": plan_id,
            "phase": phase,
            "workspace": str(workspace.expanduser().resolve()),
            "title": title,
        },
    )


def plan_state_json_path(workspace: Path, thread_id: str) -> Path:
    """
    plan_state.json 路径。

    @param workspace 工作区根
    @param thread_id plan thread
    @return 绝对路径
    """
    return session_thread_dir(workspace, thread_id) / "plan_state.json"


def subgraph_messages_path(workspace: Path, thread_id: str, task_id: str) -> Path:
    """
    Worker 子图 trace 路径。

    @param workspace 工作区根
    @param thread_id plan thread
    @param task_id Task ID
    @return messages.jsonl 路径
    """
    return session_thread_dir(workspace, thread_id) / "subgraphs" / task_id / "messages.jsonl"
