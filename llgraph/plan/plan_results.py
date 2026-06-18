"""Plan 执行结果展示（终端 /plan results，Web 可复用）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.plan.config import resolve_plan_settings
from llgraph.plan.plan_registry import PlanSummary, discover_plan_sessions, subgraph_messages_path
from llgraph.plan.plan_state_store import load_plan_state
from llgraph.plan.plan_store import load_plan, load_task_result
from llgraph.plan.subgraphs.planner import PLANNER_SUBGRAPH_SPEC
from llgraph.plan.subgraphs.worker import WORKER_SUBGRAPH_SPEC
from llgraph.session.session_edits import SessionEditTracker


def _resolve_plan_item(workspace: Path, query: str) -> PlanSummary | None:
    """
    解析 query 为 Plan 摘要。

    @param workspace 工作区根
    @param query plan-* / plan_id / 空=最近
    @return PlanSummary 或 None
    """
    import re

    plans = discover_plan_sessions(workspace)
    if not plans:
        return None
    q = query.strip()
    if not q:
        return plans[0]
    lower = q.lower()
    if re.match(r"^plan-[0-9a-f]{8}$", q, re.I):
        for item in plans:
            if item.thread_id.lower() == lower:
                return item
        return None
    if re.match(r"^[0-9a-f]{8}$", q, re.I):
        for item in plans:
            if item.plan_id.lower() == lower:
                return item
        return None
    for item in plans:
        if lower in item.title.lower() or lower in item.thread_id.lower():
            return item
    return None


def collect_subagent_file_changes(workspace: Path, sub_session_id: str) -> list[str]:
    """
    读取子 Agent 会话编辑账本中的变更路径。

    @param workspace 工作区根
    @param sub_session_id 子图 thread（如 plan-xxx:worker:w1）
    @return 相对路径列表
    """
    try:
        tracker = SessionEditTracker(workspace, session_id=sub_session_id)
        return tracker.unique_paths()
    except Exception:
        return []


def _format_files_section(paths: list[str], *, indent: str = "    ") -> list[str]:
    if not paths:
        return [f"{indent}（无文件变更）"]
    return [f"{indent}- {p}" for p in paths]


def format_plan_results(
    workspace: Path,
    *,
    query: str = "",
    task_id: str = "",
    current_thread_id: str | None = None,
) -> str:
    """
    格式化 Plan 执行结果：整体 + 各 subagent + 文件变更。

    返回 Markdown 原文；终端 emit_report 仅分色，Web 自行渲染。

    @param workspace 工作区根
    @param query plan-* / plan_id；空则当前或最近
    @param task_id 仅展示指定 Worker task
    @param current_thread_id Plan 模式内默认 thread
    @return 多行文本
    """
    root = workspace.expanduser().resolve()
    settings = resolve_plan_settings(root)

    item = _resolve_plan_item(root, query or (current_thread_id or ""))
    if item is None:
        return "未找到 Plan 会话。用 /plan list 查看，或 /plan results plan-xxxxxxxx。"

    state = load_plan_state(root, item.thread_id)
    plan: dict[str, Any] = {}
    if item.plan_id:
        plan = load_plan(root, item.plan_id, plans_dir=settings.plans_dir)

    lines = [
        f"=== Plan 结果 · {item.thread_id} ===",
        f"plan_id: {item.plan_id or '?'}  phase: {item.phase}  title: {item.title}",
    ]
    goal = str(state.get("opening_goal") or plan.get("goal") or "").strip()
    if goal:
        lines.append(f"goal: {goal[:400]}{'…' if len(goal) > 400 else ''}")

    report = str(state.get("final_report") or "").strip()
    if report:
        lines.append("")
        lines.append("【整体汇总 final_report】")
        lines.append(report if len(report) <= 3000 else report[:2999] + "…")

    version = int(state.get("plan_version") or plan.get("version") or 1)
    planner_thread = f"{item.thread_id}{PLANNER_SUBGRAPH_SPEC.thread_suffix.format(version=version)}"
    planner_files = collect_subagent_file_changes(root, planner_thread)
    lines.append("")
    lines.append(f"【Subagent · Planner v{version}】")
    lines.append(f"  thread: {planner_thread}")
    if plan:
        lines.append(f"  产出: {len(plan.get('tasks') or [])} 个 task · {plan.get('title') or ''}")
    lines.append("  文件变更:")
    lines.extend(_format_files_section(planner_files))

    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    task_results = state.get("task_results") if isinstance(state.get("task_results"), dict) else {}
    filter_tid = task_id.strip().lower()

    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "")
        if filter_tid and tid.lower() != filter_tid:
            continue
        worker_thread = f"{item.thread_id}{WORKER_SUBGRAPH_SPEC.thread_suffix.format(task_id=tid)}"
        result = load_task_result(root, item.plan_id, tid, plans_dir=settings.plans_dir) if item.plan_id else {}
        if not result and tid in task_results:
            result = task_results[tid]
        summary = str(result.get("summary") or "").strip()
        status = str(result.get("status") or task.get("status") or "?")
        artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), list) else []
        files_from_result = result.get("files_changed") if isinstance(result.get("files_changed"), list) else []
        files = [str(f) for f in files_from_result if str(f).strip()]
        if not files:
            files = collect_subagent_file_changes(root, worker_thread)
        msg_path = subgraph_messages_path(root, item.thread_id, tid)

        lines.append("")
        lines.append(f"【Subagent · Worker {tid}】 [{status}] {task.get('title') or tid}")
        lines.append(f"  thread: {worker_thread}")
        if summary:
            lines.append(f"  summary: {summary[:500]}{'…' if len(summary) > 500 else ''}")
        if artifacts:
            lines.append(f"  artifacts: {', '.join(str(a) for a in artifacts[:20])}")
        lines.append("  文件变更:")
        lines.extend(_format_files_section(files))
        if msg_path.is_file():
            lines.append(f"  trace: {msg_path}")

    if filter_tid and not any(str(t.get("id") or "").lower() == filter_tid for t in tasks if isinstance(t, dict)):
        lines.append("")
        lines.append(f"未找到 task: {task_id}")

    lines.append("")
    lines.append("产物目录: .llgraph/plans/" + (item.plan_id or "?"))
    return "\n".join(lines)
