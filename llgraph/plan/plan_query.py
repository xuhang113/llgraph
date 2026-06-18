"""Plan 会话查询（Agent 只读工具底层）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from llgraph.plan.config import resolve_plan_settings
from llgraph.plan.plan_registry import (
    PlanSummary,
    discover_plan_sessions,
    plan_state_json_path,
    subgraph_messages_path,
)
from llgraph.plan.plan_state_store import load_plan_state
from llgraph.plan.plan_store import load_plan, load_task_result, plan_json_path
from llgraph.session.user_storage import session_thread_dir

_PLAN_THREAD_RE = re.compile(r"^plan-[0-9a-f]{8}$", re.I)
_PLAN_ID_RE = re.compile(r"^[0-9a-f]{8}$", re.I)

_FINAL_REPORT_MAX = 1200
_TASK_SUMMARY_MAX = 200


def _match_plans(plans: list[PlanSummary], query: str) -> list[PlanSummary]:
    """
    按 query 筛选 Plan 会话。

    @param plans 全部 Plan 摘要
    @param query 空=全部；plan thread / plan_id / 标题关键词
    @return 匹配列表
    """
    q = query.strip()
    if not q:
        return plans
    lower = q.lower()
    if _PLAN_THREAD_RE.match(q):
        return [p for p in plans if p.thread_id.lower() == lower]
    if _PLAN_ID_RE.match(q):
        return [p for p in plans if p.plan_id.lower() == lower]
    return [
        p
        for p in plans
        if lower in p.title.lower()
        or lower in p.thread_id.lower()
        or lower in p.plan_id.lower()
    ]


def _artifact_paths(workspace: Path, item: PlanSummary, settings_plans_dir: str) -> list[str]:
    """
    生成 read_file 可用的产物路径提示。

    @param workspace 工作区根
    @param item Plan 摘要
    @param settings_plans_dir plans 相对目录
    @return 路径字符串列表
    """
    paths: list[str] = []
    if item.plan_id:
        rel = plan_json_path(workspace, item.plan_id, plans_dir=settings_plans_dir)
        try:
            paths.append(str(rel.relative_to(workspace.resolve())))
        except ValueError:
            paths.append(str(rel))
    session_dir = session_thread_dir(workspace, item.thread_id)
    ps = plan_state_json_path(workspace, item.thread_id)
    paths.append(str(ps))
    if session_dir.is_dir():
        paths.append(str(session_dir / "subgraphs"))
    return paths


def _format_plan_line(item: PlanSummary) -> str:
    """
    格式化列表行。

    @param item Plan 摘要
    @return 单行文本
    """
    updated = item.updated_at[:19].replace("T", " ") if item.updated_at else "未知"
    progress = f"{item.tasks_done}/{item.tasks_total}"
    title = item.title if len(item.title) <= 40 else item.title[:39] + "…"
    return (
        f"- {item.thread_id}  plan_id={item.plan_id or '?'}  [{item.phase}]  "
        f"{title}  · tasks {progress}  · {updated}"
    )


def _format_task_results(
    workspace: Path,
    plan_id: str,
    plan: dict[str, Any],
    *,
    plans_dir: str,
) -> list[str]:
    """
    格式化 task 与 result 摘要行。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param plan plan.json
    @param plans_dir 相对目录
    @return 文本行
    """
    lines: list[str] = []
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "?")
        status = str(task.get("status") or "?")
        title = str(task.get("title") or tid)
        result = load_task_result(workspace, plan_id, tid, plans_dir=plans_dir)
        summary = str(result.get("summary") or "").strip()
        if len(summary) > _TASK_SUMMARY_MAX:
            summary = summary[: _TASK_SUMMARY_MAX - 1] + "…"
        line = f"  · {tid} [{status}] {title}"
        if summary:
            line += f": {summary}"
        lines.append(line)
    return lines


def format_plan_detail(workspace: Path, item: PlanSummary) -> str:
    """
    单个 Plan 会话详情（紧凑）。

    @param workspace 工作区根
    @param item Plan 摘要
    @return 多行文本
    """
    settings = resolve_plan_settings(workspace)
    lines = [
        f"Plan 会话: {item.thread_id}",
        f"plan_id: {item.plan_id or '（未知）'}  phase: {item.phase}  title: {item.title}",
    ]
    state = load_plan_state(workspace, item.thread_id)
    if state.get("opening_goal"):
        goal = str(state.get("opening_goal") or "")
        if len(goal) > 300:
            goal = goal[:299] + "…"
        lines.append(f"goal: {goal}")

    plan: dict[str, Any] = {}
    if item.plan_id:
        plan = load_plan(workspace, item.plan_id, plans_dir=settings.plans_dir)
    if plan:
        lines.append("tasks:")
        lines.extend(
            _format_task_results(
                workspace,
                item.plan_id,
                plan,
                plans_dir=settings.plans_dir,
            )
        )
    elif item.tasks_total:
        lines.append(f"tasks: {item.tasks_done}/{item.tasks_total}（plan.json 未找到）")

    report = str(state.get("final_report") or "").strip()
    if report:
        if len(report) > _FINAL_REPORT_MAX:
            report = report[: _FINAL_REPORT_MAX - 1] + "…"
        lines.append("final_report:")
        lines.append(report)

    paths = _artifact_paths(workspace, item, settings.plans_dir)
    lines.append("paths（read_file / 目录）:")
    for p in paths:
        lines.append(f"  {p}")
    lines.append("进入 Plan 模式: /plan switch " + item.thread_id)
    return "\n".join(lines)


def query_plans_text(workspace: Path, *, query: str = "", limit: int = 10) -> str:
    """
    查询 Plan 会话：列表或详情。

    @param workspace 工作区根
    @param query 空=最近列表；plan-* / plan_id=详情；否则标题关键词
    @param limit 列表最大条数
    @return 格式化文本
    """
    root = workspace.expanduser().resolve()
    try:
        cap = max(1, min(20, int(limit)))
    except (TypeError, ValueError):
        cap = 10

    all_plans = discover_plan_sessions(root)
    if not all_plans:
        return (
            "当前工作区无 Plan 会话（plan-*）。\n"
            "说明: thread_id 形如 plan-xxxxxxxx；产物在 .llgraph/plans/{plan_id}/。"
        )

    q = query.strip()
    if not q:
        lines = [f"最近 Plan 会话（最多 {cap} 条）:", ""]
        for item in all_plans[:cap]:
            lines.append(_format_plan_line(item))
        if len(all_plans) > cap:
            lines.append(f"… 共 {len(all_plans)} 条，用 query_plans(query='plan-xxxxxxxx') 查详情。")
        lines.append("")
        lines.append("详情: query_plans(query='plan-*' 或 8 位 plan_id)")
        return "\n".join(lines)

    matched = _match_plans(all_plans, q)
    if not matched:
        return f"未找到匹配 Plan: {q!r}\n提示: 空 query 列最近会话；或用 plan-xxxxxxxx / 8 位 plan_id。"

    explicit = _PLAN_THREAD_RE.match(q) or _PLAN_ID_RE.match(q)
    if explicit or len(matched) == 1:
        return format_plan_detail(root, matched[0])

    lines = [
        f"匹配 {len(matched)} 个 Plan（关键词 {q!r}），用 plan-* 或 plan_id 查详情:",
        "",
    ]
    for item in matched[:cap]:
        lines.append(_format_plan_line(item))
    if len(matched) > cap:
        lines.append(f"… 还有 {len(matched) - cap} 条，请缩小关键词或指定 plan-*。")
    return "\n".join(lines)
