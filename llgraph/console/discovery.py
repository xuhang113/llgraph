"""扫描 ~/.llgraph/context 下的工作区与会话。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgraph.plan.plan_registry import discover_plan_sessions
from llgraph.plan.plan_phase_resolve import resolve_plan_phase
from llgraph.plan.plan_store import load_plan, load_task_result, pick_richer_plan
from llgraph.plan.config import resolve_plan_settings
from llgraph.plan.state import PlanPhase
from llgraph.plan.workflow_view import GRAPH_DEFINITION, build_workflow_snapshot
from llgraph.core.llm_response import normalize_stored_llm_text
from llgraph.session.session_meta import load_session_meta
from llgraph.session.session_registry import discover_sessions
from llgraph.session.user_storage import (
    session_edits_path,
    session_messages_path,
    session_thread_dir,
)


def _mtime_iso(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


@dataclass(frozen=True)
class WorkspaceInfo:
    """工作区摘要。"""

    slug: str
    path: str
    session_count: int
    plan_count: int
    updated_at: str | None


def resolve_context_root() -> Path:
    """
    解析 llgraph 用户 context 根目录。

    @return ~/.llgraph/context 或 LLGRAPH_HOME/context
    """
    from llgraph.gateway.services.workspace_catalog import resolve_context_root as _root

    return _root()


def discover_workspaces() -> list[WorkspaceInfo]:
    """
    扫描 context 下所有工作区。

    @return WorkspaceInfo 列表，按最近活跃排序
    """
    from llgraph.gateway import get_control_gateway

    return [
        WorkspaceInfo(
            slug=r.slug,
            path=r.path,
            session_count=r.session_count,
            plan_count=r.plan_count,
            updated_at=r.updated_at,
        )
        for r in get_control_gateway().list_workspaces()
    ]


def workspace_path_from_slug(slug: str) -> Path:
    """
    由 slug 解析工作区根路径。

    @param slug context 目录名
    @return 工作区 Path
    @raises FileNotFoundError slug 不存在或无 workspace.json
    """
    from llgraph.gateway import get_control_gateway

    return Path(get_control_gateway().resolve_workspace_path(slug))


def read_json_file(path: Path) -> dict[str, Any] | None:
    """
    安全读取 JSON 文件。

    @param path 文件路径
    @return dict 或 None
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data
    return None


from llgraph.session.jsonl_read import iter_jsonl_text_lines, open_jsonl_for_read


def read_jsonl_lines(path: Path, *, offset: int = 0, limit: int = 100) -> tuple[list[dict[str, Any]], int]:
    """
    分页读取 jsonl。

    @param path jsonl 路径
    @param offset 跳过行数
    @param limit 最大返回行数
    @return (行列表, 总行数)
    """
    if not path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    total = 0
    try:
        with open_jsonl_for_read(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if total >= offset and len(rows) < limit:
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            rows.append(item)
                    except json.JSONDecodeError:
                        pass
                total += 1
    except OSError:
        return [], 0
    return rows, total


def _strip_user_injected_context(text: str) -> str:
    """剥离 user 消息中的 workspace-context 等注入块，保留用户原文。"""
    import re

    out = text
    out = re.sub(r"<workspace-context>[\s\S]*?</workspace-context>\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<session-manifest>[\s\S]*?</session-manifest>\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<custom-command[\s\S]*?</custom-command>\s*", "", out, flags=re.IGNORECASE)
    return out.strip()


def read_jsonl_lines_recent(path: Path, *, limit: int = 200) -> tuple[list[dict[str, Any]], int]:
    """
    读取 jsonl 末尾最近 limit 条（长 ReAct 会话最新消息在文件尾）。

    @param path jsonl 路径
    @param limit 最大返回行数
    @return (行列表, 总行数)
    """
    if not path.is_file():
        return [], 0
    try:
        lines = list(iter_jsonl_text_lines(path))
    except OSError:
        return [], 0
    total = len(lines)
    if total == 0:
        return [], 0
    tail_lines = lines[max(0, total - limit) :]
    rows: list[dict[str, Any]] = []
    for line in tail_lines:
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        except json.JSONDecodeError:
            continue
    return rows, total


def simplify_message(
    row: dict[str, Any],
    *,
    slug: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """
    将 LangChain messages_to_dict 行简化为前端友好结构。

    @param row jsonl 单行
    @param slug Web 工作区 slug（附件 URL）
    @param thread_id 会话 ID（附件 URL）
    @return 简化消息
    """
    from llgraph.context.message_normalize import _message_text
    from llgraph.core.user_message_content import extract_images_from_human_content
    from llgraph.session.session_image_store import attachment_api_path

    msg_type = str(row.get("type") or row.get("role") or "unknown")
    data = row.get("data") if isinstance(row.get("data"), dict) else row
    content = data.get("content") if isinstance(data, dict) else row.get("content")
    tool_calls = None
    if isinstance(data, dict):
        tool_calls = data.get("tool_calls") or data.get("additional_kwargs", {}).get("tool_calls")
    name = data.get("name") if isinstance(data, dict) else None
    display_text = _message_text(content).strip()

    def _attachment_url(image_id: str) -> str:
        if slug and thread_id:
            return attachment_api_path(slug, thread_id, image_id)
        return ""

    images = extract_images_from_human_content(
        content,
        attachment_url_for=_attachment_url if slug and thread_id else None,
    )
    kind: str | None = None
    if "human" in msg_type.lower():
        display_text = _strip_user_injected_context(display_text)
        from llgraph.core.agent_turn import THINK_CONTINUE_NUDGE

        if display_text.strip() == THINK_CONTINUE_NUDGE.strip():
            kind = "think_nudge"
    elif "ai" in msg_type.lower() or "assistant" in msg_type.lower():
        from llgraph.context.message_normalize import format_agent_chat_display_text

        display_text = format_agent_chat_display_text(display_text)
    has_tool_calls = bool(tool_calls)
    if not display_text and not has_tool_calls and isinstance(data, dict):
        extra = data.get("additional_kwargs") if isinstance(data.get("additional_kwargs"), dict) else {}
        llgraph_meta = extra.get("llgraph") if isinstance(extra.get("llgraph"), dict) else {}
        thinking = llgraph_meta.get("thinking_text")
        if isinstance(thinking, str) and thinking.strip():
            display_text = format_agent_chat_display_text(thinking.strip())
    out: dict[str, Any] = {
        "type": msg_type,
        "content": content,
        "display_text": display_text,
        "name": name,
        "tool_calls": tool_calls,
        "raw": row,
    }
    if images:
        out["images"] = images
    if kind:
        out["kind"] = kind
    return out


def list_edits(workspace: Path, thread_id: str) -> list[dict[str, Any]]:
    """
    读取 edits.jsonl。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 编辑记录列表
    """
    path = session_edits_path(workspace, thread_id)
    rows, _ = read_jsonl_lines(path, offset=0, limit=500)
    return rows


def _plan_task_ids(plan: dict[str, Any]) -> set[str]:
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    return {str(t.get("id")) for t in tasks if isinstance(t, dict) and t.get("id")}


def _snapshot_task_ids(snapshot: dict[str, Any]) -> set[str]:
    tasks = snapshot.get("tasks") if isinstance(snapshot.get("tasks"), list) else []
    return {str(t.get("id")) for t in tasks if isinstance(t, dict) and t.get("id")}


def _phase_rank(phase: str) -> int:
    from llgraph.plan.plan_phase_resolve import _phase_rank as _rank

    return _rank(phase)


def _task_statuses(plan: dict[str, Any]) -> list[str]:
    from llgraph.plan.plan_phase_resolve import task_statuses

    return task_statuses(plan)


def _resolve_plan_phase(
    *,
    plan_state: dict[str, Any],
    meta: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    return resolve_plan_phase(plan_state=plan_state, meta=meta, plan=plan)


def _snapshot_task_status_stale(snapshot: dict[str, Any], plan: dict[str, Any]) -> bool:
    """snapshot 内 task status 与 plan.json 不一致。"""
    snap_tasks = snapshot.get("tasks") if isinstance(snapshot.get("tasks"), list) else []
    snap_by_id = {
        str(t.get("id")): str(t.get("status") or "")
        for t in snap_tasks
        if isinstance(t, dict) and t.get("id")
    }
    for task in plan.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "")
        if not tid:
            continue
        if snap_by_id.get(tid) != str(task.get("status") or ""):
            return True
    return False


def _workflow_current_node(plan_state: dict[str, Any], *, phase: str) -> str | None:
    if phase == PlanPhase.COMPLETED:
        return None
    ws = plan_state.get("workflow_snapshot")
    if isinstance(ws, dict) and ws.get("current_node"):
        return str(ws.get("current_node"))
    return None


def _snapshot_node_status_stale(snapshot: dict[str, Any], phase: str) -> bool:
    """completed 但 synthesize 仍 running 的旧 snapshot（汇总完成后未刷新节点态）。"""
    if phase != PlanPhase.COMPLETED:
        return False
    for node in snapshot.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("id") == "synthesize" and str(node.get("status") or "") == "running":
            return True
    return False


def load_plan_detail(workspace: Path, thread_id: str) -> dict[str, Any]:
    """
    加载 Plan 详情（plan_state + plan.json + workflow）。

    @param workspace 工作区根
    @param thread_id plan-* thread
    @return 详情 dict
    """
    settings = resolve_plan_settings(workspace)
    thread_dir = session_thread_dir(workspace, thread_id)
    plan_state = read_json_file(thread_dir / "plan_state.json") or {}
    meta = load_session_meta(workspace, thread_id)
    plan_id = str(plan_state.get("plan_id") or meta.get("plan_id") or "")

    plan_inline = plan_state.get("plan") if isinstance(plan_state.get("plan"), dict) else None
    plan_file = load_plan(workspace, plan_id, plans_dir=settings.plans_dir) if plan_id else None
    plan = pick_richer_plan(plan_file, plan_inline)
    phase = _resolve_plan_phase(plan_state=plan_state, meta=meta, plan=plan)

    snapshot = plan_state.get("workflow_snapshot")
    stale_snapshot = (
        isinstance(snapshot, dict)
        and snapshot
        and _snapshot_task_ids(snapshot) != _plan_task_ids(plan)
    )
    phase_stale = isinstance(snapshot, dict) and snapshot and str(snapshot.get("phase") or "") != phase
    node_stale = isinstance(snapshot, dict) and snapshot and _snapshot_node_status_stale(snapshot, phase)
    task_stale = isinstance(snapshot, dict) and snapshot and _snapshot_task_status_stale(snapshot, plan)
    if (
        not isinstance(snapshot, dict)
        or not snapshot
        or stale_snapshot
        or phase_stale
        or node_stale
        or task_stale
    ):
        ws = plan_state.get("workflow_snapshot") if isinstance(plan_state.get("workflow_snapshot"), dict) else {}
        snapshot = build_workflow_snapshot(
            thread_id=thread_id,
            phase=phase,
            plan=plan,
            current_node=_workflow_current_node(plan_state, phase=phase),
            current_task_id=plan_state.get("current_task_id") or ws.get("current_task_id"),
        )
    elif "graph_definition" not in snapshot:
        snapshot = {**snapshot, "graph_definition": GRAPH_DEFINITION}

    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    raw_report = plan_state.get("final_report")
    final_report = None
    if phase == PlanPhase.COMPLETED and raw_report:
        final_report = normalize_stored_llm_text(raw_report)
    return {
        "thread_id": thread_id,
        "plan_id": plan_id,
        "title": str(plan.get("title") or meta.get("title") or thread_id),
        "goal": str(plan.get("goal") or ""),
        "phase": phase,
        "plan": plan,
        "plan_state": plan_state,
        "workflow_snapshot": snapshot,
        "final_report": final_report,
        "error": plan_state.get("error"),
        "tasks": tasks,
        "meta": meta,
        "updated_at": _mtime_iso(thread_dir / "plan_state.json"),
    }


def load_worker_detail(
    workspace: Path,
    thread_id: str,
    task_id: str,
) -> dict[str, Any]:
    """
    加载 Worker 任务详情。

    @param workspace 工作区根
    @param thread_id plan thread
    @param task_id 如 w1
    @return Worker 详情
    """
    settings = resolve_plan_settings(workspace)
    detail = load_plan_detail(workspace, thread_id)
    plan_id = detail["plan_id"]
    result = load_task_result(workspace, plan_id, task_id, plans_dir=settings.plans_dir) if plan_id else None

    subgraph_path = session_thread_dir(workspace, thread_id) / "subgraphs" / task_id / "messages.jsonl"
    messages, msg_total = read_jsonl_lines(subgraph_path, offset=0, limit=200)
    simplified = [simplify_message(m) for m in messages]

    worker_thread = f"{thread_id}:worker:{task_id}"
    edits = list_edits(workspace, worker_thread)

    task_info: dict[str, Any] = {}
    for t in detail.get("tasks") or []:
        if isinstance(t, dict) and str(t.get("id")) == task_id:
            task_info = t
            break

    return {
        "thread_id": thread_id,
        "task_id": task_id,
        "worker_thread_id": worker_thread,
        "task": task_info,
        "result": result,
        "messages": simplified,
        "message_total": msg_total,
        "edits": edits,
    }


def workspace_sessions_payload(workspace: Path) -> dict[str, Any]:
    """
    工作区下 Agent 会话列表。

    @param workspace 工作区根
    @return sessions 列表
    """
    sessions = discover_sessions(workspace)
    return {
        "sessions": [asdict(s) for s in sessions],
    }


def workspace_plans_payload(workspace: Path) -> dict[str, Any]:
    """
    工作区下 Plan 列表。

    @param workspace 工作区根
    @return plans 列表
    """
    plans = discover_plan_sessions(workspace)
    return {
        "plans": [asdict(p) for p in plans],
    }


def register_workspace_path(path: str) -> WorkspaceInfo:
    """
    注册工作区到 context（写入 workspace.json）。

    @param path 工作区绝对路径
    @return WorkspaceInfo
    """
    from llgraph.gateway import get_control_gateway

    record = get_control_gateway().register_workspace(path)
    return WorkspaceInfo(
        slug=record.slug,
        path=record.path,
        session_count=record.session_count,
        plan_count=record.plan_count,
        updated_at=record.updated_at,
    )


def dismiss_workspace_from_recent(slug: str) -> None:
    """
    从最近工作区列表隐藏（不删除会话数据）。

    @param slug context 目录名
    """
    from llgraph.gateway import get_control_gateway

    get_control_gateway().dismiss_workspace_from_recent(slug)


def touch_workspace_opened(slug: str) -> None:
    """
    记录工作区最近打开时间。

    @param slug context 目录名
    """
    from llgraph.gateway import get_control_gateway

    get_control_gateway().touch_workspace_opened(slug)


def build_session_tree(workspace: Path) -> dict[str, Any]:
    """
    构建工作区会话树（Agent + Plan + Worker 子节点）。

    @param workspace 工作区根
    @return 树形结构
    """
    agents = discover_sessions(workspace)
    plans = discover_plan_sessions(workspace)
    agent_nodes = [
        {
            "kind": "agent",
            "thread_id": s.thread_id,
            "title": s.title,
            "title_full": s.title_full,
            "updated_at": s.updated_at,
            "children": [],
        }
        for s in agents
    ]
    plan_nodes: list[dict[str, Any]] = []
    for p in plans:
        children = [
            {
                "kind": "worker",
                "thread_id": f"{p.thread_id}:worker:{t.id}",
                "task_id": t.id,
                "title": t.title,
                "status": t.status,
                "children": [],
            }
            for t in p.task_stubs
        ]
        plan_nodes.append(
            {
                "kind": "plan",
                "thread_id": p.thread_id,
                "plan_id": p.plan_id,
                "title": p.title,
                "title_full": p.title,
                "phase": p.phase,
                "updated_at": p.updated_at,
                "children": children,
            }
        )
    return {"agents": agent_nodes, "plans": plan_nodes}

