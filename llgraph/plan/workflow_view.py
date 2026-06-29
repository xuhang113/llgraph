"""Plan 工作流状态图：ASCII 渲染与 workflow_snapshot。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llgraph.plan.state import NodeStatus, PlanPhase, TaskStatus
from llgraph.plan.subgraph_registry import subgraph_definitions_for_workflow
from llgraph.plan.task_scheduling import all_work_task_ids

GRAPH_REVISION = "plan-graph-v2"

GRAPH_DEFINITION: dict[str, Any] = {
    "revision": GRAPH_REVISION,
    "nodes": [
        {
            "id": "planner",
            "label": "Planner",
            "subgraph": "planner",
            "subgraph_engine": "langgraph_react",
        },
        {"id": "confirm", "label": "Confirm", "interrupt": True},
        {"id": "supervisor", "label": "Supervisor"},
        {
            "id": "worker",
            "label": "Worker",
            "repeatable": True,
            "subgraph": "worker",
            "subgraph_engine": "langgraph_react",
        },
        {"id": "synthesize", "label": "Synthesize"},
    ],
    "subgraphs": subgraph_definitions_for_workflow(),
    "edges": [
        {"from": "START", "to": "planner"},
        {"from": "planner", "to": "confirm"},
        {"from": "confirm", "to": "supervisor", "when": "approve"},
        {"from": "confirm", "to": "planner", "when": "revise"},
        {"from": "confirm", "to": "END", "when": "cancel"},
        {"from": "supervisor", "to": "worker"},
        {"from": "worker", "to": "supervisor"},
        {"from": "supervisor", "to": "synthesize", "when": "all_done"},
        {"from": "synthesize", "to": "END"},
    ],
}


def _node_icon(status: str) -> str:
    mapping = {
        NodeStatus.DONE: "✓",
        NodeStatus.RUNNING: "●",
        NodeStatus.WAITING: "⏸",
        NodeStatus.FAILED: "✗",
        NodeStatus.PENDING: "○",
        TaskStatus.DONE: "✓",
        TaskStatus.RUNNING: "●",
        TaskStatus.FAILED: "✗",
        TaskStatus.PENDING: "○",
        TaskStatus.SKIPPED: "○",
    }
    return mapping.get(status, "○")


def _infer_node_statuses(
    phase: str,
    *,
    current_node: str | None,
    plan: dict[str, Any] | None,
) -> dict[str, str]:
    statuses = {
        "planner": NodeStatus.PENDING,
        "confirm": NodeStatus.PENDING,
        "supervisor": NodeStatus.PENDING,
        "synthesize": NodeStatus.PENDING,
    }
    if phase == PlanPhase.PLANNING:
        statuses["planner"] = NodeStatus.RUNNING
    elif phase == PlanPhase.AWAITING_CONFIRM:
        statuses["planner"] = NodeStatus.DONE
        statuses["confirm"] = NodeStatus.WAITING
    elif phase == PlanPhase.EXECUTING:
        statuses["planner"] = NodeStatus.DONE
        statuses["confirm"] = NodeStatus.DONE
        statuses["supervisor"] = NodeStatus.RUNNING if current_node != "worker" else NodeStatus.DONE
        if current_node == "worker":
            statuses["supervisor"] = NodeStatus.RUNNING
        if current_node == "synthesize":
            statuses["supervisor"] = NodeStatus.DONE
            statuses["synthesize"] = NodeStatus.RUNNING
    elif phase == PlanPhase.COMPLETED:
        for key in statuses:
            statuses[key] = NodeStatus.DONE
    elif phase == PlanPhase.CANCELLED:
        statuses["confirm"] = NodeStatus.FAILED
    if phase not in (PlanPhase.COMPLETED, PlanPhase.CANCELLED):
        if current_node == "planner":
            statuses["planner"] = NodeStatus.RUNNING
        if current_node == "confirm":
            statuses["confirm"] = NodeStatus.WAITING
        if current_node == "supervisor":
            statuses["supervisor"] = NodeStatus.RUNNING
        if current_node == "synthesize":
            statuses["supervisor"] = NodeStatus.DONE
            statuses["synthesize"] = NodeStatus.RUNNING
    return statuses


def build_workflow_snapshot(
    *,
    thread_id: str,
    phase: str,
    plan: dict[str, Any] | None,
    current_node: str | None = None,
    current_task_id: str | None = None,
) -> dict[str, Any]:
    """
    构建 workflow_snapshot JSON。

    @param thread_id Plan thread
    @param phase 当前 phase
    @param plan plan.json 内容
    @param current_node 当前 LangGraph node
    @param current_task_id 当前 task
    @return snapshot dict
    """
    node_statuses = _infer_node_statuses(phase, current_node=current_node, plan=plan)
    tasks_out: list[dict[str, Any]] = []
    tasks = plan.get("tasks") if isinstance(plan, dict) and isinstance(plan.get("tasks"), list) else []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "")
        status = str(task.get("status") or TaskStatus.PENDING)
        worker_status = status
        if tid == current_task_id and status == TaskStatus.RUNNING:
            worker_status = TaskStatus.RUNNING
        tasks_out.append(
            {
                "id": tid,
                "title": str(task.get("title") or tid),
                "status": status,
                "worker_node_status": worker_status,
                "depends_on": list(task.get("depends_on") or []),
            }
        )

    return {
        "graph_revision": GRAPH_REVISION,
        "thread_id": thread_id,
        "phase": phase,
        "current_node": current_node,
        "current_task_id": current_task_id,
        "nodes": [
            {"id": nid, "status": node_statuses.get(nid, NodeStatus.PENDING)}
            for nid in ("planner", "confirm", "supervisor", "synthesize")
        ],
        "tasks": tasks_out,
        "synthesize_depends_on": all_work_task_ids(plan) if isinstance(plan, dict) else [],
        "graph_definition": GRAPH_DEFINITION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def render_ascii_graph(
    snapshot: dict[str, Any],
    *,
    plan_title: str = "",
) -> str:
    """
    渲染终端 ASCII 状态图。

    @param snapshot workflow_snapshot
    @param plan_title Plan 标题
    @return 多行 ASCII
    """
    thread_id = str(snapshot.get("thread_id") or "")
    phase = str(snapshot.get("phase") or "")
    tasks = snapshot.get("tasks") if isinstance(snapshot.get("tasks"), list) else []
    nodes = {n["id"]: n.get("status", NodeStatus.PENDING) for n in snapshot.get("nodes", []) if isinstance(n, dict)}

    done_count = sum(1 for t in tasks if isinstance(t, dict) and t.get("status") == TaskStatus.DONE)
    total = len(tasks)
    header = f"Plan {thread_id}"
    if plan_title:
        header += f" · {plan_title}"
    header += f" · {phase} · {done_count}/{total} tasks"

    lines = [header, ""]
    lines.append(f"  [{_node_icon(nodes.get('planner', NodeStatus.PENDING))} Planner]")
    lines.append("       ↓")
    lines.append(f"  [{_node_icon(nodes.get('confirm', NodeStatus.PENDING))} Confirm]")
    lines.append("       ↓")
    lines.append(f"  [{_node_icon(nodes.get('supervisor', NodeStatus.PENDING))} Supervisor]")

    if tasks:
        for idx, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            prefix = "──→" if idx == 0 else "  ╲"
            if idx > 0:
                lines.append("  ╲")
            status = str(task.get("worker_node_status") or task.get("status") or TaskStatus.PENDING)
            title = str(task.get("title") or task.get("id") or "")
            if len(title) > 32:
                title = title[:31] + "…"
            suffix = ""
            if status == TaskStatus.RUNNING:
                suffix = "  running…"
            elif status == TaskStatus.PENDING:
                suffix = "  pending"
            elif status == TaskStatus.FAILED:
                suffix = "  failed"
            lines.append(
                f"  {prefix} [{_node_icon(status)} Worker {task.get('id')}: {title}]{suffix}"
            )
    else:
        lines.append("       ↓")
        lines.append("  [○ Worker×N]  (无 task)")

    lines.append("       ↓")
    synth_deps = snapshot.get("synthesize_depends_on")
    synth_suffix = ""
    if isinstance(synth_deps, list) and synth_deps:
        synth_suffix = f"  依赖: {', '.join(str(d) for d in synth_deps)}"
    lines.append(f"  [{_node_icon(nodes.get('synthesize', NodeStatus.PENDING))} Synthesize]{synth_suffix}")
    lines.append("       ↓")
    lines.append("  [○ END]")
    lines.append("")
    lines.append("图例: ✓ done  ● running/waiting  ○ pending  ✗ failed  ⏸ interrupt")
    return "\n".join(lines)
