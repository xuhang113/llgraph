"""Synthesize 节点：汇总各 Worker 结果生成最终报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_response import llm_response_text
from llgraph.plan.plan_registry import init_plan_session_meta
from llgraph.plan.plan_store import load_plan, save_plan
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase
from llgraph.plan.workflow_view import build_workflow_snapshot


def _build_synthesize_prompt(state: dict[str, Any], *, workspace: Path, plans_dir: str) -> str:
    from llgraph.plan.task_results_hydrate import hydrate_task_results

    hydrated = hydrate_task_results(workspace, state, plans_dir=plans_dir)
    plan = hydrated.get("plan") if isinstance(hydrated.get("plan"), dict) else {}
    title = str(plan.get("title") or "未命名计划")
    goal = str(plan.get("goal") or hydrated.get("opening_goal") or "")
    task_results = hydrated.get("task_results") if isinstance(hydrated.get("task_results"), dict) else {}
    lines = [f"计划：{title}", f"目标：{goal}", "", "各 Work 结果："]
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "")
        row = task_results.get(tid) if isinstance(task_results.get(tid), dict) else {}
        summary = str(row.get("summary") or task.get("description") or "（无摘要）")
        status = str(row.get("status") or task.get("status") or "?")
        files = row.get("files_changed") if isinstance(row.get("files_changed"), list) else []
        file_note = f" files_changed={files}" if files else " files_changed=[]"
        lines.append(f"- [{tid}] {task.get('title') or tid} ({status}): {summary}{file_note}")
    return "\n".join(lines)


def _synthesize_cancelled(state: dict[str, Any], ctx: PlanRuntimeContext, *, plan: dict[str, Any]) -> dict[str, Any]:
    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.CANCELLED,
        plan=plan,
        current_node=None,
    )
    return {
        "plan": plan,
        "phase": PlanPhase.CANCELLED,
        "cancel_requested": True,
        "workflow_snapshot": snapshot,
        "parallel_batch": [],
        "current_task_id": None,
    }


def synthesize_node(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """
    Synthesize 节点：生成 final_report 并标记 completed。

    @param state PlanState
    @param ctx 运行时上下文
    @return state 更新
    """
    from llgraph.plan.execution_coordinator import is_cancel_requested

    if is_cancel_requested(ctx.thread_id) or state.get("cancel_requested"):
        plan = dict(state.get("plan") or {})
        return _synthesize_cancelled(state, ctx, plan=plan)

    plan = dict(state.get("plan") or {})
    plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")
    if plan_id:
        disk = load_plan(ctx.workspace, plan_id, plans_dir=ctx.settings.plans_dir)
        if disk:
            plan = disk

    from llgraph.loaders.prompt_loader import compose_plan_synthesize_system

    llm = create_gateway_llm(ctx.workspace)
    user_prompt = _build_synthesize_prompt(
        {**state, "plan": plan},
        workspace=ctx.workspace,
        plans_dir=ctx.settings.plans_dir,
    )
    response = llm.invoke(
        [
            SystemMessage(content=compose_plan_synthesize_system()),
            HumanMessage(content=user_prompt),
        ]
    )

    if is_cancel_requested(ctx.thread_id) or state.get("cancel_requested"):
        return _synthesize_cancelled(state, ctx, plan=plan)

    report = llm_response_text(response, fallback_thinking=True).strip() or "（汇总为空）"

    plan["phase"] = PlanPhase.COMPLETED
    if plan_id:
        save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)
        init_plan_session_meta(
            ctx.workspace,
            ctx.thread_id,
            plan_id,
            phase=PlanPhase.COMPLETED,
            title=str(plan.get("title") or ""),
        )

    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.COMPLETED,
        plan=plan,
        current_node=None,
    )
    return {
        "plan": plan,
        "phase": PlanPhase.COMPLETED,
        "final_report": report,
        "workflow_snapshot": snapshot,
        "parallel_batch": [],
        "current_task_id": None,
    }
