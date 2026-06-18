"""Synthesize 节点：汇总各 Worker 结果生成最终报告。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_response import llm_response_text
from llgraph.plan.plan_registry import init_plan_session_meta
from llgraph.plan.plan_store import load_plan, save_plan
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase
from llgraph.plan.workflow_view import build_workflow_snapshot


def _build_synthesize_prompt(state: dict[str, Any]) -> str:
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    title = str(plan.get("title") or "未命名计划")
    goal = str(plan.get("goal") or state.get("opening_goal") or "")
    task_results = state.get("task_results") if isinstance(state.get("task_results"), dict) else {}
    lines = [f"计划：{title}", f"目标：{goal}", "", "各 Work 结果："]
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "")
        row = task_results.get(tid) if isinstance(task_results.get(tid), dict) else {}
        summary = str(row.get("summary") or task.get("description") or "（无摘要）")
        status = str(row.get("status") or task.get("status") or "?")
        lines.append(f"- [{tid}] {task.get('title') or tid} ({status}): {summary}")
    return "\n".join(lines)


def synthesize_node(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """
    Synthesize 节点：生成 final_report 并标记 completed。

    @param state PlanState
    @param ctx 运行时上下文
    @return state 更新
    """
    plan = dict(state.get("plan") or {})
    plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")
    if plan_id:
        disk = load_plan(ctx.workspace, plan_id, plans_dir=ctx.settings.plans_dir)
        if disk:
            plan = disk

    llm = create_gateway_llm(ctx.workspace)
    user_prompt = _build_synthesize_prompt({**state, "plan": plan})
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是 Plan 汇总助手。根据各 Work 执行结果，生成面向用户的最终报告。\n"
                    "使用 Markdown；结构清晰；不要编造未在结果中出现的内容。"
                )
            ),
            HumanMessage(content=user_prompt),
        ]
    )
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
        current_node="synthesize",
    )
    return {
        "plan": plan,
        "phase": PlanPhase.COMPLETED,
        "final_report": report,
        "workflow_snapshot": snapshot,
        "parallel_batch": [],
        "current_task_id": None,
    }
