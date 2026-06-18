"""Planner 节点：只读调研 + 生成 plan.json。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_response import llm_response_text
from llgraph.plan.plan_store import empty_plan, save_plan
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase, TaskStatus


_PLAN_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_plan_from_text(text: str, *, plan_id: str, goal: str) -> dict[str, Any]:
    """
    从 LLM 输出解析 plan JSON。

    @param text LLM 输出
    @param plan_id Plan ID
    @param goal 用户目标
    @return plan dict
    """
    match = _PLAN_JSON_BLOCK.search(text)
    raw = match.group(1) if match else text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    plan = empty_plan(plan_id=plan_id, title=str(data.get("title") or "未命名计划"), goal=goal)
    if data.get("title"):
        plan["title"] = str(data["title"])
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        normalized: list[dict[str, Any]] = []
        for idx, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            tid = str(task.get("id") or f"w{idx + 1}")
            normalized.append(
                {
                    "id": tid,
                    "title": str(task.get("title") or f"Task {idx + 1}"),
                    "description": str(task.get("description") or ""),
                    "scope": task.get("scope") if isinstance(task.get("scope"), dict) else {"path_globs": ["."]},
                    "depends_on": list(task.get("depends_on") or []),
                    "status": TaskStatus.PENDING,
                    "retry_count": 0,
                    "max_retries": int(task.get("max_retries") or 2),
                    "readonly": bool(task.get("readonly", False)),
                }
            )
        plan["tasks"] = normalized
    plan["phase"] = PlanPhase.AWAITING_CONFIRM
    return plan


def run_planner_node(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """
    Planner 节点：生成 plan.json。

    @param state PlanState
    @param ctx 运行时上下文
    @return state 更新
    """
    from llgraph.plan.subgraphs.planner import run_planner_subagent

    from llgraph.plan.agent_context import build_planner_user_prompt

    plan_id = str(state.get("plan_id") or "")
    goal = str(state.get("opening_goal") or "").strip()
    revision = str(state.get("revision_note") or "").strip()
    agent_context = str(state.get("agent_context") or "").strip()
    version = int(state.get("plan_version") or 1)

    if revision:
        version += 1

    prompt = build_planner_user_prompt(
        opening_goal=goal,
        agent_context=agent_context,
        revision_note=revision,
        plan_version=version,
    )

    text = run_planner_subagent(ctx, user_prompt=prompt, version=version)

    plan = _parse_plan_from_text(text, plan_id=plan_id, goal=goal)
    parse_fallback = not plan.get("tasks")
    if parse_fallback:
        plan["tasks"] = [
            {
                "id": "w1",
                "title": "调研与梳理",
                "description": goal,
                "scope": {"path_globs": ["."]},
                "depends_on": [],
                "status": TaskStatus.PENDING,
                "retry_count": 0,
                "max_retries": 2,
                "readonly": True,
            }
        ]
    plan["version"] = version
    save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)

    from llgraph.plan.workflow_view import build_workflow_snapshot

    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.AWAITING_CONFIRM,
        plan=plan,
        current_node="planner",
    )
    return {
        "plan": plan,
        "phase": PlanPhase.AWAITING_CONFIRM,
        "plan_version": version,
        "revision_note": None,
        "workflow_snapshot": snapshot,
        "error": (
            "Planner 未能解析完整任务列表，已降级为单个默认 Work（w1）。可 /plan revise 重新规划。"
            if parse_fallback
            else None
        ),
    }


def run_planner_node_fast(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """
    轻量 Planner：无 ReAct，直接 LLM 生成 plan（用于测试或快速路径）。

    @param state PlanState
    @param ctx 运行时上下文
    @return state 更新
    """
    from llgraph.plan.agent_context import build_planner_user_prompt
    from llgraph.plan.subgraphs.planner import planner_system_prompt

    plan_id = str(state.get("plan_id") or "")
    goal = str(state.get("opening_goal") or "").strip()
    revision = str(state.get("revision_note") or "").strip()
    agent_context = str(state.get("agent_context") or "").strip()
    version = int(state.get("plan_version") or 1)
    if revision:
        version += 1

    llm = create_gateway_llm(ctx.workspace)
    sys_msg = SystemMessage(content=planner_system_prompt(ctx))
    user_content = build_planner_user_prompt(
        opening_goal=goal,
        agent_context=agent_context,
        revision_note=revision,
        plan_version=version,
    )
    response = llm.invoke([sys_msg, HumanMessage(content=user_content)])
    text = llm_response_text(response, fallback_thinking=True)
    plan = _parse_plan_from_text(text, plan_id=plan_id, goal=goal)
    parse_fallback = not plan.get("tasks")
    if parse_fallback:
        plan["tasks"] = [
            {
                "id": "w1",
                "title": "调研与梳理",
                "description": goal,
                "scope": {"path_globs": ["."]},
                "depends_on": [],
                "status": TaskStatus.PENDING,
                "retry_count": 0,
                "max_retries": 2,
                "readonly": True,
            }
        ]
    plan["version"] = version
    save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)

    from llgraph.plan.workflow_view import build_workflow_snapshot

    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.AWAITING_CONFIRM,
        plan=plan,
        current_node="planner",
    )
    return {
        "plan": plan,
        "phase": PlanPhase.AWAITING_CONFIRM,
        "plan_version": version,
        "revision_note": None,
        "workflow_snapshot": snapshot,
        "error": (
            "Planner 未能解析完整任务列表，已降级为单个默认 Work（w1）。可 /plan revise 重新规划。"
            if parse_fallback
            else None
        ),
    }
