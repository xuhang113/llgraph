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


_PLAN_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _find_json_object_end(text: str, start: int) -> int:
    """从 start 处的 `{` 找匹配的 `}`，返回索引；找不到返回 -1。"""
    if start < 0 or start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_plan_json_raw(text: str) -> str:
    """
    从 LLM 输出提取 plan JSON 字符串。

    @param text LLM 输出
    @return 候选 JSON 文本
    """
    match = _PLAN_JSON_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = text.find("{")
    while start >= 0:
        end = _find_json_object_end(text, start)
        if end > start:
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        start = text.find("{", start + 1)
    return stripped


def _apply_planner_parse_fallback(
    plan: dict[str, Any],
    *,
    parse_fallback: bool,
    revision: str,
    previous_plan: dict[str, Any] | None,
    goal: str,
    version: int,
) -> tuple[dict[str, Any], str | None]:
    """
    解析失败时的降级策略：修订场景保留上一版计划，首次规划才用 w1 默认任务。

    @return (plan, error_message)
    """
    if not parse_fallback:
        return plan, None
    prev_tasks = (
        previous_plan.get("tasks")
        if isinstance(previous_plan, dict) and isinstance(previous_plan.get("tasks"), list)
        else []
    )
    if revision.strip() and prev_tasks:
        kept = dict(previous_plan or {})
        kept["version"] = version
        kept["phase"] = PlanPhase.AWAITING_CONFIRM
        return kept, (
            "Planner 未能解析修订后的任务列表，已保留上一版计划。"
            "请用更具体的修订说明重试（需含完整 ```json plan 代码块）。"
        )
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
    return plan, "Planner 未能解析完整任务列表，已降级为单个默认 Work（w1）。可 /plan revise 重新规划。"


def _parse_plan_from_text(text: str, *, plan_id: str, goal: str) -> dict[str, Any]:
    """
    从 LLM 输出解析 plan JSON。

    @param text LLM 输出
    @param plan_id Plan ID
    @param goal 用户目标
    @return plan dict
    """
    raw = _extract_plan_json_raw(text)
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


def _planner_cancelled_update(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """Planner 已取消：保留现有 plan，不写 awaiting_confirm。"""
    from llgraph.plan.workflow_view import build_workflow_snapshot

    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.CANCELLED,
        plan=plan,
        current_node="planner",
    )
    return {
        "phase": PlanPhase.CANCELLED,
        "cancel_requested": True,
        "workflow_snapshot": snapshot,
        "plan": plan,
    }


def run_planner_node(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """
    Planner 节点：生成 plan.json。

    @param state PlanState
    @param ctx 运行时上下文
    @return state 更新
    """
    from llgraph.plan.execution_coordinator import is_cancel_requested
    from llgraph.plan.subgraphs.planner import run_planner_subagent

    from llgraph.plan.agent_context import build_planner_user_prompt

    if is_cancel_requested(ctx.thread_id) or state.get("cancel_requested"):
        from llgraph.plan.workflow_view import build_workflow_snapshot

        plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
        snapshot = build_workflow_snapshot(
            thread_id=ctx.thread_id,
            phase=PlanPhase.CANCELLED,
            plan=plan,
            current_node="planner",
        )
        return {
            "phase": PlanPhase.CANCELLED,
            "cancel_requested": True,
            "workflow_snapshot": snapshot,
        }

    plan_id = str(state.get("plan_id") or "")
    goal = str(state.get("opening_goal") or "").strip()
    revision = str(state.get("revision_note") or "").strip()
    agent_context = str(state.get("agent_context") or "").strip()
    version = int(state.get("plan_version") or 1)
    previous_plan = state.get("plan") if isinstance(state.get("plan"), dict) else None

    if revision:
        version += 1

    prompt = build_planner_user_prompt(
        opening_goal=goal,
        agent_context=agent_context,
        revision_note=revision,
        plan_version=version,
        current_plan=previous_plan,
    )

    text = run_planner_subagent(ctx, user_prompt=prompt, version=version, plan_state=state)

    if is_cancel_requested(ctx.thread_id) or state.get("cancel_requested"):
        return _planner_cancelled_update(state, ctx)

    plan = _parse_plan_from_text(text, plan_id=plan_id, goal=goal)
    parse_fallback = not plan.get("tasks")
    plan, planner_error = _apply_planner_parse_fallback(
        plan,
        parse_fallback=parse_fallback,
        revision=revision,
        previous_plan=previous_plan,
        goal=goal,
        version=version,
    )
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
        "error": planner_error,
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
    previous_plan = state.get("plan") if isinstance(state.get("plan"), dict) else None
    if revision:
        version += 1

    llm = create_gateway_llm(ctx.workspace)
    sys_msg = SystemMessage(content=planner_system_prompt(ctx))
    user_content = build_planner_user_prompt(
        opening_goal=goal,
        agent_context=agent_context,
        revision_note=revision,
        plan_version=version,
        current_plan=previous_plan,
    )
    response = llm.invoke([sys_msg, HumanMessage(content=user_content)])
    text = llm_response_text(response, fallback_thinking=False)
    plan = _parse_plan_from_text(text, plan_id=plan_id, goal=goal)
    parse_fallback = not plan.get("tasks")
    plan, planner_error = _apply_planner_parse_fallback(
        plan,
        parse_fallback=parse_fallback,
        revision=revision,
        previous_plan=previous_plan,
        goal=goal,
        version=version,
    )
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
        "error": planner_error,
    }
