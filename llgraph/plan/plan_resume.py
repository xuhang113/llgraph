"""Plan 跨重启续跑：pending_interrupt 与无 checkpoint 降级。"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

from llgraph.plan.state import PlanPhase


def serialize_interrupt(intr: Any) -> dict[str, Any]:
    """@param intr LangGraph interrupt payload @return 可 JSON 序列化 dict"""
    if isinstance(intr, dict):
        return dict(intr)
    return {"type": "unknown", "value": str(intr)}


def apply_pending_interrupt_to_state(state: dict[str, Any], intr: Any | None) -> dict[str, Any]:
    """
    将 interrupt 写入 plan_state.pending_interrupt（或清除）。

    @param state PlanState
    @param intr interrupt payload
    @return 更新后的 state 副本
    """
    merged = dict(state)
    if intr is not None:
        merged["pending_interrupt"] = serialize_interrupt(intr)
    else:
        merged.pop("pending_interrupt", None)
    return merged


def graph_has_interrupt(graph, config: dict[str, Any]) -> bool:
    """@return checkpoint 是否处于 interrupt"""
    try:
        snap = graph.get_state(config)
        return bool(snap and snap.interrupts)
    except Exception:
        return False


def resume_after_confirm_without_checkpoint(
    ctx,
    graph,
    state: dict[str, Any],
    decision: dict[str, Any],
    *,
    run_until_interrupt,
    resume_executing_plan,
    persist_state,
) -> tuple[dict[str, Any], Any | None]:
    """
    checkpoint 丢失时直接应用 confirm 决策并续跑。

    @return (新 state, interrupt)
    """
    from llgraph.plan.nodes.confirm import _apply_confirm_decision

    updates = _apply_confirm_decision(state, ctx, decision)
    merged = {**state, **updates}
    merged.pop("pending_interrupt", None)
    merged = persist_state(ctx, merged)
    phase = str(merged.get("phase") or "")
    if phase == PlanPhase.PLANNING:
        return run_until_interrupt(ctx, graph, merged)
    if phase == PlanPhase.CANCELLED:
        return merged, None
    return resume_executing_plan(ctx, graph, merged)


def resume_interrupt_or_fallback(
    ctx,
    graph,
    state: dict[str, Any],
    *,
    config: dict[str, Any],
    run_until_interrupt,
    resume_executing_plan,
    persist_state,
    resume_payload: Any | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """
    优先 Command(resume)；无 checkpoint 时按 pending_interrupt / phase 降级。

    @param resume_payload Command(resume=...) 或 None
    @return (新 state, interrupt)
    """
    if graph_has_interrupt(graph, config):
        payload = resume_payload if resume_payload is not None else Command(resume={})
        return run_until_interrupt(ctx, graph, state, input_payload=payload)

    pending = state.get("pending_interrupt")
    if isinstance(pending, dict) and pending.get("type") == "plan_confirm":
        return state, pending

    return resume_executing_plan(ctx, graph, state)
