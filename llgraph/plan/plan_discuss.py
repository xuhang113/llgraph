"""Plan 终止后只读问答（基于 final_report，不调度 Worker）。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_response import llm_response_text, normalize_stored_llm_text
from llgraph.plan.plan_lifecycle import can_discuss
from llgraph.plan.plan_state_store import load_plan_state, save_plan_state
from llgraph.plan.runtime import PlanRuntimeContext


def _build_discuss_system(state: dict[str, Any]) -> str:
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    title = str(plan.get("title") or "未命名计划")
    goal = str(plan.get("goal") or "")
    report = normalize_stored_llm_text(state.get("final_report"))
    task_results = state.get("task_results") if isinstance(state.get("task_results"), dict) else {}
    lines = [f"计划「{title}」", f"目标: {goal}", "", "## 最终报告", report]
    if task_results:
        lines.append("")
        lines.append("## 各 Work 摘要")
        for tid, row in task_results.items():
            if isinstance(row, dict):
                lines.append(f"- {tid}: {row.get('summary') or '（无）'}")
    return (
        "你是 Plan 会话助手。Plan 已全部执行并完成汇总，用户会追问结果。\n"
        "请仅根据下列报告与摘要回答，不要编造未出现的内容。\n"
        "不要建议重新执行 Work 或修改代码；若用户要改计划，提示使用 /plan revise。\n\n"
        + "\n".join(lines)
    )


def run_plan_discuss(
    ctx: PlanRuntimeContext,
    state: dict[str, Any],
    user_message: str,
) -> str:
    """
    Plan 终止后单轮问答。

    @param ctx 运行时上下文
    @param state PlanState
    @param user_message 用户问题
    @return 助手回复
    """
    if not can_discuss(state):
        phase = str(state.get("phase") or "")
        if phase != "completed":
            return "Plan 尚未完成汇总，请等待执行结束或使用 /plan continue。"
        return "尚无最终报告，请 /plan continue 完成汇总后再追问。"
    llm = create_gateway_llm(ctx.workspace)
    response = llm.invoke(
        [
            SystemMessage(content=_build_discuss_system(state)),
            HumanMessage(content=user_message.strip()),
        ]
    )
    text = llm_response_text(response).strip()
    user_messages = list(state.get("user_messages") or [])
    user_messages.append(user_message.strip())
    state = dict(state)
    state["user_messages"] = user_messages
    save_plan_state(ctx.workspace, ctx.thread_id, state)
    return text or "（无回复）"
