"""Plan 终止后只读问答（基于 final_report，不调度 Worker）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_response import llm_response_text, normalize_stored_llm_text
from llgraph.plan.plan_lifecycle import can_discuss
from llgraph.plan.plan_state_store import save_plan_state
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.task_results_hydrate import hydrate_task_results


def _discuss_messages(state: dict[str, Any]) -> list[dict[str, str]]:
    raw = state.get("discuss_messages")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip()
        content = str(row.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def _build_discuss_system(state: dict[str, Any], *, workspace: Path, plans_dir: str) -> str:
    hydrated = hydrate_task_results(workspace, state, plans_dir=plans_dir)
    plan = hydrated.get("plan") if isinstance(hydrated.get("plan"), dict) else {}
    title = str(plan.get("title") or "未命名计划")
    goal = str(plan.get("goal") or "")
    report = normalize_stored_llm_text(hydrated.get("final_report"))
    task_results = hydrated.get("task_results") if isinstance(hydrated.get("task_results"), dict) else {}
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

    question = user_message.strip()
    llm = create_gateway_llm(ctx.workspace)
    messages: list[Any] = [
        SystemMessage(
            content=_build_discuss_system(
                state,
                workspace=ctx.workspace,
                plans_dir=ctx.settings.plans_dir,
            )
        )
    ]
    for row in _discuss_messages(state):
        if row["role"] == "user":
            messages.append(HumanMessage(content=row["content"]))
        else:
            messages.append(AIMessage(content=row["content"]))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    text = llm_response_text(response).strip() or "（无回复）"

    history = _discuss_messages(state)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": text})

    merged = dict(state)
    user_messages = list(merged.get("user_messages") or [])
    user_messages.append(question)
    merged["user_messages"] = user_messages
    merged["discuss_messages"] = history
    save_plan_state(ctx.workspace, ctx.thread_id, merged)
    return text
