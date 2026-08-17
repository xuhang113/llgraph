"""Agent 模式：spawn_subagent / explore 工具（与 Plan 共用 run_subagent）。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from llgraph.subagent.parent_slot import SubagentParentSlot
from llgraph.subagent.runner import run_subagent
from llgraph.subagent.runtime import SubagentRuntime


def create_subagent_tools(
    *,
    parent_runtime: SubagentRuntime | None = None,
    parent_slot: SubagentParentSlot | None = None,
    nestable: bool = False,
) -> list:
    """
    创建父 Agent 可用的 subagent 工具。

    @param parent_runtime 固定父运行时（测试 / 嵌套）
    @param parent_slot 可变槽（主 Agent 推荐：每轮 bind SSE）
    @param nestable 子 Agent 内是否再挂 spawn（默认否）
    @return Tool 列表
    """

    def _resolve_parent() -> SubagentRuntime | None:
        if parent_slot is not None and parent_slot.runtime is not None:
            return parent_slot.runtime
        return parent_runtime

    def spawn_subagent(
        prompt: str,
        kind: str = "explore",
        title: str = "",
    ) -> str:
        """
        在隔离子会话中执行调研/任务，仅把摘要返回本会话，避免主上下文被检索中间结果污染。

        何时使用（类似 Cursor Explore）：
        - 需要广搜代码库、多模块摸底、并行定位文件时
        - 预期会产生大量 grep/read 中间输出时
        何时不要用：已知文件的小范围修改、一两步就能定位的针点查询。

        @param prompt 给子 Agent 的完整任务说明（含范围、要回答的问题、返回格式要求）
        @param kind explore（只读广搜，默认）| general（只读通用）
        @param title 侧栏展示标题（可选）
        @return 子 Agent 摘要文本（含 sub_thread，可在 UI 点进查看完整轨迹）
        """
        import time

        from llgraph.display.trace_emit import (
            emit_explore_trace_step,
            update_explore_trace_step,
        )

        parent = _resolve_parent()
        if parent is None:
            return "错误：subagent 父运行时未绑定（请重试本轮对话）"
        kind_key = (kind or "explore").strip().lower()
        if kind_key not in ("explore", "general"):
            kind_key = "explore"
        explore_step_id = 0

        def on_started(sub_thread: str, label: str, started_kind: str) -> None:
            nonlocal explore_step_id
            step_title = (
                label if started_kind == "explore" else f"Subagent · {label}"
            )
            explore_step_id = emit_explore_trace_step(
                parent.trace_session,
                title=step_title,
                summary="执行中…",
                elapsed=0.0,
                sub_thread=sub_thread,
                body_lines=["status=running"],
            )

        t0 = time.perf_counter()
        result = run_subagent(
            parent,
            kind=kind_key,
            user_prompt=prompt,
            title=title.strip(),
            enable_spawn_tools=nestable,
            on_started=on_started,
        )
        elapsed = time.perf_counter() - t0
        label = (title.strip() or kind_key.capitalize() or "Explore")
        summary = (result.summary or "").strip().replace("\n", " ")
        if len(summary) > 120:
            summary = summary[:119] + "…"
        body_lines = [
            f"status={result.status}",
            *(result.summary or "").strip().splitlines()[:12],
        ]
        step_title = label if kind_key == "explore" else f"Subagent · {label}"
        if explore_step_id > 0:
            updated = update_explore_trace_step(
                parent.trace_session,
                explore_step_id,
                summary=summary or result.status,
                elapsed=elapsed,
                body_lines=body_lines,
            )
            if not updated:
                emit_explore_trace_step(
                    parent.trace_session,
                    title=step_title,
                    summary=summary or result.status,
                    elapsed=elapsed,
                    sub_thread=result.sub_thread,
                    body_lines=body_lines,
                )
        else:
            emit_explore_trace_step(
                parent.trace_session,
                title=step_title,
                summary=summary or result.status,
                elapsed=elapsed,
                sub_thread=result.sub_thread,
                body_lines=body_lines,
            )
        return result.as_tool_output()

    return [
        StructuredTool.from_function(
            func=spawn_subagent,
            name="spawn_subagent",
            description=(
                "Spawn an isolated read-only explore subagent for BROAD codebase research. "
                "USE THIS FIRST when the user asks to map/梳理 a business flow, call chain, "
                "module overview, or when you would otherwise run many search/read rounds. "
                "Child keeps intermediate tool output; you only get a FINAL SUMMARY + sub_thread. "
                "Args: prompt=full task brief — include scope, questions, and required summary shape "
                "(conclusion; evidence with path:line; related paths; uncovered gaps). "
                "Reject outline-only returns: if summary lacks evidence, search again before answering. "
                "kind=explore (default) or general; title=optional short label. "
                "Do NOT use for needle lookups (known file/symbol, 1–2 grep/read steps)."
            ),
        ),
    ]


def attach_subagent_runtime_to_session(session: Any) -> SubagentRuntime:
    """从 AgentSessionContext 得到可 spawn 的父运行时。"""
    from llgraph.subagent.runtime import runtime_from_agent_session

    return runtime_from_agent_session(session)
