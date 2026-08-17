"""Subagent 运行时（与 PlanRuntimeContext 解耦的共享宿主）。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llgraph.context.context_session import ContextSession
from llgraph.display.trace_display import TraceSession
from llgraph.session.session_edits import SessionEditTracker


@dataclass
class SubagentRuntime:
    """
    子 Agent 执行宿主（Plan / Agent 共用）。

    parent_thread_id 为父会话；fork 后仍保留该字段，子会话用 sub_thread 参数区分。
    """

    workspace: Path
    parent_thread_id: str
    trace_session: TraceSession
    context_session: ContextSession
    allow_write: bool = False
    mcp_tools: list = field(default_factory=list)
    sandbox_policy: Any = None
    web_search_enabled: bool = False
    write_failure_tracker: Any = None
    on_file_changed: Callable[[str], None] | None = None
    sse_emit: Callable[[dict[str, Any]], None] | None = None
    sse_loop: Any = None
    max_turns: int | None = None
    cancel_check: Callable[[], bool] | None = None

    @property
    def thread_id(self) -> str:
        """兼容 Plan 侧 ctx.thread_id（父会话 id）。"""
        return self.parent_thread_id


def fork_subagent_runtime(
    parent: SubagentRuntime,
    *,
    sub_thread: str,
    subgraph_kind: str,
    task_id: str | None = None,
    allow_write: bool | None = None,
    max_turns: int | None = None,
) -> SubagentRuntime:
    """
    为子 Agent 创建独立运行时（独立 context / trace / SSE channel）。

    @param parent 父运行时
    @param sub_thread 如 cli-xxx:explore:a1b2 或 plan-xxx:worker:t1
    @param subgraph_kind explore | worker | planner | general
    @param task_id 可选任务 id（Worker / 展示标签）
    @param allow_write 覆盖父写权限；None 继承
    @param max_turns 覆盖步数上限
    @return 子运行时
    """
    from llgraph.console.runtime.event_hub import HUB
    from llgraph.config.edit_settings import resolve_edit_settings
    from llgraph.core.write_failure_tracker import WriteFailureTracker

    child_cs = parent.context_session.fork()
    sub = sub_thread.strip()
    kind = subgraph_kind.strip()
    tid = (task_id or "").strip()
    child_allow = parent.allow_write if allow_write is None else bool(allow_write)

    parent_emit = parent.sse_emit
    parent_loop = parent.sse_loop

    def session_emit(event: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            **event,
            "thread_id": sub,
            "sub_thread": sub,
            "subgraph_kind": kind,
        }
        if tid:
            payload["task_id"] = tid
        if parent_loop is not None:
            HUB.publish_sync(f"session:{sub}", payload, parent_loop)
        else:
            HUB.publish(f"session:{sub}", payload)
        if parent_emit is not None:
            parent_emit(payload)

    wft = None
    if child_allow and kind in ("worker", "general"):
        edit_settings = resolve_edit_settings(parent.workspace)
        wft = WriteFailureTracker(
            child_cs,
            failures_before_hint=edit_settings.write_failures_before_hint,
            chunk_max_chars=edit_settings.write_chunk_max_chars,
        )

    return SubagentRuntime(
        workspace=parent.workspace,
        parent_thread_id=parent.parent_thread_id,
        trace_session=TraceSession(mode=parent.trace_session.mode),
        context_session=child_cs,
        allow_write=child_allow,
        mcp_tools=list(parent.mcp_tools),
        sandbox_policy=parent.sandbox_policy,
        web_search_enabled=parent.web_search_enabled,
        write_failure_tracker=wft,
        on_file_changed=parent.on_file_changed,
        sse_emit=session_emit,
        sse_loop=parent_loop,
        max_turns=max_turns if max_turns is not None else parent.max_turns,
        cancel_check=parent.cancel_check,
    )


def isolated_subagent_trace(
    runtime: SubagentRuntime,
    *,
    sub_thread: str,
    subgraph_kind: str,
    task_id: str | None = None,
) -> TraceSession:
    """
    为子图 invoke 创建独立 TraceSession（Web 时带 sub_thread 标签）。

    @param runtime 子运行时（通常已 fork）
    @param sub_thread 子图 thread_id
    @param subgraph_kind 类型标签
    @param task_id 可选
    @return TraceSession
    """
    if runtime.sse_emit is None:
        return runtime.trace_session

    from llgraph.console.runtime.sse_sink import PersistingSseTraceSink, SseTraceSink

    parent_mode = runtime.trace_session.mode
    kind = subgraph_kind.strip()
    tid = (task_id or "").strip()
    sub = sub_thread.strip()

    def _emit(event: dict[str, Any]) -> None:
        payload = {
            **event,
            "sub_thread": sub,
            "subgraph_kind": kind,
        }
        if tid:
            payload["task_id"] = tid
        runtime.sse_emit(payload)

    trace = TraceSession(mode=parent_mode)
    inner = SseTraceSink(_emit)
    trace.trace_sink = PersistingSseTraceSink(
        inner,
        workspace=runtime.workspace,
        thread_id=sub,
    )
    return trace


def subagent_edit_tracker(
    runtime: SubagentRuntime,
    sub_session_id: str,
    *,
    allow_write: bool,
) -> SessionEditTracker | None:
    """可写子 Agent 的会话级编辑账本。"""
    if not allow_write:
        return None
    return SessionEditTracker(runtime.workspace, session_id=sub_session_id)


def runtime_from_plan_context(ctx: Any) -> SubagentRuntime:
    """
    从 PlanRuntimeContext 构造共享运行时视图（不 fork）。

    @param ctx PlanRuntimeContext
    @return SubagentRuntime
    """
    settings = getattr(ctx, "settings", None)
    max_turns = None
    if settings is not None:
        max_turns = getattr(settings, "worker_max_turns", None)
    return SubagentRuntime(
        workspace=ctx.workspace,
        parent_thread_id=ctx.thread_id,
        trace_session=ctx.trace_session,
        context_session=ctx.context_session,
        allow_write=bool(getattr(ctx, "allow_write_cli", False)),
        mcp_tools=list(getattr(ctx, "mcp_tools", None) or []),
        sandbox_policy=getattr(ctx, "sandbox_policy", None),
        web_search_enabled=bool(getattr(ctx, "web_search_enabled", False)),
        write_failure_tracker=getattr(ctx, "write_failure_tracker", None),
        on_file_changed=getattr(ctx, "on_file_changed", None),
        sse_emit=getattr(ctx, "sse_emit", None),
        sse_loop=getattr(ctx, "sse_loop", None),
        max_turns=max_turns,
    )


def runtime_from_agent_session(session: Any) -> SubagentRuntime:
    """
    从 AgentSessionContext 构造共享运行时视图。

    @param session AgentSessionContext
    @return SubagentRuntime
    """
    from llgraph.core.react_limits import resolve_agent_max_turns

    return SubagentRuntime(
        workspace=session.workspace,
        parent_thread_id=session.thread_id,
        trace_session=session.trace_session,
        context_session=session.context_session,
        allow_write=bool(getattr(session, "allow_write", False)),
        mcp_tools=list(getattr(session, "mcp_tools", None) or []),
        sandbox_policy=getattr(session, "sandbox_policy", None),
        web_search_enabled=bool(getattr(session, "web_search_enabled", False)),
        write_failure_tracker=getattr(session, "write_failure_tracker", None),
        on_file_changed=getattr(session, "on_file_changed", None),
        sse_emit=getattr(session, "sse_emit", None),
        sse_loop=getattr(session, "sse_loop", None),
        max_turns=resolve_agent_max_turns(session.workspace),
    )
