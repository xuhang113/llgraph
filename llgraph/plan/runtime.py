"""Plan 运行时上下文（节点共享）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from llgraph.context.context_session import ContextSession
from llgraph.plan.config import PlanSettings
from llgraph.display.trace_display import TraceSession
from llgraph.session.session_edits import SessionEditTracker


@dataclass
class PlanRuntimeContext:
    """PlanGraph 编译与执行共享上下文。"""

    workspace: Path
    thread_id: str
    settings: PlanSettings
    trace_session: TraceSession
    context_session: ContextSession
    allow_write_cli: bool = False
    mcp_tools: list = field(default_factory=list)
    sandbox_policy: Any = None
    web_search_enabled: bool = False
    write_failure_tracker: Any = None
    on_file_changed: Callable[[str], None] | None = None
    # Web SSE：子 Agent trace 独立 emit，Plan 父会话仅收里程碑
    sse_emit: Callable[[dict[str, Any]], None] | None = None
    sse_loop: Any = None

    def fork_subagent_runtime(
        self,
        *,
        sub_thread: str,
        subgraph_kind: str,
        task_id: str | None = None,
    ) -> PlanRuntimeContext:
        """
        为 Planner / Worker 创建独立运行时（独立 context_session / trace / SSE channel）。

        @param sub_thread 如 plan-xxx:planner:v1 或 plan-xxx:worker:t1
        @param subgraph_kind planner | worker
        @param task_id Worker task id（Planner 可省略）
        @return 仅供该子 Agent 使用的上下文
        """
        from llgraph.console.runtime.event_hub import HUB
        from llgraph.core.write_failure_tracker import WriteFailureTracker
        from llgraph.config.edit_settings import resolve_edit_settings

        child_cs = self.context_session.fork()
        sub = sub_thread.strip()
        kind = subgraph_kind.strip()
        tid = (task_id or "").strip()

        parent_emit = self.sse_emit
        parent_loop = self.sse_loop

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
        if self.allow_write_cli and kind == "worker":
            edit_settings = resolve_edit_settings(self.workspace)
            wft = WriteFailureTracker(
                child_cs,
                failures_before_hint=edit_settings.write_failures_before_hint,
                chunk_max_chars=edit_settings.write_chunk_max_chars,
            )

        trace = TraceSession(mode=self.trace_session.mode)
        return PlanRuntimeContext(
            workspace=self.workspace,
            thread_id=self.thread_id,
            settings=self.settings,
            trace_session=trace,
            context_session=child_cs,
            allow_write_cli=self.allow_write_cli,
            mcp_tools=list(self.mcp_tools),
            sandbox_policy=self.sandbox_policy,
            web_search_enabled=self.web_search_enabled,
            write_failure_tracker=wft,
            on_file_changed=self.on_file_changed,
            sse_emit=session_emit,
            sse_loop=parent_loop,
        )

    def fork_worker_runtime(
        self,
        *,
        task_id: str,
        sub_thread: str,
    ) -> PlanRuntimeContext:
        """@param task_id 如 t3 @param sub_thread plan-xxx:worker:t3"""
        return self.fork_subagent_runtime(
            sub_thread=sub_thread,
            subgraph_kind="worker",
            task_id=task_id,
        )

    def isolated_subagent_trace(
        self,
        *,
        sub_thread: str,
        subgraph_kind: str,
        task_id: str | None = None,
    ) -> TraceSession:
        """
        为子图 invoke 创建独立 TraceSession（Web 时带 sub_thread/task_id 标签）。

        @param sub_thread 子图 checkpoint thread_id
        @param subgraph_kind planner | worker
        @param task_id Worker task id（可选）
        @return 独立 trace 会话
        """
        if self.sse_emit is None:
            return self.trace_session

        from llgraph.console.runtime.sse_sink import PersistingSseTraceSink, SseTraceSink
        from llgraph.display.trace_display import TraceSession

        parent_mode = self.trace_session.mode
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
            self.sse_emit(payload)

        trace = TraceSession(mode=parent_mode)
        inner = SseTraceSink(_emit)
        trace.trace_sink = PersistingSseTraceSink(
            inner,
            workspace=self.workspace,
            thread_id=sub,
        )
        return trace

    def worker_allow_write(self, state: dict[str, Any]) -> bool:
        """
        Worker 是否允许写文件。

        @param state PlanState
        @return 是否可写
        """
        if state.get("allow_worker_write"):
            return True
        plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
        execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
        if execution.get("allow_worker_write"):
            return True
        return self.settings.default_allow_write and self.allow_write_cli

    def subagent_manifest_allow_write(
        self,
        state: dict[str, Any] | None,
        *,
        allow_write: bool | None = None,
    ) -> bool:
        """
        子 Agent manifest / workspace-context 与 Worker 工具一致的写权限。

        @param state 父 PlanState（Worker 调用时传入）
        @param allow_write 显式覆盖（Planner 传 False）
        @return 是否可写
        """
        if allow_write is not None:
            return allow_write
        if state is not None:
            return self.worker_allow_write(state)
        return self.allow_write_cli

    def subagent_edit_tracker(self, sub_session_id: str, *, allow_write: bool) -> SessionEditTracker | None:
        """
        子 Agent 会话级编辑账本（Worker 写文件追踪）。

        @param sub_session_id 子图 thread_id
        @param allow_write 是否可写
        @return SessionEditTracker 或 None
        """
        if not allow_write:
            return None
        return SessionEditTracker(self.workspace, session_id=sub_session_id)
