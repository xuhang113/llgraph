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
