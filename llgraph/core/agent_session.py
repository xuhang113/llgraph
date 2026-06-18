"""Agent 会话上下文（元命令调用 Agent 用）。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llgraph.context.context_session import ContextSession
from llgraph.survey.edit_confirm import EditConfirmGate
from llgraph.session.session_edits import SessionEditTracker
from llgraph.display.trace_display import TraceSession
from llgraph.sandbox.policy import SandboxPolicy
from llgraph.session.mode_switch import SessionModeTransition


@dataclass
class AgentSessionContext:
    """交互会话中 Agent 相关句柄。"""

    agent: Any
    workspace: Path
    thread_id: str
    trace_session: TraceSession
    context_session: ContextSession
    with_memory: bool = True
    edit_tracker: SessionEditTracker | None = None
    context_spill: Any | None = None
    write_failure_tracker: Any | None = None
    allow_write: bool = False
    mcp_tools: list | None = None
    mcp_registry: Any | None = None
    on_file_changed: Callable[[str], None] | None = None
    watch_service: Any | None = None
    web_search_enabled: bool = False
    edit_confirm_gate: EditConfirmGate | None = None
    sandbox_policy: SandboxPolicy | None = None
    sandbox_cli_enabled: bool | None = None
    mode_switch: SessionModeTransition | None = None
    return_agent_thread_id: str | None = None
