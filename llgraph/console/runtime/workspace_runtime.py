"""工作区 Runtime：MCP、Trace 配置等进程级资源。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llgraph.config.config import load_llgraph_env
from llgraph.context.context_session import ContextSession
from llgraph.core.tools import load_mcp_tool_bundle
from llgraph.display.trace_display import TraceMode, TraceSession
from llgraph.runtime.shutdown import shutdown_agent_resources
from llgraph.sandbox.policy import build_sandbox_policy
from llgraph.config.sandbox_settings import resolve_sandbox_settings
from llgraph.session.session_web_search import resolve_initial_web_search_enabled


@dataclass
class WorkspaceRuntime:
    """单工作区运行时资源。"""

    workspace: Path
    allow_write: bool = False
    mcp_tools: list = field(default_factory=list)
    mcp_registry: Any = None
    mcp_summary: str = ""
    sandbox_policy: Any = None
    sandbox_cli_enabled: bool | None = None
    web_search_enabled: bool = False
    trace_session: TraceSession = field(default_factory=TraceSession)
    context_session: ContextSession = field(default_factory=ContextSession)


class WorkspaceRuntimeManager:
    """按工作区路径缓存 Runtime。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtimes: dict[str, WorkspaceRuntime] = {}
        load_llgraph_env()

    def get(self, workspace: Path, *, allow_write: bool = False) -> WorkspaceRuntime:
        """
        获取或创建工作区 Runtime。

        @param workspace 工作区根
        @param allow_write 是否允许写（影响 MCP 加载与沙箱）
        @return WorkspaceRuntime
        """
        key = str(workspace.expanduser().resolve())
        with self._lock:
            if key in self._runtimes:
                rt = self._runtimes[key]
                if rt.allow_write != allow_write:
                    self._refresh_write_mode(rt, allow_write=allow_write)
                return rt
            mcp_tools, mcp_registry, mcp_summary = load_mcp_tool_bundle(
                workspace,
                allow_write=allow_write,
            )
            sandbox_settings = resolve_sandbox_settings(workspace)
            sandbox_policy = build_sandbox_policy(
                workspace,
                sandbox_settings,
                cli_enabled=None,
                allow_write=allow_write,
            )
            runtime = WorkspaceRuntime(
                workspace=workspace,
                allow_write=allow_write,
                mcp_tools=mcp_tools,
                mcp_registry=mcp_registry,
                mcp_summary=mcp_summary,
                sandbox_policy=sandbox_policy,
                sandbox_cli_enabled=None,
                web_search_enabled=resolve_initial_web_search_enabled(workspace),
            )
            self._runtimes[key] = runtime
            return runtime

    def _refresh_write_mode(self, rt: WorkspaceRuntime, *, allow_write: bool) -> None:
        """
        Web「允许写」切换时刷新 MCP 与沙箱（不重建整个 Runtime）。

        @param rt 工作区 Runtime
        @param allow_write 目标写权限
        """
        workspace = rt.workspace
        mcp_tools, mcp_registry, mcp_summary = load_mcp_tool_bundle(
            workspace,
            allow_write=allow_write,
        )
        sandbox_settings = resolve_sandbox_settings(workspace)
        sandbox_policy = build_sandbox_policy(
            workspace,
            sandbox_settings,
            cli_enabled=rt.sandbox_cli_enabled,
            allow_write=allow_write,
        )
        rt.allow_write = allow_write
        rt.mcp_tools = mcp_tools
        rt.mcp_registry = mcp_registry
        rt.mcp_summary = mcp_summary
        rt.sandbox_policy = sandbox_policy

    def _rebuild_sandbox_policy(self, rt: WorkspaceRuntime) -> None:
        """
        按当前 cli 覆盖与工作区写模式重建沙箱策略。

        @param rt 工作区 Runtime
        """
        sandbox_settings = resolve_sandbox_settings(rt.workspace)
        rt.sandbox_policy = build_sandbox_policy(
            rt.workspace,
            sandbox_settings,
            cli_enabled=rt.sandbox_cli_enabled,
            allow_write=rt.allow_write,
        )

    def set_sandbox_enabled(self, workspace: Path, *, enabled: bool) -> Any:
        """
        切换工作区 Runtime 的 OS 沙箱开关（覆盖 sandbox.json）。

        @param workspace 工作区根
        @param enabled 是否启用
        @return 更新后的 SandboxPolicy
        """
        rt = self.get(workspace)
        rt.sandbox_cli_enabled = enabled
        self._rebuild_sandbox_policy(rt)
        policy = rt.sandbox_policy
        if enabled and not policy.enabled:
            warning = policy.startup_warning()
            raise ValueError(warning or "沙箱后端不可用，无法启用")
        return policy

    def set_web_search_enabled(self, workspace: Path, *, enabled: bool) -> bool:
        """
        切换工作区 Runtime 的联网搜索开关（影响后续 Agent 与能力清单）。

        @param workspace 工作区根
        @param enabled 是否启用
        @return 当前是否启用
        """
        if enabled:
            from llgraph.config.web_search_settings import validate_web_search_ready

            ok, err = validate_web_search_ready(workspace)
            if not ok:
                raise ValueError(err or "Web 搜索未就绪")
        rt = self.get(workspace)
        rt.web_search_enabled = enabled
        return enabled

    def set_trace_mode(self, workspace: Path, mode: str) -> TraceMode:
        """
        设置 trace 模式。

        @param workspace 工作区根
        @param mode all/steps/reply/none
        @return TraceMode
        """
        from llgraph.display.trace_display import parse_trace_mode

        parsed = parse_trace_mode(mode) or TraceMode.STEPS
        rt = self.get(workspace)
        rt.trace_session.mode = parsed
        return parsed

    def shutdown_all(self) -> None:
        """释放 MCP 等资源。"""
        with self._lock:
            for rt in self._runtimes.values():
                shutdown_agent_resources(mcp_registry=rt.mcp_registry)
            self._runtimes.clear()


RUNTIME_MANAGER = WorkspaceRuntimeManager()
