"""Agent 会话资源回收（watch / MCP / 编辑账本）。"""

from __future__ import annotations

from typing import Any


def shutdown_agent_resources(
    *,
    watch_service: Any | None = None,
    mcp_registry: Any | None = None,
    edit_tracker: Any | None = None,
) -> None:
    """
    交互结束或进程退出前释放后台资源，避免 atexit 阶段 join 阻塞。

    @param watch_service IndexWatchService
    @param mcp_registry McpToolRegistry
    @param edit_tracker SessionEditTracker
    """
    if watch_service is not None:
        try:
            watch_service.stop()
        except Exception:
            pass

    if mcp_registry is not None:
        try:
            mcp_registry.stop()
        except Exception:
            pass

    if edit_tracker is not None:
        try:
            summary = edit_tracker.exit_summary()
            if summary:
                from llgraph.terminal.output import emit_report

                emit_report(summary)
        except Exception:
            pass
