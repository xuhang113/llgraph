"""MCP 工具装载入口（与 `core.tools` 分开，专为冷启动）。

`core.tools` 顶层要 `langchain_core.tools` + 全部内置工具模块（约 0.4s），
但启动时先跑的只是「有没有配 MCP」这一步。没配 Server 就不该付这笔钱。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_mcp_tool_bundle(
    workspace: Path,
    *,
    allow_write: bool = False,
) -> tuple[list, Any | None, str]:
    """
    加载 MCP 工具与 registry。

    失败可降级：不抛异常，仅跳过失败 Server；不影响 Agent 其它工具。

    @param workspace 工作区根
    @param allow_write 是否允许 MCP 写类工具
    @return (tools, registry, summary)
    """
    from llgraph.config.mcp_config import format_mcp_summary, resolve_mcp_settings

    try:
        settings = resolve_mcp_settings(workspace, allow_write=allow_write)
    except Exception as exc:
        return [], None, f"MCP: 配置解析失败（已跳过）{exc}"

    summary = format_mcp_summary(settings)
    if not settings.servers:
        return [], None, summary

    from llgraph.core.mcp_tools import create_mcp_tools

    try:
        tools, registry = create_mcp_tools(settings)
    except Exception as exc:
        return [], None, f"MCP: 加载失败（已跳过，不影响其它功能）{exc}"

    if registry is not None:
        summary = registry.summary()
    return tools, registry, summary
