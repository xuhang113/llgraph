"""MCP 工具权限：只读模式下过滤写类工具。"""

from __future__ import annotations

_WRITE_TOOL_KEYWORDS = frozenset({
    "write",
    "edit",
    "delete",
    "create",
    "update",
    "insert",
    "remove",
    "patch",
})


def is_write_mcp_tool(name: str, description: str) -> bool:
    """
    根据工具名与描述判断是否属于写类 MCP 工具。

    @param name MCP 工具名
    @param description MCP 工具描述
    @return 是否应视为写工具
    """
    text = f"{name} {description}".lower()
    return any(keyword in text for keyword in _WRITE_TOOL_KEYWORDS)
