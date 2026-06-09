"""Agent 权限策略：路径边界、写模式、Shell/MCP 拦截（统一入口）。"""

from llgraph.permissions.file_write import (
    FILE_WRITE_DENIED_MESSAGE,
    require_file_write,
)
from llgraph.permissions.mcp import is_write_mcp_tool
from llgraph.permissions.paths import resolve_read_path, resolve_workspace_path
from llgraph.permissions.shell import check_shell_command

__all__ = [
    "FILE_WRITE_DENIED_MESSAGE",
    "check_shell_command",
    "is_write_mcp_tool",
    "require_file_write",
    "resolve_read_path",
    "resolve_workspace_path",
]
