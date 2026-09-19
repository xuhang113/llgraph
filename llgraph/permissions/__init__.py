"""Agent 权限策略：路径边界、写模式、Shell/MCP 拦截、逐次授权（统一入口）。"""

from llgraph.permissions.approval import (
    ApprovalDecision,
    ApprovalRequest,
    check_approval,
    request_approval,
    use_approval_gate,
)
from llgraph.permissions.file_write import (
    FILE_WRITE_DENIED_MESSAGE,
    require_file_write,
)
from llgraph.permissions.mcp import McpToolAccess, classify_mcp_tool
from llgraph.permissions.paths import resolve_read_path, resolve_workspace_path
from llgraph.permissions.shell import check_shell_command

__all__ = [
    "FILE_WRITE_DENIED_MESSAGE",
    "ApprovalDecision",
    "ApprovalRequest",
    "McpToolAccess",
    "check_approval",
    "check_shell_command",
    "classify_mcp_tool",
    "request_approval",
    "require_file_write",
    "resolve_read_path",
    "resolve_workspace_path",
    "use_approval_gate",
]
