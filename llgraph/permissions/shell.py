"""Shell 命令权限：只读黑名单与全局高危拦截。"""

from __future__ import annotations

import re

# 只读模式下禁止的 shell 模式（写盘、提交、危险删除等）
_READ_ONLY_BLOCKED = re.compile(
    r"(?:^|[;&|]\s*)"
    r"(?:"
    r"rm\s|"
    r"mv\s|"
    r"cp\s|"
    r"chmod\s|"
    r"chown\s|"
    r"sudo\s|"
    r"tee\s|"
    r"git\s+(?:commit|push|pull|reset|checkout|merge|rebase|cherry-pick|stash\s+(?:push|drop|clear))\b|"
    r"(?:npm|pnpm|yarn)\s+(?:install|uninstall|publish)\b|"
    r"pip\s+install\b|"
    r"mvn\s+install\b"
    r")",
    re.IGNORECASE,
)

# 任意模式均禁止（即使 -w）
_ALWAYS_BLOCKED = re.compile(
    r"(?:^|[;&|]\s*)"
    r"(?:"
    r"rm\s+-[a-zA-Z]*f[a-zA-Z]*\s+/\b|"
    r"rm\s+-[a-zA-Z]*f[a-zA-Z]*\s+/\s|"
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"
    r")",
    re.IGNORECASE,
)

_REDIRECT_PATTERN = re.compile(r"[><]{1,2}")


def check_shell_command(command: str, *, allow_write: bool) -> str | None:
    """
    校验 shell 命令是否允许执行。

    @param command shell 命令
    @param allow_write 是否 -w 模式
    @return 拒绝原因；允许则返回 None
    """
    stripped = command.strip()
    if not stripped:
        return "命令不能为空"
    if _ALWAYS_BLOCKED.search(stripped):
        return "该命令被安全策略禁止（高危删除/ fork 炸弹）"
    if not allow_write:
        if _READ_ONLY_BLOCKED.search(stripped):
            return (
                "只读模式禁止该命令（写盘/git 提交/安装依赖等）。"
                "请使用 llgraph -w 启动，或改用 read_file/grep_files 等只读工具。"
            )
        if _REDIRECT_PATTERN.search(stripped):
            return "只读模式禁止 shell 重定向（> >>）。请使用 llgraph -w 或只读工具。"
    return None
