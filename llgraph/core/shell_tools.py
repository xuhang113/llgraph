"""工作区内 Shell 命令执行（类 Cursor run_terminal_cmd）。"""

from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.config.shell_settings import ShellSettings, resolve_shell_settings
from llgraph.core.workspace import WorkspaceContext

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


def _command_blocked(command: str, *, allow_write: bool) -> str | None:
    """
    校验命令是否允许执行。

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


def _append_terminal_log(
    workspace: Path,
    settings: ShellSettings,
    *,
    command: str,
    cwd: str,
    exit_code: int,
    output_chars: int,
) -> None:
    """追加命令执行摘要到终端日志。"""
    if not settings.log_commands:
        return
    log_dir = workspace / settings.terminal_log_dir
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "commands.jsonl"
        line = (
            f'{{"ts":"{datetime.now(timezone.utc).isoformat()}",'
            f'"cwd":{cwd!r},"exit_code":{exit_code},'
            f'"output_chars":{output_chars},"command":{command[:500]!r}}}\n'
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass


def create_shell_tools(
    ctx: WorkspaceContext,
    *,
    allow_write: bool = False,
    settings: ShellSettings | None = None,
) -> list:
    """
    创建 shell 工具。

    @param ctx 工作区上下文
    @param allow_write 是否 -w 模式
    @param settings shell 配置
    @return Tool 列表；未启用时返回空列表
    """
    shell_settings = settings or resolve_shell_settings(ctx.root)
    if not shell_settings.enabled:
        return []

    def run_shell_command(command: str, working_directory: str = ".") -> str:
        """
        在工作区内执行 shell 命令并返回合并输出（stdout+stderr）。

        用于 pwd、git status、mvn test、构建脚本等；只读模式禁止 rm/git commit/重定向写文件。
        working_directory 为相对工作区路径，默认工作区根。

        @param command 要执行的 shell 命令（单条）
        @param working_directory 相对工作区的执行目录，默认 .
        @return 命令输出；非零退出码时前缀 [exit N]
        """
        blocked = _command_blocked(command, allow_write=allow_write)
        if blocked:
            return f"错误: {blocked}"

        try:
            work_dir = ctx.resolve_path(working_directory)
        except (ValueError, RuntimeError) as exc:
            return f"工作目录无效: {exc}"

        if not work_dir.is_dir():
            return f"工作目录不存在或不是目录: {working_directory}"

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=shell_settings.timeout_sec,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return (
                f"错误: 命令超时（>{shell_settings.timeout_sec:.0f}s）: {command[:200]}"
            )
        except OSError as exc:
            return f"执行失败: {exc}"

        elapsed = time.perf_counter() - started
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        combined = stdout
        if stderr:
            if combined and not combined.endswith("\n"):
                combined += "\n"
            combined += stderr

        if len(combined) > shell_settings.max_output_chars:
            combined = (
                combined[: shell_settings.max_output_chars]
                + f"\n…(输出已截断，共超过 {shell_settings.max_output_chars} 字符；"
                "完整输出可能已落盘 tool-results)"
            )

        rel_cwd = working_directory
        try:
            rel_cwd = work_dir.relative_to(ctx.root).as_posix()
        except ValueError:
            pass

        _append_terminal_log(
            ctx.root,
            shell_settings,
            command=command,
            cwd=rel_cwd,
            exit_code=completed.returncode,
            output_chars=len(stdout) + len(stderr),
        )

        header = (
            f"--- shell (cwd={rel_cwd}, exit={completed.returncode}, "
            f"{elapsed:.2f}s) ---\n"
        )
        body = combined if combined.strip() else "(无输出)"
        if completed.returncode != 0:
            return header + body + f"\n[exit {completed.returncode}]"
        return header + body

    return [
        StructuredTool.from_function(
            func=run_shell_command,
            name="run_shell_command",
            description=run_shell_command.__doc__ or "",
        ),
    ]
