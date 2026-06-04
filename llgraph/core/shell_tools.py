"""工作区内 Shell 命令执行（类 Cursor run_terminal_cmd）。"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.config.sandbox_settings import format_sandbox_config_hint
from llgraph.config.shell_settings import ShellSettings, resolve_shell_settings
from llgraph.core.workspace import WorkspaceContext
from llgraph.permissions.shell import check_shell_command
from llgraph.sandbox.policy import SandboxPolicy
from llgraph.sandbox.runner import run_sandboxed_shell


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


def _inactive_sandbox_policy(workspace: Path) -> SandboxPolicy:
    from llgraph.config.sandbox_settings import resolve_sandbox_settings
    from llgraph.sandbox.policy import build_sandbox_policy

    settings = resolve_sandbox_settings(workspace)
    return build_sandbox_policy(workspace, settings, cli_enabled=False)


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

    sandbox = ctx.sandbox_policy or _inactive_sandbox_policy(ctx.root)

    def run_shell_command(command: str, working_directory: str = ".") -> str:
        """
        在工作区内执行 shell 命令并返回合并输出（stdout+stderr）。

        启用沙箱时命令在 OS 沙箱子进程中执行（macOS sandbox-exec / Linux bwrap）。
        只读模式禁止 rm/git commit/重定向写文件。
        working_directory 为相对工作区路径，默认工作区根。

        @param command 要执行的 shell 命令（单条）
        @param working_directory 相对工作区的执行目录，默认 .
        @return 命令输出；非零退出码时前缀 [exit N]
        """
        blocked = check_shell_command(command, allow_write=allow_write)
        if blocked:
            return f"错误: {blocked}"

        try:
            work_dir = ctx.resolve_path(working_directory)
        except (ValueError, RuntimeError, PermissionError) as exc:
            return f"工作目录无效: {exc}"

        if not work_dir.is_dir():
            return f"工作目录不存在或不是目录: {working_directory}"

        started = time.perf_counter()
        result = run_sandboxed_shell(
            sandbox,
            command=command,
            cwd=work_dir,
            timeout_sec=shell_settings.timeout_sec,
            env=os.environ.copy(),
        )

        if result.error == "timeout":
            return (
                f"错误: 命令超时（>{shell_settings.timeout_sec:.0f}s）: {command[:200]}"
            )
        if result.error:
            return f"执行失败: {result.error}"

        stdout = result.stdout
        stderr = result.stderr
        combined = stdout
        if stderr:
            if combined and not combined.endswith("\n"):
                combined += "\n"
            combined += stderr

        if sandbox.enabled and result.returncode != 0:
            combined = (
                combined
                + "\n[沙箱] 命令可能被 sandbox.json 限制（路径/网络/只读模式）。\n"
                + format_sandbox_config_hint(ctx.root)
            )

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
            exit_code=result.returncode,
            output_chars=len(stdout) + len(stderr),
        )

        sandbox_tag = "sandbox" if result.sandboxed else "nosandbox"
        header = (
            f"--- shell ({sandbox_tag}, cwd={rel_cwd}, exit={result.returncode}, "
            f"{time.perf_counter() - started:.2f}s) ---\n"
        )
        body = combined if combined.strip() else "(无输出)"
        if result.returncode != 0:
            return header + body + f"\n[exit {result.returncode}]"
        return header + body

    return [
        StructuredTool.from_function(
            func=run_shell_command,
            name="run_shell_command",
            description=run_shell_command.__doc__ or "",
        ),
    ]
