"""工作区内 Shell 命令执行（对齐 Cursor run_terminal_cmd / Claude Code Bash）。"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.config.sandbox_settings import format_sandbox_config_hint
from llgraph.config.shell_settings import ShellSettings, resolve_shell_settings
from llgraph.context.runtime_context import get_active_thread_id
from llgraph.core.shell_cwd import apply_cd_hops, peel_all_leading_cd
from llgraph.core.shell_jobs import (
    ShellJob,
    command_fingerprint,
    get_shell_registry,
    new_job_id,
)
from llgraph.core.shell_output import clip_shell_output, combine_stdio
from llgraph.core.shell_schemas import AwaitShellInput, RunShellCommandInput
from llgraph.core.tool_arg_coerce import format_tool_validation_error
from llgraph.core.workspace import WorkspaceContext
from llgraph.permissions.shell import check_shell_command
from llgraph.sandbox.exec import LiveShellProcess, spawn_sandboxed_shell
from llgraph.sandbox.policy import SandboxPolicy


def _append_terminal_log(
    workspace: Path,
    settings: ShellSettings,
    *,
    command: str,
    cwd: str,
    exit_code: int,
    output_chars: int,
    job_id: str = "",
    status: str = "",
) -> None:
    """追加命令执行摘要到终端日志。"""
    if not settings.log_commands:
        return
    log_dir = workspace / settings.terminal_log_dir
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "commands.jsonl"
        extra = ""
        if job_id:
            extra += f',"job_id":{job_id!r}'
        if status:
            extra += f',"status":{status!r}'
        line = (
            f'{{"ts":"{datetime.now(timezone.utc).isoformat()}",'
            f'"cwd":{cwd!r},"exit_code":{exit_code},'
            f'"output_chars":{output_chars},"command":{command[:500]!r}{extra}}}\n'
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass


def _inactive_sandbox_policy(workspace: Path, *, allow_write: bool = False) -> SandboxPolicy:
    from llgraph.config.sandbox_settings import resolve_sandbox_settings
    from llgraph.sandbox.policy import build_sandbox_policy

    settings = resolve_sandbox_settings(workspace)
    return build_sandbox_policy(workspace, settings, cli_enabled=False, allow_write=allow_write)


def _thread_key() -> str:
    return get_active_thread_id() or "anon"


def _cancel_check() -> bool:
    from llgraph.core.react_invoke import agent_cancel_requested

    return agent_cancel_requested()


def _resolve_work_dir(
    ctx: WorkspaceContext,
    rel: str,
) -> tuple[Path | None, str, str]:
    """
    解析相对工作目录。

    @return (绝对路径, 相对路径, 错误)
    """
    try:
        work_dir = ctx.resolve_path(rel or ".")
    except (ValueError, RuntimeError, PermissionError) as exc:
        return None, rel or ".", f"工作目录无效: {exc}"
    if not work_dir.is_dir():
        return None, rel or ".", f"工作目录不存在或不是目录: {rel or '.'}"
    try:
        rel_cwd = work_dir.relative_to(ctx.root).as_posix()
    except ValueError:
        rel_cwd = rel or "."
    if rel_cwd == ".":
        rel_cwd = "."
    return work_dir, rel_cwd or ".", ""


def _format_header(
    *,
    sandboxed: bool,
    cwd: str,
    elapsed: float,
    exit_code: int | None,
    job_id: str = "",
    status: str = "",
) -> str:
    sandbox_tag = "sandbox" if sandboxed else "nosandbox"
    parts = [sandbox_tag, f"cwd={cwd}"]
    if status:
        parts.append(status)
    elif exit_code is not None:
        parts.append(f"exit={exit_code}")
    if job_id:
        parts.append(f"job={job_id}")
    parts.append(f"{elapsed:.2f}s")
    return f"--- shell ({', '.join(parts)}) ---\n"


def _clip_body(text: str, max_chars: int) -> str:
    clipped, _omitted = clip_shell_output(text, max_chars)
    body = clipped if clipped.strip() else "(无输出)"
    return body


def _format_finished(
    live: LiveShellProcess,
    *,
    cwd_rel: str,
    max_chars: int,
    sandbox_enabled: bool,
    workspace: Path,
    job_id: str = "",
) -> str:
    stdout, stderr = live.snapshot_stdio()
    combined = combine_stdio(stdout, stderr)
    code = live.returncode()
    if code is None:
        code = -1
    if sandbox_enabled and code != 0:
        combined = (
            combined
            + "\n[沙箱] 命令可能被 sandbox.json 限制（路径/网络/只读模式）。\n"
            + format_sandbox_config_hint(workspace)
        )
    body = _clip_body(combined, max_chars)
    status = ""
    footer = ""
    if live.error == "timeout":
        status = "timeout"
        footer = (
            f"\n错误: 命令超时已终止；上面是超时前捕获的 stdout/stderr。"
            f"若测试/构建需要更久：传 block_until_ms=0 或 30000，再用 await_shell。"
        )
    elif live.error == "cancelled":
        status = "cancelled"
        footer = "\n[llgraph] 用户已停止当前生成。"
    elif code != 0:
        footer = f"\n[exit {code}]"
    header = _format_header(
        sandboxed=live.sandboxed,
        cwd=cwd_rel,
        elapsed=live.elapsed_sec(),
        exit_code=None if status else code,
        job_id=job_id,
        status=status,
    )
    return header + body + footer


def _format_running(
    job: ShellJob,
    *,
    max_chars: int,
    note: str = "",
) -> str:
    stdout, stderr = job.live.snapshot_stdio()
    combined = combine_stdio(stdout, stderr)
    body = _clip_body(combined, max_chars)
    header = _format_header(
        sandboxed=job.live.sandboxed,
        cwd=job.cwd_rel,
        elapsed=job.live.elapsed_sec(),
        exit_code=None,
        job_id=job.job_id,
        status="running",
    )
    hint = (
        f"\n[llgraph] 命令仍在运行。请 await_shell(job_id=\"{job.job_id}\") 继续等待；"
        "可设 pattern 在输出匹配时提前返回。不要再启动同一条命令。"
    )
    extra = f"\n{note}" if note else ""
    return header + body + hint + extra


def _pattern_hits(text: str, pattern: str) -> tuple[bool, str]:
    needle = (pattern or "").strip()
    if not needle:
        return False, ""
    try:
        compiled = re.compile(needle)
    except re.error as exc:
        return False, f"pattern 不是合法正则: {exc}"
    return compiled.search(text) is not None, ""


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

    sandbox = ctx.sandbox_policy or _inactive_sandbox_policy(ctx.root, allow_write=ctx.allow_write)
    registry = get_shell_registry()

    def _log(
        command: str,
        cwd: str,
        *,
        exit_code: int,
        output_chars: int,
        job_id: str = "",
        status: str = "",
    ) -> None:
        _append_terminal_log(
            ctx.root,
            shell_settings,
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            output_chars=output_chars,
            job_id=job_id,
            status=status,
        )

    def run_shell_command(
        command: str,
        working_directory: str = "",
        block_until_ms: int | None = None,
    ) -> str:
        """
        在工作区内执行 shell 命令，返回合并输出（stdout+stderr）。

        超时/取消仍返回已捕获输出（失败信息通常在末尾）。超长输出保留开头+末尾。
        长任务请传 block_until_ms（0=立即后台）；未完成时返回 job_id，再用 await_shell。
        会话会记住 cd 后的工作目录；working_directory 为空则沿用。

        @param command 要执行的 shell 命令（单条）
        @param working_directory 相对工作区目录；空=沿用本会话上次目录
        @param block_until_ms 前台等待毫秒；省略则等到 timeout_sec 后终止
        @return 命令输出或后台 job 状态
        """
        blocked = check_shell_command(command, allow_write=allow_write)
        if blocked:
            return f"错误: {blocked}"

        thread_id = _thread_key()
        if (working_directory or "").strip():
            start_rel = working_directory.strip()
        else:
            start_rel = registry.get_cwd(thread_id)

        start_abs, start_rel, err = _resolve_work_dir(ctx, start_rel)
        if err:
            return err

        hops, rest = peel_all_leading_cd(command)
        if hops:
            new_rel, cd_err = apply_cd_hops(ctx, start_rel, hops)
            if cd_err:
                return cd_err
            start_rel = new_rel or "."
            start_abs, start_rel, err = _resolve_work_dir(ctx, start_rel)
            if err:
                return err
        registry.set_cwd(thread_id, start_rel)

        if not (rest or "").strip():
            _log(command, start_rel, exit_code=0, output_chars=0, status="cwd")
            return (
                _format_header(
                    sandboxed=sandbox.enabled,
                    cwd=start_rel,
                    elapsed=0.0,
                    exit_code=0,
                    status="cwd",
                )
                + f"(已切换工作目录 → {start_rel}；后续省略 working_directory 的 shell 将落在此处)"
            )

        fp = command_fingerprint(rest, start_rel)
        existing = registry.find_running(thread_id, fp)
        if existing is not None:
            return _format_running(
                existing,
                max_chars=shell_settings.max_output_chars,
                note="[llgraph] 相同命令已在运行，未再启动新进程。",
            )

        if registry.running_count(thread_id) >= shell_settings.max_jobs:
            running = [
                job.job_id
                for job in registry.list_jobs(thread_id)
                if job.live.returncode() is None
            ]
            return (
                f"错误: 本会话在途 shell 已达上限 {shell_settings.max_jobs}。"
                f"请先 await_shell：{', '.join(running) or '(未知)'}"
            )

        live, spawn_err = spawn_sandboxed_shell(
            sandbox,
            command=rest,
            cwd=start_abs or ctx.root,
            env=os.environ.copy(),
            hard_timeout_sec=shell_settings.background_timeout_sec,
        )
        if spawn_err or live is None:
            return f"执行失败: {spawn_err or '启动失败'}"

        job = ShellJob(
            job_id=new_job_id(),
            thread_id=thread_id,
            command=rest,
            cwd_rel=start_rel,
            fingerprint=fp,
            live=live,
            created_at=time.perf_counter(),
        )
        registry.register(job)

        if block_until_ms is None:
            wait_sec = shell_settings.timeout_sec
            after = "kill"
        else:
            wait_sec = max(0, int(block_until_ms)) / 1000.0
            after = "background"

        finished = live.wait(wait_sec, cancel_check=_cancel_check)
        if finished:
            out = _format_finished(
                live,
                cwd_rel=start_rel,
                max_chars=shell_settings.max_output_chars,
                sandbox_enabled=sandbox.enabled,
                workspace=ctx.root,
                job_id=job.job_id,
            )
            code = live.returncode() if live.returncode() is not None else -1
            _log(
                rest,
                start_rel,
                exit_code=code,
                output_chars=len(combine_stdio(*live.snapshot_stdio())),
                job_id=job.job_id,
                status=live.error or "done",
            )
            return out

        if after == "kill":
            live.kill("timeout")
            out = _format_finished(
                live,
                cwd_rel=start_rel,
                max_chars=shell_settings.max_output_chars,
                sandbox_enabled=sandbox.enabled,
                workspace=ctx.root,
                job_id=job.job_id,
            )
            _log(
                rest,
                start_rel,
                exit_code=-1,
                output_chars=len(combine_stdio(*live.snapshot_stdio())),
                job_id=job.job_id,
                status="timeout",
            )
            return out

        _log(
            rest,
            start_rel,
            exit_code=-1,
            output_chars=len(combine_stdio(*live.snapshot_stdio())),
            job_id=job.job_id,
            status="running",
        )
        return _format_running(job, max_chars=shell_settings.max_output_chars)

    def await_shell(
        job_id: str,
        block_until_ms: int = 30_000,
        pattern: str = "",
    ) -> str:
        """
        等待后台 shell 任务结束或输出匹配 pattern。

        @param job_id run_shell_command 返回的 job id
        @param block_until_ms 本次最多再等毫秒；0=立即看当前输出
        @param pattern 可选正则，匹配到已捕获输出则提前返回
        @return 当前输出；仍在运行时带 job_id 提示
        """
        jid = (job_id or "").strip()
        if not jid:
            return "错误: job_id 不能为空"
        job = registry.get(jid)
        if job is None:
            known = [item.job_id for item in registry.list_jobs(_thread_key())]
            extra = f" 本会话已知: {', '.join(known)}" if known else " 本会话没有后台 shell。"
            return f"错误: 找不到 job {jid!r}。{extra}"

        wait_sec = max(0, int(block_until_ms)) / 1000.0
        deadline = time.monotonic() + wait_sec
        needle = (pattern or "").strip()

        while True:
            if job.live.returncode() is not None:
                out = _format_finished(
                    job.live,
                    cwd_rel=job.cwd_rel,
                    max_chars=shell_settings.max_output_chars,
                    sandbox_enabled=sandbox.enabled,
                    workspace=ctx.root,
                    job_id=job.job_id,
                )
                code = job.live.returncode() if job.live.returncode() is not None else -1
                _log(
                    job.command,
                    job.cwd_rel,
                    exit_code=code,
                    output_chars=len(combine_stdio(*job.live.snapshot_stdio())),
                    job_id=job.job_id,
                    status=job.live.error or "done",
                )
                return out
            stdout, stderr = job.live.snapshot_stdio()
            combined = combine_stdio(stdout, stderr)
            if needle:
                hit, perr = _pattern_hits(combined, needle)
                if perr:
                    return f"错误: {perr}"
                if hit:
                    return _format_running(
                        job,
                        max_chars=shell_settings.max_output_chars,
                        note=f'[llgraph] pattern={needle!r} 已命中，进程仍在运行。',
                    )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _format_running(job, max_chars=shell_settings.max_output_chars)
            slice_sec = min(0.2, remaining)
            job.live.wait(slice_sec, cancel_check=_cancel_check)
            if _cancel_check():
                return _format_finished(
                    job.live,
                    cwd_rel=job.cwd_rel,
                    max_chars=shell_settings.max_output_chars,
                    sandbox_enabled=sandbox.enabled,
                    workspace=ctx.root,
                    job_id=job.job_id,
                )

    return [
        StructuredTool.from_function(
            func=run_shell_command,
            name="run_shell_command",
            description=run_shell_command.__doc__ or "",
            args_schema=RunShellCommandInput,
            handle_validation_error=format_tool_validation_error,
        ),
        StructuredTool.from_function(
            func=await_shell,
            name="await_shell",
            description=await_shell.__doc__ or "",
            args_schema=AwaitShellInput,
            handle_validation_error=format_tool_validation_error,
        ),
    ]
