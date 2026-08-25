"""在 OS 沙箱中执行 Shell 命令。"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llgraph.sandbox.exec import spawn_sandboxed_shell
from llgraph.sandbox.policy import SandboxPolicy


@dataclass(frozen=True)
class SandboxShellResult:
    """沙箱 Shell 执行结果。"""

    stdout: str
    stderr: str
    returncode: int
    sandboxed: bool
    error: str | None = None


def run_sandboxed_shell(
    policy: SandboxPolicy,
    *,
    command: str,
    cwd: Path,
    timeout_sec: float,
    env: dict[str, str] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SandboxShellResult:
    """
    在 OS 沙箱中执行 shell 命令；未启用沙箱时回退普通 /bin/sh。

    超时或取消时仍返回已捕获的 stdout/stderr（不再丢空）。

    @param policy 沙箱策略
    @param command shell 命令
    @param cwd 工作目录
    @param timeout_sec 超时秒数（到期杀进程组）
    @param env 环境变量
    @param cancel_check 可选；返回 True 时终止
    @return SandboxShellResult
    """
    run_env = dict(env or os.environ)
    live, err = spawn_sandboxed_shell(
        policy,
        command=command,
        cwd=cwd,
        env=run_env,
        hard_timeout_sec=max(5.0, timeout_sec),
    )
    if err or live is None:
        return SandboxShellResult("", "", -1, False, error=err or "启动失败")

    finished = live.wait(timeout_sec, cancel_check=cancel_check)
    if not finished:
        live.kill("timeout")
    stdout, stderr = live.snapshot_stdio()
    code = live.returncode()
    if code is None:
        code = -1
    return SandboxShellResult(
        stdout,
        stderr,
        code,
        live.sandboxed,
        error=live.error,
    )
