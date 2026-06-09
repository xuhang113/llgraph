"""在 OS 沙箱中执行 Shell 命令。"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
) -> SandboxShellResult:
    """
    在 OS 沙箱中执行 shell 命令；未启用沙箱时回退普通 subprocess。

    @param policy 沙箱策略
    @param command shell 命令
    @param cwd 工作目录
    @param timeout_sec 超时秒数
    @param env 环境变量
    @return SandboxShellResult
    """
    run_env = dict(env or os.environ)

    if not policy.enabled:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=run_env,
            )
        except subprocess.TimeoutExpired:
            return SandboxShellResult("", "", -1, False, error="timeout")
        except OSError as exc:
            return SandboxShellResult("", "", -1, False, error=str(exc))
        return SandboxShellResult(
            completed.stdout or "",
            completed.stderr or "",
            completed.returncode,
            False,
        )

    if policy.backend == "macos_seatbelt":
        return _run_macos(policy, command=command, cwd=cwd, timeout_sec=timeout_sec, env=run_env)
    if policy.backend == "linux_bwrap":
        return _run_linux(policy, command=command, cwd=cwd, timeout_sec=timeout_sec, env=run_env)
    return SandboxShellResult(
        "",
        "",
        -1,
        False,
        error="沙箱后端不可用",
    )


def _run_macos(
    policy: SandboxPolicy,
    *,
    command: str,
    cwd: Path,
    timeout_sec: float,
    env: dict[str, str],
) -> SandboxShellResult:
    profile_path = policy.create_seatbelt_profile_file()
    try:
        argv = [
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile_path),
            "/bin/sh",
            "-c",
            command,
        ]
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return SandboxShellResult("", "", -1, True, error="timeout")
    except OSError as exc:
        return SandboxShellResult("", "", -1, True, error=str(exc))
    finally:
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            pass

    return SandboxShellResult(
        completed.stdout or "",
        completed.stderr or "",
        completed.returncode,
        True,
    )


def _run_linux(
    policy: SandboxPolicy,
    *,
    command: str,
    cwd: Path,
    timeout_sec: float,
    env: dict[str, str],
) -> SandboxShellResult:
    from llgraph.sandbox.linux import build_bwrap_command

    argv = build_bwrap_command(policy, command=command, cwd=cwd)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return SandboxShellResult("", "", -1, True, error="timeout")
    except OSError as exc:
        return SandboxShellResult("", "", -1, True, error=str(exc))

    return SandboxShellResult(
        completed.stdout or "",
        completed.stderr or "",
        completed.returncode,
        True,
    )
