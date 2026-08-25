"""Popen 执行沙箱/普通 shell：超时保留已输出，可后台保活。"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llgraph.sandbox.policy import SandboxPolicy

_MAX_CAPTURE_BYTES = 2_000_000
_HEAD_KEEP_BYTES = 800_000
_POLL_SEC = 0.05


class CappedByteBuffer:
    """有上限的字节缓冲：超出后丢中间、留头尾，避免 `yes` 撑爆内存。"""

    def __init__(self, cap: int = _MAX_CAPTURE_BYTES, head_keep: int = _HEAD_KEEP_BYTES) -> None:
        self._cap = max(64, cap)
        self._head_keep = max(16, min(head_keep, self._cap // 2))
        self._tail_keep = max(16, self._cap - self._head_keep)
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0
        self._lock = threading.Lock()

    def write(self, data: bytes) -> None:
        """追加一块字节。"""
        if not data:
            return
        with self._lock:
            self.total += len(data)
            if len(self.head) < self._head_keep:
                need = self._head_keep - len(self.head)
                self.head.extend(data[:need])
                data = data[need:]
            if not data:
                return
            self.tail.extend(data)
            overflow = len(self.tail) - self._tail_keep
            if overflow > 0:
                del self.tail[:overflow]

    def snapshot(self) -> bytes:
        """当前已捕获字节（头 + 尾）。"""
        with self._lock:
            if not self.tail:
                return bytes(self.head)
            omitted = self.total - len(self.head) - len(self.tail)
            if omitted <= 0:
                return bytes(self.head) + bytes(self.tail)
            marker = f"\n…(进程输出中间省略 {omitted} 字节)\n".encode("utf-8")
            return bytes(self.head) + marker + bytes(self.tail)


def _decode_bytes(raw: bytes) -> str:
    """非法 UTF-8 用替换字符，避免整轮对话崩溃。"""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """杀掉进程组（含子进程）；已退出则忽略。"""
    if proc.poll() is not None:
        return
    pid = proc.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=1.5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def build_shell_argv(
    policy: SandboxPolicy,
    *,
    command: str,
    cwd: Path,
) -> tuple[list[str], Path | None, bool, list[Path], str]:
    """
    构造 Popen argv。

    @param policy 沙箱策略
    @param command shell 命令
    @param cwd 工作目录
    @return (argv, popen_cwd, sandboxed, 结束后需删除的文件, 错误)
    """
    if not policy.enabled:
        return ["/bin/sh", "-c", command], cwd, False, [], ""

    if policy.backend == "macos_seatbelt":
        profile_path = policy.create_seatbelt_profile_file()
        argv = [
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile_path),
            "/bin/sh",
            "-c",
            command,
        ]
        return argv, cwd, True, [profile_path], ""

    if policy.backend == "linux_bwrap":
        from llgraph.sandbox.linux import build_bwrap_command

        argv = build_bwrap_command(policy, command=command, cwd=cwd)
        return argv, None, True, [], ""

    return [], None, False, [], "沙箱后端不可用"


@dataclass
class LiveShellProcess:
    """正在运行或刚结束的 shell 子进程。"""

    proc: subprocess.Popen[bytes]
    sandboxed: bool
    command: str
    cwd: Path
    started_at: float
    stdout_buf: CappedByteBuffer
    stderr_buf: CappedByteBuffer
    cleanup_paths: list[Path] = field(default_factory=list)
    error: str | None = None
    hard_deadline: float = 0.0
    _readers: list[threading.Thread] = field(default_factory=list)
    _readers_joined: bool = False
    _kill_lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot_stdio(self) -> tuple[str, str]:
        """当前 stdout / stderr 文本。"""
        return (
            _decode_bytes(self.stdout_buf.snapshot()),
            _decode_bytes(self.stderr_buf.snapshot()),
        )

    def returncode(self) -> int | None:
        """子进程退出码；仍在运行则为 None。"""
        return self.proc.poll()

    def elapsed_sec(self) -> float:
        """已运行秒数。"""
        return max(0.0, time.perf_counter() - self.started_at)

    def _join_readers(self) -> None:
        if self._readers_joined:
            return
        for thread in self._readers:
            thread.join(timeout=1.0)
        self._readers_joined = True
        self._cleanup_sidecar()

    def _cleanup_sidecar(self) -> None:
        for path in self.cleanup_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self.cleanup_paths = []

    def kill(self, reason: str) -> None:
        """
        终止进程组并记录原因（幂等）。

        @param reason timeout | cancelled | 其它
        """
        with self._kill_lock:
            if self.error is None:
                self.error = reason
            _kill_process_group(self.proc)
        self._join_readers()

    def wait(
        self,
        timeout_sec: float | None,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> bool:
        """
        等到退出、取消、硬超时或调用方等待到期。

        @param timeout_sec 本次等待上限；None 表示直到退出/硬超时/取消
        @param cancel_check 返回 True 时杀进程
        @return 进程是否已退出（含被杀）
        """
        deadline = (
            None if timeout_sec is None else time.monotonic() + max(0.0, timeout_sec)
        )
        while self.proc.poll() is None:
            if cancel_check is not None and cancel_check():
                self.kill("cancelled")
                return True
            if self.hard_deadline and time.monotonic() >= self.hard_deadline:
                self.kill("timeout")
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(_POLL_SEC)
        self._join_readers()
        return True


def _start_reader(pipe: object, buf: CappedByteBuffer) -> threading.Thread:
    def _run() -> None:
        try:
            read = getattr(pipe, "read", None)
            if read is None:
                return
            while True:
                chunk = read(4096)
                if not chunk:
                    break
                buf.write(chunk)
        except OSError:
            pass
        finally:
            try:
                close = getattr(pipe, "close", None)
                if close is not None:
                    close()
            except OSError:
                pass

    thread = threading.Thread(target=_run, name="llgraph-shell-reader", daemon=True)
    thread.start()
    return thread


def spawn_sandboxed_shell(
    policy: SandboxPolicy,
    *,
    command: str,
    cwd: Path,
    env: dict[str, str] | None = None,
    hard_timeout_sec: float = 1800.0,
) -> tuple[LiveShellProcess | None, str]:
    """
    启动 shell 子进程（不阻塞到结束）。

    @param policy 沙箱策略
    @param command 命令
    @param cwd 工作目录
    @param env 环境变量
    @param hard_timeout_sec 硬超时（后台任务上限）
    @return (进程, 错误)；失败时进程为 None
    """
    argv, popen_cwd, sandboxed, cleanup, err = build_shell_argv(
        policy, command=command, cwd=cwd
    )
    if err:
        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return None, err

    run_env = dict(env or os.environ)
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(popen_cwd) if popen_cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=run_env,
            start_new_session=True,
            bufsize=0,
        )
    except OSError as exc:
        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return None, str(exc)

    stdout_buf = CappedByteBuffer()
    stderr_buf = CappedByteBuffer()
    live = LiveShellProcess(
        proc=proc,
        sandboxed=sandboxed,
        command=command,
        cwd=cwd,
        started_at=time.perf_counter(),
        stdout_buf=stdout_buf,
        stderr_buf=stderr_buf,
        cleanup_paths=cleanup,
        hard_deadline=time.monotonic() + max(5.0, hard_timeout_sec),
    )
    if proc.stdout is not None:
        live._readers.append(_start_reader(proc.stdout, stdout_buf))
    if proc.stderr is not None:
        live._readers.append(_start_reader(proc.stderr, stderr_buf))
    return live, ""
