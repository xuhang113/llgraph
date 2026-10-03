"""命令跑在编辑器的终端里：输出边跑边看。

``run_shell_command`` 自己起子进程时，输出要等命令跑完才一次性回填——编辑器里
一条 `pytest -q` 的几分钟内只有一行「执行中」，看不到滚动的日志。ACP 客户端声明
``terminal`` 能力时就多了一条执行路径：命令交给编辑器开一个终端去跑，编辑器自己
渲染实时输出，我们只在需要文本时（跑完、或 ``await_shell`` 看一眼）取一次回来。

来源放在 ContextVar 上，理由与 `core/editor_fs.py` 的文件来源一样：工具可能在
LangGraph 的线程池里跑，全局变量会在多会话同进程时串台。
没有入口登记来源时 ``spawn_editor_terminal`` 一律返回 None——CLI / Web Console
的行为因此完全不变。

命令能不能跑（``permissions.shell`` 的拦截、写模式下的授权闸门）仍然全部在交出去
**之前**判完：编辑器只负责跑和显示，不负责「这条命令该不该跑」。
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

_SHELL_PATH = "/bin/sh"
"""与 ``sandbox.exec.build_shell_argv`` 同一个解释器：命令串里有管道和 && ，必须 -c 起。"""

_KILL_GRACE_SEC = 2.0
"""杀完等编辑器回退出状态的宽限；它不回也不能让这条 job 永远算在运行中。"""

_MIN_OUTPUT_BYTE_LIMIT = 131_072
_MAX_OUTPUT_BYTE_LIMIT = 1_048_576

_TRUNCATED_NOTE = "…(编辑器终端只保留了最近的输出)\n"
_LOST_NOTE = "\n[llgraph] 编辑器终端失联，上面是失联前捕获的输出。"


@dataclass(frozen=True)
class TerminalExit:
    """一条编辑器终端命令的退出状态。"""

    exit_code: int | None = None
    signal: str | None = None


@dataclass(frozen=True)
class TerminalSnapshot:
    """``terminal/output`` 的一次快照。"""

    output: str = ""
    truncated: bool = False
    exit: TerminalExit | None = None
    """None 表示还在跑。"""


@runtime_checkable
class EditorTerminalSource(Protocol):
    """编辑器侧的终端（ACP 下是 ``terminal/*`` 五条反向请求）。"""

    def create(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str,
        output_byte_limit: int,
    ) -> str | None:
        """@return 终端 id；编辑器开不出来时 None"""

    def output(self, terminal_id: str) -> TerminalSnapshot | None:
        """@param terminal_id 终端 id @return 当前输出；问不到时 None"""

    def wait_for_exit(self, terminal_id: str) -> TerminalExit | None:
        """@param terminal_id 终端 id @return 退出状态（阻塞到退出）；问不到时 None"""

    def kill(self, terminal_id: str) -> None:
        """@param terminal_id 终端 id"""

    def release(self, terminal_id: str) -> None:
        """@param terminal_id 终端 id"""


_terminal_source: ContextVar[EditorTerminalSource | None] = ContextVar(
    "llgraph_editor_terminal", default=None
)


def set_editor_terminal_source(source: EditorTerminalSource | None) -> Token:
    """
    登记编辑器终端来源。

    @param source 来源；None 表示本地起子进程
    @return ContextVar token，交给 ``reset_editor_terminal_source``
    """
    return _terminal_source.set(source)


def reset_editor_terminal_source(token: Token) -> None:
    """
    还原上一层来源。

    @param token ``set_editor_terminal_source`` 的返回值
    """
    _terminal_source.reset(token)


@contextmanager
def use_editor_terminal_source(
    source: EditorTerminalSource | None,
) -> Iterator[None]:
    """
    在一段执行期间登记来源（一轮 invoke 外面套一层）。

    @param source 来源；None 时等于什么都不做
    """
    token = set_editor_terminal_source(source)
    try:
        yield
    finally:
        reset_editor_terminal_source(token)


def current_editor_terminal_source() -> EditorTerminalSource | None:
    """@return 当前来源；无则 None"""
    return _terminal_source.get()


def terminal_output_byte_limit(max_output_chars: int) -> int:
    """
    编辑器终端该留多少输出。

    这个上限同时管着**编辑器里能往上翻多少**（超出后它从头丢），所以比模型可见的
    ``max_output_chars`` 给得宽：人往回翻日志比模型读得多。但也不能无上限——
    跑完取一次要整段过 stdio。

    @param max_output_chars 模型可见的输出上限（字符）
    @return 字节上限
    """
    try:
        chars = int(max_output_chars)
    except (TypeError, ValueError):
        chars = 0
    return max(_MIN_OUTPUT_BYTE_LIMIT, min(chars * 4, _MAX_OUTPUT_BYTE_LIMIT))


class EditorTerminalProcess:
    """
    编辑器终端里的那条命令，对外长得和 ``LiveShellProcess`` 一样。

    shell 工具只通过 ``snapshot_stdio`` / ``returncode`` / ``elapsed_sec`` /
    ``wait`` / ``kill`` 几个口子用进程，所以换一条执行路径不必动结果格式化、
    后台 job 表与 ``await_shell``——它们一行都没改。

    退出状态走 ``terminal/wait_for_exit``（一条挂着的反向请求，由后台线程等），
    输出则**按需**取：编辑器自己在渲染实时输出，我们没必要边跑边把同一段文本
    一遍遍搬过来。
    """

    def __init__(
        self,
        source: EditorTerminalSource,
        terminal_id: str,
        *,
        command: str,
        cwd: Path,
        hard_timeout_sec: float = 1800.0,
    ) -> None:
        """
        @param source 编辑器终端来源
        @param terminal_id ``terminal/create`` 给的 id
        @param command 命令（原始 shell 串，仅用于留档）
        @param cwd 工作目录
        @param hard_timeout_sec 硬超时（后台任务上限）
        """
        self.terminal_id = terminal_id
        self.command = command
        self.cwd = cwd
        self.sandboxed = False
        self.error: str | None = None
        self.started_at = time.perf_counter()
        self.hard_deadline = time.monotonic() + max(5.0, hard_timeout_sec)
        self._source = source
        self._lock = threading.Lock()
        self._exited = threading.Event()
        self._exit: TerminalExit | None = None
        self._final_output: str | None = None
        self._released = False
        self._lost = False
        self._waiter = threading.Thread(
            target=self._wait_for_exit,
            name="llgraph-editor-terminal",
            daemon=True,
        )
        self._waiter.start()

    # ---- LiveShellProcess 同形接口 ----

    def snapshot_stdio(self) -> tuple[str, str]:
        """
        当前输出。

        编辑器终端把 stdout / stderr 合在一条流里，所以 stderr 一律空串——
        调用方走的是 ``combine_stdio``，空串原样返回前一段。

        已经退出过一次之后不再问编辑器：收尾时就把最终输出留下来了，
        而终端此时已经 ``terminal/release``（编辑器仍继续显示那段输出）。

        @return (合并输出, "")
        """
        with self._lock:
            cached = self._final_output
        if cached is not None:
            return cached, ""

        snapshot: TerminalSnapshot | None
        try:
            snapshot = self._source.output(self.terminal_id)
        except Exception:  # noqa: BLE001 - 编辑器侧出问题不能升级成工具崩溃
            snapshot = None
        if snapshot is None:
            self._lost = True
            text = ""
        else:
            text = snapshot.output or ""
            if snapshot.truncated:
                text = _TRUNCATED_NOTE + text
            if snapshot.exit is not None:
                self._note_exit(snapshot.exit)
        if self._lost:
            text += _LOST_NOTE
        if self._exited.is_set():
            with self._lock:
                if self._final_output is None:
                    self._final_output = text
            self._release()
        return text, ""

    def returncode(self) -> int | None:
        """@return 退出码；仍在运行则 None。被信号打死或编辑器没给码时为 -1"""
        if not self._exited.is_set():
            return None
        with self._lock:
            status = self._exit
        if status is None or status.exit_code is None:
            return -1
        return int(status.exit_code)

    def elapsed_sec(self) -> float:
        """@return 已运行秒数"""
        return max(0.0, time.perf_counter() - self.started_at)

    def wait(
        self,
        timeout_sec: float | None,
        *,
        cancel_check: object = None,
    ) -> bool:
        """
        等到退出、取消、硬超时或调用方等待到期。

        @param timeout_sec 本次等待上限；None 表示直到退出/硬超时/取消
        @param cancel_check 返回 True 时杀掉终端里的命令
        @return 命令是否已退出（含被杀）
        """
        deadline = (
            None if timeout_sec is None else time.monotonic() + max(0.0, timeout_sec)
        )
        while not self._exited.is_set():
            if callable(cancel_check) and cancel_check():
                self.kill("cancelled")
                return True
            if self.hard_deadline and time.monotonic() >= self.hard_deadline:
                self.kill("timeout")
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            self._exited.wait(0.05)
        return True

    def kill(self, reason: str) -> None:
        """
        让编辑器终止这条命令并记录原因（幂等）。

        @param reason timeout | cancelled | 其它
        """
        with self._lock:
            if self.error is None:
                self.error = reason
        if self._exited.is_set():
            return
        try:
            self._source.kill(self.terminal_id)
        except Exception:  # noqa: BLE001 - 杀不动也要把这条 job 收掉
            pass
        if not self._exited.wait(_KILL_GRACE_SEC):
            self._note_exit(TerminalExit())

    # ---- 内部 ----

    def _wait_for_exit(self) -> None:
        """后台等退出状态：一条挂着的 ``terminal/wait_for_exit``。"""
        try:
            status = self._source.wait_for_exit(self.terminal_id)
        except Exception:  # noqa: BLE001 - 等不到就按失联收场
            status = None
        if status is None:
            self._lost = True
            with self._lock:
                if self.error is None:
                    self.error = "terminal-lost"
        self._note_exit(status if status is not None else TerminalExit())

    def _note_exit(self, status: TerminalExit) -> None:
        with self._lock:
            if self._exited.is_set():
                return
            self._exit = status
        self._exited.set()

    def _release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        try:
            self._source.release(self.terminal_id)
        except Exception:  # noqa: BLE001 - 放不掉只是编辑器多留一个终端
            pass


def spawn_editor_terminal(
    *,
    command: str,
    cwd: Path,
    output_byte_limit: int,
    hard_timeout_sec: float = 1800.0,
) -> EditorTerminalProcess | None:
    """
    把命令交给编辑器的终端跑。

    建好之后顺手报一次（``tool_progress.notify_terminal_created``）：入口据此把
    终端挂到当前那条 ``tool_call`` 上，编辑器里那一行才会展开成实时终端。

    @param command shell 命令（单条）
    @param cwd 绝对工作目录
    @param output_byte_limit 编辑器为这个终端保留的输出字节上限
    @param hard_timeout_sec 硬超时（后台任务上限）
    @return 进程；没有编辑器终端 / 开不出来时 None（调用方回落本地子进程）
    """
    from llgraph.core.tool_progress import notify_terminal_created

    source = _terminal_source.get()
    if source is None:
        return None
    try:
        raw_id = source.create(
            command=_SHELL_PATH,
            args=["-c", command],
            cwd=str(cwd),
            output_byte_limit=int(output_byte_limit),
        )
    except Exception:  # noqa: BLE001 - 开不出来就回落本地进程
        return None
    terminal_id = str(raw_id or "").strip()
    if not terminal_id:
        return None
    notify_terminal_created(terminal_id)
    return EditorTerminalProcess(
        source,
        terminal_id,
        command=command,
        cwd=cwd,
        hard_timeout_sec=hard_timeout_sec,
    )
