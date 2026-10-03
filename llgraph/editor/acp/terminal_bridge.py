"""ACP 终端反向请求：命令跑在**编辑器的终端**里，输出边跑边看。

``terminal/create`` / ``terminal/output`` / ``terminal/wait_for_exit`` /
``terminal/kill`` / ``terminal/release`` 都是 Agent → Client 方向，
与授权弹窗、``fs/*`` 走同一条反向链路：工作线程发出并阻塞等待，读循环收回包唤醒它。

客户端在 ``initialize`` 里声明 ``terminal`` 能力才有这条路；没声明时
``core/shell_terminal.py`` 那边拿不到来源，``run_shell_command`` 照旧本地起子进程。

编辑器答不上来不是致命错误——起不来就回落本地进程。但连着失败就别再问了：
每次失败都要等一个超时，一轮里几条命令能把对话拖死。
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from llgraph.core.shell_terminal import TerminalExit, TerminalSnapshot
from llgraph.editor.acp.jsonrpc import (
    JsonRpcConnection,
    RequestCancelled,
    RequestFailed,
)

_DEFAULT_TIMEOUT_SEC = 15.0
"""建终端 / 取输出这几步没有人参与（不像授权弹窗要等人点按钮），超时给短的。"""

_MAX_FAILURES = 3
"""连着这么多次失败就关掉这条路，本会话后面的命令只走本地进程。"""


def client_terminal_capability(raw: Any) -> bool:
    """
    读 ``initialize`` 里客户端声明的终端能力。

    @param raw clientCapabilities
    @return 是否能用 ``terminal/*``
    """
    if not isinstance(raw, dict):
        return False
    return bool(raw.get("terminal"))


def parse_exit_status(raw: Any) -> TerminalExit | None:
    """
    回包里的退出状态 → ``TerminalExit``。

    ``terminal/output`` 把它放在 ``exitStatus`` 下，``terminal/wait_for_exit``
    直接放在顶层，两种形状都认：拍错一种就会把跑完的命令当成还在跑。

    @param raw 回包（或其中的 exitStatus）
    @return 退出状态；还在跑 / 认不出来时 None
    """
    if not isinstance(raw, dict):
        return None
    status = raw.get("exitStatus")
    if isinstance(status, dict):
        raw = status
    elif "exitCode" not in raw and "signal" not in raw:
        return None
    code = raw.get("exitCode")
    signal = raw.get("signal")
    exit_code = (
        int(code) if isinstance(code, (int, float)) and not isinstance(code, bool) else None
    )
    return TerminalExit(
        exit_code=exit_code,
        signal=signal.strip() if isinstance(signal, str) and signal.strip() else None,
    )


class AcpTerminalBridge:
    """一个 ACP 会话的编辑器终端（`core.shell_terminal.EditorTerminalSource`）。"""

    def __init__(
        self,
        connection: JsonRpcConnection,
        session_id: str,
        *,
        cancel_check: Callable[[], bool] | None = None,
        timeout_sec: float | None = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._conn = connection
        self._session_id = session_id
        self._cancel_check = cancel_check
        self._timeout_sec = timeout_sec
        self._failures = 0
        self.disabled = False

    # ---- 来源接口 ----

    def create(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str,
        output_byte_limit: int,
    ) -> str | None:
        """
        @param command 可执行文件（``/bin/sh``）
        @param args 参数（``["-c", 命令串]``）
        @param cwd 绝对工作目录
        @param output_byte_limit 编辑器为这个终端保留的输出字节上限
        @return 终端 id；没这个能力 / 已熔断 / 编辑器开不出来时 None
        """
        if self.disabled:
            return None
        if self._cancel_check is not None and self._cancel_check():
            return None
        params: dict[str, Any] = {
            "sessionId": self._session_id,
            "command": command,
            "args": list(args),
            "cwd": cwd,
            "outputByteLimit": int(output_byte_limit),
        }
        try:
            result = self._conn.request(
                "terminal/create",
                params,
                timeout=self._timeout_sec,
                cancel_check=self._cancel_check,
            )
        except RequestCancelled:
            return None
        except RequestFailed as exc:
            self._note_failure("terminal/create", str(exc))
            return None
        terminal_id = result.get("terminalId") if isinstance(result, dict) else None
        if not isinstance(terminal_id, str) or not terminal_id.strip():
            self._note_failure("terminal/create", "回包里没有 terminalId 字段")
            return None
        self._failures = 0
        return terminal_id.strip()

    def output(self, terminal_id: str) -> TerminalSnapshot | None:
        """
        @param terminal_id 终端 id
        @return 当前输出与退出状态；问不到时 None

        这一步**不看取消标记**：用户点了停止之后，模型仍然需要看到命令跑出了什么，
        否则下一轮它只知道「被打断了」，不知道打断前发生了什么。
        """
        try:
            result = self._conn.request(
                "terminal/output",
                {"sessionId": self._session_id, "terminalId": terminal_id},
                timeout=self._timeout_sec,
            )
        except RequestCancelled:
            return None
        except RequestFailed as exc:
            self._note_failure("terminal/output", str(exc))
            return None
        if not isinstance(result, dict):
            self._note_failure("terminal/output", "回包不是对象")
            return None
        output = result.get("output")
        if not isinstance(output, str):
            self._note_failure("terminal/output", "回包里没有 output 字段")
            return None
        self._failures = 0
        return TerminalSnapshot(
            output=output,
            truncated=bool(result.get("truncated")),
            exit=parse_exit_status(result),
        )

    def wait_for_exit(self, terminal_id: str) -> TerminalExit | None:
        """
        @param terminal_id 终端 id
        @return 退出状态；问不到时 None

        这条请求一直挂着不设超时：命令可能跑一小时，设了超时只会让我们误判成
        「失联」。取消与硬超时走 ``terminal/kill``——杀掉命令，编辑器自然回退出状态。
        连接断开时 ``jsonrpc`` 会把挂着的请求一起放掉，不至于永远等下去。
        """
        try:
            result = self._conn.request(
                "terminal/wait_for_exit",
                {"sessionId": self._session_id, "terminalId": terminal_id},
                timeout=None,
            )
        except RequestCancelled:
            return None
        except RequestFailed as exc:
            self._note_failure("terminal/wait_for_exit", str(exc))
            return None
        status = parse_exit_status(result)
        if status is None:
            self._note_failure("terminal/wait_for_exit", "回包里没有退出状态")
            return None
        self._failures = 0
        return status

    def kill(self, terminal_id: str) -> None:
        """@param terminal_id 终端 id"""
        self._fire("terminal/kill", terminal_id)

    def release(self, terminal_id: str) -> None:
        """
        @param terminal_id 终端 id

        放掉之后编辑器仍继续显示那段输出（ACP 如此约定），所以收尾取完最终输出
        就该放：不放会让编辑器一直替我们留着终端。
        """
        self._fire("terminal/release", terminal_id)

    # ---- 内部 ----

    def _fire(self, method: str, terminal_id: str) -> None:
        """@param method 反向请求名 @param terminal_id 终端 id"""
        try:
            self._conn.request(
                method,
                {"sessionId": self._session_id, "terminalId": terminal_id},
                timeout=self._timeout_sec,
            )
        except (RequestCancelled, RequestFailed) as exc:
            # 杀不动 / 放不掉只影响编辑器那一侧，不该把工具结果带崩
            print(f"[acp] {method} 失败: {exc}", file=sys.stderr, flush=True)

    def _note_failure(self, method: str, reason: str) -> None:
        """@param method 反向请求名 @param reason 失败说明"""
        self._failures += 1
        print(f"[acp] {method} 失败: {reason}", file=sys.stderr, flush=True)
        if self._failures >= _MAX_FAILURES and not self.disabled:
            self.disabled = True
            print(
                f"[acp] {method} 连续失败 {self._failures} 次，本会话的命令改在本地跑。",
                file=sys.stderr,
                flush=True,
            )
