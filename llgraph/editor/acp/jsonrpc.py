"""ACP 传输层：stdio 上的换行分隔 JSON-RPC 2.0。"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Callable, TextIO

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
AUTH_REQUIRED = -32000

DEFERRED = object()
"""处理函数返回它表示「稍后自行回包」（如 session/prompt 交给工作线程）。"""


class JsonRpcError(Exception):
    """带 JSON-RPC 错误码的异常。"""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_payload(self) -> dict[str, Any]:
        """@return error 对象"""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return payload


def invalid_params(message: str) -> JsonRpcError:
    """@param message 说明 @return -32602 异常"""
    return JsonRpcError(INVALID_PARAMS, message)


class RequestFailed(Exception):
    """我们发出的请求没拿到结果（对端回了 error、超时、或连接已断）。"""


class RequestCancelled(Exception):
    """等对端回复期间本轮被取消。"""


class _Pending:
    """一条等着对端回复的出向请求。"""

    __slots__ = ("done", "result", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: Any = None
        self.error: str | None = None


class JsonRpcConnection:
    """
    一条 ndjson JSON-RPC 连接。

    写入端加锁：``session/prompt`` 在工作线程里跑，边跑边推 ``session/update``，
    与读循环的回包并发。

    也能反向发请求（``session/request_permission``）：请求由工作线程发出并阻塞等待，
    回包由读循环收下后唤醒它——所以等待期间读循环必须仍在跑，不能在这里读流。
    """

    def __init__(self, reader: TextIO, writer: TextIO) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._next_request_id = 0
        self._closed = False

    def _send(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            self._writer.write(line + "\n")
            self._writer.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """
        发送通知（无 id、不等回包）。

        @param method 方法名
        @param params 参数
        """
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def respond(self, request_id: Any, result: Any) -> None:
        """
        回成功包。

        @param request_id 请求 id
        @param result 结果
        """
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def respond_error(self, request_id: Any, error: JsonRpcError) -> None:
        """
        回错误包。

        @param request_id 请求 id（解析失败时为 None）
        @param error 错误
        """
        self._send({"jsonrpc": "2.0", "id": request_id, "error": error.to_payload()})

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
        poll_interval: float = 0.05,
    ) -> Any:
        """
        向对端（编辑器）发一条请求并等回复。

        @param method 方法名
        @param params 参数
        @param timeout 等待上限秒；None 表示一直等（人可能去泡咖啡了）
        @param cancel_check 返回 True 时放弃等待
        @param poll_interval 轮询 cancel_check 的间隔秒
        @return 对端的 result
        @raise RequestFailed 对端回 error / 超时 / 连接已断
        @raise RequestCancelled 等待期间被取消
        """
        with self._pending_lock:
            if self._closed:
                raise RequestFailed("连接已关闭")
            self._next_request_id += 1
            request_id = f"agent-{self._next_request_id}"
            pending = _Pending()
            self._pending[request_id] = pending
        try:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params if params is not None else {},
                }
            )
            deadline = None if timeout is None else time.monotonic() + timeout
            while not pending.done.wait(poll_interval):
                if cancel_check is not None and cancel_check():
                    raise RequestCancelled(f"{method} 等待期间被取消")
                if deadline is not None and time.monotonic() >= deadline:
                    raise RequestFailed(f"{method} 等待对端回复超时")
            if pending.error is not None:
                raise RequestFailed(pending.error)
            return pending.result
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def _resolve_pending(self, message: dict[str, Any]) -> None:
        """对端回包 → 唤醒对应的 ``request``。"""
        raw_id = message.get("id")
        if raw_id is None:
            return
        with self._pending_lock:
            pending = self._pending.get(str(raw_id))
        if pending is None:
            return
        error = message.get("error")
        if isinstance(error, dict):
            pending.error = str(error.get("message") or "对端返回错误") or "对端返回错误"
        elif error is not None:
            pending.error = str(error)
        else:
            pending.result = message.get("result")
        pending.done.set()

    def _fail_all_pending(self, reason: str) -> None:
        """连接断开时把等待中的请求一起放掉，否则工作线程会永远挂着。"""
        with self._pending_lock:
            self._closed = True
            pendings = list(self._pending.values())
        for pending in pendings:
            if not pending.done.is_set():
                pending.error = reason
                pending.done.set()

    def serve(
        self,
        dispatch: Callable[[str, dict[str, Any], Any], Any],
    ) -> None:
        """
        读循环：一行一条消息，直到对端关闭 stdin。

        @param dispatch ``(method, params, request_id) -> result | DEFERRED``
        """
        try:
            self._serve_loop(dispatch)
        finally:
            self._fail_all_pending("连接已关闭")

    def _serve_loop(
        self,
        dispatch: Callable[[str, dict[str, Any], Any], Any],
    ) -> None:
        for raw in self._reader:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self.respond_error(None, JsonRpcError(PARSE_ERROR, f"JSON 解析失败: {exc}"))
                continue
            if not isinstance(message, dict):
                self.respond_error(None, JsonRpcError(INVALID_REQUEST, "消息必须是 JSON 对象"))
                continue
            method = message.get("method")
            if not isinstance(method, str) or not method:
                self._resolve_pending(message)
                continue
            request_id = message.get("id")
            params = message.get("params")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                if request_id is not None:
                    self.respond_error(request_id, invalid_params("params 必须是对象"))
                continue
            try:
                result = dispatch(method, params, request_id)
            except JsonRpcError as exc:
                if request_id is not None:
                    self.respond_error(request_id, exc)
                continue
            except Exception as exc:  # 处理函数崩了也不能断连接
                if request_id is not None:
                    self.respond_error(request_id, JsonRpcError(INTERNAL_ERROR, str(exc)))
                else:
                    print(f"[acp] 通知 {method} 处理失败: {exc}", file=sys.stderr, flush=True)
                continue
            if request_id is None or result is DEFERRED:
                continue
            self.respond(request_id, result if result is not None else {})
