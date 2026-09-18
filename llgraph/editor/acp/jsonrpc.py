"""ACP 传输层：stdio 上的换行分隔 JSON-RPC 2.0。"""

from __future__ import annotations

import json
import sys
import threading
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


class JsonRpcConnection:
    """
    一条 ndjson JSON-RPC 连接。

    写入端加锁：``session/prompt`` 在工作线程里跑，边跑边推 ``session/update``，
    与读循环的回包并发。
    """

    def __init__(self, reader: TextIO, writer: TextIO) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = threading.Lock()

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

    def serve(
        self,
        dispatch: Callable[[str, dict[str, Any], Any], Any],
    ) -> None:
        """
        读循环：一行一条消息，直到对端关闭 stdin。

        @param dispatch ``(method, params, request_id) -> result | DEFERRED``
        """
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
                # 回包（对端响应我们发出的请求）：本轮不主动发请求，忽略即可
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
