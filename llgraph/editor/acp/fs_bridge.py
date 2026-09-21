"""ACP 文件反向请求：读写**编辑器里**那份文件（含未保存的缓冲区）。

``fs/read_text_file`` / ``fs/write_text_file`` 是 Agent → Client 方向，
与授权弹窗走同一条反向链路：工作线程发出并阻塞等待，读循环收回包唤醒它。

只对**工作区内**的路径问编辑器。客户端只认自己打开的项目，
``~/.llgraph/skills`` 这类外部读仍走磁盘；工作区外的路径问过去也只会被拒。

编辑器答不上来不是致命错误——调用方一律回落磁盘。但连着失败就别再问了：
每次失败都要等一个超时，一轮里十几次读文件会把对话拖死。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from llgraph.editor.acp.jsonrpc import (
    JsonRpcConnection,
    RequestCancelled,
    RequestFailed,
)

_DEFAULT_TIMEOUT_SEC = 15.0
"""编辑器这一步没有人参与（不像授权弹窗要等人点按钮），超时给短的。"""

_MAX_FAILURES = 3
"""连着这么多次失败就关掉这条来源，本会话后面只走磁盘。"""


def client_fs_capabilities(raw: Any) -> tuple[bool, bool]:
    """
    读 ``initialize`` 里客户端声明的文件能力。

    @param raw clientCapabilities
    @return (能读 fs/read_text_file, 能写 fs/write_text_file)
    """
    fs = raw.get("fs") if isinstance(raw, dict) else None
    if not isinstance(fs, dict):
        return False, False
    return bool(fs.get("readTextFile")), bool(fs.get("writeTextFile"))


class AcpFileBridge:
    """一个 ACP 会话的编辑器文件来源（`core.editor_fs.EditorFileSource`）。"""

    def __init__(
        self,
        connection: JsonRpcConnection,
        session_id: str,
        *,
        workspace: Path,
        can_read: bool = True,
        can_write: bool = True,
        cancel_check: Callable[[], bool] | None = None,
        timeout_sec: float | None = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._conn = connection
        self._session_id = session_id
        self._workspace = Path(workspace).resolve()
        self._can_read = can_read
        self._can_write = can_write
        self._cancel_check = cancel_check
        self._timeout_sec = timeout_sec
        self._failures = 0
        self.disabled = False

    # ---- 来源接口 ----

    def read_text(self, path: Path) -> str | None:
        """
        @param path 绝对路径
        @return 编辑器里的正文；没这个能力 / 路径在工作区外 / 编辑器答不上来时 None
        """
        target = self._payload_path(path) if self._usable(self._can_read) else None
        if target is None:
            return None
        try:
            result = self._conn.request(
                "fs/read_text_file",
                {"sessionId": self._session_id, "path": target},
                timeout=self._timeout_sec,
                cancel_check=self._cancel_check,
            )
        except RequestCancelled:
            return None
        except RequestFailed as exc:
            self._note_failure("fs/read_text_file", str(exc))
            return None
        content = result.get("content") if isinstance(result, dict) else None
        if not isinstance(content, str):
            self._note_failure("fs/read_text_file", "回包里没有 content 字段")
            return None
        self._failures = 0
        return content

    def write_text(self, path: Path, text: str) -> bool:
        """
        @param path 绝对路径
        @param text 全量正文
        @return 是否已由编辑器写下；False 时调用方自己落盘
        """
        target = self._payload_path(path) if self._usable(self._can_write) else None
        if target is None:
            return False
        try:
            self._conn.request(
                "fs/write_text_file",
                {
                    "sessionId": self._session_id,
                    "path": target,
                    "content": text,
                },
                timeout=self._timeout_sec,
                cancel_check=self._cancel_check,
            )
        except RequestCancelled:
            return False
        except RequestFailed as exc:
            self._note_failure("fs/write_text_file", str(exc))
            return False
        self._failures = 0
        return True

    # ---- 内部 ----

    def _usable(self, capability: bool) -> bool:
        """@param capability 客户端是否声明了这个方法 @return 这次还要不要问编辑器"""
        if not capability or self.disabled:
            return False
        return not (self._cancel_check is not None and self._cancel_check())

    def _payload_path(self, path: Path) -> str | None:
        """
        @param path 工具解析出的路径
        @return ACP 要的绝对路径；工作区外返回 None
        """
        try:
            candidate = Path(path).resolve()
        except OSError:
            return None
        try:
            candidate.relative_to(self._workspace)
        except ValueError:
            return None
        return str(candidate)

    def _note_failure(self, method: str, reason: str) -> None:
        """@param method 反向请求名 @param reason 失败说明"""
        self._failures += 1
        print(f"[acp] {method} 失败，本次回落磁盘: {reason}", file=sys.stderr, flush=True)
        if self._failures >= _MAX_FAILURES and not self.disabled:
            self.disabled = True
            print(
                f"[acp] {method} 连续失败 {self._failures} 次，本会话不再读写编辑器缓冲区。",
                file=sys.stderr,
                flush=True,
            )
