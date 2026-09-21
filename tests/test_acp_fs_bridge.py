"""编辑器缓冲区：文件工具的第二条来源 + ACP 的 ``fs/*`` 反向请求。

分四层测：
1. `core/editor_fs.py` 的来源语义（没登记来源时 CLI / Web 行为不变）
2. 文件工具真的以缓冲区为准，并把脏文件的写入交回编辑器
3. ACP 侧的桥：载荷、工作区边界、失败回落与熔断
4. 协议层：真 pipe 上走一趟 ``fs/read_text_file``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llgraph.core.editor_fs import (
    current_editor_file_source,
    editor_buffer_text,
    editor_write_text,
    use_editor_file_source,
)
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.workspace import WorkspaceContext
from llgraph.editor.acp.fs_bridge import AcpFileBridge, client_fs_capabilities
from llgraph.editor.acp.jsonrpc import RequestCancelled, RequestFailed
from llgraph.editor.acp.turn import AcpTurnRequest, AcpTurnResult

from tests.test_acp_server import _Harness

_DISK = "def run():\n    return 1\n"
_BUFFER = "def run():\n    return 1  # 用户刚改的，还没保存\n"


class _FakeEditor:
    """假编辑器：缓冲区放在字典里。"""

    def __init__(self, buffers: dict[Path, str], *, write_ok: bool = True) -> None:
        self.buffers = {Path(k): v for k, v in buffers.items()}
        self.write_ok = write_ok
        self.reads: list[Path] = []
        self.writes: list[tuple[Path, str]] = []

    def read_text(self, path: Path) -> str | None:
        self.reads.append(Path(path))
        return self.buffers.get(Path(path))

    def write_text(self, path: Path, text: str) -> bool:
        self.writes.append((Path(path), text))
        if not self.write_ok:
            return False
        self.buffers[Path(path)] = text
        return True


def _fs_tool(root: Path, name: str, **ctx_kwargs: Any):
    ctx = WorkspaceContext(root, allow_write=True, **ctx_kwargs)
    return next(t for t in create_filesystem_tools(ctx) if t.name == name)


@pytest.fixture
def dirty(tmp_path: Path) -> tuple[Path, _FakeEditor]:
    """磁盘一份、编辑器缓冲区另一份的工作区。"""
    target = tmp_path / "app.py"
    target.write_text(_DISK, encoding="utf-8")
    return target, _FakeEditor({target: _BUFFER})


# ---- 来源语义 ----


def test_no_source_means_disk_only(tmp_path: Path) -> None:
    """CLI / Web 不登记来源：读写不能因为本轮改造绕去别处。"""
    assert current_editor_file_source() is None
    assert editor_buffer_text(tmp_path / "a.py") is None
    assert editor_write_text(tmp_path / "a.py", "x") is False


def test_source_is_scoped_to_the_with_block(tmp_path: Path) -> None:
    editor = _FakeEditor({tmp_path / "a.py": "buffered\n"})
    with use_editor_file_source(editor):
        assert editor_buffer_text(tmp_path / "a.py") == "buffered\n"
    assert editor_buffer_text(tmp_path / "a.py") is None


def test_source_exception_falls_back_to_disk(tmp_path: Path) -> None:
    """编辑器侧炸了只能少一条来源，不能让工具报错。"""

    class _Boom:
        def read_text(self, path: Path) -> str | None:
            raise RuntimeError("编辑器没了")

        def write_text(self, path: Path, text: str) -> bool:
            raise RuntimeError("编辑器没了")

    with use_editor_file_source(_Boom()):
        assert editor_buffer_text(tmp_path / "a.py") is None
        assert editor_write_text(tmp_path / "a.py", "x") is False


def test_garbage_reply_is_ignored(tmp_path: Path) -> None:
    class _Garbage:
        def read_text(self, path: Path) -> Any:
            return 42

        def write_text(self, path: Path, text: str) -> bool:
            return True

    with use_editor_file_source(_Garbage()):
        assert editor_buffer_text(tmp_path / "a.py") is None


# ---- 文件工具 ----


def test_read_file_prefers_unsaved_buffer(dirty: tuple[Path, _FakeEditor]) -> None:
    target, editor = dirty
    tool = _fs_tool(target.parent, "read_file")
    with use_editor_file_source(editor):
        out = tool.invoke({"path": "app.py"})
    assert "还没保存" in out
    assert "未保存" in out.splitlines()[0]
    assert editor.reads == [target]


def test_read_file_without_unsaved_changes_says_nothing(tmp_path: Path) -> None:
    """缓冲区与磁盘一致时不该每次读都挂一句提示。"""
    target = tmp_path / "app.py"
    target.write_text(_DISK, encoding="utf-8")
    tool = _fs_tool(tmp_path, "read_file")
    with use_editor_file_source(_FakeEditor({target: _DISK})):
        out = tool.invoke({"path": "app.py"})
    assert "未保存" not in out
    assert "return 1" in out


def test_read_files_batch_also_prefers_buffer(dirty: tuple[Path, _FakeEditor]) -> None:
    target, editor = dirty
    tool = _fs_tool(target.parent, "read_files")
    with use_editor_file_source(editor):
        out = tool.invoke({"paths": ["app.py"]})
    assert "还没保存" in out


def test_oversized_buffer_is_rejected(tmp_path: Path) -> None:
    """磁盘 stat 拦不住缓冲区：未保存那份可能大得多。"""
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = _FakeEditor({target: "y = 2\n" * 20_000})
    tool = _fs_tool(tmp_path, "read_file", max_read_bytes=50_000)
    with use_editor_file_source(editor):
        out = tool.invoke({"path": "app.py"})
    assert "过大" in out
    assert "保存" in out


def test_search_replace_applies_to_buffer_and_writes_back_to_editor(
    dirty: tuple[Path, _FakeEditor],
) -> None:
    """脏文件必须由编辑器落地：我们自己写磁盘，用户按一次保存就盖回去了。"""
    target, editor = dirty
    tool = _fs_tool(target.parent, "search_replace")
    with use_editor_file_source(editor):
        out = tool.invoke(
            {"path": "app.py", "old_string": "return 1", "new_string": "return 2"}
        )
    assert "交回编辑器" in out
    assert len(editor.writes) == 1
    written = editor.writes[0][1]
    # 改的是缓冲区那份：用户没保存的注释还在
    assert "还没保存" in written
    assert "return 2" in written
    assert target.read_text(encoding="utf-8") == _DISK


def test_search_replace_falls_back_to_disk_when_editor_refuses(
    tmp_path: Path,
) -> None:
    target = tmp_path / "app.py"
    target.write_text(_DISK, encoding="utf-8")
    editor = _FakeEditor({target: _BUFFER}, write_ok=False)
    tool = _fs_tool(tmp_path, "search_replace")
    with use_editor_file_source(editor):
        out = tool.invoke(
            {"path": "app.py", "old_string": "return 1", "new_string": "return 2"}
        )
    assert "已直接落盘" in out
    disk = target.read_text(encoding="utf-8")
    assert "return 2" in disk
    assert "还没保存" in disk


def test_clean_file_still_lands_on_disk(tmp_path: Path) -> None:
    """没有未保存改动的文件照旧原子落盘，不绕编辑器。"""
    target = tmp_path / "app.py"
    target.write_text(_DISK, encoding="utf-8")
    editor = _FakeEditor({target: _DISK})
    tool = _fs_tool(tmp_path, "search_replace")
    with use_editor_file_source(editor):
        out = tool.invoke(
            {"path": "app.py", "old_string": "return 1", "new_string": "return 2"}
        )
    assert "编辑器" not in out
    assert editor.writes == []
    assert "return 2" in target.read_text(encoding="utf-8")


def test_new_file_is_created_on_disk(tmp_path: Path) -> None:
    """编辑器里没有这个文件，写入不该指望它。"""
    editor = _FakeEditor({})
    tool = _fs_tool(tmp_path, "write_file")
    with use_editor_file_source(editor):
        tool.invoke({"path": "new.py", "content": "print(1)\n"})
    assert editor.writes == []
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print(1)\n"


def test_write_file_overwriting_dirty_file_goes_through_editor(
    dirty: tuple[Path, _FakeEditor],
) -> None:
    target, editor = dirty
    tool = _fs_tool(target.parent, "write_file")
    with use_editor_file_source(editor):
        out = tool.invoke({"path": "app.py", "content": "print(2)\n"})
    assert "交回编辑器" in out
    assert editor.writes == [(target, "print(2)\n")]
    assert target.read_text(encoding="utf-8") == _DISK


def test_append_file_appends_to_the_buffer(dirty: tuple[Path, _FakeEditor]) -> None:
    target, editor = dirty
    tool = _fs_tool(target.parent, "append_file")
    with use_editor_file_source(editor):
        out = tool.invoke({"path": "app.py", "content": "# 尾巴\n"})
    assert "交回编辑器" in out
    written = editor.writes[0][1]
    assert written == _BUFFER + "# 尾巴\n"


# ---- ACP 桥 ----


class _FakeConn:
    """只实现 ``request``：按脚本回 result 或抛异常。"""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def request(self, method: str, params: dict[str, Any], **_kwargs: Any) -> Any:
        self.sent.append((method, params))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _bridge(
    replies: list[Any],
    workspace: Path,
    **kwargs: Any,
) -> tuple[AcpFileBridge, _FakeConn]:
    conn = _FakeConn(replies)
    bridge = AcpFileBridge(
        conn,  # type: ignore[arg-type]
        "cli-fs",
        workspace=workspace,
        timeout_sec=1.0,
        **kwargs,
    )
    return bridge, conn


def test_capabilities_parsing() -> None:
    assert client_fs_capabilities(None) == (False, False)
    assert client_fs_capabilities({}) == (False, False)
    assert client_fs_capabilities({"fs": {}}) == (False, False)
    assert client_fs_capabilities({"fs": {"readTextFile": True}}) == (True, False)
    assert client_fs_capabilities(
        {"fs": {"readTextFile": True, "writeTextFile": True}}
    ) == (True, True)


def test_read_payload_and_content(tmp_path: Path) -> None:
    bridge, conn = _bridge([{"content": _BUFFER}], tmp_path)
    assert bridge.read_text(tmp_path / "app.py") == _BUFFER
    method, params = conn.sent[0]
    assert method == "fs/read_text_file"
    assert params["sessionId"] == "cli-fs"
    assert params["path"] == str(tmp_path / "app.py")


def test_write_payload(tmp_path: Path) -> None:
    bridge, conn = _bridge([None], tmp_path)
    assert bridge.write_text(tmp_path / "app.py", "x\n") is True
    method, params = conn.sent[0]
    assert method == "fs/write_text_file"
    assert params["path"] == str(tmp_path / "app.py")
    assert params["content"] == "x\n"


def test_paths_outside_the_workspace_are_not_asked(tmp_path: Path) -> None:
    """``~/.llgraph/skills`` 这类外部读走磁盘，编辑器不认。"""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    bridge, conn = _bridge([], workspace)
    assert bridge.read_text(tmp_path / "outside.md") is None
    assert bridge.write_text(tmp_path / "outside.md", "x") is False
    assert conn.sent == []


def test_missing_capability_is_not_asked(tmp_path: Path) -> None:
    bridge, conn = _bridge([], tmp_path, can_read=False, can_write=False)
    assert bridge.read_text(tmp_path / "app.py") is None
    assert bridge.write_text(tmp_path / "app.py", "x") is False
    assert conn.sent == []


def test_garbage_read_reply_is_none(tmp_path: Path) -> None:
    bridge, _conn = _bridge([{"text": "没按协议来"}], tmp_path)
    assert bridge.read_text(tmp_path / "app.py") is None


def test_repeated_failures_disable_the_bridge(tmp_path: Path) -> None:
    """每次失败都要等一个超时，连着失败就别再问了。"""
    bridge, conn = _bridge([RequestFailed("超时")] * 3, tmp_path)
    for _ in range(3):
        assert bridge.read_text(tmp_path / "app.py") is None
    assert bridge.disabled is True
    assert len(conn.sent) == 3

    conn.replies = [{"content": "x"}]
    assert bridge.read_text(tmp_path / "app.py") is None
    assert bridge.write_text(tmp_path / "app.py", "x") is False
    assert len(conn.sent) == 3


def test_one_success_resets_the_failure_count(tmp_path: Path) -> None:
    bridge, _conn = _bridge(
        [RequestFailed("抖了一下"), {"content": "x"}, RequestFailed("又抖了")],
        tmp_path,
    )
    assert bridge.read_text(tmp_path / "app.py") is None
    assert bridge.read_text(tmp_path / "app.py") == "x"
    assert bridge.read_text(tmp_path / "app.py") is None
    assert bridge.disabled is False


def test_cancelled_turn_stops_asking(tmp_path: Path) -> None:
    bridge, conn = _bridge([], tmp_path, cancel_check=lambda: True)
    assert bridge.read_text(tmp_path / "app.py") is None
    assert conn.sent == []

    # 等待期间被取消：不算编辑器的错，不计进熔断
    bridge2, _c2 = _bridge([RequestCancelled("停止")], tmp_path)
    assert bridge2.read_text(tmp_path / "app.py") is None
    assert bridge2.disabled is False


# ---- 协议层：编辑器与 Agent 之间真走一趟 ----


def _read_runner(seen: list[Any], path: Path):
    """turn_runner 桩件：跑一轮里读一次文件，把读到的正文写进 result.text。"""

    def runner(req: AcpTurnRequest, *, send_update: Any, cancel_check: Any) -> AcpTurnResult:
        seen.append(req.editor_files)
        if req.editor_files is None:
            return AcpTurnResult(text="没有编辑器来源")
        return AcpTurnResult(text=req.editor_files.read_text(path) or "读不到")

    return runner


def _session_with_caps(h: _Harness, workspace: Path, caps: dict[str, Any]) -> str:
    h.send("initialize", {"protocolVersion": 1, "clientCapabilities": caps}, request_id=1)
    h.response(1)
    h.send("session/new", {"cwd": str(workspace), "mcpServers": []}, request_id=2)
    return h.response(2)["result"]["sessionId"]


def test_editor_gets_fs_read_request_and_content_reaches_the_tool(
    tmp_path: Path,
) -> None:
    target = tmp_path / "app.py"
    target.write_text(_DISK, encoding="utf-8")
    seen: list[Any] = []
    h = _Harness(turn_runner=_read_runner(seen, target))
    try:
        session_id = _session_with_caps(
            h, tmp_path, {"fs": {"readTextFile": True, "writeTextFile": True}}
        )
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "看一眼"}]},
            request_id=3,
        )
        ask = h.out.wait_for(lambda m: m.get("method") == "fs/read_text_file")
        assert ask["params"]["sessionId"] == session_id
        assert ask["params"]["path"] == str(target)

        h.send_raw(
            json.dumps(
                {"jsonrpc": "2.0", "id": ask["id"], "result": {"content": _BUFFER}}
            )
        )
        assert h.response(3)["result"]["stopReason"] == "end_turn"
        assert seen and seen[0] is not None
    finally:
        h.close()


def test_client_without_fs_capability_gets_no_reverse_request(tmp_path: Path) -> None:
    """编辑器没声明 fs 能力：一条反向请求都不该发。"""
    target = tmp_path / "app.py"
    target.write_text(_DISK, encoding="utf-8")
    seen: list[Any] = []
    h = _Harness(turn_runner=_read_runner(seen, target))
    try:
        session_id = _session_with_caps(h, tmp_path, {})
        h.send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "看一眼"}]},
            request_id=3,
        )
        h.response(3)
        assert seen == [None]
        assert all(
            not str(m.get("method") or "").startswith("fs/") for m in h.out.messages
        )
    finally:
        h.close()
