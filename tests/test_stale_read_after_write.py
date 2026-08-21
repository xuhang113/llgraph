"""写入后作废旧 read 快照，以及写入成功返回当前片段。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolNode

from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch
from llgraph.context.stale_read_after_write import (
    STALE_AFTER_WRITE_MARKER,
    collect_write_success_paths,
    invalidate_reads_after_writes_for_dispatch,
)
from llgraph.core.edit_apply import (
    apply_search_replace,
    format_apply_success,
    strip_read_file_artifacts,
)
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.workspace import WorkspaceContext
from llgraph.core.write_serialize import (
    gate_from_tool_calls,
    install_write_serialize_gate,
    touched_path_from_call,
    wrap_tool_node_with_write_serialize,
)


def _ai(calls: list[dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=calls)


def _call(cid: str, name: str, **args) -> dict:
    return {"id": cid, "name": name, "args": args, "type": "tool_call"}


def _read_body(path: str, text: str, *, start: int = 1) -> str:
    lines = text.splitlines()
    end = start + len(lines) - 1
    numbered = "\n".join(f"{start + i}| {line}" for i, line in enumerate(lines))
    return f"--- {path} (行 {start}-{end} / 共 {end} 行) ---\n{numbered}"


def test_collect_write_success_paths_from_tool_args() -> None:
    msgs = [
        HumanMessage(content="改"),
        _ai([_call("w1", "search_replace", path="src/Foo.java", old_string="a", new_string="b")]),
        ToolMessage(content="已替换 src/Foo.java（1 处）", tool_call_id="w1", name="search_replace"),
    ]
    hits = collect_write_success_paths(msgs)
    assert hits == [(2, "src/Foo.java")]


def test_invalidate_drops_pre_write_read_keeps_post_write_read() -> None:
    old = _read_body("src/Foo.java", "old line\nkeep")
    new = _read_body("src/Foo.java", "new line\nkeep")
    msgs = [
        HumanMessage(content="改 Foo"),
        _ai([_call("r1", "read_file", path="src/Foo.java")]),
        ToolMessage(content=old, tool_call_id="r1", name="read_file"),
        _ai([_call("w1", "search_replace", path="./src/Foo.java", old_string="old", new_string="new")]),
        ToolMessage(
            content="已替换 src/Foo.java（1 处）\n--- src/Foo.java (行 1-2 / 共 2 行) [写入后快照] ---\n1| new line",
            tool_call_id="w1",
            name="search_replace",
        ),
        _ai([_call("r2", "read_file", path="src/Foo.java")]),
        ToolMessage(content=new, tool_call_id="r2", name="read_file"),
    ]
    out = invalidate_reads_after_writes_for_dispatch(msgs)
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert STALE_AFTER_WRITE_MARKER in str(tools[0].content)
    assert "old line" not in str(tools[0].content)
    assert "已替换" in str(tools[1].content)
    assert "new line" in str(tools[2].content)


def test_invalidate_skips_unrelated_file() -> None:
    other = _read_body("src/Bar.java", "bar")
    msgs = [
        HumanMessage(content="q"),
        _ai([_call("r1", "read_file", path="src/Bar.java")]),
        ToolMessage(content=other, tool_call_id="r1", name="read_file"),
        _ai([_call("w1", "write_file", path="src/Foo.java", content="x")]),
        ToolMessage(content="已写入 src/Foo.java（1 字符）", tool_call_id="w1", name="write_file"),
    ]
    out = invalidate_reads_after_writes_for_dispatch(msgs)
    assert out[2].content == other


def test_invalidate_read_files_drops_only_stale_block() -> None:
    content = (
        "批量读取 2/2 个文件:\n\n"
        + _read_body("src/Foo.java", "FOO_OLD")
        + "\n\n"
        + _read_body("src/Bar.java", "BAR_OK")
    )
    msgs = [
        HumanMessage(content="q"),
        _ai([_call("r1", "read_files", paths=["src/Foo.java", "src/Bar.java"])]),
        ToolMessage(content=content, tool_call_id="r1", name="read_files"),
        _ai([_call("w1", "search_replace", path="src/Foo.java", old_string="FOO", new_string="X")]),
        ToolMessage(content="已替换 src/Foo.java（1 处）", tool_call_id="w1", name="search_replace"),
    ]
    out = invalidate_reads_after_writes_for_dispatch(msgs)
    body = str(out[2].content)
    assert "FOO_OLD" not in body
    assert "BAR_OK" in body
    assert STALE_AFTER_WRITE_MARKER in body


def test_failed_write_does_not_invalidate() -> None:
    old = _read_body("a.py", "alpha")
    msgs = [
        HumanMessage(content="q"),
        _ai([_call("r1", "read_file", path="a.py")]),
        ToolMessage(content=old, tool_call_id="r1", name="read_file"),
        _ai([_call("w1", "search_replace", path="a.py", old_string="nope", new_string="x")]),
        ToolMessage(content="未找到 old_string（0 处匹配）: a.py", tool_call_id="w1", name="search_replace"),
    ]
    out = invalidate_reads_after_writes_for_dispatch(msgs)
    assert "alpha" in str(out[2].content)


def test_prepare_dispatch_invalidates_stale_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLGRAPH_API_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LLGRAPH_API_KEY", "test-key")
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text(
        '{"context": {"compress_strategy": "auto", "dispatch_dedupe_read_paths": true}}',
        encoding="utf-8",
    )
    old = _read_body("pkg/A.java", "class A { int x = 1; }")
    msgs = [
        HumanMessage(content="改 A"),
        _ai([_call("r1", "read_file", path="pkg/A.java")]),
        ToolMessage(content=old, tool_call_id="r1", name="read_file"),
        _ai([_call("w1", "search_replace", path="pkg/A.java", old_string="x = 1", new_string="x = 2")]),
        ToolMessage(content="已替换 pkg/A.java（1 处）", tool_call_id="w1", name="search_replace"),
    ]
    prepared = prepare_messages_for_llm_dispatch(
        msgs,
        agent_system_content="sys",
        workspace=tmp_path,
        model_id="claude-sonnet-4-6",
    )
    tools = [m for m in prepared if isinstance(m, ToolMessage)]
    assert tools
    assert STALE_AFTER_WRITE_MARKER in str(tools[0].content)
    assert "int x = 1" not in str(tools[0].content)


def test_format_apply_success_includes_new_snippet() -> None:
    old = "alpha\nbeta\ngamma\n"
    result = apply_search_replace(old, "beta", "BETA")
    text = format_apply_success("sample.py", result, old_text=old)
    assert text.startswith("已替换 sample.py")
    assert "[写入后快照]" in text
    assert "BETA" in text
    assert "2| BETA" in text


def test_search_replace_tool_returns_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("    def foo():\n        return 1\n", encoding="utf-8")
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tool = next(t for t in create_filesystem_tools(ctx) if t.name == "search_replace")
    out = str(
        tool.invoke(
            {
                "path": "sample.py",
                "old_string": "def foo():\n    return 1\n",
                "new_string": "def foo():\n    return 2\n",
            }
        )
    )
    assert out.startswith("已替换")
    assert "[写入后快照]" in out
    assert "return 2" in out


def test_write_file_returns_head_snapshot(tmp_path: Path) -> None:
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tool = next(t for t in create_filesystem_tools(ctx) if t.name == "write_file")
    out = str(tool.invoke({"path": "n.py", "content": "hello\nworld\n"}))
    assert out.startswith("已写入")
    assert "[写入后快照]" in out
    assert "1| hello" in out


def test_strip_read_file_artifacts_accepts_write_snapshot_header() -> None:
    pasted = (
        "--- sample.py (行 1-2 / 共 2 行) [写入后快照] ---\n"
        "1| def foo():\n"
        "2|     return 2\n"
    )
    stripped = strip_read_file_artifacts(pasted)
    assert "def foo():" in stripped
    assert "1|" not in stripped
    assert "写入后快照" not in stripped


def test_touched_path_includes_read_file() -> None:
    read = _call("r1", "read_file", path="./src/a.py")
    write = _call("w1", "search_replace", path="src/a.py", old_string="a", new_string="b")
    grep = _call("g1", "grep_files", pattern="a", path=".")
    assert touched_path_from_call(read) == "src/a.py"
    assert touched_path_from_call(write) == "src/a.py"
    assert touched_path_from_call(grep) is None


def test_read_waits_for_preceding_write_same_path(tmp_path: Path) -> None:
    target = tmp_path / "seq.py"
    target.write_text("STEP=0\n", encoding="utf-8")
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tools = create_filesystem_tools(ctx)
    inner = ToolNode(tools)
    wrap_tool_node_with_write_serialize(inner)
    calls = [
        {
            "id": "w1",
            "name": "search_replace",
            "args": {"path": "seq.py", "old_string": "STEP=0", "new_string": "STEP=1"},
            "type": "tool_call",
        },
        {
            "id": "r1",
            "name": "read_file",
            "args": {"path": "seq.py"},
            "type": "tool_call",
        },
    ]
    install_write_serialize_gate(inner, calls)
    runtime = MagicMock()
    runtime.config = {}
    runtime.state = {}
    read_out: list[str] = []

    def do_read() -> None:
        result = inner._run_one(calls[1], "list", runtime)
        read_out.append(str(result))

    try:
        reader = threading.Thread(target=do_read)
        writer = threading.Thread(target=lambda: inner._run_one(calls[0], "list", runtime))
        reader.start()
        time.sleep(0.02)
        writer.start()
        writer.join(timeout=2)
        reader.join(timeout=2)
    finally:
        inner._llgraph_write_gate = None

    assert read_out
    assert "STEP=1" in read_out[0]
    assert "STEP=0" not in read_out[0]


def test_unrelated_reads_not_queued_with_write() -> None:
    calls = [
        _call("r1", "read_file", path="a.py"),
        _call("w1", "search_replace", path="b.py", old_string="x", new_string="y"),
    ]
    gate = gate_from_tool_calls(calls)
    assert gate._order == {}
