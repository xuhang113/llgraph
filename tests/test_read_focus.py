"""大文件智能读：大纲 + 命中窗（对齐 Cursor / Claude Code）。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.read_segment_dedupe import extract_read_segments
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.read_focus import (
    FOCUS_READ_MARKER,
    FULL_READ_MAX_LINES,
    extract_outline,
    format_focus_read,
    format_numbered_slice,
    plan_hit_windows,
    should_focus_read,
)
from llgraph.core.tool_execution_context import set_tool_execution_messages
from llgraph.core.workspace import WorkspaceContext


def _python_module(n_funcs: int = 24, body_lines: int = 18) -> str:
    chunks = ['"""sample module"""', "import os", ""]
    for idx in range(n_funcs):
        chunks.append(f"def func_{idx}():")
        chunks.append(f"    unique_body_{idx} = {idx}")
        chunks.extend(["    pass"] * body_lines)
        chunks.append("")
    return "\n".join(chunks) + "\n"


def test_should_focus_unscoped_large_file() -> None:
    assert should_focus_read(start_line=1, end_line=0, total_lines=FULL_READ_MAX_LINES + 1)
    assert should_focus_read(start_line=1, end_line=900, total_lines=900)


def test_should_not_focus_small_or_scoped() -> None:
    assert not should_focus_read(start_line=1, end_line=0, total_lines=40)
    assert not should_focus_read(start_line=1, end_line=0, total_lines=FULL_READ_MAX_LINES)
    assert not should_focus_read(start_line=1, end_line=120, total_lines=900)
    assert not should_focus_read(start_line=400, end_line=0, total_lines=900)


def test_extract_outline_python_defs() -> None:
    text = _python_module(n_funcs=8, body_lines=2)
    entries = extract_outline(text.splitlines(), "mod.py")
    names = [item.text for item in entries]
    assert any("def func_0" in name for name in names)
    assert any("def func_7" in name for name in names)
    assert all(item.line >= 1 for item in entries)


def test_extract_outline_java_type_and_method() -> None:
    lines = [
        "package com.foo;",
        "public class Foo {",
        "    public void bar() {",
        "        return;",
        "    }",
        "}",
    ]
    entries = extract_outline(lines, "Foo.java")
    joined = " ".join(item.text for item in entries)
    assert "class Foo" in joined
    assert "bar(" in joined


def test_format_focus_read_omits_function_bodies() -> None:
    text = _python_module(n_funcs=20, body_lines=20)
    lines = text.splitlines()
    assert len(lines) > FULL_READ_MAX_LINES
    out = format_focus_read("mod.py", lines)
    assert FOCUS_READ_MARKER in out
    assert "def func_0" in out
    assert "unique_body_12" not in out
    assert "--- mod.py (行 1-" in out
    full = format_numbered_slice("mod.py", lines, 1, len(lines))
    assert len(out) < len(full) // 3


def test_format_focus_read_includes_hit_windows() -> None:
    text = _python_module(n_funcs=20, body_lines=20)
    lines = text.splitlines()
    hit_line = next(i for i, line in enumerate(lines, start=1) if "unique_body_12" in line)
    out = format_focus_read("mod.py", lines, hit_lines=[hit_line])
    assert "unique_body_12" in out
    segs = extract_read_segments(out)
    assert segs
    assert segs[0][0] == "mod.py"
    assert any(start <= hit_line <= end for _, start, end in segs)


def test_plan_hit_windows_merges_and_caps() -> None:
    windows = plan_hit_windows([10, 12, 400], total_lines=500, radius=5, max_windows=4)
    assert windows[0][0] <= 10
    assert windows[0][1] >= 12
    assert any(start <= 400 <= end for start, end in windows)


def test_read_file_tool_small_file_is_full(tmp_path: Path) -> None:
    (tmp_path / "tiny.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    tool = next(t for t in create_filesystem_tools(WorkspaceContext(tmp_path)) if t.name == "read_file")
    out = str(tool.invoke({"path": "tiny.py"}))
    assert FOCUS_READ_MARKER not in out
    assert "def foo" in out
    assert "return 1" in out


def test_read_file_tool_large_file_folds(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text(_python_module(), encoding="utf-8")
    tool = next(t for t in create_filesystem_tools(WorkspaceContext(tmp_path)) if t.name == "read_file")
    out = str(tool.invoke({"path": "big.py"}))
    assert FOCUS_READ_MARKER in out
    assert "def func_0" in out
    assert "unique_body_18" not in out


def test_read_file_tool_explicit_range_honored(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text(_python_module(), encoding="utf-8")
    tool = next(t for t in create_filesystem_tools(WorkspaceContext(tmp_path)) if t.name == "read_file")
    out = str(tool.invoke({"path": "big.py", "start_line": 4, "end_line": 12}))
    assert FOCUS_READ_MARKER not in out
    assert "--- big.py (行 4-12 /" in out
    assert "unique_body_0" in out


def test_read_file_tool_injects_grep_hit_window(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text(_python_module(), encoding="utf-8")
    lines = (tmp_path / "big.py").read_text(encoding="utf-8").splitlines()
    hit_line = next(i for i, line in enumerate(lines, start=1) if "unique_body_18" in line)
    set_tool_execution_messages(
        [
            HumanMessage(content="find it"),
            AIMessage(
                content="",
                tool_calls=[{"id": "g1", "name": "grep_files", "args": {"pattern": "unique_body_18"}}],
            ),
            ToolMessage(
                content=f"big.py:{hit_line}: unique_body_18 = 18",
                tool_call_id="g1",
                name="grep_files",
            ),
        ]
    )
    try:
        tool = next(
            t for t in create_filesystem_tools(WorkspaceContext(tmp_path)) if t.name == "read_file"
        )
        out = str(tool.invoke({"path": "big.py"}))
    finally:
        set_tool_execution_messages(None)
    assert FOCUS_READ_MARKER in out
    assert "unique_body_18" in out


def test_read_files_batch_folds_each_large_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(_python_module(), encoding="utf-8")
    (tmp_path / "b.py").write_text(_python_module(), encoding="utf-8")
    tool = next(
        t for t in create_filesystem_tools(WorkspaceContext(tmp_path)) if t.name == "read_files"
    )
    out = str(tool.invoke({"paths": ["a.py", "b.py"]}))
    assert out.count(FOCUS_READ_MARKER) == 2
    assert "unique_body_18" not in out


def test_format_numbered_slice_matches_legacy_header() -> None:
    text = format_numbered_slice("src/a.py", ["alpha", "beta"], 1, 2)
    assert text.startswith("--- src/a.py (行 1-2 / 共 2 行) ---\n")
    assert "1| alpha" in text
    assert "2| beta" in text
