"""search_replace 容错匹配与同文件写串行。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolNode

from llgraph.core.edit_apply import (
    EditHunk,
    apply_edit_hunks,
    apply_search_replace,
    format_apply_failure,
    format_apply_success,
    parse_replacements_arg,
    strip_read_file_artifacts,
)
from llgraph.core.edit_diagnostics import DIAGNOSTIC_MARKER
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.tool_invoke_timing import wrap_tool_node_with_timing
from llgraph.core.workspace import WorkspaceContext
from llgraph.core.write_serialize import (
    install_write_serialize_gate,
    wrap_tool_node_with_write_serialize,
)


def test_exact_replace() -> None:
    text = "alpha\nbeta\ngamma\n"
    result = apply_search_replace(text, "beta", "BETA")
    assert result.ok
    assert result.strategy == "exact"
    assert result.new_text == "alpha\nBETA\ngamma\n"
    assert result.replacements == 1


def test_crlf_needle_matches_lf_file() -> None:
    text = "def foo():\n    return 1\n"
    needle = "def foo():\r\n    return 1\r\n"
    result = apply_search_replace(text, needle, "def foo():\n    return 2\n")
    assert result.ok
    assert result.strategy == "newline"
    assert "return 2" in result.new_text
    assert "\r\n" not in result.new_text


def test_lf_needle_matches_crlf_file() -> None:
    text = "def foo():\r\n    return 1\r\n"
    needle = "def foo():\n    return 1\n"
    result = apply_search_replace(text, needle, "def foo():\n    return 2\n")
    assert result.ok
    assert "\r\n" in result.new_text
    assert "return 2" in result.new_text


def test_trailing_whitespace() -> None:
    text = "value = 1   \nnext = 2\n"
    result = apply_search_replace(text, "value = 1\n", "value = 3\n")
    assert result.ok
    assert result.strategy == "trailing_ws"
    assert result.new_text == "value = 3\nnext = 2\n"


def test_indent_flex_applies_file_indent_to_new() -> None:
    text = "class A:\n    def foo():\n        return 1\n"
    needle = "def foo():\n    return 1\n"
    result = apply_search_replace(text, needle, "def foo():\n    return 2\n")
    assert result.ok
    assert result.strategy == "indent"
    assert "    def foo():\n        return 2\n" in result.new_text


def test_read_file_line_prefix_stripped() -> None:
    needle = (
        "--- src/a.py (行 1-2 / 共 10 行) ---\n"
        "1| def foo():\n"
        "2|     return 1\n"
    )
    cleaned = strip_read_file_artifacts(needle)
    assert "1|" not in cleaned
    assert "def foo():" in cleaned
    text = "def foo():\n    return 1\n"
    result = apply_search_replace(text, needle, "def foo():\n    return 2\n")
    assert result.ok
    assert "return 2" in result.new_text


def test_whitespace_flex() -> None:
    text = "foo   =    1\n"
    result = apply_search_replace(text, "foo = 1", "foo = 2")
    assert result.ok
    assert result.strategy == "whitespace"
    assert result.new_text == "foo = 2\n"


def test_unique_required() -> None:
    text = "foo\nfoo\n"
    result = apply_search_replace(text, "foo", "bar", require_unique=True)
    assert not result.ok
    assert "不唯一" in result.error
    msg = format_apply_failure("a.py", result)
    assert "a.py" in msg
    assert "replace_all=true" in msg


def test_replace_all() -> None:
    text = "foo\nfoo\n"
    result = apply_search_replace(text, "foo", "bar", replace_all=True)
    assert result.ok
    assert result.replacements == 2
    assert result.new_text == "bar\nbar\n"


def test_multi_hunk_sequential() -> None:
    text = "a=1\nb=2\n"
    result = apply_edit_hunks(
        text,
        [
            EditHunk("a=1", "a=10"),
            EditHunk("b=2", "b=20"),
        ],
    )
    assert result.ok
    assert result.hunks_applied == 2
    assert result.new_text == "a=10\nb=20\n"
    assert "2 hunk" in format_apply_success("x.py", result)


def test_multi_hunk_second_fails_does_not_partial_apply() -> None:
    text = "a=1\nb=2\n"
    result = apply_edit_hunks(
        text,
        [
            EditHunk("a=1", "a=10"),
            EditHunk("missing", "x"),
        ],
    )
    assert not result.ok
    assert "hunk 2/2" in result.error
    assert "未找到 old_string" in result.error


def test_not_found_hint_contains_nearby_line() -> None:
    text = "def authentic_token():\n    return True\n"
    result = apply_search_replace(
        text, "class CompletelyDifferent:\n    pass\n", "x", allow_fuzzy=True
    )
    assert not result.ok
    msg = format_apply_failure("mod.py", result)
    assert "未找到 old_string" in msg
    assert "已尝试匹配" in msg
    assert "fuzzy" in msg


def test_fuzzy_unique_typo_applies() -> None:
    text = (
        "def load_user(user_id: str) -> User:\n"
        "    row = db.fetch(user_id)\n"
        "    if not row:\n"
        "        raise NotFound(\"user\")\n"
        "    return User.from_row(row)\n"
    )
    needle = (
        "def load_user(user_id: str) -> User:\n"
        "    row = db.fetch(user_id)\n"
        "    if not row:\n"
        "        raise NotFound(\"missing\")\n"
        "    return User.from_row(row)\n"
    )
    new = (
        "def load_user(user_id: str) -> User:\n"
        "    row = db.fetch(user_id)\n"
        "    if not row:\n"
        "        raise NotFound(\"user\")\n"
        "    return User.from_row(row, strict=True)\n"
    )
    result = apply_search_replace(text, needle, new)
    assert result.ok
    assert result.strategy == "fuzzy"
    assert result.fuzzy_ratio >= 0.76
    assert "strict=True" in result.new_text
    assert "raise NotFound(\"user\")" in result.new_text
    msg = format_apply_success("user.py", result, old_text=text)
    assert "fuzzy" in msg


def test_fuzzy_unique_identifier_typo_applies() -> None:
    text = "def authentic_token():\n    return True\n"
    result = apply_search_replace(
        text,
        "def authentic_tokn():\n    return True\n",
        "def authentic_token():\n    return False\n",
    )
    assert result.ok
    assert result.strategy == "fuzzy"
    assert "return False" in result.new_text


def test_fuzzy_not_unique_does_not_apply() -> None:
    text = "def helper():\n    return 1\n\ndef helper():\n    return 1\n"
    result = apply_search_replace(
        text,
        "def helppr():\n    return 1\n",
        "def helper():\n    return 2\n",
    )
    assert not result.ok
    assert "未找到 old_string" in result.error


def test_fuzzy_disabled_falls_back_to_hint() -> None:
    text = (
        "def load_user(user_id: str) -> User:\n"
        "    row = db.fetch(user_id)\n"
        "    if not row:\n"
        "        raise NotFound(\"user\")\n"
        "    return User.from_row(row)\n"
    )
    needle = (
        "def load_user(user_id: str) -> User:\n"
        "    row = db.fetch(user_id)\n"
        "    if not row:\n"
        "        raise NotFound(\"missing\")\n"
        "    return User.from_row(row)\n"
    )
    result = apply_search_replace(text, needle, "x", allow_fuzzy=False)
    assert not result.ok
    assert "fuzzy" not in result.tried
    assert "user" in (result.hint or "") or "load_user" in (result.hint or "")


def test_parse_replacements_arg_aliases() -> None:
    hunks = parse_replacements_arg(
        [{"oldString": "a", "newString": "b", "replaceAll": True}]
    )
    assert len(hunks) == 1
    assert hunks[0].old_string == "a"
    assert hunks[0].new_string == "b"
    assert hunks[0].replace_all is True


def test_search_replace_tool_tolerant_and_multi(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("    def foo():\n        return 1\n", encoding="utf-8")
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tool = next(t for t in create_filesystem_tools(ctx) if t.name == "search_replace")
    out = tool.invoke(
        {
            "path": "sample.py",
            "old_string": "def foo():\n    return 1\n",
            "new_string": "def foo():\n    return 2\n",
        }
    )
    assert str(out).startswith("已替换")
    assert "return 2" in str(out)
    assert "[写入后快照]" in str(out)
    assert "return 2" in target.read_text(encoding="utf-8")

    out2 = tool.invoke(
        {
            "path": "sample.py",
            "replacements": [
                {"old_string": "return 2", "new_string": "return 3"},
                {"old_string": "def foo", "new_string": "def bar"},
            ],
        }
    )
    assert "2 hunk" in str(out2)
    body = target.read_text(encoding="utf-8")
    assert "def bar" in body
    assert "return 3" in body


def test_search_replace_tool_fuzzy_typo(tmp_path: Path) -> None:
    target = tmp_path / "user.py"
    target.write_text(
        "def load_user(user_id: str) -> User:\n"
        "    row = db.fetch(user_id)\n"
        "    if not row:\n"
        "        raise NotFound(\"user\")\n"
        "    return User.from_row(row)\n",
        encoding="utf-8",
    )
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tool = next(t for t in create_filesystem_tools(ctx) if t.name == "search_replace")
    out = str(
        tool.invoke(
            {
                "path": "user.py",
                "old_string": (
                    "def load_user(user_id: str) -> User:\n"
                    "    row = db.fetch(user_id)\n"
                    "    if not row:\n"
                    "        raise NotFound(\"missing\")\n"
                    "    return User.from_row(row)\n"
                ),
                "new_string": (
                    "def load_user(user_id: str) -> User:\n"
                    "    row = db.fetch(user_id)\n"
                    "    if not row:\n"
                    "        raise NotFound(\"user\")\n"
                    "    return User.from_row(row, strict=True)\n"
                ),
            }
        )
    )
    assert out.startswith("已替换")
    assert "fuzzy" in out
    assert "strict=True" in target.read_text(encoding="utf-8")
    assert DIAGNOSTIC_MARKER not in out


def test_search_replace_schema_requires_hunk() -> None:
    from pydantic import ValidationError

    from llgraph.core.filesystem_tool_schemas import SearchReplaceInput

    try:
        SearchReplaceInput.model_validate({"path": "a.py"})
        raise AssertionError("expected validation error")
    except ValidationError:
        pass


def test_same_path_writes_run_in_tool_call_order() -> None:
    """并行 ToolNode 下，同一 path 的写调用按声明顺序完成。"""
    order: list[str] = []
    lock = threading.Lock()

    def slow_write(path: str, old_string: str = "", new_string: str = "") -> str:
        label = old_string
        if label == "first":
            time.sleep(0.08)
        with lock:
            order.append(label)
        return f"已替换 {path}（1 处）"

    tool = StructuredTool.from_function(
        func=slow_write,
        name="search_replace",
        description="test",
    )
    inner = ToolNode([tool])
    wrap_tool_node_with_timing(inner)
    wrap_tool_node_with_write_serialize(inner)

    calls = [
        {
            "id": "c1",
            "name": "search_replace",
            "args": {"path": "a.py", "old_string": "first", "new_string": "x"},
            "type": "tool_call",
        },
        {
            "id": "c2",
            "name": "search_replace",
            "args": {"path": "a.py", "old_string": "second", "new_string": "y"},
            "type": "tool_call",
        },
    ]
    install_write_serialize_gate(inner, calls)
    runtime = MagicMock()
    runtime.config = {}
    runtime.state = {}
    try:
        waiter = threading.Thread(target=lambda: inner._run_one(calls[1], "list", runtime))
        first = threading.Thread(target=lambda: inner._run_one(calls[0], "list", runtime))
        waiter.start()
        time.sleep(0.01)
        first.start()
        waiter.join(timeout=2)
        first.join(timeout=2)
    finally:
        inner._llgraph_write_gate = None

    assert order == ["first", "second"]


def test_same_file_search_replace_hunks_apply_in_order(tmp_path: Path) -> None:
    """同文件连续两处替换：第二处依赖第一处结果，串行后才能成功。"""
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
            "args": {
                "path": "seq.py",
                "old_string": "STEP=0",
                "new_string": "STEP=1",
            },
            "type": "tool_call",
        },
        {
            "id": "w2",
            "name": "search_replace",
            "args": {
                "path": "seq.py",
                "old_string": "STEP=1",
                "new_string": "STEP=2",
            },
            "type": "tool_call",
        },
    ]
    install_write_serialize_gate(inner, calls)
    runtime = MagicMock()
    runtime.config = {}
    runtime.state = {}
    try:
        waiter = threading.Thread(target=lambda: inner._run_one(calls[1], "list", runtime))
        first = threading.Thread(target=lambda: inner._run_one(calls[0], "list", runtime))
        waiter.start()
        time.sleep(0.01)
        first.start()
        first.join(timeout=2)
        waiter.join(timeout=2)
    finally:
        inner._llgraph_write_gate = None
    assert target.read_text(encoding="utf-8") == "STEP=2\n"
