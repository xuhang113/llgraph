"""本问内重复工具短路径拦截。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolNode

from llgraph.core.react_limits import parse_identical_tool_guard, resolve_identical_tool_guard
from llgraph.core.tool_loop_guard import (
    IDENTICAL_BLOCK_MARKER,
    IDENTICAL_FAIL_MARKER,
    compute_blocked_tool_messages,
    install_tool_loop_guard,
    shape_from_call,
    wrap_tool_node_with_loop_guard,
)


def _ai(calls: list[dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=calls)


def _call(cid: str, name: str, **args) -> dict:
    return {"id": cid, "name": name, "args": args, "type": "tool_call"}


def test_parse_identical_tool_guard() -> None:
    assert parse_identical_tool_guard(None) is True
    assert parse_identical_tool_guard(False) is False
    assert parse_identical_tool_guard("off") is False
    assert parse_identical_tool_guard("true") is True


def test_resolve_identical_tool_guard_from_workspace(tmp_path: Path) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text(
        json.dumps({"agent": {"identical_tool_guard": False}}),
        encoding="utf-8",
    )
    assert resolve_identical_tool_guard(tmp_path) is False
    assert resolve_identical_tool_guard(None) is True


def test_duplicate_grep_across_rounds_is_blocked() -> None:
    first = _call("c1", "grep_files", pattern="FooService", path=".")
    msgs = [
        HumanMessage(content="查 FooService"),
        _ai([first]),
        ToolMessage(content="匹配结果:\nFoo.java:1", tool_call_id="c1", name="grep_files"),
        _ai([_call("c2", "grep_files", pattern="FooService", path=".")]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("c2", "grep_files", pattern="FooService", path=".")]
    )
    assert "c2" in blocked
    body = str(blocked["c2"].content)
    assert IDENTICAL_BLOCK_MARKER in body
    assert "FooService" in body
    assert "Foo.java:1" in body


def test_grep_workspace_covers_subdirectory() -> None:
    first = _call("c1", "grep_files", pattern="bar_id", path=".")
    msgs = [
        HumanMessage(content="查"),
        _ai([first]),
        ToolMessage(content="未找到匹配内容: pattern='bar_id' path='.'。", tool_call_id="c1", name="grep_files"),
        _ai([_call("c2", "grep_files", pattern="bar_id", path="src")]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("c2", "grep_files", pattern="bar_id", path="src")]
    )
    assert "c2" in blocked


def test_different_grep_pattern_not_blocked() -> None:
    first = _call("c1", "grep_files", pattern="alpha", path=".")
    msgs = [
        HumanMessage(content="查"),
        _ai([first]),
        ToolMessage(content="匹配结果:\nhit", tool_call_id="c1", name="grep_files"),
        _ai([_call("c2", "grep_files", pattern="beta", path=".")]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("c2", "grep_files", pattern="beta", path=".")]
    )
    assert blocked == {}


def test_full_file_read_covers_narrow_range() -> None:
    first = _call("c1", "read_file", path="a.py", start_line=1, end_line=0)
    msgs = [
        HumanMessage(content="读"),
        _ai([first]),
        ToolMessage(
            content="--- a.py (行 1-80 / 共 80 行) ---\nprint(1)\n",
            tool_call_id="c1",
            name="read_file",
        ),
        _ai([_call("c2", "read_file", path="a.py", start_line=10, end_line=20)]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("c2", "read_file", path="./a.py", start_line=10, end_line=20)]
    )
    assert "c2" in blocked
    assert IDENTICAL_BLOCK_MARKER in str(blocked["c2"].content)


def test_narrow_read_does_not_cover_wider() -> None:
    first = _call("c1", "read_file", path="a.py", start_line=10, end_line=20)
    msgs = [
        HumanMessage(content="读"),
        _ai([first]),
        ToolMessage(content="snippet", tool_call_id="c1", name="read_file"),
        _ai([_call("c2", "read_file", path="a.py", start_line=1, end_line=200)]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("c2", "read_file", path="a.py", start_line=1, end_line=200)]
    )
    assert blocked == {}


def test_read_files_covers_later_single_read() -> None:
    first = _call(
        "c1",
        "read_files",
        paths=["src/a.py", "src/b.py"],
        start_line=1,
        end_line=0,
    )
    msgs = [
        HumanMessage(content="读"),
        _ai([first]),
        ToolMessage(content="批量读取 2/2 个文件:\n...", tool_call_id="c1", name="read_files"),
        _ai([_call("c2", "read_file", path="src/b.py", start_line=1, end_line=40)]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("c2", "read_file", path="src/b.py", start_line=1, end_line=40)]
    )
    assert "c2" in blocked


def test_same_batch_duplicate_keeps_first() -> None:
    calls = [
        _call("a", "grep_files", pattern="x", path="."),
        _call("b", "grep_files", pattern="x", path="."),
    ]
    msgs = [HumanMessage(content="q"), _ai(calls)]
    blocked = compute_blocked_tool_messages(msgs, calls)
    assert "a" not in blocked
    assert "b" in blocked
    assert "本批" in str(blocked["b"].content)


def test_write_invalidates_read_cache() -> None:
    read_call = _call("r1", "read_file", path="a.py", start_line=1, end_line=0)
    write_call = _call(
        "w1",
        "search_replace",
        path="a.py",
        old_string="a",
        new_string="b",
    )
    msgs = [
        HumanMessage(content="改"),
        _ai([read_call]),
        ToolMessage(content="--- a.py (行 1-2 / 共 2 行) ---\na\n", tool_call_id="r1", name="read_file"),
        _ai([write_call]),
        ToolMessage(content="已替换 a.py（1 处）", tool_call_id="w1", name="search_replace"),
        _ai([_call("r2", "read_file", path="a.py", start_line=1, end_line=0)]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("r2", "read_file", path="a.py", start_line=1, end_line=0)]
    )
    assert blocked == {}


def test_same_batch_write_skips_read_short_circuit() -> None:
    read_call = _call("r1", "read_file", path="a.py", start_line=1, end_line=0)
    msgs = [
        HumanMessage(content="改"),
        _ai([read_call]),
        ToolMessage(content="body", tool_call_id="r1", name="read_file"),
        _ai(
            [
                _call(
                    "w1",
                    "search_replace",
                    path="a.py",
                    old_string="a",
                    new_string="b",
                ),
                _call("r2", "read_file", path="a.py", start_line=1, end_line=0),
            ]
        ),
    ]
    blocked = compute_blocked_tool_messages(
        msgs,
        [
            _call("w1", "search_replace", path="a.py", old_string="a", new_string="b"),
            _call("r2", "read_file", path="a.py", start_line=1, end_line=0),
        ],
    )
    assert "r2" not in blocked


def test_identical_failed_search_replace_blocked() -> None:
    first = _call(
        "w1",
        "search_replace",
        path="a.py",
        old_string="missing",
        new_string="x",
    )
    msgs = [
        HumanMessage(content="改"),
        _ai([first]),
        ToolMessage(
            content="未找到 old_string（0 处匹配）: a.py\n请先 read_file",
            tool_call_id="w1",
            name="search_replace",
        ),
        _ai(
            [
                _call(
                    "w2",
                    "search_replace",
                    path="a.py",
                    old_string="missing",
                    new_string="x",
                )
            ]
        ),
    ]
    blocked = compute_blocked_tool_messages(
        msgs,
        [
            _call(
                "w2",
                "search_replace",
                path="a.py",
                old_string="missing",
                new_string="x",
            )
        ],
    )
    assert "w2" in blocked
    assert IDENTICAL_FAIL_MARKER in str(blocked["w2"].content)


def test_changed_old_string_not_blocked() -> None:
    first = _call(
        "w1",
        "search_replace",
        path="a.py",
        old_string="missing",
        new_string="x",
    )
    msgs = [
        HumanMessage(content="改"),
        _ai([first]),
        ToolMessage(content="未找到 old_string（0 处匹配）: a.py", tool_call_id="w1", name="search_replace"),
        _ai(
            [
                _call(
                    "w2",
                    "search_replace",
                    path="a.py",
                    old_string="actual line",
                    new_string="x",
                )
            ]
        ),
    ]
    blocked = compute_blocked_tool_messages(
        msgs,
        [
            _call(
                "w2",
                "search_replace",
                path="a.py",
                old_string="actual line",
                new_string="x",
            )
        ],
    )
    assert blocked == {}


def test_shell_not_cached() -> None:
    first = _call("s1", "run_shell_command", command="pwd")
    assert shape_from_call(first) is None
    msgs = [
        HumanMessage(content="跑"),
        _ai([first]),
        ToolMessage(content="/tmp", tool_call_id="s1", name="run_shell_command"),
        _ai([_call("s2", "run_shell_command", command="pwd")]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("s2", "run_shell_command", command="pwd")]
    )
    assert blocked == {}


def test_new_user_turn_resets_cache() -> None:
    first = _call("c1", "grep_files", pattern="x", path=".")
    msgs = [
        HumanMessage(content="第一问"),
        _ai([first]),
        ToolMessage(content="hit", tool_call_id="c1", name="grep_files"),
        HumanMessage(content="第二问，再搜 x"),
        _ai([_call("c2", "grep_files", pattern="x", path=".")]),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [_call("c2", "grep_files", pattern="x", path=".")]
    )
    assert blocked == {}


def test_wrap_skips_original_run() -> None:
    inner = ToolNode([])
    executed: list[str] = []

    def run(call, *args, **kwargs):
        executed.append(call["id"])
        return ToolMessage(content="ran", tool_call_id=call["id"], name="grep_files")

    inner._run_one = run  # type: ignore[method-assign]
    wrap_tool_node_with_loop_guard(inner)
    calls = [
        _call("a", "grep_files", pattern="x", path="."),
        _call("b", "grep_files", pattern="x", path="."),
    ]
    msgs = [HumanMessage(content="q"), _ai(calls)]
    install_tool_loop_guard(inner, msgs, calls)
    out_a = inner._run_one(calls[0], "list", MagicMock())
    out_b = inner._run_one(calls[1], "list", MagicMock())
    assert executed == ["a"]
    assert str(out_a.content) == "ran"
    assert IDENTICAL_BLOCK_MARKER in str(out_b.content)


def test_blocked_write_releases_serialize_gate(monkeypatch) -> None:
    """拦截的写调用仍须 mark_done，避免同 path 后续写空等超时。"""
    import time

    from llgraph.core.write_serialize import (
        install_write_serialize_gate,
        wrap_tool_node_with_write_serialize,
    )

    monkeypatch.setattr("llgraph.core.write_serialize._WRITE_WAIT_TIMEOUT_SEC", 0.4)
    inner = ToolNode([])
    executed: list[str] = []

    def run(call, *args, **kwargs):
        executed.append(call["id"])
        return ToolMessage(content="ok", tool_call_id=call["id"], name="search_replace")

    inner._run_one = run  # type: ignore[method-assign]
    wrap_tool_node_with_loop_guard(inner)
    wrap_tool_node_with_write_serialize(inner)

    failed = _call("w1", "search_replace", path="a.py", old_string="x", new_string="y")
    calls = [
        _call("w2", "search_replace", path="a.py", old_string="x", new_string="y"),
        _call("w3", "search_replace", path="a.py", old_string="z", new_string="q"),
    ]
    msgs = [
        HumanMessage(content="改"),
        _ai([failed]),
        ToolMessage(
            content="未找到 old_string（0 处匹配）: a.py",
            tool_call_id="w1",
            name="search_replace",
        ),
        _ai(calls),
    ]
    install_tool_loop_guard(inner, msgs, calls)
    install_write_serialize_gate(inner, calls)
    started = time.perf_counter()
    out2 = inner._run_one(calls[0], "list", MagicMock())
    out3 = inner._run_one(calls[1], "list", MagicMock())
    elapsed = time.perf_counter() - started
    assert IDENTICAL_FAIL_MARKER in str(out2.content)
    assert str(out3.content) == "ok"
    assert executed == ["w3"]
    assert elapsed < 0.25
