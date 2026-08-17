"""连续单工具轮后 ToolMessage 注入并行提示。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.core.react_limits import parse_batch_tools_nudge_after
from llgraph.core.react_tools import (
    count_single_tool_rounds_since_user,
    maybe_append_batch_tools_hint,
)


def _ai(n: int, *, prefix: str = "c") -> AIMessage:
    calls = [
        {"id": f"{prefix}{i}", "name": "grep_files", "args": {"pattern": "x"}}
        for i in range(n)
    ]
    return AIMessage(content="", tool_calls=calls)


def test_count_single_tool_streak() -> None:
    msgs = [
        HumanMessage(content="q"),
        _ai(1, prefix="a"),
        ToolMessage(content="1", tool_call_id="a0", name="grep_files"),
        _ai(1, prefix="b"),
        ToolMessage(content="2", tool_call_id="b0", name="grep_files"),
        _ai(1, prefix="c"),
    ]
    assert count_single_tool_rounds_since_user(msgs) == 3


def test_multi_tool_resets_streak() -> None:
    msgs = [
        HumanMessage(content="q"),
        _ai(1, prefix="a"),
        ToolMessage(content="1", tool_call_id="a0", name="grep_files"),
        _ai(2, prefix="b"),
        ToolMessage(content="2", tool_call_id="b0", name="grep_files"),
        _ai(1, prefix="c"),
    ]
    assert count_single_tool_rounds_since_user(msgs) == 0


def test_append_hint_when_threshold_met(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "llgraph.core.react_tools.resolve_batch_tools_nudge_after",
        lambda _ws: 3,
    )
    prior = [
        HumanMessage(content="q"),
        _ai(1, prefix="a"),
        ToolMessage(content="1", tool_call_id="a0", name="grep_files"),
        _ai(1, prefix="b"),
        ToolMessage(content="2", tool_call_id="b0", name="grep_files"),
        _ai(1, prefix="c"),
    ]
    out = {
        "messages": [
            ToolMessage(content="hit", tool_call_id="c0", name="grep_files"),
        ]
    }
    patched = maybe_append_batch_tools_hint(out, prior_messages=prior, workspace=tmp_path)
    body = str(patched["messages"][0].content)
    assert "已连续多轮每次仅 1 个工具调用" in body
    assert "并行多个工具调用" in body


def test_no_hint_below_threshold(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "llgraph.core.react_tools.resolve_batch_tools_nudge_after",
        lambda _ws: 3,
    )
    prior = [
        HumanMessage(content="q"),
        _ai(1, prefix="a"),
    ]
    out = {
        "messages": [
            ToolMessage(content="hit", tool_call_id="a0", name="grep_files"),
        ]
    }
    patched = maybe_append_batch_tools_hint(out, prior_messages=prior, workspace=tmp_path)
    assert patched["messages"][0].content == "hit"


def test_parse_batch_nudge() -> None:
    assert parse_batch_tools_nudge_after(None) == 3
    assert parse_batch_tools_nudge_after(0) == 0
    assert parse_batch_tools_nudge_after("5") == 5
