"""ReAct 步间工具结果增量写回 checkpoint。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.incremental_context import (
    maybe_prune_tools_during_react,
    prune_stale_tool_messages,
)
from llgraph.context.context_settings import resolve_context_settings


class _FakeAgent:
    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self.updated: list[list[Any]] = []

    def get_state(self, _config: dict[str, Any]) -> Any:
        class _State:
            values = {"messages": self._messages}

        return _State()

    def update_state(self, _config: dict[str, Any], patch: dict[str, Any]) -> None:
        self._messages = list(patch["messages"])
        self.updated.append(self._messages)


def _write_agent_json(tmp_path: Path, *, keep: int = 2) -> Path:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir(exist_ok=True)
    (llgraph_dir / "agent.json").write_text(
        f'{{"context": {{"budget_source": "config", "max_tokens_estimate": 1000, '
        f'"auto_compress_ratio": 0.85, "incremental_tool_prune": true, '
        f'"keep_recent_tool_messages": {keep}, '
        f'"compress_tool_mask_max_chars": 200, "read_tool_mask_max_chars": 200, '
        f'"tool_prune_token_ratio": 0.0}}}}',
        encoding="utf-8",
    )
    return tmp_path


def test_prune_stale_tool_messages_masks_older(tmp_path: Path) -> None:
    ws = _write_agent_json(tmp_path, keep=2)
    settings = resolve_context_settings(ws)
    long_body = "x" * 500
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "grep", "args": {}}]),
        ToolMessage(content=long_body, tool_call_id="1", name="grep"),
        AIMessage(content="", tool_calls=[{"id": "2", "name": "read_file", "args": {}}]),
        ToolMessage(content=long_body, tool_call_id="2", name="read_file"),
        AIMessage(content="", tool_calls=[{"id": "3", "name": "read_file", "args": {}}]),
        ToolMessage(content=long_body, tool_call_id="3", name="read_file"),
    ]
    pruned, count = prune_stale_tool_messages(messages, ws, settings)
    assert count == 1
    assert long_body not in pruned[2].content
    assert long_body in pruned[4].content
    assert long_body in pruned[6].content


def test_maybe_prune_tools_during_react_writes_checkpoint(tmp_path: Path, monkeypatch) -> None:
    ws = _write_agent_json(tmp_path, keep=2)
    long_body = "y" * 600
    messages = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"id": "a", "name": "grep", "args": {}}]),
        ToolMessage(content=long_body, tool_call_id="a", name="grep"),
        AIMessage(content="", tool_calls=[{"id": "b", "name": "read_file", "args": {}}]),
        ToolMessage(content=long_body, tool_call_id="b", name="read_file"),
        AIMessage(content="", tool_calls=[{"id": "c", "name": "read_file", "args": {}}]),
        ToolMessage(content=long_body, tool_call_id="c", name="read_file"),
    ]
    agent = _FakeAgent(messages)
    saved: list[Any] = []

    def _fake_save(_ws: Path, _tid: str, msgs: list[Any]) -> None:
        saved.append(list(msgs))

    monkeypatch.setattr(
        "llgraph.session.session_file_store.save_session_messages",
        _fake_save,
    )
    report = maybe_prune_tools_during_react(
        agent,
        thread_id="t-react",
        workspace=ws,
    )
    assert report is not None
    assert report.pruned_count >= 1
    assert report.trigger == "react"
    assert agent.updated
    assert not saved
    assert long_body not in agent._messages[2].content
