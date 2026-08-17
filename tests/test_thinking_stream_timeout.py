"""thinking-only 流式超时。"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessageChunk

from llgraph.core.llm_settings import DEFAULT_THINKING_STREAM_TIMEOUT_SEC
from llgraph.core.react_invoke import _consume_runnable_stream
from llgraph.session.session_run_log import ThinkingStreamTimeoutError


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __iter__(self):
        for chunk in self._chunks:
            time.sleep(0.02)
            yield chunk

    def close(self) -> None:
        return None


def _thinking_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=[{"type": "thinking", "thinking": text}])


class _SlowThinkingStream:
    def __iter__(self):
        for i in range(20):
            time.sleep(0.03)
            yield _thinking_chunk(f"chunk-{i}")

    def close(self) -> None:
        return None


def test_thinking_stream_timeout(monkeypatch) -> None:
    ws = Path("/tmp/ws-thinking-timeout")
    runnable = MagicMock()
    runnable.llgraph_workspace = ws
    runnable.stream.return_value = _SlowThinkingStream()

    monkeypatch.setattr(
        "llgraph.core.react_invoke.resolve_llm_settings",
        lambda _ws: MagicMock(thinking_stream_timeout_sec=0.05),
    )
    monkeypatch.setattr("llgraph.core.react_invoke.agent_cancel_requested", lambda: False)

    with pytest.raises(ThinkingStreamTimeoutError):
        _consume_runnable_stream(runnable, [{"role": "user", "content": "hi"}], {})


def test_tool_chunk_resets_thinking_timer(monkeypatch) -> None:
    ws = Path("/tmp/ws-tool-reset")
    runnable = MagicMock()
    runnable.llgraph_workspace = ws
    tool_chunk = AIMessageChunk(content="", tool_calls=[{"id": "t1", "name": "grep", "args": {}}])
    runnable.stream.return_value = _FakeStream([_thinking_chunk("plan"), tool_chunk])

    monkeypatch.setattr(
        "llgraph.core.react_invoke.resolve_llm_settings",
        lambda _ws: MagicMock(thinking_stream_timeout_sec=DEFAULT_THINKING_STREAM_TIMEOUT_SEC),
    )
    monkeypatch.setattr("llgraph.core.react_invoke.agent_cancel_requested", lambda: False)

    result = _consume_runnable_stream(runnable, [{"role": "user", "content": "hi"}], {})
    assert result is not None
