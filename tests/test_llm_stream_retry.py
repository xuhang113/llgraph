"""网关流式瞬时故障重放。

出站走 `.stream()`：响应头一旦回来，SDK 的 `max_retries` 就不管了。
中途被网关掐断会把整轮 ReAct 打掉——前面十几个工具白跑。
这里锁住三件事：哪些错该重放、重放不能吞掉已经流出去的正文、退避可被 Stop 打断。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from llgraph.core.llm_retry import (
    DEFAULT_MAX_ATTEMPTS,
    LlmStreamRetryPolicy,
    classify_stream_failure,
    compute_retry_delay,
    resolve_llm_stream_retry_policy,
)


class _FakeStatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class RemoteProtocolError(Exception):
    """同名于 httpx；分类器按类名判定，不 import 真 SDK。"""


class BadRequestError(Exception):
    pass


def test_transient_status_is_retryable() -> None:
    assert classify_stream_failure(_FakeStatusError("bad gateway", 502)) == "http_502"
    assert classify_stream_failure(_FakeStatusError("slow down", 429)) == "http_429"


def test_client_error_is_not_retryable() -> None:
    assert classify_stream_failure(_FakeStatusError("bad input", 400)) is None
    assert classify_stream_failure(BadRequestError("tool schema invalid")) is None


def test_mid_stream_disconnect_is_retryable() -> None:
    assert classify_stream_failure(RemoteProtocolError("peer closed connection")) is not None
    assert classify_stream_failure(ConnectionResetError(104, "reset by peer")) is not None


def test_gateway_text_only_failure_is_retryable() -> None:
    """网关把上游故障塞进 200 的 SSE body，langchain 抛出来只是普通 Exception。"""
    assert classify_stream_failure(Exception("upstream connect error or disconnect")) is not None
    assert classify_stream_failure(Exception("Error: 503 Service Unavailable")) is not None


def test_context_overflow_never_retried() -> None:
    """重放上下文超限只会再烧一次同样的 prompt。"""
    assert classify_stream_failure(Exception("prompt is too long: 250000 tokens")) is None
    assert classify_stream_failure(Exception("maximum context length exceeded")) is None


def test_fatal_cause_beats_transient_wrapper() -> None:
    """SDK 常把 400 包在看着像网络错的壳里，内层致命就不许重放。"""
    inner = _FakeStatusError("invalid_request_error", 400)
    outer = RemoteProtocolError("connection closed")
    outer.__cause__ = inner
    assert classify_stream_failure(outer) is None


def test_cancellation_is_never_retried() -> None:
    from llgraph.session.session_run_log import ThinkingStreamTimeoutError, UserCancelledError

    assert classify_stream_failure(UserCancelledError("stop")) is None
    assert classify_stream_failure(ThinkingStreamTimeoutError("idle")) is None


def test_retry_delay_backs_off_and_caps() -> None:
    policy = LlmStreamRetryPolicy(max_attempts=5, base_delay_sec=1.0, max_delay_sec=4.0)
    assert compute_retry_delay(1, policy, jitter=0.0) == 1.0
    assert compute_retry_delay(2, policy, jitter=0.0) == 2.0
    assert compute_retry_delay(3, policy, jitter=0.0) == 4.0
    assert compute_retry_delay(9, policy, jitter=0.0) == 4.0
    assert compute_retry_delay(1, policy, jitter=1.0) == 1.25


def test_policy_defaults_and_agent_json_override(tmp_path: Path) -> None:
    assert resolve_llm_stream_retry_policy(None).max_attempts == DEFAULT_MAX_ATTEMPTS
    assert resolve_llm_stream_retry_policy(tmp_path).max_attempts == DEFAULT_MAX_ATTEMPTS

    cfg_dir = tmp_path / ".llgraph"
    cfg_dir.mkdir()
    (cfg_dir / "agent.json").write_text(
        json.dumps({"llm": {"stream_retry": {"max_attempts": 4, "base_delay_sec": 0.5}}}),
        encoding="utf-8",
    )
    policy = resolve_llm_stream_retry_policy(tmp_path)
    assert policy.max_attempts == 4
    assert policy.base_delay_sec == 0.5


def test_policy_rejects_garbage(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".llgraph"
    cfg_dir.mkdir()
    (cfg_dir / "agent.json").write_text(
        json.dumps({"llm": {"stream_retry": {"max_attempts": "many", "base_delay_sec": -9}}}),
        encoding="utf-8",
    )
    policy = resolve_llm_stream_retry_policy(tmp_path)
    assert policy.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert policy.base_delay_sec > 0


class _ScriptedRunnable:
    """按脚本吐 chunk / 抛异常的假 bound_model。"""

    def __init__(self, workspace: Path, scripts: list[list[Any]]) -> None:
        self.llgraph_workspace = workspace
        self._scripts = scripts
        self.calls = 0

    def stream(self, _state: Any, _config: Any):
        script = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1

        def _gen():
            for item in script:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return _gen()


def _text_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=[{"type": "text", "text": text, "index": 0}])


@pytest.fixture()
def fast_retry_workspace(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / ".llgraph"
    cfg_dir.mkdir()
    (cfg_dir / "agent.json").write_text(
        json.dumps(
            {"llm": {"stream_retry": {"max_attempts": 3, "base_delay_sec": 0.05, "max_delay_sec": 0.05}}}
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_stream_replays_transient_failure(fast_retry_workspace: Path) -> None:
    from llgraph.core.react_invoke import _consume_runnable_stream

    runnable = _ScriptedRunnable(
        fast_retry_workspace,
        [
            [RemoteProtocolError("server disconnected without sending a response")],
            [_text_chunk("ok")],
        ],
    )
    response = _consume_runnable_stream(runnable, {"messages": []}, {})
    assert runnable.calls == 2
    assert "ok" in str(response.content)


def test_stream_gives_up_after_budget(fast_retry_workspace: Path) -> None:
    from llgraph.core.react_invoke import _consume_runnable_stream

    runnable = _ScriptedRunnable(
        fast_retry_workspace,
        [[RemoteProtocolError("connection reset by peer")]],
    )
    with pytest.raises(RemoteProtocolError):
        _consume_runnable_stream(runnable, {"messages": []}, {})
    assert runnable.calls == 3


def test_stream_does_not_replay_after_visible_text(fast_retry_workspace: Path) -> None:
    """正文已经流到终端，重放会让用户看到半截重复。"""
    from llgraph.core.react_invoke import _consume_runnable_stream

    runnable = _ScriptedRunnable(
        fast_retry_workspace,
        [[_text_chunk("已改好 "), RemoteProtocolError("connection reset by peer")]],
    )
    with pytest.raises(RemoteProtocolError):
        _consume_runnable_stream(runnable, {"messages": []}, {})
    assert runnable.calls == 1


def test_fatal_error_not_replayed(fast_retry_workspace: Path) -> None:
    from llgraph.core.react_invoke import _consume_runnable_stream

    runnable = _ScriptedRunnable(fast_retry_workspace, [[BadRequestError("tool schema invalid")]])
    with pytest.raises(BadRequestError):
        _consume_runnable_stream(runnable, {"messages": []}, {})
    assert runnable.calls == 1


def test_empty_gateway_response_is_replayed(fast_retry_workspace: Path) -> None:
    """空 body 以前被当成用户 Stop 静默收场，至少要先重放。"""
    from llgraph.core.react_invoke import _consume_runnable_stream

    runnable = _ScriptedRunnable(fast_retry_workspace, [[], [_text_chunk("ok")]])
    response = _consume_runnable_stream(runnable, {"messages": []}, {})
    assert runnable.calls == 2
    assert "ok" in str(response.content)


def test_persistent_empty_response_still_ends_as_cancelled(fast_retry_workspace: Path) -> None:
    """重放耗尽后维持旧的收场语义，不把内部异常泄到 CLI。"""
    from llgraph.core.react_invoke import _consume_runnable_stream
    from llgraph.session.session_run_log import UserCancelledError

    runnable = _ScriptedRunnable(fast_retry_workspace, [[]])
    with pytest.raises(UserCancelledError):
        _consume_runnable_stream(runnable, {"messages": []}, {})
    assert runnable.calls == 3


def test_retry_backoff_is_interruptible_by_stop(
    fast_retry_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llgraph.core import react_invoke
    from llgraph.session.session_run_log import UserCancelledError

    runnable = _ScriptedRunnable(
        fast_retry_workspace,
        [[RemoteProtocolError("connection reset by peer")], [_text_chunk("ok")]],
    )
    state = {"cancelled": False}

    def _cancel_requested() -> bool:
        # 首次流式失败后才置位：模拟用户在退避窗口里按了 Stop
        if runnable.calls >= 1:
            state["cancelled"] = True
        return state["cancelled"]

    monkeypatch.setattr(react_invoke, "agent_cancel_requested", _cancel_requested)
    with pytest.raises(UserCancelledError):
        react_invoke._consume_runnable_stream(runnable, {"messages": []}, {})
    assert runnable.calls == 1
