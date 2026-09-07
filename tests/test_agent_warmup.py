"""LazyAgent：交互模式让 banner 先出来，Agent 在后台建。

关注三件事：转发要透明（调用方拿不到区别）、只建一次、后台失败要能在主线程
原样抛出（否则「缺少环境变量」这类提示会变成首轮才报、且吞在后台线程里）。
"""

from __future__ import annotations

import threading

import pytest

from llgraph.runtime.agent_warmup import LazyAgent, unwrap_agent


class _FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get_state(self, config):
        self.calls.append(("get_state", config))
        return {"values": {"messages": []}}

    def update_state(self, config, patch):
        self.calls.append(("update_state", config, patch))


def test_forwards_attributes_to_real_agent() -> None:
    real = _FakeAgent()
    lazy = LazyAgent(lambda: real, background=False)

    assert lazy.get_state({"thread_id": "t1"}) == {"values": {"messages": []}}
    lazy.update_state({"thread_id": "t1"}, {"messages": []})
    assert real.calls == [
        ("get_state", {"thread_id": "t1"}),
        ("update_state", {"thread_id": "t1"}, {"messages": []}),
    ]


def test_builder_runs_only_once_under_concurrent_access() -> None:
    counter = {"n": 0}
    gate = threading.Event()

    def builder():
        counter["n"] += 1
        gate.wait(5)
        return _FakeAgent()

    lazy = LazyAgent(builder, background=False)
    resolved: list = []
    threads = [
        threading.Thread(target=lambda: resolved.append(lazy.resolve())) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join(10)

    assert counter["n"] == 1
    assert len({id(a) for a in resolved}) == 1


def test_background_build_is_reused_by_resolve() -> None:
    counter = {"n": 0}

    def builder():
        counter["n"] += 1
        return _FakeAgent()

    lazy = LazyAgent(builder)
    first = lazy.resolve()

    assert lazy.ready is True
    assert lazy.resolve() is first
    assert counter["n"] == 1


def test_background_failure_reraises_on_main_thread() -> None:
    """后台线程吞掉异常，resolve() 在调用点重跑并原样抛出。"""
    attempts = {"n": 0}

    def builder():
        attempts["n"] += 1
        raise RuntimeError("缺少环境变量: LLGRAPH_API_KEY")

    lazy = LazyAgent(builder)
    with pytest.raises(RuntimeError, match="缺少环境变量"):
        lazy.resolve()
    assert attempts["n"] >= 2
    assert lazy.ready is False


def test_unwrap_agent_passes_through_plain_agent() -> None:
    real = _FakeAgent()
    assert unwrap_agent(real) is real
    assert unwrap_agent(LazyAgent(lambda: real, background=False)) is real
