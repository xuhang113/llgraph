"""thinking 流式合并与 Trace 落盘补全。"""

from __future__ import annotations

from llgraph.display.trace_display import _merge_thinking_capture


def test_merge_thinking_cumulative_growth() -> None:
    assert _merge_thinking_capture("", "用户想知道") == "用户想知道"
    assert (
        _merge_thinking_capture("用户想知道", "用户想知道业务")
        == "用户想知道业务"
    )


def test_merge_thinking_rejects_shorter_prefix_replay() -> None:
    assert _merge_thinking_capture("用户想知道业务吗", "用户想知道") == "用户想知道业务吗"


def test_merge_thinking_appends_delta() -> None:
    # 网关按增量推送时，短片段应拼到已有缓冲后
    assert (
        _merge_thinking_capture(
            "用户想知道工作区中是否有网易云联名会员相关的业务。这是一个广搜/摸底类问题，应该优先",
            "使用 spawn_subagent 进行探索。",
        )
        == "用户想知道工作区中是否有网易云联名会员相关的业务。这是一个广搜/摸底类问题，应该优先使用 spawn_subagent 进行探索。"
    )
