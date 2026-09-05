"""出站压缩纪元：prompt cache 前缀稳定性回归。

对标 Claude Code / Codex CLI 的「只追加 + 低频压缩」。核心断言不是「压得多狠」，
而是「纪元内一条都不新压」——因为 Anthropic prompt cache 按精确前缀命中，
每步改写历史等于每步全价重算。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.context_settings import ContextSettings, resolve_context_settings
from llgraph.context.dispatch_compaction import (
    plan_dispatch_compaction,
    quantized_compact_amount,
    record_dispatch_prefix,
    reset_dispatch_compaction_state,
    resolve_dispatch_budget,
)
from llgraph.context.incremental_context import prune_tool_messages_for_dispatch
from llgraph.context.runtime_context import set_active_thread_id

_WS = Path("/tmp/llgraph-epoch-ws")


def _settings(**overrides: object) -> ContextSettings:
    base = dict(
        max_tokens_estimate=200_000,
        auto_compress_ratio=0.85,
        keep_recent_turns=2,
        keep_recent_token_ratio=0.25,
        compress_model=None,
        session_archive_on_compress=True,
        compress_tool_mask_max_chars=6000,
        read_tool_mask_max_chars=12000,
        tool_result_max_chars=12000,
        read_tool_result_max_chars=36000,
        read_file_max_bytes=600_000,
        read_file_max_lines=2000,
        tool_result_preview_lines=40,
        tool_result_preview_head_lines=25,
        spill_dir=".llgraph/context/tool-results",
        spill_enabled=True,
        spill_exempt_tools=(),
        budget_source="model",
        context_model_id="test",
        context_fallback_max_tokens=200_000,
        incremental_tool_prune=True,
        keep_recent_tool_messages=6,
        compress_trigger_max_tokens=None,
        session_history_search_enabled=True,
        session_history_search_top_k=8,
        compress_strategy="auto",
        compress_during_react=True,
        compress_summary_chunk_chars=120_000,
        dispatch_tool_chain_compress=True,
        dispatch_keep_full_tool_messages=2,
        dispatch_full_tool_hysteresis=2.5,
        dispatch_full_tool_budget_tokens=4_000,
        dispatch_compact_low_ratio=0.4,
        dispatch_dedupe_read_paths=False,
        grep_context_lines=5,
        grep_max_inline_chars=48_000,
        spill_hit_context_lines=100,
        tool_prune_token_ratio=0.7,
        protect_cited_tool_messages=True,
        max_protected_cited_tool_messages=8,
    )
    base.update(overrides)
    return ContextSettings(**base)


def _grep_round(i: int, *, chars: int = 3000) -> list[AIMessage | ToolMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "grep_files", "args": {"query": f"q{i}"}, "id": f"g{i}"}],
        ),
        ToolMessage(
            content=f"匹配结果（ripgrep）:\n\n--- src/F{i}.java:10 ---\n>>> 10| hit-{i}\n"
            + ("x" * chars),
            tool_call_id=f"g{i}",
            name="grep_files",
        ),
    ]


def _full_text_tool_ids(messages: list, settings: ContextSettings, thread_id: str | None) -> set[str]:
    out = prune_tool_messages_for_dispatch(messages, _WS, settings, thread_id=thread_id)
    return {
        str(m.tool_call_id)
        for m in out
        if isinstance(m, ToolMessage) and "[历史" not in str(m.content)
    }


@pytest.fixture(autouse=True)
def _clean_state():
    reset_dispatch_compaction_state()
    set_active_thread_id(None)
    yield
    reset_dispatch_compaction_state()
    set_active_thread_id(None)


def test_quantized_amount_only_moves_on_epoch_boundary() -> None:
    """低于高水位不动；跨过高水位一次压到低水位；纪元内恒定。"""
    low, high = 40, 100
    step = high - low
    assert quantized_compact_amount(0, low=low, high=high) == 0
    assert quantized_compact_amount(99, low=low, high=high) == 0
    # 刚跨过高水位：压掉 step，剩余正好回到低水位
    assert quantized_compact_amount(100, low=low, high=high) == step
    # 纪元内（100..159）目标恒定 → 出站前缀不变
    assert quantized_compact_amount(130, low=low, high=high) == step
    assert quantized_compact_amount(159, low=low, high=high) == step
    # 下一纪元
    assert quantized_compact_amount(160, low=low, high=high) == 2 * step


def test_quantized_amount_is_monotone() -> None:
    prev = 0
    for total in range(0, 600, 7):
        cur = quantized_compact_amount(total, low=40, high=100)
        assert cur >= prev
        prev = cur


def test_no_new_compaction_below_high_water() -> None:
    """高水位以下：连续多步出站结果逐字节不变（缓存整段命中的前提）。"""
    settings = _settings(dispatch_keep_full_tool_messages=4, dispatch_full_tool_hysteresis=2.5)
    messages: list = [HumanMessage(content="q")]
    snapshots: list[str] = []
    for i in range(4):
        messages += _grep_round(i, chars=200)
        out = prune_tool_messages_for_dispatch(messages, _WS, settings, thread_id="t-low")
        snapshots.append(
            json.dumps([str(m.content) for m in out[: 1 + 2 * i]], ensure_ascii=False)
        )
    # 每一步的公共前缀部分与上一步完全一致
    for older, newer in zip(snapshots, snapshots[1:]):
        assert newer.startswith(older[:-1])
    assert all("[历史" not in str(m.content) for m in out if isinstance(m, ToolMessage))


def test_epoch_compacts_to_low_water_when_high_water_crossed() -> None:
    """跨过条数高水位（2×2.5=5）时一次压到低水位 2 条。"""
    settings = _settings(dispatch_keep_full_tool_messages=2, dispatch_full_tool_hysteresis=2.5)
    messages: list = [HumanMessage(content="q")]
    for i in range(4):
        messages += _grep_round(i, chars=100)
    kept = _full_text_tool_ids(messages, settings, None)
    assert kept == {"g0", "g1", "g2", "g3"}, "4 条未到高水位 5，不应压缩"

    messages += _grep_round(4, chars=100)
    kept = _full_text_tool_ids(messages, settings, None)
    assert kept == {"g3", "g4"}, "第 5 条跨过高水位，应压到低水位 2 条"


def test_sliding_window_regression_hysteresis_one() -> None:
    """滞回 1.0 应精确复现旧 recency 滑窗（每步保留最近 N 条）。"""
    settings = _settings(
        dispatch_keep_full_tool_messages=2,
        dispatch_full_tool_hysteresis=1.0,
        dispatch_full_tool_budget_tokens=10_000_000,
    )
    messages: list = [HumanMessage(content="q")]
    for i in range(5):
        messages += _grep_round(i, chars=100)
        kept = _full_text_tool_ids(messages, settings, None)
        assert kept == {f"g{j}" for j in range(max(0, i - 1), i + 1)}


def test_epoch_reduces_history_rewrites_over_long_loop() -> None:
    """20 步工具循环：纪元压缩改写历史的步数应显著少于滑窗。"""

    def count_rewrites(hysteresis: float, budget: int) -> tuple[int, int]:
        thread = f"t-{hysteresis}-{budget}"
        reset_dispatch_compaction_state(thread)
        settings = _settings(
            dispatch_keep_full_tool_messages=2,
            dispatch_full_tool_hysteresis=hysteresis,
            dispatch_full_tool_budget_tokens=budget,
        )
        messages: list = [HumanMessage(content="q")]
        prev: set[str] | None = None
        rewrites = 0
        for i in range(20):
            messages += _grep_round(i, chars=1500)
            kept = _full_text_tool_ids(messages, settings, thread)
            if prev is not None and (prev - {f"g{i}"}) - kept:
                rewrites += 1
            prev = kept
        return rewrites, len(kept)

    sliding, _ = count_rewrites(1.0, 10_000_000)
    epoch, kept_n = count_rewrites(2.5, 4_000)
    assert sliding >= 15, "滑窗应几乎每步改写历史"
    assert epoch * 3 <= sliding, f"纪元压缩改写步数应远少于滑窗: {epoch} vs {sliding}"
    assert kept_n <= 12, "上下文仍需有界"


def test_compaction_is_monotone_even_when_cited_later() -> None:
    """已压缩条目不因后到的引用复活：否则前缀回退，缓存与上下文双输。"""
    thread = "t-monotone"
    settings = _settings(dispatch_keep_full_tool_messages=1, dispatch_full_tool_hysteresis=2.0)
    cited_body = (
        "匹配结果（ripgrep）:\n\n--- src/Foo.java:167 ---\n>>> 167| throw new BizException()\n"
        + ("x" * 3000)
    )
    messages: list = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "grep_files", "args": {}, "id": "old"}]),
        ToolMessage(content=cited_body, tool_call_id="old", name="grep_files"),
    ]
    for i in range(3):
        messages += _grep_round(i, chars=1500)

    kept = _full_text_tool_ids(messages, settings, thread)
    assert "old" not in kept, "跨过高水位后最旧一条应已压缩"

    # 模型此时才引用被压缩条目里的 path:line
    messages.append(AIMessage(content="结论：Foo.java:167 抛 BizException。"))
    messages += _grep_round(9, chars=1500)
    kept_after = _full_text_tool_ids(messages, settings, thread)
    assert "old" not in kept_after, "已压缩条目不得复活为全文"


def test_cited_message_protected_before_it_is_compacted() -> None:
    """尚未压缩时的引用保护保留（无 thread 的纯函数模式）。"""
    settings = _settings(dispatch_keep_full_tool_messages=1, dispatch_full_tool_hysteresis=1.0)
    cited = (
        "匹配结果（ripgrep）:\n\n--- src/Foo.java:167 ---\n>>> 167| throw new BizException()\n"
        + ("y" * 2000)
    )
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "grep_files", "args": {}, "id": "c1"}]),
        ToolMessage(content=cited, tool_call_id="c1", name="grep_files"),
        *_grep_round(2, chars=2000),
        *_grep_round(3, chars=20),
        AIMessage(content="结论：Foo.java:167 抛 BizException。"),
    ]
    kept = _full_text_tool_ids(messages, settings, None)
    assert "c1" in kept
    assert "g2" not in kept


def test_write_snapshot_still_pinned_across_epochs() -> None:
    """写入后快照按路径钉住最新一份，不因纪元推进丢掉改码依据。"""
    thread = "t-pin"
    settings = _settings(dispatch_keep_full_tool_messages=1, dispatch_full_tool_hysteresis=2.0)
    snapshot = (
        "已替换 src/Foo.java（1 处）\n"
        "--- src/Foo.java (行 1-2 / 共 2 行) [写入后快照] ---\n"
        "1| new line\n"
        "后续 search_replace 请以此快照为准，勿使用写入前的 read。"
    )
    messages: list = [
        HumanMessage(content="改 Foo"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_replace", "args": {"path": "src/Foo.java"}, "id": "w1"}],
        ),
        ToolMessage(content=snapshot, tool_call_id="w1", name="search_replace"),
    ]
    for i in range(6):
        messages += _grep_round(i, chars=1500)
        out = prune_tool_messages_for_dispatch(messages, _WS, settings, thread_id=thread)
        pinned = next(m for m in out if isinstance(m, ToolMessage) and m.tool_call_id == "w1")
        assert pinned.content == snapshot, f"第 {i} 步写入快照被压掉了"


def test_plan_reports_epoch_and_budget() -> None:
    settings = _settings(dispatch_keep_full_tool_messages=2, dispatch_full_tool_hysteresis=2.5)
    budget = resolve_dispatch_budget(settings)
    assert budget.low_count == 2
    assert budget.high_count == 5
    assert budget.low_tokens < budget.high_tokens

    messages: list = [HumanMessage(content="q")]
    for i in range(6):
        messages += _grep_round(i, chars=1500)
    plan = plan_dispatch_compaction(messages, settings, thread_id="t-plan")
    assert plan.triggered
    assert plan.kept_tokens <= plan.heavy_tokens
    assert plan.epoch == 1


def test_prefix_report_detects_append_only_growth() -> None:
    """只追加时前缀报告应显示公共前缀 = 上一次全部消息。"""
    base = [HumanMessage(content="q"), AIMessage(content="a")]
    assert record_dispatch_prefix(base, thread_id="t-prefix") is None
    grown = [*base, HumanMessage(content="b")]
    report = record_dispatch_prefix(grown, thread_id="t-prefix")
    assert report is not None
    assert report.stable_messages == 2
    assert report.stable_ratio > 0.5


def test_prefix_report_detects_history_rewrite() -> None:
    base = [HumanMessage(content="q"), AIMessage(content="x" * 1000)]
    record_dispatch_prefix(base, thread_id="t-rw")
    rewritten = [HumanMessage(content="q"), AIMessage(content="[历史已归档]")]
    report = record_dispatch_prefix(rewritten, thread_id="t-rw")
    assert report is not None
    assert report.stable_messages == 1
    assert report.first_changed_index == 1


def test_default_settings_enable_hysteresis(tmp_path: Path) -> None:
    """默认配置必须带滞回，否则等于回到每步击穿缓存。"""
    (tmp_path / ".llgraph").mkdir()
    (tmp_path / ".llgraph" / "agent.json").write_text("{}", encoding="utf-8")
    settings = resolve_context_settings(tmp_path)
    assert settings.dispatch_full_tool_hysteresis > 1.0
    assert settings.dispatch_full_tool_budget_tokens >= 8_000
    assert 0.05 <= settings.dispatch_compact_low_ratio <= 0.95
    budget = resolve_dispatch_budget(settings)
    assert budget.high_count > budget.low_count
