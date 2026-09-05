"""对话缓存断点位置：断点必须落在「下一步仍原样存在」的块上。

Anthropic 只在 cache_control 处写入缓存前缀，读取时匹配的是**扁平 block 序列**的
精确前缀。原实现用 top-level cache_control，langchain-anthropic 会把断点放在出站
最后一块 —— 那是 ephemeral 的 <system-reminder>，下一步就没了，于是写进去的前缀
永远读不回来（实测缓存命中 0%）。本文件用 langchain-anthropic 真正的序列化器
_format_messages 把消息展平成 block 序列，直接断言前缀能对上。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_anthropic.chat_models import _format_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llgraph.core.prompt_cache import (
    MAX_MESSAGE_BREAKPOINTS,
    apply_message_cache_breakpoints,
    last_stable_message_index,
)

_CC = {"type": "ephemeral", "ttl": "5m"}
_REMINDER = "<system-reminder>\n工具往返预算：上限 100；已用 1。\n</system-reminder>"


def _flatten(messages: list) -> list[tuple[str, bool]]:
    """展平为 (去掉 cache_control 的块指纹, 是否断点)，即 Anthropic 眼里的 prompt。"""
    _system, formatted = _format_messages(messages)
    out: list[tuple[str, bool]] = []
    for msg in formatted:
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        for block in blocks:
            if not isinstance(block, dict):
                block = {"type": "text", "text": str(block)}
            body = {k: v for k, v in block.items() if k != "cache_control"}
            out.append(
                (json.dumps(body, sort_keys=True, ensure_ascii=False), bool(block.get("cache_control")))
            )
    return out


def _common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x[0] != y[0]:
            break
        n += 1
    return n


def _tool_round(i: int) -> list:
    return [
        AIMessage(
            content="（调用工具中，无可见正文。）",
            tool_calls=[{"name": "read_file", "args": {"target_file": f"src/m{i}.py"}, "id": f"c{i}"}],
        ),
        ToolMessage(
            content=f"--- src/m{i}.py (行 1-80 / 共 80 行) ---\n"
            + "\n".join(f"{n}| body-{i}-{n}" for n in range(1, 81)),
            tool_call_id=f"c{i}",
            name="read_file",
        ),
    ]


def _dispatch(steps: int) -> list[list]:
    """模拟 steps 步工具循环，每步返回打好断点的出站列表（尾部带 ephemeral 提醒）。"""
    history: list = [SystemMessage(content="规范"), HumanMessage(content="逐个读模块")]
    out: list[list] = []
    for i in range(1, steps + 1):
        history = [*history, *_tool_round(i)]
        outbound = [*history, HumanMessage(content=_REMINDER)]
        out.append(apply_message_cache_breakpoints(outbound, cache_control=_CC))
    return out


def test_breakpoint_not_on_ephemeral_reminder() -> None:
    messages = [
        HumanMessage(content="q"),
        ToolMessage(content="结果正文", tool_call_id="t1", name="grep_files"),
        HumanMessage(content=_REMINDER),
    ]
    assert last_stable_message_index(messages) == 1
    tagged = apply_message_cache_breakpoints(messages, cache_control=_CC)
    blocks = _flatten(tagged)
    bp_positions = [i for i, (_h, bp) in enumerate(blocks) if bp]
    assert bp_positions, "至少要有一个断点"
    assert bp_positions[-1] < len(blocks) - 1, "断点不能落在最后一块（ephemeral 提醒）"
    # 最后一块正是提醒本身
    assert "system-reminder" in blocks[-1][0]
    assert not blocks[-1][1]


def test_breakpoint_lands_on_tool_result() -> None:
    tagged = apply_message_cache_breakpoints(
        [
            HumanMessage(content="q"),
            ToolMessage(content="结果正文", tool_call_id="t1", name="grep_files"),
            HumanMessage(content=_REMINDER),
        ],
        cache_control=_CC,
    )
    _system, formatted = _format_messages(tagged)
    tool_results = [
        b
        for m in formatted
        for b in (m.get("content") or [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0].get("cache_control") == _CC, "cache_control 应提升到 tool_result 层"


def test_turn_anchor_gets_second_breakpoint() -> None:
    tagged = apply_message_cache_breakpoints(
        [
            SystemMessage(content="规范"),
            HumanMessage(content="本轮问题"),
            *_tool_round(1),
            HumanMessage(content=_REMINDER),
        ],
        cache_control=_CC,
    )
    assert sum(1 for _h, bp in _flatten(tagged) if bp) == 2
    assert MAX_MESSAGE_BREAKPOINTS == 2


def test_prefix_is_stable_across_tool_loop_steps() -> None:
    """核心回归：每步出站的 block 前缀必须覆盖上一步断点位置，否则缓存读不回来。"""
    dispatches = _dispatch(6)
    for step, (prev, cur) in enumerate(zip(dispatches, dispatches[1:]), start=2):
        pb, cb = _flatten(prev), _flatten(cur)
        last_bp = max(i for i, (_h, bp) in enumerate(pb) if bp)
        lcp = _common_prefix(pb, cb)
        assert lcp > last_bp, (
            f"第 {step} 步：公共前缀 {lcp} 未覆盖上一步断点 {last_bp}，缓存必然 0 命中"
        )


def test_tool_result_shape_identical_tagged_and_untagged() -> None:
    """带断点与不带断点的 tool_result 正文形状必须一致（曾因 str→block 转换而分叉）。"""
    dispatches = _dispatch(3)
    tagged_step, later_step = _flatten(dispatches[1]), _flatten(dispatches[2])
    # 第 2 步的最后一个断点块，在第 3 步里应以完全相同的正文出现在同一位置
    last_bp = max(i for i, (_h, bp) in enumerate(tagged_step) if bp)
    assert tagged_step[last_bp][1] is True
    assert later_step[last_bp][0] == tagged_step[last_bp][0]
    assert later_step[last_bp][1] is False, "同一块在下一步不应再带断点"


def test_breakpoint_count_never_exceeds_budget() -> None:
    """system + tools 已占 2 个，消息级最多再用 2 个，否则 Anthropic 直接报错。"""
    for steps in (1, 3, 7):
        for messages in _dispatch(steps):
            assert sum(1 for _h, bp in _flatten(messages) if bp) <= MAX_MESSAGE_BREAKPOINTS


def test_no_breakpoint_when_only_ephemeral_messages() -> None:
    messages = [SystemMessage(content="规范"), HumanMessage(content=_REMINDER)]
    assert last_stable_message_index(messages) is None
    assert apply_message_cache_breakpoints(messages, cache_control=_CC) == messages


def test_disabled_by_settings(tmp_path: Path) -> None:
    from llgraph.core.prompt_cache import apply_cache_breakpoints_for_dispatch

    (tmp_path / ".llgraph").mkdir()
    (tmp_path / ".llgraph" / "agent.json").write_text(
        json.dumps({"context": {"prompt_cache": {"enabled": False}}}), encoding="utf-8"
    )
    messages = [HumanMessage(content="q"), ToolMessage(content="r", tool_call_id="t", name="g")]
    out = apply_cache_breakpoints_for_dispatch(messages, workspace=tmp_path, model_id="claude-x")
    assert out == messages


@pytest.mark.parametrize("steps", [2, 5, 9])
def test_cache_hit_ratio_is_high_on_append_only_loop(steps: int) -> None:
    """只追加的工具循环里，缓存可读比例应接近上一步全量。"""
    dispatches = _dispatch(steps)
    hit = total = 0
    for prev, cur in zip(dispatches, dispatches[1:]):
        pb, cb = _flatten(prev), _flatten(cur)
        last_bp = max(i for i, (_h, bp) in enumerate(pb) if bp)
        lcp = _common_prefix(pb, cb)
        hit += (last_bp + 1) if lcp > last_bp else 0
        total += len(cb)
    if steps >= 5:
        assert hit / total > 0.6, f"命中块占比过低: {hit}/{total}"


def _readable_prefix_per_step(workspace: Path, steps: int, chars: int) -> tuple[list[int], float]:
    """走真实出站管线，按 Anthropic 语义算每步能读回多少块。

    Anthropic 在断点位置写下的条目覆盖块 [0, B]，下一次请求只有自己的 [0, B]
    与之逐字节全等才能读回该条目——不是任意最长公共前缀。

    @param workspace 工作区根（提供 .llgraph/agent.json）
    @param steps 工具循环步数
    @param chars 每步工具正文字符数
    @return (每步可读回块数, 总可读字符占比)
    """
    from llgraph.context.dispatch_compaction import reset_dispatch_compaction_state
    from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch
    from llgraph.context.runtime_context import set_active_thread_id

    reset_dispatch_compaction_state()
    set_active_thread_id("cache-growth")
    try:
        history: list = [HumanMessage(content="逐个读模块，找出 token 刷新逻辑")]
        entries: list[tuple[int, list[str]]] = []
        readable: list[int] = []
        total = read_total = 0

        for i in range(steps):
            outbound = prepare_messages_for_llm_dispatch(
                history,
                agent_system_content="你是 llgraph。" + "规范。" * 100,
                workspace=workspace,
                model_id="claude-sonnet-4-6",
            )
            system, _formatted = _format_messages(outbound)
            seq = _flatten_with_system(system, outbound)
            bodies = [h for h, _bp in seq]
            sizes = [len(h) for h in bodies]
            msg_bps = sum(1 for _h, bp in _flatten(outbound) if bp)
            assert msg_bps <= MAX_MESSAGE_BREAKPOINTS, f"第 {i} 步消息级断点超预算: {msg_bps}"

            best = 0
            for upto, prefix in entries:
                if upto + 1 > best and bodies[: upto + 1] == prefix:
                    best = upto + 1
            readable.append(best)
            read_total += sum(sizes[:best])
            total += sum(sizes)
            for pos, (_h, bp) in enumerate(seq):
                if bp:
                    entries.append((pos, bodies[: pos + 1]))

            history = [
                *history,
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read_file", "args": {"target_file": f"src/m{i}.py"}, "id": f"c{i}"}
                    ],
                ),
                ToolMessage(
                    content=f"--- src/m{i}.py ---\n" + f"line{i} " * (chars // 7),
                    tool_call_id=f"c{i}",
                    name="read_file",
                ),
            ]
        return readable, read_total / total
    finally:
        set_active_thread_id(None)
        reset_dispatch_compaction_state()


def _flatten_with_system(system: object, messages: list) -> list[tuple[str, bool]]:
    """system 块 + 消息块，即 Anthropic 缓存前缀里除 tools 外的全部内容。"""
    out: list[tuple[str, bool]] = []
    if isinstance(system, list):
        for block in system:
            body = {k: v for k, v in block.items() if k != "cache_control"}
            out.append(
                (
                    json.dumps(body, sort_keys=True, ensure_ascii=False),
                    bool(block.get("cache_control")),
                )
            )
    elif system:
        out.append((str(system), False))
    return [*out, *_flatten(messages)]


@pytest.mark.parametrize("chars", [3_000, 12_000])
def test_readable_cache_prefix_grows_with_conversation(tmp_path: Path, chars: int) -> None:
    """核心回归：可读回的缓存前缀必须随对话**增长**，不能冻结在开头。

    这是原实现真正的失效形态，且两个修复缺一不可，所以必须走完整出站管线来测：
    - 断点打在 ephemeral 尾巴上 → 每步的条目下一步都作废，永远只读回 system 那 1 块；
    - recency 滑窗每步改写历史 → 断点修好了也只在头几步有效，之后塌回 2 块。
    修好后是「锯齿上升」：纪元内逐步增长，压缩那一步回落到本轮 user 断点再重新爬。
    """
    (tmp_path / ".llgraph").mkdir()
    (tmp_path / ".llgraph" / "agent.json").write_text(
        json.dumps({"context": {"dispatch_keep_full_tool_messages": 3}}), encoding="utf-8"
    )

    readable, ratio = _readable_prefix_per_step(tmp_path, steps=12, chars=chars)

    third = len(readable) // 3
    early, late = max(readable[:third]), max(readable[-third:])
    assert late > early * 2, f"可读前缀未随对话增长（前段 {early} 块 → 后段 {late} 块）: {readable}"
    # 冻结形态下这个比值只有 4% 上下；修好后实测约 70%
    assert ratio > 0.5, f"缓存可读字符占比过低: {ratio:.1%}，每步可读块数 {readable}"
