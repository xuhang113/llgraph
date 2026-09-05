"""出站 tool 链的「压缩纪元」：让 prompt cache 前缀在工具循环内保持稳定。

对标 Claude Code / Codex CLI：会话记录只追加，压缩是低频事件。
Anthropic prompt cache 按**精确前缀**命中，历史被改写位置之后的全部内容都要重新
计费与重算 TTFT。原出站实现按 recency 滑窗（永远只保留最近 N 条重工具结果全文），
每走一步就恰好有一条历史 ToolMessage 由全文变指针，等于每步都击穿缓存。

本模块把「保留最近 N 条」换成「超过高水位才压到低水位」：压缩点按纪元跳变，
纪元内出站前缀逐字节不变，因此除压缩那一步外，工具循环每步都能整段命中缓存。
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from math import ceil
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from llgraph.context.context_compressor import estimate_tokens
from llgraph.context.context_settings import ContextSettings

# 出站已压缩/拦截短文案：不占预算，也不再二次归档
_COMPACT_TOOL_MARKERS = (
    "[历史",
    "[工具结果已落盘",
    "[llgraph] 重复工具已拦截",
    "[llgraph] 重复失败已拦截",
    "已 superseded",
)

# 每个 thread 记住的「已压缩 tool_call_id」上限（防长会话无界增长）
_COMPACTED_ID_CAP = 8192
# 同时跟踪的 thread 数上限（LRU 淘汰）
_TRACKED_THREAD_CAP = 64


def tool_content_is_compact(content: str) -> bool:
    """
    @param content 工具正文
    @return 是否已是短指针/拦截文案（不计入全文预算）
    """
    return any(marker in content for marker in _COMPACT_TOOL_MARKERS)


def _tool_text(msg: ToolMessage) -> str:
    raw = msg.content
    return raw if isinstance(raw, str) else str(raw or "")


@dataclass(frozen=True)
class DispatchBudget:
    """出站全文工具结果的水位配置。"""

    high_tokens: int
    low_tokens: int
    high_count: int
    low_count: int

    @property
    def token_step(self) -> int:
        """@return 相邻纪元之间新增的全文 token 量"""
        return max(1, self.high_tokens - self.low_tokens)

    @property
    def count_step(self) -> int:
        """@return 相邻纪元之间新增的全文条数"""
        return max(1, self.high_count - self.low_count)


def resolve_dispatch_budget(settings: ContextSettings) -> DispatchBudget:
    """
    解析出站全文工具结果的高/低水位。

    低水位沿用 dispatch_keep_full_tool_messages（压缩后保留条数）；
    高水位 = 低水位 × dispatch_full_tool_hysteresis，滞回越大压缩越低频。

    @param settings 上下文配置
    @return DispatchBudget
    """
    low_count = max(1, settings.dispatch_keep_full_tool_messages)
    hysteresis = max(1.0, settings.dispatch_full_tool_hysteresis)
    high_count = max(low_count, ceil(low_count * hysteresis))

    high_tokens = max(1, settings.dispatch_full_tool_budget_tokens)
    low_ratio = min(0.95, max(0.05, settings.dispatch_compact_low_ratio))
    low_tokens = max(1, int(high_tokens * low_ratio))
    if low_tokens >= high_tokens:
        low_tokens = max(1, high_tokens - 1)
    return DispatchBudget(
        high_tokens=high_tokens,
        low_tokens=low_tokens,
        high_count=high_count,
        low_count=low_count,
    )


def quantized_compact_amount(total: int, *, low: int, high: int) -> int:
    """
    按纪元量化「应压缩掉的量」：total 跨过 high 才动，一次压到 low。

    纪元 e = floor((total - low) / (high - low))；返回 e × step。
    total 在 [low + e·step, low + (e+1)·step) 内取值时结果恒定，
    因此纪元内出站前缀不变——这正是 prompt cache 命中的前提。

    @param total 当前全文总量（token 或条数）
    @param low 压缩目标（低水位）
    @param high 触发压缩的阈值（高水位）
    @return 应压缩掉的量；未触发时 0
    """
    if total < high:
        return 0
    step = max(1, high - low)
    excess = total - low
    if excess <= 0:
        return 0
    return (excess // step) * step


@dataclass
class _ThreadCompactionState:
    """单 thread 的单调压缩水位（只增不减）。"""

    compacted_ids: OrderedDict[str, None]
    epoch: int = 0
    prefix_hashes: tuple[str, ...] = ()
    prefix_sizes: tuple[int, ...] = ()
    last_report: PrefixStabilityReport | None = None

    def remember(self, call_ids: list[str]) -> None:
        """@param call_ids 本次新压缩的 tool_call_id"""
        for cid in call_ids:
            self.compacted_ids.pop(cid, None)
            self.compacted_ids[cid] = None
        while len(self.compacted_ids) > _COMPACTED_ID_CAP:
            self.compacted_ids.popitem(last=False)


_STATES: OrderedDict[str, _ThreadCompactionState] = OrderedDict()
_STATES_LOCK = threading.Lock()


def _state_for(thread_id: str) -> _ThreadCompactionState:
    with _STATES_LOCK:
        state = _STATES.get(thread_id)
        if state is None:
            state = _ThreadCompactionState(compacted_ids=OrderedDict())
            _STATES[thread_id] = state
            while len(_STATES) > _TRACKED_THREAD_CAP:
                _STATES.popitem(last=False)
        else:
            _STATES.move_to_end(thread_id)
        return state


def reset_dispatch_compaction_state(thread_id: str | None = None) -> None:
    """
    清空压缩水位（/compress、会话切换、测试用）。

    @param thread_id 线程 ID；None 时清空全部
    """
    with _STATES_LOCK:
        if thread_id is None:
            _STATES.clear()
        else:
            _STATES.pop(thread_id, None)


@dataclass(frozen=True)
class CompactionPlan:
    """本次出站的压缩决策。"""

    compact_indices: frozenset[int]
    epoch: int
    heavy_tokens: int
    kept_tokens: int
    budget: DispatchBudget
    newly_compacted: int

    @property
    def triggered(self) -> bool:
        """@return 本次是否新压缩了历史工具结果（会击穿缓存前缀）"""
        return self.newly_compacted > 0


@dataclass(frozen=True)
class PrefixStabilityReport:
    """相邻两次出站的公共前缀（prompt cache 可命中部分）。"""

    messages: int
    stable_messages: int
    total_chars: int
    stable_chars: int
    first_changed_index: int | None

    @property
    def stable_ratio(self) -> float:
        """@return 可缓存前缀字符占比"""
        if self.total_chars <= 0:
            return 1.0
        return self.stable_chars / self.total_chars


def _heavy_tool_slots(
    messages: list[BaseMessage],
    *,
    pinned: set[int],
) -> list[tuple[int, str, int]]:
    """
    收集参与全文预算的「重」工具结果。

    @param messages 出站消息
    @param pinned 始终保留全文的下标（写入快照等）
    @return [(下标, tool_call_id, 估算 token)]，按出现顺序
    """
    slots: list[tuple[int, str, int]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage) or idx in pinned:
            continue
        content = _tool_text(msg)
        if tool_content_is_compact(content):
            continue
        slots.append(
            (idx, str(getattr(msg, "tool_call_id", "") or ""), estimate_tokens(content))
        )
    return slots


def _boundary_for_amount(
    amounts: list[int],
    target: int,
) -> int:
    """
    从最旧一侧累计压缩，直到达到 target。

    @param amounts 按时间顺序的每条量
    @param target 需要压缩掉的总量
    @return 压缩边界（前 boundary 条压缩）
    """
    if target <= 0:
        return 0
    acc = 0
    for i, amount in enumerate(amounts):
        if acc >= target:
            return i
        acc += amount
    return len(amounts)


def plan_dispatch_compaction(
    messages: list[BaseMessage],
    settings: ContextSettings,
    *,
    thread_id: str | None = None,
    pinned: set[int] | None = None,
    protected: set[int] | None = None,
) -> CompactionPlan:
    """
    决定本次出站要把哪些历史 ToolMessage 压成指针。

    与 recency 滑窗的区别：全文量在高水位以下时**一条都不新压**，出站字节与上一步
    完全一致，prompt cache 整段命中；跨过高水位才一次压到低水位（纪元 +1）。

    单调性：本 thread 压过的 tool_call_id 不会再被「引用保护」救回全文，
    否则一次晚到的引用就会让前缀回退，缓存与上下文双输。

    @param messages 已 canonical / repair 后的出站消息
    @param settings 上下文配置
    @param thread_id 会话线程；None 时为纯函数模式（不记水位）
    @param pinned 始终保留全文的下标（写入快照）
    @param protected 尚未压缩时可延后压缩的下标（下游引用）
    @return CompactionPlan
    """
    budget = resolve_dispatch_budget(settings)
    pinned_set = set(pinned or ())
    slots = _heavy_tool_slots(messages, pinned=pinned_set)
    heavy_tokens = sum(tokens for _idx, _cid, tokens in slots)

    state = _state_for(thread_id) if thread_id else None
    sticky_ids = set(state.compacted_ids) if state else set()

    # 已压缩过的条目：不参与预算，也不因引用保护复活
    forced: set[int] = set()
    live: list[tuple[int, str, int]] = []
    for idx, cid, tokens in slots:
        if cid and cid in sticky_ids:
            forced.add(idx)
            continue
        live.append((idx, cid, tokens))

    live_tokens = sum(tokens for _idx, _cid, tokens in live)
    token_target = quantized_compact_amount(
        live_tokens,
        low=budget.low_tokens,
        high=budget.high_tokens,
    )
    count_target = quantized_compact_amount(
        len(live),
        low=budget.low_count,
        high=budget.high_count,
    )
    boundary = max(
        _boundary_for_amount([tokens for _i, _c, tokens in live], token_target),
        _boundary_for_amount([1] * len(live), count_target),
    )

    protected_set = set(protected or ())
    newly: list[tuple[int, str]] = []
    for idx, cid, _tokens in live[:boundary]:
        if idx in protected_set:
            continue
        newly.append((idx, cid))

    compact_indices = forced | {idx for idx, _cid in newly}
    kept_tokens = sum(
        tokens for idx, _cid, tokens in slots if idx not in compact_indices
    )

    epoch = state.epoch if state else 0
    if state is not None and newly:
        state.remember([cid for _idx, cid in newly if cid])
        state.epoch += 1
        epoch = state.epoch

    return CompactionPlan(
        compact_indices=frozenset(compact_indices),
        epoch=epoch,
        heavy_tokens=heavy_tokens,
        kept_tokens=kept_tokens,
        budget=budget,
        newly_compacted=len(newly),
    )


def _message_fingerprint(msg: BaseMessage) -> tuple[str, int]:
    """
    @param msg 出站消息
    @return (内容指纹, 字符数)
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        body = content
    else:
        try:
            body = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            body = str(content)
    extra = ""
    calls = getattr(msg, "tool_calls", None)
    if calls:
        try:
            extra = json.dumps(calls, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            extra = str(calls)
    payload = f"{type(msg).__name__}\x00{body}\x00{extra}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return digest, len(body) + len(extra)


def record_dispatch_prefix(
    messages: list[BaseMessage],
    *,
    thread_id: str | None,
) -> PrefixStabilityReport | None:
    """
    记录本次出站指纹并给出与上一次的公共前缀（prompt cache 可命中量）。

    这是本模块唯一的效果度量：比值接近 1 说明工具循环里几乎整段命中缓存，
    掉到 0 附近说明历史又被中途改写了。

    @param messages 即将提交网关的消息
    @param thread_id 会话线程；None 时不记录
    @return 与上一次出站的前缀报告；首次调用返回 None
    """
    if not thread_id or not messages:
        return None
    fingerprints = [_message_fingerprint(m) for m in messages]
    hashes = tuple(h for h, _n in fingerprints)
    sizes = tuple(n for _h, n in fingerprints)
    total_chars = sum(sizes)

    state = _state_for(thread_id)
    prev_hashes = state.prefix_hashes
    state.prefix_hashes = hashes
    state.prefix_sizes = sizes
    if not prev_hashes:
        state.last_report = None
        return None

    stable = 0
    for a, b in zip(prev_hashes, hashes):
        if a != b:
            break
        stable += 1
    report = PrefixStabilityReport(
        messages=len(hashes),
        stable_messages=stable,
        total_chars=total_chars,
        stable_chars=sum(sizes[:stable]),
        first_changed_index=None if stable == len(prev_hashes) == len(hashes) else stable,
    )
    state.last_report = report
    return report


def last_dispatch_prefix_report(thread_id: str | None) -> PrefixStabilityReport | None:
    """
    @param thread_id 会话线程
    @return 最近一次出站的前缀报告
    """
    if not thread_id:
        return None
    with _STATES_LOCK:
        state = _STATES.get(thread_id)
    return state.last_report if state else None


def format_prefix_stability(report: PrefixStabilityReport | None) -> str:
    """
    @param report 前缀报告
    @return 单行摘要
    """
    if report is None:
        return "cache_prefix: 暂无（本会话仅一次出站）"
    return (
        f"cache_prefix: {report.stable_chars}/{report.total_chars} 字符稳定 "
        f"({report.stable_ratio * 100:.0f}%)，"
        f"{report.stable_messages}/{report.messages} 条未改写"
    )


def dispatch_compaction_debug(plan: CompactionPlan) -> dict[str, Any]:
    """
    @param plan 压缩决策
    @return 供 run_log 落盘的结构
    """
    return {
        "epoch": plan.epoch,
        "heavy_tokens": plan.heavy_tokens,
        "kept_tokens": plan.kept_tokens,
        "compacted": len(plan.compact_indices),
        "newly_compacted": plan.newly_compacted,
        "high_water_tokens": plan.budget.high_tokens,
        "low_water_tokens": plan.budget.low_tokens,
    }
