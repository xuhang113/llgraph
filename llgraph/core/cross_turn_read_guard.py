"""跨轮重复读同一文件的短路径拦截（性能：token 浪费）。

`tool_loop_guard` 的去重窗口从**最近一条真实 user 消息**开始。用户追问一句
「再顺手改一下 B」，窗口就清零，模型往往把上一轮刚读过的同一批文件原样再读一遍：
一个 1400 行的源文件 ≈ 8000 token，重读一次就是一次全价 input，
还会把出站全文预算顶到高水位、触发一次压缩纪元——prompt cache 前缀跟着断一次。

本模块只在能证明「重读拿不到任何新信息」时才拦：

- 这次 read **真正会返回的行段**（与 `read_focus` 的折叠规则一致：大文件无行段时
  返回文件头 + 本问检索命中窗）已被更早的 read 结果整段覆盖；
- 那些历史结果的正文与**磁盘逐行一致**、文件总行数也没变；
- 它们**还是全文**（没被出站压缩 / checkpoint 掩码），模型确实还看得见；
- 该路径本会话没有成功写入过。

四条里任一条不成立就放行真实 read——宁可白读一次，也不能让模型拿过期正文拼 old_string。
另有兜底：同一文件在同一问里只拦一次，模型坚持再读就放行，避免「看不见正文又读不到」死转。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from llgraph.context.read_content_verify import (
    ReadBlock,
    parse_read_blocks,
    path_content_unchanged,
)

CROSS_TURN_READ_MARKER = "[llgraph] 跨轮重复读已拦截"

READ_TOOL_NAMES = frozenset({"read_file", "read_files"})

# 已被掩码 / 落盘 / 拦截的正文：拿不到原文，不能当作去重依据
_NON_FULL_PREFIXES = ("[历史", "[工具结果已落盘", "[llgraph]", "【llgraph")
_BLOCKED_PATH_RE = re.compile(r"^-\s+`([^`]+)`", re.MULTILINE)


@dataclass(frozen=True)
class CarryRead:
    """更早轮次里一个仍是全文的 read 文件块。"""

    call_id: str
    block: ReadBlock


@dataclass(frozen=True)
class CarryCoverage:
    """某个请求路径被历史 read 覆盖的结果。"""

    path: str
    total: int
    segments: tuple[tuple[int, int], ...]
    hits: tuple[CarryRead, ...]

    @property
    def call_ids(self) -> tuple[str, ...]:
        """@return 参与覆盖的 tool_call_id（按出现顺序去重）"""
        out: list[str] = []
        for hit in self.hits:
            if hit.call_id not in out:
                out.append(hit.call_id)
        return tuple(out)


def _tool_text(msg: BaseMessage) -> str:
    raw = getattr(msg, "content", "")
    return raw if isinstance(raw, str) else str(raw or "")


def _is_full_read_body(content: str) -> bool:
    return not content.lstrip().startswith(_NON_FULL_PREFIXES)


def last_real_user_index(messages: list[BaseMessage]) -> int:
    """
    @param messages 图消息
    @return 最近一条真实 user 的下标；没有则 -1
    """
    from llgraph.context.investigate_harness import is_ephemeral_harness_human

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, HumanMessage) and not is_ephemeral_harness_human(msg):
            return idx
    return -1


def collect_carry_reads(
    messages: list[BaseMessage],
    *,
    end_index: int,
) -> dict[str, list[CarryRead]]:
    """
    收集**更早轮次**里仍是全文的 read 结果，按路径归档。

    只看 ToolMessage 正文（自带路径、行段与总行数），不必回溯 tool_calls 参数。

    @param messages 图消息
    @param end_index 只扫这个下标之前的消息（通常是最近一条真实 user）
    @return 路径 → 文件块（按出现顺序，越后越新）
    """
    out: dict[str, list[CarryRead]] = {}
    if end_index <= 0:
        return out
    for msg in messages[:end_index]:
        if not isinstance(msg, ToolMessage):
            continue
        if str(getattr(msg, "name", "") or "") not in READ_TOOL_NAMES:
            continue
        content = _tool_text(msg)
        if not _is_full_read_body(content):
            continue
        cid = str(getattr(msg, "tool_call_id", "") or "").strip()
        if not cid:
            continue
        for block in parse_read_blocks(content):
            out.setdefault(block.path, []).append(CarryRead(call_id=cid, block=block))
    return out


def cross_turn_blocked_paths(messages: list[BaseMessage], *, start_index: int) -> set[str]:
    """
    本问内已经拦过一次跨轮重读的路径（兜底放行依据）。

    @param messages 图消息
    @param start_index 从这个下标开始扫（最近一条真实 user 之后）
    @return 路径集合
    """
    blocked: set[str] = set()
    for msg in messages[max(0, start_index) :]:
        if not isinstance(msg, ToolMessage):
            continue
        content = _tool_text(msg)
        if not content.lstrip().startswith(CROSS_TURN_READ_MARKER):
            continue
        blocked.update(match.strip() for match in _BLOCKED_PATH_RE.findall(content))
    return blocked


def written_paths(messages: list[BaseMessage]) -> set[str]:
    """
    @param messages 图消息
    @return 本会话里成功写入过的归一化路径（这些路径不参与跨轮去重）
    """
    from llgraph.context.stale_read_after_write import collect_write_success_paths

    return {path for _idx, path in collect_write_success_paths(messages)}


def request_segments(
    path: str,
    *,
    start: int,
    end: int,
    total: int,
    messages: list[BaseMessage],
) -> list[tuple[int, int]]:
    """
    这次 read 真正会返回的行段。

    必须与 `read_focus` 的折叠规则一致：大文件不带行段时返回的是「文件头 +
    本问检索命中窗」，而不是全文。按全文要求覆盖会让本层几乎永不生效；
    按折叠后的行段要求覆盖，才既准确又拦得住真正的重复。

    @param path 相对路径
    @param start 请求起始行
    @param end 请求结束行；<=0 表示到末尾
    @param total 当前总行数
    @param messages 图消息（取本问检索命中行）
    @return [(start, end), ...]；无法判定时为空
    """
    from llgraph.core.read_focus import HEAD_LINES, plan_hit_windows, should_focus_read

    if total <= 0:
        return []
    start_n = max(1, start)
    if not should_focus_read(start_line=start_n, end_line=end, total_lines=total):
        end_n = total if end <= 0 else min(end, total)
        return [(start_n, end_n)] if start_n <= end_n else []

    from llgraph.context.search_hit_lines import collect_search_hits_from_messages

    hits: set[int] = set()
    want = path.strip("/")
    for raw_path, line_nos in collect_search_hits_from_messages(messages).items():
        norm = raw_path.strip("/")
        if norm == want or norm.endswith("/" + want) or want.endswith("/" + norm):
            hits.update(line_nos)
    windows = plan_hit_windows(sorted(hits), total_lines=total)
    return [(1, min(HEAD_LINES, total)), *windows]


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _lookup_path(carry: dict[str, list[CarryRead]], path: str) -> list[CarryRead]:
    direct = carry.get(path)
    if direct:
        return direct
    # read 输出里的展示路径可能带 ./ 前缀或与请求写法不同，按尾部匹配兜底
    want = path.strip("/")
    for key, hits in carry.items():
        norm = key.strip("/")
        if norm == want or norm.endswith("/" + want) or want.endswith("/" + norm):
            return hits
    return []


def find_carry_coverage(
    carry: dict[str, list[CarryRead]],
    *,
    path: str,
    start: int,
    end: int,
    messages: list[BaseMessage],
) -> CarryCoverage | None:
    """
    历史 read 是否整段覆盖这次请求会返回的行段。

    允许多条历史结果拼起来覆盖（例如上一轮分两段读完同一个类），
    但它们必须报同一个总行数，否则说明文件已经变过。

    @param carry collect_carry_reads 结果
    @param path 请求路径
    @param start 请求起始行
    @param end 请求结束行；<=0 到末尾
    @param messages 图消息
    @return 覆盖结果；未覆盖时 None
    """
    hits = _lookup_path(carry, path)
    if not hits:
        return None
    totals = {hit.block.total for hit in hits}
    if len(totals) != 1:
        return None
    total = totals.pop()
    wanted = request_segments(path, start=start, end=end, total=total, messages=messages)
    if not wanted:
        return None

    merged = _merge_ranges([(hit.block.start, hit.block.end) for hit in hits])
    used: list[CarryRead] = []
    for seg_start, seg_end in wanted:
        if not any(lo <= seg_start and hi >= seg_end for lo, hi in merged):
            return None
        used.extend(
            hit
            for hit in hits
            if hit.block.end >= seg_start and hit.block.start <= seg_end
        )
    if not used:
        return None
    return CarryCoverage(
        path=hits[0].block.path,
        total=total,
        segments=tuple(wanted),
        hits=tuple(dict.fromkeys(used)),
    )


def find_carry_coverages(
    carry: dict[str, list[CarryRead]],
    *,
    paths: tuple[str, ...],
    start: int,
    end: int,
    messages: list[BaseMessage],
) -> list[CarryCoverage] | None:
    """
    @param carry collect_carry_reads 结果
    @param paths 请求路径（read_files 可能多个）
    @param start 请求起始行
    @param end 请求结束行
    @param messages 图消息
    @return 每个路径的覆盖结果；任一路径没覆盖则 None
    """
    if not paths:
        return None
    out: list[CarryCoverage] = []
    for path in paths:
        coverage = find_carry_coverage(
            carry,
            path=path,
            start=start,
            end=end,
            messages=messages,
        )
        if coverage is None:
            return None
        out.append(coverage)
    return out


def coverages_still_valid(
    coverages: list[CarryCoverage],
    *,
    workspace: Path,
    compacted_ids: frozenset[str],
    write_paths: set[str],
    skip_paths: set[str],
) -> bool:
    """
    覆盖结果能否拿来短路径拦截。

    @param coverages 覆盖结果
    @param workspace 工作区根
    @param compacted_ids 已被出站压缩的 tool_call_id（正文模型已看不见）
    @param write_paths 本会话成功写入过的路径
    @param skip_paths 本问已拦过一次的路径（兜底放行）
    @return 是否全部可用
    """
    if not coverages:
        return False
    for coverage in coverages:
        norm = coverage.path.strip("/")
        if coverage.path in skip_paths or norm in skip_paths:
            return False
        if coverage.path in write_paths or norm in write_paths:
            return False
        if any(cid in compacted_ids for cid in coverage.call_ids):
            return False
        blocks = [hit.block for hit in coverage.hits]
        if not path_content_unchanged(coverage.path, blocks, workspace):
            return False
    return True


def format_cross_turn_read_block(
    coverages: list[CarryCoverage],
    *,
    tool_name: str,
) -> str:
    """
    跨轮重读的短指针文案。

    @param coverages 覆盖结果
    @param tool_name 被拦截的工具名
    @return 占位正文
    """
    lines = [
        CROSS_TURN_READ_MARKER,
        f"{tool_name} 要读的行段在本会话前面已经读过，且磁盘正文与那次**逐行一致**"
        "（文件没被改动），因此不再把全文重复注入上下文：",
    ]
    for coverage in coverages:
        span = "、".join(f"{lo}-{hi}" for lo, hi in coverage.segments[:4])
        ids = "、".join(coverage.call_ids[:3])
        lines.append(
            f"- `{coverage.path}` 需要的行 {span}（共 {coverage.total} 行）"
            f"已在 tool_call_id={ids} 的结果里"
        )
    lines.append("请直接以上文那条 read 结果为原文作答或 search_replace（行号仍然有效）。")
    lines.append(
        "需要别的行段：带 start_line/end_line 精读未覆盖的区段。"
        "若上文那条结果已被压缩、你确实需要全文，原样再调一次即可——"
        "同一文件本问只拦这一次。"
    )
    return "\n".join(lines)
