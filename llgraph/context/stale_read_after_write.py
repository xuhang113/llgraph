"""写入后作废旧 read 快照：对齐 Cursor / Claude Code / Codex 的改码上下文。

出站去重只处理「后来又 read 了同一段」；文件被 search_replace / write / append
改过之后，写入前的全文仍会进下一轮 LLM，既浪费 token，又让模型拿过期行号/正文
去拼 old_string。本模块仅改出站消息，不改落盘。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from llgraph.context.chat_history_repair import ai_message_tool_calls
from llgraph.context.read_segment_dedupe import extract_read_segments
from llgraph.core.write_failure_tracker import WRITE_TOOL_NAMES
from llgraph.core.write_serialize import normalize_write_path

STALE_AFTER_WRITE_MARKER = "[历史 read 已失效]"

_READ_TOOL_NAMES = frozenset({"read_file", "read_files"})
_WRITE_PATH_RE = re.compile(
    r"^(?:已写入|已追加|已替换)\s+(.+?)(?:（|\s+\(|$)",
)
_READ_RANGE_HDR = re.compile(
    r"^---\s+(.+?)\s+\(行\s+(\d+)-(\d+)\s+/ 共\s+\d+\s+行\)",
    re.MULTILINE,
)
_ARCHIVED_MARKERS = (
    "[历史",
    "[工具结果已落盘",
    "已 superseded",
)


def _content_text(msg: ToolMessage) -> str:
    raw = msg.content
    return raw if isinstance(raw, str) else str(raw or "")


def _is_archived(content: str) -> bool:
    return any(marker in content for marker in _ARCHIVED_MARKERS)


def _write_succeeded(name: str, content: str) -> bool:
    if name not in WRITE_TOOL_NAMES:
        return False
    return content.startswith(("已写入", "已追加", "已替换"))


def _path_from_write_content(content: str) -> str:
    match = _WRITE_PATH_RE.match(content.strip())
    if not match:
        return ""
    return normalize_write_path(match.group(1).strip())


def _path_from_call_args(call: Any) -> str:
    if isinstance(call, dict):
        args = call.get("args")
        cid_name = str(call.get("name") or "")
    else:
        args = getattr(call, "args", None)
        cid_name = str(getattr(call, "name", "") or "")
    if cid_name not in WRITE_TOOL_NAMES:
        return ""
    if isinstance(args, str):
        from llgraph.core.tool_arg_coerce import maybe_parse_json

        args = maybe_parse_json(args)
    if not isinstance(args, dict):
        return ""
    if cid_name:
        from llgraph.core.tool_arg_coerce import coerce_tool_args

        args = coerce_tool_args(cid_name, args)
    return normalize_write_path(args.get("path"))


def collect_write_success_paths(messages: list[BaseMessage]) -> list[tuple[int, str]]:
    """
    收集成功写入的 (消息下标, 归一化路径)，按出现顺序。

    @param messages 出站消息
    @return 写入记录
    """
    pending: dict[str, str] = {}
    out: list[tuple[int, str]] = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            for call in ai_message_tool_calls(msg):
                cid = ""
                if isinstance(call, dict):
                    cid = str(call.get("id") or "").strip()
                else:
                    cid = str(getattr(call, "id", "") or "").strip()
                path = _path_from_call_args(call)
                if cid and path:
                    pending[cid] = path
            continue
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "")
        content = _content_text(msg)
        if not _write_succeeded(name, content):
            continue
        cid = str(getattr(msg, "tool_call_id", "") or "").strip()
        path = pending.get(cid) or _path_from_write_content(content)
        if path:
            out.append((idx, path))
    return out


def format_stale_read_pointer(
    segments: list[tuple[str, int, int]],
    *,
    write_paths: tuple[str, ...] = (),
) -> str:
    """
    @param segments 被作废的行段
    @param write_paths 导致作废的写入路径
    @return 短指针
    """
    if not segments:
        target = write_paths[0] if write_paths else ""
        hint = f"`{target}`" if target else "先前 read"
    elif len(segments) == 1:
        path, start, end = segments[0]
        hint = f"`{path}` 行 {start}-{end}"
    else:
        path, start, end = segments[0]
        hint = f"`{path}` 行 {start}-{end} 等 {len(segments)} 段"
    written = "、".join(f"`{p}`" for p in write_paths[:4]) or "该文件"
    return (
        f"{STALE_AFTER_WRITE_MARKER} {hint} 已在后续写入（{written}）后过期；"
        "禁止用此快照作为 search_replace 的 old_string。"
        "请以写入工具返回的「写入后快照」为准，必要时再 read_file。"
    )


def split_read_output_blocks(content: str) -> tuple[str, list[tuple[str, str]]]:
    """
    将 read_file / read_files 输出拆成前缀 + [(path, block), ...]。

    @param content 工具输出
    @return (header 前缀, 文件块)
    """
    matches = list(_READ_RANGE_HDR.finditer(content))
    if not matches:
        return content, []
    prefix = content[: matches[0].start()]
    blocks: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        path = match.group(1).strip()
        blocks.append((path, content[match.start() : end].rstrip("\n")))
    return prefix, blocks


def mask_stale_read_content(
    content: str,
    stale_paths: set[str],
) -> str | None:
    """
    去掉写入后已过期的文件块。

    @param content read 工具输出
    @param stale_paths 归一化后的过期路径
    @return 新正文；整段作废时返回 None
    """
    segments = extract_read_segments(content)
    if not segments:
        return content
    stale_segs = [
        seg for seg in segments if normalize_write_path(seg[0]) in stale_paths
    ]
    if not stale_segs:
        return content
    live_segs = [
        seg for seg in segments if normalize_write_path(seg[0]) not in stale_paths
    ]
    if not live_segs:
        return None
    _prefix, blocks = split_read_output_blocks(content)
    kept: list[str] = []
    dropped: list[tuple[str, int, int]] = []
    seg_iter = list(segments)
    seg_i = 0
    for path, block in blocks:
        is_stale = normalize_write_path(path) in stale_paths
        if is_stale:
            if seg_i < len(seg_iter):
                dropped.append(seg_iter[seg_i])
            seg_i += 1
            continue
        kept.append(block)
        seg_i += 1
    if not kept:
        return None
    note = format_stale_read_pointer(
        dropped or stale_segs,
        write_paths=tuple(sorted(stale_paths)),
    )
    header = f"批量读取（已去掉 {len(dropped) or len(stale_segs)} 个写入后失效文件）:\n\n"
    return header + "\n\n".join(kept) + "\n\n" + note


def invalidate_reads_after_writes_for_dispatch(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """
    出站：写入成功之后，将该 path 上更早的 read 全文换成短指针。

    写入之后的 read 保留（新快照）。只改返回列表，不改入参。

    @param messages 已 prune / canonical 的消息
    @return 作废过期 read 后的消息
    """
    writes = collect_write_success_paths(messages)
    if not writes:
        return messages

    changed = False
    new_messages: list[BaseMessage] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            new_messages.append(msg)
            continue
        name = str(getattr(msg, "name", "") or "")
        if name not in _READ_TOOL_NAMES:
            new_messages.append(msg)
            continue
        content = _content_text(msg)
        if _is_archived(content):
            new_messages.append(msg)
            continue
        stale_paths = {path for w_idx, path in writes if w_idx > idx}
        if not stale_paths:
            new_messages.append(msg)
            continue
        segments = extract_read_segments(content)
        hit_paths = {
            normalize_write_path(path)
            for path, _s, _e in segments
            if normalize_write_path(path) in stale_paths
        }
        if not hit_paths:
            new_messages.append(msg)
            continue
        masked = mask_stale_read_content(content, hit_paths)
        if masked is None:
            pointer = format_stale_read_pointer(
                segments if all(normalize_write_path(p) in hit_paths for p, _s, _e in segments) else [
                    seg for seg in segments if normalize_write_path(seg[0]) in hit_paths
                ],
                write_paths=tuple(sorted(hit_paths)),
            )
            new_messages.append(
                ToolMessage(
                    content=pointer,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
            )
            changed = True
            continue
        if masked != content:
            new_messages.append(
                ToolMessage(
                    content=masked,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
            )
            changed = True
            continue
        new_messages.append(msg)
    return new_messages if changed else messages
