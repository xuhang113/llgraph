"""本问内重复工具短路径：对齐 Cursor / Claude Code / Codex 的工具循环治理。

出站 read 去重只能少发旧结果，拦不住模型再调一次、再塞一遍全文。
本模块在 ToolNode 执行前拦截：

- 只读工具：相同参数或已被更宽范围覆盖的调用直接返回短指针
- 写工具：相同参数已失败（或已成功写入相同内容）则禁止原样重试
- 同批 tool_calls 内的重复调用只执行首次
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import ai_message_tool_calls
from llgraph.context.investigate_harness import is_ephemeral_harness_human
from llgraph.core.write_failure_tracker import WRITE_TOOL_NAMES
from llgraph.core.write_serialize import normalize_write_path, write_path_from_call

IDENTICAL_BLOCK_MARKER = "[llgraph] 重复工具已拦截"
IDENTICAL_FAIL_MARKER = "[llgraph] 重复失败已拦截"

CACHEABLE_READ_TOOLS = frozenset(
    {
        "read_file",
        "read_files",
        "grep_files",
        "glob_files",
        "list_directory",
        "search_files",
        "search_workspace",
        "search_code_semantic",
        "search_code_parallel",
        "search_session_history",
        "search_memory",
        "web_search",
    }
)

_MCP_PREFIX = "mcp__"
_EXCERPT_LIMIT = 900
_EOF_LINE = 10**9

_WRITE_OK_PREFIXES = ("已写入", "已追加", "已替换")
_FAIL_MARKERS = (
    "错误:",
    "失败:",
    "缺少必填",
    "validation error",
    "field required",
    "未找到 old_string",
    "不唯一",
    "文件不存在",
    "路径不存在",
    "【llgraph 拦截】",
)


@dataclass(frozen=True)
class ToolFp:
    """工具调用指纹（精确去重键）。"""

    name: str
    key: tuple[Any, ...]


@dataclass
class ToolShape:
    """一次调用的结构化形态，供覆盖判定。"""

    fp: ToolFp
    name: str
    kind: str
    path: str = ""
    paths: tuple[str, ...] = ()
    start: int = 1
    end: int = 0
    pattern: str = ""
    glob: str = ""
    file_glob: str = ""
    summary: str = ""


@dataclass
class ToolRecord:
    """本问内一次已完成工具调用。"""

    shape: ToolShape
    call_id: str
    content: str
    failed: bool


@dataclass
class _HistoryIndex:
    """自最近 user 以来的已完成调用索引。"""

    exact: dict[ToolFp, ToolRecord] = field(default_factory=dict)
    reads: dict[str, list[ToolRecord]] = field(default_factory=dict)
    greps: list[ToolRecord] = field(default_factory=list)
    globs: list[ToolRecord] = field(default_factory=list)


def call_args(call: Any) -> dict[str, Any]:
    """
    取出 tool_call 参数 dict。

    @param call LangGraph / OpenAI 风格 tool_call
    @return 参数字典；无法解析则为空
    """
    raw: Any = None
    if isinstance(call, dict):
        raw = call.get("args")
        if raw is None:
            raw = call.get("arguments")
    else:
        raw = getattr(call, "args", None)
        if raw is None:
            raw = getattr(call, "arguments", None)
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        raw = {}
    name = call_name(call)
    if name:
        from llgraph.core.tool_arg_coerce import coerce_tool_args

        return coerce_tool_args(name, raw)
    return raw


def call_name(call: Any) -> str:
    """@return 工具名"""
    if isinstance(call, dict):
        return str(call.get("name") or "").strip()
    return str(getattr(call, "name", "") or "").strip()


def call_id(call: Any) -> str:
    """@return tool_call_id"""
    if isinstance(call, dict):
        return str(call.get("id") or "").strip()
    return str(getattr(call, "id", "") or "").strip()


def _int_arg(args: dict[str, Any], key: str, default: int) -> int:
    raw = args.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _norm_path(raw: object) -> str:
    path = normalize_write_path(raw)
    return path if path else "."


def _norm_paths(raw: object) -> tuple[str, ...]:
    from llgraph.core.tool_arg_coerce import coerce_path_list

    seen: set[str] = set()
    out: list[str] = []
    for item in coerce_path_list(raw):
        path = _norm_path(item)
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return tuple(out)


def _end_line(end: int) -> int:
    return _EOF_LINE if end <= 0 else end


def _range_covers(prior_start: int, prior_end: int, new_start: int, new_end: int) -> bool:
    return prior_start <= new_start and _end_line(prior_end) >= _end_line(new_end)


def _path_covers(prior_path: str, new_path: str) -> bool:
    prior = prior_path or "."
    new = new_path or "."
    if prior in {".", ""}:
        return True
    if new == prior:
        return True
    return new.startswith(prior.rstrip("/") + "/")


def _glob_covers(prior_glob: str, new_glob: str) -> bool:
    if prior_glob == new_glob:
        return True
    return (not prior_glob) and bool(new_glob)


def describe_shape(shape: ToolShape) -> str:
    """
    生成给人看的调用摘要。

    @param shape 结构化形态
    @return 短摘要
    """
    if shape.summary:
        return shape.summary
    return f"{shape.name}()"


def shape_from_call(call: Any) -> ToolShape | None:
    """
    将 tool_call 转为可缓存形态；副作用工具返回 None。

    @param call tool_call
    @return 形态；不可缓存则为 None
    """
    name = call_name(call)
    if not name or name.startswith(_MCP_PREFIX):
        return None
    if name not in CACHEABLE_READ_TOOLS and name not in WRITE_TOOL_NAMES:
        return None
    args = call_args(call)

    if name == "read_file":
        path = _norm_path(args.get("path"))
        start = max(1, _int_arg(args, "start_line", 1))
        end = max(0, _int_arg(args, "end_line", 0))
        end_label = "EOF" if end <= 0 else str(end)
        return ToolShape(
            fp=ToolFp(name, ("read", path, start, end)),
            name=name,
            kind="read",
            path=path,
            paths=(path,),
            start=start,
            end=end,
            summary=f'read_file(path="{path}", {start}-{end_label})',
        )

    if name == "read_files":
        paths = _norm_paths(args.get("paths"))
        if not paths:
            return None
        start = max(1, _int_arg(args, "start_line", 1))
        end = max(0, _int_arg(args, "end_line", 0))
        end_label = "EOF" if end <= 0 else str(end)
        return ToolShape(
            fp=ToolFp(name, ("read_files", paths, start, end)),
            name=name,
            kind="read",
            paths=paths,
            start=start,
            end=end,
            summary=f"read_files({len(paths)} files, {start}-{end_label})",
        )

    if name == "grep_files":
        pattern = str(args.get("pattern") or "").strip()
        path = _norm_path(args.get("path"))
        file_glob = str(args.get("file_glob") or "").strip()
        output_mode = str(args.get("output_mode") or "auto").strip().lower() or "auto"
        head_limit = _int_arg(args, "head_limit", 0)
        return ToolShape(
            fp=ToolFp(name, ("grep", pattern, path, file_glob, output_mode, head_limit)),
            name=name,
            kind="grep",
            path=path,
            pattern=pattern,
            file_glob=file_glob,
            summary=f'grep_files(pattern={pattern!r}, path="{path}")',
        )

    if name == "glob_files":
        glob_pat = str(args.get("glob_pattern") or args.get("pattern") or "").strip()
        path = _norm_path(args.get("path"))
        return ToolShape(
            fp=ToolFp(name, ("glob", glob_pat, path)),
            name=name,
            kind="glob",
            path=path,
            glob=glob_pat,
            summary=f'glob_files(glob_pattern={glob_pat!r}, path="{path}")',
        )

    if name == "list_directory":
        path = _norm_path(args.get("path"))
        return ToolShape(
            fp=ToolFp(name, ("list", path)),
            name=name,
            kind="list",
            path=path,
            summary=f'list_directory(path="{path}")',
        )

    if name in WRITE_TOOL_NAMES:
        path = _norm_path(args.get("path"))
        if name == "search_replace":
            hunks = _search_replace_hunks(args)
            return ToolShape(
                fp=ToolFp(name, ("search_replace", path, hunks)),
                name=name,
                kind="write",
                path=path,
                summary=f'search_replace(path="{path}")',
            )
        content = str(args.get("content") or "")
        return ToolShape(
            fp=ToolFp(name, (name, path, content)),
            name=name,
            kind="write",
            path=path,
            summary=f'{name}(path="{path}")',
        )

    # 其它只读工具：精确参数去重
    stable = _stable_args(args)
    return ToolShape(
        fp=ToolFp(name, (name, stable)),
        name=name,
        kind="other_read",
        summary=f"{name}()",
    )


def _search_replace_hunks(args: dict[str, Any]) -> tuple[tuple[str, str, bool], ...]:
    hunks: list[tuple[str, str, bool]] = []
    old = args.get("old_string")
    if isinstance(old, str) and old:
        hunks.append((old, str(args.get("new_string") or ""), bool(args.get("replace_all"))))
    raw = args.get("replacements")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            hunk_old = item.get("old_string")
            if hunk_old is None:
                hunk_old = item.get("oldString")
            if not isinstance(hunk_old, str) or not hunk_old:
                continue
            hunk_new = item.get("new_string")
            if hunk_new is None:
                hunk_new = item.get("newString")
            hunks.append(
                (
                    hunk_old,
                    str(hunk_new or ""),
                    bool(item.get("replace_all") or item.get("replaceAll")),
                )
            )
    return tuple(hunks)


def _stable_args(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(args)


def tool_result_failed(name: str, content: str) -> bool:
    """
    工具返回是否为失败（空 grep 不算失败，仍可缓存）。

    @param name 工具名
    @param content 返回正文
    @return 是否失败
    """
    text = str(content or "")
    if not text.strip():
        return False
    if name in WRITE_TOOL_NAMES and any(text.startswith(p) for p in _WRITE_OK_PREFIXES):
        return False
    lowered = text.lower()
    return any(m in text or m in lowered for m in _FAIL_MARKERS)


def _write_succeeded(name: str, content: str) -> bool:
    if name not in WRITE_TOOL_NAMES:
        return False
    text = str(content or "")
    return any(text.startswith(p) for p in _WRITE_OK_PREFIXES)


def _excerpt(content: str) -> str:
    text = str(content or "").strip()
    if len(text) <= _EXCERPT_LIMIT:
        return text
    head = 360
    tail = _EXCERPT_LIMIT - head - 8
    return text[:head].rstrip() + "\n…\n" + text[-tail:].lstrip()


def format_duplicate_block(
    shape: ToolShape,
    prior: ToolRecord,
    *,
    same_batch: bool,
) -> str:
    """
    重复只读/已成功写的拦截文案。

    @param shape 当前调用
    @param prior 先前记录
    @param same_batch 是否同批 tool_calls
    @return 短指针正文
    """
    where = "本批 tool_calls" if same_batch else "本问先前工具结果"
    prior_id = prior.call_id or "(无 id)"
    lines = [
        IDENTICAL_BLOCK_MARKER,
        f"{describe_shape(shape)} 已在{where}执行过（tool_call_id={prior_id}）。",
        "禁止用相同参数再调；请换 pattern/path/行段，或基于已有结果作答/search_replace。",
    ]
    if prior.failed:
        lines[0] = IDENTICAL_FAIL_MARKER
        lines[2] = "上次已失败。禁止原样重试；请改参数或换工具。"
    excerpt = _excerpt(prior.content)
    if excerpt and IDENTICAL_BLOCK_MARKER not in excerpt and IDENTICAL_FAIL_MARKER not in excerpt:
        lines.append("上次返回摘录:")
        lines.append(excerpt)
    return "\n".join(lines)


def format_fail_block(shape: ToolShape, prior: ToolRecord) -> str:
    """写工具相同参数失败后的拦截文案。"""
    lines = [
        IDENTICAL_FAIL_MARKER,
        f"{describe_shape(shape)} 已用相同参数失败过（tool_call_id={prior.call_id or '(无 id)'}）。",
        "禁止原样重试。请先 read_file 核对原文，改 old_string / replacements，或换策略。",
    ]
    excerpt = _excerpt(prior.content)
    if excerpt:
        lines.append("上次错误摘录:")
        lines.append(excerpt)
    return "\n".join(lines)


def _index_record(index: _HistoryIndex, record: ToolRecord) -> None:
    index.exact[record.shape.fp] = record
    shape = record.shape
    if shape.kind == "read" and not record.failed:
        for path in shape.paths:
            index.reads.setdefault(path, []).append(record)
    elif shape.kind == "grep" and not record.failed:
        index.greps.append(record)
    elif shape.kind == "glob" and not record.failed:
        index.globs.append(record)


def _invalidate_path(index: _HistoryIndex, path: str) -> None:
    if not path:
        return
    stale_fps = [
        fp
        for fp, rec in index.exact.items()
        if rec.shape.kind == "read" and path in rec.shape.paths
    ]
    for fp in stale_fps:
        index.exact.pop(fp, None)
    index.reads.pop(path, None)


def _lookup_cover(index: _HistoryIndex, shape: ToolShape) -> ToolRecord | None:
    exact = index.exact.get(shape.fp)
    if exact is not None:
        return exact
    if shape.kind == "read":
        if not shape.paths:
            return None
        hit: ToolRecord | None = None
        for path in shape.paths:
            covered = False
            for rec in index.reads.get(path, []):
                if _range_covers(rec.shape.start, rec.shape.end, shape.start, shape.end):
                    covered = True
                    hit = rec
                    break
            if not covered:
                return None
        return hit
    if shape.kind == "grep":
        for rec in index.greps:
            prior = rec.shape
            if prior.pattern != shape.pattern:
                continue
            if not _path_covers(prior.path, shape.path):
                continue
            if not _glob_covers(prior.file_glob, shape.file_glob):
                continue
            return rec
    if shape.kind == "glob":
        for rec in index.globs:
            prior = rec.shape
            if prior.glob != shape.glob:
                continue
            if _path_covers(prior.path, shape.path):
                return rec
    return None


def _batch_write_paths(calls: list[Any]) -> set[str]:
    paths: set[str] = set()
    for call in calls:
        item = dict(call) if isinstance(call, dict) else None
        if item is None:
            item = {
                "name": call_name(call),
                "id": call_id(call),
                "args": call_args(call),
            }
        path = write_path_from_call(item)
        if path:
            paths.add(path)
    return paths


def _read_hits_pending_write(shape: ToolShape, write_paths: set[str]) -> bool:
    if shape.kind != "read" or not write_paths:
        return False
    return any(path in write_paths for path in shape.paths)


def build_history_index(messages: list[BaseMessage]) -> _HistoryIndex:
    """
    扫描自最近真实 user 以来的已完成工具结果。

    @param messages 图状态消息（含本轮 AI tool_calls）
    @return 索引
    """
    start = 0
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, HumanMessage) and not is_ephemeral_harness_human(msg):
            start = idx + 1
            break
    index = _HistoryIndex()
    pending: dict[str, Any] = {}
    for msg in messages[start:]:
        if isinstance(msg, AIMessage):
            for call in ai_message_tool_calls(msg):
                cid = call_id(call)
                if cid:
                    pending[cid] = call
            continue
        if not isinstance(msg, ToolMessage):
            continue
        cid = str(getattr(msg, "tool_call_id", "") or "").strip()
        call = pending.get(cid)
        if call is None:
            continue
        shape = shape_from_call(call)
        if shape is None:
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        failed = tool_result_failed(shape.name, content)
        if _write_succeeded(shape.name, content):
            _invalidate_path(index, shape.path)
        record = ToolRecord(shape=shape, call_id=cid, content=content, failed=failed)
        _index_record(index, record)
    return index


def compute_blocked_tool_messages(
    messages: list[BaseMessage],
    calls: list[Any],
) -> dict[str, ToolMessage]:
    """
    计算本批应拦截的 tool_call_id → 占位 ToolMessage。

    @param messages 工具执行前的图消息
    @param calls 本批待执行 tool_calls
    @return 拦截表
    """
    index = build_history_index(messages)
    write_paths = _batch_write_paths(calls)
    blocked: dict[str, ToolMessage] = {}
    seen_batch: dict[ToolFp, ToolRecord] = {}

    for call in calls:
        shape = shape_from_call(call)
        cid = call_id(call)
        if shape is None or not cid:
            continue
        name = shape.name

        batch_hit = seen_batch.get(shape.fp)
        if batch_hit is not None:
            blocked[cid] = ToolMessage(
                content=format_duplicate_block(shape, batch_hit, same_batch=True),
                tool_call_id=cid,
                name=name,
            )
            continue

        if _read_hits_pending_write(shape, write_paths):
            seen_batch[shape.fp] = ToolRecord(
                shape=shape, call_id=cid, content="", failed=False
            )
            continue

        prior = _lookup_cover(index, shape)
        if prior is not None:
            if shape.kind == "write" and prior.failed:
                body = format_fail_block(shape, prior)
            else:
                body = format_duplicate_block(shape, prior, same_batch=False)
            blocked[cid] = ToolMessage(content=body, tool_call_id=cid, name=name)
            continue

        seen_batch[shape.fp] = ToolRecord(
            shape=shape, call_id=cid, content="", failed=False
        )
    return blocked


def install_tool_loop_guard(
    inner: Any,
    messages: list[BaseMessage],
    calls: list[Any],
    *,
    enabled: bool = True,
) -> None:
    """把本批拦截表挂到 ToolNode（跨线程可见）。"""
    if not enabled:
        inner._llgraph_loop_blocks = {}
        return
    inner._llgraph_loop_blocks = compute_blocked_tool_messages(messages, calls)


def clear_tool_loop_guard(inner: Any) -> None:
    """清除 ToolNode 上的重复工具拦截表。"""
    inner._llgraph_loop_blocks = {}


def wrap_tool_node_with_loop_guard(inner: Any) -> None:
    """
    包装 ToolNode._run_one / _arun_one：命中拦截表则不执行真实工具。

    必须在 timing / write_serialize 之前调用，使拦截仍走写串行门闩的 mark_done，
    避免同 path 后续写等待已拦截的前驱导致超时。

    @param inner ToolNode 实例
    """
    if getattr(inner, "_llgraph_loop_guard_wrapped", False):
        return

    inner._llgraph_loop_blocks = {}
    original_run = inner._run_one
    original_arun = inner._arun_one

    def _blocked_message(call: dict[str, Any]) -> ToolMessage | None:
        blocks = getattr(inner, "_llgraph_loop_blocks", None) or {}
        cid = str(call.get("id") or "").strip()
        msg = blocks.get(cid)
        return msg if isinstance(msg, ToolMessage) else None

    def guarded_run_one(call: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        blocked = _blocked_message(call)
        if blocked is not None:
            return blocked
        return original_run(call, *args, **kwargs)

    async def guarded_arun_one(call: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        blocked = _blocked_message(call)
        if blocked is not None:
            return blocked
        return await original_arun(call, *args, **kwargs)

    inner._run_one = guarded_run_one  # type: ignore[method-assign]
    inner._arun_one = guarded_arun_one  # type: ignore[method-assign]
    inner._llgraph_loop_guard_wrapped = True  # type: ignore[attr-defined]
