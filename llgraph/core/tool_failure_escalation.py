"""同一文件改不动时的升级策略：诊断 → 强制重读 → 停手。

`tool_loop_guard` 只拦**参数逐字节相同**的重试。真实死循环长这样：
`search_replace` 每次换一点 `old_string`（缩进、变量名、上下文各差一点），
指纹次次不同，护栏次次放行，模型可以一路试到 `max_turns`——每次都是一次完整
LLM 往返 + 一份错误回灌，最后以「Sorry, need more steps」收场。

本模块按**路径**而不是按参数聚合失败，给出商用 Agent 的三级升级：

- 连失 N 次：把「你已经失败几次、都试过什么、这类错误该怎么破」贴回工具结果
- 连失 N+2 次：锁住该路径的写工具，必须先 `read_file` 拿到当前原文才解锁
- 连失 N+4 次：彻底停手，让模型向用户说明卡在哪，而不是继续烧轮次
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import ai_message_tool_calls
from llgraph.context.investigate_harness import is_ephemeral_harness_human
from llgraph.core.tool_loop_guard import (
    IDENTICAL_BLOCK_MARKER,
    IDENTICAL_FAIL_MARKER,
    call_args,
    call_id,
    call_name,
    tool_result_failed,
)
from llgraph.core.write_failure_tracker import WRITE_TOOL_NAMES
from llgraph.core.write_serialize import normalize_write_path

ESCALATION_MARKER = "[llgraph] 同一文件连续改不动"
RELOCK_MARKER = "[llgraph] 该文件写入已锁"
STOP_MARKER = "[llgraph] 该文件写入已停"

_READ_TOOL_NAMES = frozenset({"read_file", "read_files"})
# 未真正执行工具的占位正文，一律以标记开头。升级提示是**追加**在真失败末尾的，
# 不在此列——把它当占位会让失败计数停在阈值上，锁定与停手永远不触发。
_INTERCEPT_MARKERS = (
    RELOCK_MARKER,
    STOP_MARKER,
    IDENTICAL_BLOCK_MARKER,
    IDENTICAL_FAIL_MARKER,
)

_SIGNATURE_CHARS = 64
_MAX_LISTED_ATTEMPTS = 5

REASON_NOT_FOUND = "not_found"
REASON_NOT_UNIQUE = "not_unique"
REASON_MISSING_FILE = "missing_file"
REASON_BAD_ARGS = "bad_args"
REASON_OTHER = "other"

_REASON_LABELS = {
    REASON_NOT_FOUND: "old_string 没匹配上",
    REASON_NOT_UNIQUE: "old_string 不唯一",
    REASON_MISSING_FILE: "路径不存在",
    REASON_BAD_ARGS: "参数不合法",
    REASON_OTHER: "写入失败",
}

_PRESCRIPTIONS = {
    REASON_NOT_FOUND: (
        "连续没匹配上，几乎一定是你手里的原文不对——凭印象重打、或用了被改前的旧快照。"
        "**不要再换着法子猜 old_string**：先 `read_file(path, start_line, end_line)` 把目标那几十行读出来，"
        "从返回正文里**逐字符复制**一段作 old_string（含原缩进），再提交一次。"
    ),
    REASON_NOT_UNIQUE: (
        "片段在文件里出现多次。把 old_string 往上下各扩几行到整段唯一；"
        "同一文件多处要改就用 `replacements` 一次提交；确实要全改才用 `replace_all=true`。"
    ),
    REASON_MISSING_FILE: (
        "路径不对，别再拿相近的名字试。先 `glob_files` / `list_directory` 确认真实路径；"
        "确实要新建文件就用 `write_file` 并给出完整 content。"
    ),
    REASON_BAD_ARGS: (
        "参数没给全。`write_file` / `append_file` 必须**同时**给 `path` 与 `content`；"
        "`search_replace` 必须给 `old_string`（或 `replacements`）。"
    ),
    REASON_OTHER: (
        "同一处反复失败。先 `read_file` 核对当前原文与权限，再决定是换 `write_file` 整文件重写，"
        "还是这条路本来就走不通。"
    ),
}


@dataclass(frozen=True)
class FailedAttempt:
    """一次失败的写工具调用。"""

    tool: str
    reason: str
    signature: str


@dataclass
class PathFailureState:
    """某个路径自最近 user 以来的连续写失败状态。"""

    path: str
    attempts: list[FailedAttempt] = field(default_factory=list)
    read_after_last_failure: bool = False

    @property
    def count(self) -> int:
        """连续失败次数。"""
        return len(self.attempts)

    @property
    def dominant_reason(self) -> str:
        """出现最多的失败原因（并列取最近一次）。"""
        if not self.attempts:
            return REASON_OTHER
        tally: dict[str, int] = {}
        for item in self.attempts:
            tally[item.reason] = tally.get(item.reason, 0) + 1
        best = max(tally.values())
        for item in reversed(self.attempts):
            if tally[item.reason] == best:
                return item.reason
        return REASON_OTHER


def classify_failure(text: str) -> str:
    """
    从工具返回正文判定失败类别。

    @param text 工具返回
    @return REASON_* 之一
    """
    body = str(text or "")
    lowered = body.lower()
    if "不唯一" in body:
        return REASON_NOT_UNIQUE
    if "未找到 old_string" in body:
        return REASON_NOT_FOUND
    if "文件不存在" in body or "路径不存在" in body:
        return REASON_MISSING_FILE
    if (
        "缺少必填" in body
        or "缺少有效 path" in body
        or "必须提供 old_string" in body
        or "缺少 old_string" in body
        or "validation error" in lowered
        or "field required" in lowered
    ):
        return REASON_BAD_ARGS
    return REASON_OTHER


def _is_intercept_notice(text: str) -> bool:
    return str(text or "").lstrip().startswith(_INTERCEPT_MARKERS)


def _write_succeeded(text: str) -> bool:
    return str(text or "").startswith(("已写入", "已追加", "已替换"))


def _attempt_signature(name: str, args: dict[str, Any]) -> str:
    if name == "search_replace":
        old = args.get("old_string")
        if not isinstance(old, str) or not old:
            replacements = args.get("replacements")
            if isinstance(replacements, list) and replacements:
                first = replacements[0]
                if isinstance(first, dict):
                    old = first.get("old_string") or first.get("oldString")
        if isinstance(old, str) and old.strip():
            head = old.strip().splitlines()[0].strip()
            if len(head) > _SIGNATURE_CHARS:
                head = head[: _SIGNATURE_CHARS - 1] + "…"
            return f'old_string 首行 "{head}"'
        return "未给出 old_string"
    content = args.get("content")
    if isinstance(content, str):
        return f"content {len(content)} 字符"
    return "未给出 content"


def _read_paths(name: str, args: dict[str, Any]) -> set[str]:
    if name == "read_file":
        path = normalize_write_path(args.get("path"))
        return {path} if path else set()
    from llgraph.core.tool_arg_coerce import coerce_path_list

    out = {normalize_write_path(item) for item in coerce_path_list(args.get("paths"))}
    return {path for path in out if path}


def _history_start(messages: list[BaseMessage]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, HumanMessage) and not is_ephemeral_harness_human(msg):
            return idx + 1
    return 0


def scan_path_failures(messages: list[BaseMessage]) -> dict[str, PathFailureState]:
    """
    扫描自最近真实 user 以来，每个路径上的**连续**写失败。

    写成功清零该路径；成功 read 只解锁、不清零（读完还改不动要继续升级）。
    本模块自己贴出的提示与 loop guard 的拦截文案都不计入失败。

    @param messages 工具执行前的图消息
    @return 路径 → 失败状态（无失败的路径不出现）
    """
    states: dict[str, PathFailureState] = {}
    pending: dict[str, Any] = {}
    for msg in messages[_history_start(messages) :]:
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
        name = call_name(call)
        text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        args = call_args(call)

        if name in _READ_TOOL_NAMES:
            if _is_intercept_notice(text) or tool_result_failed(name, text):
                continue
            for path in _read_paths(name, args):
                state = states.get(path)
                if state is not None:
                    state.read_after_last_failure = True
            continue

        if name not in WRITE_TOOL_NAMES:
            continue
        path = normalize_write_path(args.get("path"))
        if not path:
            continue
        if _write_succeeded(text):
            states.pop(path, None)
            continue
        if _is_intercept_notice(text) or not tool_result_failed(name, text):
            continue
        state = states.setdefault(path, PathFailureState(path=path))
        state.attempts.append(
            FailedAttempt(
                tool=name,
                reason=classify_failure(text),
                signature=_attempt_signature(name, args),
            )
        )
        state.read_after_last_failure = False
    return states


def format_attempt_history(state: PathFailureState) -> str:
    """
    列出已经试过什么，让模型看见自己在打转。

    @param state 失败状态
    @return 多行摘要
    """
    listed = state.attempts[-_MAX_LISTED_ATTEMPTS:]
    skipped = state.count - len(listed)
    lines: list[str] = []
    if skipped > 0:
        lines.append(f"  （更早 {skipped} 次省略）")
    for offset, item in enumerate(listed, start=skipped + 1):
        label = _REASON_LABELS.get(item.reason, _REASON_LABELS[REASON_OTHER])
        lines.append(f"  {offset}. {item.tool} · {label} · {item.signature}")
    return "\n".join(lines)


def format_escalation_hint(state: PathFailureState, *, lock_at: int, stop_at: int) -> str:
    """
    连失达到阈值后追加在工具结果末尾的升级提示。

    @param state 失败状态
    @param lock_at 锁定阈值
    @param stop_at 停手阈值
    @return 提示块
    """
    lines = [
        "",
        f"{ESCALATION_MARKER}：`{state.path}` 本问已连续失败 {state.count} 次。",
        format_attempt_history(state),
        _PRESCRIPTIONS.get(state.dominant_reason, _PRESCRIPTIONS[REASON_OTHER]),
    ]
    left_to_lock = lock_at - state.count
    if left_to_lock > 0:
        lines.append(
            f"再失败 {left_to_lock} 次，该文件的写工具会被锁住，必须先 read_file 才能继续。"
        )
    else:
        lines.append(
            f"再失败 {max(1, stop_at - state.count)} 次，本问将不再允许写这个文件，"
            "届时请直接向用户说明卡点。"
        )
    return "\n".join(lines)


def format_lock_block(state: PathFailureState) -> str:
    """路径被锁时的拦截文案（read 之后自动解锁）。"""
    return "\n".join(
        [
            f"{RELOCK_MARKER}：`{state.path}` 已连续失败 {state.count} 次，写工具暂时不执行。",
            format_attempt_history(state),
            "解锁方式只有一个：先 `read_file` 读这个文件的目标区段，"
            "拿到当前真实原文之后再提交写入。",
            _PRESCRIPTIONS.get(state.dominant_reason, _PRESCRIPTIONS[REASON_OTHER]),
        ]
    )


def format_stop_block(state: PathFailureState) -> str:
    """路径彻底停写时的拦截文案。"""
    return "\n".join(
        [
            f"{STOP_MARKER}：`{state.path}` 本问已失败 {state.count} 次，"
            "读过之后仍然改不动，继续试只会空耗。",
            format_attempt_history(state),
            "本问不会再执行这个文件的写工具。请停止改这个文件："
            "要么改用别的落地方式（例如改动别处、或整文件 `write_file` 重写），"
            "要么直接向用户说明卡在哪一步、你需要什么信息。",
        ]
    )


def compute_escalation_blocks(
    messages: list[BaseMessage],
    calls: list[Any],
    *,
    hint_after: int,
    lock_offset: int,
    stop_offset: int,
) -> dict[str, ToolMessage]:
    """
    计算本批因连续失败而应拦截的写调用。

    @param messages 工具执行前的图消息
    @param calls 本批 tool_calls
    @param hint_after 提示阈值（<=0 关闭整个机制）
    @param lock_offset 锁定阈值相对提示阈值的偏移
    @param stop_offset 停手阈值相对提示阈值的偏移
    @return tool_call_id → 占位 ToolMessage
    """
    if hint_after <= 0:
        return {}
    states = scan_path_failures(messages)
    if not states:
        return {}
    lock_at = hint_after + lock_offset
    stop_at = hint_after + stop_offset
    blocked: dict[str, ToolMessage] = {}
    for call in calls:
        name = call_name(call)
        if name not in WRITE_TOOL_NAMES:
            continue
        cid = call_id(call)
        if not cid:
            continue
        path = normalize_write_path(call_args(call).get("path"))
        state = states.get(path)
        if state is None:
            continue
        if state.count >= stop_at:
            body = format_stop_block(state)
        elif state.count >= lock_at and not state.read_after_last_failure:
            body = format_lock_block(state)
        else:
            continue
        blocked[cid] = ToolMessage(content=body, tool_call_id=cid, name=name)
    return blocked


def annotate_escalation_hints(
    out: dict[str, Any],
    *,
    prior_messages: list[BaseMessage],
    hint_after: int,
    lock_offset: int,
    stop_offset: int,
) -> dict[str, Any]:
    """
    在本批新产生的写失败结果末尾追加升级提示。

    @param out ToolNode 输出
    @param prior_messages 工具执行前的图消息
    @param hint_after 提示阈值（<=0 关闭）
    @param lock_offset 锁定阈值偏移
    @param stop_offset 停手阈值偏移
    @return 可能改写后的 out
    """
    if hint_after <= 0:
        return out
    new_msgs = list(out.get("messages") or [])
    if not new_msgs:
        return out
    combined = list(prior_messages) + new_msgs
    states = scan_path_failures(combined)
    if not states:
        return out

    pending: dict[str, str] = {}
    for msg in reversed(prior_messages):
        if isinstance(msg, AIMessage):
            for call in ai_message_tool_calls(msg):
                cid = call_id(call)
                if not cid or call_name(call) not in WRITE_TOOL_NAMES:
                    continue
                pending[cid] = normalize_write_path(call_args(call).get("path"))
            break

    lock_at = hint_after + lock_offset
    stop_at = hint_after + stop_offset
    changed = False
    annotated: set[str] = set()
    for idx, msg in enumerate(new_msgs):
        if not isinstance(msg, ToolMessage):
            continue
        cid = str(getattr(msg, "tool_call_id", "") or "").strip()
        path = pending.get(cid)
        if not path or path in annotated:
            continue
        body = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if _is_intercept_notice(body) or ESCALATION_MARKER in body:
            continue
        state = states.get(path)
        if state is None or state.count < hint_after:
            continue
        annotated.add(path)
        new_msgs[idx] = ToolMessage(
            content=body.rstrip()
            + "\n"
            + format_escalation_hint(state, lock_at=lock_at, stop_at=stop_at),
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", None),
        )
        changed = True
    if not changed:
        return out
    return {**out, "messages": new_msgs}


def install_edit_failure_blocks(
    inner: Any,
    messages: list[BaseMessage],
    calls: list[Any],
    *,
    hint_after: int,
    lock_offset: int,
    stop_offset: int,
) -> None:
    """
    把连续失败拦截并入 ToolNode 已有的短路径表（复用 loop guard 的包装）。

    必须在 `install_tool_loop_guard` 之后调用：后者会重置整张表。

    @param inner ToolNode 实例
    @param messages 工具执行前的图消息
    @param calls 本批 tool_calls
    @param hint_after 提示阈值（<=0 关闭）
    @param lock_offset 锁定阈值偏移
    @param stop_offset 停手阈值偏移
    """
    blocks = compute_escalation_blocks(
        messages,
        calls,
        hint_after=hint_after,
        lock_offset=lock_offset,
        stop_offset=stop_offset,
    )
    if not blocks:
        return
    existing = getattr(inner, "_llgraph_loop_blocks", None)
    if not isinstance(existing, dict):
        existing = {}
        inner._llgraph_loop_blocks = existing
    for cid, msg in blocks.items():
        existing.setdefault(cid, msg)
