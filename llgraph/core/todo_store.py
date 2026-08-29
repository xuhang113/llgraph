"""会话任务清单：落盘在 checkpoint 之外，压缩也不会丢计划。

对标 Cursor Agent 任务列表 / Claude Code TodoWrite / Codex plan：
多步改码时模型把步骤写成结构化清单；每轮 <workspace-context> 钉住当前表，
结束 turn 前若仍有 pending/in_progress 则图内 nudge 续跑。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import ai_message_tool_calls
from llgraph.context.investigate_harness import is_ephemeral_harness_human
from llgraph.context.runtime_context import get_active_thread_id
from llgraph.session.user_storage import session_todos_path

TODO_TOOL_NAME = "todo_write"
TODO_NUDGE_MARKER = "[系统·未完成任务]"

TODO_STATUSES = ("pending", "in_progress", "completed", "cancelled")
_STATUS_ALIASES = {
    "in-progress": "in_progress",
    "inprogress": "in_progress",
    "doing": "in_progress",
    "done": "completed",
    "complete": "completed",
    "canceled": "cancelled",
    "cancel": "cancelled",
    "todo": "pending",
    "open": "pending",
}

MAX_TODOS = 20
MAX_CONTENT_CHARS = 240
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,23}$")

_STATUS_MARK = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "cancelled": "[-]",
}
_STATUS_LABEL = {
    "pending": "待做",
    "in_progress": "进行中",
    "completed": "已完成",
    "cancelled": "已取消",
}


@dataclass(frozen=True)
class TodoItem:
    """一条任务。"""

    id: str
    content: str
    status: str


@dataclass
class TodoState:
    """会话任务清单。"""

    todos: list[TodoItem] = field(default_factory=list)
    updated_at: str = ""

    def open_items(self) -> list[TodoItem]:
        return [t for t in self.todos if t.status in ("pending", "in_progress")]

    def counts(self) -> dict[str, int]:
        out = {key: 0 for key in TODO_STATUSES}
        for item in self.todos:
            if item.status in out:
                out[item.status] += 1
        return out


def normalize_todo_status(raw: object) -> str:
    """
    规范化 status。

    @param raw 模型传入值
    @return pending | in_progress | completed | cancelled
    """
    key = str(raw or "").strip().lower().replace(" ", "_")
    key = _STATUS_ALIASES.get(key, key)
    if key in TODO_STATUSES:
        return key
    return "pending"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_content(raw: object) -> str:
    text = " ".join(str(raw or "").split())
    if len(text) > MAX_CONTENT_CHARS:
        return text[: MAX_CONTENT_CHARS - 1] + "…"
    return text


def _clean_id(raw: object) -> str:
    value = str(raw or "").strip()
    if _ID_RE.fullmatch(value):
        return value
    return ""


def _next_id(taken: set[str]) -> str:
    n = 1
    while f"t{n}" in taken:
        n += 1
    return f"t{n}"


def empty_todo_state() -> TodoState:
    """@return 空清单"""
    return TodoState(todos=[], updated_at="")


def load_todo_state(workspace: Path, thread_id: str) -> TodoState:
    """
    读取会话 todos.json。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @return 清单；缺失或损坏时为空
    """
    path = session_todos_path(workspace, thread_id)
    if not path.is_file():
        return empty_todo_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_todo_state()
    return _state_from_dict(data if isinstance(data, dict) else {})


def save_todo_state(workspace: Path, thread_id: str, state: TodoState) -> None:
    """
    原子写入 todos.json。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param state 清单
    """
    path = session_todos_path(workspace, thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": state.updated_at or _utc_now(),
        "todos": [
            {"id": item.id, "content": item.content, "status": item.status}
            for item in state.todos
        ],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _state_from_dict(data: dict[str, Any]) -> TodoState:
    raw_items = data.get("todos")
    if not isinstance(raw_items, list):
        return empty_todo_state()
    items: list[TodoItem] = []
    seen: set[str] = set()
    for row in raw_items:
        if not isinstance(row, dict):
            continue
        item_id = _clean_id(row.get("id"))
        content = _clean_content(row.get("content"))
        if not item_id or not content or item_id in seen:
            continue
        seen.add(item_id)
        items.append(
            TodoItem(
                id=item_id,
                content=content,
                status=normalize_todo_status(row.get("status")),
            )
        )
        if len(items) >= MAX_TODOS:
            break
    return TodoState(
        todos=_enforce_single_in_progress(items),
        updated_at=str(data.get("updated_at") or ""),
    )


def _enforce_single_in_progress(items: list[TodoItem]) -> list[TodoItem]:
    """最多保留一条 in_progress（取最后一条）。"""
    last_idx = -1
    for idx, item in enumerate(items):
        if item.status == "in_progress":
            last_idx = idx
    if last_idx < 0:
        return items
    out: list[TodoItem] = []
    for idx, item in enumerate(items):
        if item.status == "in_progress" and idx != last_idx:
            out.append(TodoItem(id=item.id, content=item.content, status="pending"))
        else:
            out.append(item)
    return out


def parse_todo_inputs(raw: object) -> list[dict[str, str]]:
    """
    把工具参数规范成 [{id, content, status}, ...]。

    @param raw todos 列表
    @return 规范化条目（可能仍缺 id）
    """
    if not isinstance(raw, list):
        from llgraph.core.tool_arg_coerce import maybe_parse_json

        parsed = maybe_parse_json(raw)
        if isinstance(parsed, dict):
            raw = [parsed]
        elif isinstance(parsed, list):
            raw = parsed
        else:
            return []
    out: list[dict[str, str]] = []
    for row in raw:
        if hasattr(row, "model_dump"):
            row = row.model_dump()
        elif hasattr(row, "dict"):
            row = row.dict()
        if not isinstance(row, dict):
            continue
        content = _clean_content(row.get("content") or row.get("task") or row.get("text"))
        if not content:
            continue
        out.append(
            {
                "id": _clean_id(row.get("id")),
                "content": content,
                "status": normalize_todo_status(row.get("status")),
            }
        )
        if len(out) >= MAX_TODOS:
            break
    return out


def apply_todo_write(
    current: TodoState,
    incoming: list[dict[str, str]],
    *,
    merge: bool,
) -> tuple[TodoState, list[str]]:
    """
    合并或整表替换任务清单。

    @param current 当前落盘状态
    @param incoming 本轮提交的条目
    @param merge True 按 id/正文合并；False 整表替换
    @return (新状态, 备注)
    """
    notes: list[str] = []
    if merge:
        by_id = {item.id: item for item in current.todos}
        by_content = {item.content.lower(): item.id for item in current.todos}
        taken = set(by_id)
        for row in incoming:
            item_id = row["id"]
            content = row["content"]
            status = row["status"]
            if not item_id:
                item_id = by_content.get(content.lower(), "")
            if not item_id:
                item_id = _next_id(taken)
            taken.add(item_id)
            by_id[item_id] = TodoItem(id=item_id, content=content, status=status)
            by_content[content.lower()] = item_id
        items = list(by_id.values())
        if len(items) > MAX_TODOS:
            items = items[:MAX_TODOS]
            notes.append(f"清单超过 {MAX_TODOS} 条，已截断。")
    else:
        taken: set[str] = set()
        items = []
        for row in incoming:
            item_id = row["id"] if row["id"] and row["id"] not in taken else _next_id(taken)
            taken.add(item_id)
            items.append(TodoItem(id=item_id, content=row["content"], status=row["status"]))
        if not incoming:
            notes.append("已清空任务清单。")

    enforced = _enforce_single_in_progress(items)
    if sum(1 for item in items if item.status == "in_progress") > 1:
        notes.append("同时只能有 1 条 in_progress，其余已改回 pending。")
    in_progress = [item for item in enforced if item.status == "in_progress"]
    open_pending = [item for item in enforced if item.status == "pending"]
    if not in_progress and open_pending:
        notes.append("仍有 pending 但没有 in_progress：下一动作前把当前项标为 in_progress。")
    return TodoState(todos=enforced, updated_at=_utc_now()), notes


def format_todo_markdown(state: TodoState, *, heading: bool = True) -> str:
    """
    渲染给人/模型看的清单。

    @param state 清单
    @param heading 是否带标题
    @return Markdown；空清单时说明尚无任务
    """
    counts = state.counts()
    done = counts["completed"]
    total = len(state.todos)
    open_n = counts["pending"] + counts["in_progress"]
    lines: list[str] = []
    if heading:
        if total:
            lines.append(f"## 本轮任务清单（todo_write）· {done}/{total} 完成")
        else:
            lines.append("## 本轮任务清单（todo_write）")
            lines.append("尚无条目。两步以上的改码/排查应先 `todo_write` 列出步骤。")
            return "\n".join(lines)
    elif not total:
        return "（空清单）"
    for item in state.todos:
        mark = _STATUS_MARK.get(item.status, "[ ]")
        label = _STATUS_LABEL.get(item.status, item.status)
        lines.append(f"- {mark} `{item.id}` {label}：{item.content}")
    if heading and open_n:
        lines.append(
            f"未完成 {open_n} 项。结束 turn 前须全部 completed/cancelled，"
            "或 `todo_write(merge=true)` 更新。"
        )
    return "\n".join(lines)


def format_todo_tool_result(state: TodoState, notes: list[str]) -> str:
    """
    工具返回文案（短，适合留在 tool 链里）。

    @param state 更新后清单
    @param notes 合并备注
    @return 纯文本
    """
    counts = state.counts()
    summary = (
        f"任务清单 {counts['completed']}/{len(state.todos)} 完成"
        f" · {counts['in_progress']} 进行中 · {counts['pending']} 待做"
        f" · {counts['cancelled']} 取消"
    )
    lines = [summary, format_todo_markdown(state, heading=False)]
    for note in notes:
        lines.append(f"注：{note}")
    return "\n".join(lines).strip()


def format_todo_workspace_context(workspace: Path, thread_id: str | None) -> str:
    """
    注入 <workspace-context> 的当前清单。空清单不占 token。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @return Markdown 或空串
    """
    if not thread_id:
        return ""
    state = load_todo_state(workspace, thread_id)
    if not state.todos:
        return ""
    return format_todo_markdown(state, heading=True)


def format_edited_paths_workspace_context(edited_paths: list[str] | None) -> str:
    """
    每轮钉住本会话已改文件（不只续写口吻），减少丢工作集后再广搜。

    @param edited_paths 已改相对路径
    @return Markdown 或空串
    """
    paths = [p.strip() for p in (edited_paths or []) if str(p).strip()]
    if not paths:
        return ""
    lines = ["## 本会话已改文件", "继续改这些文件时用写入后快照或再 read，勿用写入前的全文。"]
    for path in paths[:12]:
        lines.append(f"- `{path}`")
    return "\n".join(lines)


def todo_nudge_pending(messages: list[BaseMessage]) -> bool:
    """
    本问是否已经注入过未完成任务 nudge。

    @param messages 图消息
    @return 已注入则为 True
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not is_ephemeral_harness_human(msg):
            return False
        if isinstance(msg, HumanMessage):
            text = str(getattr(msg, "content", "") or "").strip()
            if text.startswith(TODO_NUDGE_MARKER):
                return True
    return False


def _todo_write_since_last_user(messages: list[BaseMessage]) -> bool:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not is_ephemeral_harness_human(msg):
            return False
        if isinstance(msg, ToolMessage) and str(getattr(msg, "name", "") or "") == TODO_TOOL_NAME:
            return True
        if isinstance(msg, AIMessage):
            for call in ai_message_tool_calls(msg):
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if str(name or "") == TODO_TOOL_NAME:
                    return True
    return False


def should_nudge_open_todos(
    messages: list[BaseMessage],
    workspace: Path | None,
    *,
    remaining_steps: int | None = None,
) -> bool:
    """
    模型已输出可见正文、本问建过清单、仍有未完成项时，禁止提前结束。

    @param messages 图消息
    @param workspace 工作区根
    @param remaining_steps 图剩余步；过少时不强行续跑
    @return 是否应注入 todo_nudge
    """
    if workspace is None:
        return False
    if remaining_steps is not None and remaining_steps <= 3:
        return False
    thread_id = get_active_thread_id()
    if not thread_id:
        return False
    state = load_todo_state(workspace, thread_id)
    if not state.open_items():
        return False
    if todo_nudge_pending(messages):
        return False
    return _todo_write_since_last_user(messages)


def format_todo_nudge(workspace: Path | None) -> str:
    """
    图内续跑 Human 文案（ephemeral，不落盘 jsonl）。

    @param workspace 工作区根
    @return nudge 正文
    """
    thread_id = get_active_thread_id()
    listing = "（无法读取清单）"
    if workspace is not None and thread_id:
        listing = format_todo_markdown(load_todo_state(workspace, thread_id), heading=False)
    return (
        f"{TODO_NUDGE_MARKER} 任务清单仍有未完成项。禁止现在结束 turn。\n"
        f"{listing}\n"
        "请继续执行当前 in_progress，或 `todo_write(merge=true)` "
        "将剩余项标为 completed/cancelled 并说明原因后再作答。"
    )
