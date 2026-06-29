"""会话元数据：标题（类似 Cursor 侧边栏）、更新时间等。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from llgraph.session.jsonl_read import open_jsonl_for_read
from llgraph.session.user_storage import session_messages_path, session_thread_dir, user_sessions_root

_TITLE_DISPLAY_MAX_LEN = 24
_TITLE_FULL_MAX_LEN = 30
_TITLE_SOURCE_MANUAL = "manual"
_TITLE_SOURCE_AUTO = "auto"
_TITLE_SOURCE_FALLBACK = "fallback"

# 不作为会话标题的交互/元命令（避免 exit; 等脏标题）
_SKIP_AUTO_TITLE_MESSAGES = frozenset({
    "exit",
    "quit",
    "q",
    "help",
    "?",
    "/help",
    "/exit",
    "/quit",
    "/sessions",
    "/session",
})

_INJECTED_USER_BLOCK_PATTERNS: tuple[tuple[str, int], ...] = (
    (r"<workspace-context>[\s\S]*?</workspace-context>\s*", re.IGNORECASE),
    (r"<session-manifest>[\s\S]*?</session-manifest>\s*", re.IGNORECASE),
    (r"<custom-command[\s\S]*?</custom-command>\s*", re.IGNORECASE),
)


def session_meta_json_path(workspace: Path, thread_id: str) -> Path:
    """
    会话 meta.json 路径（与 messages.jsonl 同目录）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return meta.json 绝对路径
    """
    return session_thread_dir(workspace, thread_id) / "meta.json"


def load_session_meta(workspace: Path, thread_id: str) -> dict[str, Any]:
    """
    读取会话 meta.json。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 元数据 dict；无文件返回空 dict
    """
    path = session_meta_json_path(workspace, thread_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_session_meta(
    workspace: Path,
    thread_id: str,
    patch: dict[str, Any],
) -> None:
    """
    合并写入 meta.json（保留已有 title 等字段）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param patch 待合并字段
    """
    path = session_meta_json_path(workspace, thread_id)
    merged = load_session_meta(workspace, thread_id)
    merged.update(patch)
    merged["thread_id"] = thread_id
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def touch_session_activity(workspace: Path, thread_id: str, **extra: Any) -> None:
    """
    刷新 meta.updated_at，供 Web 侧栏按最近活动排序（发消息、Agent 轮次结束等）。

    @param workspace 工作区根
    @param thread_id 会话 thread
    @param extra 可选合并字段
    """
    if not thread_id.strip():
        return
    save_session_meta(workspace, thread_id, dict(extra))


def strip_injected_context_from_user_message(text: str) -> str:
    """
    去掉 user 消息中的 workspace-context / manifest 等注入块（标题生成用）。

    @param text 原始 user 消息
    @return 用户可见正文
    """
    out = str(text or "").strip()
    if not out:
        return ""
    for pattern, flags in _INJECTED_USER_BLOCK_PATTERNS:
        out = re.sub(pattern, "", out, flags=flags)
    return out.strip()


def _looks_like_injected_title_fragment(text: str) -> bool:
    """标题是否像 XML 注入块残片（如 `<workspace-c`）。"""
    t = str(text or "").strip().lower()
    return (
        t.startswith("<workspace")
        or t.startswith("<session-manifest")
        or t.startswith("<custom-command")
        or t.startswith("<")
    )


def normalize_session_title(
    text: str,
    *,
    max_len: int = _TITLE_DISPLAY_MAX_LEN,
    use_ellipsis: bool = True,
) -> str:
    """
    规范化标题：去空白、压平换行、截断至 max_len。

    @param text 原始文本
    @param max_len 最大长度
    @param use_ellipsis 超长时是否追加 …
    @return 可用标题；过短返回空串
    """
    if not text or not str(text).strip():
        return ""
    line = str(text).strip().splitlines()[0].strip()
    line = re.sub(r"\s+", " ", line)
    line = line.strip("#").strip()
    if len(line) > max_len:
        if use_ellipsis and max_len > 1:
            line = line[: max_len - 1].rstrip() + "…"
        else:
            line = line[:max_len].rstrip()
    return line if len(line) >= 2 else ""


def format_display_session_title(full_title: str) -> str:
    """
    侧栏/列表用的短标题（≤24 字）。

    @param full_title 完整标题
    @return 展示标题
    """
    return normalize_session_title(
        full_title,
        max_len=_TITLE_DISPLAY_MAX_LEN,
        use_ellipsis=True,
    )


def extract_title_candidate(text: str) -> str:
    """
    从首条用户消息提取标题候选（首句/去代码块/简化 curl 等）。

    @param text 用户消息
    @return 候选文本；无法提取返回空串
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    cleaned = strip_injected_context_from_user_message(raw)
    if not cleaned:
        return ""
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`[^`\n]+`", " ", cleaned)
    line = cleaned.splitlines()[0].strip()
    line = re.sub(r"\s+", " ", line).strip("#").strip()
    if not line or _looks_like_injected_title_fragment(line):
        return ""

    curl_match = re.match(
        r"curl\b.+?(https?://[^\s'\"]+)",
        line,
        flags=re.IGNORECASE,
    )
    if curl_match:
        from urllib.parse import urlparse

        parsed = urlparse(curl_match.group(1))
        path = (parsed.path or "").strip("/")
        if path:
            line = path.split("/")[-1] or path
        elif parsed.netloc:
            line = parsed.netloc

    if line.startswith("/") and " " not in line[:96]:
        segments = [s for s in line.strip("/").split("/") if s]
        if segments:
            line = segments[-1]

    for sep in ("。", "！", "？", ".", "!", "?"):
        if sep in line:
            line = line.split(sep, 1)[0].strip()
            break

    return line


def disambiguate_session_titles(entries: list[tuple[str, str]]) -> list[str]:
    """
    列表展示时，为重复标题追加 thread 短后缀以便区分。

    @param entries (thread_id, title) 列表
    @return 展示用标题（与 entries 同序）
    """
    from collections import Counter

    titles = [title for _, title in entries]
    counts = Counter(titles)
    out: list[str] = []
    for thread_id, title in entries:
        if counts[title] <= 1:
            out.append(title)
            continue
        suffix = short_thread_suffix(thread_id)
        tag = f" · {suffix}"
        room = _TITLE_DISPLAY_MAX_LEN - len(tag)
        if room < 4:
            out.append(f"{title[:8]}…{tag}")
            continue
        base = title
        if len(base) > room:
            base = base[: room - 1].rstrip() + "…"
        out.append(base + tag)
    return out


def should_use_message_for_auto_title(text: str) -> bool:
    """
    判断是否可用用户输入作为自动标题。

    @param text 用户消息
    @return 是否采用
    """
    stripped = strip_injected_context_from_user_message(text)
    if not stripped:
        return False
    from llgraph.terminal.keys import is_exit_command

    if is_exit_command(stripped):
        return False
    lower = stripped.lower().rstrip(";")
    if lower in _SKIP_AUTO_TITLE_MESSAGES:
        return False
    if lower.startswith("/") and len(stripped) < 48:
        return False
    if _looks_like_injected_title_fragment(stripped.splitlines()[0]):
        return False
    return len(stripped) >= 2


def suggest_full_title_from_text(text: str) -> str:
    """
    从用户首条消息生成完整标题（编辑/悬停用，≤30 字）。

    @param text 用户消息
    @return 建议标题
    """
    if not should_use_message_for_auto_title(text):
        return ""
    visible = strip_injected_context_from_user_message(text)
    candidate = extract_title_candidate(visible or text)
    if candidate and _looks_like_injected_title_fragment(candidate):
        candidate = ""
    return normalize_session_title(
        candidate or visible or text,
        max_len=_TITLE_FULL_MAX_LEN,
        use_ellipsis=False,
    )


def suggest_title_from_text(text: str) -> str:
    """
    从用户首条消息生成短标题（对齐 Cursor 首行摘要习惯）。

    @param text 用户消息
    @return 建议标题
    """
    full = suggest_full_title_from_text(text)
    return format_display_session_title(full) if full else ""


def short_thread_suffix(thread_id: str) -> str:
    """
    从 thread_id 提取短后缀（cli-xxxxxxxx → xxxxxxxx）。

    @param thread_id 线程 ID
    @return 短后缀
    """
    tid = thread_id.strip()
    if tid.startswith("cli-") and len(tid) > 4:
        return tid[4:]
    return tid[-8:] if len(tid) > 8 else tid


def default_session_title(thread_id: str) -> str:
    """
    无对话内容时的默认标题。

    @param thread_id 线程 ID
    @return 默认标题（≤24 字）
    """
    suffix = short_thread_suffix(thread_id)
    return normalize_session_title(f"会话 {suffix}") or f"会话 {suffix[:8]}"


def get_session_title(workspace: Path, thread_id: str) -> str | None:
    """
    读取会话标题。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 标题；未设置返回 None
    """
    meta = load_session_meta(workspace, thread_id)
    title = meta.get("title")
    if title and str(title).strip():
        return str(title).strip()
    return None


def set_session_title(
    workspace: Path,
    thread_id: str,
    title: str,
    *,
    source: Literal["manual", "auto", "fallback"] = "manual",
) -> tuple[bool, str]:
    """
    设置会话标题。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param title 标题文本
    @param source manual 不会被自动覆盖；auto/fallback 可被首条用户消息覆盖
    @return (是否成功, 提示)
    """
    normalized = normalize_session_title(
        title,
        max_len=_TITLE_FULL_MAX_LEN,
        use_ellipsis=False,
    )
    if not normalized:
        return False, "标题至少需要 2 个有效字符。"
    save_session_meta(
        workspace,
        thread_id,
        {
            "title": normalized,
            "title_source": source,
            "workspace": str(workspace.expanduser().resolve()),
        },
    )
    return True, f"已设置会话标题: {normalized}"


def _extract_human_content_from_jsonl_row(row: dict[str, Any]) -> str:
    data = row.get("data") if isinstance(row.get("data"), dict) else row
    role = (data or {}).get("type") or row.get("type") or row.get("role") or ""
    if "human" not in str(role).lower() and row.get("role") != "user":
        return ""
    content = (data or {}).get("content") or row.get("content") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        content = "".join(parts)
    return str(content)


def _peek_first_human_title_from_jsonl_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with open_jsonl_for_read(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                content = _extract_human_content_from_jsonl_row(row)
                if not content.strip():
                    continue
                title = suggest_full_title_from_text(content)
                if title:
                    return title
                continue
    except (OSError, json.JSONDecodeError):
        return None
    return None


def peek_title_from_messages_jsonl(workspace: Path, thread_id: str) -> str | None:
    """
    从 messages.jsonl 首条 user 消息推导标题。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 推导标题或 None
    """
    return _peek_first_human_title_from_jsonl_file(
        session_messages_path(workspace, thread_id)
    )


def _peek_title_from_archive_jsonl(workspace: Path, thread_id: str) -> str | None:
    root = user_sessions_root(workspace.expanduser().resolve())
    candidates = [
        root / f"{thread_id}.jsonl",
        root / thread_id / "archive.jsonl",
    ]
    for path in candidates:
        title = _peek_first_human_title_from_jsonl_file(path)
        if title:
            return title
    return None


def _peek_title_from_anchor(workspace: Path, thread_id: str) -> str | None:
    from llgraph.session.session_manifest import conversation_anchor_json_path

    path = conversation_anchor_json_path(workspace, thread_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sections = data.get("sections") if isinstance(data, dict) else None
    if not isinstance(sections, dict):
        goal = data.get("session_goal") if isinstance(data, dict) else None
        if goal:
            return suggest_full_title_from_text(str(goal))
        return None
    goal = sections.get("session_goal") or ""
    if goal:
        first = str(goal).strip().splitlines()[0]
        return suggest_full_title_from_text(first)
    return None


def _peek_title_from_edits(workspace: Path, thread_id: str) -> str | None:
    from llgraph.session.user_storage import legacy_workspace_session_dir, session_edits_path

    edits_path = session_edits_path(workspace, thread_id)
    if not edits_path.is_file():
        edits_path = legacy_workspace_session_dir(workspace, thread_id) / "edits.jsonl"
    if not edits_path.is_file():
        return None
    try:
        with edits_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rel = str(data.get("rel_path") or "").strip()
                if not rel:
                    continue
                name = Path(rel).name
                return normalize_session_title(f"编辑 {name}")
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _peek_title_from_started_at(workspace: Path, thread_id: str) -> str | None:
    meta_path = (
        workspace.expanduser().resolve()
        / ".llgraph"
        / "sessions"
        / thread_id
        / "meta.json"
    )
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        started = str(data.get("started_at") or "").strip()
        if not started:
            return None
        # 2026-05-29T11:24:54 → 05-29 11:24
        if "T" in started:
            date_part, time_part = started.split("T", 1)
            md = date_part[5:10] if len(date_part) >= 10 else date_part
            hm = time_part[:5] if len(time_part) >= 5 else ""
            label = f"{md} {hm}".strip()
            return normalize_session_title(f"会话 {label}")
    except (OSError, json.JSONDecodeError):
        return None
    return None


def infer_session_title(workspace: Path, thread_id: str) -> str | None:
    """
    从对话/归档/锚点/编辑记录推断标题（不落盘）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 推断标题；无法推断返回 None
    """
    for fn in (
        peek_title_from_messages_jsonl,
        _peek_title_from_archive_jsonl,
        _peek_title_from_anchor,
        _peek_title_from_edits,
        _peek_title_from_started_at,
    ):
        title = fn(workspace, thread_id)
        if title:
            return title
    return None


def backfill_session_title_if_missing(workspace: Path, thread_id: str) -> str | None:
    """
    列表/恢复时补写自动标题：有内容则推断，否则用「会话 xxxxxxxx」。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 写入的标题；已有手动标题则返回 None
    """
    meta = load_session_meta(workspace, thread_id)
    if meta.get("title") and str(meta.get("title")).strip():
        return str(meta["title"]).strip()
    if meta.get("title_source") == _TITLE_SOURCE_MANUAL:
        return None
    inferred = infer_session_title(workspace, thread_id)
    if not inferred:
        return None
    set_session_title(workspace, thread_id, inferred, source="fallback")
    return inferred


def backfill_session_titles(workspace: Path, thread_ids: list[str]) -> int:
    """
    批量补全会话标题。

    @param workspace 工作区根
    @param thread_ids 待处理 ID 列表
    @return 新写入标题的数量
    """
    count = 0
    for tid in thread_ids:
        before = get_session_title(workspace, tid)
        backfill_session_title_if_missing(workspace, tid)
        after = get_session_title(workspace, tid)
        if after and after != before:
            count += 1
    return count


def _sync_plan_json_title_after_auto(
    workspace: Path,
    thread_id: str,
    title: str,
) -> None:
    """
    Plan 会话自动标题写入后，同步 plan.json（替换 Plan {plan_id} 占位）。

    @param workspace 工作区根
    @param thread_id plan thread
    @param title 自动标题
    """
    meta = load_session_meta(workspace, thread_id)
    is_plan = meta.get("session_kind") == "plan" or thread_id.startswith("plan-")
    if not is_plan:
        return
    plan_id = str(meta.get("plan_id") or "").strip()
    if not plan_id:
        return
    from llgraph.plan.config import resolve_plan_settings
    from llgraph.plan.plan_store import is_placeholder_plan_title, load_plan, save_plan

    settings = resolve_plan_settings(workspace)
    plan = load_plan(workspace, plan_id, plans_dir=settings.plans_dir)
    if not plan:
        return
    current = str(plan.get("title") or "")
    if not is_placeholder_plan_title(current, plan_id):
        return
    plan["title"] = title
    save_plan(workspace, plan, plans_dir=settings.plans_dir)


def sync_plan_json_title_from_session_meta(workspace: Path, thread_id: str) -> None:
    """
    将 meta 中已有自动标题同步到 plan.json（plan 文件晚于 meta 创建时补写）。

    @param workspace 工作区根
    @param thread_id plan thread
    """
    title = get_session_title(workspace, thread_id)
    if title:
        _sync_plan_json_title_after_auto(workspace, thread_id, title)


def ensure_session_title_auto(
    workspace: Path,
    thread_id: str,
    user_message: str,
) -> str | None:
    """
    若尚无标题且非手动锁定，用首条用户消息自动生成标题。

    已有任意非空标题（含 auto/fallback）时不再覆盖。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param user_message 本轮用户输入
    @return 新标题；未生成返回 None
    """
    meta = load_session_meta(workspace, thread_id)
    if meta.get("title_source") == _TITLE_SOURCE_MANUAL:
        return None
    existing = get_session_title(workspace, thread_id)
    if existing and str(existing).strip():
        return None
    if not should_use_message_for_auto_title(user_message):
        return None
    suggested = suggest_full_title_from_text(user_message)
    if not suggested:
        return None
    set_session_title(workspace, thread_id, suggested, source="auto")
    _sync_plan_json_title_after_auto(workspace, thread_id, suggested)
    return suggested


def resolve_session_full_title(workspace: Path, thread_id: str) -> str:
    """
    完整会话标题（编辑/悬停/存储用，优先首条用户消息全文摘要）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 完整标题
    """
    meta = load_session_meta(workspace, thread_id)
    source = meta.get("title_source")
    stored = get_session_title(workspace, thread_id)
    if source == _TITLE_SOURCE_MANUAL and stored:
        return str(stored).strip()

    from_msg = peek_title_from_messages_jsonl(workspace, thread_id)
    if from_msg:
        return from_msg

    for fn in (
        _peek_title_from_archive_jsonl,
        _peek_title_from_anchor,
        _peek_title_from_edits,
        _peek_title_from_started_at,
    ):
        title = fn(workspace, thread_id)
        if title:
            return str(title).strip()

    if stored and not _looks_like_injected_title_fragment(str(stored)):
        return str(stored).strip()

    inferred = infer_session_title(workspace, thread_id)
    if inferred:
        return str(inferred).strip()

    return default_session_title(thread_id) or "未命名会话"


def resolve_session_display_title(workspace: Path, thread_id: str) -> str:
    """
    列表展示用标题：完整标题截断至 ≤24 字。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 展示标题
    """
    full = resolve_session_full_title(workspace, thread_id)
    short = format_display_session_title(full)
    return short or "未命名会话"
