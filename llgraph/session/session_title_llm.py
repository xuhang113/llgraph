"""首轮结束后用 LLM 生成更友好的会话标题。"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import HumanMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_response import llm_response_text
from llgraph.session.jsonl_read import open_jsonl_for_read
from llgraph.session.session_meta import (
    _extract_human_content_from_jsonl_row,
    _sync_plan_json_title_after_auto,
    extract_user_body_for_title,
    get_session_title,
    is_weak_auto_session_title,
    load_session_meta,
    save_session_meta,
    set_session_title,
    suggest_full_title_from_text,
)
from llgraph.session.user_storage import session_messages_path

logger = logging.getLogger(__name__)

_TITLE_REFRESH_LOCK = threading.Lock()
_IN_FLIGHT: set[str] = set()
_TITLE_LLM_MAX_TOKENS = 64
_TITLE_USER_BODY_MAX = 2400
_TITLE_REPLY_MAX = 800


def _count_user_messages(workspace: Path, thread_id: str) -> int:
    path = session_messages_path(workspace, thread_id)
    if not path.is_file():
        return 0
    count = 0
    try:
        with open_jsonl_for_read(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                content = _extract_human_content_from_jsonl_row(row)
                if content.strip():
                    count += 1
    except (OSError, json.JSONDecodeError):
        return count
    return count


def _should_refresh_title_with_llm(workspace: Path, thread_id: str) -> bool:
    meta = load_session_meta(workspace, thread_id)
    if meta.get("title_source") == "manual":
        return False
    if meta.get("title_llm_refreshed"):
        return False
    if _count_user_messages(workspace, thread_id) != 1:
        return False
    stored = get_session_title(workspace, thread_id) or ""
    return not stored.strip() or is_weak_auto_session_title(stored)


def _build_title_prompt(user_body: str, assistant_preview: str) -> str:
    return (
        "你是 coding agent 的会话标题生成器。根据用户首条消息与助手首轮回复，"
        "生成一个简短中文标题。\n"
        "要求：\n"
        "- 8～20 字，概括用户核心问题或任务\n"
        "- 不要代码片段、键名字段、URL、类名包名\n"
        "- 不要引号，不要句号结尾\n"
        "- 只输出标题一行\n\n"
        f"用户消息：\n{user_body[:_TITLE_USER_BODY_MAX]}\n\n"
        f"助手回复摘要：\n{(assistant_preview or '（无）')[:_TITLE_REPLY_MAX]}"
    )


def _normalize_llm_title(raw: str) -> str:
    line = str(raw or "").strip().splitlines()[0].strip()
    line = line.strip("\"'「」『』")
    line = re.sub(r"\s+", " ", line).strip()
    if line.endswith("。"):
        line = line[:-1].strip()
    return line


def _invoke_session_title_llm(
    workspace: Path,
    user_body: str,
    assistant_preview: str,
) -> str | None:
    if not user_body.strip():
        return None
    try:
        llm = create_gateway_llm(workspace)
        llm = llm.bind(max_tokens=_TITLE_LLM_MAX_TOKENS)
        response = llm.invoke([
            HumanMessage(content=_build_title_prompt(user_body, assistant_preview)),
        ])
        candidate = _normalize_llm_title(llm_response_text(response))
        if not candidate or is_weak_auto_session_title(candidate):
            return None
        return candidate
    except Exception:
        logger.debug("session title LLM failed", exc_info=True)
        return None


def _first_user_message_from_disk(workspace: Path, thread_id: str) -> str | None:
    path = session_messages_path(workspace, thread_id)
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
                if content.strip():
                    return content
    except (OSError, json.JSONDecodeError):
        return None
    return None


def maybe_refresh_session_title_with_llm(
    workspace: Path,
    thread_id: str,
    *,
    user_message: str,
    assistant_reply: str = "",
) -> str | None:
    """
    首轮结束后尝试用 LLM 刷新劣质/缺失的自动标题。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param user_message 本轮用户输入
    @param assistant_reply 助手首轮可见回复
    @return 新标题；未刷新返回 None
    """
    if not _should_refresh_title_with_llm(workspace, thread_id):
        return None

    user_body = extract_user_body_for_title(user_message)
    if not user_body.strip():
        user_body = extract_user_body_for_title(
            _first_user_message_from_disk(workspace, thread_id) or user_message,
        )
    title = _invoke_session_title_llm(workspace, user_body, assistant_reply)
    if not title:
        fallback = suggest_full_title_from_text(user_body or user_message)
        if not fallback or is_weak_auto_session_title(fallback):
            save_session_meta(workspace, thread_id, {"title_llm_refreshed": True})
            return None
        title = fallback

    ok, _ = set_session_title(workspace, thread_id, title, source="auto")
    if not ok:
        return None
    _sync_plan_json_title_after_auto(workspace, thread_id, title)
    save_session_meta(workspace, thread_id, {"title_llm_refreshed": True})
    return get_session_title(workspace, thread_id)


def schedule_session_title_llm_refresh(
    workspace: Path,
    thread_id: str,
    *,
    user_message: str,
    assistant_reply: str = "",
    on_updated: Callable[[str], None] | None = None,
) -> None:
    """
    后台线程刷新会话标题（不阻塞 turn_done / invoke 返回）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param user_message 用户输入
    @param assistant_reply 助手回复
    @param on_updated 标题更新回调（Web SSE 用）
    """
    ws = workspace.expanduser().resolve()
    key = f"{ws}:{thread_id}"
    with _TITLE_REFRESH_LOCK:
        if key in _IN_FLIGHT:
            return
        _IN_FLIGHT.add(key)

    def worker() -> None:
        try:
            title = maybe_refresh_session_title_with_llm(
                ws,
                thread_id,
                user_message=user_message,
                assistant_reply=assistant_reply,
            )
            if title and on_updated is not None:
                on_updated(title)
        finally:
            with _TITLE_REFRESH_LOCK:
                _IN_FLIGHT.discard(key)

    threading.Thread(
        target=worker,
        daemon=True,
        name=f"title-{thread_id[:12]}",
    ).start()
