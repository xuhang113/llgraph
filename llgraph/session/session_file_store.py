"""会话对话持久化：messages.jsonl（可读可维护，无 SQLite）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_from_dict, messages_to_dict

from llgraph.core.user_message_content import strip_inline_images_from_messages
from llgraph.session.jsonl_read import open_jsonl_for_read
from llgraph.session.user_storage import session_messages_path, session_thread_dir, user_sessions_root


def save_session_messages(
    workspace: Path,
    thread_id: str,
    messages: list[BaseMessage],
) -> str | None:
    """
    将当前对话写入 messages.jsonl。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param messages LangChain 消息列表
    @return 落盘路径；失败返回 None
    """
    if not thread_id.strip():
        return None
    path = session_messages_path(workspace, thread_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in messages_to_dict(messages):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        _write_session_meta(workspace, thread_id, len(messages))
        return str(path)
    except OSError:
        return None


def load_session_messages(workspace: Path, thread_id: str) -> list[BaseMessage]:
    """
    从 messages.jsonl 加载对话。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 消息列表；无文件或解析失败返回空列表
    """
    path = session_messages_path(workspace, thread_id)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open_jsonl_for_read(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not rows:
        return []
    try:
        loaded = list(messages_from_dict(rows))
    except Exception:
        return []
    from llgraph.context.message_canonical import to_canonical_v2_messages

    cleaned, _report = to_canonical_v2_messages(loaded)
    cleaned, stripped = strip_inline_images_from_messages(cleaned)
    if stripped:
        save_session_messages(workspace, thread_id, cleaned)
    return cleaned


def session_has_messages_file(workspace: Path, thread_id: str) -> bool:
    """
    是否存在可恢复的 messages.jsonl。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 是否可恢复对话正文
    """
    return session_messages_path(workspace, thread_id).is_file()


def restore_session_to_agent(
    agent: Any,
    workspace: Path,
    thread_id: str,
) -> int:
    """
    从 jsonl 恢复到 Agent 内存状态（MemorySaver）。

    @param agent LangGraph agent
    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 加载的消息条数
    """
    from llgraph.context.message_canonical import to_canonical_v2_messages

    path = session_messages_path(workspace, thread_id)
    if not path.is_file():
        return 0
    rows: list[dict] = []
    try:
        with open_jsonl_for_read(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if not rows:
        return 0
    try:
        raw_messages = list(messages_from_dict(rows))
    except Exception:
        return 0

    messages, report = to_canonical_v2_messages(raw_messages)
    messages, stripped = strip_inline_images_from_messages(messages)
    if report.changed or stripped:
        save_session_messages(workspace, thread_id, messages)

    config = {"configurable": {"thread_id": thread_id}}
    try:
        agent.update_state(config, {"messages": messages})
    except Exception:
        return 0
    return len(messages)


def prepare_resumable_agent_session(
    agent: Any,
    workspace: Path,
    thread_id: str,
    context_session: ContextSession,
    *,
    user_message: str = "",
) -> int:
    """
    非交互/交互续聊前：刷新 manifest 并从 messages.jsonl 恢复到 Agent 状态。

    @param agent LangGraph agent
    @param workspace 工作区根
    @param thread_id 会话 ID
    @param context_session Rule/Skill 会话
    @param user_message 当前用户消息（manifest 目录用）
    @return 恢复的消息条数；无历史为 0
    """
    from llgraph.session.session_manifest import sync_session_manifest_to_agent_state

    sync_session_manifest_to_agent_state(
        agent,
        thread_id=thread_id,
        workspace=workspace,
        session=context_session,
        user_message=user_message,
        with_memory=True,
    )
    return restore_session_to_agent(agent, workspace, thread_id)


def append_pending_user_turn(
    workspace: Path,
    thread_id: str,
    user_message: str,
    *,
    image_refs: list | None = None,
) -> str | None:
    """
    轮次开始即将用户原文追加到 messages.jsonl，刷新页面时可恢复最新提问。

    整轮结束后 persist_agent_session 会用 Agent 状态全量覆盖。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param user_message 用户输入原文（不含 workspace-context）
    @param image_refs 可选图片附件引用（Web 多模态）
    @return 落盘路径；无内容或失败返回 None
    """
    from llgraph.context.context_continuity import strip_workspace_context_wrapper
    from llgraph.core.user_message_content import (
        StoredImageRef,
        build_stored_user_content,
        extract_text_from_human_content,
    )

    text = user_message.strip()
    refs: list[StoredImageRef] = list(image_refs or [])
    if not thread_id.strip() or (not text and not refs):
        return None

    stored_content = build_stored_user_content(text, image_refs=refs)

    existing = load_session_messages(workspace, thread_id)
    if existing:
        last = existing[-1]
        if isinstance(last, HumanMessage):
            last_text = strip_workspace_context_wrapper(
                extract_text_from_human_content(last.content)
            ).strip()
            if last_text == text and not refs:
                return str(session_messages_path(workspace, thread_id))
        elif not isinstance(last, AIMessage):
            pass

    new_messages = [*existing, HumanMessage(content=stored_content)]
    from llgraph.context.message_canonical import to_canonical_v2_messages

    cleaned, _report = to_canonical_v2_messages(new_messages)
    return save_session_messages(workspace, thread_id, cleaned)


def persist_agent_session(
    agent: Any,
    workspace: Path,
    thread_id: str,
    *,
    turn_image_refs: list | None = None,
) -> str | None:
    """
    从 Agent 状态读取并落盘 messages.jsonl。

    @param agent LangGraph agent
    @param workspace 工作区根
    @param thread_id 会话 ID
    @param turn_image_refs 本轮新上传附件；落盘前将内联图替换为 image_ref
    @return 落盘路径；无消息或失败返回 None
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        messages = list((state.values or {}).get("messages") or [])
    except Exception:
        return None
    if not messages:
        return None
    from llgraph.context.message_canonical import to_canonical_v2_messages
    from llgraph.session.session_image_store import canonicalize_messages_image_refs

    if turn_image_refs:
        messages = canonicalize_messages_image_refs(
            messages,
            turn_image_refs=turn_image_refs,
        )
        try:
            agent.update_state(config, {"messages": messages})
        except Exception:
            pass

    cleaned, _report = to_canonical_v2_messages(messages)
    cleaned, stripped = strip_inline_images_from_messages(cleaned)
    if stripped or turn_image_refs:
        try:
            agent.update_state(config, {"messages": cleaned})
        except Exception:
            pass

    return save_session_messages(workspace, thread_id, cleaned)


def purge_legacy_inline_images_in_workspace(workspace: Path) -> int:
    """
    扫描工作区全部会话，清除 messages.jsonl 中废弃的内联 base64 image 块。

    @param workspace 工作区根
    @return 清理的会话数
    """
    root = user_sessions_root(workspace)
    if not root.is_dir():
        return 0
    cleaned_sessions = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        thread_id = child.name
        msg_path = session_messages_path(workspace, thread_id)
        if not msg_path.is_file():
            continue
        before = msg_path.read_text(encoding="utf-8")
        load_session_messages(workspace, thread_id)
        after = msg_path.read_text(encoding="utf-8") if msg_path.is_file() else ""
        if before != after:
            cleaned_sessions += 1
    return cleaned_sessions


def _write_session_meta(workspace: Path, thread_id: str, message_count: int) -> None:
    from llgraph.session.session_meta import load_session_meta, save_session_meta

    existing = load_session_meta(workspace, thread_id)
    from llgraph.context.message_canonical import CANONICAL_FORMAT_VERSION

    patch: dict[str, object] = {
        "workspace": str(workspace.expanduser().resolve()),
        "message_count": message_count,
        "store": "messages.jsonl",
        "messages_format": f"canonical_v{CANONICAL_FORMAT_VERSION}",
    }
    if existing.get("title"):
        patch["title"] = existing["title"]
    if existing.get("title_source"):
        patch["title_source"] = existing["title_source"]
    save_session_meta(workspace, thread_id, patch)
