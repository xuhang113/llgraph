"""会话对话持久化：messages.jsonl（可读可维护，无 SQLite）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from llgraph.session.user_storage import session_messages_path, session_thread_dir


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
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    if not rows:
        return []
    try:
        loaded = list(messages_from_dict(rows))
    except Exception:
        return []
    from llgraph.context.message_canonical import to_canonical_v2_messages

    cleaned, _report = to_canonical_v2_messages(loaded)
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
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return 0
    if not rows:
        return 0
    try:
        raw_messages = list(messages_from_dict(rows))
    except Exception:
        return 0

    messages, report = to_canonical_v2_messages(raw_messages)
    if report.changed:
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


def persist_agent_session(
    agent: Any,
    workspace: Path,
    thread_id: str,
) -> str | None:
    """
    从 Agent 状态读取并落盘 messages.jsonl。

    @param agent LangGraph agent
    @param workspace 工作区根
    @param thread_id 会话 ID
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

    cleaned, _report = to_canonical_v2_messages(messages)
    return save_session_messages(workspace, thread_id, cleaned)


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
