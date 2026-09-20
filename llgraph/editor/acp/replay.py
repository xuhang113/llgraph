"""``session/load`` 续聊：把落盘的对话回放成 ``session/update``。

编辑器重启后拿着旧 sessionId 接回来时，它自己没有历史——ACP 规定由 Agent 侧
把整段对话重新推一遍，客户端照单渲染。llgraph 的历史在 ``messages.jsonl``，
所以这里做的是「存储里的消息 → 编辑器里的更新」这一层映射。

回放拿不到实时信息：工具步骤没有耗时（trace 那份没落盘），
思考也没有单独存过。所以工具只发标题 + 折过的输出，思考整段不发。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.editor.acp.updates import (
    acp_tool_kind,
    agent_message_chunk,
    agent_thought_chunk,
    text_content,
    user_message_chunk,
)

DEFAULT_MAX_MESSAGES = 200
"""回放条数上限：更早的历史模型侧仍在（由 Agent 自己从 jsonl 恢复），只是不往编辑器推。"""

DEFAULT_MAX_TOOL_OUTPUT_LINES = 20
"""每条工具结果回放的行数上限（实时那条是 40 行，回放整段历史要更省）。"""

_MAX_TEXT_CHARS = 20000


def _clip_text(text: str) -> str:
    if len(text) <= _MAX_TEXT_CHARS:
        return text
    return text[:_MAX_TEXT_CHARS] + f"\n…（本条另有 {len(text) - _MAX_TEXT_CHARS} 字未回放）"


def _tool_output_content(text: str, max_lines: int) -> list[dict[str, Any]]:
    """工具结果文本 → ACP tool_call 的 content 块（空则不给）。"""
    body = (text or "").strip()
    if not body:
        return []
    lines = body.splitlines()
    shown = lines[:max_lines]
    hidden = len(lines) - len(shown)
    joined = "\n".join(shown)
    if hidden > 0:
        joined = f"{joined}\n… 还有 {hidden} 行"
    return [{"type": "content", "content": text_content(_clip_text(joined))}]


def _tool_call_update(
    *,
    tool_call_id: str,
    tool_name: str,
    args: Any,
    output: str,
    failed: bool,
    max_output_lines: int,
) -> dict[str, Any]:
    """一次历史工具调用 → ACP ``tool_call``（直接 completed / failed，没有中间态）。"""
    # 标题沿用 trace 的「执行 工具(对象)」：同一个会话在 CLI、实时回合、回放里长一个样
    from llgraph.display.trace_display import _format_tool_step_title

    payload: dict[str, Any] = {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_call_id,
        "title": _format_tool_step_title(tool_name, args),
        "kind": acp_tool_kind(tool_name),
        "status": "failed" if failed else "completed",
    }
    content = _tool_output_content(output, max_output_lines)
    if content:
        payload["content"] = content
    return payload


def _tool_results_by_call_id(messages: list[Any]) -> dict[str, Any]:
    from langchain_core.messages import ToolMessage

    out: dict[str, Any] = {}
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        call_id = getattr(msg, "tool_call_id", "")
        if isinstance(call_id, str) and call_id:
            out[call_id] = msg
    return out


def _tool_failed(msg: Any) -> bool:
    return str(getattr(msg, "status", "") or "") == "error"


def replay_updates(
    messages: list[Any],
    *,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_tool_output_lines: int = DEFAULT_MAX_TOOL_OUTPUT_LINES,
) -> list[dict[str, Any]]:
    """
    历史消息 → ``session/update`` 序列。

    system 消息与注入的 manifest / anchor / 摘要不回放：那些是给模型看的上下文，
    编辑器里显示出来只会让用户以为自己发过这些话。

    @param messages 已恢复的消息列表（``load_session_messages`` 的结果）
    @param max_messages 回放条数上限，从最近的一端往回取
    @param max_tool_output_lines 每条工具结果的行数上限
    @return update 载荷列表，按时间顺序
    """
    from langchain_core.messages import AIMessage, ToolMessage
    from llgraph.context.message_normalize import (
        _is_business_user_message,
        _message_text,
        format_agent_chat_display_text,
    )
    from llgraph.core.user_message_content import extract_text_from_human_content
    from llgraph.session.session_meta import strip_injected_context_from_user_message

    kept = list(messages or [])
    dropped = 0
    if max_messages > 0 and len(kept) > max_messages:
        dropped = len(kept) - max_messages
        kept = kept[-max_messages:]

    results = _tool_results_by_call_id(kept)
    consumed: set[str] = set()
    updates: list[dict[str, Any]] = []
    if dropped > 0:
        # 没有「系统提示」这类更新可用；折进思考块，至少用户知道自己看到的不是全部
        updates.append(agent_thought_chunk(f"（更早的 {dropped} 条历史未回放，模型侧仍然记得）"))

    seq = 0
    for msg in kept:
        if isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", "")
            if isinstance(call_id, str) and call_id in consumed:
                continue
            # 孤立的工具结果（它那条 AIMessage 被条数上限切掉了）也要给编辑器一行
            seq += 1
            updates.append(
                _tool_call_update(
                    tool_call_id=f"load_{seq}",
                    tool_name=str(getattr(msg, "name", "") or "tool"),
                    args=None,
                    output=_message_text(getattr(msg, "content", "")),
                    failed=_tool_failed(msg),
                    max_output_lines=max_tool_output_lines,
                )
            )
            continue

        if isinstance(msg, AIMessage):
            body = format_agent_chat_display_text(_message_text(getattr(msg, "content", "")))
            if body:
                updates.append(agent_message_chunk(_clip_text(body)))
            for call in getattr(msg, "tool_calls", None) or []:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                result = results.get(call_id) if isinstance(call_id, str) else None
                if isinstance(call_id, str) and call_id:
                    consumed.add(call_id)
                seq += 1
                updates.append(
                    _tool_call_update(
                        tool_call_id=f"load_{seq}",
                        tool_name=str(call.get("name") or "tool"),
                        args=call.get("args"),
                        output=(
                            _message_text(getattr(result, "content", "")) if result else ""
                        ),
                        failed=_tool_failed(result) if result else False,
                        max_output_lines=max_tool_output_lines,
                    )
                )
            continue

        if _is_business_user_message(msg):
            raw = extract_text_from_human_content(getattr(msg, "content", ""))
            body = strip_injected_context_from_user_message(raw)
            if body:
                updates.append(user_message_chunk(_clip_text(body)))

    return updates


def load_session_updates(
    workspace: Path,
    thread_id: str,
    *,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_tool_output_lines: int = DEFAULT_MAX_TOOL_OUTPUT_LINES,
) -> list[dict[str, Any]]:
    """
    读 ``messages.jsonl`` 并回放成 ``session/update``。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param max_messages 回放条数上限
    @param max_tool_output_lines 每条工具结果的行数上限
    @return update 载荷列表；没有历史时为空
    """
    from llgraph.session.session_file_store import load_session_messages

    messages = load_session_messages(workspace, thread_id)
    return replay_updates(
        messages,
        max_messages=max_messages,
        max_tool_output_lines=max_tool_output_lines,
    )


def session_is_resumable(workspace: Path, thread_id: str) -> bool:
    """
    这个工作区下有没有这个会话。

    只有正文 jsonl 不够：``session/new`` 之后一句话都没说就重启，
    会话只有 meta.json，那也该能接回去（回放为空而已）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 是否可续
    """
    from llgraph.session.session_file_store import session_has_messages_file
    from llgraph.session.session_meta import session_meta_json_path

    if session_has_messages_file(workspace, thread_id):
        return True
    return session_meta_json_path(workspace, thread_id).is_file()
