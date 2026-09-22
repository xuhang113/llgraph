"""trace 事件 → ACP ``session/update`` 载荷。"""

from __future__ import annotations

import re
from typing import Any

ACP_PROTOCOL_VERSION = 1

_TOOL_TITLE_RE = re.compile(r"^执行\s+([A-Za-z_][\w.:-]*)")

# ACP 的 ToolKind：编辑器据此选图标/折叠样式，认不出的一律 other
_TOOL_KIND_BY_NAME: dict[str, str] = {
    "read_file": "read",
    "list_dir": "read",
    "read_many_files": "read",
    "write_file": "edit",
    "search_replace": "edit",
    "apply_patch": "edit",
    "delete_file": "delete",
    "grep": "search",
    "glob_file_search": "search",
    "codebase_search": "search",
    "search_code": "search",
    "search_code_hybrid": "search",
    "search_history": "search",
    "run_terminal_cmd": "execute",
    "shell": "execute",
    "web_search": "fetch",
    "fetch_url": "fetch",
    "spawn_subagent": "think",
}


def tool_name_from_title(title: str) -> str:
    """
    从 trace 步骤标题里取工具名（标题形如 ``执行 read_file(a.txt)``）。

    @param title 步骤标题
    @return 工具名；取不到则返回原标题
    """
    match = _TOOL_TITLE_RE.match((title or "").strip())
    if match:
        return match.group(1)
    return (title or "").strip()


def _kind_by_verb() -> dict[str, str]:
    groups = {
        "read": (
            "read", "get", "cat", "view", "show", "list", "ls", "head", "tail",
            "describe", "inspect", "info", "status", "log", "logs", "diff",
        ),
        "search": ("search", "grep", "find", "query", "lookup", "glob"),
        "edit": (
            "write", "edit", "patch", "apply", "create", "update", "insert",
            "append", "replace", "rename", "move", "commit", "post", "push",
            "upload", "add", "set",
        ),
        "delete": ("delete", "remove", "rm", "drop", "purge", "destroy"),
        "execute": (
            "run", "exec", "execute", "shell", "bash", "invoke", "call",
            "start", "stop", "restart", "build",
        ),
        "fetch": ("fetch", "download", "crawl", "browse", "navigate", "open"),
        "think": ("think", "plan", "reason", "spawn"),
    }
    return {verb: kind for kind, verbs in groups.items() for verb in verbs}


_KIND_BY_VERB = _kind_by_verb()

# MCP 命名常带一层 Server 前缀（`github_list_issues`），前两个词都算动词位
_VERB_ZONE = 2


def acp_tool_kind(tool_name: str) -> str:
    """
    工具名映射到 ACP ToolKind。

    llgraph 自带工具查表（`search_replace` 打头是 search，其实是编辑，靠表纠正）；
    MCP 工具按动词位判，分词沿用 ``permissions.mcp.split_tool_name``，
    免得同一批工具名在两处被切成不同的词。

    @param tool_name 工具名
    @return read|edit|delete|search|execute|fetch|think|other
    """
    from llgraph.permissions.mcp import split_tool_name

    name = (tool_name or "").strip()
    kind = _TOOL_KIND_BY_NAME.get(name)
    if kind:
        return kind
    tokens = split_tool_name(name)
    for token in tokens[:_VERB_ZONE]:
        if token in _KIND_BY_VERB:
            return _KIND_BY_VERB[token]
    for token in tokens[_VERB_ZONE:]:
        if token in _KIND_BY_VERB:
            return _KIND_BY_VERB[token]
    return "other"


def text_content(text: str) -> dict[str, Any]:
    """@param text 文本 @return ACP ContentBlock"""
    return {"type": "text", "text": text}


def agent_message_chunk(text: str) -> dict[str, Any]:
    """@param text 助手正文片段 @return session/update 载荷"""
    return {"sessionUpdate": "agent_message_chunk", "content": text_content(text)}


def agent_thought_chunk(text: str) -> dict[str, Any]:
    """@param text 思考片段 @return session/update 载荷"""
    return {"sessionUpdate": "agent_thought_chunk", "content": text_content(text)}


def user_message_chunk(text: str) -> dict[str, Any]:
    """@param text 用户消息 @return session/update 载荷"""
    return {"sessionUpdate": "user_message_chunk", "content": text_content(text)}


def prompt_text(blocks: Any) -> str:
    """
    ACP prompt（ContentBlock 数组）取纯文本。

    resource / resource_link 折成一行路径提示：llgraph 有自己的文件工具，
    正文里给出路径比把整份文件塞进 user message 更省上下文。

    @param blocks prompt 数组
    @return 拼好的文本
    """
    if isinstance(blocks, str):
        return blocks.strip()
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        elif btype == "resource_link":
            uri = str(block.get("uri") or "").strip()
            name = str(block.get("name") or "").strip()
            label = name or uri
            if label:
                parts.append(f"[引用] {label}" + (f" ({uri})" if name and uri else ""))
        elif btype == "resource":
            resource = block.get("resource")
            if isinstance(resource, dict):
                uri = str(resource.get("uri") or "").strip()
                text = resource.get("text")
                if isinstance(text, str) and text.strip():
                    header = f"[引用 {uri}]" if uri else "[引用]"
                    parts.append(f"{header}\n{text}")
                elif uri:
                    parts.append(f"[引用] {uri}")
    return "\n\n".join(p.strip() for p in parts if p.strip()).strip()


def tool_call_pending(
    tool_call_id: str,
    *,
    title: str,
    tool_name: str,
) -> dict[str, Any]:
    """
    模型刚决定要调的工具 → ACP ``tool_call``（pending）。

    @param tool_call_id 本会话内唯一的工具调用 id
    @param title 步骤标题（与跑完那条一致，编辑器里不会换名字）
    @param tool_name 工具名（定 kind）
    @return session/update 载荷
    """
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_call_id,
        "title": title or "执行工具",
        "kind": acp_tool_kind(tool_name),
        "status": "pending",
    }


def tool_call_status(tool_call_id: str, status: str) -> dict[str, Any]:
    """
    只改状态的 ``tool_call_update``（pending → in_progress / failed）。

    @param tool_call_id 已发过 ``tool_call`` 的那个 id
    @param status ACP ToolCallStatus
    @return session/update 载荷
    """
    return {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
        "status": status,
    }


def is_tool_call_step(step: dict[str, Any]) -> bool:
    """@param step trace 步骤 dict @return 是否对应编辑器里的一次工具调用"""
    return str(step.get("kind") or "") in ("tool", "explore")


def tool_call_from_step(
    step: dict[str, Any],
    *,
    tool_call_id: str,
    max_content_lines: int = 40,
    as_update: bool = False,
) -> dict[str, Any] | None:
    """
    trace 工具步骤 → ACP 工具调用更新（completed）。

    trace 的步骤在工具跑完后才登记（那时才有耗时与输出）。这条调用此前若已经以
    pending 报过（``tool_call_pending``），收尾必须走 ``tool_call_update``——
    再发一条 ``tool_call`` 会让编辑器多画一行。

    ``as_update`` 时只报状态与输出，不重发标题 / kind：工具节点的输出里没有调用参数，
    照它算出来的标题会从 ``执行 read_file(a.txt)`` 退化成 ``执行 read_file``。

    @param step trace 步骤 dict
    @param tool_call_id 本会话内唯一的工具调用 id
    @param max_content_lines 回填给编辑器的输出行数上限
    @param as_update True 时发 ``tool_call_update``（这条调用已经报过 pending）
    @return session/update 载荷；非工具步骤返回 None
    """
    if not is_tool_call_step(step):
        return None
    kind = str(step.get("kind") or "")
    title = str(step.get("title") or "").strip() or "执行工具"
    tool_name = tool_name_from_title(title)
    body_lines = step.get("body_lines")
    lines = [str(x) for x in body_lines] if isinstance(body_lines, list) else []
    payload: dict[str, Any] = {
        "sessionUpdate": "tool_call_update" if as_update else "tool_call",
        "toolCallId": tool_call_id,
        "status": "completed",
    }
    if not as_update:
        payload["title"] = title
        payload["kind"] = "think" if kind == "explore" else acp_tool_kind(tool_name)
    if lines:
        shown = lines[:max_content_lines]
        hidden = len(lines) - len(shown)
        text = "\n".join(shown)
        if hidden > 0:
            text = f"{text}\n… 还有 {hidden} 行"
        payload["content"] = [{"type": "content", "content": text_content(text)}]
    return payload
