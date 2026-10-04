"""trace 事件 → ACP ``session/update`` 载荷。"""

from __future__ import annotations

import re
from pathlib import Path
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


def absolute_workspace_path(path: Any, workspace: Path | None) -> str | None:
    """
    工具参数里的路径 → ACP 要的绝对路径。

    ACP（locations 与 diff 块）只认绝对路径，相对路径按工作区根补齐；
    没有工作区根时相对路径直接丢掉——报一条编辑器打不开的路径，
    点下去是个报错，不如不报。

    @param path 路径（相对工作区或绝对）
    @param workspace 工作区根
    @return 绝对路径；补不出来时 None
    """
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = Path(path.strip()).expanduser()
    if not candidate.is_absolute():
        if workspace is None:
            return None
        candidate = Path(workspace) / candidate
    return str(candidate)


def tool_call_locations(
    paths: Any,
    workspace: Path | None,
) -> list[dict[str, str]]:
    """
    受影响文件 → ACP ``ToolCallLocation``（编辑器据此让那一行可点开跳转）。

    @param paths 路径列表（相对工作区或绝对）
    @param workspace 工作区根
    @return locations 列表；一个都补不出来时为空
    """
    if not isinstance(paths, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in paths:
        text = absolute_workspace_path(item, workspace)
        if text is None or text in seen:
            continue
        seen.add(text)
        out.append({"path": text})
    return out


# 一个 diff 块要把改前 + 改后全文都推给编辑器。超过这个量就不发了：
# 生成的文件动辄几十万字符，一条更新能把 stdio 通道堵住，而那种文件的逐行 diff
# 在编辑器里本来也没人读。此时那次调用仍有原来的纯文本输出。
MAX_DIFF_CHARS = 200_000


def tool_call_diffs(edits: Any, workspace: Path | None) -> list[dict[str, Any]]:
    """
    写工具报上来的改动 → ACP ``diff`` 内容块（编辑器里渲染成改动预览）。

    新建文件的 ``oldText`` 是 null（ACP 据此画成整份新增），与授权弹窗
    （``permission.py``）同一个写法，免得同一次改动在弹窗与结果里长得不一样。

    @param edits ``[{"path": ..., "old_text": ..., "new_text": ...}]``
    @param workspace 工作区根（相对路径补绝对用）
    @return diff 块列表；一个都发不出时为空
    """
    if not isinstance(edits, list):
        return []
    out: list[dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        abs_path = absolute_workspace_path(edit.get("path"), workspace)
        if abs_path is None:
            continue
        old_text = edit.get("old_text") or ""
        new_text = edit.get("new_text")
        if not isinstance(new_text, str) or not isinstance(old_text, str):
            continue
        if old_text == new_text:
            continue
        if len(old_text) + len(new_text) > MAX_DIFF_CHARS:
            continue
        out.append(
            {
                "type": "diff",
                "path": abs_path,
                "oldText": old_text or None,
                "newText": new_text,
            }
        )
    return out


def terminal_content(terminal_id: str) -> dict[str, Any]:
    """
    编辑器终端 → ACP ``terminal`` 内容块（编辑器据此在那一行里画实时终端）。

    @param terminal_id ``terminal/create`` 给的 id
    @return ContentBlock
    """
    return {"type": "terminal", "terminalId": terminal_id}


def tool_call_terminal(
    tool_call_id: str,
    terminal_ids: list[str],
) -> dict[str, Any]:
    """
    命令刚挂上编辑器终端 → 只换 ``content`` 的 ``tool_call_update``。

    这条必须**马上**发：实时输出就靠编辑器拿着这个 id 自己渲染，
    攒到收尾再发就只剩一份跑完的文本，和本轮之前没区别。

    @param tool_call_id 已发过 ``tool_call`` 的那个 id
    @param terminal_ids 这次调用的终端 id（给全量：content 是整块替换的）
    @return session/update 载荷
    """
    return {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
        "content": [terminal_content(tid) for tid in terminal_ids],
    }


# ACP 的 PlanEntryStatus 只有三档。llgraph 的 cancelled 没有对应项：留成 pending
# 会让编辑器里那张表永远显示「还有活没干」，所以按「不用再做了」归到 completed，
# 正文前面加一句标记，免得看起来像真做完了。
_PLAN_STATUS = {
    "pending": "pending",
    "in_progress": "in_progress",
    "completed": "completed",
    "cancelled": "completed",
}
CANCELLED_PLAN_PREFIX = "（已取消）"

# ACP 的 PlanEntry 必填 priority，而 llgraph 的清单没有优先级这个字段。
# 按顺序硬编一个高低只是凭空造数据，一律中档。
PLAN_PRIORITY = "medium"

# 与 ``todo_store.MAX_TODOS`` 同量级的保险：清单本来就在工具侧截过，
# 这里再兜一层，免得别的调用方塞进来一张几千条的表把一条更新撑爆
MAX_PLAN_ENTRIES = 20


def plan_entries(items: Any) -> list[dict[str, Any]]:
    """
    任务清单条目 → ACP ``PlanEntry`` 列表。

    清单里的 id 不往外发：ACP 的计划是全量快照，没有「按 id 更新某一条」这回事，
    编辑器只按顺序渲染。

    @param items ``[{"content": ..., "status": ...}]``（status 用 llgraph 的四档）
    @return entries 列表；一条都发不出时为空
    """
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = " ".join(str(item.get("content") or "").split())
        if not content:
            continue
        raw_status = str(item.get("status") or "").strip()
        if raw_status == "cancelled":
            content = f"{CANCELLED_PLAN_PREFIX}{content}"
        out.append(
            {
                "content": content,
                "priority": PLAN_PRIORITY,
                "status": _PLAN_STATUS.get(raw_status, "pending"),
            }
        )
        if len(out) >= MAX_PLAN_ENTRIES:
            break
    return out


def session_plan(items: Any) -> dict[str, Any]:
    """
    任务清单 → ACP ``plan``（编辑器里的计划清单）。

    这条是**全量替换**：每次都把整张表发过去，编辑器照单重画。

    @param items ``[{"content": ..., "status": ...}]``
    @return session/update 载荷
    """
    return {"sessionUpdate": "plan", "entries": plan_entries(items)}


def tool_call_pending(
    tool_call_id: str,
    *,
    title: str,
    tool_name: str,
    locations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    模型刚决定要调的工具 → ACP ``tool_call``（pending）。

    @param tool_call_id 本会话内唯一的工具调用 id
    @param title 步骤标题（与跑完那条一致，编辑器里不会换名字）
    @param tool_name 工具名（定 kind）
    @param locations 受影响文件（绝对路径）；空则不带这个字段
    @return session/update 载荷
    """
    payload: dict[str, Any] = {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_call_id,
        "title": title or "执行工具",
        "kind": acp_tool_kind(tool_name),
        "status": "pending",
    }
    if locations:
        payload["locations"] = locations
    return payload


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
    diffs: list[dict[str, Any]] | None = None,
    terminals: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    trace 工具步骤 → ACP 工具调用更新（completed / failed）。

    trace 的步骤在工具跑完后才登记（那时才有耗时与输出）。这条调用此前若已经以
    pending 报过（``tool_call_pending``），收尾必须走 ``tool_call_update``——
    再发一条 ``tool_call`` 会让编辑器多画一行。

    ``as_update`` 时只报状态与输出，不重发标题 / kind：工具节点的输出里没有调用参数，
    照它算出来的标题会从 ``执行 read_file(a.txt)`` 退化成 ``执行 read_file``。

    工具没抛异常也可能是失败的（参数校验错、``old_string`` 没匹配上），
    trace 在登记步骤时已经判过（``tool_failed``），这里照它报 ``failed``：
    一律 ``completed`` 的话，编辑器里一次失败的改写和一次成功的改写长得一模一样。

    ``diffs`` 排在纯文本输出前面：编辑器里第一眼要看的是「这一刀改了什么」，
    工具返回的那段文本（诊断、分块提示）是补充说明。

    这次调用挂了编辑器终端时**不再回填文本输出**：终端里已经是同一段输出的全文
    （还带退出状态），再在下面附一份截断过的副本只是同样的东西看两遍。
    收尾仍要把终端块重发一遍——``content`` 是整块替换的，不重发就被文本顶掉了。

    @param step trace 步骤 dict
    @param tool_call_id 本会话内唯一的工具调用 id
    @param max_content_lines 回填给编辑器的输出行数上限
    @param as_update True 时发 ``tool_call_update``（这条调用已经报过 pending）
    @param diffs 这次调用落下的改动块（``tool_call_diffs`` 的结果）
    @param terminals 这次调用挂上的编辑器终端 id
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
        "status": "failed" if step.get("tool_failed") else "completed",
    }
    if not as_update:
        payload["title"] = title
        payload["kind"] = "think" if kind == "explore" else acp_tool_kind(tool_name)
    content: list[dict[str, Any]] = list(diffs or [])
    content.extend(terminal_content(tid) for tid in terminals or [])
    if lines and not terminals:
        shown = lines[:max_content_lines]
        hidden = len(lines) - len(shown)
        text = "\n".join(shown)
        if hidden > 0:
            text = f"{text}\n… 还有 {hidden} 行"
        content.append({"type": "content", "content": text_content(text)})
    if content:
        payload["content"] = content
    return payload
