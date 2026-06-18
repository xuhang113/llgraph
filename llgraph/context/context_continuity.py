"""续写/重写场景：向 <workspace-context> 注入会话连续性提示。"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import ai_message_has_tool_calls
from llgraph.context.message_normalize import _message_text

_WORKSPACE_CONTEXT_RE = re.compile(
    r"<workspace-context>.*?</workspace-context>\s*",
    re.DOTALL | re.IGNORECASE,
)
_CONTINUE_INTENT_RE = re.compile(
    r"(重写|再来一份|再写一份|继续|接着|那份|刚才|上一轮|允许写了)",
    re.IGNORECASE,
)
_PATH_IN_TOOL_RE = re.compile(r"[`'\"]?([^\s`'\"><|]+\.(?:md|mdc|txt|json|yaml|yml))[`'\"]?", re.I)


def strip_workspace_context_wrapper(user_message: str) -> str:
    """
    去掉 <workspace-context> 外壳，取用户真实输入。

    @param user_message 可能含 workspace-context 的消息
    @return 用户正文
    """
    text = user_message.strip()
    if not text:
        return ""
    return _WORKSPACE_CONTEXT_RE.sub("", text).strip()


def is_continue_or_rewrite_intent(user_message: str) -> bool:
    """
    是否像「重写/继续/那份」类延续指令。

    @param user_message 用户消息
    @return 是否命中
    """
    tail = strip_workspace_context_wrapper(user_message)
    if not tail:
        return False
    return bool(_CONTINUE_INTENT_RE.search(tail))


def _last_assistant_text(messages: list[BaseMessage], *, max_chars: int) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if ai_message_has_tool_calls(msg):
            continue
        text = _message_text(getattr(msg, "content", "")).strip()
        if not text or text.startswith("【规划】"):
            continue
        if len(text) > max_chars:
            return text[: max_chars - 20] + "\n…（上轮助手正文已截断）"
        return text
    return ""


def _recent_read_paths(messages: list[BaseMessage], *, limit: int = 8) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "")
        if name not in ("read_file", "read_files"):
            continue
        content = _message_text(getattr(msg, "content", ""))
        if "已省略" in content or "已跳过" in content or "已归档" in content:
            continue
        for match in _PATH_IN_TOOL_RE.finditer(content[:4000]):
            rel = match.group(1).strip().lstrip("./")
            if rel and rel not in seen and "/" not in rel[:1]:
                seen.add(rel)
                paths.append(rel)
        if name == "read_files":
            for line in content.splitlines()[:12]:
                line = line.strip()
                if line.startswith("--- ") and "(" in line:
                    inner = line.removeprefix("--- ").split(" (", 1)[0].strip()
                    if inner and inner not in seen:
                        seen.add(inner)
                        paths.append(inner)
        if len(paths) >= limit:
            break
    return paths[:limit]


def build_continuity_context_hint(
    messages: list[BaseMessage] | None,
    *,
    user_message: str,
    edited_paths: list[str] | None = None,
    max_assistant_chars: int = 1200,
) -> str:
    """
    续写/重写时拼入 workspace-context 的连续性块。

    @param messages 当前会话消息（落盘/canonical）
    @param user_message 本轮用户消息
    @param edited_paths 本会话已改文件路径
    @param max_assistant_chars 上轮助手正文上限
    @return Markdown；无命中时空串
    """
    if not is_continue_or_rewrite_intent(user_message):
        return ""
    if not messages:
        messages = []

    lines = ["## 会话连续性（续写/重写）", ""]
    lines.append(
        "用户意图为延续上一轮产出；**优先复用下列信息与历史助手正文**，"
        "勿重复 list_directory/glob 摸底，除非路径已失效。"
    )

    if edited_paths:
        lines.append("")
        lines.append("本会话已改文件（优先在此基础上修改）：")
        for path in edited_paths[:12]:
            lines.append(f"- `{path}`")

    read_paths = _recent_read_paths(messages)
    if read_paths:
        lines.append("")
        lines.append("近期已读文件（可直接 read_file 精读或复用结论）：")
        for path in read_paths:
            lines.append(f"- `{path}`")

    assistant = _last_assistant_text(messages, max_chars=max_assistant_chars)
    if assistant:
        lines.append("")
        lines.append("上轮助手正文摘要（落盘/重写时请以此为底稿）：")
        lines.append(assistant)

    lines.append("")
    return "\n".join(lines).strip()
