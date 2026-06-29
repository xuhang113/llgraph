"""从 Agent 会话加载上下文，供 Plan Planner 制定计划。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.session.session_file_store import load_session_messages

_TOOL_PREVIEW_CHARS = 400
_ASSISTANT_PREVIEW_CHARS = 2000


def _message_role(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, ToolMessage):
        return f"tool:{msg.name or 'tool'}"
    role = getattr(msg, "type", None) or getattr(msg, "role", "")
    return str(role or "unknown")


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "thinking":
                    continue
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content).strip()


def _clip(text: str, limit: int) -> str:
    raw = (text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 1)] + "…"


def format_messages_as_plan_context(
    messages: list[BaseMessage],
    *,
    max_chars: int = 12000,
    max_messages: int = 48,
) -> str:
    """
    将 Agent 消息转为 Planner 可读的对话摘录（取最近若干条）。

    @param messages LangChain 消息列表
    @param max_chars 总字符上限
    @param max_messages 最多条数（从尾部取）
    @return 格式化文本；无内容返回空串
    """
    if not messages:
        return ""

    tail = messages[-max_messages:] if len(messages) > max_messages else messages
    lines: list[str] = []
    used = 0

    for msg in tail:
        role = _message_role(msg)
        if role in ("system", "unknown") and not _content_text(getattr(msg, "content", "")):
            continue
        text = _content_text(getattr(msg, "content", ""))
        if isinstance(msg, ToolMessage):
            text = _clip(text, _TOOL_PREVIEW_CHARS)
        elif isinstance(msg, AIMessage):
            text = _clip(text, _ASSISTANT_PREVIEW_CHARS)
        if not text and role.startswith("tool"):
            continue
        if not text and role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                names = [
                    str(c.get("name", "?"))
                    for c in tool_calls
                    if isinstance(c, dict)
                ]
                text = f"（调用工具: {', '.join(names[:6])}）"
        if not text:
            continue
        line = f"[{role}] {text}"
        if used + len(line) + 1 > max_chars:
            remain = max_chars - used - 20
            if remain > 80:
                lines.append(_clip(line, remain))
            lines.append("…（Agent 对话已截断）")
            break
        lines.append(line)
        used += len(line) + 1

    return "\n".join(lines).strip()


def load_agent_context_for_plan(
    workspace,
    agent_thread_id: str,
    *,
    max_chars: int = 12000,
    max_messages: int = 48,
) -> str:
    """
    从 Agent cli 会话 messages.jsonl 加载 Planner 上下文。

    @param workspace 工作区根
    @param agent_thread_id 来源 Agent thread_id（cli-*）
    @param max_chars 总字符上限
    @param max_messages 最多消息条数
    @return 对话摘录；无文件或空则 ""
    """
    tid = (agent_thread_id or "").strip()
    if not tid or not tid.startswith("cli-"):
        return ""
    from pathlib import Path

    root = Path(workspace).expanduser().resolve()
    messages = load_session_messages(root, tid)
    return format_messages_as_plan_context(
        messages,
        max_chars=max_chars,
        max_messages=max_messages,
    )


def format_plan_for_revision_prompt(plan: dict[str, Any]) -> str:
    """
    将当前 plan.json 摘要序列化，供修订时作为基线。

    @param plan plan dict
    @return JSON 文本；无 tasks 时返回空串
    """
    if not isinstance(plan, dict):
        return ""
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ""
    import json

    payload = {
        "title": str(plan.get("title") or ""),
        "goal": str(plan.get("goal") or ""),
        "tasks": tasks,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_planner_user_prompt(
    *,
    opening_goal: str,
    agent_context: str = "",
    revision_note: str = "",
    plan_version: int = 1,
    current_plan: dict[str, Any] | None = None,
) -> str:
    """
    组装 Planner 用户提示（含可选 Agent 会话摘录）。

    @param opening_goal 计划目标
    @param agent_context 来源 Agent 对话摘录
    @param revision_note 修订说明
    @param plan_version 计划版本
    @param current_plan 修订前的 plan.json（含 tasks）
    @return 完整 user prompt
    """
    blocks: list[str] = []
    ctx = (agent_context or "").strip()
    if ctx:
        blocks.append("--- 来源 Agent 会话（进入 Plan 前的讨论，制定计划时须参考）---")
        blocks.append(ctx)
        blocks.append("")

    goal = (opening_goal or "").strip()
    revision = (revision_note or "").strip()
    if revision:
        blocks.append(f"请根据修订说明更新计划（v{plan_version}）：\n{revision}")
        plan_snapshot = format_plan_for_revision_prompt(current_plan or {})
        if plan_snapshot:
            blocks.append("\n--- 当前计划（须在此基础上修改，勿清空已有 Work）---")
            blocks.append(plan_snapshot)
        blocks.append(
            "\n说明：最终报告由系统「汇总(Synthesize)」节点自动生成，不要把它写成 Work 任务。"
            "回复末尾必须输出完整 ```json plan 代码块（含 title 与 tasks 数组）。"
        )
        if goal:
            blocks.append(f"\n原目标：{goal}")
    elif goal:
        blocks.append(f"请为以下需求制定多步骤计划：\n{goal}")
    else:
        blocks.append(
            "请根据上文 Agent 会话中已讨论的需求，制定多步骤可执行计划。"
        )
    return "\n".join(blocks)
