"""按 Rule / Skill 目录 + 文档索引组装上下文（目录在 session-manifest；此处仅每轮 ephemeral 提示）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from llgraph.context.context_session import ContextSession
from llgraph.config.catalog_paths import format_catalog_path, scope_label
from llgraph.loaders.rules_loader import (
    RuleEntry,
    discover_rules,
    glob_matches_message,
    select_rules_for_turn,
)
from llgraph.loaders.skills_loader import discover_skills

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _format_manual_skill_pin_hint(workspace: Path, session: ContextSession) -> str:
    """
    /skill 手动启用的技能：仅路径指针，不注入 SKILL.md 正文。

    @param workspace 工作区根
    @param session 会话 skill 状态
    @return Markdown 块；无手动启用时空串
    """
    if not session.active_skills:
        return ""
    by_name = {s.name.lower(): s for s in discover_skills(workspace)}
    lines = ["## 本会话已启用技能（/skill）", ""]
    for name in session.active_skills:
        skill = by_name.get(name.strip().lower())
        if skill is None:
            lines.append(f"- **{name}**（未找到目录，请 /skill list）")
            continue
        path = format_catalog_path(workspace, skill.skill_dir / "SKILL.md", skill.scope)
        origin = scope_label(skill.scope)
        lines.append(f"- **{skill.name}** [{origin}]: {skill.description}")
        lines.append(f"  - 路径: `{path}`（需要时用 read_file 读取正文）")
    lines.append("")
    return "\n".join(lines).strip()


# 用户在追问「压缩前 / 更早轮次」的细节、结论、对比时的口吻
_HISTORY_RECALL_RE = re.compile(
    r"(之前|先前|早些|早前|前面|前文|上一轮|上次|上回|刚才|刚刚|前几轮|更早|历史里|压缩前|"
    r"你(说过|讲过|提到过|分析过|查过|得出)|当时(说|的结论)|earlier|previously|before you)",
)

# 广搜/摸底类：近端强调 spawn（与针点并行搜读区分）
_SPAWN_RESEARCH_RE = re.compile(
    r"(整理|梳理|摸底|调研|全貌|概览|链路|流程|业务|架构|调用链|数据流|端到端|"
    r"怎么串|如何串|模块.*关系|仓库.*结构|代码库.*结构|"
    r"overview|map\b|explore|end-?to-?end|call\s*chain|architecture)",
    re.IGNORECASE,
)


def format_spawn_research_hint(user_message: str) -> str:
    """
    广搜/业务链路类任务的近端调研路由（注入 workspace-context）。

    @param user_message 当前用户消息
    @return Markdown；非广搜口吻时返回空串
    """
    text = (user_message or "").strip()
    if not text or not _SPAWN_RESEARCH_RE.search(text):
        return ""
    return (
        "## 调研路由（本轮优先）\n"
        "该问偏**广搜/摸底/链路整理**：**先**调用 `spawn_subagent`"
        "（`kind=\"explore\"`，`prompt` 写清范围与要回答的问题）。\n"
        "主会话本回合**不要**自己连开 `search_code_parallel` + 大批量 `read_files` 摸底；"
        "等子 Agent 摘要返回后再精读或作答。\n"
        "针点查询（已知文件/符号）才用主会话并行 grep/read。"
    )


def format_history_recall_hint(
    user_message: str,
    recent_messages: list[BaseMessage] | None,
) -> str:
    """
    压缩后追问压缩前细节时的回捞提示（不加轮、不重跑，只提醒先检索）。

    仅当上下文里已存在会话锚点（说明发生过压缩），且用户在追问更早轮次的
    细节/结论时才注入，避免模型凭锚点摘要或指针预览臆答。

    @param user_message 当前用户消息
    @param recent_messages 当前会话历史消息
    @return Markdown 提示；不满足条件时空串
    """
    if not user_message or not recent_messages:
        return ""
    if not _HISTORY_RECALL_RE.search(user_message):
        return ""
    from llgraph.context.conversation_anchor import is_conversation_anchor_message

    if not any(is_conversation_anchor_message(m) for m in recent_messages):
        return ""
    return (
        "## 压缩历史回捞\n"
        "本会话较早内容已被压缩为锚点/指针，细节可能不全。"
        "用户在追问更早轮次的结论/证据/对比时，**先** `search_session_history` "
        "或按指针 `read_file` 归档，再作答；**禁止**仅凭锚点摘要或预览臆断。"
        "确认锚点未覆盖该细节前，不要说「之前没有/未提及」。"
    )


def build_workspace_context_block(
    workspace: Path,
    session: ContextSession,
    user_message: str,
    *,
    allow_write: bool = False,
    recent_messages: list[BaseMessage] | None = None,
    edited_paths: list[str] | None = None,
) -> str:
    """
    构建每轮拼入用户消息前的 <workspace-context>（仅 ephemeral 提示）。

    Skills/Rules 全量目录与简介在置顶 <session-manifest>；模型按需 read_file。
    续写/重写时注入会话连续性提示，减少重复侦察。

    @param workspace 工作区根
    @param session 会话 rule/skill 状态
    @param user_message 当前用户消息
    @param allow_write 是否可写
    @param recent_messages 当前会话消息（用于续写 pin）
    @param edited_paths 本会话已改文件路径
    @return 上下文 Markdown，可为空
    """
    from llgraph.session.session_write_mode import format_file_access_workspace_context

    sections: list[str] = [format_file_access_workspace_context(allow_write)]

    sections.append(
        "## 本轮目标\n"
        "只服务下方 `<user_query>`。"
        "意图与排查/归因策略由你自行判断；框架不替你分流。"
        "正确性优先：顺着调用/数据上下文核证，禁止断章取义；"
        "有证据后结论或动作先行；因果断言须有源码依据，勿只靠注释。"
    )

    spawn_hint = format_spawn_research_hint(user_message)
    if spawn_hint:
        sections.append(spawn_hint)

    sections.append(
        "## 工具并行（针点任务）\n"
        "已知范围、无需广搜摸底时："
        "无依赖的 grep/read/list/glob 必须在**同一条** assistant 消息里并行发出；"
        "先在思考里列齐「要凑齐答案还缺什么」，再一次性发出工具调用。"
        "多词 → 一条 `grep_files(pattern=\"a|b|c\")`；已有字面量时**不要**先 `search_code_parallel`。\n"
        "广搜/多模块/业务链路整理 → 用上文 `spawn_subagent`，不要用主会话并行搜读代替。"
    )

    from llgraph.core.react_limits import format_tool_round_budget_line

    sections.append(format_tool_round_budget_line(recent_messages or [], workspace=workspace))

    from llgraph.context.context_continuity import build_continuity_context_hint

    continuity = build_continuity_context_hint(
        recent_messages,
        user_message=user_message,
        edited_paths=edited_paths,
    )
    if continuity:
        sections.append(continuity)

    recall_hint = format_history_recall_hint(user_message, recent_messages)
    if recall_hint:
        sections.append(recall_hint)

    manual_hint = _format_manual_skill_pin_hint(workspace, session)
    if manual_hint:
        sections.append(manual_hint)

    hint = session.write_failure_hint.strip()
    if hint:
        sections.append(hint)

    from llgraph.core.search_path_tracker import format_retrieval_batch_hint

    batch_hint = format_retrieval_batch_hint(user_message)
    if batch_hint:
        sections.append(batch_hint)

    return "\n\n".join(sections)


def wrap_user_message_with_context(user_message: str, context_block: str) -> str:
    """
    将上下文块与用户消息合并；用户正文放入 `<user_query>`，便于全程钉目标。

    @param user_message 原始用户输入
    @param context_block build_workspace_context_block 返回值
    @return 发给模型的完整用户消息
    """
    query = (user_message or "").strip()
    query_block = f"<user_query>\n{query}\n</user_query>" if query else (user_message or "")
    ctx = (context_block or "").strip()
    if not ctx:
        return query_block
    return f"<workspace-context>\n{ctx}\n</workspace-context>\n\n{query_block}"


def format_rules_list(workspace: Path, session: ContextSession, user_message: str) -> str:
    """
    生成 /rule 命令用的规则列表文本。

    @param workspace 工作区根
    @param session 当前会话
    @param user_message 用于显示 glob 是否命中（可为空）
    @return 多行说明
    """
    rules = discover_rules(workspace)
    if not rules:
        return (
            "未找到 llgraph 规则。请执行:\n"
            "  llgraph --init-config -C <工作区>   → .llgraph/rules/\n"
            "  llgraph --init-user-config         → ~/.llgraph/rules/\n"
            "（不读取 .cursorrules；同名文件个人优先）"
        )

    lines = ["规则列表（项目 + 个人，同名个人优先）:", ""]
    for rule in rules:
        status: list[str] = []
        if rule.always_apply:
            status.append("always")
        if rule.rule_id in session.disabled_rules:
            status.append("已禁用")
        elif rule.rule_id in session.forced_rules:
            status.append("强制启用")
        elif rule.globs and user_message and glob_matches_message_for_rule(rule, user_message):
            status.append("本句命中")
        elif rule.globs:
            status.append(f"glob:{rule.globs}")
        tag = ", ".join(status) if status else "未启用"
        path = format_catalog_path(workspace, rule.source_path, rule.scope)
        origin = scope_label(rule.scope)
        lines.append(f"  [{rule.rule_id}] [{origin}] {rule.description}  ({tag})")
        lines.append(f"      路径: {path}")
    lines.append("")
    lines.append("命令: /rule list | /rule on <id> | /rule off <id> | /rule reset")
    lines.append("全量规则目录见 <session-manifest>；正文 read_file，不会自动注入对话。")
    return "\n".join(lines)


def glob_matches_message_for_rule(rule: RuleEntry, message: str) -> bool:
    """判断 glob 规则是否命中消息。"""
    if not rule.globs:
        return False
    import re

    for pattern in re.split(r"[,，\s]+", rule.globs):
        pattern = pattern.strip()
        if pattern and glob_matches_message(pattern, message):
            return True
    return False
