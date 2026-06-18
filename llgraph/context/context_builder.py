"""按 Rule / Skill 目录 + 文档索引组装上下文（目录在 session-manifest；此处仅每轮 ephemeral 提示）。"""

from __future__ import annotations

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

    from llgraph.context.context_continuity import build_continuity_context_hint

    continuity = build_continuity_context_hint(
        recent_messages,
        user_message=user_message,
        edited_paths=edited_paths,
    )
    if continuity:
        sections.append(continuity)

    manual_hint = _format_manual_skill_pin_hint(workspace, session)
    if manual_hint:
        sections.append(manual_hint)

    hint = session.write_failure_hint.strip()
    if hint:
        sections.append(hint)

    return "\n\n".join(sections)


def wrap_user_message_with_context(user_message: str, context_block: str) -> str:
    """
    将上下文块与用户消息合并。

    @param user_message 原始用户输入
    @param context_block build_workspace_context_block 返回值
    @return 发给模型的完整用户消息
    """
    if not context_block.strip():
        return user_message
    return (
        "<workspace-context>\n"
        f"{context_block.strip()}\n"
        "</workspace-context>\n\n"
        f"{user_message}"
    )


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
