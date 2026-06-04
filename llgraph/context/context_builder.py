"""按 Rule / Skill 目录 + 文档索引组装每轮注入的上下文块（正文按需 read_file）。"""

from pathlib import Path

from llgraph.context.context_session import ContextSession
from llgraph.cli.markdowns_index import build_markdowns_index
from llgraph.loaders.rules_loader import (
    RuleEntry,
    discover_rules,
    glob_matches_message,
    select_rules_for_turn,
)
from llgraph.config.catalog_paths import format_catalog_path, scope_label
from llgraph.session.session_manifest import _CATALOG_READ_HINT
from llgraph.loaders.skills_loader import discover_skills, resolve_active_skills

DEFAULT_CATALOG_MAX_CHARS = 8_000
DEFAULT_TOTAL_MAX_CHARS = 12_000


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + f"\n\n…（{label} 已截断，共 {len(text)} 字符；完整目录见 <session-manifest>）"


def _format_skill_catalog(workspace: Path, session: ContextSession, user_message: str) -> str:
    """技能目录：全量列出，⭐ 标记本回合推荐。"""
    all_skills = discover_skills(workspace)
    if not all_skills:
        return ""
    recommended = {
        s.name.lower()
        for s in resolve_active_skills(
            all_skills,
            session_active=session.active_skills,
            user_message=user_message,
            auto_match=session.auto_match_skills,
        )
    }
    lines = ["## 技能目录（Skills）", "", _CATALOG_READ_HINT, ""]
    for skill in all_skills:
        flag = "⭐ " if skill.name.lower() in recommended else ""
        path = format_catalog_path(workspace, skill.skill_dir / "SKILL.md", skill.scope)
        origin = scope_label(skill.scope)
        lines.append(f"- {flag}**{skill.name}** [{origin}]: {skill.description}")
        lines.append(f"  - 路径: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _format_rule_catalog(
    workspace: Path,
    active_rules: list[RuleEntry],
    session: ContextSession,
    user_message: str,
) -> str:
    """本回合适用规则目录（仅路径+描述）。"""
    if not active_rules:
        return ""
    lines = ["## 本回合适用规则（Rules）", ""]
    for rule in active_rules:
        path = format_catalog_path(workspace, rule.source_path, rule.scope)
        origin = scope_label(rule.scope)
        lines.append(f"- **[{rule.rule_id}]** [{origin}] {rule.description}")
        lines.append(f"  - 路径: `{path}`")
        if rule.globs and not rule.always_apply:
            lines.append(f"  - glob: {rule.globs}")
        lines.append("")
    return "\n".join(lines).strip()


def build_workspace_context_block(
    workspace: Path,
    session: ContextSession,
    user_message: str,
    *,
    catalog_max_chars: int = DEFAULT_CATALOG_MAX_CHARS,
    total_max_chars: int = DEFAULT_TOTAL_MAX_CHARS,
) -> str:
    """
    构建每轮拼入用户消息前的 <workspace-context>（Skill/Rule 仅目录，不含正文）。

    @param workspace 工作区根
    @param session 会话 rule/skill 状态
    @param user_message 当前用户消息（glob 规则匹配、技能推荐）
    @return 上下文 Markdown，可为空
    """
    sections: list[str] = []

    skill_catalog = _format_skill_catalog(workspace, session, user_message)
    if skill_catalog:
        sections.append(_truncate(skill_catalog.strip(), catalog_max_chars, "Skills 目录"))

    all_rules = discover_rules(workspace)
    active_rules = select_rules_for_turn(
        all_rules,
        user_message=user_message,
        session_disabled=session.disabled_rules,
        session_forced=session.forced_rules,
    )
    rule_catalog = _format_rule_catalog(workspace, active_rules, session, user_message)
    if rule_catalog:
        sections.append(_truncate(rule_catalog, catalog_max_chars, "Rules 目录"))

    if session.include_markdowns_index:
        index = build_markdowns_index(workspace)
        if index:
            sections.append(index)

    hint = session.write_failure_hint.strip()
    if hint:
        sections.append(hint)

    if not sections:
        return ""

    body = "\n\n".join(sections)
    body = _truncate(body, total_max_chars, "workspace-context")
    return body


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
    lines.append("正文请 read_file 上述路径，不会自动注入对话。")
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
