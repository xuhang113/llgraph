"""从工作区 .llgraph/skills/ 与 ~/.llgraph/skills/ 加载 Skill（个人优先）。"""

import re
from dataclasses import dataclass
from pathlib import Path

from llgraph.loaders.rules_loader import _parse_frontmatter
from llgraph.session.user_storage import user_skills_dir

_SKILL_FILENAME = "SKILL.md"


def _keyword_terms(text: str) -> set[str]:
    """
    从文本提取用于自动匹配的关键词（含 2 字滑动窗，便于「整理」「文档」类短词命中）。

    @param text 原文
    @return 小写关键词集合
    """
    lowered = text.lower()
    terms: set[str] = set()
    for token in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z][\w-]{2,}", lowered):
        terms.add(token)
        if len(token) >= 4 and re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for i in range(len(token) - 1):
                terms.add(token[i : i + 2])
    return terms


@dataclass(frozen=True)
class SkillEntry:
    """单个技能定义。"""

    name: str
    skill_dir: Path
    description: str
    body: str
    scope: str = "workspace"


def _scan_skills_root(skills_root: Path, *, scope: str) -> list[SkillEntry]:
    if not skills_root.is_dir():
        return []
    entries: list[SkillEntry] = []
    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / _SKILL_FILENAME
        if not skill_file.is_file():
            continue
        try:
            raw = skill_file.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", skill_dir.name).strip() or skill_dir.name
        description = meta.get("description", name).strip()
        entries.append(
            SkillEntry(
                name=name,
                skill_dir=skill_dir,
                description=description,
                body=body.strip(),
                scope=scope,
            )
        )
    return entries


def discover_skills(workspace: Path) -> list[SkillEntry]:
    """
    扫描工作区与个人技能目录；同名技能个人优先。

    @param workspace 工作区根
    @return 技能列表
    """
    ws_root = workspace.expanduser().resolve()
    by_name: dict[str, SkillEntry] = {}
    for skill in _scan_skills_root(ws_root / ".llgraph" / "skills", scope="workspace"):
        by_name[skill.name.lower()] = skill
    for skill in _scan_skills_root(user_skills_dir(), scope="user"):
        by_name[skill.name.lower()] = skill
    return sorted(by_name.values(), key=lambda s: s.name.lower())


def resolve_skill_by_token(workspace: Path, token: str) -> SkillEntry | None:
    """
    按斜杠命令 token 查找技能（不含 /）。

    @param workspace 工作区根
    @param token 首 token，如 tracking
    @return 匹配技能；无则 None
    """
    key = token.strip().lstrip("/").lower()
    if not key:
        return None
    for skill in discover_skills(workspace):
        if skill.name.lower() == key:
            return skill
    return None


def match_skills_by_message(skills: list[SkillEntry], message: str) -> list[str]:
    """
    根据用户消息自动匹配技能名。

    匹配顺序：技能名 / 别名出现在消息中 → description 词重叠 → body 首段关键词。

    @param skills 全部技能
    @param message 用户输入
    @return 匹配到的技能名列表
    """
    if not message.strip():
        return []
    msg = message.lower()
    matched: list[str] = []
    for skill in skills:
        name = skill.name.lower()
        if name in msg or name.replace("-", "") in msg.replace("-", ""):
            matched.append(skill.name)
            continue
        desc = skill.description.lower()
        if not desc or desc in (">-", ">", "|-", "|"):
            continue
        msg_terms = _keyword_terms(message)
        desc_terms = _keyword_terms(skill.description)
        overlap = msg_terms & desc_terms
        if len(overlap) >= 2:
            matched.append(skill.name)
    return matched


def resolve_active_skills(
    skills: list[SkillEntry],
    *,
    session_active: list[str],
) -> list[SkillEntry]:
    """
    合并 /skill 手动置顶的技能（不自动匹配）。

    @param skills 全部技能
    @param session_active /skill 指定的名称
    @return 置顶技能实体
    """
    by_name = {s.name.lower(): s for s in skills}
    order: list[str] = []
    for name in session_active:
        key = name.lower()
        if key in by_name and key not in [n.lower() for n in order]:
            order.append(by_name[key].name)

    return [by_name[n.lower()] for n in order]
