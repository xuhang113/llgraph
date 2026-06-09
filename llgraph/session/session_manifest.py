"""会话锚点：Skill/Rule 目录 + manifest.json，压缩后仍保留指针。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from llgraph.config.catalog_paths import format_catalog_path, scope_label
from llgraph.context.context_session import ContextSession
from llgraph.loaders.rules_loader import discover_rules
from llgraph.loaders.skills_loader import SkillEntry, discover_skills
from llgraph.context.message_normalize import reorder_pinned_system_messages
from llgraph.session.user_storage import session_messages_path, user_sessions_root

SESSION_MANIFEST_TAG = "<session-manifest>"
_MANIFEST_VERSION = 1

_CATALOG_READ_HINT = (
    "Skills/Rules **全量目录**（描述+路径）；正文不在上下文中，需要时用 `read_file <path>` "
    "（path 可为工作区相对路径或 ~/.llgraph 下绝对路径）。"
    "⭐ 仅表示 /skill 手动置顶；不自动匹配、不注入 SKILL.md 正文。"
    "Rule 是否 alwaysApply/glob 见备注；模型自行判断 read_file 哪些规则。"
    "长文用 `start_line`/`end_line` 分段。"
    "压缩/换模型后远早对话不在每轮上下文：先看 <conversation-anchor>，"
    "细节用 `search_session_history(query=关键词)` 或 read_file archive_path；勿 cat 全量 messages.jsonl。"
)


@dataclass(frozen=True)
class CatalogEntry:
    """目录项（Skill 或 Rule）。"""

    kind: str
    id: str
    description: str
    path: str
    recommended: bool
    extra: str = ""


def _rel_workspace_path(workspace: Path, path: Path) -> str:
    """
    工作区相对路径（posix）。

    @param workspace 工作区根
    @param path 绝对或相对路径
    @return 相对路径字符串
    """
    root = workspace.expanduser().resolve()
    try:
        return path.expanduser().resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_thread_dir_name(thread_id: str) -> str:
    return thread_id.replace("/", "_").strip() or "default"


def session_manifest_dir(workspace: Path, thread_id: str) -> Path:
    """
    会话 manifest 目录（用户目录，按工作区键隔离）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return ~/.llgraph/context/<工作区名>/sessions/<thread_id>/
    """
    safe_id = _safe_thread_dir_name(thread_id)
    return user_sessions_root(workspace) / safe_id


def session_manifest_json_path(workspace: Path, thread_id: str) -> Path:
    """manifest.json 路径。"""
    return session_manifest_dir(workspace, thread_id) / "manifest.json"


def session_archive_jsonl_path(workspace: Path, thread_id: str) -> Path:
    """对话归档 jsonl 路径（与压缩导出一致）。"""
    safe_id = _safe_thread_dir_name(thread_id)
    return user_sessions_root(workspace) / f"{safe_id}.jsonl"


def conversation_anchor_json_path(workspace: Path, thread_id: str) -> Path:
    """
    结构化会话锚点 JSON 路径。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return conversation_anchor.json 路径
    """
    return session_manifest_dir(workspace, thread_id) / "conversation_anchor.json"


def is_session_manifest_message(msg: BaseMessage) -> bool:
    """
    是否为会话锚点 SystemMessage（压缩时保留）。

    @param msg LangChain 消息
    @return 是否锚点消息
    """
    if not isinstance(msg, SystemMessage):
        return False
    content = getattr(msg, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return SESSION_MANIFEST_TAG in content



def build_catalog_entries(
    workspace: Path,
    session: ContextSession,
    user_message: str,
) -> tuple[list[CatalogEntry], list[CatalogEntry]]:
    """
    构建 Skill / Rule 全量目录（对齐 Cursor：仅描述+路径，正文 read_file）。

    @param workspace 工作区根
    @param session 会话状态
    @param user_message 用户消息（用于 Rule 状态标注 glob 命中）
    @return (skill_entries, rule_entries)
    """
    all_skills = discover_skills(workspace)
    recommended = {name.strip().lower() for name in session.active_skills if name.strip()}
    skill_entries: list[CatalogEntry] = []
    for skill in all_skills:
        skill_entries.append(
            CatalogEntry(
                kind="skill",
                id=skill.name,
                description=skill.description,
                path=format_catalog_path(
                    workspace, skill.skill_dir / "SKILL.md", skill.scope
                ),
                recommended=skill.name.lower() in recommended,
                extra=scope_label(skill.scope),
            )
        )

    all_rules = discover_rules(workspace)
    rule_entries: list[CatalogEntry] = []
    for rule in all_rules:
        if rule.rule_id in session.disabled_rules:
            status = "disabled"
        elif rule.always_apply:
            status = "always"
        elif rule.rule_id in session.forced_rules:
            status = "forced"
        elif rule.globs and user_message:
            from llgraph.context.context_builder import glob_matches_message_for_rule

            status = "glob-hit" if glob_matches_message_for_rule(rule, user_message) else "glob"
        elif rule.globs:
            status = "glob"
        else:
            status = "off"
        extra_parts = [scope_label(rule.scope), status]
        if rule.globs:
            extra_parts.append(f"glob={rule.globs}")
        rule_entries.append(
            CatalogEntry(
                kind="rule",
                id=rule.rule_id,
                description=rule.description,
                path=format_catalog_path(workspace, rule.source_path, rule.scope),
                recommended=status in ("always", "forced", "glob-hit"),
                extra=";".join(extra_parts),
            )
        )
    return skill_entries, rule_entries


def _format_catalog_section(title: str, entries: list[CatalogEntry]) -> str:
    if not entries:
        return ""
    lines = [f"## {title}", ""]
    for entry in entries:
        flag = "⭐推荐 " if entry.recommended else ""
        lines.append(f"- {flag}**{entry.id}**: {entry.description}")
        lines.append(f"  - 路径: `{entry.path}`")
        if entry.extra:
            lines.append(f"  - 备注: {entry.extra}")
    lines.append("")
    return "\n".join(lines)


def build_session_manifest_payload(
    workspace: Path,
    thread_id: str,
    session: ContextSession,
    user_message: str,
    *,
    archive_path: str | None = None,
    spill_dir: str | None = None,
    anchor_path: str | None = None,
) -> dict[str, Any]:
    """
    构建可落盘的 manifest 结构。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param session 会话状态
    @param user_message 用户消息
    @param archive_path 对话归档 jsonl（相对或绝对）
    @param spill_dir 工具落盘目录（相对工作区）
    @param anchor_path 结构化锚点 json 路径
    @return JSON 可序列化 dict
    """
    skill_entries, rule_entries = build_catalog_entries(workspace, session, user_message)
    manifest_rel = _rel_workspace_path(workspace, session_manifest_json_path(workspace, thread_id))
    archive = archive_path
    if archive is None:
        archive_rel = _rel_workspace_path(workspace, session_archive_jsonl_path(workspace, thread_id))
        if session_archive_jsonl_path(workspace, thread_id).is_file():
            archive = archive_rel
    anchor = anchor_path
    if anchor is None:
        anchor_file = conversation_anchor_json_path(workspace, thread_id)
        if anchor_file.is_file():
            anchor = _rel_workspace_path(workspace, anchor_file)
    messages_file = session_messages_path(workspace, thread_id)
    messages_path: str | None = None
    if messages_file.is_file():
        messages_path = str(messages_file.resolve())
    return {
        "version": _MANIFEST_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "manifest_path": manifest_rel,
        "archive_path": archive,
        "messages_path": messages_path,
        "anchor_path": anchor,
        "spill_dir": spill_dir or ".llgraph/context/tool-results",
        "active_skills_manual": list(session.active_skills),
        "skills": [
            {
                "name": e.id,
                "description": e.description,
                "path": e.path,
                "recommended": e.recommended,
            }
            for e in skill_entries
        ],
        "rules": [
            {
                "id": e.id,
                "description": e.description,
                "path": e.path,
                "status": e.extra,
            }
            for e in rule_entries
        ],
    }


def write_session_manifest_json(
    workspace: Path,
    thread_id: str,
    payload: dict[str, Any],
) -> str | None:
    """
    写入 manifest.json。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param payload manifest 内容
    @return 相对工作区路径；失败返回 None
    """
    path = session_manifest_json_path(workspace, thread_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return _rel_workspace_path(workspace, path)
    except OSError:
        return None


def build_session_manifest_message_content(
    workspace: Path,
    thread_id: str,
    session: ContextSession,
    user_message: str,
    *,
    archive_path: str | None = None,
    spill_dir: str | None = None,
    anchor_path: str | None = None,
) -> str:
    """
    构建置顶 SystemMessage 正文（压缩时不摘要此消息）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param session 会话状态
    @param user_message 用户消息
    @param archive_path 归档路径
    @param spill_dir 工具落盘目录
    @param anchor_path 结构化锚点路径
    @return 锚点消息正文
    """
    payload = build_session_manifest_payload(
        workspace,
        thread_id,
        session,
        user_message,
        archive_path=archive_path,
        spill_dir=spill_dir,
        anchor_path=anchor_path,
    )
    manifest_rel = payload.get("manifest_path") or _rel_workspace_path(
        workspace, session_manifest_json_path(workspace, thread_id)
    )
    skill_entries, rule_entries = build_catalog_entries(workspace, session, user_message)

    parts = [
        SESSION_MANIFEST_TAG,
        f"会话: {thread_id}",
        f"manifest: `{manifest_rel}`（完整 JSON，可用 read_file 读取）",
    ]
    archive = payload.get("archive_path")
    if archive:
        parts.append(f"对话归档: `{archive}`（可用 search_session_history / read_file）")
    messages = payload.get("messages_path")
    if messages:
        parts.append(f"会话落盘: `{messages}`（search_session_history 优先；勿 cat 全文件）")
    anchor = payload.get("anchor_path")
    if anchor:
        parts.append(f"结构化锚点: `{anchor}`")
    parts.append("历史检索工具: search_session_history（按 query 搜 archive/messages/anchor 章节）")
    parts.append(f"工具结果落盘目录: `{payload.get('spill_dir')}`")
    parts.append("")
    parts.append(_CATALOG_READ_HINT)
    parts.append("")
    skills_sec = _format_catalog_section("技能目录（Skills）", skill_entries)
    if skills_sec:
        parts.append(skills_sec)
    rules_sec = _format_catalog_section("规则目录（Rules）", rule_entries)
    if rules_sec:
        parts.append(rules_sec)
    parts.append("## 文档目录\n")
    parts.append(
        "新业务梳理落盘: `docs/`（工作区 `docs/{业务域}/`、各仓 `{repo}/docs/`）；"
        "tmp 模式（不覆盖）时工作区与各仓**全部**写 `{原名}.tmp.md`，即使该仓原先无 doc。"
    )
    parts.append(
        "历史参考（只读，禁止落盘）: `markdowns/`（按需 list_directory / read_file；不内联索引正文）"
    )
    parts.append("")
    parts.append("</session-manifest>")
    return "\n".join(parts).strip()


def build_session_manifest_system_message(
    workspace: Path,
    thread_id: str,
    session: ContextSession,
    user_message: str,
    *,
    archive_path: str | None = None,
    spill_dir: str | None = None,
    anchor_path: str | None = None,
) -> SystemMessage:
    """
    构建会话锚点 SystemMessage。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param session 会话状态
    @param user_message 用户消息
    @param archive_path 归档路径
    @param spill_dir 落盘目录
    @param anchor_path 结构化锚点路径
    @return SystemMessage
    """
    content = build_session_manifest_message_content(
        workspace,
        thread_id,
        session,
        user_message,
        archive_path=archive_path,
        spill_dir=spill_dir,
        anchor_path=anchor_path,
    )
    return SystemMessage(content=content)


def strip_manifest_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """移除消息列表中的旧锚点消息。"""
    return [m for m in messages if not is_session_manifest_message(m)]


def sync_session_manifest_to_agent_state(
    agent: Any,
    *,
    thread_id: str,
    workspace: Path,
    session: ContextSession,
    user_message: str,
    with_memory: bool,
    archive_path: str | None = None,
) -> str | None:
    """
    落盘 manifest.json 并将锚点 SystemMessage 置顶到 agent 状态。

    @param agent LangGraph agent
    @param thread_id 线程 ID
    @param workspace 工作区根
    @param session Rule/Skill 会话
    @param user_message 当前用户消息
    @param with_memory 是否写入 agent 状态
    @param archive_path 最近压缩归档路径
    @return manifest 相对路径；失败 None
    """
    from llgraph.context.context_settings import resolve_context_settings

    ctx = resolve_context_settings(workspace)
    spill_dir = ctx.spill_dir
    anchor_file = conversation_anchor_json_path(workspace, thread_id)
    default_anchor = (
        _rel_workspace_path(workspace, anchor_file) if anchor_file.is_file() else None
    )
    payload = build_session_manifest_payload(
        workspace,
        thread_id,
        session,
        user_message,
        archive_path=archive_path,
        spill_dir=spill_dir,
        anchor_path=default_anchor,
    )
    manifest_rel = write_session_manifest_json(workspace, thread_id, payload)
    if not with_memory:
        return manifest_rel

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        messages = list((state.values or {}).get("messages") or [])
    except Exception:
        messages = []

    pinned = build_session_manifest_system_message(
        workspace,
        thread_id,
        session,
        user_message,
        archive_path=archive_path or payload.get("archive_path"),
        spill_dir=spill_dir,
        anchor_path=payload.get("anchor_path"),
    )
    rest = strip_manifest_messages(messages)
    new_messages = reorder_pinned_system_messages([pinned, *rest])
    try:
        agent.update_state(config, {"messages": new_messages})
    except Exception:
        return manifest_rel
    return manifest_rel


def sync_session_manifest_after_compress(
    agent: Any,
    *,
    thread_id: str,
    workspace: Path,
    session: ContextSession,
    archive_path: str | None,
    anchor_path: str | None,
) -> str | None:
    """
    压缩后刷新 manifest.json 与置顶 manifest（更新 archive/anchor 路径）。

    @param agent LangGraph agent
    @param thread_id 线程 ID
    @param workspace 工作区根
    @param session 会话状态
    @param archive_path 归档路径
    @param anchor_path 锚点路径
    @return manifest 相对路径
    """
    from llgraph.context.context_settings import resolve_context_settings

    spill_dir = resolve_context_settings(workspace).spill_dir
    payload = build_session_manifest_payload(
        workspace,
        thread_id,
        session,
        "",
        archive_path=archive_path,
        spill_dir=spill_dir,
        anchor_path=anchor_path,
    )
    manifest_rel = write_session_manifest_json(workspace, thread_id, payload)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        messages = list((state.values or {}).get("messages") or [])
    except Exception:
        return manifest_rel

    manifest_msg = build_session_manifest_system_message(
        workspace,
        thread_id,
        session,
        "",
        archive_path=archive_path or payload.get("archive_path"),
        spill_dir=spill_dir,
        anchor_path=anchor_path or payload.get("anchor_path"),
    )
    new_messages = reorder_pinned_system_messages(
        [manifest_msg, *strip_manifest_messages(messages)]
    )
    try:
        agent.update_state(config, {"messages": new_messages})
        from llgraph.session.session_file_store import save_session_messages

        save_session_messages(workspace, thread_id, new_messages)
    except Exception:
        pass
    return manifest_rel
