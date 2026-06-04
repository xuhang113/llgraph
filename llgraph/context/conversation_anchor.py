"""结构化会话锚点（Tier 2）：增量合并 + 编辑账本 + 压缩检索（Tier 3）。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.session.session_manifest import (
    _rel_workspace_path,
    conversation_anchor_json_path,
    is_session_manifest_message,
    session_manifest_json_path,
)

CONVERSATION_ANCHOR_TAG = "<conversation-anchor>"
CONVERSATION_SUMMARY_TAG = "<conversation-summary>"
ANCHOR_FILENAME = "conversation_anchor.json"
_ANCHOR_VERSION = 1

SECTION_SESSION_GOAL = "session_goal"
SECTION_FILES_MODIFIED = "files_modified"
SECTION_DECISIONS = "decisions"
SECTION_ERRORS_RESOLVED = "errors_resolved"
SECTION_PENDING_TASKS = "pending_tasks"
SECTION_RELATED_CODE = "related_code"
SECTION_DETAIL_POINTERS = "detail_pointers"

ANCHOR_SECTION_KEYS = (
    SECTION_SESSION_GOAL,
    SECTION_FILES_MODIFIED,
    SECTION_DECISIONS,
    SECTION_ERRORS_RESOLVED,
    SECTION_PENDING_TASKS,
    SECTION_RELATED_CODE,
    SECTION_DETAIL_POINTERS,
)

_SECTION_LABELS = {
    SECTION_SESSION_GOAL: "会话目标",
    SECTION_FILES_MODIFIED: "已修改文件",
    SECTION_DECISIONS: "关键决策与结论",
    SECTION_ERRORS_RESOLVED: "错误与处理",
    SECTION_PENDING_TASKS: "未完成与下一步",
    SECTION_RELATED_CODE: "相关代码（检索）",
    SECTION_DETAIL_POINTERS: "细节去哪找",
}


def is_conversation_anchor_message(msg: BaseMessage) -> bool:
    """
    是否为结构化会话锚点 SystemMessage。

    @param msg LangChain 消息
    @return 是否锚点
    """
    if not isinstance(msg, SystemMessage):
        return False
    content = getattr(msg, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return CONVERSATION_ANCHOR_TAG in content


def is_conversation_summary_message(msg: BaseMessage) -> bool:
    """
    是否为旧版自由摘要消息（压缩时移除）。

    @param msg LangChain 消息
    @return 是否旧摘要
    """
    if not isinstance(msg, SystemMessage):
        return False
    content = getattr(msg, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return CONVERSATION_SUMMARY_TAG in content


def is_pinned_session_message(msg: BaseMessage) -> bool:
    """压缩时保留的置顶消息（manifest 或 anchor）。"""
    return (
        is_session_manifest_message(msg)
        or is_conversation_anchor_message(msg)
        or is_conversation_summary_message(msg)
    )


def empty_anchor_sections() -> dict[str, str]:
    """
    空锚点各章节。

    @return 章节 dict
    """
    return {key: "" for key in ANCHOR_SECTION_KEYS}


def load_anchor_sections(workspace: Path, thread_id: str) -> dict[str, str]:
    """
    读取已有锚点章节。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 章节 dict
    """
    path = conversation_anchor_json_path(workspace, thread_id)
    if not path.is_file():
        return empty_anchor_sections()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sections = data.get("sections")
        if not isinstance(sections, dict):
            return empty_anchor_sections()
        result = empty_anchor_sections()
        for key in ANCHOR_SECTION_KEYS:
            val = sections.get(key)
            if val is not None:
                result[key] = str(val).strip()
        return result
    except (OSError, json.JSONDecodeError):
        return empty_anchor_sections()


def save_anchor_sections(
    workspace: Path,
    thread_id: str,
    sections: dict[str, str],
    *,
    compression_count_delta: int = 1,
) -> str | None:
    """
    落盘 conversation_anchor.json。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param sections 章节内容
    @param compression_count_delta 本次压缩计数增量
    @return 路径字符串；失败 None
    """
    path = conversation_anchor_json_path(workspace, thread_id)
    prev_count = 0
    if path.is_file():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            prev_count = int(prev.get("compression_count", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            prev_count = 0
    payload = {
        "version": _ANCHOR_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "compression_count": prev_count + compression_count_delta,
        "sections": {key: sections.get(key, "") for key in ANCHOR_SECTION_KEYS},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
    except OSError:
        return None


def format_anchor_system_message(sections: dict[str, str], *, anchor_path: str) -> str:
    """
    构建置顶锚点 SystemMessage 正文。

    @param sections 章节
    @param anchor_path 锚点文件路径（展示用）
    @return 消息正文
    """
    lines = [
        CONVERSATION_ANCHOR_TAG,
        "结构化会话摘要（压缩后任务状态；细节用 read_file / 检索工具按需加载）",
        f"完整锚点 JSON: `{anchor_path}`",
        "",
    ]
    for key in ANCHOR_SECTION_KEYS:
        label = _SECTION_LABELS[key]
        body = sections.get(key, "").strip()
        if not body:
            continue
        lines.append(f"## {label}")
        lines.append(body)
        lines.append("")
    lines.append(
        "需要逐条对话、完整 tool 输出时：read_file manifest.json 的 archive_path；"
        "大工具结果见 spill_dir。"
    )
    lines.append("</conversation-anchor>")
    return "\n".join(lines).strip()


def build_conversation_anchor_system_message(
    workspace: Path,
    thread_id: str,
    sections: dict[str, str],
) -> SystemMessage:
    """
    构建锚点 SystemMessage。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param sections 章节
    @return SystemMessage
    """
    path = conversation_anchor_json_path(workspace, thread_id)
    rel = _rel_workspace_path(workspace, path)
    content = format_anchor_system_message(sections, anchor_path=rel)
    return SystemMessage(content=content)


def _merge_file_lines(existing: str, new_part: str) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for block in (existing, new_part):
        for line in block.splitlines():
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


def merge_anchor_sections(existing: dict[str, str], delta: dict[str, str]) -> dict[str, str]:
    """
    将本轮 LLM 增量合并进已有锚点（Factory 式 anchored merge）。

    @param existing 已有章节
    @param delta 本轮增量
    @return 合并后章节
    """
    merged = dict(existing)
    for key in ANCHOR_SECTION_KEYS:
        new_val = (delta.get(key) or "").strip()
        if not new_val:
            continue
        if key == SECTION_FILES_MODIFIED:
            merged[key] = _merge_file_lines(merged.get(key, ""), new_val)
        elif key == SECTION_DETAIL_POINTERS:
            old = merged.get(key, "").strip()
            merged[key] = f"{old}\n{new_val}".strip() if old else new_val
        else:
            old = merged.get(key, "").strip()
            merged[key] = f"{old}\n{new_val}".strip() if old else new_val
    return merged


def build_artifact_trail(workspace: Path, thread_id: str) -> str:
    """
    从本会话 edits.jsonl 提取已修改文件清单（Tier 1，非 LLM）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @return 多行文本
    """
    from llgraph.session.user_storage import (
        legacy_workspace_session_dir,
        migrate_legacy_workspace_session_dir,
        session_edits_path,
        session_thread_dir,
    )

    target = session_thread_dir(workspace, thread_id)
    migrate_legacy_workspace_session_dir(workspace, thread_id, target)
    edits_path = session_edits_path(workspace, thread_id)
    if not edits_path.is_file():
        edits_path = legacy_workspace_session_dir(workspace, thread_id) / "edits.jsonl"
    if not edits_path.is_file():
        return ""
    paths: list[str] = []
    seen: set[str] = set()
    try:
        for line in edits_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rel = str(data.get("rel_path", "")).strip()
            if not rel or rel in seen:
                continue
            seen.add(rel)
            op = str(data.get("op", "")).strip()
            paths.append(f"- `{rel}` ({op})")
    except (OSError, json.JSONDecodeError):
        return ""
    return "\n".join(paths)


def _extract_path_hints_from_messages(messages: list[BaseMessage]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"(?:[\w.-]+/)+[\w./-]+\.(?:java|py|md|mdc|json|xml|yml|yaml|ts|tsx|js|go|kt)"
        r"|[\w][\w.-]*-(?:service|api|gw|server|worker)[\w-]*",
        re.IGNORECASE,
    )
    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        for match in pattern.findall(content):
            m = match.strip()
            if m and m not in seen:
                seen.add(m)
                hints.append(f"- `{m}`（对话提及）")
    return hints


def build_artifact_trail_for_compress(
    workspace: Path,
    thread_id: str,
    messages: list[BaseMessage],
) -> str:
    """
    合并 edits 账本与对话中的路径提示。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param messages 待压缩消息
    @return 供 LLM 参考的硬性事实块
    """
    parts: list[str] = []
    edits = build_artifact_trail(workspace, thread_id)
    if edits:
        parts.append(edits)
    path_hints = _extract_path_hints_from_messages(messages)
    if path_hints:
        parts.extend(path_hints)
    return "\n".join(parts)


def extract_compress_search_query(
    messages: list[BaseMessage],
    *,
    max_terms: int = 6,
) -> str | None:
    """
    从待压缩段提取 hybrid 检索 query（Tier 3）。

    @param messages 消息列表
    @param max_terms 最多关键词数
    @return 查询串；无则 None
    """
    terms: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][\w.-]{2,}", content):
            low = token.lower()
            if low in seen or len(low) < 2:
                continue
            if low in {"你好", "谢谢", "帮忙", "一下", "这个", "那个", "怎么", "什么"}:
                continue
            seen.add(low)
            terms.append(token)
            if len(terms) >= max_terms:
                break
        if len(terms) >= max_terms:
            break
    if not terms:
        return None
    return " ".join(terms)


def retrieve_related_code_for_compress(
    workspace: Path,
    query: str,
    *,
    top_k: int = 5,
    max_chars: int = 4000,
) -> str:
    """
    压缩前用 code index 补全相关代码片段（Tier 3）。

    @param workspace 工作区根
    @param query 检索词
    @param top_k 条数
    @param max_chars 返回最大字符
    @return 格式化文本；无索引或失败返回空
    """
    try:
        from llgraph.code_index.hybrid import search_hybrid
        from llgraph.code_index.store import get_index_status

        if not get_index_status(workspace).exists:
            return ""
        text = search_hybrid(
            workspace,
            query,
            top_k=top_k,
            source="compress",
            tool="search_code_hybrid",
        )
        text = text.strip()
        if len(text) > max_chars:
            return text[: max_chars - 20] + "\n…(检索结果已截断)"
        return text
    except (RuntimeError, OSError, ValueError):
        return ""


def _messages_to_transcript(messages: list[BaseMessage], *, per_msg_limit: int = 4000) -> str:
    lines: list[str] = []
    for msg in messages:
        if is_session_manifest_message(msg) or is_conversation_anchor_message(msg):
            continue
        role = type(msg).__name__
        content = getattr(msg, "content", "")
        if isinstance(content, str) and len(content) > per_msg_limit:
            content = content[:per_msg_limit] + "\n…(截断)"
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines)


def summarize_span_to_anchor_delta(
    workspace: Path,
    span_messages: list[BaseMessage],
    *,
    existing_sections: dict[str, str],
    artifact_trail: str,
    retrieval_block: str,
    model_name: str | None,
) -> dict[str, str]:
    """
    仅摘要新挤出消息段，输出结构化章节增量（JSON）。

    @param workspace 工作区根
    @param span_messages 本轮新挤出段
    @param existing_sections 已有锚点（供 LLM 参考）
    @param artifact_trail 硬性文件清单
    @param retrieval_block Tier3 检索结果
    @param model_name 压缩用模型
    @return 章节增量 dict
    """
    transcript = _messages_to_transcript(span_messages)
    existing_preview = json.dumps(existing_sections, ensure_ascii=False, indent=2)
    if len(existing_preview) > 6000:
        existing_preview = existing_preview[:6000] + "\n…"

    prompt_parts = [
        "你是 coding agent 的会话压缩器。根据「本轮新挤出的对话片段」更新会话锚点。",
        "只输出一个 JSON 对象，键必须且仅能包含：",
        "session_goal, files_modified, decisions, errors_resolved, pending_tasks, related_code, detail_pointers",
        "每个值为中文字符串；无信息则空字符串。禁止编造。",
        "files_modified 须为列表行，每行 `- 相对路径: 说明`；可合并 artifact 中的路径。",
        "detail_pointers 可写 archive/spill 以外的补充说明。",
        "related_code 可吸收下方检索结果要点。",
        "",
        f"已有锚点（勿重复堆砌，仅补充新信息）：\n{existing_preview}",
    ]
    if artifact_trail.strip():
        prompt_parts.append(f"\n硬性事实（必须写入 files_modified 或 decisions）：\n{artifact_trail}")
    if retrieval_block.strip():
        prompt_parts.append(f"\n代码检索结果：\n{retrieval_block}")
    prompt_parts.append(f"\n本轮新挤出对话片段：\n{transcript}")

    llm = create_gateway_llm(workspace)
    if model_name:
        llm = llm.bind(model=model_name)
    response = llm.invoke([HumanMessage(content="\n".join(prompt_parts))])
    text = getattr(response, "content", str(response))
    if isinstance(text, list):
        text = "".join(str(x) for x in text)
    text = str(text).strip()
    return _parse_anchor_delta_json(text)


def _parse_anchor_delta_json(text: str) -> dict[str, str]:
    result = empty_anchor_sections()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return result
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return result
    if not isinstance(data, dict):
        return result
    for key in ANCHOR_SECTION_KEYS:
        val = data.get(key)
        if val is not None:
            result[key] = str(val).strip()
    return result


def update_detail_pointers(
    sections: dict[str, str],
    *,
    archive_path: str | None,
    spill_dir: str,
    anchor_path: str,
) -> dict[str, str]:
    """
    更新「细节去哪找」章节。

    @param sections 章节 dict
    @param archive_path 归档 jsonl
    @param spill_dir 工具落盘目录
    @param anchor_path 锚点 json 路径
    @return 更新后 sections
    """
    lines = [
        f"- 结构化锚点: `{anchor_path}`（read_file）",
        f"- 工具大结果目录: `{spill_dir}`",
    ]
    if archive_path:
        lines.append(f"- 压缩前完整对话归档: `{archive_path}`（read_file）")
    lines.append("- 代码细节: `search_code_hybrid` / `grep_files` / `read_file`")
    sections = dict(sections)
    sections[SECTION_DETAIL_POINTERS] = "\n".join(lines)
    return sections


def run_anchor_update(
    workspace: Path,
    thread_id: str,
    span_messages: list[BaseMessage],
    *,
    archive_path: str | None,
    spill_dir: str,
    compress_model: str | None,
    retrieval_enabled: bool,
    retrieval_top_k: int,
) -> tuple[dict[str, str], str | None]:
    """
    Tier 2+3：增量更新锚点并落盘。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param span_messages 新挤出段
    @param archive_path 归档路径
    @param spill_dir 落盘目录
    @param compress_model 模型
    @param retrieval_enabled 是否 Tier3 检索
    @param retrieval_top_k 检索条数
    @return (合并后 sections, anchor 文件路径)
    """
    existing = load_anchor_sections(workspace, thread_id)
    artifact = build_artifact_trail_for_compress(workspace, thread_id, span_messages)
    retrieval_block = ""
    if retrieval_enabled:
        query = extract_compress_search_query(span_messages)
        if query:
            retrieval_block = retrieve_related_code_for_compress(
                workspace,
                query,
                top_k=retrieval_top_k,
            )

    delta = summarize_span_to_anchor_delta(
        workspace,
        span_messages,
        existing_sections=existing,
        artifact_trail=artifact,
        retrieval_block=retrieval_block,
        model_name=compress_model,
    )
    if artifact.strip() and not delta.get(SECTION_FILES_MODIFIED):
        delta[SECTION_FILES_MODIFIED] = artifact

    if retrieval_block.strip():
        old_rc = delta.get(SECTION_RELATED_CODE, "").strip()
        delta[SECTION_RELATED_CODE] = (
            f"{old_rc}\n{retrieval_block}".strip() if old_rc else retrieval_block
        )

    merged = merge_anchor_sections(existing, delta)
    anchor_file = conversation_anchor_json_path(workspace, thread_id)
    rel_anchor = _rel_workspace_path(workspace, anchor_file)
    merged = update_detail_pointers(
        merged,
        archive_path=archive_path,
        spill_dir=spill_dir,
        anchor_path=rel_anchor,
    )
    saved = save_anchor_sections(workspace, thread_id, merged)
    return merged, saved


def load_session_from_manifest(workspace: Path, thread_id: str):
    """
    从 manifest.json 恢复 ContextSession（压缩时重建 manifest 用）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @return ContextSession
    """
    from llgraph.context.context_session import ContextSession

    path = session_manifest_json_path(workspace, thread_id)
    session = ContextSession()
    if not path.is_file():
        return session
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        manual = data.get("active_skills_manual")
        if isinstance(manual, list):
            session.active_skills = [str(x) for x in manual]
        auto = data.get("auto_match_skills")
        if auto is not None:
            if isinstance(auto, str):
                session.auto_match_skills = auto.strip().lower() not in (
                    "0",
                    "false",
                    "no",
                )
            else:
                session.auto_match_skills = bool(auto)
    except (OSError, json.JSONDecodeError):
        pass
    return session
