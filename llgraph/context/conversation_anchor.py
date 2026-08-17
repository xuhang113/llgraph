"""结构化会话锚点（Tier 2）：增量合并 + 编辑账本。"""

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
SECTION_DETAIL_POINTERS = "detail_pointers"

ANCHOR_SECTION_KEYS = (
    SECTION_SESSION_GOAL,
    SECTION_FILES_MODIFIED,
    SECTION_DECISIONS,
    SECTION_ERRORS_RESOLVED,
    SECTION_DETAIL_POINTERS,
)

_SECTION_LABELS = {
    SECTION_SESSION_GOAL: "会话目标",
    SECTION_FILES_MODIFIED: "已修改文件",
    SECTION_DECISIONS: "关键决策与结论",
    SECTION_ERRORS_RESOLVED: "错误与处理",
    SECTION_DETAIL_POINTERS: "细节去哪找",
}

_SEMANTIC_SECTION_KEYS = (
    SECTION_SESSION_GOAL,
    SECTION_DECISIONS,
    SECTION_ERRORS_RESOLVED,
)

_SESSION_GOAL_MAX_CHARS = 4_000
_DECISIONS_MAX_CHARS = 12_000
_ASSISTANT_SUBSTANTIVE_MIN_CHARS = 400
_ASSISTANT_BULLET_EXTRACT_MIN_CHARS = 80
_DECISIONS_FALLBACK_BULLET_MAX = 24
_DECISIONS_FALLBACK_EXCERPT_CHARS = 600


def _system_message_text(msg: BaseMessage) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def is_conversation_anchor_message(msg: BaseMessage) -> bool:
    """
    是否为结构化会话锚点上下文消息（非 system）。

    @param msg LangChain 消息
    @return 是否锚点
    """
    return _system_message_text(msg).lstrip().startswith(CONVERSATION_ANCHOR_TAG)


def is_pinned_session_context_message(msg: BaseMessage) -> bool:
    """manifest / anchor / 旧版 summary 等会话上下文 Human 消息（非真实用户发言）。"""
    return (
        is_session_manifest_message(msg)
        or is_conversation_anchor_message(msg)
        or is_conversation_summary_message(msg)
    )


def is_conversation_summary_message(msg: BaseMessage) -> bool:
    """
    是否为旧版自由摘要消息（压缩时移除）。

    @param msg LangChain 消息
    @return 是否旧摘要
    """
    return _system_message_text(msg).lstrip().startswith(CONVERSATION_SUMMARY_TAG)


def is_pinned_session_message(msg: BaseMessage) -> bool:
    """压缩时保留的会话上下文消息（manifest 或 anchor）。"""
    return is_pinned_session_context_message(msg)


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
        "需要逐条对话、完整 tool 输出时：read_file manifest 的 archive_path（完整对话归档文件）；"
        "大工具结果见 spill_dir；可用 search_session_history 按关键词检索历史。"
        "若用户追问压缩前的调研结论/对比/细节，且锚点 decisions 不足，必须先 search_session_history，"
        "禁止凭猜测续答。"
    )
    lines.append("</conversation-anchor>")
    return "\n".join(lines).strip()


def build_conversation_anchor_message(
    workspace: Path,
    thread_id: str,
    sections: dict[str, str],
) -> HumanMessage:
    """
    构建锚点上下文消息（HumanMessage，注入历史尾段而非 system）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param sections 章节
    @return HumanMessage
    """
    path = conversation_anchor_json_path(workspace, thread_id)
    rel = _rel_workspace_path(workspace, path)
    content = format_anchor_system_message(sections, anchor_path=rel)
    return HumanMessage(content=content)


def build_conversation_anchor_system_message(
    workspace: Path,
    thread_id: str,
    sections: dict[str, str],
) -> HumanMessage:
    """兼容旧名；返回 HumanMessage。"""
    return build_conversation_anchor_message(workspace, thread_id, sections)


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
    从 edits 账本提取已修改文件（供 LLM 摘要参考；不扫对话路径避免噪音）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param messages 待压缩消息（保留参数以兼容调用方）
    @return 供 LLM 参考的硬性事实块
    """
    _ = messages
    return build_artifact_trail(workspace, thread_id)


def _messages_to_transcript(messages: list[BaseMessage]) -> str:
    """
    将消息转为摘要输入 transcript（不截断单条 content；超长 tool 应已掩码为指针）。

    @param messages 消息列表
    @return  transcript 文本
    """
    lines: list[str] = []
    for msg in messages:
        if is_session_manifest_message(msg) or is_conversation_anchor_message(msg):
            continue
        role = type(msg).__name__
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines)


def _chunk_messages_for_summary(
    messages: list[BaseMessage],
    *,
    max_chars: int,
) -> list[list[BaseMessage]]:
    """
    将待摘要消息按字符预算切成多段（保持 segment 边界）。

    @param messages 消息列表
    @param max_chars 每段 transcript 字符上限
    @return 消息段列表
    """
    from llgraph.context.context_message_split import _segment_messages

    segments = _segment_messages(messages)
    chunks: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    current_chars = 0
    for seg in segments:
        seg_text = _messages_to_transcript(seg)
        seg_chars = len(seg_text)
        if current and current_chars + seg_chars > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.extend(seg)
        current_chars += seg_chars
    if current:
        chunks.append(current)
    return chunks if chunks else [list(messages)]


def _summarize_prompt_header() -> str:
    return (
        "你是 coding agent 的会话压缩器。根据对话片段做**智能摘要**（不是机械截断或删句）。"
        "只输出一个 JSON 对象，键必须且仅能包含："
        "session_goal, files_modified, decisions, errors_resolved, detail_pointers。"
        "每个值为中文字符串；无信息则空字符串。"
        "必须保留：用户目标、已改文件路径、关键决策与结论、错误根因；禁止编造。"
        "若本段含用户消息，session_goal 不得为空，须用 1～3 句概括用户核心诉求（勿粘贴 assistant 长文）。"
        "files_modified 须为列表行，每行 `- 相对路径: 说明`（仅真实改动，勿列对话提及路径）。"
        "decisions 必填（assistant 有实质回复时不可空）：从 assistant 回复提炼关键结论、对比、定义、"
        "架构说明、方案取舍、列表要点；纯调研/问答且无代码改动时，仍须写入 5～20 条 `- ` 列表。"
        "禁止只保留 session_goal 而丢弃 assistant 正文；错误根因写入 errors_resolved。"
        "detail_pointers 可写需回查 archive/spill 的说明。"
        "禁止输出 pending_tasks、related_code 或 markdown 代码块，仅输出 JSON。"
    )


def _decisions_only_prompt_header() -> str:
    return (
        "你是 coding agent 的会话压缩器。本段 assistant 有实质回答，但 decisions 缺失或过薄。"
        "只输出一个 JSON 对象，键必须且仅能包含：decisions（中文字符串）。"
        "从 assistant 回复提炼 5～20 条 `- ` 列表：关键结论、对比、定义、方案、注意事项；禁止编造。"
        "禁止输出 markdown 代码块，仅输出 JSON。"
    )


def _assistant_visible_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content or "").strip()


def _assistant_visible_chars(messages: list[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        total += len(_assistant_visible_text(getattr(msg, "content", "")))
    return total


def _decisions_min_chars_for_span(span_messages: list[BaseMessage]) -> int:
    assistant_chars = _assistant_visible_chars(span_messages)
    if assistant_chars < _ASSISTANT_SUBSTANTIVE_MIN_CHARS:
        return 0
    return min(1_200, max(200, assistant_chars // 15))


def _decisions_needs_enrichment(delta: dict[str, str], span_messages: list[BaseMessage]) -> bool:
    min_chars = _decisions_min_chars_for_span(span_messages)
    if min_chars <= 0:
        return False
    decisions = (delta.get(SECTION_DECISIONS) or "").strip()
    return len(decisions) < min_chars


def _fallback_decisions_from_assistant(messages: list[BaseMessage]) -> str:
    """
    LLM 摘要未写出 decisions 时，从 assistant 回复机械提取要点（兜底，非替代智能摘要）。

    @param messages 待压缩段或上下文消息
    @return decisions 章节正文
    """
    bullets: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        text = _assistant_visible_text(getattr(msg, "content", ""))
        if len(text) < _ASSISTANT_BULLET_EXTRACT_MIN_CHARS:
            continue
        extracted = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("## ", "### ", "#### ")):
                item = stripped.lstrip("#").strip()
            elif re.match(r"^[-*•]\s+", stripped):
                item = re.sub(r"^[-*•]\s+", "- ", stripped)
            elif re.match(r"^\d+[.)]\s+", stripped):
                item = f"- {re.sub(r'^\d+[.)]\s+', '', stripped)}"
            else:
                continue
            item = item[:500].strip()
            if not item or item in seen:
                continue
            seen.add(item)
            bullets.append(item if item.startswith("- ") else f"- {item}")
            extracted = True
            if len(bullets) >= _DECISIONS_FALLBACK_BULLET_MAX:
                break
        if not extracted and len(text) >= _ASSISTANT_SUBSTANTIVE_MIN_CHARS:
            excerpt = " ".join(text.split())
            excerpt = excerpt[:_DECISIONS_FALLBACK_EXCERPT_CHARS].strip()
            if excerpt and excerpt not in seen:
                seen.add(excerpt)
                bullets.append(f"- （助手回复要点）{excerpt}")
        if len(bullets) >= _DECISIONS_FALLBACK_BULLET_MAX:
            break
    body = "\n".join(bullets)
    return body[:_DECISIONS_MAX_CHARS]


def _invoke_anchor_summary_llm(
    workspace: Path,
    prompt_body: str,
    *,
    model_name: str | None,
) -> dict[str, str]:
    """
    调用 LLM 生成锚点章节增量。

    @param workspace 工作区根
    @param prompt_body 完整 prompt
    @param model_name 压缩用模型
    @return 章节增量
    """
    llm = create_gateway_llm(workspace)
    if model_name:
        llm = llm.bind(model=model_name)
    response = llm.invoke([HumanMessage(content=prompt_body)])
    text = getattr(response, "content", str(response))
    if isinstance(text, list):
        text = "".join(str(x) for x in text)
    return _parse_anchor_delta_json(str(text).strip())


def _merge_anchor_delta_dicts(deltas: list[dict[str, str]]) -> dict[str, str]:
    """
    合并多段摘要增量。

    @param deltas 各段 LLM 输出
    @return 合并后的章节增量
    """
    merged = empty_anchor_sections()
    for delta in deltas:
        merged = merge_anchor_sections(merged, delta)
    return merged


def summarize_span_to_anchor_delta(
    workspace: Path,
    span_messages: list[BaseMessage],
    *,
    existing_sections: dict[str, str],
    artifact_trail: str,
    model_name: str | None,
    summary_chunk_chars: int = 120_000,
) -> dict[str, str]:
    """
    智能摘要待压缩段，输出结构化章节增量（支持分块 map-reduce，不截断单条消息）。

    @param workspace 工作区根
    @param span_messages 本轮新挤出段
    @param existing_sections 已有锚点（供 LLM 参考）
    @param artifact_trail 硬性文件清单
    @param model_name 压缩用模型
    @param summary_chunk_chars 单段 LLM 输入字符上限，超出则分块摘要后合并
    @return 章节增量 dict
    """
    existing_preview = json.dumps(existing_sections, ensure_ascii=False, indent=2)
    chunks = _chunk_messages_for_summary(span_messages, max_chars=summary_chunk_chars)
    partial_deltas: list[dict[str, str]] = []

    for idx, chunk_msgs in enumerate(chunks):
        transcript = _messages_to_transcript(chunk_msgs)
        prompt_parts = [
            _summarize_prompt_header(),
            f"（第 {idx + 1}/{len(chunks)} 段）" if len(chunks) > 1 else "",
            "",
            f"已有锚点（勿重复堆砌，仅补充新信息）：\n{existing_preview}",
        ]
        if artifact_trail.strip() and idx == 0:
            prompt_parts.append(
                f"\n硬性事实（必须写入 files_modified 或 decisions）：\n{artifact_trail}"
            )
        prompt_parts.append(f"\n本段对话：\n{transcript}")
        partial_deltas.append(
            _invoke_anchor_summary_llm(
                workspace,
                "\n".join(prompt_parts),
                model_name=model_name,
            )
        )

    if len(partial_deltas) == 1:
        delta = partial_deltas[0]
    else:
        consolidate_prompt = [
            _summarize_prompt_header(),
            "",
            "以下为多段对话分别摘要的 JSON，请合并为一份无重复、信息完整的 JSON：",
            json.dumps(partial_deltas, ensure_ascii=False, indent=2),
            "",
            f"已有锚点：\n{existing_preview}",
        ]
        delta = _invoke_anchor_summary_llm(
            workspace,
            "\n".join(consolidate_prompt),
            model_name=model_name,
        )

    if span_messages and not _delta_has_semantic_content(delta):
        retry_prompt = [
            _summarize_prompt_header(),
            "",
            "上次摘要无效（session_goal/decisions/errors_resolved 均为空）。",
            "请仅根据下列对话重新输出完整 JSON；session_goal 必须概括用户诉求。",
            f"已有锚点：\n{existing_preview}",
        ]
        if artifact_trail.strip():
            retry_prompt.append(f"\n硬性事实：\n{artifact_trail}")
        retry_prompt.append(f"\n本段对话：\n{_messages_to_transcript(span_messages)}")
        retry_delta = _invoke_anchor_summary_llm(
            workspace,
            "\n".join(retry_prompt),
            model_name=model_name,
        )
        if _delta_has_semantic_content(retry_delta):
            delta = merge_anchor_sections(delta, retry_delta)

    if _decisions_needs_enrichment(delta, span_messages):
        decisions_prompt = [
            _decisions_only_prompt_header(),
            "",
            f"已有锚点：\n{existing_preview}",
            f"\n本段对话：\n{_messages_to_transcript(span_messages)}",
        ]
        decisions_delta = _invoke_anchor_summary_llm(
            workspace,
            "\n".join(decisions_prompt),
            model_name=model_name,
        )
        decisions_text = (decisions_delta.get(SECTION_DECISIONS) or "").strip()
        if decisions_text:
            delta = merge_anchor_sections(delta, {SECTION_DECISIONS: decisions_text})

    if _decisions_needs_enrichment(delta, span_messages):
        fallback = _fallback_decisions_from_assistant(span_messages)
        if fallback.strip():
            delta = merge_anchor_sections(delta, {SECTION_DECISIONS: fallback})
    return delta


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


def _delta_has_semantic_content(delta: dict[str, str]) -> bool:
    return any((delta.get(key) or "").strip() for key in _SEMANTIC_SECTION_KEYS)


def anchor_sections_have_semantic_content(sections: dict[str, str]) -> bool:
    """锚点是否含任务级语义（非仅文件列表/指针）。"""
    return any((sections.get(key) or "").strip() for key in _SEMANTIC_SECTION_KEYS)


def _human_visible_text(content: Any) -> str:
    from llgraph.context.context_continuity import strip_workspace_context_wrapper
    from llgraph.core.user_message_content import extract_text_from_human_content

    text = extract_text_from_human_content(content)
    return strip_workspace_context_wrapper(text).strip()


def _collect_user_texts(messages: list[BaseMessage]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        text = _human_visible_text(getattr(msg, "content", ""))
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return texts


def _peek_first_user_text_from_archive(workspace: Path, thread_id: str) -> str:
    from llgraph.session.session_manifest import session_archive_jsonl_path

    path = session_archive_jsonl_path(workspace, thread_id)
    if not path.is_file():
        return ""
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                role = data.get("role") or data.get("type")
                if role not in ("user", "human"):
                    continue
                content = data.get("content", "")
                if isinstance(content, list):
                    parts = [
                        str(block.get("text", ""))
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    content = "\n".join(parts)
                text = _human_visible_text(str(content or ""))
                if text:
                    return text
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return ""
    return ""


def _sanitize_files_modified_section(text: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or "对话提及" in line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def ensure_anchor_sections_minimum(
    sections: dict[str, str],
    *,
    workspace: Path,
    thread_id: str,
    span_messages: list[BaseMessage] | None = None,
    all_messages: list[BaseMessage] | None = None,
) -> dict[str, str]:
    """
    保证锚点必有可读的语义内容（LLM 失败时从对话/edits 兜底）。

    @param sections 当前章节
    @param workspace 工作区根
    @param thread_id 会话 ID
    @param span_messages 本轮待压缩段
    @param all_messages 全量会话（注入上下文时用）
    @return 补齐后的章节
    """
    result = dict(sections)
    sources: list[BaseMessage] = []
    if span_messages:
        sources.extend(span_messages)
    if all_messages:
        sources.extend(all_messages)

    user_texts = _collect_user_texts(sources)
    if not result.get(SECTION_SESSION_GOAL, "").strip():
        goal = user_texts[0] if user_texts else ""
        if not goal:
            goal = _peek_first_user_text_from_archive(workspace, thread_id)
        if goal:
            result[SECTION_SESSION_GOAL] = goal[:_SESSION_GOAL_MAX_CHARS]

    edits = build_artifact_trail(workspace, thread_id)
    cleaned_files = _sanitize_files_modified_section(result.get(SECTION_FILES_MODIFIED, ""))
    if edits:
        result[SECTION_FILES_MODIFIED] = _merge_file_lines(cleaned_files, edits)
    else:
        result[SECTION_FILES_MODIFIED] = cleaned_files

    if not anchor_sections_have_semantic_content(result) and not result.get(
        SECTION_FILES_MODIFIED, ""
    ).strip():
        if user_texts:
            result[SECTION_SESSION_GOAL] = user_texts[0][:_SESSION_GOAL_MAX_CHARS]

    if not (result.get(SECTION_DECISIONS) or "").strip() and sources:
        fallback = _fallback_decisions_from_assistant(sources)
        if fallback.strip():
            result[SECTION_DECISIONS] = fallback

    return result


def _anchor_compression_count(workspace: Path, thread_id: str) -> int:
    path = conversation_anchor_json_path(workspace, thread_id)
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("compression_count", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def ensure_messages_include_conversation_anchor(
    workspace: Path,
    thread_id: str,
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """
    若存在压缩锚点 JSON，则将锚点上下文消息注入消息链（末条 user 之前）。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param messages 当前消息
    @return 注入后的消息
    """
    if not thread_id.strip() or not messages:
        return messages
    if _anchor_compression_count(workspace, thread_id) < 1:
        return messages

    from llgraph.context.message_normalize import reorder_pinned_session_messages

    stripped = [
        m
        for m in messages
        if not is_conversation_anchor_message(m) and not is_conversation_summary_message(m)
    ]
    loaded = load_anchor_sections(workspace, thread_id)
    enriched = ensure_anchor_sections_minimum(
        loaded,
        workspace=workspace,
        thread_id=thread_id,
        all_messages=messages,
    )
    if enriched != loaded:
        save_anchor_sections(
            workspace,
            thread_id,
            enriched,
            compression_count_delta=0,
        )
    anchor_msg = build_conversation_anchor_message(workspace, thread_id, enriched)
    return reorder_pinned_session_messages([*stripped, anchor_msg])


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
    lines.append("- 代码细节: `search_code_parallel` / `grep_files` / `read_file`")
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
    summary_chunk_chars: int = 120_000,
) -> tuple[dict[str, str], str | None]:
    """
    Tier 2：增量更新锚点并落盘。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param span_messages 新挤出段
    @param archive_path 归档路径
    @param spill_dir 落盘目录
    @param compress_model 模型
    @return (合并后 sections, anchor 文件路径)
    """
    existing = load_anchor_sections(workspace, thread_id)
    artifact = build_artifact_trail_for_compress(workspace, thread_id, span_messages)

    delta = summarize_span_to_anchor_delta(
        workspace,
        span_messages,
        existing_sections=existing,
        artifact_trail=artifact,
        model_name=compress_model,
        summary_chunk_chars=summary_chunk_chars,
    )
    if artifact.strip() and not delta.get(SECTION_FILES_MODIFIED):
        delta[SECTION_FILES_MODIFIED] = artifact

    merged = merge_anchor_sections(existing, delta)
    merged = ensure_anchor_sections_minimum(
        merged,
        workspace=workspace,
        thread_id=thread_id,
        span_messages=span_messages,
    )
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
    except (OSError, json.JSONDecodeError):
        pass
    return session
