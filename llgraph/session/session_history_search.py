"""会话全历史检索：归档 jsonl、messages.jsonl、结构化锚点（按需注入，省 API token）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import messages_from_dict

from llgraph.context.conversation_anchor import ANCHOR_SECTION_KEYS, load_anchor_sections
from llgraph.cli.search_terms import build_search_terms
from llgraph.session.session_manifest import (
    conversation_anchor_json_path,
    session_archive_jsonl_path,
    session_manifest_json_path,
)
from llgraph.session.user_storage import session_messages_path

_EXCERPT_MAX = 900
_TOOL_EXCERPT_MAX = 500
_SHELL_QUERY_RE = re.compile(
    r"(shell|find\b|wc\b|git\b|command|run_shell|工具|命令|历史|刚才|之前|archive|messages\.jsonl)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HistoryHit:
    """单条历史命中。"""

    source: str
    line_no: int | None
    role: str
    score: int
    excerpt: str
    source_path: str | None = None


def _query_wants_tool_results(query: str) -> bool:
    """
    检索 shell/命令/历史类问题时，默认纳入 tool 长输出与 tool_calls。

    @param query 检索问句
    @return 是否扩展 tool 命中
    """
    return bool(_SHELL_QUERY_RE.search(query or ""))


def _append_tool_calls_text(parts: list[str], tool_calls: object) -> None:
    """
    将 tool_calls 参数拼入可检索正文（shell 命令常在 AI 的 tool_calls 而非 tool 结果里）。

    @param parts 正文片段列表
    @param tool_calls 消息中的 tool_calls 字段
    """
    if not isinstance(tool_calls, list):
        return
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not name and isinstance(call.get("function"), dict):
            name = call["function"].get("name")
        args = call.get("args")
        if args is None and isinstance(call.get("function"), dict):
            args = call["function"].get("arguments")
        if name:
            parts.append(f"tool_call:{name}")
        if args is not None:
            parts.append(str(args))


def _resolve_path(workspace: Path, raw: str | None) -> Path | None:
    if not raw or not str(raw).strip():
        return None
    p = Path(str(raw).strip()).expanduser()
    if p.is_absolute():
        return p if p.is_file() else None
    candidate = (workspace / p).resolve()
    return candidate if candidate.is_file() else None


def _load_manifest_archive(workspace: Path, thread_id: str) -> Path | None:
    manifest = session_manifest_json_path(workspace, thread_id)
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _resolve_path(workspace, data.get("archive_path"))


def _history_source_paths(workspace: Path, thread_id: str) -> list[tuple[str, Path]]:
    """
    按优先级列出可检索的历史文件。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @return (来源标签, 路径) 列表
    """
    seen: set[Path] = set()
    ordered: list[tuple[str, Path]] = []

    def add(label: str, path: Path | None) -> None:
        if path is None or not path.is_file():
            return
        key = path.resolve()
        if key in seen:
            return
        seen.add(key)
        ordered.append((label, path))

    add("archive", _load_manifest_archive(workspace, thread_id))
    add("archive", session_archive_jsonl_path(workspace, thread_id))
    add("messages", session_messages_path(workspace, thread_id))
    anchor = conversation_anchor_json_path(workspace, thread_id)
    add("anchor_json", anchor)
    return ordered


def _message_text_from_record(record: dict[str, Any]) -> tuple[str, str]:
    """
    从归档行或 LangChain dict 提取 role 与正文。

    @param record JSON 对象
    @return (role, text)
    """
    if "role" in record:
        role = str(record.get("role") or "unknown")
        content = record.get("content", "")
        parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
        elif content:
            parts.append(str(content))
        _append_tool_calls_text(parts, record.get("tool_calls"))
        return role, "\n".join(parts).strip()

    if "type" in record and "data" in record:
        data = record.get("data")
        if isinstance(data, dict):
            role = str(record.get("type") or "unknown")
            parts = []
            content = data.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            kwargs = data.get("additional_kwargs")
            if isinstance(kwargs, dict):
                llgraph = kwargs.get("llgraph")
                if isinstance(llgraph, dict):
                    thinking = llgraph.get("thinking_text")
                    if isinstance(thinking, str) and thinking.strip():
                        parts.append(thinking.strip())
            _append_tool_calls_text(parts, data.get("tool_calls"))
            meta = data.get("response_metadata")
            if isinstance(meta, dict):
                model_name = meta.get("model_name")
                if model_name:
                    parts.append(f"model={model_name}")
            if parts:
                return role, "\n".join(parts)
        try:
            msgs = messages_from_dict([record])
            if msgs:
                m = msgs[0]
                role = getattr(m, "type", "unknown")
                content = getattr(m, "content", "")
                parts = []
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
                _append_tool_calls_text(parts, getattr(m, "tool_calls", None))
                if parts:
                    return str(role), "\n".join(parts)
                return str(role), str(content)
        except Exception:
            pass
    return "unknown", str(record)


def _score_text(text: str, terms: list[str]) -> int:
    if not text.strip() or not terms:
        return 0
    lower = text.lower()
    score = 0
    for term in terms:
        key = term.lower()
        if len(key) < 2:
            continue
        if key in lower:
            score += 2
            score += min(3, len(key) // 4)
    return score


def _excerpt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n…（已截断，完整内容见 read_file 对应归档行）"


def _iter_archive_lines(path: Path) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                role, text = _message_text_from_record(record)
                if text:
                    rows.append((line_no, role, text))
    except OSError:
        return []
    return rows


def _iter_messages_jsonl(path: Path) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, list):
                    for item in record:
                        if isinstance(item, dict):
                            role, text = _message_text_from_record(item)
                            if text:
                                rows.append((line_no, role, text))
                    continue
                if isinstance(record, dict):
                    role, text = _message_text_from_record(record)
                    if text:
                        rows.append((line_no, role, text))
    except OSError:
        return []
    return rows


def _search_anchor_sections(
    workspace: Path,
    thread_id: str,
    terms: list[str],
) -> list[HistoryHit]:
    sections = load_anchor_sections(workspace, thread_id)
    hits: list[HistoryHit] = []
    for key in ANCHOR_SECTION_KEYS:
        body = (sections.get(key) or "").strip()
        if not body:
            continue
        score = _score_text(body, terms)
        if score <= 0 and terms:
            continue
        if not terms:
            score = 1
        label = key
        hits.append(
            HistoryHit(
                source="anchor_json",
                line_no=None,
                role=f"anchor:{label}",
                score=score,
                excerpt=_excerpt(body, _EXCERPT_MAX),
            )
        )
    return hits


def search_session_history(
    workspace: Path,
    thread_id: str,
    query: str,
    *,
    top_k: int = 8,
    include_tool_results: bool = False,
) -> str:
    """
    按查询词在会话归档/落盘消息/锚点章节中检索相关片段。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param query 检索问句或关键词（中英文均可）
    @param top_k 返回条数上限
    @param include_tool_results 是否包含 tool 角色长输出
    @return 格式化命中列表；无命中时返回说明
    """
    query = (query or "").strip()
    if not query:
        return "search_session_history: query 不能为空。"

    terms = build_search_terms(topic=query, keywords=query)
    if not terms:
        terms = [query]

    if not include_tool_results and _query_wants_tool_results(query):
        include_tool_results = True

    sources = _history_source_paths(workspace, thread_id)
    if not sources:
        return (
            f"会话 {thread_id} 尚无落盘历史（无 archive/messages/anchor）。"
            "请先进行有记忆的对话或执行压缩生成归档。"
        )

    hits: list[HistoryHit] = []
    hits.extend(_search_anchor_sections(workspace, thread_id, terms))

    for label, path in sources:
        if label == "anchor_json":
            continue
        if label == "archive":
            rows = _iter_archive_lines(path)
        else:
            rows = _iter_messages_jsonl(path)

        for line_no, role, text in rows:
            if role == "tool" and not include_tool_results:
                if len(text) > 400 and _score_text(text, terms) < 3:
                    continue
            score = _score_text(text, terms)
            if score <= 0:
                continue
            cap = _TOOL_EXCERPT_MAX if role == "tool" else _EXCERPT_MAX
            if role in ("ai", "assistant") and _query_wants_tool_results(query):
                cap = max(cap, 1400)
            hits.append(
                HistoryHit(
                    source=label,
                    line_no=line_no,
                    role=role,
                    score=score,
                    excerpt=_excerpt(text, cap),
                    source_path=str(path.resolve()),
                )
            )

    if not hits:
        paths_hint = ", ".join(f"{lbl}={p.name}" for lbl, p in sources)
        return (
            f"未在会话 {thread_id} 历史中找到与「{query}」相关的片段（检索词: {', '.join(terms[:12])}）。\n"
            f"已扫描: {paths_hint}\n"
            "建议：换更具体的关键词；或 read_file 置顶 <conversation-anchor> / manifest 中的 archive_path 全文。"
        )

    hits.sort(key=lambda h: (-h.score, h.source, h.line_no or 0))
    selected = hits[: max(1, min(20, top_k))]

    lines = [
        f"会话历史检索 thread={thread_id} query={query!r} 命中 {len(selected)}/{len(hits)} 条",
        "说明：以下为按需片段（已含 tool_calls/ tool 输出时自动扩展）。",
        "若仍不足：用各命中下方的 read_file 行段精读；禁止 search_files/cat 整文件 messages.jsonl。",
        "",
    ]
    for idx, hit in enumerate(selected, start=1):
        loc = f"{hit.source}"
        if hit.line_no is not None:
            loc += f":{hit.line_no}"
        lines.append(f"--- 命中 {idx} [{loc}] {hit.role} (score={hit.score}) ---")
        lines.append(hit.excerpt)
        if hit.source_path and hit.line_no is not None:
            lines.append(
                f"read_file: `{hit.source_path}` start_line={hit.line_no} end_line={hit.line_no}"
            )
        lines.append("")

    messages_path = session_messages_path(workspace, thread_id)
    if messages_path.is_file():
        lines.append(f"messages: {messages_path}")
    manifest_rel = session_manifest_json_path(workspace, thread_id)
    if manifest_rel.is_file():
        lines.append(f"manifest: {manifest_rel}")
    return "\n".join(lines).strip()
