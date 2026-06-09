"""Hybrid 检索：路径/文件名 + 内容 grep + 向量，RRF 融合。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llgraph.code_index.path_hits import path_hits
from llgraph.code_index.search_path_filter import is_junk_search_path
from llgraph.code_index.vector_search import perform_vector_search
from llgraph.config.logging_settings import get_search_logger
from llgraph.core.workspace import WorkspaceContext

RRF_K = 60
GREP_TOP = 50
VECTOR_TOP = 50
PATH_TOP = 40
# 与 filesystem_tools.read_files 单次上限对齐
_HYBRID_SUGGEST_READ_FILES = 8


def _unique_file_paths(
    ranked: list[tuple[str, float]],
    *,
    limit: int = _HYBRID_SUGGEST_READ_FILES,
) -> list[str]:
    """
    从 hybrid doc_id（rel_path:line）提取去重文件路径，供 read_files 建议。

    @param ranked 排序后的 (doc_id, score) 列表
    @param limit 最多条数
    @return 相对工作区路径列表
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc_id, _score in ranked:
        rel = doc_id.split(":", 1)[0]
        if not rel or rel in seen:
            continue
        seen.add(rel)
        out.append(rel)
        if len(out) >= limit:
            break
    return out


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _grep_hits(
    ctx: WorkspaceContext,
    query: str,
    path_prefix: str,
    *,
    limit: int = GREP_TOP,
) -> list[tuple[str, str]]:
    """
    稀疏检索：提取 query 中的标识符，优先 ripgrep 全工作区 grep。

    @return (doc_id, snippet) 列表，doc_id 为 rel_path:start_line
    """
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z_][\w.-]{2,}", query)
    if not tokens:
        tokens = [query.strip()[:40]]
    tokens = tokens[:8]

    from llgraph.core.ripgrep_search import ripgrep_available, ripgrep_content

    if ripgrep_available():
        results: list[tuple[str, str]] = []
        seen: set[str] = set()
        per_token = max(8, limit // max(1, len(tokens)))
        for token in tokens:
            if not token.strip():
                continue
            hits, _err = ripgrep_content(
                ctx.root,
                token,
                path_prefix=path_prefix or ".",
                limit=per_token,
                skip_dirs=ctx._extra_skip_dirs,
            )
            for line in hits:
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                rel, line_no, snippet_body = parts[0], parts[1], parts[2]
                doc_id = f"{rel}:{line_no}"
                if doc_id in seen:
                    continue
                if is_junk_search_path(rel):
                    continue
                seen.add(doc_id)
                snippet = snippet_body.strip()[:160]
                results.append((doc_id, snippet))
                if len(results) >= limit:
                    return results
        return results

    results = []
    seen: set[str] = set()
    for token in tokens:
        try:
            pattern = re.compile(re.escape(token), re.IGNORECASE)
        except re.error:
            continue
        for rel in ctx.iter_files(path_prefix or "."):
            full = ctx.resolve_path(rel)
            if not full.is_file():
                continue
            try:
                if full.stat().st_size > 500_000:
                    continue
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    if is_junk_search_path(rel):
                        continue
                    doc_id = f"{rel}:{line_no}"
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    snippet = line.strip()[:160]
                    results.append((doc_id, snippet))
                    if len(results) >= limit:
                        return results
    return results


def _vector_hit_ids(
    workspace: Path,
    query: str,
    path_prefix: str,
    top_k: int,
    *,
    source: str = "cli",
    tool: str = "search_code_hybrid",
) -> tuple[list[tuple[str, str]], str]:
    """
    向量路：返回 (doc_id, preview) 与错误信息。

    @param workspace 工作区根
    @param query 查询
    @param path_prefix 路径前缀
    @param top_k 条数
    @param source 调用来源
    @param tool 工具名
    @return (命中列表, 错误文案；无错为空串)
    """
    logger = get_search_logger()
    try:
        outcome = perform_vector_search(
            workspace,
            query,
            top_k=top_k,
            path_prefix=path_prefix,
            source=source,
            tool=tool,
        )
    except Exception as exc:
        logger.error(
            "[vector] hybrid 向量路失败 source=%s tool=%s error=%s",
            source,
            tool,
            exc,
        )
        return [], str(exc)

    if outcome.skipped:
        return [], "索引为空"

    out: list[tuple[str, str]] = []
    for hit in outcome.hits:
        rel = hit.get("rel_path", "")
        if is_junk_search_path(str(rel)):
            continue
        start = hit.get("start_line", 0)
        doc_id = f"{rel}:{start}"
        preview = hit.get("text_preview", "")
        out.append((doc_id, preview))
    return out, ""


def search_hybrid(
    workspace: Path,
    query: str,
    *,
    top_k: int = 10,
    path_prefix: str = ".",
    source: str = "cli",
    tool: str = "search_code_hybrid",
) -> str:
    """
    RRF 融合路径/文件名、内容 grep 与向量检索（含原 search_files 路径能力）。

    @param workspace 工作区根
    @param query 查询
    @param top_k 最终条数
    @param path_prefix 路径前缀
    @param source 调用来源（cli / agent）
    @param tool 工具名（日志用）
    @return 格式化文本
    """
    from llgraph.code_index.index_settings import resolve_index_settings

    logger = get_search_logger()
    skip_dirs = frozenset(resolve_index_settings(workspace).skip_dirs)
    ctx = WorkspaceContext(workspace, allow_write=False, extra_skip_dirs=skip_dirs)
    prefix = path_prefix.strip() if path_prefix else "."

    try:
        return _search_hybrid_impl(
            workspace,
            query,
            top_k=top_k,
            path_prefix=prefix,
            source=source,
            tool=tool,
            ctx=ctx,
            skip_dirs=skip_dirs,
            logger=logger,
        )
    except UnicodeDecodeError as exc:
        logger.error(
            "[hybrid] 文本解码失败 source=%s tool=%s path_prefix=%r error=%s",
            source,
            tool,
            prefix,
            exc,
        )
        return (
            f"Hybrid 检索失败：命中行含非 UTF-8 文本（{exc}）。"
            f"请缩小 path_prefix（当前 {prefix!r}），或排除 target/二进制目录后重试；"
            "也可改用 grep_files(pattern=..., path=<仓库名>)。"
        )
    except OSError as exc:
        logger.error(
            "[hybrid] IO 失败 source=%s tool=%s path_prefix=%r error=%s",
            source,
            tool,
            prefix,
            exc,
        )
        return f"Hybrid 检索失败（IO）: {exc}"


def _search_hybrid_impl(
    workspace: Path,
    query: str,
    *,
    top_k: int,
    path_prefix: str,
    source: str,
    tool: str,
    ctx: WorkspaceContext,
    skip_dirs: frozenset[str],
    logger,
) -> str:
    """search_hybrid 核心实现（便于统一捕获解码/IO 异常）。"""
    prefix = path_prefix

    path_list = path_hits(ctx, query, prefix, limit=PATH_TOP)
    grep_list = _grep_hits(ctx, query, prefix, limit=GREP_TOP)
    vector_list, vector_err = _vector_hit_ids(
        workspace,
        query,
        prefix,
        VECTOR_TOP,
        source=source,
        tool=tool,
    )

    logger.info(
        "[vector] hybrid 汇总 source=%s tool=%s path_hits=%d grep_hits=%d vector_hits=%d "
        "top_k=%d vector_err=%r",
        source,
        tool,
        len(path_list),
        len(grep_list),
        len(vector_list),
        top_k,
        vector_err or "",
    )

    scores: dict[str, float] = {}
    previews: dict[str, str] = {}

    for rank, (doc_id, snippet) in enumerate(path_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        previews.setdefault(doc_id, snippet)

    for rank, (doc_id, snippet) in enumerate(grep_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        previews.setdefault(doc_id, snippet)

    for rank, (doc_id, snippet) in enumerate(vector_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        previews.setdefault(doc_id, snippet)

    if not scores:
        msg = "Hybrid 检索无结果。"
        if vector_err:
            msg += f" 向量路失败: {vector_err}"
        msg += " 请先 llgraph index -C . 或调整关键词。"
        return msg

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    filtered: list[tuple[str, float]] = []
    for doc_id, score in ranked:
        rel = doc_id.split(":", 1)[0]
        if is_junk_search_path(rel):
            continue
        filtered.append((doc_id, score))
        if len(filtered) >= top_k:
            break
    if not filtered:
        msg = "Hybrid 检索无有效结果（已过滤 .git/target 等噪声）。"
        if vector_err:
            msg += f" 向量路: {vector_err}"
        msg += " 请调整 query 或 llgraph index -C . 重建索引。"
        return msg

    lines = [
        f"Hybrid 检索 Top{len(filtered)}（RRF k={RRF_K}, "
        f"path={len(path_list)}, grep={len(grep_list)}, vector={len(vector_list)}）:",
        "",
    ]
    for i, (doc_id, score) in enumerate(filtered, start=1):
        prev = previews.get(doc_id, "")
        lines.append(f"{i}. [{score:.4f}] {doc_id}")
        if prev:
            lines.append(f"   {prev}")
    if vector_err:
        lines.append(f"\n（向量路告警: {vector_err}）")
    suggest_paths = _unique_file_paths(filtered)
    if len(suggest_paths) >= 2:
        paths_json = json.dumps(suggest_paths, ensure_ascii=False)
        lines.append(
            f"\n【批量读建议】下一轮请**一次** read_files(paths={paths_json})，"
            f"勿逐个 read_file（最多 {_HYBRID_SUGGEST_READ_FILES} 个/次）。"
        )
    elif len(suggest_paths) == 1:
        lines.append(f"\n精读建议: read_file(path={suggest_paths[0]!r})")
    else:
        lines.append("\n请对命中位置使用 read_files / read_file 精读。")
    return "\n".join(lines)
