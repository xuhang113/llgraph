"""字面量 ripgrep 检索（供 parallel_search 并行调用）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from llgraph.code_index.search_format import truncate_search_snippet
from llgraph.code_index.search_path_filter import is_junk_search_path
from llgraph.core.workspace import WorkspaceContext

LITERAL_GREP_TOP = 60
_PARALLEL_WORKERS = 4


def _parse_ripgrep_line(line: str) -> tuple[str, str, str] | None:
    """解析 ripgrep 输出行 rel:line:snippet。"""
    parts = line.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def collect_ripgrep_hits(
    ctx: WorkspaceContext,
    pattern: str,
    path_prefix: str,
    *,
    limit: int,
) -> list[tuple[str, str]]:
    """
    单次 ripgrep 命中列表。

    @param ctx 工作区上下文
    @param pattern 正则或字面量
    @param path_prefix 路径前缀
    @param limit 本 pattern 最多条数
    @return (doc_id, snippet) 列表，doc_id 为 rel_path:line
    """
    from llgraph.core.ripgrep_search import ripgrep_content

    if not pattern.strip():
        return []
    hits, _err = ripgrep_content(
        ctx.root,
        pattern,
        path_prefix=path_prefix or ".",
        limit=limit,
        skip_dirs=ctx._extra_skip_dirs,
    )
    out: list[tuple[str, str]] = []
    for line in hits:
        parsed = _parse_ripgrep_line(line)
        if parsed is None:
            continue
        rel, line_no, snippet_body = parsed
        if is_junk_search_path(rel):
            continue
        doc_id = f"{rel}:{line_no}"
        snippet = truncate_search_snippet(snippet_body.strip())
        out.append((doc_id, snippet))
    return out


def literal_grep_hits(
    ctx: WorkspaceContext,
    patterns: tuple[str, ...],
    path_prefix: str,
    *,
    limit: int = LITERAL_GREP_TOP,
) -> list[tuple[str, str]]:
    """
    字面量 grep：并行执行多组 ripgrep 模式（对齐 Cursor 多路 Grep）。

    @param ctx 工作区上下文
    @param patterns ripgrep 正则列表
    @param path_prefix 路径前缀
    @param limit 总条数上限
    @return (doc_id, snippet) 列表
    """
    if not patterns:
        return []

    from llgraph.core.ripgrep_search import ripgrep_available

    if not ripgrep_available():
        return []

    per_pattern = max(12, limit // max(1, len(patterns)))
    batches: list[list[tuple[str, str]]] = []

    if len(patterns) == 1:
        batches.append(
            collect_ripgrep_hits(ctx, patterns[0], path_prefix, limit=limit)
        )
    else:
        with ThreadPoolExecutor(max_workers=min(len(patterns), _PARALLEL_WORKERS)) as pool:
            futures = [
                pool.submit(
                    collect_ripgrep_hits,
                    ctx,
                    pat,
                    path_prefix,
                    limit=per_pattern,
                )
                for pat in patterns
            ]
            for fut in futures:
                batches.append(fut.result())

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    if not batches:
        return deduped
    max_len = max(len(batch) for batch in batches)
    # 轮询合并：每路 pattern 均有机会进 Top（避免单 pattern 独占配额）
    for idx in range(max_len):
        for batch in batches:
            if idx >= len(batch):
                continue
            doc_id, snippet = batch[idx]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            deduped.append((doc_id, snippet))
            if len(deduped) >= limit:
                return deduped
    return deduped
