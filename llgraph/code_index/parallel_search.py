"""并行检索：主 Agent query → 字面量 grep ∥ 向量，RRF 融合（无内嵌 LLM / chain）。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llgraph.code_index.literal_grep import literal_grep_hits
from llgraph.code_index.paths import DEFAULT_SEARCH_TOP_K
from llgraph.code_index.search_format import format_search_hit
from llgraph.code_index.search_params import build_parallel_search_params_with_timing
from llgraph.code_index.search import collect_vector_hits
from llgraph.code_index.search_path_filter import is_junk_search_path
from llgraph.config.logging_settings import get_search_logger
from llgraph.core.workspace import WorkspaceContext

RRF_K = 60
LITERAL_RRF_BOOST = 2.4
LITERAL_TOP = 40
VECTOR_TOP = 30
_MAX_HITS_PER_REPO = 3
_PARALLEL_WORKERS = 2
_SUGGEST_READ_FILES = 5


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _unique_file_paths(
    ranked: list[tuple[str, float]],
    *,
    limit: int = _SUGGEST_READ_FILES,
) -> list[str]:
    """
    从 doc_id（rel_path:line）提取去重文件路径，供 read_files 建议。

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


def _repo_key(doc_id: str) -> str:
    rel = doc_id.split(":", 1)[0]
    if "/" in rel:
        return rel.split("/", 1)[0]
    return rel or "."


def _apply_repo_cap(
    ranked: list[tuple[str, float]],
    *,
    per_repo: int = _MAX_HITS_PER_REPO,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    多仓限流：单仓最多 per_repo 条，避免单仓库占满 TopK。

    @param ranked 全量排序结果
    @param per_repo 单仓上限
    @param top_k 最终条数
    @return 限流后列表
    """
    repo_counts: dict[str, int] = {}
    out: list[tuple[str, float]] = []
    for doc_id, score in ranked:
        repo = _repo_key(doc_id)
        if repo_counts.get(repo, 0) >= per_repo:
            continue
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        out.append((doc_id, score))
        if len(out) >= top_k:
            break
    return out


def search_parallel(
    workspace: Path,
    query: str,
    *,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    path_prefix: str = ".",
    source: str = "cli",
    tool: str = "search_code_parallel",
) -> str:
    """
    并行代码检索：启发式 literal grep ∥ 向量 → RRF（无内嵌 LLM）。

    @param workspace 工作区根
    @param query 主 Agent 扩展后的检索词
    @param top_k 最终条数
    @param path_prefix 路径前缀
    @param source 调用来源（cli / agent / compress）
    @param tool 工具名（日志用）
    @return 格式化文本
    """
    from llgraph.code_index.index_settings import resolve_index_settings

    logger = get_search_logger()
    skip_dirs = frozenset(resolve_index_settings(workspace).skip_dirs)
    ctx = WorkspaceContext(workspace, allow_write=False, extra_skip_dirs=skip_dirs)
    prefix = path_prefix.strip() if path_prefix else "."

    try:
        return _search_parallel_impl(
            workspace,
            query,
            top_k=top_k,
            path_prefix=prefix,
            source=source,
            tool=tool,
            ctx=ctx,
            logger=logger,
        )
    except UnicodeDecodeError as exc:
        logger.error(
            "[parallel] 文本解码失败 source=%s tool=%s path_prefix=%r error=%s",
            source,
            tool,
            prefix,
            exc,
        )
        return (
            f"并行检索失败：命中行含非 UTF-8 文本（{exc}）。"
            f"请缩小 path_prefix（当前 {prefix!r}），或改用 grep_files。"
        )
    except OSError as exc:
        logger.error(
            "[parallel] IO 失败 source=%s tool=%s path_prefix=%r error=%s",
            source,
            tool,
            prefix,
            exc,
        )
        return f"并行检索失败（IO）: {exc}"


def _search_parallel_impl(
    workspace: Path,
    query: str,
    *,
    top_k: int,
    path_prefix: str,
    source: str,
    tool: str,
    ctx: WorkspaceContext,
    logger,
) -> str:
    """search_parallel 核心：启发式拆参 → literal grep ∥ 向量 → RRF。"""
    params_result = build_parallel_search_params_with_timing(query)
    params = params_result.params

    if source == "agent":
        from llgraph.display.trace_emit import emit_parallel_search_params_trace

        emit_parallel_search_params_trace(
            params,
            elapsed=params_result.elapsed_sec,
            parent_tool=tool,
            user_query=query,
        )

    literal_list: list[tuple[str, str]] = []
    vector_list: list[tuple[str, str]] = []
    vector_err = ""

    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as pool:
        fut_literal = pool.submit(
            literal_grep_hits,
            ctx,
            params.literal_grep_patterns,
            path_prefix,
            limit=LITERAL_TOP,
        )
        fut_vector = pool.submit(
            collect_vector_hits,
            workspace,
            params.semantic_query,
            top_k=VECTOR_TOP,
            path_prefix=path_prefix,
            source=source,
            tool=tool,
        )
        literal_list = fut_literal.result()
        vector_list, vector_err = fut_vector.result()

    logger.info(
        "[parallel] 汇总 source=%s tool=%s literal=%d vector=%d top_k=%d vector_err=%r",
        source,
        tool,
        len(literal_list),
        len(vector_list),
        top_k,
        vector_err or "",
    )

    scores: dict[str, float] = {}
    previews: dict[str, str] = {}

    for rank, (doc_id, snippet) in enumerate(literal_list, start=1):
        rel = doc_id.split(":", 1)[0]
        if is_junk_search_path(rel):
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank) * LITERAL_RRF_BOOST
        previews.setdefault(doc_id, snippet)

    for rank, (doc_id, snippet) in enumerate(vector_list, start=1):
        rel = doc_id.split(":", 1)[0]
        if is_junk_search_path(rel):
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        previews.setdefault(doc_id, snippet)

    if not scores:
        msg = "并行检索无结果。"
        if vector_err:
            msg += f" 向量路: {vector_err}"
        msg += " 请先 llgraph index -C . 或调整 query / path_prefix。"
        return msg

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    filtered = _apply_repo_cap(ranked, top_k=top_k)
    if not filtered:
        msg = "并行检索无有效结果（已过滤噪声路径）。"
        if vector_err:
            msg += f" 向量路: {vector_err}"
        return msg

    lines = [f"并行检索 Top{len(filtered)}:", ""]
    for i, (doc_id, _score) in enumerate(filtered, start=1):
        prev = previews.get(doc_id, "")
        lines.append(format_search_hit(i, doc_id, prev))
    if vector_err:
        lines.append(f"\n（向量路告警: {vector_err}）")

    suggest_paths = _unique_file_paths(filtered)
    if len(suggest_paths) >= 2:
        paths_json = json.dumps(suggest_paths, ensure_ascii=False)
        lines.append(
            f"\n精读: read_files(paths={paths_json})，"
            f"再 grep_files 追上下游（最多 {_SUGGEST_READ_FILES} 个/次）。"
        )
    elif len(suggest_paths) == 1:
        lines.append(f"\n精读: read_file(path={suggest_paths[0]!r})")
    else:
        lines.append("\n精读: read_files / read_file。")
    lines.append(
        "勿重复 search_code_parallel（再次调用将被拦截）。"
        "下一步：grep_files(pattern=\"词A|词B\", path=\".\") 定行号，再 read_files/read_file(宽段)；"
        "path 须从本结果原样复制，禁止猜目录名。"
    )
    return "\n".join(lines)
