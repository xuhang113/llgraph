"""search_code_parallel 检索参数：从主 Agent query 启发式拆 literal + semantic。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

_MAX_LITERAL = 8
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][\w.-]{2,}")
_RG_METACHAR = re.compile(r"[\\.*+?^$|[\](){}|]")


@dataclass(frozen=True)
class ParallelSearchParams:
    """并行检索参数（主 Agent query 启发式拆分）。"""

    original: str
    literal_grep_patterns: tuple[str, ...]
    semantic_query: str


@dataclass(frozen=True)
class ParallelSearchParamsResult:
    """检索参数与耗时（供 trace / parallel_search）。"""

    params: ParallelSearchParams
    elapsed_sec: float


def _literal_pattern(tok: str) -> str | None:
    """token 是否适合作 ripgrep literal；纯中文等仅走向量。"""
    tok = tok.strip()
    if not tok:
        return None
    if not re.search(r"[A-Za-z_\\.*+?^$|[\](){}|]", tok):
        return None
    if _RG_METACHAR.search(tok):
        return tok
    if _IDENTIFIER_RE.fullmatch(tok):
        return re.escape(tok)
    if re.search(r"[A-Za-z_]", tok):
        return re.escape(tok)
    return None


def build_parallel_search_params(query: str) -> ParallelSearchParams:
    """
    从主 Agent query 拆检索参数（literal 用 ripgrep，整句走向量）。

    @param query 主 Agent 检索词（可自行扩展类名/关键字，空格分隔）
    @return 检索参数
    """
    raw = (query or "").strip()
    patterns: list[str] = []
    seen: set[str] = set()

    for tok in raw.split():
        pat = _literal_pattern(tok)
        if pat is None or pat in seen:
            continue
        seen.add(pat)
        patterns.append(pat)
        if len(patterns) >= _MAX_LITERAL:
            break

    if not patterns:
        for token in _IDENTIFIER_RE.findall(raw):
            pat = re.escape(token)
            if pat in seen:
                continue
            seen.add(pat)
            patterns.append(pat)
            if len(patterns) >= _MAX_LITERAL:
                break

    return ParallelSearchParams(
        original=raw,
        literal_grep_patterns=tuple(patterns),
        semantic_query=raw,
    )


def build_parallel_search_params_with_timing(
    query: str,
) -> ParallelSearchParamsResult:
    """
    拆检索参数并记录耗时（供 trace）。

    @param query 检索词
    @return 参数 + 耗时
    """
    t0 = time.perf_counter()
    params = build_parallel_search_params(query)
    elapsed = time.perf_counter() - t0
    return ParallelSearchParamsResult(params=params, elapsed_sec=elapsed)
