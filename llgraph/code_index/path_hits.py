"""工作区路径/文件名关键字匹配（供 hybrid 与 search_files 共用）。"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from llgraph.code_index.search_path_filter import is_junk_search_path
from llgraph.core.workspace import WorkspaceContext

PATH_HIT_TOP = 40


def extract_path_search_tokens(query: str) -> list[str]:
    """
    从 query 提取用于路径匹配的词元。

    @param query 用户查询
    @return 去重后的词元列表
    """
    raw = re.findall(
        r"[\u4e00-\u9fff]{2,}|[A-Za-z_][\w.-]{2,}|\S+\.(?:sh|py|java|md|yml|yaml|json)",
        query,
    )
    if not raw:
        text = query.strip()
        return [text] if text else []
    seen: set[str] = set()
    out: list[str] = []
    for token in raw:
        key = token.lower()
        if key in seen or len(key) < 2:
            continue
        seen.add(key)
        out.append(token)
    return out[:12]


def match_paths_by_keyword(
    ctx: WorkspaceContext,
    keyword_lower: str,
    path: str,
    glob_pattern: str,
    limit: int,
) -> list[str]:
    """
    按关键字匹配相对路径（含顶层目录名优先扫描）。

    @param ctx 工作区上下文
    @param keyword_lower 小写关键字
    @param path 起始相对目录
    @param glob_pattern 文件名 glob
    @param limit 最多条数
    @return 相对路径列表
    """
    matches: list[str] = []
    seen: set[str] = set()
    base = ctx.resolve_path(path)

    def add(rel: str) -> bool:
        if rel in seen or is_junk_search_path(rel):
            return len(matches) >= limit
        if glob_pattern and glob_pattern not in ("**/*", "*"):
            name = Path(rel.rstrip("/")).name
            if not fnmatch.fnmatch(name, glob_pattern.lstrip("**/")):
                if not fnmatch.fnmatch(rel, glob_pattern):
                    return len(matches) >= limit
        seen.add(rel)
        matches.append(rel)
        return len(matches) >= limit

    if keyword_lower and base.is_dir():
        try:
            entries = sorted(
                base.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError:
            entries = []
        for entry in entries:
            if entry.is_dir() and ctx.should_skip_dir(entry.name):
                continue
            try:
                rel = entry.relative_to(ctx.root).as_posix()
            except ValueError:
                continue
            if entry.is_dir():
                rel = f"{rel}/"
            if keyword_lower in rel.lower():
                if add(rel):
                    return matches

    for rel in ctx.iter_files(path):
        if keyword_lower and keyword_lower not in rel.lower():
            continue
        if add(rel):
            return matches
    return matches


def path_hits(
    ctx: WorkspaceContext,
    query: str,
    path_prefix: str,
    *,
    limit: int = PATH_HIT_TOP,
) -> list[tuple[str, str]]:
    """
    路径/文件名路：对 query 中各词元做子串匹配。

    @param ctx 工作区上下文
    @param query 查询
    @param path_prefix 起始相对目录
    @param limit 最多条数
    @return (doc_id, snippet) 列表，doc_id 为 rel_path:0
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in extract_path_search_tokens(query):
        for rel in match_paths_by_keyword(
            ctx,
            token.lower(),
            path_prefix or ".",
            "**/*",
            limit,
        ):
            doc_id = f"{rel}:0"
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append((doc_id, f"(路径/文件名匹配: {token})"))
            if len(results) >= limit:
                return results
    return results
