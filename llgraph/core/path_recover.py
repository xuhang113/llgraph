"""路径容错：猜错 path 时唯一解析或给出相近路径。

对标 Cursor / Claude Code / Codex：模型常漏仓库前缀、写错中间目录、只用 basename。
工具层当场纠正，避免再空转一轮 LLM 去 list_directory / glob。
写工具仅在高置信唯一命中时改写已有文件；write_file 新建不改路径。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from llgraph.code_index.search_path_filter import is_junk_search_path
from llgraph.core.workspace import WorkspaceContext

Want = Literal["file", "dir", "any"]

AUTO_RESOLVE_MARKER = "[llgraph] path 已唯一解析"
_MAX_FILES = 30_000
_MAX_SUGGEST = 5
_CACHE_TTL_SEC = 2.0
_RG_TIMEOUT_SEC = 3.0
_AUTO_MIN_SCORE = 0.90
_AUTO_GAP = 0.08
_SUGGEST_MIN_SCORE = 0.52

_cache_lock = threading.Lock()
_listing_cache: dict[str, tuple[float, tuple[str, ...], tuple[str, ...]]] = {}


@dataclass(frozen=True)
class PathHit:
    """一条候选路径。"""

    rel: str
    score: float
    reason: str


@dataclass(frozen=True)
class PathResolution:
    """工具 path 解析结果。"""

    ok: bool
    rel: str = ""
    note: str = ""
    kind: str = "missing"
    is_file: bool = False
    is_dir: bool = False
    candidates: tuple[str, ...] = ()


def normalize_rel_path(path: str) -> str:
    """
    规范化相对路径（POSIX、去掉 ./ 与首尾斜杠）。

    @param path 原始 path
    @return 规范化相对路径；空或 "." 返回空串表示工作区根
    """
    raw = (path or "").strip().replace("\\", "/")
    if not raw or raw == ".":
        return ""
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.strip("/")


def is_external_read_path(path: str) -> bool:
    """是否为工作区外绝对/家目录路径（技能/规则），不做仓库内恢复。"""
    raw = (path or "").strip()
    if not raw:
        return False
    if raw.startswith("~") or raw.startswith("/"):
        return True
    if len(raw) >= 3 and raw[1] == ":" and raw[0].isalpha():
        return True
    return False


def invalidate_path_listing_cache(root: Path | None = None) -> None:
    """
    写文件后作废路径缓存。

    @param root 工作区根；None 清空全部
    """
    with _cache_lock:
        if root is None:
            _listing_cache.clear()
            return
        _listing_cache.pop(str(Path(root).expanduser().resolve()), None)


def extract_glob_literal_name(glob_pattern: str) -> str:
    """
    从 glob 取出不含通配符的文件名（如 **/Foo.java → Foo.java）。

    @param glob_pattern glob
    @return 字面文件名；含通配或过宽则空串
    """
    name = Path(normalize_rel_path(glob_pattern) or glob_pattern).name
    name = (name or "").strip()
    if not name or name in {"*", "**", "*.*", "**/*"}:
        return ""
    if any(ch in name for ch in "*?[]"):
        return ""
    if len(name) < 3:
        return ""
    return name


def format_auto_resolve_note(guess: str, resolved: str) -> str:
    """唯一解析提示（拼在工具结果前）。"""
    return f"{AUTO_RESOLVE_MARKER}: {guess} → {resolved}\n"


def format_missing_path(kind: str, guess: str, candidates: list[str]) -> str:
    """
    路径不存在时的工具返回（保留「文件不存在/路径不存在」前缀供失败检测）。

    @param kind 文件|目录
    @param guess 原始 path
    @param candidates 相近路径
    @return 错误文案
    """
    label = "文件不存在" if kind == "文件" else "路径不存在"
    if not candidates:
        return (
            f"{label}: {guess}\n"
            "未找到相近路径。请改用 path=\".\" 做 glob/grep，或从工具输出复制完整相对路径。"
        )
    lines = [f"{label}: {guess}", "相近路径:"]
    lines.extend(f"- {item}" for item in candidates)
    lines.append("请改用上述完整相对路径重试，勿再猜仓库名前缀或中间目录。")
    return "\n".join(lines)


def resolve_tool_path(
    ctx: WorkspaceContext,
    guess: str,
    *,
    want: Want = "any",
    allow_auto: bool = True,
) -> PathResolution:
    """
    解析工具 path：已存在则原样；否则唯一高置信命中自动改写，否则给建议。

    @param ctx 工作区
    @param guess 模型传入 path
    @param want 期望文件/目录/任意
    @param allow_auto 是否允许改写为唯一命中（write_file 新建应 False）
    @return 解析结果
    """
    if is_external_read_path(guess):
        return PathResolution(ok=False, rel=guess.strip(), kind="external")

    norm = normalize_rel_path(guess)
    if not norm:
        try:
            target = ctx.resolve_path(".")
        except ValueError as exc:
            return PathResolution(ok=False, rel=".", note=str(exc), kind="invalid")
        return PathResolution(
            ok=True,
            rel=".",
            kind="exists",
            is_dir=target.is_dir(),
            is_file=target.is_file(),
        )

    existing = _existing_resolution(ctx, norm, want=want)
    if existing is not None:
        return existing

    stripped = _strip_workspace_prefix(ctx, norm)
    if stripped and stripped != norm:
        existing = _existing_resolution(ctx, stripped, want=want)
        if existing is not None and existing.ok:
            return PathResolution(
                ok=True,
                rel=existing.rel,
                note=format_auto_resolve_note(guess, existing.rel),
                kind="auto",
                is_file=existing.is_file,
                is_dir=existing.is_dir,
            )

    hits = suggest_paths(ctx, norm, want=want, limit=8)
    if allow_auto:
        auto = _pick_unique_auto(hits)
        if auto is not None:
            try:
                target = ctx.resolve_path(auto.rel)
            except ValueError:
                auto = None
            else:
                if _matches_want(target, want):
                    return PathResolution(
                        ok=True,
                        rel=auto.rel,
                        note=format_auto_resolve_note(guess, auto.rel),
                        kind="auto",
                        is_file=target.is_file(),
                        is_dir=target.is_dir(),
                    )

    candidates = tuple(hit.rel for hit in hits[:_MAX_SUGGEST] if hit.score >= _SUGGEST_MIN_SCORE)
    kind_label = "文件" if want == "file" else "路径"
    return PathResolution(
        ok=False,
        rel=norm,
        note=format_missing_path(kind_label, guess, list(candidates)),
        kind="suggest" if candidates else "missing",
        candidates=candidates,
    )


def suggest_paths(
    ctx: WorkspaceContext,
    guess: str,
    *,
    want: Want = "any",
    limit: int = _MAX_SUGGEST,
) -> list[PathHit]:
    """
    按 basename / 后缀 / 模糊度给相近路径。

    @param ctx 工作区
    @param guess 错误 path
    @param want 过滤文件或目录
    @param limit 最多条数
    @return 按分数降序的命中
    """
    norm = normalize_rel_path(guess)
    if not norm:
        return []
    files, dirs = list_workspace_rel_paths(ctx)
    pool = _pool_for_want(files, dirs, want)
    if not pool:
        return []

    scored: list[PathHit] = []
    g_lower = norm.lower()
    g_name = Path(norm).name.lower()
    g_stem = Path(norm).stem.lower()
    suffix_hits: list[PathHit] = []
    name_hits: list[PathHit] = []

    for rel in pool:
        c_lower = rel.lower()
        if c_lower == g_lower:
            scored.append(PathHit(rel, 1.0, "exact"))
            continue
        if c_lower.endswith("/" + g_lower):
            suffix_hits.append(PathHit(rel, 0.97, "suffix"))
            continue
        c_name = Path(rel).name.lower()
        if g_name and c_name == g_name:
            name_hits.append(PathHit(rel, 0.93, "basename"))

    scored.extend(suffix_hits)
    scored.extend(name_hits)

    if not scored:
        stem_hits: list[PathHit] = []
        if g_stem and len(g_stem) >= 3:
            for rel in pool:
                if Path(rel).stem.lower() == g_stem:
                    stem_hits.append(PathHit(rel, 0.78, "stem"))
        if stem_hits:
            scored.extend(stem_hits)
        else:
            scored.extend(_fuzzy_name_hits(pool, g_name, g_lower))

    scored.sort(key=lambda hit: (-hit.score, hit.rel))
    deduped: list[PathHit] = []
    seen: set[str] = set()
    for hit in scored:
        if hit.rel in seen:
            continue
        seen.add(hit.rel)
        deduped.append(hit)
        if len(deduped) >= max(limit, 8):
            break
    return deduped[:limit]


def list_workspace_rel_paths(
    ctx: WorkspaceContext,
    *,
    max_files: int = _MAX_FILES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    列出工作区相对文件与目录（短 TTL 缓存；优先 rg --files）。

    @param ctx 工作区
    @param max_files 文件数上限
    @return (files, dirs)
    """
    key = str(ctx.root)
    now = time.monotonic()
    with _cache_lock:
        cached = _listing_cache.get(key)
        if cached is not None and now - cached[0] < _CACHE_TTL_SEC:
            return cached[1], cached[2]

    files, dirs = _list_rel_paths_uncached(ctx, max_files=max_files)
    file_t = tuple(files)
    dir_t = tuple(dirs)
    with _cache_lock:
        _listing_cache[key] = (time.monotonic(), file_t, dir_t)
    return file_t, dir_t


def _existing_resolution(
    ctx: WorkspaceContext,
    rel: str,
    *,
    want: Want,
) -> PathResolution | None:
    try:
        target = ctx.resolve_path(rel or ".")
    except ValueError:
        return None
    if not target.exists():
        return None
    is_file = target.is_file()
    is_dir = target.is_dir()
    if not _matches_want(target, want):
        return PathResolution(
            ok=False,
            rel=rel or ".",
            kind="wrong_type",
            is_file=is_file,
            is_dir=is_dir,
            note=_wrong_type_note(rel or ".", want, is_file=is_file, is_dir=is_dir),
        )
    return PathResolution(
        ok=True,
        rel=rel or ".",
        kind="exists",
        is_file=is_file,
        is_dir=is_dir,
    )


def _matches_want(target: Path, want: Want) -> bool:
    if want == "file":
        return target.is_file()
    if want == "dir":
        return target.is_dir()
    return target.exists()


def _wrong_type_note(rel: str, want: Want, *, is_file: bool, is_dir: bool) -> str:
    if want == "file" and is_dir:
        return f"这是目录不是文件: {rel}"
    if want == "dir" and is_file:
        return f"不是目录: {rel}（这是文件）。请用 read_file(path={rel!r})。"
    return f"路径类型不符: {rel}"


def _strip_workspace_prefix(ctx: WorkspaceContext, rel: str) -> str:
    name = ctx.root.name.replace("\\", "/").strip("/")
    if not name:
        return ""
    prefix = name.lower() + "/"
    if rel.lower().startswith(prefix):
        return rel[len(name) + 1 :]
    return ""


def _pick_unique_auto(hits: list[PathHit]) -> PathHit | None:
    auto_hits = [hit for hit in hits if hit.score >= _AUTO_MIN_SCORE]
    if not auto_hits:
        return None
    best = auto_hits[0]
    if best.reason not in {"exact", "suffix", "basename"}:
        return None
    if len(auto_hits) == 1:
        return best
    second = auto_hits[1]
    if best.score - second.score >= _AUTO_GAP:
        return best
    if best.reason == "suffix" and all(hit.rel == best.rel or hit.reason != "suffix" for hit in auto_hits[1:]):
        return best
    return None


def _pool_for_want(
    files: tuple[str, ...],
    dirs: tuple[str, ...],
    want: Want,
) -> tuple[str, ...]:
    if want == "file":
        return files
    if want == "dir":
        return dirs
    return files + dirs


def _fuzzy_name_hits(pool: tuple[str, ...], g_name: str, g_lower: str) -> list[PathHit]:
    if not g_name or len(g_name) < 4:
        return []
    prefix = g_name[:2]
    scored: list[PathHit] = []
    for rel in pool:
        c_name = Path(rel).name.lower()
        if prefix not in c_name and not c_name.startswith(prefix):
            continue
        name_ratio = SequenceMatcher(None, g_name, c_name).ratio()
        if name_ratio < 0.72:
            continue
        path_ratio = SequenceMatcher(None, g_lower, rel.lower()).ratio()
        score = 0.62 * name_ratio + 0.28 * path_ratio
        if score >= _SUGGEST_MIN_SCORE:
            scored.append(PathHit(rel, score, "fuzzy"))
        if len(scored) >= 40:
            break
    scored.sort(key=lambda hit: (-hit.score, hit.rel))
    return scored[:12]


def _list_rel_paths_uncached(
    ctx: WorkspaceContext,
    *,
    max_files: int,
) -> tuple[list[str], list[str]]:
    files = _try_rg_files(ctx, max_files=max_files)
    if files is None:
        files, dirs = _walk_rel_paths(ctx, max_files=max_files)
        return files, dirs
    return files, _dirs_from_files(files)


def _try_rg_files(ctx: WorkspaceContext, *, max_files: int) -> list[str] | None:
    try:
        from llgraph.core.ripgrep_search import ripgrep_available, ripgrep_files
    except Exception:
        return None
    if not ripgrep_available():
        return None
    try:
        paths, err = ripgrep_files(
            ctx.root,
            "**/*",
            path_prefix=".",
            limit=max_files,
            skip_dirs=ctx._extra_skip_dirs,
            timeout=_RG_TIMEOUT_SEC,
            all_files=True,
        )
    except Exception:
        return None
    if err:
        return None
    out: list[str] = []
    for rel in paths:
        if is_junk_search_path(rel):
            continue
        out.append(rel)
    return out


def _walk_rel_paths(
    ctx: WorkspaceContext,
    *,
    max_files: int,
) -> tuple[list[str], list[str]]:
    files: list[str] = []
    dirs: list[str] = []
    root = ctx.root
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [name for name in dirnames if not ctx.should_skip_dir(name)]
        try:
            rel_dir = Path(dirpath).resolve().relative_to(root).as_posix()
        except ValueError:
            dirnames[:] = []
            continue
        if rel_dir != ".":
            if is_junk_search_path(rel_dir + "/"):
                dirnames[:] = []
                continue
            dirs.append(rel_dir)
        for filename in filenames:
            rel = filename if rel_dir == "." else f"{rel_dir}/{filename}"
            if is_junk_search_path(rel):
                continue
            files.append(rel)
            if len(files) >= max_files:
                return files, dirs
    return files, dirs


def _dirs_from_files(files: list[str]) -> list[str]:
    seen: set[str] = set()
    dirs: list[str] = []
    for rel in files:
        parent = Path(rel).parent.as_posix()
        while parent and parent != ".":
            if parent not in seen:
                seen.add(parent)
                dirs.append(parent)
            nxt = Path(parent).parent.as_posix()
            if nxt == parent:
                break
            parent = nxt
    dirs.sort()
    return dirs
