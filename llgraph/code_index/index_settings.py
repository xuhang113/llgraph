"""代码索引行为配置（.llgraph/embedding.json 内 index 段）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from llgraph.code_index.embedding_config import load_embedding_config

# 0 表示不限制可索引文件数（全工作区渐进同步）
DEFAULT_MAX_FILES = 0
DEFAULT_MANIFEST_FLUSH_EVERY = 50
DEFAULT_PROGRESS_LOG_EVERY = 500
DEFAULT_PREPARE_WORKERS = 4
DEFAULT_EMBED_ACCUMULATE_CHUNKS = 128
# 默认关闭：首跑/全量以 manifest 跳过未变文件即可；开启后查 embed_cache.db 复用向量
DEFAULT_USE_EMBED_CACHE = False


def _parse_bool(value: object, default: bool) -> bool:
    """
    解析布尔配置。

    @param value 配置值
    @param default 无法识别时的默认值
    @return 布尔结果
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _parse_skip_dirs(index_cfg: dict) -> tuple[str, ...]:
    """
    解析 index.skip_dirs：工作区根下要整目录跳过的目录名。

    @param index_cfg embedding.json 的 index 段
    @return 目录名元组（不含路径分隔符）
    """
    raw = index_cfg.get("skip_dirs")
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip().strip("/")
        if name and name not in names:
            names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class IndexSettings:
    """索引扫描与渐进同步参数。"""

    max_files: int
    manifest_flush_every: int
    progress_log_every: int
    progressive: bool
    prepare_workers: int
    embed_accumulate_chunks: int
    use_embed_cache: bool
    skip_dirs: tuple[str, ...]
    watch_enabled: bool
    watch_with_agent: bool
    watch_debounce_sec: float
    show_progress: bool


def _parse_watch_settings(index_cfg: dict) -> tuple[bool, bool, float]:
    """
    解析 index.watch 段。

    @param index_cfg embedding.json 的 index 段
    @return (enabled, with_agent, debounce_sec)
    """
    watch = index_cfg.get("watch")
    if not isinstance(watch, dict):
        return True, True, 5.0
    enabled = _parse_bool(watch.get("enabled"), True)
    with_agent = _parse_bool(watch.get("with_agent"), True)
    debounce = watch.get("debounce_sec", 5.0)
    try:
        debounce = max(1.0, float(debounce))
    except (TypeError, ValueError):
        debounce = 5.0
    return enabled, with_agent, debounce


def resolve_index_settings(workspace: Path) -> IndexSettings:
    """
    解析索引配置。

    @param workspace 工作区根
    @return IndexSettings；max_files=0 表示不限制
    """
    cfg = load_embedding_config(workspace)
    index_cfg = cfg.get("index") if isinstance(cfg.get("index"), dict) else {}

    max_files = index_cfg.get("max_files", DEFAULT_MAX_FILES)
    try:
        max_files = int(max_files)
    except (TypeError, ValueError):
        max_files = DEFAULT_MAX_FILES
    if max_files < 0:
        max_files = 0

    flush = index_cfg.get("manifest_flush_every", DEFAULT_MANIFEST_FLUSH_EVERY)
    try:
        flush = max(1, int(flush))
    except (TypeError, ValueError):
        flush = DEFAULT_MANIFEST_FLUSH_EVERY

    log_every = index_cfg.get("progress_log_every", DEFAULT_PROGRESS_LOG_EVERY)
    try:
        log_every = max(50, int(log_every))
    except (TypeError, ValueError):
        log_every = DEFAULT_PROGRESS_LOG_EVERY

    progressive = index_cfg.get("progressive", True)
    if isinstance(progressive, str):
        progressive = progressive.strip().lower() not in ("0", "false", "no")
    else:
        progressive = bool(progressive)

    workers = index_cfg.get("prepare_workers", DEFAULT_PREPARE_WORKERS)
    try:
        workers = max(1, int(workers))
    except (TypeError, ValueError):
        workers = DEFAULT_PREPARE_WORKERS
    # 切块线程过多会与 embed/写库抢 CPU，通常 8～16 即可
    cpu_cap = max(8, (os.cpu_count() or 4) * 2)
    workers = min(workers, cpu_cap)

    accum = index_cfg.get("embed_accumulate_chunks", DEFAULT_EMBED_ACCUMULATE_CHUNKS)
    try:
        accum = max(16, int(accum))
    except (TypeError, ValueError):
        accum = DEFAULT_EMBED_ACCUMULATE_CHUNKS

    env_max = os.getenv("LLGRAPH_INDEX_MAX_FILES", "").strip()
    if env_max.isdigit():
        max_files = int(env_max)

    env_workers = os.getenv("LLGRAPH_INDEX_PREPARE_WORKERS", "").strip()
    if env_workers.isdigit():
        workers = max(1, int(env_workers))

    env_accum = os.getenv("LLGRAPH_INDEX_EMBED_ACCUMULATE", "").strip()
    if env_accum.isdigit():
        accum = max(16, int(env_accum))

    use_embed_cache = _parse_bool(
        index_cfg.get("use_embed_cache"),
        DEFAULT_USE_EMBED_CACHE,
    )
    env_cache = os.getenv("LLGRAPH_INDEX_USE_EMBED_CACHE", "").strip()
    if env_cache:
        use_embed_cache = _parse_bool(env_cache, use_embed_cache)

    skip_dirs = _parse_skip_dirs(index_cfg)
    # 始终跳过 .llgraph（配置、索引产物、上下文落盘）
    merged: list[str] = []
    for name in (*skip_dirs, ".llgraph"):
        if name and name not in merged:
            merged.append(name)
    skip_dirs = tuple(merged)
    watch_enabled, watch_with_agent, watch_debounce = _parse_watch_settings(index_cfg)

    show_progress = _parse_bool(index_cfg.get("show_progress"), True)

    return IndexSettings(
        max_files=max_files,
        manifest_flush_every=flush,
        progress_log_every=log_every,
        progressive=progressive,
        prepare_workers=workers,
        embed_accumulate_chunks=accum,
        use_embed_cache=use_embed_cache,
        skip_dirs=skip_dirs,
        watch_enabled=watch_enabled,
        watch_with_agent=watch_with_agent,
        watch_debounce_sec=watch_debounce,
        show_progress=show_progress,
    )


def effective_max_files(settings: IndexSettings) -> int | None:
    """
    供 file_scan 使用的上限；None 表示不限制。

    @param settings 索引配置
    @return 最大文件数或 None
    """
    if settings.max_files <= 0:
        return None
    return settings.max_files
