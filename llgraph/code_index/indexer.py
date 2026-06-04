"""全量/增量/重建索引编排（支持全工作区渐进同步 + 并行加速）。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from llgraph.code_index.embedder import (
    format_embedding_status,
    get_embedding_dimension,
)
from llgraph.code_index.embedding_config import resolve_embedding_profile
from llgraph.code_index.file_scan import file_sha256, iter_indexable_files
from llgraph.code_index.index_logging import get_index_logger
from llgraph.code_index.index_pipeline import run_parallel_index_loop
from llgraph.code_index.index_progress import IndexProgressDisplay, should_update_scan
from llgraph.code_index.index_settings import effective_max_files, resolve_index_settings
from llgraph.code_index.index_skip import purge_skipped_index_entries
from llgraph.code_index.manifest import load_manifest, save_manifest
from llgraph.code_index.paths import DEFAULT_VECTOR_DIM
from llgraph.code_index.rebuild import prepare_rebuild
from llgraph.code_index.store import (
    delete_chunks_for_file,
    delete_chunks_not_in_files,
    touch_index_meta,
)
from llgraph.core.workspace import WorkspaceContext


@dataclass
class IndexRunResult:
    """索引运行结果。"""

    files_scanned: int
    files_updated: int
    files_skipped: int
    chunks_written: int
    errors: list[str] = field(default_factory=list)
    log_file: str | None = None
    hit_file_cap: bool = False


def _persist_manifest(
    workspace: Path,
    manifest: dict[str, str],
    *,
    path_prefix: str,
) -> None:
    """写入 manifest（全工作区或子目录合并）。"""
    prefix = path_prefix.strip() or "."
    if prefix in (".", ""):
        save_manifest(workspace, manifest)
        return
    merged = load_manifest(workspace)
    merged.update(manifest)
    save_manifest(workspace, merged)


def run_index(
    workspace: Path,
    *,
    incremental: bool = False,
    rebuild: bool = False,
    path_prefix: str = ".",
    use_ast: bool = False,
    dry_run: bool = False,
    clear_embedding_cache: bool = False,
    on_progress: Callable[[str], None] | None = None,
    progress: IndexProgressDisplay | None = None,
) -> IndexRunResult:
    """执行索引（渐进同步 + 并行切块 + 跨文件批量 embed）。"""
    logger = get_index_logger()
    index_settings = resolve_index_settings(workspace)
    scan_limit = effective_max_files(index_settings)
    profile = resolve_embedding_profile(workspace)

    def emit(msg: str, level: int = 20) -> None:
        logger.log(level, msg)
        if progress is not None:
            if level >= 30:
                progress.note(msg)
            elif not progress.enabled:
                progress.emit_fallback(msg, level=level)
        elif on_progress:
            on_progress(msg)

    if rebuild and not dry_run:
        emit("执行重建准备：清理旧索引数据…")
        if progress is not None:
            progress.set_phase("prepare")
        prepare_rebuild(
            workspace,
            path_prefix=path_prefix,
            clear_embedding_cache=clear_embedding_cache,
        )

    incremental_effective = incremental and not rebuild
    prefix = path_prefix.strip() or "."
    skip_dirs = frozenset(index_settings.skip_dirs)
    ctx = WorkspaceContext(workspace, allow_write=False, extra_skip_dirs=skip_dirs)
    old_manifest = load_manifest(workspace) if not rebuild else {}
    manifest: dict[str, str] = dict(old_manifest)

    if skip_dirs and not dry_run:
        purged = purge_skipped_index_entries(workspace, manifest, skip_dirs)
        if purged:
            emit(
                f"已忽略目录 {', '.join(sorted(skip_dirs))}，"
                f"清理旧索引 {purged} 个文件"
            )
            _persist_manifest(workspace, manifest, path_prefix=prefix)

    files_scanned = 0
    files_skipped = 0
    files_updated = 0
    chunks_written = 0
    errors: list[str] = []
    hit_file_cap = False

    if scan_limit is None:
        cache_hint = "开" if index_settings.use_embed_cache else "关(默认)"
        emit(
            f"渐进同步（prefix={path_prefix}）| "
            f"workers={index_settings.prepare_workers} | "
            f"每批 {index_settings.embed_accumulate_chunks} chunks | "
            f"embed_cache={cache_hint}"
        )
    else:
        emit(f"扫描（prefix={path_prefix}），最多 {scan_limit} 个文件…")

    if dry_run:
        if progress is not None:
            progress.set_phase("sync")
        pending = 0
        for rel, full in iter_indexable_files(
            ctx, path_prefix=path_prefix, max_files=scan_limit
        ):
            files_scanned += 1
            try:
                digest = file_sha256(full)
            except OSError:
                continue
            manifest[rel] = digest
            if rebuild or old_manifest.get(rel) != digest:
                pending += 1
            else:
                files_skipped += 1
            if progress is not None and should_update_scan(
                files_scanned, index_settings
            ):
                progress.update_scan(files_scanned, files_skipped)
        if scan_limit and files_scanned >= scan_limit:
            hit_file_cap = True
        emit(f"[dry-run] 扫描 {files_scanned} 文件, 待处理约 {pending}")
        if progress is not None:
            progress.update_scan(files_scanned, files_skipped)
        return IndexRunResult(
            files_scanned=files_scanned,
            files_updated=pending,
            files_skipped=files_skipped,
            chunks_written=0,
            errors=errors,
            hit_file_cap=hit_file_cap,
        )

    if rebuild and prefix in (".", ""):
        emit("重建：已清空索引，开始全量渐进同步…")
    elif incremental_effective:
        emit("增量同步：仅处理内容变化的文件…")
    else:
        emit("同步索引：新增/变更文件（并行加速）…")

    if progress is not None:
        progress.set_phase("sync")

    vector_dim = DEFAULT_VECTOR_DIM
    try:
        vector_dim = get_embedding_dimension(workspace)
        emit(
            f"Embedding: {format_embedding_status(workspace, probe_dim=True)}, "
            f"dim={vector_dim}"
        )
    except RuntimeError as exc:
        logger.error("Embedding 配置错误: %s", exc)
        raise

    def work_iter() -> Iterator[tuple[str, Path, str, bool]]:
        nonlocal files_scanned, files_skipped
        for rel, full in iter_indexable_files(
            ctx, path_prefix=path_prefix, max_files=scan_limit
        ):
            files_scanned += 1
            if should_update_scan(files_scanned, index_settings):
                if progress is not None:
                    progress.update_scan(files_scanned, files_skipped)
                else:
                    emit(f"已扫描 {files_scanned} 文件（跳过 {files_skipped}）…")

            try:
                digest = file_sha256(full)
            except OSError as exc:
                msg = f"{rel}: hash失败 {exc}"
                errors.append(msg)
                logger.warning(msg)
                continue

            manifest[rel] = digest
            if not rebuild and old_manifest.get(rel) == digest:
                files_skipped += 1
                continue

            # 仅 manifest 中已有记录的文件需删 Lance 旧 chunk；首次入库可跳过 delete
            replace_existing = rel in old_manifest
            yield rel, full, digest, replace_existing

            if files_scanned % index_settings.manifest_flush_every == 0:
                _persist_manifest(workspace, manifest, path_prefix=prefix)
                logger.debug("已 checkpoint manifest（%d 文件）", len(manifest))

    files_updated, chunks_written, vector_dim = run_parallel_index_loop(
        workspace,
        work_iter(),
        index_settings=index_settings,
        use_ast=use_ast,
        vector_dim=vector_dim,
        embed_profile_batch_size=profile.batch_size,
        errors=errors,
        logger=logger,
        emit=emit,
        progress=progress,
    )

    if progress is not None:
        progress.update_scan(files_scanned, files_skipped)
        progress.update_embed(files_updated, chunks_written)

    if scan_limit is not None and files_scanned >= scan_limit:
        hit_file_cap = True
        emit(
            f"已达 max_files={scan_limit} 上限；请调大 index.max_files 或 --path 分批",
            level=30,
        )

    _persist_manifest(workspace, manifest, path_prefix=prefix)
    if prefix in (".", "") and not hit_file_cap:
        removed = delete_chunks_not_in_files(workspace, set(manifest.keys()))
        if removed:
            emit(f"已清理 {removed} 个已从磁盘删除的文件的 chunk")

    touch_index_meta(
        workspace,
        vector_dim=vector_dim,
        sync_files=len(manifest),
        sync_complete=not hit_file_cap,
    )
    emit(
        f"同步结束: 遍历 {files_scanned} 文件, 写入/更新 {files_updated}, "
        f"跳过 {files_skipped}, chunks {chunks_written}"
    )

    if errors:
        emit(f"错误 {len(errors)} 个（见日志）", level=30)
        for err in errors[:10]:
            logger.warning("  %s", err)

    return IndexRunResult(
        files_scanned=files_scanned,
        files_updated=files_updated,
        files_skipped=files_skipped,
        chunks_written=chunks_written,
        errors=errors,
        hit_file_cap=hit_file_cap,
    )


def run_index_paths(
    workspace: Path,
    rel_paths: list[str],
    *,
    on_progress: Callable[[str], None] | None = None,
    progress: IndexProgressDisplay | None = None,
) -> IndexRunResult:
    """
    仅索引指定相对路径（watch 增量用）。

    @param workspace 工作区根
    @param rel_paths 变更的相对路径列表
    @param on_progress 进度回调
    @return IndexRunResult
    """
    logger = get_index_logger()
    index_settings = resolve_index_settings(workspace)
    profile = resolve_embedding_profile(workspace)
    skip_dirs = frozenset(index_settings.skip_dirs)

    def emit(msg: str, level: int = 20) -> None:
        logger.log(level, msg)
        if progress is not None:
            if level >= 30:
                progress.note(msg)
            elif not progress.enabled:
                progress.emit_fallback(msg, level=level)
        elif on_progress:
            on_progress(msg)

    unique_paths = sorted({p.strip().lstrip("/") for p in rel_paths if p.strip()})
    if not unique_paths:
        return IndexRunResult(
            files_scanned=0,
            files_updated=0,
            files_skipped=0,
            chunks_written=0,
        )

    old_manifest = load_manifest(workspace)
    manifest: dict[str, str] = dict(old_manifest)
    errors: list[str] = []
    files_scanned = 0
    files_skipped = 0

    try:
        vector_dim = get_embedding_dimension(workspace)
    except RuntimeError as exc:
        logger.error("Embedding 配置错误: %s", exc)
        raise

    def work_iter() -> Iterator[tuple[str, Path, str, bool]]:
        nonlocal files_scanned, files_skipped
        for rel in unique_paths:
            if skip_dirs and rel.split("/", 1)[0] in skip_dirs:
                continue
            full = workspace / rel
            files_scanned += 1
            if not full.is_file():
                if rel in manifest:
                    delete_chunks_for_file(workspace, rel)
                    del manifest[rel]
                    emit(f"watch 删除索引: {rel}")
                continue
            try:
                digest = file_sha256(full)
            except OSError as exc:
                msg = f"{rel}: hash失败 {exc}"
                errors.append(msg)
                logger.warning(msg)
                continue
            manifest[rel] = digest
            if old_manifest.get(rel) == digest:
                files_skipped += 1
                continue
            replace_existing = rel in old_manifest
            yield rel, full, digest, replace_existing

    files_updated, chunks_written, vector_dim = run_parallel_index_loop(
        workspace,
        work_iter(),
        index_settings=index_settings,
        use_ast=False,
        vector_dim=vector_dim,
        embed_profile_batch_size=profile.batch_size,
        errors=errors,
        logger=logger,
        emit=emit,
        progress=progress,
    )

    save_manifest(workspace, manifest)
    touch_index_meta(
        workspace,
        vector_dim=vector_dim,
        sync_files=len(manifest),
    )
    if files_updated:
        emit(
            f"watch 增量: 处理 {files_scanned} 路径, 更新 {files_updated}, "
            f"chunks {chunks_written}"
        )

    return IndexRunResult(
        files_scanned=files_scanned,
        files_updated=files_updated,
        files_skipped=files_skipped,
        chunks_written=chunks_written,
        errors=errors,
    )

