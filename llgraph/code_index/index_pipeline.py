"""并行切块 + 跨文件批量 embed，提升索引吞吐。"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from llgraph.code_index.chunker import CodeChunk, chunk_file_path
from llgraph.code_index.embedder import embed_texts
from llgraph.code_index.file_scan import file_sha256
from llgraph.code_index.index_settings import IndexSettings
from llgraph.code_index.index_write_session import IndexWriteSession


@dataclass
class PreparedFile:
    """已切块、待写入向量的文件。"""

    rel: str
    digest: str
    chunks: list[CodeChunk]
    replace_existing: bool = False


def prepare_file(
    rel: str,
    full: Path,
    *,
    use_ast: bool,
    digest: str,
    replace_existing: bool,
) -> PreparedFile:
    """
    切块（在线程池中执行，hash 由主线程预先计算）。

    @param rel 相对路径
    @param full 绝对路径
    @param use_ast 是否 AST 切块
    @param digest 文件内容 hash
    @return PreparedFile
    """
    chunks = chunk_file_path(rel, full, use_ast=use_ast)
    return PreparedFile(
        rel=rel,
        digest=digest,
        chunks=chunks,
        replace_existing=replace_existing,
    )


def flush_embed_batch(
    workspace: Path,
    batch: list[PreparedFile],
    *,
    vector_dim: int,
    embed_batch_size: int,
    use_embed_cache: bool,
    write_session: IndexWriteSession,
    errors: list[str],
    logger,
) -> tuple[int, int, int]:
    """
    对一批文件跨文件批量 embed 并写入 Lance。

    @return (chunks_written, files_in_batch, new_vector_dim)
    """
    if not batch:
        return 0, 0, vector_dim

    texts: list[str] = []
    hashes: list[str] = []
    owners: list[tuple[int, int]] = []
    paths_to_delete: set[str] = set()

    for file_idx, prepared in enumerate(batch):
        if prepared.replace_existing:
            paths_to_delete.add(prepared.rel)
        for chunk_idx, chunk in enumerate(prepared.chunks):
            texts.append(chunk.text)
            hashes.append(chunk.content_hash)
            owners.append((file_idx, chunk_idx))

    if not texts:
        for prepared in batch:
            logger.debug("%s: 0 chunks", prepared.rel)
        return 0, len(batch), vector_dim

    new_files = len(batch) - len(paths_to_delete)
    t0 = time.perf_counter()
    if paths_to_delete:
        write_session.delete_rel_paths(paths_to_delete)
    delete_ms = int((time.perf_counter() - t0) * 1000)

    try:
        t1 = time.perf_counter()
        vectors, embed_stats = embed_texts(
            workspace,
            texts,
            hashes,
            batch_size=embed_batch_size,
            use_embed_cache=use_embed_cache,
            return_stats=True,
        )
        embed_ms = int((time.perf_counter() - t1) * 1000)
    except Exception as exc:
        rels = ", ".join(p.rel for p in batch[:3])
        msg = f"批量 embed 失败 ({len(batch)} 文件, 示例 {rels}): {exc}"
        errors.append(msg)
        logger.error(msg)
        return 0, 0, vector_dim

    if vectors and len(vectors[0]) != vector_dim:
        vector_dim = len(vectors[0])
        logger.info("向量维度更新为 %d", vector_dim)

    records_by_file: dict[int, list[dict]] = defaultdict(list)
    all_records: list[dict] = []
    for (file_idx, chunk_idx), vec in zip(owners, vectors):
        chunk = batch[file_idx].chunks[chunk_idx]
        record = {
            "chunk_id": chunk.chunk_id,
            "rel_path": chunk.rel_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "language": chunk.language,
            "symbol": chunk.symbol,
            "content_hash": chunk.content_hash,
            "text_preview": chunk.text_preview,
            "vector": vec,
        }
        records_by_file[file_idx].append(record)
        all_records.append(record)

    written = 0
    files_ok = 0
    try:
        t2 = time.perf_counter()
        written = write_session.add_records(all_records)
        lance_ms = int((time.perf_counter() - t2) * 1000)
        files_ok = len(records_by_file)
        for file_idx, records in records_by_file.items():
            logger.debug("%s: 写入 %d chunks", batch[file_idx].rel, len(records))
        if embed_stats.get("cache_enabled"):
            embed_note = (
                f"embed {embed_ms}ms(缓存命中{embed_stats.get('cache_hits', 0)}"
                f"/{embed_stats.get('total', len(texts))})"
            )
        else:
            embed_note = f"embed {embed_ms}ms(无SQLite缓存)"

        logger.info(
            "批次耗时 %dms: lance删 %dms(%d路径,跳过新文件%d) | %s | "
            "lance写 %dms（%d 文件 %d chunks）",
            delete_ms + embed_ms + lance_ms,
            delete_ms,
            len(paths_to_delete),
            new_files,
            embed_note,
            lance_ms,
            len(batch),
            written,
        )
    except Exception as exc:
        rels = ", ".join(p.rel for p in batch[:3])
        msg = f"Lance 批量写入失败 ({len(batch)} 文件, 示例 {rels}): {exc}"
        errors.append(msg)
        logger.exception(msg)

    return written, files_ok, vector_dim


def run_parallel_index_loop(
    workspace: Path,
    work_iter,
    *,
    index_settings: IndexSettings,
    use_ast: bool,
    vector_dim: int,
    embed_profile_batch_size: int,
    write_session: IndexWriteSession | None = None,
    errors: list[str],
    logger,
    emit,
    progress: IndexProgressDisplay | None = None,
) -> tuple[int, int, int]:
    """
    并行准备 + 跨文件批量 embed。

    @param work_iter 迭代 (rel, full, digest, replace_existing)，仅包含需要索引的文件
    @return (files_updated, chunks_written, vector_dim)
    """
    files_updated = 0
    chunks_written = 0
    pending_batch: list[PreparedFile] = []
    pending_chunks = 0
    accum_limit = index_settings.embed_accumulate_chunks
    workers = index_settings.prepare_workers
    session = write_session or IndexWriteSession(workspace)

    def submit(
        executor: ThreadPoolExecutor,
        rel: str,
        full: Path,
        digest: str,
        replace_existing: bool,
    ) -> Future:
        return executor.submit(
            prepare_file,
            rel,
            full,
            use_ast=use_ast,
            digest=digest,
            replace_existing=replace_existing,
        )

    def drain_future(fut: Future) -> PreparedFile | None:
        try:
            return fut.result()
        except Exception as exc:
            rel_hint = inflight.get(fut, "?")
            msg = f"{rel_hint}: 切块失败 {exc}"
            errors.append(msg)
            logger.exception(msg)
            return None

    def flush_pending() -> None:
        nonlocal pending_batch, pending_chunks, chunks_written, vector_dim, files_updated
        if not pending_batch:
            return
        written, n_files, vector_dim = flush_embed_batch(
            workspace,
            pending_batch,
            vector_dim=vector_dim,
            embed_batch_size=embed_profile_batch_size,
            use_embed_cache=index_settings.use_embed_cache,
            write_session=session,
            errors=errors,
            logger=logger,
        )
        chunks_written += written
        files_updated += n_files
        if progress is not None:
            progress.update_embed(files_updated, chunks_written)
        else:
            emit(
                f"批量 embed 完成: {n_files} 文件, {written} chunks"
                f"（累计 {files_updated} 文件 / {chunks_written} chunks）"
            )
        pending_batch = []
        pending_chunks = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        inflight: dict[Future, str] = {}
        work_exhausted = False

        while True:
            while not work_exhausted and len(inflight) < workers * 2:
                try:
                    rel, full, digest, replace_existing = next(work_iter)
                except StopIteration:
                    work_exhausted = True
                    break
                inflight[submit(executor, rel, full, digest, replace_existing)] = rel

            if not inflight:
                break

            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut, None)
                prepared = drain_future(fut)
                if prepared is None:
                    continue
                if not prepared.chunks:
                    files_updated += 1
                    logger.debug("%s: 0 chunks", prepared.rel)
                    continue

                pending_batch.append(prepared)
                pending_chunks += len(prepared.chunks)
                if pending_chunks >= accum_limit:
                    flush_pending()

        flush_pending()

    return files_updated, chunks_written, vector_dim
