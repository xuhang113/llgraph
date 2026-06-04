"""索引 CLI 与交互 /index 共用调度。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llgraph.code_index.index_lock import IndexLock
from llgraph.code_index.index_logging import (
    index_log_dir,
    log_index_banner,
    setup_index_logging,
)
from llgraph.code_index.indexer import run_index
from llgraph.code_index.index_progress import IndexProgressDisplay, resolve_show_progress
from llgraph.config.config import load_llgraph_env
from llgraph.code_index.embedder import format_embedding_status
from llgraph.code_index.embedding_config import EMBEDDING_CONFIG_FILENAME
from llgraph.code_index.index_settings import resolve_index_settings
from llgraph.code_index.manifest import load_manifest
from llgraph.code_index.paths import embed_cache_path, manifest_path, meta_path
from llgraph.code_index.store import get_index_status
from llgraph.config.logging_settings import search_log_path


@dataclass
class IndexDispatchResult:
    """索引调度结果。"""

    exit_code: int
    log_path: Path | None = None


def build_index_argument_parser(*, prog: str = "llgraph index") -> argparse.ArgumentParser:
    """
    构建 index 参数解析器（CLI 与 /index 共用）。

    @param prog 程序名（帮助文案）
    @return ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="构建/更新/重建工作区代码向量索引（LanceDB，.llgraph/index/）",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default=None,
        choices=("status", "full", "incremental", "rebuild", "dry-run", "help"),
        help="交互模式子命令；CLI 可省略（默认全量）",
    )
    parser.add_argument(
        "-C",
        "--workspace",
        default=".",
        metavar="DIR",
        help="工作区根目录",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="仅显示索引状态",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量：仅处理 hash 变化的文件",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="重建索引：清理旧 Lance/manifest 后全量重扫",
    )
    parser.add_argument(
        "--clear-embed-cache",
        action="store_true",
        help="与 --rebuild 合用：同时删除 embed_cache.db",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描与统计，不写入",
    )
    parser.add_argument(
        "--path",
        default=".",
        metavar="PREFIX",
        help="只索引该相对子目录",
    )
    parser.add_argument(
        "--ast",
        action="store_true",
        help="使用 tree-sitter AST 切块",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="FILE",
        help="指定日志文件路径",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="控制台少输出（详情仍在日志文件）",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="仅监听文件变更并增量索引（Ctrl+C 退出，不进入 Agent）",
    )
    return parser


def print_index_status(workspace: Path) -> None:
    """
    打印索引状态。

    @param workspace 工作区根
    """
    from llgraph.ui.output import emit_report

    status = get_index_status(workspace)
    log_dir = index_log_dir(workspace)
    latest_log = log_dir / "latest.log"
    lines = [
        f"workspace: {workspace}",
        f"lance: {status.lance_path}",
        f"exists: {status.exists}",
        f"chunks: {status.chunk_count}",
        f"vector_dim: {status.vector_dim}",
        f"last_indexed_at: {status.last_indexed_at or '(无)'}",
    ]
    manifest = load_manifest(workspace)
    lines.append(f"manifest_files: {len(manifest)}")
    idx_cfg = resolve_index_settings(workspace)
    cap = idx_cfg.max_files if idx_cfg.max_files > 0 else "无限制"
    lines.extend(
        [
            f"index.max_files: {cap}  progressive: {idx_cfg.progressive}",
            f"index.use_embed_cache: {idx_cfg.use_embed_cache}",
        ]
    )
    if idx_cfg.skip_dirs:
        lines.append(f"index.skip_dirs: {', '.join(idx_cfg.skip_dirs)}")
    lines.extend(
        [
            f"index.watch: enabled={idx_cfg.watch_enabled} "
            f"with_agent={idx_cfg.watch_with_agent} "
            f"debounce={idx_cfg.watch_debounce_sec}s",
            f"index.show_progress: {idx_cfg.show_progress}",
        ]
    )
    if meta_path(workspace).is_file():
        import json

        try:
            meta = json.loads(meta_path(workspace).read_text(encoding="utf-8"))
            complete = meta.get("sync_complete")
            if complete is not None:
                lines.append(f"sync_complete: {complete}")
        except (OSError, json.JSONDecodeError):
            pass
    lines.extend(
        [
            f"manifest: {manifest_path(workspace)}",
            f"embed_cache: {embed_cache_path(workspace)}",
            f"embedding: {format_embedding_status(workspace)}",
        ]
    )
    ws_cfg = workspace / ".llgraph" / EMBEDDING_CONFIG_FILENAME
    lines.append(
        f"embedding_config: {ws_cfg} {'(存在)' if ws_cfg.is_file() else '(默认/用户级)'}"
    )
    lines.append(f"log_dir: {log_dir}")
    if latest_log.exists():
        lines.append(f"latest_log: {latest_log.resolve()}")
    search_log = search_log_path(workspace)
    if search_log.is_file():
        lines.append(
            f"search_log: {search_log.resolve()}（向量检索审计，/log 控制级别）"
        )
    else:
        lines.append(
            f"search_log: {search_log}（尚无记录；--log-level info 或 /log info 后产生）"
        )
    emit_report("\n".join(lines))


def print_index_interactive_help() -> None:
    """交互会话 /index 用法。"""
    from llgraph.ui.output import emit_block

    emit_block(
        "命令:\n"
        "  /index              查看状态 + 简要说明\n"
        "  /index status       仅查看状态\n"
        "  /index full         全量索引（等同 llgraph index）\n"
        "  /index incremental  增量索引\n"
        "  /index rebuild      强制重建（清 Lance + manifest）\n"
        "  /index dry-run      试运行，不写库\n"
        "  /watch on|off       会话内启停索引监听（同 /watch status）\n"
        "  可选: --path <子目录>  --ast  --clear-embed-cache  -q（-q 关闭进度条）\n"
        "  日志: .llgraph/index/logs/latest.log"
    )


def _apply_action_to_args(args: argparse.Namespace) -> None:
    """将 positional action 映射到 flag。"""
    action = getattr(args, "action", None)
    if not action:
        return
    if action == "status":
        args.status = True
    elif action == "full":
        pass
    elif action == "incremental":
        args.incremental = True
    elif action == "rebuild":
        args.rebuild = True
    elif action == "dry-run":
        args.dry_run = True
    elif action == "help":
        args.status = False


def dispatch_index(
    workspace: Path,
    argv: list[str] | None = None,
    *,
    prog: str = "llgraph index",
    bare_means_status: bool = False,
) -> IndexDispatchResult:
    """
    解析参数并执行索引或打印状态。

    @param workspace 工作区根（已 resolve）
    @param argv 参数列表；空列表时 bare_means_status 决定默认行为
    @param prog argparse 程序名
    @param bare_means_status True 时无参数仅 status（交互 /index）
    @return IndexDispatchResult
    """
    parser = build_index_argument_parser(prog=prog)
    argv = list(argv or [])

    if bare_means_status and not argv:
        print_index_status(workspace)
        return IndexDispatchResult(exit_code=0)

    if "-C" not in argv and "--workspace" not in argv:
        argv = ["-C", str(workspace), *argv]

    args = parser.parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"错误: 不是目录: {workspace}", file=sys.stderr)
        return IndexDispatchResult(exit_code=1)
    _apply_action_to_args(args)

    if args.action == "help":
        print_index_interactive_help()
        return IndexDispatchResult(exit_code=0)

    if args.status:
        print_index_status(workspace)
        return IndexDispatchResult(exit_code=0)

    if args.watch:
        from llgraph.code_index.index_watch import IndexWatchService, attach_watch_shutdown

        service = IndexWatchService(workspace)
        if not service.start():
            return IndexDispatchResult(exit_code=1)
        attach_watch_shutdown(service)
        print("监听中… Ctrl+C 退出", flush=True)
        try:
            import time

            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\n已停止 watch", flush=True)
        finally:
            service.stop()
        return IndexDispatchResult(exit_code=0)

    if args.rebuild and args.incremental:
        print("错误: rebuild 与 incremental 不能同时使用", file=sys.stderr)
        return IndexDispatchResult(exit_code=1)

    try:
        load_llgraph_env()
    except Exception as exc:
        print(f"错误: 加载 Gateway 配置失败: {exc}", file=sys.stderr)
        return IndexDispatchResult(exit_code=1)

    log_path = setup_index_logging(
        workspace,
        log_file=Path(args.log_file) if args.log_file else None,
        verbose=not args.quiet,
    )

    if args.rebuild:
        mode = "rebuild"
    elif args.incremental:
        mode = "incremental"
    else:
        mode = "full"

    log_index_banner(
        workspace,
        log_path,
        mode=mode,
        path_prefix=args.path,
        use_ast=args.ast,
        dry_run=args.dry_run,
    )

    index_lock = IndexLock(workspace)
    if not index_lock.try_acquire():
        print(
            "错误: 索引锁被占用（可能 llgraph Agent 的 index-watch 正在运行）",
            file=sys.stderr,
        )
        return IndexDispatchResult(exit_code=1)

    progress: IndexProgressDisplay | None = None
    try:
        if resolve_show_progress(workspace, quiet=args.quiet):
            progress = IndexProgressDisplay()

        result = run_index(
            workspace,
            incremental=args.incremental,
            rebuild=args.rebuild,
            path_prefix=args.path,
            use_ast=args.ast,
            dry_run=args.dry_run,
            clear_embedding_cache=args.clear_embed_cache,
            progress=progress,
            on_progress=(
                None
                if progress is not None
                else (lambda msg: print(msg, flush=True))
            ),
        )
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        print(f"日志文件: {log_path}", file=sys.stderr)
        index_lock.release()
        return IndexDispatchResult(exit_code=1, log_path=log_path)
    finally:
        index_lock.release()

    print(f"\n日志文件: {log_path}", flush=True)
    if progress is not None:
        progress.finish(
            files_scanned=result.files_scanned,
            files_updated=result.files_updated,
            files_skipped=result.files_skipped,
            chunks_written=result.chunks_written,
            ok=not result.errors,
        )
    else:
        print(
            f"完成: 扫描 {result.files_scanned} 文件, "
            f"更新 {result.files_updated}, 跳过 {result.files_skipped}, "
            f"写入 {result.chunks_written} chunks",
            flush=True,
        )
    if result.errors:
        print(f"错误 {len(result.errors)} 个（完整列表见日志）", file=sys.stderr)
        for err in result.errors[:5]:
            print(f"  {err}", file=sys.stderr)
        code = 2 if result.chunks_written == 0 else 0
        _write_index_execution_log(
            workspace,
            mode=mode,
            result=result,
            log_path=log_path,
            settings=resolve_index_settings(workspace),
        )
        return IndexDispatchResult(exit_code=code, log_path=log_path)

    _write_index_execution_log(
        workspace,
        mode=mode,
        result=result,
        log_path=log_path,
        settings=resolve_index_settings(workspace),
    )
    return IndexDispatchResult(exit_code=0, log_path=log_path)


def _write_index_execution_log(
    workspace: Path,
    *,
    mode: str,
    result: Any,
    log_path: Path,
    settings: Any,
) -> None:
    """索引结束后写入 execution.jsonl 并清理过期日志。"""
    from llgraph.display.execution_log import log_index_event
    from llgraph.display.log_retention import run_log_retention

    log_index_event(
        workspace,
        mode=mode,
        files_scanned=result.files_scanned,
        files_updated=result.files_updated,
        chunks_written=result.chunks_written,
        embed_cache_enabled=bool(getattr(settings, "use_embed_cache", False)),
        log_path=str(log_path),
        error_count=len(result.errors),
    )
    run_log_retention(workspace, quiet=True)
