"""索引过程日志：控制台 + .llgraph/index/logs/ 落盘。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from llgraph.code_index.paths import ensure_index_dirs, index_root

_LOGGER_NAME = "llgraph.index"


def index_log_dir(workspace: Path) -> Path:
    """日志目录：.llgraph/index/logs。"""
    return index_root(workspace) / "logs"


def setup_index_logging(
    workspace: Path,
    *,
    log_file: Path | None = None,
    verbose: bool = True,
) -> Path:
    """
    配置索引专用 logger（文件 + 控制台）。

    @param workspace 工作区根
    @param log_file 指定日志路径；默认 logs/index-YYYYMMDD-HHMMSS.log
    @param verbose 控制台是否输出 DEBUG
    @return 实际日志文件路径
    """
    ensure_index_dirs(workspace)
    log_dir = index_log_dir(workspace)
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"index-{stamp}.log"
    else:
        log_path = log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    latest = log_dir / "latest.log"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_path.name)
    except OSError:
        try:
            latest.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass

    from llgraph.display.log_retention import run_log_retention

    run_log_retention(workspace, quiet=True)

    return log_path


def get_index_logger() -> logging.Logger:
    """获取索引 logger（需先 setup_index_logging）。"""
    return logging.getLogger(_LOGGER_NAME)


def log_index_banner(
    workspace: Path,
    log_path: Path,
    *,
    mode: str,
    path_prefix: str,
    use_ast: bool,
    dry_run: bool,
) -> None:
    """
    记录索引任务开头信息。

    @param workspace 工作区根
    @param log_path 日志文件路径
    @param mode 运行模式说明（full/incremental/rebuild）
    @param path_prefix 索引子目录
    @param use_ast 是否 AST
    @param dry_run 是否试运行
    """
    logger = get_index_logger()
    logger.info("=" * 60)
    logger.info("llgraph 代码索引开始")
    logger.info("workspace: %s", workspace)
    logger.info("log_file: %s", log_path)
    logger.info("mode: %s", mode)
    logger.info("path_prefix: %s", path_prefix)
    logger.info("use_ast: %s", use_ast)
    logger.info("dry_run: %s", dry_run)
    logger.info("=" * 60)
