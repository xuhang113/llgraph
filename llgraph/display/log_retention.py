"""日志保留与过期清理（默认 30 天）。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from llgraph.config.edit_settings import load_agent_config
from llgraph.session.user_storage import workspace_context_dir

DEFAULT_RETENTION_DAYS = 30
_MIN_RETENTION_DAYS = 1
_MAX_RETENTION_DAYS = 3650


@dataclass
class PurgeDirectoryReport:
    """单目录清理结果。"""

    directory: str
    deleted_files: int = 0
    deleted_bytes: int = 0
    errors: int = 0


@dataclass
class PurgeLogsReport:
    """工作区相关日志清理汇总。"""

    retention_days: int
    directories: list[PurgeDirectoryReport] = field(default_factory=list)

    @property
    def total_deleted_files(self) -> int:
        return sum(d.deleted_files for d in self.directories)

    @property
    def total_deleted_bytes(self) -> int:
        return sum(d.deleted_bytes for d in self.directories)


def resolve_retention_days(workspace: Path | None = None) -> int:
    """
    解析日志保留天数（agent.json → logging.retention_days）。

    @param workspace 工作区根
    @return 保留天数
    """
    days = DEFAULT_RETENTION_DAYS
    if workspace is not None:
        cfg = load_agent_config(workspace)
        logging_cfg = cfg.get("logging") if isinstance(cfg.get("logging"), dict) else {}
        raw = logging_cfg.get("retention_days")
        if raw is not None:
            try:
                days = int(raw)
            except (TypeError, ValueError):
                days = DEFAULT_RETENTION_DAYS
    return max(_MIN_RETENTION_DAYS, min(_MAX_RETENTION_DAYS, days))


def workspace_log_directories(workspace: Path) -> list[Path]:
    """
    工作区与用户目录下需定期清理的日志目录。

    @param workspace 工作区根
    @return 目录列表（存在或将被创建的不强制存在）
    """
    root = workspace.expanduser().resolve()
    dirs = [
        root / ".llgraph" / "index" / "logs",
        workspace_context_dir(root) / "logs",
        root / ".llgraph" / "context" / "terminals",
    ]
    shell_cfg = load_agent_config(root).get("shell")
    if isinstance(shell_cfg, dict):
        term_dir = shell_cfg.get("terminal_log_dir")
        if isinstance(term_dir, str) and term_dir.strip():
            custom = root / term_dir.strip().lstrip("/")
            if custom not in dirs:
                dirs.append(custom)
    return dirs


def purge_directory(
    directory: Path,
    retention_days: int,
    *,
    now: float | None = None,
) -> PurgeDirectoryReport:
    """
    删除目录内 mtime 早于保留期的普通文件（不删子目录本身）。

    @param directory 目标目录
    @param retention_days 保留天数
    @param now 当前时间戳（测试用）
    @return 清理报告
    """
    report = PurgeDirectoryReport(directory=str(directory))
    if retention_days < 1 or not directory.is_dir():
        return report

    cutoff = (now if now is not None else time.time()) - retention_days * 86400
    try:
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                size = path.stat().st_size
                path.unlink()
                report.deleted_files += 1
                report.deleted_bytes += size
            except OSError:
                report.errors += 1
    except OSError:
        report.errors += 1
    return report


def purge_workspace_logs(
    workspace: Path,
    *,
    retention_days: int | None = None,
) -> PurgeLogsReport:
    """
    清理当前工作区相关的全部日志目录。

    @param workspace 工作区根
    @param retention_days 保留天数；None 时从配置读取
    @return 汇总报告
    """
    days = retention_days if retention_days is not None else resolve_retention_days(workspace)
    summary = PurgeLogsReport(retention_days=days)
    for log_dir in workspace_log_directories(workspace):
        if log_dir.is_dir():
            summary.directories.append(purge_directory(log_dir, days))
    return summary


def format_purge_report(report: PurgeLogsReport) -> str:
    """
    格式化清理报告。

    @param report 清理汇总
    @return 多行文本
    """
    lines = [
        f"日志保留: {report.retention_days} 天",
        f"已删除文件: {report.total_deleted_files} 个"
        f"（约 {report.total_deleted_bytes / 1024:.1f} KB）",
    ]
    for item in report.directories:
        if item.deleted_files or item.errors:
            lines.append(
                f"  {item.directory}: 删除 {item.deleted_files} 个"
                + (f", 失败 {item.errors}" if item.errors else "")
            )
    return "\n".join(lines)


def run_log_retention(workspace: Path, *, quiet: bool = True) -> PurgeLogsReport:
    """
    执行日志清理（启动、索引后等入口统一调用）。

    @param workspace 工作区根
    @param quiet 为 False 时在 stderr 打印摘要
    @return 清理报告
    """
    report = purge_workspace_logs(workspace)
    if not quiet and report.total_deleted_files:
        import sys

        print(f"[llgraph] {format_purge_report(report)}", file=sys.stderr, flush=True)
    return report
