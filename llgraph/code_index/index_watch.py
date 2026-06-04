"""工作区文件监听 + 增量索引（P5）。"""

from __future__ import annotations

import atexit
import threading
from datetime import datetime, timezone
from pathlib import Path

from llgraph.code_index.index_lock import IndexLock
from llgraph.code_index.index_logging import get_index_logger
from llgraph.code_index.index_settings import resolve_index_settings
from llgraph.code_index.indexer import run_index_paths
from llgraph.core.workspace import WorkspaceContext

_WATCHDOG_MISSING = (
    "未安装 watchdog，无法启用 index watch。"
    "请执行: pip install 'llgraph[watch]'"
)


class IndexWatchService:
    """
    文件变更 debounce 后触发 run_index_paths；生命周期由调用方管理。
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()
        self._settings = resolve_index_settings(self._workspace)
        self._ctx = WorkspaceContext(
            self._workspace,
            allow_write=False,
            extra_skip_dirs=frozenset(self._settings.skip_dirs),
        )
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()
        self._observer = None
        self._running = False
        self._shutting_down = False
        self._last_run_at: str | None = None
        self._last_files_updated = 0
        self._logger = get_index_logger()

    @property
    def active(self) -> bool:
        """是否正在监听。"""
        return self._running

    @property
    def last_run_at(self) -> str | None:
        """上次增量索引完成时间（UTC ISO）。"""
        return self._last_run_at

    @property
    def last_files_updated(self) -> int:
        """上次增量索引更新文件数。"""
        return self._last_files_updated

    def start(self) -> bool:
        """
        启动文件监听。

        @return 是否成功启动
        """
        if self._running:
            return True
        self._shutting_down = False
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            from llgraph.ui.output import emit_warn

            emit_warn(_WATCHDOG_MISSING)
            return False

        lock = IndexLock(self._workspace)
        if not lock.try_acquire():
            from llgraph.ui.output import emit_warn

            emit_warn("[index-watch] 已有全量 llgraph index 在运行，跳过自动监听")
            return False
        self._index_lock = lock

        service = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event) -> None:
                if event.is_directory:
                    return
                src = getattr(event, "src_path", "") or ""
                service._enqueue_path(src)

            def on_moved(self, event) -> None:
                dest = getattr(event, "dest_path", "") or ""
                if dest and not dest.endswith("/"):
                    service._enqueue_path(dest)
                src = getattr(event, "src_path", "") or ""
                if src:
                    service._enqueue_deleted(src)

        self._observer = Observer()
        self._observer.schedule(Handler(), str(self._workspace), recursive=True)
        self._observer.start()
        self._running = True
        debounce = self._settings.watch_debounce_sec
        from llgraph.ui.context import ui_notify

        ui_notify("index-watch", f"已启动 debounce={debounce}s")
        return True

    def stop(self) -> None:
        """停止监听并释放索引锁。"""
        self._shutting_down = True
        if not self._running and not getattr(self, "_index_lock", None):
            return
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        with self._pending_lock:
            self._pending.clear()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=1.5)
            except Exception:
                pass
            self._observer = None
        self._running = False
        lock = getattr(self, "_index_lock", None)
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass
            self._index_lock = None
        from llgraph.ui.context import ui_notify

        ui_notify("index-watch", "已停止")

    def notify_changed(self, rel_path: str) -> None:
        """
        Agent 写文件后通知 watch 队列（与 P7 衔接）。

        @param rel_path 相对工作区路径
        """
        if not self._running:
            return
        self._pending.add(rel_path.strip().lstrip("/"))
        self._schedule_flush()

    def _enqueue_path(self, abs_path: str) -> None:
        """文件系统事件入队。"""
        try:
            rel = Path(abs_path).resolve().relative_to(self._workspace).as_posix()
        except ValueError:
            return
        if self._should_ignore(rel):
            return
        with self._pending_lock:
            self._pending.add(rel)
        self._schedule_flush()

    def _enqueue_deleted(self, abs_path: str) -> None:
        """删除事件入队。"""
        self._enqueue_path(abs_path)

    def _should_ignore(self, rel: str) -> bool:
        """是否忽略该路径。"""
        parts = rel.split("/")
        if not parts:
            return True
        if parts[0] in (".git", ".llgraph", "node_modules", "target", ".venv", "venv"):
            return True
        if self._ctx.should_skip_dir(parts[0]):
            return True
        if parts[0] in self._settings.skip_dirs:
            return True
        return False

    def _schedule_flush(self) -> None:
        """debounce 后执行增量索引。"""
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
            delay = self._settings.watch_debounce_sec
            self._timer = threading.Timer(delay, self._flush_pending)
            self._timer.daemon = True
            self._timer.start()

    def _flush_pending(self) -> None:
        """执行 pending 路径的增量索引。"""
        if self._shutting_down or not self._running:
            return
        with self._pending_lock:
            batch = list(self._pending)
            self._pending.clear()
        if not batch:
            return
        try:
            result = run_index_paths(self._workspace, batch)
            self._last_run_at = datetime.now(timezone.utc).isoformat()
            self._last_files_updated = result.files_updated
        except Exception as exc:
            self._logger.warning("watch 增量索引失败: %s", exc)


def start_index_watch_with_agent(
    workspace: Path,
    *,
    no_watch: bool = False,
) -> IndexWatchService | None:
    """
    按配置随 Agent 启动 watch。

    @param workspace 工作区根
    @param no_watch CLI --no-watch-index
    @return 服务实例或 None
    """
    settings = resolve_index_settings(workspace)
    if no_watch or not settings.watch_enabled or not settings.watch_with_agent:
        return None
    service = IndexWatchService(workspace)
    if not service.start():
        return None
    return service


def attach_watch_shutdown(service: IndexWatchService | None) -> None:
    """
    注册 atexit，确保进程退出时 stop watch。

    @param service IndexWatchService 或 None
    """
    if service is None:
        return
    if getattr(service, "_shutdown_registered", False):
        return
    service._shutdown_registered = True

    def _stop() -> None:
        if getattr(service, "_shutting_down", False):
            return
        service.stop()

    atexit.register(_stop)


def format_watch_status(
    service: IndexWatchService | None,
    workspace: Path,
) -> str:
    """
    格式化 /watch status 输出。

    @param service 当前 watch 服务
    @param workspace 工作区根
    @return 多行文本
    """
    settings = resolve_index_settings(workspace)
    lines = [
        "Index watch",
        "===========",
        (
            f"配置: enabled={settings.watch_enabled}  "
            f"with_agent={settings.watch_with_agent}  "
            f"debounce={settings.watch_debounce_sec}s"
        ),
    ]
    if service is None or not service.active:
        lines.append("状态: 未运行")
        if not settings.watch_enabled:
            lines.append("提示: embedding.json index.watch.enabled=false，/watch on 仍可临时启动")
    else:
        lines.append("状态: 监听中（保存文件后 debounce 增量索引）")
        if service.last_run_at:
            lines.append(
                f"上次增量: {service.last_run_at}（更新 {service.last_files_updated} 个文件）"
            )
        else:
            lines.append("上次增量: （尚无）")
    lines.extend(
        [
            "",
            "命令: /watch on  |  /watch off  |  /watch status",
            "启动参数: --no-watch-index 可禁止随 Agent 自动开启",
        ]
    )
    return "\n".join(lines)


def ensure_index_watch(
    workspace: Path,
    service: IndexWatchService | None,
) -> tuple[IndexWatchService | None, str | None]:
    """
    确保 watch 服务存在并已启动。

    @param workspace 工作区根
    @param service 已有服务或 None
    @return (服务实例, 错误说明)；成功时错误为 None
    """
    if service is not None and service.active:
        return service, None
    if service is None:
        service = IndexWatchService(workspace)
    if service.start():
        attach_watch_shutdown(service)
        return service, None
    return service, "启动失败（见上方 [index-watch] 日志；可能未安装 watchdog 或索引锁被占用）"
