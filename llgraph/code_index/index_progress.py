"""索引同步进度条（终端单行刷新 + Web 可轮询落盘）。"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgraph.code_index.index_settings import IndexSettings, resolve_index_settings
from llgraph.code_index.paths import ensure_index_dirs, live_progress_path


@dataclass
class IndexProgressSnapshot:
    """进度快照。"""

    phase: str
    files_scanned: int = 0
    files_skipped: int = 0
    files_updated: int = 0
    chunks_written: int = 0
    files_total: int | None = None


def resolve_show_progress(workspace: Path, *, quiet: bool = False) -> bool:
    """
    是否展示终端同步进度条。

    @param workspace 工作区根
    @param quiet CLI -q
    @return 是否启用
    """
    if quiet:
        return False
    settings = resolve_index_settings(workspace)
    return settings.show_progress


def estimate_progress_percent(
    *,
    files_scanned: int = 0,
    files_skipped: int = 0,
    files_updated: int = 0,
    files_total: int | None = None,
    phase: str = "sync",
) -> float | None:
    """
    由计数估算进度百分比（供落盘与 API 读时统一算法）。

    @return 0–99.9 或 None
    """
    return _compute_percent(
        IndexProgressSnapshot(
            phase=phase,
            files_scanned=files_scanned,
            files_skipped=files_skipped,
            files_updated=files_updated,
            files_total=files_total,
        )
    )


def _compute_percent(snap: IndexProgressSnapshot) -> float | None:
    """
    估算进度百分比；无法估算时返回 None（前端用不确定条）。

    以文件遍历为主（增量大量 skip 时也能匀速前进）；若 embed 明显落后于
    待更新数，则略微压低，避免扫完却仍在 embed 时显示 99%。

    @param snap 当前快照
    @return 0–99.9 或 None
    """
    total = snap.files_total
    pending = max(0, snap.files_scanned - snap.files_skipped)
    if total is None or total <= 0:
        if pending > 0 and snap.files_updated > 0:
            return min(99.0, snap.files_updated / pending * 100.0)
        if snap.files_scanned > 0:
            return None
        return 0.0

    scan_pct = min(99.0, snap.files_scanned / total * 100.0)
    if pending <= 0:
        return scan_pct
    if snap.files_updated <= 0:
        # 已扫到待更新文件但尚未写出：略压低，避免虚高
        return min(99.0, scan_pct * 0.92)
    embed_ratio = min(1.0, snap.files_updated / pending)
    if embed_ratio >= 0.95:
        # embed 跟上扫描：百分比跟扫描走（解决「卡在 82%」假死感）
        return scan_pct
    # embed 落后：在扫描进度上按落后程度打折
    return min(99.0, scan_pct * (0.7 + 0.3 * embed_ratio))


def write_live_progress(workspace: Path, payload: dict[str, Any]) -> None:
    """
    写入 live_progress.json。

    @param workspace 工作区根
    @param payload 进度字典
    """
    ensure_index_dirs(workspace)
    path = live_progress_path(workspace)
    data = dict(payload)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def read_live_progress(workspace: Path) -> dict[str, Any] | None:
    """
    读取 live_progress.json。

    @param workspace 工作区根
    @return 进度字典或 None
    """
    path = live_progress_path(workspace)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def clear_live_progress(workspace: Path) -> None:
    """
    删除进度文件。

    @param workspace 工作区根
    """
    path = live_progress_path(workspace)
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


class IndexProgressDisplay:
    """
    不确定总长度的同步进度条：滑动条 + 扫描/索引计数。

    非 TTY 时退化为普通换行输出；若传入 workspace 则始终落盘供 Web 轮询。
    """

    BAR_WIDTH = 28

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_interval: float = 0.2,
        workspace: Path | None = None,
        action: str | None = None,
        files_total: int | None = None,
    ) -> None:
        self._tty = sys.stderr.isatty()
        self.enabled = enabled and self._tty
        self.min_interval = min_interval
        self._workspace = workspace.expanduser().resolve() if workspace else None
        self._action = action
        self._started = time.monotonic()
        self._last_draw = 0.0
        self._last_persist = 0.0
        self._tick = 0
        self._running = True
        self._error: str | None = None
        self._snapshot = IndexProgressSnapshot(phase="prepare", files_total=files_total)
        self._persist(force=True)

    def set_files_total(self, files_total: int | None) -> None:
        """
        设置软总量（用于估算百分比）。

        @param files_total 预计文件数
        """
        self._snapshot.files_total = files_total
        self._persist(force=True)

    def set_phase(self, phase: str) -> None:
        """
        切换阶段。

        @param phase prepare | scan | sync | embed | done
        """
        self._snapshot.phase = phase
        self._draw(force=True)

    def update_scan(self, files_scanned: int, files_skipped: int) -> None:
        """
        更新扫描进度。

        @param files_scanned 已遍历文件数
        @param files_skipped 未变更跳过数
        """
        if self._snapshot.phase == "prepare":
            self._snapshot.phase = "sync"
        self._snapshot.files_scanned = files_scanned
        self._snapshot.files_skipped = files_skipped
        self._draw()

    def update_embed(self, files_updated: int, chunks_written: int) -> None:
        """
        更新切块/embed 进度。

        @param files_updated 已写入文件数
        @param chunks_written 已写入 chunk 数
        """
        if self._snapshot.phase == "prepare":
            self._snapshot.phase = "sync"
        self._snapshot.files_updated = files_updated
        self._snapshot.chunks_written = chunks_written
        self._draw()

    def note(self, msg: str) -> None:
        """
        输出重要信息（换行，不覆盖进度条）。

        @param msg 文本
        """
        if self.enabled:
            self._clear_line()
        print(msg, flush=True)

    def finish(
        self,
        *,
        files_scanned: int,
        files_updated: int,
        files_skipped: int,
        chunks_written: int,
        ok: bool = True,
        error: str | None = None,
        emit_summary: bool = True,
    ) -> None:
        """
        结束进度条并打印汇总行。

        @param files_scanned 扫描文件数
        @param files_updated 更新文件数
        @param files_skipped 跳过文件数
        @param chunks_written 写入 chunk 数
        @param ok 是否成功结束
        @param error 失败信息
        @param emit_summary 是否打印终端汇总行
        """
        self._snapshot.phase = "done"
        self._snapshot.files_scanned = files_scanned
        self._snapshot.files_updated = files_updated
        self._snapshot.files_skipped = files_skipped
        self._snapshot.chunks_written = chunks_written
        self._running = False
        self._error = error
        if self.enabled:
            self._clear_line()
        if emit_summary:
            label = "同步完成" if ok else "同步结束"
            print(
                f"{label}: 扫描 {files_scanned} | 更新 {files_updated} | "
                f"跳过 {files_skipped} | chunks {chunks_written}",
                flush=True,
            )
        self._persist(
            force=True,
            extra={"ok": ok, "percent": 100.0 if ok else _compute_percent(self._snapshot)},
        )

    def emit_fallback(self, msg: str, *, level: int = 20) -> None:
        """
        非 TTY 时的进度文本输出。

        @param msg 消息
        @param level 日志级别（>=30 始终输出）
        """
        if self.enabled:
            return
        if level >= 30:
            print(msg, flush=True)

    def _payload(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        snap = self._snapshot
        payload: dict[str, Any] = {
            "running": self._running,
            "action": self._action,
            "phase": snap.phase,
            "files_scanned": snap.files_scanned,
            "files_skipped": snap.files_skipped,
            "files_updated": snap.files_updated,
            "chunks_written": snap.chunks_written,
            "files_total": snap.files_total,
            "percent": 100.0 if (not self._running and self._error is None) else _compute_percent(snap),
            "elapsed_sec": round(time.monotonic() - self._started, 1),
            "error": self._error,
        }
        if extra:
            payload.update(extra)
        return payload

    def _persist(self, *, force: bool = False, extra: dict[str, Any] | None = None) -> None:
        if self._workspace is None:
            return
        now = time.monotonic()
        if not force and now - self._last_persist < self.min_interval:
            return
        self._last_persist = now
        write_live_progress(self._workspace, self._payload(extra=extra))

    def _draw(self, force: bool = False) -> None:
        self._persist(force=force)
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_draw < self.min_interval:
            return
        self._last_draw = now
        self._tick += 1

        snap = self._snapshot
        pos = self._tick % (self.BAR_WIDTH + 10)
        bar_chars = ["-"] * self.BAR_WIDTH
        for offset in range(7):
            idx = pos - offset
            if 0 <= idx < self.BAR_WIDTH:
                bar_chars[idx] = "#"
        bar = "[" + "".join(bar_chars) + "]"

        if snap.phase == "prepare":
            detail = "准备中…"
        else:
            detail = (
                f"扫描 {snap.files_scanned}（跳过 {snap.files_skipped}）"
                f" | 索引 {snap.files_updated} / {snap.chunks_written} chunks"
            )

        line = f"\r同步 {bar} {detail}"
        padding = max(0, 100 - len(line) + 1)
        sys.stderr.write(line + " " * padding)
        sys.stderr.flush()

    def _clear_line(self) -> None:
        sys.stderr.write("\r" + " " * 100 + "\r")
        sys.stderr.flush()


def should_update_scan(files_scanned: int, settings: IndexSettings) -> bool:
    """
    是否刷新扫描进度。

    @param files_scanned 当前扫描数
    @param settings 索引配置
    @return 是否刷新
    """
    if files_scanned <= 1:
        return True
    return files_scanned % settings.progress_log_every == 0
