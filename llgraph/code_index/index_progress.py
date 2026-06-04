"""索引同步进度条（终端单行刷新，无需预扫 total）。"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from llgraph.code_index.index_settings import IndexSettings, resolve_index_settings


@dataclass
class IndexProgressSnapshot:
    """进度快照。"""

    phase: str
    files_scanned: int = 0
    files_skipped: int = 0
    files_updated: int = 0
    chunks_written: int = 0


def resolve_show_progress(workspace: Path, *, quiet: bool = False) -> bool:
    """
    是否展示同步进度条。

    @param workspace 工作区根
    @param quiet CLI -q
    @return 是否启用
    """
    if quiet:
        return False
    settings = resolve_index_settings(workspace)
    return settings.show_progress


class IndexProgressDisplay:
    """
    不确定总长度的同步进度条：滑动条 + 扫描/索引计数。

    非 TTY 时退化为普通换行输出。
    """

    BAR_WIDTH = 28

    def __init__(self, *, enabled: bool = True, min_interval: float = 0.2) -> None:
        self._tty = sys.stderr.isatty()
        self.enabled = enabled and self._tty
        self.min_interval = min_interval
        self._last_draw = 0.0
        self._tick = 0
        self._snapshot = IndexProgressSnapshot(phase="prepare")

    def set_phase(self, phase: str) -> None:
        """
        切换阶段。

        @param phase prepare | scan | embed | done
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
    ) -> None:
        """
        结束进度条并打印汇总行。

        @param files_scanned 扫描文件数
        @param files_updated 更新文件数
        @param files_skipped 跳过文件数
        @param chunks_written 写入 chunk 数
        @param ok 是否成功结束
        """
        if self.enabled:
            self._clear_line()
        label = "同步完成" if ok else "同步结束"
        print(
            f"{label}: 扫描 {files_scanned} | 更新 {files_updated} | "
            f"跳过 {files_skipped} | chunks {chunks_written}",
            flush=True,
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

    def _draw(self, force: bool = False) -> None:
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
