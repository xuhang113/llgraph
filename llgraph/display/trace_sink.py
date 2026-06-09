"""追踪输出 Sink：终端 stdout 与静默模式。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from llgraph.display.trace_display import TraceStepRecord

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """
    去掉 ANSI 转义。

    @param text 原文
    @return 纯文本
    """
    return _ANSI_RE.sub("", text)


class TraceSink(Protocol):
    """追踪输出接口。"""

    def line(self, text: str) -> None:
        """追加一行日志。"""
        ...

    def stream(self, text: str) -> None:
        """流式追加（最终回复）。"""
        ...

    def stream_end(self) -> None:
        """流式段落结束（换行）。"""
        ...

    def step_added(self, step: TraceStepRecord) -> None:
        """新步骤完成（用于左侧列表）。"""
        ...

    def step_selected(self, step_id: int) -> None:
        """用户选中步骤（展开详情）。"""
        ...


class StdoutTraceSink:
    """经典终端：过程与回复写入 stdout（保留 ANSI 颜色与缩进）。"""

    preserves_ansi: bool = True

    def line(self, text: str) -> None:
        print(text, flush=True)

    def stream(self, text: str) -> None:
        if text:
            print(text, end="", flush=True)

    def stream_end(self) -> None:
        print(flush=True)

    def step_added(self, step: TraceStepRecord) -> None:
        pass

    def step_selected(self, step_id: int) -> None:
        pass


class NullTraceSink:
    """无 UI 时的空实现（静默模式）。"""

    def line(self, text: str) -> None:
        pass

    def stream(self, text: str) -> None:
        pass

    def stream_end(self) -> None:
        pass

    def step_added(self, step: TraceStepRecord) -> None:
        pass

    def step_selected(self, step_id: int) -> None:
        pass
