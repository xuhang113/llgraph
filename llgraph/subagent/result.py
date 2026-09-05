"""Subagent 执行结果（父 Agent 只消费摘要级字段）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubagentResult:
    """
    子 Agent 一轮交付。

    @param sub_thread 子会话 thread_id
    @param kind explore | general
    @param sub_id 子会话短 id（如 explore-a1b2 / w1）
    @param summary 回传父侧的摘要正文
    @param status ok | failed | cancelled
    @param files_changed 可写子 Agent 变更路径
    @param raw_text 子图最终可见正文（含 JSON 交付物时）
    @param meta 扩展字段
    """

    sub_thread: str
    kind: str
    sub_id: str
    summary: str
    status: str = "ok"
    files_changed: list[str] = field(default_factory=list)
    raw_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def as_tool_output(self) -> str:
        """供父 Agent tool 返回的紧凑文本。"""
        lines = [
            f"[subagent kind={self.kind} id={self.sub_id} status={self.status}]",
            f"thread: {self.sub_thread}",
        ]
        if self.files_changed:
            lines.append("files_changed: " + ", ".join(self.files_changed[:40]))
        summary = (self.summary or self.raw_text or "").strip()
        if summary:
            lines.append("")
            lines.append(summary)
        return "\n".join(lines)
