"""助手回合结构化内容（Markdown 原文；终端 / Web 分层消费）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONTENT_FORMAT_MARKDOWN = "markdown"


@dataclass
class AssistantTurnContent:
    """
    单轮助手输出载荷。

    - Web / API：取 markdown / raw_markdown，由前端或网关渲染
    - 终端：trace 层对 markdown 做 Rich/ANSI 展示，不修改本对象
    """

    markdown: str
    """展示用 Markdown（已去掉 survey 块等终端向导接管的片段）。"""

    raw_markdown: str = ""
    """模型原始 Markdown（含 survey 块，供解析与落盘）。"""

    format: str = CONTENT_FORMAT_MARKDOWN
    """内容格式标识，供 Web Content-Type / 渲染器选择。"""

    tool_names: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        """
        序列化为 Web/API 友好 dict（不含终端 ANSI / Rich 产物）。

        @return 含 format、markdown、raw_markdown 等字段
        """
        return {
            "format": self.format,
            "markdown": self.markdown,
            "raw_markdown": self.raw_markdown or self.markdown,
            "tool_names": list(self.tool_names),
            "duration_sec": self.duration_sec,
        }


def build_assistant_turn_content(
    *,
    markdown: str,
    raw_markdown: str = "",
    tool_names: list[str] | None = None,
    duration_sec: float = 0.0,
) -> AssistantTurnContent:
    """
    构建助手回合内容（始终保留 Markdown 原文）。

    @param markdown 展示用 Markdown
    @param raw_markdown 原始 Markdown，默认同 markdown
    @param tool_names 本轮工具名
    @param duration_sec 耗时秒
    @return AssistantTurnContent
    """
    raw = raw_markdown.strip() or markdown
    return AssistantTurnContent(
        markdown=markdown,
        raw_markdown=raw,
        tool_names=list(tool_names or []),
        duration_sec=duration_sec,
    )
