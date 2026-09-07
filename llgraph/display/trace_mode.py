"""过程展示档位（/trace 四档）。

单独成模块只为一件事：CLI 参数解析要用 `TraceMode`，但不该为此把
`trace_display` 整条链（langchain_core.messages 等）拉进冷启动。
`trace_display` 仍然 re-export 这三个名字，老 import 路径不变。
"""

from __future__ import annotations

from enum import Enum

DEFAULT_PREVIEW_LINES = 4


class TraceMode(str, Enum):
    """过程展示档位。"""

    ALL = "all"
    """完整过程（规划详情、工具参数与输出，对应截图效果）。"""

    STEPS = "steps"
    """展示步骤（折叠摘要，默认）。"""

    REPLY = "reply"
    """不展示步骤，仅流式输出最终回复。"""

    NONE = "none"
    """都不展示（无过程行，仅最终回复文本）。"""


TRACE_MODE_LABELS: dict[TraceMode, str] = {
    TraceMode.ALL: "完整过程（规划+工具详情）",
    TraceMode.STEPS: "展示步骤（折叠摘要）",
    TraceMode.REPLY: "仅回复（不展示步骤）",
    TraceMode.NONE: "都不展示",
}


def parse_trace_mode(name: str) -> TraceMode | None:
    """
    解析 /trace 参数。

    @param name 模式名或别名
    @return 对应 TraceMode，无法识别时返回 None
    """
    key = name.strip().lower()
    aliases = {
        "all": TraceMode.ALL,
        "full": TraceMode.ALL,
        "完整": TraceMode.ALL,
        "全部": TraceMode.ALL,
        "steps": TraceMode.STEPS,
        "step": TraceMode.STEPS,
        "步骤": TraceMode.STEPS,
        "展示步骤": TraceMode.STEPS,
        "reply": TraceMode.REPLY,
        "off": TraceMode.REPLY,
        "回复": TraceMode.REPLY,
        "仅回复": TraceMode.REPLY,
        "不展示": TraceMode.REPLY,
        "none": TraceMode.NONE,
        "quiet": TraceMode.NONE,
        "静默": TraceMode.NONE,
        "都不展示": TraceMode.NONE,
    }
    return aliases.get(key)
