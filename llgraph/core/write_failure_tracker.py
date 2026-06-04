"""写工具失败计数与下一轮上下文提醒。"""

from __future__ import annotations

from llgraph.context.context_session import ContextSession

WRITE_TOOL_NAMES = frozenset({"write_file", "append_file", "search_replace"})

_WRITE_HINT_TEMPLATE = """## 写文件提醒（连续 {n} 次写工具未成功）

- 每次 `write_file` / `append_file` **必须同时提供** `path` 与 `content`；**禁止只传 path**。
- 长文档（预计 >{chunk} 字符）请分块：
  1. `write_file` 写标题与目录骨架（含占位符如 `<!--sec-2-->`）；
  2. 按节 `append_file` 或 `search_replace` 追加，单次 content 建议 <{chunk} 字符。
- 修改已有文件优先 `search_replace`；新建文件用 `write_file`，后续节用 `append_file`。
- 最近一次错误: {detail}
"""


class WriteFailureTracker:
    """跟踪写工具连续失败，向 ContextSession 注入提醒。"""

    def __init__(
        self,
        context_session: ContextSession,
        *,
        failures_before_hint: int = 2,
        chunk_max_chars: int = 8000,
    ) -> None:
        self._session = context_session
        self._threshold = max(1, failures_before_hint)
        self._chunk_max_chars = max(1000, chunk_max_chars)
        self._consecutive_failures = 0

    @property
    def chunk_max_chars(self) -> int:
        """单次 write/append 建议字符上限。"""
        return self._chunk_max_chars

    def note_success(self) -> None:
        """写工具成功后清零失败计数。"""
        self._consecutive_failures = 0
        self._session.write_failure_hint = ""

    def note_failure(self, tool_name: str, detail: str) -> None:
        """
        记录写工具失败。

        @param tool_name 工具名
        @param detail 错误摘要
        """
        self._consecutive_failures += 1
        clipped = detail.strip().replace("\n", " ")[:400]
        if self._consecutive_failures >= self._threshold:
            self._session.write_failure_hint = _WRITE_HINT_TEMPLATE.format(
                n=self._consecutive_failures,
                chunk=self._chunk_max_chars,
                detail=clipped or tool_name,
            )

    def inspect_tool_messages(self, messages: list) -> None:
        """
        根据 ToolMessage 更新失败计数。

        @param messages 本轮 tools 节点消息
        """
        from langchain_core.messages import ToolMessage

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            name = msg.name or ""
            if name not in WRITE_TOOL_NAMES:
                continue
            text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            if _is_write_success(text):
                self.note_success()
            elif _is_write_failure(text):
                self.note_failure(name, text)

    def consume_hint_for_context(self) -> str:
        """
        取出并保留写失败提醒（供 workspace-context 注入）。

        @return 提醒文本或空
        """
        return self._session.write_failure_hint.strip()


def _is_write_success(text: str) -> bool:
    """是否为写工具成功返回。"""
    prefixes = ("已写入", "已追加", "已替换")
    return any(text.startswith(p) for p in prefixes)


def _is_write_failure(text: str) -> bool:
    """是否为写工具失败或校验错误。"""
    lowered = text.lower()
    markers = (
        "错误:",
        "缺少必填",
        "validation error",
        "field required",
        "未找到 old_string",
        "不唯一",
        "文件不存在:",
    )
    if any(m in text or m in lowered for m in markers):
        return True
    return False
