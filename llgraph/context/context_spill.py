"""动态上下文：大工具结果落盘 + 指针（P6）。"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from llgraph.context.context_settings import SpillSettings, resolve_spill_settings

# 单行错误/状态类结果不 spill
_SHORT_RESULT_MAX_CHARS = 280
_ERROR_PREFIXES = (
    "错误",
    "未找到",
    "文件不存在",
    "路径不存在",
    "不是目录",
    "无法",
    "读取失败",
    "MCP 错误",
    "MCP 调用失败",
)


@dataclass
class SpillRecord:
    """单次落盘记录。"""

    tool_name: str
    rel_path: str
    total_chars: int
    total_lines: int
    spilled_at: float


@dataclass
class ContextSpill:
    """
    工具结果落盘管理器。

    超过阈值时将全文写入 .llgraph/context/tool-results/，
    返回给模型的消息仅含路径、行数与末 N 行预览。
    """

    workspace: Path
    session_id: str
    settings: SpillSettings
    disabled: bool = False
    records: list[SpillRecord] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        workspace: Path,
        *,
        session_id: str,
        disabled: bool = False,
    ) -> ContextSpill:
        """
        创建工作区级 spill 实例。

        @param workspace 工作区根
        @param session_id 会话 ID（用于文件名）
        @param disabled 是否禁用（--no-spill）
        @return ContextSpill
        """
        settings = resolve_spill_settings(workspace)
        spill_dir = workspace / settings.spill_dir
        spill_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            workspace=workspace,
            session_id=session_id,
            settings=settings,
            disabled=disabled or not settings.enabled,
        )

    def _should_spill(self, tool_name: str, content: str) -> bool:
        """判断是否应对结果落盘。"""
        if self.disabled or not self.settings.enabled:
            return False
        if tool_name in self.settings.spill_exempt_tools:
            return False
        text = content.strip()
        if not text:
            return False
        if len(text) <= self.settings.tool_result_max_chars:
            return False
        if len(text) <= _SHORT_RESULT_MAX_CHARS:
            return False
        first_line = text.split("\n", 1)[0].strip()
        if any(first_line.startswith(p) for p in _ERROR_PREFIXES):
            return False
        return True

    def _next_spill_path(self, tool_name: str) -> Path:
        """生成唯一落盘路径。"""
        safe_tool = re.sub(r"[^a-zA-Z0-9_-]+", "_", tool_name)[:40]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        filename = f"{stamp}-{safe_tool}-{short_id}.txt"
        return self.workspace / self.settings.spill_dir / filename

    def maybe_spill(self, tool_name: str, content: str) -> str:
        """
        若内容超长则落盘并返回指针消息。

        @param tool_name 工具名
        @param content 原始工具输出
        @return 可能已替换为指针的文本
        """
        if not self._should_spill(tool_name, content):
            return content

        spill_path = self._next_spill_path(tool_name)
        try:
            spill_path.parent.mkdir(parents=True, exist_ok=True)
            spill_path.write_text(content, encoding="utf-8")
        except OSError:
            return content

        lines = content.splitlines()
        total_lines = len(lines) if lines else (1 if content else 0)
        preview_n = self.settings.tool_result_preview_lines
        tail_lines = lines[-preview_n:] if lines else [content[:500]]
        clipped: list[str] = []
        for line in tail_lines:
            if len(line) > 200:
                clipped.append(line[:200] + "…")
            else:
                clipped.append(line)
        preview = "\n".join(clipped)

        rel_path = spill_path.relative_to(self.workspace).as_posix()
        self.records.append(
            SpillRecord(
                tool_name=tool_name,
                rel_path=rel_path,
                total_chars=len(content),
                total_lines=total_lines,
                spilled_at=time.time(),
            )
        )
        return self.format_pointer(
            rel_path=rel_path,
            tool_name=tool_name,
            total_lines=total_lines,
            total_chars=len(content),
            preview=preview,
        )

    @staticmethod
    def format_pointer(
        *,
        rel_path: str,
        tool_name: str,
        total_lines: int,
        total_chars: int,
        preview: str,
    ) -> str:
        """
        生成工具结果指针模板。

        @param rel_path 相对工作区落盘路径
        @param tool_name 工具名
        @param total_lines 总行数
        @param total_chars 总字符数
        @param preview 末 N 行预览
        @return 给模型的指针文本
        """
        return (
            f"[工具结果已落盘 — {tool_name}]\n"
            f"全文路径（相对工作区）: {rel_path}\n"
            f"规模: {total_lines} 行 / {total_chars} 字符\n"
            f"说明: 下文仅为末尾预览；需要全文或指定段落时请 read_file(path, start_line, end_line)，"
            f"或对落盘文件 grep_files（勿重复 read 源码）。\n"
            f"--- 末尾预览 ---\n"
            f"{preview}\n"
            f"--- 预览结束 ---"
        )

    def spilled_bytes_on_disk(self) -> int:
        """
        统计落盘目录内文件总字节。

        @return 字节数
        """
        spill_root = self.workspace / self.settings.spill_dir
        if not spill_root.is_dir():
            return 0
        total = 0
        try:
            for path in spill_root.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        except OSError:
            pass
        return total

    def spill_count(self) -> int:
        """本会话 spill 次数。"""
        return len(self.records)


def apply_spill_to_tools(tools: list[Any], spill: ContextSpill | None) -> list[Any]:
    """
    为 LangChain 工具包装 spill 逻辑。

    @param tools 原始工具列表
    @param spill spill 实例
    @return 包装后的工具列表
    """
    if spill is None or spill.disabled:
        return tools

    wrapped: list[Any] = []
    for tool in tools:
        name = getattr(tool, "name", "tool")
        if hasattr(tool, "func") and callable(tool.func):
            original = tool.func

            def make_wrapped(fn, tool_name: str):
                def _wrapped(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    if isinstance(result, str):
                        return spill.maybe_spill(tool_name, result)
                    return result

                return _wrapped

            try:
                from langchain_core.tools import StructuredTool

                wrapped.append(
                    StructuredTool.from_function(
                        func=make_wrapped(original, name),
                        name=name,
                        description=getattr(tool, "description", "") or "",
                        args_schema=getattr(tool, "args_schema", None),
                    )
                )
                continue
            except (TypeError, ValueError):
                pass
        wrapped.append(tool)
    return wrapped


def mask_tool_message_content(
    msg: ToolMessage,
    workspace: Path,
    *,
    max_chars: int,
) -> ToolMessage:
    """
    将超长 ToolMessage 替换为简短指针（保留 tool_call_id）。

    @param msg 工具消息
    @param workspace 工作区根
    @param max_chars 超过此长度则替换
    @return 原样或掩码后的 ToolMessage
    """
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    if len(content) <= max_chars:
        return msg
    if "[历史工具输出已归档]" in content or "[历史工具输出已省略" in content:
        return msg
    path_match = re.search(r"全文路径[^:]*:\s*(\S+)", content)
    if path_match:
        rel = path_match.group(1)
        short = (
            f"[历史工具输出已归档] 详见 {rel}；"
            f"需要时用 read_file 或 grep_files 读取。"
        )
    else:
        short = (
            f"[历史工具输出已省略，原长 {len(content)} 字符] "
            f"详见 manifest archive_path 或重新执行工具。"
        )
    return ToolMessage(
        content=short,
        tool_call_id=msg.tool_call_id,
        name=getattr(msg, "name", None),
    )


def compact_tool_messages_for_compress(
    messages: list[BaseMessage],
    workspace: Path,
    *,
    max_chars: int,
) -> list[BaseMessage]:
    """
    压缩前将超长 ToolMessage 替换为简短指针（无全文时仅截断说明）。

    @param messages 消息列表
    @param workspace 工作区根
    @param max_chars 超过此长度则压缩 tool 内容
    @return 新消息列表
    """
    new_messages: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            new_messages.append(msg)
            continue
        new_messages.append(
            mask_tool_message_content(msg, workspace, max_chars=max_chars)
        )
    return new_messages


def format_spill_stats(
    *,
    messages_tokens: int,
    spilled_bytes: int,
    spill_count: int,
    cacheable_prefix_estimate: int,
) -> str:
    """
    格式化 /trace stats 输出。

    @param messages_tokens 当前消息估算 token
    @param spilled_bytes 落盘总字节
    @param spill_count 本会话 spill 次数
    @param cacheable_prefix_estimate 可缓存前缀估算
    @return 多行文本
    """
    return (
        f"messages_tokens≈{messages_tokens}\n"
        f"spilled_bytes_on_disk={spilled_bytes} ({spill_count} 次落盘)\n"
        f"cacheable_prefix_estimate≈{cacheable_prefix_estimate}"
    )
