"""被短指针引用的历史工具结果：不许被裁剪 / 压缩掉。

llgraph 的短路径拦截（重复工具、跨轮重复读）返回的是「你已经在
tool_call_id=X 拿到过这个结果」。这类指针只有在 X 的全文**还在上下文里**时才成立；
一旦 X 被 `tool_prune` 掩码或被出站压缩成指针，模型就同时失去了正文与重新取正文的
入口——省下的 token 换成了丢上下文。

本模块把「谁被引用了」算出来，供裁剪与出站压缩当作钉子。两条硬约束：

- 只钉**还是全文**的那些结果。已经压过的不许因为一条晚到的引用而复活：
  `dispatch_compaction` 的前缀单调性靠这一点吃饭，复活一次等于击穿 prompt cache。
- 有条数上限。指针只增不减，无上限地钉住等于把出站预算作废。
"""

from __future__ import annotations

import re

from langchain_core.messages import BaseMessage, ToolMessage

_REF_RE = re.compile(r"tool_call_id=([A-Za-z0-9_\-:.]+)")

# 只有 llgraph 自己产生的短指针才算「引用」，模型正文里出现的 id 不算
_POINTER_MARKERS = (
    "[llgraph] 重复工具已拦截",
    "[llgraph] 重复失败已拦截",
    "[llgraph] 跨轮重复读已拦截",
)


def _tool_text(msg: ToolMessage) -> str:
    raw = msg.content
    return raw if isinstance(raw, str) else str(raw or "")


def is_llgraph_pointer(content: str) -> bool:
    """@param content 工具正文 @return 是否 llgraph 短指针文案"""
    text = content.lstrip()
    return any(text.startswith(marker) for marker in _POINTER_MARKERS)


def _already_archived(content: str) -> bool:
    """@param content 工具正文 @return 是否已被掩码/落盘（钉住也拿不回正文）"""
    text = content.lstrip()
    return text.startswith(("[历史", "[工具结果已落盘"))


def referenced_tool_call_ids(messages: list[BaseMessage]) -> set[str]:
    """
    收集被 llgraph 短指针引用过的 tool_call_id。

    @param messages 消息列表
    @return 被引用的 id 集合
    """
    refs: set[str] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = _tool_text(msg)
        if not is_llgraph_pointer(content):
            continue
        refs.update(_REF_RE.findall(content))
    return refs


def pinned_referenced_tool_indices(
    messages: list[BaseMessage],
    *,
    cap: int,
    exclude_ids: frozenset[str] | set[str] = frozenset(),
    is_full_text: bool = True,
) -> set[int]:
    """
    被短指针引用、且仍是全文的 ToolMessage 下标。

    @param messages 消息列表
    @param cap 最多钉住的条数（<=0 关闭）
    @param exclude_ids 已被压缩过的 tool_call_id（不许复活）
    @param is_full_text 是否只钉全文条目（出站/裁剪都应为 True）
    @return 下标集合（按出现顺序取最近 cap 条）
    """
    if cap <= 0:
        return set()
    refs = referenced_tool_call_ids(messages)
    if not refs:
        return set()
    hits: list[int] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        cid = str(getattr(msg, "tool_call_id", "") or "").strip()
        if not cid or cid not in refs or cid in exclude_ids:
            continue
        content = _tool_text(msg)
        if is_full_text and (is_llgraph_pointer(content) or _already_archived(content)):
            continue
        hits.append(idx)
    return set(hits[-cap:])
