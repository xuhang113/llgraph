"""检索结果格式化（Cursor 风格：路径 + 行号 + 极短摘要）。"""

from __future__ import annotations

from llgraph.code_index.paths import SEARCH_SNIPPET_MAX


def truncate_search_snippet(text: str, *, max_len: int = SEARCH_SNIPPET_MAX) -> str:
    """
    将命中正文压成单行极短摘要。

    @param text 原始 snippet 或 text_preview
    @param max_len 最大字符数
    @return 单行摘要；空输入返回空串
    """
    preview = (text or "").replace("\n", " ").strip()
    if not preview:
        return ""
    if len(preview) > max_len:
        return preview[: max_len - 1] + "…"
    return preview


def format_search_hit(
    rank: int,
    location: str,
    snippet: str,
    *,
    snippet_max: int = SEARCH_SNIPPET_MAX,
) -> str:
    """
    格式化单条命中为单行。

    @param rank 1-based 序号
    @param location 路径:行号（或 path:start-end）
    @param snippet 摘要正文
    @param snippet_max 摘要上限
    @return 如 ``1. repo/Foo.java:42  public void sync()``
    """
    short = truncate_search_snippet(snippet, max_len=snippet_max)
    if short:
        return f"{rank}. {location}  {short}"
    return f"{rank}. {location}"
