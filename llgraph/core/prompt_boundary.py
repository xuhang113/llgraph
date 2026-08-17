"""System prompt 动静边界拆分。"""

from __future__ import annotations

# 之前为稳定/可缓存前缀；之后为会话/环境相关动态段。
SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"


def split_system_prompt_at_boundary(text: str) -> tuple[str, str]:
    """
    按动态边界拆分 system prompt。

    @param text 完整 system 文本
    @return (static_prefix, dynamic_suffix)；无边界时 dynamic 为空
    """
    raw = str(text or "")
    marker = SYSTEM_PROMPT_DYNAMIC_BOUNDARY
    if marker not in raw:
        return raw.strip(), ""
    before, _, after = raw.partition(marker)
    return before.strip(), after.strip()
