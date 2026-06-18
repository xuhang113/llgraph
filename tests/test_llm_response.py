"""llm_response 单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

from llgraph.core.llm_response import (
    llm_content_text,
    llm_response_text,
    normalize_stored_llm_text,
)


def test_llm_content_text_extracts_text_blocks_only() -> None:
    content = [
        {"type": "thinking", "thinking": "internal only"},
        {"type": "text", "text": "# 报告\n\n正文"},
    ]
    assert llm_content_text(content) == "# 报告\n\n正文"
    assert llm_content_text(content, fallback_thinking=True).startswith("# 报告")


def test_llm_response_text_from_message() -> None:
    msg = SimpleNamespace(content=[{"type": "text", "text": "hello"}])
    assert llm_response_text(msg) == "hello"


def test_normalize_stored_repr_list() -> None:
    dirty = str(
        [
            {"signature": None, "thinking": "hidden", "type": "thinking"},
            {"type": "text", "text": "# Title\n\nbody"},
        ]
    )
    assert normalize_stored_llm_text(dirty) == "# Title\n\nbody"
