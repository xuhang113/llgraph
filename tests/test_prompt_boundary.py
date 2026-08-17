"""System prompt 动静边界拆分。"""

from __future__ import annotations

from llgraph.core.prompt_boundary import (
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    split_system_prompt_at_boundary,
)
from llgraph.core.prompt_cache import build_cached_system_message


def test_split_strips_marker() -> None:
    text = f"STATIC\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\nDYNAMIC"
    static, dynamic = split_system_prompt_at_boundary(text)
    assert static == "STATIC"
    assert dynamic == "DYNAMIC"
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY not in static
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY not in dynamic


def test_cached_system_has_two_blocks() -> None:
    text = f"# Doing tasks\n...\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\n# Environment\n..."
    msg = build_cached_system_message(
        stable_text=text,
        cache_control={"type": "ephemeral", "ttl": "5m"},
    )
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert msg.content[0]["cache_control"]["type"] == "ephemeral"
    assert "cache_control" not in msg.content[1]
    assert "# Doing tasks" in msg.content[0]["text"]
    assert "# Environment" in msg.content[1]["text"]
