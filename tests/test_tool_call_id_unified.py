"""tool_call_id 规范化交叉测试。"""

from __future__ import annotations

from llgraph.adapters.inbound.streaming import _normalize_tool_call_id as inbound_normalize
from llgraph.context.chat_history_repair import _canonical_tool_call_id
from llgraph.context.tool_call_id import canonical_tool_call_id


def test_canonical_tool_call_id_matches_kimi_variants() -> None:
    assert canonical_tool_call_id("functions_read_files_2") == "functions_read_files_2"
    assert canonical_tool_call_id("functions.read_files:2") == "functions_read_files_2"


def test_inbound_and_repair_tool_call_id_aligned() -> None:
    variants = [
        "functions.read_files:2",
        "functions_read_files_2",
        "grep_files:65",
    ]
    for raw in variants:
        assert inbound_normalize(raw) == _canonical_tool_call_id(raw)
        assert canonical_tool_call_id(raw) == _canonical_tool_call_id(raw)
