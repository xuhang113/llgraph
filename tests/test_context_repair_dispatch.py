"""tool_call_id 修链与出站窗口自动裁剪测试。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import (
    _canonical_tool_call_id,
    rebuild_provider_safe_messages,
)
from llgraph.context.context_compressor import estimate_tokens
from llgraph.context.context_dispatch_window import trim_messages_for_dispatch_window_auto
from llgraph.context.context_settings import ContextSettings, is_auto_compress_strategy


def test_canonical_tool_call_id_matches_kimi_variants() -> None:
    assert _canonical_tool_call_id("functions_read_files_2") == "functions_read_files_2"
    assert _canonical_tool_call_id("functions.read_files:2") == "functions_read_files_2"


def test_rebuild_does_not_patch_missing_read_files_result() -> None:
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_files",
                "args": {"paths": ["a.md"]},
                "id": "functions_read_files_2",
                "type": "tool_call",
            }
        ],
    )
    tool = ToolMessage(
        content="file body",
        tool_call_id="functions.read_files:2",
        name="read_files",
    )
    safe, report = rebuild_provider_safe_messages([HumanMessage(content="hi"), ai, tool])
    assert report.patched_tool_results == 0
    tool_msgs = [m for m in safe if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "未完成" not in str(tool_msgs[0].content)


def test_auto_dispatch_keeps_extra_turn_when_budget_allows() -> None:
    pinned = HumanMessage(content="ignored")
    seg_small = [
        HumanMessage(content="turn-1"),
        AIMessage(content="short reply"),
    ]
    seg_large = [
        HumanMessage(content="turn-2"),
        AIMessage(content="x" * 5000),
    ]
    messages = [seg_small[0], seg_small[1], seg_large[0], seg_large[1]]
    settings = ContextSettings(
        max_tokens_estimate=100_000,
        auto_compress_ratio=0.85,
        keep_recent_turns=2,
        keep_recent_token_ratio=0.25,
        compress_model=None,
        session_archive_on_compress=True,
        compress_retrieval_enabled=True,
        compress_retrieval_top_k=5,
        compress_tool_mask_max_chars=2000,
        tool_result_max_chars=40_000,
        read_tool_result_max_chars=40_000,
        read_file_max_bytes=600_000,
        read_file_max_lines=6000,
        tool_result_preview_lines=40,
        tool_result_preview_head_lines=25,
        spill_dir=".llgraph/context/tool-results",
        spill_enabled=True,
        spill_exempt_tools=(),
        budget_source="model",
        context_model_id="test",
        context_fallback_max_tokens=200_000,
        incremental_tool_prune=True,
        keep_recent_tool_messages=4,
        compress_trigger_max_tokens=None,
        session_history_search_enabled=True,
        session_history_search_top_k=8,
        dispatch_keep_user_turns=0,
        dispatch_min_user_turns=2,
        dispatch_max_user_turns=8,
        dispatch_window_token_ratio=0.35,
        compress_strategy="auto",
        compress_during_react=True,
        compress_summary_chunk_chars=120_000,
        dispatch_tool_chain_compress=True,
        dispatch_keep_full_tool_messages=2,
        dispatch_dedupe_read_paths=True,
        dispatch_min_tool_rounds=12,
        grep_context_lines=5,
        spill_hit_context_lines=100,
    )
    trimmed = trim_messages_for_dispatch_window_auto(
        messages,
        settings=settings,
        estimate_tokens=estimate_tokens,
    )
    humans = [m for m in trimmed if isinstance(m, HumanMessage)]
    assert len(humans) == 2
    assert is_auto_compress_strategy("cursor")
