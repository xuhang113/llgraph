"""出站 tool 链压缩与 read spill 测试。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context.context_settings import ContextSettings
from llgraph.context.context_spill import (
    ContextSpill,
    mask_tool_message_to_dispatch_pointer,
)
from llgraph.context.incremental_context import prune_tool_messages_for_dispatch


def _settings(**overrides: object) -> ContextSettings:
    base = dict(
        max_tokens_estimate=100_000,
        auto_compress_ratio=0.85,
        keep_recent_turns=2,
        keep_recent_token_ratio=0.25,
        compress_model=None,
        session_archive_on_compress=True,
        compress_tool_mask_max_chars=2000,
        read_tool_mask_max_chars=12000,
        tool_result_max_chars=12000,
        read_tool_result_max_chars=36000,
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
        compress_strategy="auto",
        compress_during_react=True,
        compress_summary_chunk_chars=120_000,
        dispatch_tool_chain_compress=True,
        dispatch_keep_full_tool_messages=2,
        dispatch_dedupe_read_paths=True,
        grep_context_lines=5,
        grep_max_inline_chars=48000,
        spill_hit_context_lines=100,
        tool_prune_token_ratio=0.0,
        protect_cited_tool_messages=True,
        max_protected_cited_tool_messages=8,
    )
    base.update(overrides)
    return ContextSettings(**base)


def test_prune_dispatch_keeps_last_two_tool_messages_full() -> None:
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "grep_files", "args": {}, "id": "1"}]),
        ToolMessage(content="grep-old-" + "x" * 5000, tool_call_id="1", name="grep_files"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "2"}]),
        ToolMessage(
            content="--- src/Foo.java (行 1-100 / 共 500 行) ---\n1| code",
            tool_call_id="2",
            name="read_file",
        ),
        AIMessage(content="", tool_calls=[{"name": "grep_files", "args": {}, "id": "3"}]),
        ToolMessage(content="grep-new", tool_call_id="3", name="grep_files"),
    ]
    out = prune_tool_messages_for_dispatch(messages, Path("/tmp/ws"), _settings())
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert len(tools) == 3
    assert "grep-old" not in tools[0].content
    assert "[历史" in tools[0].content
    assert "Foo.java" in tools[1].content
    assert tools[2].content == "grep-new"


def test_mask_read_tool_to_dispatch_pointer() -> None:
    body = "\n".join(f"{i}| line-{i}" for i in range(1, 51))
    msg = ToolMessage(
        content=f"--- pkg/Bar.java (行 10-60 / 共 300 行) ---\n{body}",
        tool_call_id="t1",
        name="read_file",
    )
    out = mask_tool_message_to_dispatch_pointer(msg)
    assert "Bar.java" in out.content
    assert "read_file" in out.content
    assert "--- 开头预览 ---" in out.content
    assert "10| line-10" in out.content
    assert "--- 末尾预览 ---" in out.content
    assert "50| line-50" in out.content


def test_read_under_mask_threshold_not_pruned(tmp_path: Path) -> None:
    from llgraph.context.context_spill import mask_tool_message_content
    from llgraph.context.incremental_context import prune_stale_tool_messages

    ws = tmp_path
    body = "1| " + ("x" * 6400)
    content = f"--- src/PolestarExternalApiInterceptor.java (行 1-100 / 共 200 行) ---\n{body}"
    msg = ToolMessage(content=content, tool_call_id="r1", name="read_file")
    settings = _settings(read_tool_mask_max_chars=12000, keep_recent_tool_messages=0)
    pruned, count = prune_stale_tool_messages([msg], ws, settings)
    assert count == 0
    assert pruned[0].content == content

    masked = mask_tool_message_content(msg, ws, max_chars=12000)
    assert masked.content == content


def test_read_archive_includes_head_tail_preview(tmp_path: Path) -> None:
    from llgraph.context.context_spill import mask_tool_message_content

    ws = tmp_path
    body = "\n".join(f"{i}| code-line-{i}" for i in range(1, 201))
    content = f"--- src/Foo.java (行 1-200 / 共 500 行) ---\n{body}"
    msg = ToolMessage(content=content, tool_call_id="r1", name="read_file")
    masked = mask_tool_message_content(msg, ws, max_chars=1000)
    assert "[历史 read 已归档]" in masked.content
    assert "Foo.java" in masked.content
    assert "--- 开头预览 ---" in masked.content
    assert "1| code-line-1" in masked.content
    assert "--- 末尾预览 ---" in masked.content
    assert "200| code-line-200" in masked.content


def test_read_file_spills_when_over_threshold(tmp_path: Path) -> None:
    from llgraph.context.context_settings import SpillSettings

    spill = ContextSpill(
        workspace=tmp_path,
        session_id="cli-test",
        settings=SpillSettings(
            enabled=True,
            tool_result_max_chars=500,
            read_tool_result_max_chars=500,
            tool_result_preview_lines=5,
            tool_result_preview_head_lines=3,
            spill_dir=".llgraph/context/tool-results",
        spill_exempt_tools=(),
        grep_max_inline_chars=48000,
        spill_hit_context_lines=100,
        ),
    )
    big = "--- a.java (行 1-10 / 共 10 行) ---\n" + ("line\n" * 200)
    out = spill.maybe_spill("read_file", big)
    assert "[工具结果已落盘" in out
    assert spill.spill_count() == 1


def test_read_file_uses_higher_spill_threshold(tmp_path: Path) -> None:
    from llgraph.context.context_settings import SpillSettings

    spill = ContextSpill(
        workspace=tmp_path,
        session_id="cli-test",
        settings=SpillSettings(
            enabled=True,
            tool_result_max_chars=500,
            read_tool_result_max_chars=8000,
            tool_result_preview_lines=5,
            tool_result_preview_head_lines=3,
            spill_dir=".llgraph/context/tool-results",
        spill_exempt_tools=(),
        grep_max_inline_chars=48000,
        spill_hit_context_lines=100,
        ),
    )
    medium = "--- Foo.java (行 1-50 / 共 50 行) ---\n" + ("1| line\n" * 80)
    assert spill.maybe_spill("read_file", medium) == medium
    grep_out = spill.maybe_spill("grep_files", medium)
    assert "[工具结果已落盘" not in grep_out
    assert "Foo.java" in grep_out


def test_read_spill_includes_head_and_tail_preview(tmp_path: Path) -> None:
    from llgraph.context.context_settings import SpillSettings

    spill = ContextSpill(
        workspace=tmp_path,
        session_id="cli-test",
        settings=SpillSettings(
            enabled=True,
            tool_result_max_chars=200,
            read_tool_result_max_chars=200,
            tool_result_preview_lines=3,
            tool_result_preview_head_lines=2,
            spill_dir=".llgraph/context/tool-results",
        spill_exempt_tools=(),
        grep_max_inline_chars=48000,
        spill_hit_context_lines=100,
        ),
    )
    lines = [f"{i}| line-{i}" for i in range(1, 41)]
    big = "--- Bar.java (行 1-40 / 共 40 行) ---\n" + "\n".join(lines)
    out = spill.maybe_spill("read_file", big)
    assert "--- 开头预览 ---" in out
    assert "1| line-1" in out
    assert "--- 末尾预览 ---" in out
    assert "40| line-40" in out


def test_dedupe_read_keeps_non_overlapping_segments() -> None:
    from llgraph.context.incremental_context import dedupe_read_tool_messages_for_dispatch

    old = ToolMessage(
        content="--- src/Foo.java (行 1-50 / 共 200 行) ---\n1| old",
        tool_call_id="1",
        name="read_file",
    )
    new = ToolMessage(
        content="--- src/Foo.java (行 51-100 / 共 200 行) ---\n51| new",
        tool_call_id="2",
        name="read_file",
    )
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
        old,
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "2"}]),
        new,
    ]
    out = dedupe_read_tool_messages_for_dispatch(messages, _settings())
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert "1| old" in tools[0].content
    assert "51| new" in tools[1].content


def test_dedupe_read_replaces_overlapping_segment() -> None:
    from llgraph.context.incremental_context import dedupe_read_tool_messages_for_dispatch

    old = ToolMessage(
        content="--- src/Foo.java (行 1-100 / 共 200 行) ---\n1| old",
        tool_call_id="1",
        name="read_file",
    )
    new = ToolMessage(
        content="--- src/Foo.java (行 50-120 / 共 200 行) ---\n50| new",
        tool_call_id="2",
        name="read_file",
    )
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
        old,
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "2"}]),
        new,
    ]
    out = dedupe_read_tool_messages_for_dispatch(messages, _settings())
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert "[历史 read 已替换]" in tools[0].content
    assert "50| new" in tools[1].content


def test_read_spill_includes_hit_anchor_preview(tmp_path: Path) -> None:
    from langchain_core.messages import ToolMessage

    from llgraph.context.context_settings import SpillSettings
    from llgraph.core.tool_execution_context import set_tool_execution_messages

    src = tmp_path / "src" / "Foo.java"
    src.parent.mkdir(parents=True)
    lines = [f"line {i}" for i in range(1, 601)]
    lines[468] = "    String buildBizId() { return id; }"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")

    spill = ContextSpill(
        workspace=tmp_path,
        session_id="t",
        settings=SpillSettings(
            enabled=True,
            tool_result_max_chars=200,
            read_tool_result_max_chars=200,
            tool_result_preview_lines=3,
            tool_result_preview_head_lines=2,
            spill_dir=".llgraph/context/tool-results",
            spill_exempt_tools=(),
            grep_max_inline_chars=48000,
            spill_hit_context_lines=5,
        ),
    )
    set_tool_execution_messages(
        [
            HumanMessage(content="q"),
            ToolMessage(
                content="--- src/Foo.java:469 ---\n>>> 469| buildBizId",
                tool_call_id="g1",
                name="grep_files",
            ),
        ]
    )
    big = "--- src/Foo.java (行 1-600 / 共 600 行) ---\n" + "\n".join(
        f"{i}| {lines[i - 1]}" for i in range(1, 601)
    )
    out = spill.maybe_spill("read_file", big)
    assert "[工具结果已落盘" in out
    assert "--- 命中区预览" in out
    assert "buildBizId" in out
    set_tool_execution_messages(None)


def test_ripgrep_content_with_context(tmp_path: Path) -> None:
    from llgraph.core.ripgrep_search import ripgrep_available, ripgrep_content

    if not ripgrep_available():
        return
    f = tmp_path / "Demo.java"
    f.write_text(
        "line1\nline2\nMATCH_HERE\nline4\nline5\nline6\nline7\n",
        encoding="utf-8",
    )
    hits, err = ripgrep_content(
        tmp_path,
        "MATCH_HERE",
        path_prefix=".",
        context_lines=2,
        limit=5,
    )
    assert err == ""
    assert len(hits) == 1
    assert "Demo.java:3" in hits[0]
    assert ">>>" in hits[0]
    assert "line2" in hits[0]
    assert "line4" in hits[0]


def test_grep_search_tools_never_spill_to_disk(tmp_path: Path) -> None:
    from llgraph.context.context_settings import SpillSettings

    spill = ContextSpill(
        workspace=tmp_path,
        session_id="cli-test",
        settings=SpillSettings(
            enabled=True,
            tool_result_max_chars=500,
            read_tool_result_max_chars=500,
            tool_result_preview_lines=5,
            tool_result_preview_head_lines=3,
            spill_dir=".llgraph/context/tool-results",
            spill_exempt_tools=(),
            grep_max_inline_chars=48000,
            spill_hit_context_lines=100,
        ),
    )
    big = "匹配结果（ripgrep）:\n\n" + (
        "--- src/Foo.java:134 ---\n>>> 134| tableDataList.subList(fromIndex, toIndex)\n"
    ) * 200
    out = spill.maybe_spill("grep_files", big)
    assert "[工具结果已落盘" not in out
    assert spill.spill_count() == 0
    assert "subList" in out


def test_grep_archive_pointer_keeps_hit_blocks() -> None:
    from llgraph.context.context_spill import build_search_tool_archive_pointer

    body = (
        "匹配结果（ripgrep）:\n\n"
        "--- src/A.java:10 ---\n>>> 10| alpha\n"
        "--- src/B.java:134 ---\n>>> 134| tableDataList.subList(fromIndex, toIndex)\n"
    )
    archived = build_search_tool_archive_pointer(body, "grep_files")
    assert archived is not None
    assert "--- 命中预览 ---" in archived
    assert "subList" in archived
    assert "A.java:10" in archived


def test_prune_skipped_when_context_pressure_low() -> None:
    from langchain_core.messages import HumanMessage, ToolMessage

    from llgraph.context.incremental_context import prune_stale_tool_messages

    settings = _settings(tool_prune_token_ratio=0.7)
    long_body = "x" * 5000
    messages = [
        HumanMessage(content="short question"),
        ToolMessage(content=long_body, tool_call_id="1", name="grep_files"),
        ToolMessage(content=long_body, tool_call_id="2", name="grep_files"),
        ToolMessage(content=long_body, tool_call_id="3", name="grep_files"),
    ]
    pruned, count = prune_stale_tool_messages(messages, Path("/tmp/ws"), settings)
    assert count == 0
    assert pruned[1].content == long_body


def test_dispatch_prunes_even_when_context_pressure_low() -> None:
    """出站裁剪不对齐满窗压力：低占用也要把较早 grep 换成指针。"""
    messages = [
        HumanMessage(content="q"),
        ToolMessage(content="grep-old-" + "x" * 5000, tool_call_id="1", name="grep_files"),
        ToolMessage(content="grep-new", tool_call_id="2", name="grep_files"),
    ]
    out = prune_tool_messages_for_dispatch(
        messages,
        Path("/tmp/ws"),
        _settings(tool_prune_token_ratio=0.7, dispatch_keep_full_tool_messages=1),
    )
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert "grep-old" not in tools[0].content
    assert "[历史" in tools[0].content
    assert tools[1].content == "grep-new"


def test_dispatch_pins_latest_write_snapshot_outside_recency() -> None:
    """写入快照在 recency 窗口外仍保留，避免模型拿过期 read 继续改。"""
    snapshot = (
        "已替换 src/Foo.java（1 处）\n"
        "--- src/Foo.java (行 1-2 / 共 2 行) [写入后快照] ---\n"
        "1| new line\n"
        "后续 search_replace 请以此快照为准，勿使用写入前的 read。"
    )
    greps = [
        ToolMessage(content=f"grep-{i}-" + "x" * 2000, tool_call_id=f"g{i}", name="grep_files")
        for i in range(4)
    ]
    messages = [
        HumanMessage(content="改 Foo"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_replace", "args": {"path": "src/Foo.java"}, "id": "w1"}],
        ),
        ToolMessage(content=snapshot, tool_call_id="w1", name="search_replace"),
        *[
            item
            for i, grep in enumerate(greps)
            for item in (
                AIMessage(content="", tool_calls=[{"name": "grep_files", "args": {}, "id": f"g{i}"}]),
                grep,
            )
        ],
    ]
    out = prune_tool_messages_for_dispatch(
        messages,
        Path("/tmp/ws"),
        _settings(dispatch_keep_full_tool_messages=2, tool_prune_token_ratio=0.7),
    )
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert tools[0].content == snapshot
    assert "grep-0" not in tools[1].content
    assert "grep-1" not in tools[2].content
    assert "grep-2-" in tools[3].content
    assert "grep-3-" in tools[4].content


def test_dispatch_compacts_older_write_to_same_path() -> None:
    old_snap = (
        "已替换 src/Foo.java（1 处）\n"
        "--- src/Foo.java (行 1-1 / 共 1 行) [写入后快照] ---\n"
        "1| first"
    )
    new_snap = (
        "已替换 src/Foo.java（1 处）\n"
        "--- src/Foo.java (行 1-1 / 共 1 行) [写入后快照] ---\n"
        "1| second"
    )
    messages = [
        HumanMessage(content="改"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_replace", "args": {"path": "src/Foo.java"}, "id": "w1"}],
        ),
        ToolMessage(content=old_snap, tool_call_id="w1", name="search_replace"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_replace", "args": {"path": "src/Foo.java"}, "id": "w2"}],
        ),
        ToolMessage(content=new_snap, tool_call_id="w2", name="search_replace"),
        ToolMessage(content="grep-a-" + "x" * 2000, tool_call_id="g1", name="grep_files"),
        ToolMessage(content="grep-b-" + "x" * 2000, tool_call_id="g2", name="grep_files"),
        ToolMessage(content="grep-c-" + "x" * 2000, tool_call_id="g3", name="grep_files"),
    ]
    out = prune_tool_messages_for_dispatch(
        messages,
        Path("/tmp/ws"),
        _settings(dispatch_keep_full_tool_messages=2),
    )
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert "[历史 search_replace 已归档]" in tools[0].content
    assert "1| first" not in tools[0].content
    assert tools[1].content == new_snap


def test_mask_write_snapshot_is_not_treated_as_read() -> None:
    from llgraph.context.context_spill import mask_tool_message_to_dispatch_pointer

    snapshot = (
        "已替换 src/Foo.java（1 处）\n"
        "--- src/Foo.java (行 1-2 / 共 2 行) [写入后快照] ---\n"
        "1| new line"
    )
    msg = ToolMessage(content=snapshot, tool_call_id="w1", name="search_replace")
    out = mask_tool_message_to_dispatch_pointer(msg)
    assert "[历史 read" not in out.content
    assert "search_replace" in out.content
    assert "src/Foo.java" in out.content


def test_dispatch_compact_intercept_does_not_consume_keep_slots() -> None:
    intercept = (
        "[llgraph] 重复工具已拦截\n"
        "read_file 已在本问先前工具结果执行过。\n"
        "上次返回摘录:\n1| already"
    )
    messages = [
        HumanMessage(content="q"),
        ToolMessage(content="grep-old-" + "x" * 3000, tool_call_id="1", name="grep_files"),
        ToolMessage(content=intercept, tool_call_id="2", name="read_file"),
        ToolMessage(content="grep-new-" + "y" * 100, tool_call_id="3", name="grep_files"),
    ]
    out = prune_tool_messages_for_dispatch(
        messages,
        Path("/tmp/ws"),
        _settings(dispatch_keep_full_tool_messages=1),
    )
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert "grep-old" not in tools[0].content
    assert tools[1].content == intercept
    assert "grep-new" in tools[2].content


def test_dispatch_protects_cited_grep_outside_recency() -> None:
    cited = "--- src/Foo.java:167 ---\n>>> 167| throw new BizException()\n" + ("x" * 2000)
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "grep_files", "args": {}, "id": "1"}]),
        ToolMessage(content=cited, tool_call_id="1", name="grep_files"),
        AIMessage(content="", tool_calls=[{"name": "grep_files", "args": {}, "id": "2"}]),
        ToolMessage(content="grep-mid-" + "y" * 2000, tool_call_id="2", name="grep_files"),
        AIMessage(content="", tool_calls=[{"name": "grep_files", "args": {}, "id": "3"}]),
        ToolMessage(content="grep-new", tool_call_id="3", name="grep_files"),
        AIMessage(content="结论：Foo.java:167 处 BizException。"),
    ]
    out = prune_tool_messages_for_dispatch(
        messages,
        Path("/tmp/ws"),
        _settings(dispatch_keep_full_tool_messages=1),
    )
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert "BizException" in tools[0].content
    assert "grep-mid" not in tools[1].content
    assert tools[2].content == "grep-new"


