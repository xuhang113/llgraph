"""下游引用信号：引用检测、裁剪保护、引用行预览、回捞提示。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.context import incremental_context
from llgraph.context.context_builder import format_history_recall_hint
from llgraph.context.context_citation import (
    build_cited_line_preview,
    cited_line_pairs_for_tool,
    cited_tool_indices,
    extract_reference_anchors,
)
from llgraph.context.context_settings import resolve_context_settings
from llgraph.context.incremental_context import prune_stale_tool_messages


def test_extract_reference_anchors_pathline_and_file() -> None:
    anchors = extract_reference_anchors("见 src/biz/ChartDataController.java:167 抛出 BizException")
    assert "pathline:chartdatacontroller.java:167" in anchors
    assert "file:chartdatacontroller.java" in anchors


def test_extract_reference_anchors_bare_filename() -> None:
    anchors = extract_reference_anchors("logback-spring.xml 里配置了 appender")
    assert "file:logback-spring.xml" in anchors


def test_cited_tool_indices_detects_downstream_citation() -> None:
    messages = [
        HumanMessage(content="查一下"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "grep", "args": {}}]),
        ToolMessage(content="src/Foo.java:167: throw new BizException()", tool_call_id="1", name="grep"),
        AIMessage(content="", tool_calls=[{"id": "2", "name": "read_file", "args": {}}]),
        ToolMessage(content="无关内容 abcdef", tool_call_id="2", name="read_file"),
        AIMessage(content="结论：Foo.java:167 处 BizException 被重新抛出，log.error 不执行。"),
    ]
    cited = cited_tool_indices(messages)
    assert 2 in cited
    assert 4 not in cited


def _four_tool_messages(cited_body: str, stale_body: str) -> list:
    r = "r" * 500
    return [
        HumanMessage(content="查一下"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "grep", "args": {}}]),
        ToolMessage(content=cited_body, tool_call_id="1", name="grep"),  # idx2 被引用
        AIMessage(content="", tool_calls=[{"id": "2", "name": "read_file", "args": {}}]),
        ToolMessage(content=stale_body, tool_call_id="2", name="read_file"),  # idx4 无引用
        AIMessage(content="", tool_calls=[{"id": "3", "name": "read_file", "args": {}}]),
        ToolMessage(content=r + "a", tool_call_id="3", name="read_file"),  # idx6 recency
        AIMessage(content="", tool_calls=[{"id": "4", "name": "read_file", "args": {}}]),
        ToolMessage(content=r + "b", tool_call_id="4", name="read_file"),  # idx8 recency
        AIMessage(content="结论：Foo.java:167 抛 BizException，日志没打出。"),
    ]


def test_prune_protects_cited_tool_message(tmp_path: Path, monkeypatch) -> None:
    ws = _write_agent_json(tmp_path, keep=2)
    settings = resolve_context_settings(ws)
    monkeypatch.setattr(incremental_context, "compute_tool_prune_pressure", lambda *_a, **_k: 0.9)
    cited_body = "src/Foo.java:167: throw new BizException()  " + "x" * 500
    stale_body = "y" * 500
    messages = _four_tool_messages(cited_body, stale_body)
    pruned, count = prune_stale_tool_messages(messages, ws, settings)
    # recency 保 idx6/idx8；idx2 被引用被保护，idx4 无引用被裁剪
    assert cited_body in pruned[2].content
    assert stale_body not in pruned[4].content
    assert count == 1


def test_prune_no_protection_at_full_pressure(tmp_path: Path, monkeypatch) -> None:
    ws = _write_agent_json(tmp_path, keep=2)
    settings = resolve_context_settings(ws)
    monkeypatch.setattr(incremental_context, "compute_tool_prune_pressure", lambda *_a, **_k: 1.0)
    cited_body = "src/Foo.java:167: throw new BizException()  " + "x" * 500
    stale_body = "y" * 500
    messages = _four_tool_messages(cited_body, stale_body)
    pruned, _count = prune_stale_tool_messages(messages, ws, settings)
    # 满窗压力：不再保护被引用项，idx2 也被裁剪
    assert cited_body not in pruned[2].content


def test_cited_line_pairs_and_preview() -> None:
    read_body = "\n".join(
        [
            "--- src/Foo.java (行 160-172 / 共 300 行) ---",
            *[f"   {n}| line {n}" for n in range(160, 173)],
        ]
    )
    messages = [
        HumanMessage(content="查一下"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "read_file", "args": {}}]),
        ToolMessage(content=read_body, tool_call_id="1", name="read_file"),
        AIMessage(content="关键在 Foo.java:167 这一行。"),
    ]
    pairs = cited_line_pairs_for_tool(messages, 2)
    assert ("foo.java", 167) in pairs
    preview = build_cited_line_preview(read_body, pairs, radius=2)
    assert "167| line 167" in preview
    assert "165| line 165" in preview
    assert "160| line 160" not in preview


def _anchor_message() -> HumanMessage:
    return HumanMessage(
        content="<conversation-anchor>\n本轮目标: 排查日志\n详情指针: 见归档",
    )


def test_history_recall_hint_fires_with_anchor_and_recall_query() -> None:
    hist = [_anchor_message(), HumanMessage(content="旧问题")]
    hint = format_history_recall_hint("你之前说的那个结论是什么", hist)
    assert "search_session_history" in hint


def test_history_recall_hint_silent_without_anchor() -> None:
    hist = [HumanMessage(content="旧问题"), AIMessage(content="旧回答")]
    assert format_history_recall_hint("你之前说的那个结论是什么", hist) == ""


def test_history_recall_hint_silent_without_recall_intent() -> None:
    hist = [_anchor_message(), HumanMessage(content="旧问题")]
    assert format_history_recall_hint("再帮我看下这个新文件", hist) == ""


def _write_agent_json(tmp_path: Path, *, keep: int = 2) -> Path:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir(exist_ok=True)
    (llgraph_dir / "agent.json").write_text(
        f'{{"context": {{"budget_source": "config", "max_tokens_estimate": 1000, '
        f'"auto_compress_ratio": 0.85, "incremental_tool_prune": true, '
        f'"keep_recent_tool_messages": {keep}, '
        f'"compress_tool_mask_max_chars": 200, "read_tool_mask_max_chars": 200, '
        f'"tool_prune_token_ratio": 0.0}}}}',
        encoding="utf-8",
    )
    return tmp_path
