"""上下文压缩：阈值、防抖、trace 步骤。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage

from llgraph.context.context_compressor import ContextCompressor, estimate_tokens


def test_should_not_compress_below_threshold(tmp_path: Path) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir(exist_ok=True)
    (llgraph_dir / "agent.json").write_text(
        '{"context": {"compress_strategy": "auto", "auto_compress_ratio": 0.85, "budget_source": "config", "context_fallback_max_tokens": 100000}}',
        encoding="utf-8",
    )
    ws = tmp_path
    c = ContextCompressor(ws, "t1")
    msgs = [HumanMessage(content="x" * 1000)]
    assert not c.should_auto_compress(msgs)


def test_skip_small_compress_span() -> None:
    from langchain_core.messages import HumanMessage

    from llgraph.context.context_compressor import _compress_span_too_small

    assert _compress_span_too_small([HumanMessage(content="hi")])
    assert not _compress_span_too_small([HumanMessage(content="x" * 70_000)])


def test_compress_report_includes_timing(tmp_path: Path) -> None:
    from llgraph.context.context_compressor import CompressReport

    report = CompressReport(
        before_count=10,
        after_count=4,
        before_tokens=100_000,
        after_tokens=40_000,
        elapsed_sec=12.5,
        llm_sec=11.0,
        trigger="react",
    )
    from llgraph.context.context_compressor import format_compress_report

    text = format_compress_report(report)
    assert "12.50s" in text
    assert "LLM 摘要" in text
    assert "react" in text
