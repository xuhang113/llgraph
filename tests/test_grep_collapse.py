"""grep 折叠：真实总数 + 过宽命中改文件统计（对齐 Cursor / Claude Code）。"""

from __future__ import annotations

from pathlib import Path

from llgraph.core.grep_collapse import (
    AUTO_COLLAPSE_FILES,
    AUTO_COLLAPSE_MATCHES,
    FileMatchCount,
    format_grep_result,
    format_hit_block,
    normalize_output_mode,
    plan_grep,
)
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.workspace import WorkspaceContext

_TOKEN = "UNIQUE_GREP_COLLAPSE_TOKEN"


def test_normalize_output_mode_aliases() -> None:
    assert normalize_output_mode(None) == "auto"
    assert normalize_output_mode("files_with_matches") == "files"
    assert normalize_output_mode("count_matches") == "count"
    assert normalize_output_mode("CONTENT") == "content"
    assert normalize_output_mode("nope") == "auto"


def test_plan_grep_auto_content_when_small() -> None:
    counts = [FileMatchCount("a.py", 3), FileMatchCount("b.py", 2)]
    plan = plan_grep(counts, output_mode="auto", context_lines=5)
    assert plan.mode == "content"
    assert plan.auto_collapsed is False
    assert plan.total_matches == 5
    assert plan.total_files == 2
    assert plan.needs_content is True


def test_plan_grep_auto_collapses_many_matches() -> None:
    counts = [FileMatchCount(f"f{i}.py", 3) for i in range(AUTO_COLLAPSE_MATCHES)]
    plan = plan_grep(counts, output_mode="auto", context_lines=5)
    assert plan.mode == "files"
    assert plan.auto_collapsed is True
    assert plan.total_matches == AUTO_COLLAPSE_MATCHES * 3
    assert plan.needs_content is True
    assert len(plan.content_paths) <= 6


def test_plan_grep_auto_collapses_many_files() -> None:
    counts = [FileMatchCount(f"f{i}.py", 1) for i in range(AUTO_COLLAPSE_FILES + 1)]
    plan = plan_grep(counts, output_mode="auto", context_lines=0)
    assert plan.mode == "files"
    assert plan.auto_collapsed is True
    assert plan.total_files == AUTO_COLLAPSE_FILES + 1


def test_plan_grep_force_content_does_not_collapse() -> None:
    counts = [FileMatchCount(f"f{i}.py", 4) for i in range(20)]
    plan = plan_grep(counts, output_mode="content", head_limit=40, context_lines=5)
    assert plan.mode == "content"
    assert plan.auto_collapsed is False
    assert plan.total_matches == 80
    assert plan.content_limit == 40


def test_plan_grep_count_has_no_content() -> None:
    counts = [FileMatchCount("a.py", 9)]
    plan = plan_grep(counts, output_mode="count")
    assert plan.mode == "count"
    assert plan.needs_content is False
    assert plan.content_paths == ()


def test_format_hit_block_with_and_without_context() -> None:
    lines = ["alpha", "needle here", "omega"]
    flat = format_hit_block("src/a.py", 2, lines, context_lines=0)
    assert flat == "src/a.py:2: needle here"
    block = format_hit_block("src/a.py", 2, lines, context_lines=1)
    assert "--- src/a.py:2 ---" in block
    assert ">>> 2| needle here" in block
    assert "   1| alpha" in block


def test_format_grep_result_collapsed_mentions_total_and_nudge() -> None:
    counts = [FileMatchCount(f"f{i}.py", 3) for i in range(15)]
    plan = plan_grep(counts, output_mode="auto", context_lines=5)
    text = format_grep_result(
        plan=plan,
        pattern="User",
        path=".",
        file_glob="",
        content_blocks=["--- f0.py:1 ---\n>>> 1| User"],
        context_lines=5,
        engine="python",
    )
    assert "【grep 已折叠】" in text
    assert "共 45 处" in text
    assert "15 个文件" in text
    assert "禁止同一宽 pattern 再全仓 grep" in text
    assert "f0.py" in text


def _grep_tool(tmp_path: Path, monkeypatch):
    class _Ctx:
        grep_context_lines = 5
        grep_context_lines = 5

    monkeypatch.setattr(
        "llgraph.core.filesystem_tools.ripgrep_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "llgraph.context.context_settings.resolve_context_settings",
        lambda _root: _Ctx(),
    )
    ctx = WorkspaceContext(tmp_path)
    tools = create_filesystem_tools(ctx)
    return next(tool for tool in tools if tool.name == "grep_files")


def _write_hits(root: Path, file_count: int, hits_per_file: int) -> None:
    for i in range(file_count):
        body = "\n".join(f"line {j} {_TOKEN} {i}" for j in range(hits_per_file))
        (root / f"mod_{i}.py").write_text(body + "\n", encoding="utf-8")


def test_grep_files_auto_collapses_wide_hit(tmp_path: Path, monkeypatch) -> None:
    _write_hits(tmp_path, file_count=15, hits_per_file=3)
    tool = _grep_tool(tmp_path, monkeypatch)
    out = str(tool.invoke({"pattern": _TOKEN, "path": "."}))
    assert "【grep 已折叠】" in out
    assert "共 45 处" in out
    assert "15 个文件" in out
    assert out.count("--- ") <= 10
    assert "禁止同一宽 pattern 再全仓 grep" in out


def test_grep_files_small_hit_stays_content(tmp_path: Path, monkeypatch) -> None:
    _write_hits(tmp_path, file_count=2, hits_per_file=2)
    tool = _grep_tool(tmp_path, monkeypatch)
    out = str(tool.invoke({"pattern": _TOKEN, "path": "."}))
    assert "【grep 已折叠】" not in out
    assert "匹配结果 4 处" in out
    assert ">>>" in out


def test_grep_files_output_mode_count(tmp_path: Path, monkeypatch) -> None:
    _write_hits(tmp_path, file_count=3, hits_per_file=2)
    tool = _grep_tool(tmp_path, monkeypatch)
    out = str(
        tool.invoke(
            {"pattern": _TOKEN, "path": ".", "output_mode": "count_matches"}
        )
    )
    assert "【count】" in out
    assert "共 6 处" in out
    assert ">>>" not in out


def test_grep_files_force_content_still_reports_true_total(
    tmp_path: Path, monkeypatch
) -> None:
    _write_hits(tmp_path, file_count=15, hits_per_file=3)
    tool = _grep_tool(tmp_path, monkeypatch)
    out = str(
        tool.invoke(
            {
                "pattern": _TOKEN,
                "path": ".",
                "output_mode": "content",
                "head_limit": 10,
            }
        )
    )
    assert "【grep 已折叠】" not in out
    assert "显示 10/45 处" in out
    assert "还有 35 处未列出" in out
