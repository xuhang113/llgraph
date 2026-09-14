"""单文件 / 单命中文件 grep 的真值：rg 只给一个文件参数时默认不打印文件名。

没有 `-H`，`rg --count <file>` 只输出 `3`、`rg -n <file>` 只输出 `12:code`，
两处解析都拿不到路径：
- `ripgrep_count` 解析出空 → `grep_files(path="a/b.py")` 一律回「未找到匹配内容」，
  还附带「禁止同 path 重复 grep」，模型会据此下「这里没有」的错结论；
- `ripgrep_content_in_files` 解析出空 → 命中只落在一个文件时结果里没有任何正文，
  模型必须再补一轮 read_file。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.ripgrep_search import (
    ripgrep_available,
    ripgrep_content,
    ripgrep_content_in_files,
    ripgrep_count,
)
from llgraph.core.workspace import WorkspaceContext

pytestmark = pytest.mark.skipif(not ripgrep_available(), reason="未安装 ripgrep")

REL = "pkg/core/mod.py"


def _seed(ws: Path) -> None:
    target = ws / REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            [
                "def build_client():",
                "    return resolve_gateway_settings()",
                "",
                "def reload_client():",
                "    return resolve_gateway_settings()",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _grep_tool(ws: Path):
    return next(t for t in create_filesystem_tools(WorkspaceContext(ws)) if t.name == "grep_files")


def test_count_on_single_file_path(tmp_path: Path) -> None:
    _seed(tmp_path)
    counts, err = ripgrep_count(tmp_path, "resolve_gateway_settings", path_prefix=REL)
    assert err == ""
    assert counts == [(REL, 2)]


def test_content_in_single_file(tmp_path: Path) -> None:
    _seed(tmp_path)
    blocks, err = ripgrep_content_in_files(
        tmp_path,
        "resolve_gateway_settings",
        [REL],
        limit=40,
        context_lines=2,
        max_per_file=8,
    )
    assert err == ""
    assert blocks
    assert all(REL in block for block in blocks)
    assert not any(str(tmp_path) in block for block in blocks)


def test_content_search_rooted_at_one_file_keeps_path(tmp_path: Path) -> None:
    """`search_code_parallel` 的字面量路走 ripgrep_content，按 `rel:line:snippet` 解析。"""
    _seed(tmp_path)
    hits, err = ripgrep_content(
        tmp_path,
        "resolve_gateway_settings",
        path_prefix=REL,
        limit=10,
    )
    assert err == ""
    assert hits
    assert all(hit.startswith(f"{REL}:") for hit in hits)


def test_grep_tool_on_single_file_reports_hits(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = _grep_tool(tmp_path).invoke(
        {"pattern": "resolve_gateway_settings", "path": REL}
    )
    assert "未找到匹配内容" not in out
    assert "2 处" in out
    assert REL in out


def test_grep_tool_single_hit_file_still_returns_content(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = _grep_tool(tmp_path).invoke(
        {"pattern": "resolve_gateway_settings", "path": "pkg"}
    )
    assert ">>> 2|" in out or ">>> 5|" in out
    assert "显示 0/" not in out


def test_grep_tool_missing_pattern_in_single_file_is_still_empty(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = _grep_tool(tmp_path).invoke({"pattern": "no_such_symbol_here", "path": REL})
    assert "未找到匹配内容" in out
