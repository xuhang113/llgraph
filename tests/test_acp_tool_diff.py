"""编辑器里看得见「这一刀改了什么」：写工具报改动 → 收尾那条带 diff 块。

trace 的步骤里只剩工具返回的那段文本（「已写入 N 字符」），改前正文跑完就没了。
这里钉住新接的那条来源：写工具落地时报一次，入口按 tool_call_id 挂回同一行。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.tool_invoke_timing import wrap_tool_node_with_timing
from llgraph.core.tool_progress import (
    ToolEdit,
    current_tool_call_id,
    current_tool_edit_observer,
    notify_file_edited,
    use_current_tool_call,
    use_tool_edit_observer,
)
from llgraph.core.workspace import WorkspaceContext
from llgraph.editor.acp.sink import AcpTraceSink
from llgraph.editor.acp.updates import (
    MAX_DIFF_CHARS,
    tool_call_diffs,
    tool_call_from_step,
)


class _FakeToolNode:
    """够用的 ToolNode 替身：只有调用包装要接的那两个方法。"""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def _run_one(self, call: dict[str, Any], input_type: Any, tool_runtime: Any) -> str:
        self.seen.append(current_tool_call_id())
        return "ok"

    async def _arun_one(self, call: dict[str, Any], input_type: Any, tool_runtime: Any) -> str:
        self.seen.append(current_tool_call_id())
        return "ok"


def _tool_step(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "tool",
        "title": "执行 search_replace(app.py)",
        "body_lines": ["已替换 app.py（1 处）"],
    }
    base.update(kwargs)
    return base


def _write_tool(root: Path, name: str) -> Any:
    ctx = WorkspaceContext(root, allow_write=True)
    return next(t for t in create_filesystem_tools(ctx) if t.name == name)


def _collect() -> tuple[list[ToolEdit], Any]:
    seen: list[ToolEdit] = []
    return seen, seen.append


# ---- 通知本身 ----


def test_no_observer_is_a_no_op() -> None:
    assert current_tool_edit_observer() is None
    with use_current_tool_call("c1"):
        notify_file_edited("a.py", "old", "new")


def test_edit_carries_the_call_it_belongs_to() -> None:
    seen, observe = _collect()
    with use_tool_edit_observer(observe), use_current_tool_call("c1"):
        notify_file_edited("a.py", "old\n", "new\n")

    assert [(e.tool_call_id, e.path, e.old_text, e.new_text) for e in seen] == [
        ("c1", "a.py", "old\n", "new\n")
    ]


def test_edit_without_a_call_is_dropped() -> None:
    seen, observe = _collect()
    with use_tool_edit_observer(observe):
        notify_file_edited("a.py", "old", "new")

    # 入口按 tool_call_id 认行，没有归属就挂不上去，宁可不报
    assert seen == []


def test_unchanged_write_is_not_reported() -> None:
    seen, observe = _collect()
    with use_tool_edit_observer(observe), use_current_tool_call("c1"):
        notify_file_edited("a.py", "same", "same")
        notify_file_edited("  ", "old", "new")

    assert seen == []


def test_observer_exception_does_not_escape() -> None:
    def boom(edit: ToolEdit) -> None:
        raise RuntimeError("界面炸了不能带崩工具")

    with use_tool_edit_observer(boom), use_current_tool_call("c1"):
        notify_file_edited("a.py", "old", "new")


def test_observer_scope_ends_with_the_block() -> None:
    seen, observe = _collect()
    with use_tool_edit_observer(observe):
        assert current_tool_edit_observer() is observe
    assert current_tool_edit_observer() is None

    with use_current_tool_call("c1"):
        assert current_tool_call_id() == "c1"
    assert current_tool_call_id() == ""


# ---- 「当前是哪次调用」由 ToolNode 的调用包装登记 ----


def test_call_id_is_visible_inside_the_tool() -> None:
    node = _FakeToolNode()
    wrap_tool_node_with_timing(node)

    node._run_one({"id": "c1", "name": "write_file"}, None, None)
    asyncio.run(node._arun_one({"id": "c2", "name": "write_file"}, None, None))

    assert node.seen == ["c1", "c2"]
    # 出了这次调用就不该再认作「正在跑」
    assert current_tool_call_id() == ""


# ---- 写工具：落地之后才报 ----


def test_write_file_reports_the_new_file(tmp_path: Path) -> None:
    seen, observe = _collect()
    tool = _write_tool(tmp_path, "write_file")

    with use_tool_edit_observer(observe), use_current_tool_call("c1"):
        tool.invoke({"path": "new.py", "content": "x = 1\n"})

    assert [(e.path, e.old_text, e.new_text) for e in seen] == [
        ("new.py", "", "x = 1\n")
    ]


def test_search_replace_reports_whole_file_before_and_after(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    seen, observe = _collect()
    tool = _write_tool(tmp_path, "search_replace")

    with use_tool_edit_observer(observe), use_current_tool_call("c1"):
        tool.invoke(
            {"path": "app.py", "old_string": "return 1", "new_string": "return 2"}
        )

    assert len(seen) == 1
    assert seen[0].old_text == "def run():\n    return 1\n"
    assert seen[0].new_text == "def run():\n    return 2\n"


def test_append_file_reports_the_merged_text(tmp_path: Path) -> None:
    (tmp_path / "log.md").write_text("第一行\n", encoding="utf-8")
    seen, observe = _collect()
    tool = _write_tool(tmp_path, "append_file")

    with use_tool_edit_observer(observe), use_current_tool_call("c1"):
        tool.invoke({"path": "log.md", "content": "第二行\n"})

    assert [(e.old_text, e.new_text) for e in seen] == [("第一行\n", "第一行\n第二行\n")]


def test_failed_edit_reports_nothing(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    seen, observe = _collect()
    tool = _write_tool(tmp_path, "search_replace")

    with use_tool_edit_observer(observe), use_current_tool_call("c1"):
        out = str(
            tool.invoke(
                {"path": "app.py", "old_string": "没有这一段", "new_string": "x"}
            )
        )

    assert "未找到" in out
    # 没落地就没有改动可看，编辑器里那一行不该挂一个空 diff
    assert seen == []


def test_rejected_write_reports_nothing(tmp_path: Path) -> None:
    from llgraph.permissions.approval import ApprovalDecision, use_approval_gate

    seen, observe = _collect()
    tool = _write_tool(tmp_path, "write_file")

    with use_approval_gate(
        lambda req: ApprovalDecision(allowed=False, reason="拒绝")
    ), use_tool_edit_observer(observe), use_current_tool_call("c1"):
        tool.invoke({"path": "new.py", "content": "x = 1\n"})

    assert seen == []
    assert not (tmp_path / "new.py").exists()


# ---- 载荷：diff 块 ----


def test_diff_uses_absolute_paths(tmp_path: Path) -> None:
    blocks = tool_call_diffs(
        [{"path": "a.py", "old_text": "old\n", "new_text": "new\n"}], tmp_path
    )
    assert blocks == [
        {
            "type": "diff",
            "path": str(tmp_path / "a.py"),
            "oldText": "old\n",
            "newText": "new\n",
        }
    ]


def test_new_file_diff_has_null_old_text(tmp_path: Path) -> None:
    blocks = tool_call_diffs(
        [{"path": "a.py", "old_text": "", "new_text": "new\n"}], tmp_path
    )
    assert blocks[0]["oldText"] is None


def test_diff_is_skipped_without_a_workspace_root() -> None:
    assert tool_call_diffs([{"path": "a.py", "old_text": "", "new_text": "x"}], None) == []


def test_huge_diff_is_skipped(tmp_path: Path) -> None:
    big = "x" * (MAX_DIFF_CHARS + 1)
    assert tool_call_diffs([{"path": "a.py", "old_text": "", "new_text": big}], tmp_path) == []


def test_diff_ignores_junk_input(tmp_path: Path) -> None:
    assert tool_call_diffs("nope", tmp_path) == []
    assert tool_call_diffs([None, 3, {}], tmp_path) == []
    assert (
        tool_call_diffs([{"path": "a.py", "old_text": "x", "new_text": "x"}], tmp_path)
        == []
    )


def test_diff_comes_before_the_text_output(tmp_path: Path) -> None:
    diffs = tool_call_diffs(
        [{"path": "app.py", "old_text": "old\n", "new_text": "new\n"}], tmp_path
    )
    payload = tool_call_from_step(
        _tool_step(), tool_call_id="t1_call_1", as_update=True, diffs=diffs
    )

    assert payload is not None
    assert [block["type"] for block in payload["content"]] == ["diff", "content"]
    assert payload["content"][1]["content"]["text"] == "已替换 app.py（1 处）"


def test_diff_alone_still_becomes_content(tmp_path: Path) -> None:
    diffs = tool_call_diffs(
        [{"path": "app.py", "old_text": "old\n", "new_text": "new\n"}], tmp_path
    )
    payload = tool_call_from_step(
        _tool_step(body_lines=[]), tool_call_id="t1_call_1", diffs=diffs
    )

    assert payload is not None
    assert [block["type"] for block in payload["content"]] == ["diff"]


def test_step_without_output_or_diff_has_no_content() -> None:
    payload = tool_call_from_step(_tool_step(body_lines=[]), tool_call_id="t1_call_1")
    assert payload is not None
    assert "content" not in payload


# ---- Sink：挂回同一行 ----


def _sink(updates: list[dict[str, Any]], workspace: Path | None) -> AcpTraceSink:
    return AcpTraceSink(updates.append, id_prefix="t1_", workspace=workspace)


def _record(**kwargs: Any) -> Any:
    from llgraph.display.trace_display import TraceStepRecord

    base: dict[str, Any] = {
        "step_id": 1,
        "kind": "tool",
        "title": "执行 search_replace(app.py)",
        "elapsed": 0.1,
        "summary": "1 行输出",
        "body_lines": ["已替换 app.py（1 处）"],
        "tool_call_id": "raw-1",
    }
    base.update(kwargs)
    return TraceStepRecord(**base)


def test_sink_attaches_the_edit_to_the_call_that_made_it(tmp_path: Path) -> None:
    updates: list[dict[str, Any]] = []
    sink = _sink(updates, tmp_path)
    sink.tool_calls_planned(
        [{"id": "raw-1", "name": "search_replace", "title": "执行 search_replace(app.py)"}]
    )
    sink.tool_edited(ToolEdit("raw-1", "app.py", "old\n", "new\n"))
    sink.step_added(_record())

    finished = updates[-1]
    assert finished["sessionUpdate"] == "tool_call_update"
    assert finished["status"] == "completed"
    assert finished["content"][0] == {
        "type": "diff",
        "path": str(tmp_path / "app.py"),
        "oldText": "old\n",
        "newText": "new\n",
    }


def test_sink_keeps_parallel_edits_apart(tmp_path: Path) -> None:
    updates: list[dict[str, Any]] = []
    sink = _sink(updates, tmp_path)
    sink.tool_calls_planned(
        [
            {"id": "raw-1", "name": "write_file", "title": "执行 write_file(a.py)"},
            {"id": "raw-2", "name": "write_file", "title": "执行 write_file(b.py)"},
        ]
    )
    sink.tool_edited(ToolEdit("raw-2", "b.py", "", "b\n"))
    sink.tool_edited(ToolEdit("raw-1", "a.py", "", "a\n"))
    sink.step_added(_record(tool_call_id="raw-1", title="执行 write_file(a.py)"))
    sink.step_added(_record(step_id=2, tool_call_id="raw-2", title="执行 write_file(b.py)"))

    finished = [u for u in updates if u.get("status") == "completed"]
    assert [u["content"][0]["path"] for u in finished] == [
        str(tmp_path / "a.py"),
        str(tmp_path / "b.py"),
    ]


def test_sink_merges_two_edits_of_the_same_file(tmp_path: Path) -> None:
    updates: list[dict[str, Any]] = []
    sink = _sink(updates, tmp_path)
    sink.tool_calls_planned([{"id": "raw-1", "name": "write_file", "title": "执行 write_file(a.py)"}])
    sink.tool_edited(ToolEdit("raw-1", "a.py", "v1\n", "v2\n"))
    sink.tool_edited(ToolEdit("raw-1", "a.py", "v2\n", "v3\n"))
    sink.step_added(_record())

    diffs = [b for b in updates[-1]["content"] if b["type"] == "diff"]
    # 一次调用挂两个同名 diff 会让人以为改了两个文件：合成「最早的改前 + 最后的改后」
    assert len(diffs) == 1
    assert (diffs[0]["oldText"], diffs[0]["newText"]) == ("v1\n", "v3\n")


def test_sink_does_not_resend_an_edit_on_a_later_step(tmp_path: Path) -> None:
    updates: list[dict[str, Any]] = []
    sink = _sink(updates, tmp_path)
    sink.tool_edited(ToolEdit("raw-1", "a.py", "", "a\n"))
    sink.step_added(_record())
    sink.step_added(_record(step_id=2))

    finished = [u for u in updates if u.get("status") == "completed"]
    assert any(b["type"] == "diff" for b in finished[0]["content"])
    assert all(b["type"] != "diff" for b in finished[1]["content"])


def test_sink_ignores_an_edit_from_an_unrelated_call(tmp_path: Path) -> None:
    updates: list[dict[str, Any]] = []
    sink = _sink(updates, tmp_path)
    sink.tool_edited(ToolEdit("raw-9", "a.py", "", "a\n"))
    sink.tool_edited(ToolEdit("", "a.py", "", "a\n"))
    sink.step_added(_record())

    assert all(b["type"] != "diff" for b in updates[-1]["content"])
