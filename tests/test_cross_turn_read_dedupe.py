"""跨轮重复读拦截：磁盘逐行核对 + 引用钉住 + 兜底放行。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.context.read_content_verify import (
    parse_read_blocks,
    path_content_unchanged,
    resolve_verify_target,
)
from llgraph.context.tool_result_pin import (
    pinned_referenced_tool_indices,
    referenced_tool_call_ids,
)
from llgraph.core.cross_turn_read_guard import CROSS_TURN_READ_MARKER
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.react_limits import resolve_cross_turn_read_dedupe
from llgraph.core.tool_loop_guard import (
    IDENTICAL_BLOCK_MARKER,
    build_history_index,
    compute_blocked_tool_messages,
    is_llgraph_placeholder,
)
from llgraph.core.workspace import WorkspaceContext

REL = "src/service.py"


def _ai(calls: list[dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=calls)


def _call(cid: str, name: str, **args: object) -> dict:
    return {"id": cid, "name": name, "args": args, "type": "tool_call"}


def _module_text(n: int = 120, marker: str = "v1") -> str:
    lines = ['"""service"""', "import os", ""]
    for idx in range(n):
        lines.append(f"def handler_{idx}(payload):")
        lines.append(f"    return payload + {idx}  # {marker}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_module(ws: Path, text: str, rel: str = REL) -> Path:
    target = ws / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _read_tool(ws: Path, name: str = "read_file"):
    return next(t for t in create_filesystem_tools(WorkspaceContext(ws)) if t.name == name)


def _read_output(ws: Path, rel: str = REL, **kwargs: object) -> str:
    return _read_tool(ws).func(path=rel, **kwargs)


def _two_turn_messages(ws: Path, *, cid: str = "r1", rel: str = REL) -> list[BaseMessage]:
    """第一轮读过 rel；用户又追问一句；第二轮模型准备原样再读。"""
    body = _read_output(ws, rel, start_line=1, end_line=90)
    return [
        HumanMessage(content="改一下 handler_3"),
        _ai([_call(cid, "read_file", path=rel, start_line=1, end_line=90)]),
        ToolMessage(content=body, tool_call_id=cid, name="read_file"),
        AIMessage(content="已看过实现"),
        HumanMessage(content="顺手把 handler_4 也改掉"),
    ]


def _blocked(
    messages: list[BaseMessage],
    calls: list[dict],
    ws: Path,
    *,
    thread_id: str | None = "t-cross",
) -> dict[str, ToolMessage]:
    return compute_blocked_tool_messages(
        list(messages) + [_ai(calls)],
        calls,
        workspace=ws,
        thread_id=thread_id,
        cross_turn_reads=True,
    )


# --- 磁盘核对 ---------------------------------------------------------------


def test_parse_read_blocks_from_real_read_output(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    blocks = parse_read_blocks(_read_output(tmp_path, start_line=5, end_line=40))
    assert len(blocks) == 1
    block = blocks[0]
    assert block.path == REL
    assert (block.start, block.end) == (5, 40)
    assert block.lines[0][0] == 5
    assert block.covers(5, 40) and block.covers(10, 20)
    assert not block.covers(1, 40) and not block.covers(5, 41)
    assert not block.covers(5, 0)  # 请求读到末尾，旧结果只覆盖到 40 行


def test_unchanged_file_verifies_changed_file_does_not(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    blocks = parse_read_blocks(_read_output(tmp_path, start_line=1, end_line=60))
    assert path_content_unchanged(REL, blocks, tmp_path)

    _write_module(tmp_path, _module_text(marker="v2"))
    assert not path_content_unchanged(REL, blocks, tmp_path)


def test_appended_tail_counts_as_changed(tmp_path: Path) -> None:
    """覆盖行没变但文件变长：读到末尾的旧结果已经不完整，必须算变了。"""
    text = _module_text()
    _write_module(tmp_path, text)
    blocks = parse_read_blocks(_read_output(tmp_path, start_line=1, end_line=30))
    assert path_content_unchanged(REL, blocks, tmp_path)

    _write_module(tmp_path, text + "\ndef tail_handler():\n    return 1\n")
    assert not path_content_unchanged(REL, blocks, tmp_path)


def test_verify_target_rejects_escape_and_absolute(tmp_path: Path) -> None:
    assert resolve_verify_target(REL, tmp_path) == (tmp_path / REL).resolve()
    assert resolve_verify_target("../outside.py", tmp_path) is None
    assert resolve_verify_target("/etc/passwd", tmp_path) is None
    assert resolve_verify_target("~/.llgraph/skills/x.md", tmp_path) is None


def test_missing_file_does_not_verify(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    blocks = parse_read_blocks(_read_output(tmp_path, start_line=1, end_line=30))
    (tmp_path / REL).unlink()
    assert not path_content_unchanged(REL, blocks, tmp_path)


# --- 跨轮拦截 ---------------------------------------------------------------


def test_cross_turn_repeat_read_is_blocked(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    body = _read_output(tmp_path, start_line=1, end_line=350)
    msgs: list[BaseMessage] = [
        HumanMessage(content="改一下 handler_3"),
        _ai([_call("r1", "read_file", path=REL, start_line=1, end_line=350)]),
        ToolMessage(content=body, tool_call_id="r1", name="read_file"),
        HumanMessage(content="顺手把 handler_4 也改掉"),
    ]
    calls = [_call("r2", "read_file", path=REL, start_line=1, end_line=350)]

    blocked = _blocked(msgs, calls, tmp_path)
    assert "r2" in blocked
    body = blocked["r2"].content
    assert body.startswith(CROSS_TURN_READ_MARKER)
    assert "tool_call_id=r1" in body
    assert f"`{REL}`" in body
    # 指针必须比全文短一个数量级，否则不值得拦
    assert len(body) * 10 < len(msgs[2].content)


def test_narrower_range_inside_prior_read_is_blocked(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    calls = [_call("r2", "read_file", path=REL, start_line=20, end_line=40)]
    assert "r2" in _blocked(msgs, calls, tmp_path)


def test_uncovered_range_still_executes(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    calls = [_call("r2", "read_file", path=REL, start_line=200, end_line=260)]
    assert _blocked(msgs, calls, tmp_path) == {}


def test_changed_file_still_executes(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    _write_module(tmp_path, _module_text(marker="v2"))
    calls = [_call("r2", "read_file", path=REL, start_line=1, end_line=90)]
    assert _blocked(msgs, calls, tmp_path) == {}


def test_same_turn_read_is_not_cross_turn_case(tmp_path: Path) -> None:
    """同一问内的重复读仍由本问精确去重接管（拦截文案不同）。"""
    _write_module(tmp_path, _module_text())
    body = _read_output(tmp_path, start_line=1, end_line=90)
    msgs: list[BaseMessage] = [
        HumanMessage(content="改一下 handler_3"),
        _ai([_call("r1", "read_file", path=REL, start_line=1, end_line=90)]),
        ToolMessage(content=body, tool_call_id="r1", name="read_file"),
    ]
    calls = [_call("r2", "read_file", path=REL, start_line=1, end_line=90)]
    blocked = _blocked(msgs, calls, tmp_path)
    assert blocked["r2"].content.startswith(IDENTICAL_BLOCK_MARKER)


def test_written_path_never_deduped(tmp_path: Path) -> None:
    """写过的文件不参与跨轮去重：出站会作废写入前的 read。"""
    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    msgs = (
        msgs[:4]
        + [
            _ai([_call("w1", "search_replace", path=REL, old_string="a", new_string="b")]),
            ToolMessage(content=f"已替换 {REL}（1 处）", tool_call_id="w1", name="search_replace"),
        ]
        + msgs[4:]
    )
    calls = [_call("r2", "read_file", path=REL, start_line=1, end_line=90)]
    assert _blocked(msgs, calls, tmp_path) == {}


def test_compacted_prior_read_is_not_deduped(tmp_path: Path) -> None:
    """历史正文已被出站压缩：模型看不见，绝不能拦。"""
    from llgraph.context.dispatch_compaction import (
        _state_for,
        reset_dispatch_compaction_state,
    )

    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    calls = [_call("r2", "read_file", path=REL, start_line=1, end_line=90)]
    thread = "t-compacted"
    reset_dispatch_compaction_state(thread)
    try:
        assert "r2" in _blocked(msgs, calls, tmp_path, thread_id=thread)
        _state_for(thread).remember(["r1"])
        assert _blocked(msgs, calls, tmp_path, thread_id=thread) == {}
    finally:
        reset_dispatch_compaction_state(thread)


def test_masked_prior_read_is_not_deduped(tmp_path: Path) -> None:
    """历史正文已被 checkpoint 掩码成指针：同样不能拦。"""
    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    msgs[2] = ToolMessage(
        content=f"[历史 read 已归档] `{REL}` 行 1-90",
        tool_call_id="r1",
        name="read_file",
    )
    calls = [_call("r2", "read_file", path=REL, start_line=1, end_line=90)]
    assert _blocked(msgs, calls, tmp_path) == {}


def test_second_attempt_this_turn_is_allowed(tmp_path: Path) -> None:
    """兜底：同一文件本问只拦一次，模型坚持再读就放行，不许死转。"""
    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    first = [_call("r2", "read_file", path=REL, start_line=1, end_line=90)]
    blocked = _blocked(msgs, first, tmp_path)
    assert "r2" in blocked

    msgs = list(msgs) + [_ai(first), blocked["r2"]]
    again = [_call("r3", "read_file", path=REL, start_line=1, end_line=90)]
    assert _blocked(msgs, again, tmp_path) == {}


def test_placeholder_not_indexed_as_history(tmp_path: Path) -> None:
    """拦截占位不算工具结果：否则兜底放行会被本问精确去重吃掉。"""
    assert is_llgraph_placeholder(f"{CROSS_TURN_READ_MARKER}\n- `x.py` 行 1-2")
    assert is_llgraph_placeholder(f"{IDENTICAL_BLOCK_MARKER} ...")
    assert not is_llgraph_placeholder("--- src/a.py (行 1-2 / 共 2 行) ---\n1| x")

    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    calls = [_call("r2", "read_file", path=REL, start_line=1, end_line=90)]
    blocked = _blocked(msgs, calls, tmp_path)
    msgs = list(msgs) + [_ai(calls), blocked["r2"]]
    index = build_history_index(msgs)
    assert index.exact == {}


def test_collapsed_large_read_is_blocked(tmp_path: Path) -> None:
    """大文件不带行段读的是「文件头 + 命中窗」，重读同样应被拦。"""
    from llgraph.core.read_focus import FOCUS_READ_MARKER

    _write_module(tmp_path, _module_text(n=200))
    body = _read_output(tmp_path)
    assert body.startswith(FOCUS_READ_MARKER)
    msgs: list[BaseMessage] = [
        HumanMessage(content="看看这个模块"),
        _ai([_call("r1", "read_file", path=REL)]),
        ToolMessage(content=body, tool_call_id="r1", name="read_file"),
        HumanMessage(content="再改一处"),
    ]
    calls = [_call("r2", "read_file", path=REL)]
    assert "r2" in _blocked(msgs, calls, tmp_path)


def test_collapsed_read_executes_when_new_search_hits_uncovered(tmp_path: Path) -> None:
    """本问新 grep 命中了旧折叠结果没覆盖的区段：重读能带来新信息，必须放行。"""
    _write_module(tmp_path, _module_text(n=200))
    body = _read_output(tmp_path)
    msgs: list[BaseMessage] = [
        HumanMessage(content="看看这个模块"),
        _ai([_call("r1", "read_file", path=REL)]),
        ToolMessage(content=body, tool_call_id="r1", name="read_file"),
        HumanMessage(content="handler_180 有 bug"),
        _ai([_call("g1", "grep_files", pattern="handler_180")]),
        ToolMessage(
            content=f"--- {REL}:545 ---\n545| def handler_180(payload):",
            tool_call_id="g1",
            name="grep_files",
        ),
    ]
    calls = [_call("r2", "read_file", path=REL)]
    assert _blocked(msgs, calls, tmp_path) == {}


def test_union_of_two_prior_reads_covers_request(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text(n=120))
    first = _read_output(tmp_path, start_line=1, end_line=100)
    second = _read_output(tmp_path, start_line=95, end_line=200)
    msgs: list[BaseMessage] = [
        HumanMessage(content="分两段读完"),
        _ai([_call("r1", "read_file", path=REL, start_line=1, end_line=100)]),
        ToolMessage(content=first, tool_call_id="r1", name="read_file"),
        _ai([_call("r2", "read_file", path=REL, start_line=95, end_line=200)]),
        ToolMessage(content=second, tool_call_id="r2", name="read_file"),
        HumanMessage(content="继续改"),
    ]
    calls = [_call("r3", "read_file", path=REL, start_line=1, end_line=200)]
    blocked = _blocked(msgs, calls, tmp_path)
    assert "r3" in blocked
    assert "tool_call_id=r1、r2" in blocked["r3"].content


def test_gap_between_prior_reads_executes(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text(n=120))
    first = _read_output(tmp_path, start_line=1, end_line=80)
    second = _read_output(tmp_path, start_line=120, end_line=200)
    msgs: list[BaseMessage] = [
        HumanMessage(content="读两段"),
        _ai([_call("r1", "read_file", path=REL, start_line=1, end_line=80)]),
        ToolMessage(content=first, tool_call_id="r1", name="read_file"),
        _ai([_call("r2", "read_file", path=REL, start_line=120, end_line=200)]),
        ToolMessage(content=second, tool_call_id="r2", name="read_file"),
        HumanMessage(content="继续改"),
    ]
    calls = [_call("r3", "read_file", path=REL, start_line=1, end_line=200)]
    assert _blocked(msgs, calls, tmp_path) == {}


def test_carry_scan_is_bounded(tmp_path: Path) -> None:
    """护栏自身的解析成本要有上界：只回溯最近若干条 read 结果。"""
    from llgraph.core.cross_turn_read_guard import collect_carry_reads

    _write_module(tmp_path, _module_text(n=10))
    msgs: list[BaseMessage] = []
    for i in range(4):
        rel = f"src/m{i}.py"
        _write_module(tmp_path, _module_text(n=10), rel=rel)
        msgs.append(_ai([_call(f"r{i}", "read_file", path=rel, start_line=1, end_line=20)]))
        msgs.append(
            ToolMessage(
                content=_read_output(tmp_path, rel, start_line=1, end_line=20),
                tool_call_id=f"r{i}",
                name="read_file",
            )
        )
    carry = collect_carry_reads(msgs, end_index=len(msgs), max_messages=2)
    assert set(carry) == {"src/m2.py", "src/m3.py"}


def test_read_files_needs_every_path_covered(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    other = "src/other.py"
    _write_module(tmp_path, _module_text(n=40), rel=other)
    msgs = _two_turn_messages(tmp_path)
    calls = [_call("r2", "read_files", paths=[REL, other], start_line=1, end_line=90)]
    assert _blocked(msgs, calls, tmp_path) == {}

    batch_body = _read_tool(tmp_path, "read_files").func(
        paths=[REL, other], start_line=1, end_line=90
    )
    msgs = [
        HumanMessage(content="看这两个文件"),
        _ai([_call("b1", "read_files", paths=[REL, other], start_line=1, end_line=90)]),
        ToolMessage(content=batch_body, tool_call_id="b1", name="read_files"),
        HumanMessage(content="再改一处"),
    ]
    assert "r2" in _blocked(msgs, calls, tmp_path)


def test_dedupe_disabled_by_flag(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_text())
    msgs = _two_turn_messages(tmp_path)
    calls = [_call("r2", "read_file", path=REL, start_line=1, end_line=90)]
    assert compute_blocked_tool_messages(
        list(msgs) + [_ai(calls)],
        calls,
        workspace=tmp_path,
        thread_id="t",
        cross_turn_reads=False,
    ) == {}

    (tmp_path / ".llgraph").mkdir(exist_ok=True)
    (tmp_path / ".llgraph" / "agent.json").write_text(
        json.dumps({"agent": {"cross_turn_read_dedupe": False}}), encoding="utf-8"
    )
    assert resolve_cross_turn_read_dedupe(tmp_path) is False
    assert resolve_cross_turn_read_dedupe(None) is True


# --- 真 ToolNode 端到端 -----------------------------------------------------


def _node_config(thread_id: str) -> object:
    """真 ToolNode 需要 langgraph 注入的 runtime；内部键变动时退回普通 config。"""
    from langchain_core.runnables import ensure_config

    configurable: dict[str, object] = {"thread_id": thread_id}
    try:
        from langgraph._internal._constants import CONFIG_KEY_RUNTIME
        from langgraph.runtime import Runtime

        configurable[CONFIG_KEY_RUNTIME] = Runtime()
    except ImportError:  # pragma: no cover - langgraph 内部结构变动
        pass
    return ensure_config({"configurable": configurable})


def test_tool_node_end_to_end_second_turn_read_short_circuits(tmp_path: Path) -> None:
    """走生产链路：真 ToolNode + 真 read_file，第二问的重读不再执行真实工具。"""
    import pytest

    from llgraph.context.dispatch_compaction import reset_dispatch_compaction_state
    from llgraph.core.react_tools import build_tool_node

    _write_module(tmp_path, _module_text(n=200))
    node = build_tool_node(create_filesystem_tools(WorkspaceContext(tmp_path)), workspace=tmp_path)
    thread = "t-e2e-cross-turn"
    reset_dispatch_compaction_state(thread)
    calls_1 = [_call("e1", "read_file", path=REL, start_line=1, end_line=200)]
    msgs: list[BaseMessage] = [HumanMessage(content="看 handler_3"), _ai(calls_1)]
    try:
        first = node.invoke({"messages": list(msgs)}, _node_config(thread))
    except ValueError as exc:  # pragma: no cover - langgraph config 契约变动
        pytest.skip(f"ToolNode 直接调用不可用: {exc}")
    finally:
        reset_dispatch_compaction_state(thread)

    first_msgs = list(first.get("messages") or [])
    assert len(first_msgs) == 1
    assert first_msgs[0].content.startswith(f"--- {REL} (行 1-200")

    calls_2 = [_call("e2", "read_file", path=REL, start_line=1, end_line=200)]
    msgs = [*msgs, *first_msgs, HumanMessage(content="再改 handler_4"), _ai(calls_2)]
    try:
        second = node.invoke({"messages": list(msgs)}, _node_config(thread))
    finally:
        reset_dispatch_compaction_state(thread)
    second_msgs = list(second.get("messages") or [])
    assert len(second_msgs) == 1
    assert second_msgs[0].content.startswith(CROSS_TURN_READ_MARKER)
    assert len(second_msgs[0].content) * 5 < len(first_msgs[0].content)


# --- 引用钉住 ---------------------------------------------------------------


def _pointer_message(cid: str = "p1", ref: str = "r1") -> ToolMessage:
    return ToolMessage(
        content=(
            f"{CROSS_TURN_READ_MARKER}\n"
            f"- `{REL}` 行 1-90 / 共 361 行（tool_call_id={ref}）"
        ),
        tool_call_id=cid,
        name="read_file",
    )


def test_referenced_ids_only_from_llgraph_pointers() -> None:
    msgs: list[BaseMessage] = [
        ToolMessage(content="普通结果 tool_call_id=nope", tool_call_id="x1", name="grep_files"),
        _pointer_message(),
    ]
    assert referenced_tool_call_ids(msgs) == {"r1"}


def test_pin_keeps_referenced_read_full_text(tmp_path: Path) -> None:
    from llgraph.context.dispatch_compaction import reset_dispatch_compaction_state
    from llgraph.context.incremental_context import dispatch_keep_tool_indices

    from tests.test_dispatch_tool_chain import _settings

    _write_module(tmp_path, _module_text())
    body = _read_output(tmp_path, start_line=1, end_line=90)
    msgs: list[BaseMessage] = [
        HumanMessage(content="q1"),
        _ai([_call("r1", "read_file", path=REL, start_line=1, end_line=90)]),
        ToolMessage(content=body, tool_call_id="r1", name="read_file"),
        HumanMessage(content="q2"),
        _ai([_call("g1", "grep_files", pattern="handler")]),
        ToolMessage(content="hit-" + "x" * 9000, tool_call_id="g1", name="grep_files"),
        _ai([_call("g2", "grep_files", pattern="payload")]),
        ToolMessage(content="hit-" + "y" * 9000, tool_call_id="g2", name="grep_files"),
    ]
    settings = _settings(
        dispatch_keep_full_tool_messages=1,
        dispatch_full_tool_hysteresis=1.0,
        dispatch_full_tool_budget_tokens=1200,
    )

    thread = "t-pin-off"
    reset_dispatch_compaction_state(thread)
    try:
        kept = dispatch_keep_tool_indices(msgs, settings, thread_id=thread)
    finally:
        reset_dispatch_compaction_state(thread)
    assert 2 not in kept  # 没有引用时旧 read 照常被压

    with_pointer = list(msgs) + [_ai([_call("r2", "read_file", path=REL)]), _pointer_message("p1")]
    thread = "t-pin-on"
    reset_dispatch_compaction_state(thread)
    try:
        kept = dispatch_keep_tool_indices(with_pointer, settings, thread_id=thread)
    finally:
        reset_dispatch_compaction_state(thread)
    assert 2 in kept  # 被指针引用 → 钉住全文


def test_pin_does_not_resurrect_compacted_result(tmp_path: Path) -> None:
    """已经压过的条目不许因为一条晚到的引用复活：前缀回退比省 token 更贵。"""
    from llgraph.context.dispatch_compaction import (
        _state_for,
        reset_dispatch_compaction_state,
    )
    from llgraph.context.incremental_context import dispatch_keep_tool_indices

    from tests.test_dispatch_tool_chain import _settings

    msgs: list[BaseMessage] = [
        HumanMessage(content="q1"),
        _ai([_call("r1", "read_file", path=REL)]),
        ToolMessage(content="--- src/a.py (行 1-2 / 共 2 行) ---\n1| a\n2| b", tool_call_id="r1", name="read_file"),
        _pointer_message("p1"),
    ]
    settings = _settings(max_pinned_referenced_tool_messages=4)
    thread = "t-pin-sticky"
    reset_dispatch_compaction_state(thread)
    try:
        _state_for(thread).remember(["r1"])
        kept = dispatch_keep_tool_indices(msgs, settings, thread_id=thread)
    finally:
        reset_dispatch_compaction_state(thread)
    assert 2 not in kept


def test_pin_cap_and_disable() -> None:
    msgs: list[BaseMessage] = []
    for i in range(6):
        msgs.append(_ai([_call(f"r{i}", "read_file", path=REL)]))
        msgs.append(
            ToolMessage(
                content=f"--- src/f{i}.py (行 1-1 / 共 1 行) ---\n1| x",
                tool_call_id=f"r{i}",
                name="read_file",
            )
        )
    for i in range(6):
        msgs.append(_pointer_message(f"p{i}", ref=f"r{i}"))

    assert len(pinned_referenced_tool_indices(msgs, cap=2)) == 2
    assert pinned_referenced_tool_indices(msgs, cap=0) == set()
    # 取最近的 cap 条
    assert pinned_referenced_tool_indices(msgs, cap=1) == {11}
    assert pinned_referenced_tool_indices(msgs, cap=6, exclude_ids=frozenset({"r0"})) == {
        3,
        5,
        7,
        9,
        11,
    }


def test_prune_state_keeps_referenced_read(tmp_path: Path) -> None:
    from llgraph.context.incremental_context import prune_stale_tool_messages

    from tests.test_dispatch_tool_chain import _settings

    big = "--- src/a.py (行 1-1 / 共 1 行) ---\n1| " + "x" * 30000
    msgs: list[BaseMessage] = [
        HumanMessage(content="q"),
        _ai([_call("r1", "read_file", path="src/a.py")]),
        ToolMessage(content=big, tool_call_id="r1", name="read_file"),
        _ai([_call("g1", "grep_files", pattern="x")]),
        ToolMessage(content="g" * 30000, tool_call_id="g1", name="grep_files"),
    ]
    settings = _settings(keep_recent_tool_messages=1, tool_prune_token_ratio=0.0)

    pruned, count = prune_stale_tool_messages(msgs, tmp_path, settings)
    assert count == 1 and pruned[2].content != big

    with_pointer = list(msgs) + [_pointer_message("p1")]
    pruned, count = prune_stale_tool_messages(with_pointer, tmp_path, settings)
    assert count == 0
    assert pruned[2].content == big
