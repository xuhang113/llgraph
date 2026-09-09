"""同一路径连续写失败的三级升级：诊断提示 → 锁定须重读 → 彻底停写。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.core.react_limits import (
    DEFAULT_EDIT_FAILURE_HINT_AFTER,
    EDIT_FAILURE_LOCK_OFFSET,
    EDIT_FAILURE_STOP_OFFSET,
    parse_edit_failure_hint_after,
    resolve_edit_failure_hint_after,
)
from llgraph.core.tool_failure_escalation import (
    ESCALATION_MARKER,
    REASON_BAD_ARGS,
    REASON_MISSING_FILE,
    REASON_NOT_FOUND,
    REASON_NOT_UNIQUE,
    RELOCK_MARKER,
    STOP_MARKER,
    annotate_escalation_hints,
    classify_failure,
    compute_escalation_blocks,
    install_edit_failure_blocks,
    scan_path_failures,
)
from llgraph.core.tool_loop_guard import IDENTICAL_FAIL_MARKER

HINT_AFTER = DEFAULT_EDIT_FAILURE_HINT_AFTER
LOCK_AT = HINT_AFTER + EDIT_FAILURE_LOCK_OFFSET
STOP_AT = HINT_AFTER + EDIT_FAILURE_STOP_OFFSET

_NOT_FOUND = "未找到 old_string（0 处匹配）: svc.py\n已尝试匹配: exact / whitespace"


def _replace_call(cid: str, path: str = "svc.py", old: str = "x") -> dict:
    return {
        "id": cid,
        "name": "search_replace",
        "args": {"path": path, "old_string": old, "new_string": old + "!"},
        "type": "tool_call",
    }


def _read_call(cid: str, path: str = "svc.py") -> dict:
    return {"id": cid, "name": "read_file", "args": {"path": path}, "type": "tool_call"}


def _ai(calls: list[dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=calls)


def _tool(cid: str, name: str, content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=cid, name=name)


def _failing_history(times: int, *, path: str = "svc.py") -> list:
    """构造 times 次「每次换个 old_string 但都没匹配上」的历史。"""
    msgs: list = [HumanMessage(content="给 submit 加折扣")]
    for idx in range(times):
        cid = f"f{idx}"
        msgs.append(_ai([_replace_call(cid, path=path, old=f"variant-{idx}")]))
        msgs.append(_tool(cid, "search_replace", _NOT_FOUND))
    return msgs


def _blocks(msgs: list, calls: list[dict]) -> dict:
    return compute_escalation_blocks(
        msgs,
        calls,
        hint_after=HINT_AFTER,
        lock_offset=EDIT_FAILURE_LOCK_OFFSET,
        stop_offset=EDIT_FAILURE_STOP_OFFSET,
    )


def _annotate(prior: list, result: ToolMessage) -> ToolMessage:
    out = annotate_escalation_hints(
        {"messages": [result]},
        prior_messages=prior,
        hint_after=HINT_AFTER,
        lock_offset=EDIT_FAILURE_LOCK_OFFSET,
        stop_offset=EDIT_FAILURE_STOP_OFFSET,
    )
    return out["messages"][0]


def test_classify_failure_reasons() -> None:
    assert classify_failure(_NOT_FOUND) == REASON_NOT_FOUND
    assert classify_failure("old_string 在 svc.py 中出现多次，不唯一。") == REASON_NOT_UNIQUE
    assert classify_failure("文件不存在: svc.py") == REASON_MISSING_FILE
    assert classify_failure("错误: write_file 缺少必填参数 content") == REASON_BAD_ARGS


def test_streak_counts_only_same_path() -> None:
    msgs = _failing_history(2)
    msgs.append(_ai([_replace_call("o1", path="other.py")]))
    msgs.append(_tool("o1", "search_replace", "文件不存在: other.py"))
    states = scan_path_failures(msgs)
    assert states["svc.py"].count == 2
    assert states["other.py"].count == 1


def test_successful_write_clears_streak() -> None:
    msgs = _failing_history(3)
    msgs.append(_ai([_replace_call("ok")]))
    msgs.append(_tool("ok", "search_replace", "已替换 svc.py（1 处）"))
    assert "svc.py" not in scan_path_failures(msgs)


def test_new_user_turn_clears_streak() -> None:
    msgs = _failing_history(STOP_AT)
    msgs.append(HumanMessage(content="换个需求"))
    assert scan_path_failures(msgs) == {}


def test_own_notices_are_not_counted_as_failures() -> None:
    """拦截文案本身也含「错误」字样，若计入失败，拦截会自我放大。"""
    msgs = _failing_history(1)
    for idx in range(3):
        cid = f"b{idx}"
        msgs.append(_ai([_replace_call(cid, old=f"blocked-{idx}")]))
        msgs.append(
            _tool(cid, "search_replace", f"{IDENTICAL_FAIL_MARKER}\n上次错误摘录: 未找到")
        )
    assert scan_path_failures(msgs)["svc.py"].count == 1


def test_hint_appended_only_after_threshold() -> None:
    prior = _failing_history(HINT_AFTER - 1)
    call = _replace_call("n1", old="another")
    prior.append(_ai([call]))
    out = annotate_escalation_hints(
        {"messages": [_tool("n1", "search_replace", _NOT_FOUND)]},
        prior_messages=prior,
        hint_after=HINT_AFTER,
        lock_offset=EDIT_FAILURE_LOCK_OFFSET,
        stop_offset=EDIT_FAILURE_STOP_OFFSET,
    )
    body = str(out["messages"][0].content)
    assert ESCALATION_MARKER in body
    assert f"连续失败 {HINT_AFTER} 次" in body
    assert "read_file" in body

    first = annotate_escalation_hints(
        {"messages": [_tool("n1", "search_replace", _NOT_FOUND)]},
        prior_messages=[HumanMessage(content="改一下"), _ai([call])],
        hint_after=HINT_AFTER,
        lock_offset=EDIT_FAILURE_LOCK_OFFSET,
        stop_offset=EDIT_FAILURE_STOP_OFFSET,
    )
    assert ESCALATION_MARKER not in str(first["messages"][0].content)


def test_hint_lists_attempts_and_reason_specific_advice() -> None:
    prior: list = [HumanMessage(content="改一下")]
    for idx in range(HINT_AFTER):
        cid = f"u{idx}"
        prior.append(_ai([_replace_call(cid, old=f"dup-{idx}")]))
        prior.append(
            _tool(cid, "search_replace", "old_string 在 svc.py 中出现多次，不唯一。")
        )
    call = _replace_call("u9", old="dup-9")
    prior.append(_ai([call]))
    out = annotate_escalation_hints(
        {"messages": [_tool("u9", "search_replace", "old_string 在 svc.py 中出现多次，不唯一。")]},
        prior_messages=prior,
        hint_after=HINT_AFTER,
        lock_offset=EDIT_FAILURE_LOCK_OFFSET,
        stop_offset=EDIT_FAILURE_STOP_OFFSET,
    )
    body = str(out["messages"][0].content)
    assert "replacements" in body and "replace_all" in body
    assert 'old_string 首行 "dup-0"' in body


def test_hint_not_duplicated_onto_intercepted_result() -> None:
    prior = _failing_history(HINT_AFTER)
    call = _replace_call("n2", old="another")
    prior.append(_ai([call]))
    out = annotate_escalation_hints(
        {"messages": [_tool("n2", "search_replace", f"{RELOCK_MARKER}\n先 read_file")]},
        prior_messages=prior,
        hint_after=HINT_AFTER,
        lock_offset=EDIT_FAILURE_LOCK_OFFSET,
        stop_offset=EDIT_FAILURE_STOP_OFFSET,
    )
    assert ESCALATION_MARKER not in str(out["messages"][0].content)


def test_lock_blocks_write_until_file_is_read() -> None:
    msgs = _failing_history(LOCK_AT)
    call = _replace_call("x1", old="yet-another")
    blocked = _blocks(msgs, [call])
    assert RELOCK_MARKER in str(blocked["x1"].content)

    msgs.append(_ai([_read_call("r1")]))
    msgs.append(_tool("r1", "read_file", "--- svc.py (行 1-7 / 共 7 行)\n1 | class X:"))
    assert _blocks(msgs, [call]) == {}


def test_lock_ignores_read_of_a_different_file() -> None:
    msgs = _failing_history(LOCK_AT)
    msgs.append(_ai([_read_call("r1", path="other.py")]))
    msgs.append(_tool("r1", "read_file", "--- other.py (行 1-3 / 共 3 行)"))
    assert "x1" in _blocks(msgs, [_replace_call("x1")])


def test_failed_read_does_not_unlock() -> None:
    msgs = _failing_history(LOCK_AT)
    msgs.append(_ai([_read_call("r1")]))
    msgs.append(_tool("r1", "read_file", "文件不存在: svc.py"))
    assert "x1" in _blocks(msgs, [_replace_call("x1")])


def test_read_unlock_is_consumed_by_the_next_failure() -> None:
    """读完再失败要继续升级，否则 read → fail → read → fail 可以永远转下去。"""
    msgs = _failing_history(LOCK_AT)
    msgs.append(_ai([_read_call("r1")]))
    msgs.append(_tool("r1", "read_file", "--- svc.py (行 1-7 / 共 7 行)"))
    msgs.append(_ai([_replace_call("x1", old="after-read")]))
    msgs.append(_tool("x1", "search_replace", _NOT_FOUND))
    state = scan_path_failures(msgs)["svc.py"]
    assert state.count == LOCK_AT + 1
    assert state.read_after_last_failure is False
    assert "x2" in _blocks(msgs, [_replace_call("x2")])


def test_stop_blocks_even_after_a_fresh_read() -> None:
    msgs = _failing_history(STOP_AT)
    msgs.append(_ai([_read_call("r1")]))
    msgs.append(_tool("r1", "read_file", "--- svc.py (行 1-7 / 共 7 行)"))
    body = str(_blocks(msgs, [_replace_call("x1")])["x1"].content)
    assert STOP_MARKER in body
    assert "向用户说明" in body


def test_other_paths_stay_writable_when_one_is_stopped() -> None:
    msgs = _failing_history(STOP_AT)
    blocked = _blocks(msgs, [_replace_call("x1"), _replace_call("x2", path="ok.py")])
    assert "x1" in blocked
    assert "x2" not in blocked


def test_escalation_disabled_by_config() -> None:
    msgs = _failing_history(STOP_AT)
    assert (
        compute_escalation_blocks(
            msgs,
            [_replace_call("x1")],
            hint_after=0,
            lock_offset=EDIT_FAILURE_LOCK_OFFSET,
            stop_offset=EDIT_FAILURE_STOP_OFFSET,
        )
        == {}
    )


def test_parse_and_resolve_threshold(tmp_path: Path) -> None:
    assert parse_edit_failure_hint_after(None) == HINT_AFTER
    assert parse_edit_failure_hint_after(False) == 0
    assert parse_edit_failure_hint_after(True) == HINT_AFTER
    assert parse_edit_failure_hint_after(5) == 5
    assert parse_edit_failure_hint_after("nonsense") == HINT_AFTER
    assert resolve_edit_failure_hint_after(None) == HINT_AFTER

    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text(
        json.dumps({"agent": {"edit_failure_escalation_after": 0}}),
        encoding="utf-8",
    )
    assert resolve_edit_failure_hint_after(tmp_path) == 0


def test_install_merges_into_loop_guard_table_without_overwriting() -> None:
    class _Node:
        pass

    node = _Node()
    kept = ToolMessage(content="loop guard 先到", tool_call_id="x1", name="search_replace")
    node._llgraph_loop_blocks = {"x1": kept}
    msgs = _failing_history(STOP_AT)
    install_edit_failure_blocks(
        node,
        msgs,
        [_replace_call("x1"), _replace_call("x2", old="second")],
        hint_after=HINT_AFTER,
        lock_offset=EDIT_FAILURE_LOCK_OFFSET,
        stop_offset=EDIT_FAILURE_STOP_OFFSET,
    )
    assert node._llgraph_loop_blocks["x1"] is kept
    assert STOP_MARKER in str(node._llgraph_loop_blocks["x2"].content)


def test_appended_hint_still_counts_as_a_failure() -> None:
    """升级提示是追加在真失败末尾的；若把它当成占位文案，失败计数会卡在阈值上。"""
    msgs: list = [HumanMessage(content="改一下")]
    for idx in range(LOCK_AT):
        cid = f"h{idx}"
        msgs.append(_ai([_replace_call(cid, old=f"guess-{idx}")]))
        msgs.append(_annotate(msgs, _tool(cid, "search_replace", _NOT_FOUND)))
    assert any(ESCALATION_MARKER in str(m.content) for m in msgs if isinstance(m, ToolMessage))
    assert scan_path_failures(msgs)["svc.py"].count == LOCK_AT
    assert "x1" in _blocks(msgs, [_replace_call("x1")])


def _run_loop(*, read_before_each_edit: bool) -> tuple[int, str]:
    """按生产链路（先算拦截，再给失败结果贴提示）跑一段死磕循环。"""
    msgs: list = [HumanMessage(content="给 submit 加折扣")]
    executed = 0
    for idx in range(30):
        if read_before_each_edit:
            rid = f"r{idx}"
            msgs.append(_ai([_read_call(rid)]))
            msgs.append(_tool(rid, "read_file", "--- svc.py (行 1-7 / 共 7 行)"))
        call = _replace_call(f"t{idx}", old=f"guess-{idx}")
        msgs.append(_ai([call]))
        blocked = _blocks(msgs, [call])
        if f"t{idx}" in blocked:
            msgs.append(blocked[f"t{idx}"])
            continue
        executed += 1
        msgs.append(_annotate(msgs, _tool(f"t{idx}", "search_replace", _NOT_FOUND)))
    return executed, str(msgs[-1].content)


def test_blind_retry_loop_stops_at_lock() -> None:
    """模型换着 old_string 死磕同一文件、不肯重读，写入必须在锁定处停住。"""
    executed, last = _run_loop(read_before_each_edit=False)
    assert executed == LOCK_AT
    assert RELOCK_MARKER in last


def test_read_then_retry_loop_still_terminates() -> None:
    """模型每次照要求先 read 再改：解锁可用，但不能无限用，最终停在 STOP。"""
    executed, last = _run_loop(read_before_each_edit=True)
    assert executed == STOP_AT
    assert STOP_MARKER in last
