"""会话自动标题：仅首条无标题时生成。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.plan.plan_store import is_placeholder_plan_title
from llgraph.session.session_meta import (
    disambiguate_session_titles,
    ensure_session_title_auto,
    extract_title_candidate,
    get_session_title,
    normalize_session_title,
    peek_title_from_messages_jsonl,
    resolve_session_full_title,
    save_session_meta,
    set_session_title,
    suggest_full_title_from_text,
    suggest_title_from_text,
)


def test_ensure_title_skips_when_title_exists(tmp_path: Path) -> None:
    workspace = tmp_path
    thread_id = "cli-abc12345"
    save_session_meta(
        workspace,
        thread_id,
        {"session_kind": "agent", "title": "已有标题", "title_source": "auto"},
    )
    result = ensure_session_title_auto(workspace, thread_id, "新的用户消息不应覆盖")
    assert result is None
    assert get_session_title(workspace, thread_id) == "已有标题"


def test_ensure_title_sets_once_from_first_message(tmp_path: Path) -> None:
    workspace = tmp_path
    thread_id = "cli-deadbeef"
    first = ensure_session_title_auto(workspace, thread_id, "帮我整理订单模块文档")
    second = ensure_session_title_auto(workspace, thread_id, "第二条消息")
    assert first == "帮我整理订单模块文档"
    assert second is None
    assert get_session_title(workspace, thread_id) == first


def test_ensure_title_respects_manual_lock(tmp_path: Path) -> None:
    workspace = tmp_path
    thread_id = "plan-12345678"
    set_session_title(workspace, thread_id, "手动标题", source="manual")
    assert ensure_session_title_auto(workspace, thread_id, "不应覆盖") is None


def test_is_placeholder_plan_title() -> None:
    assert is_placeholder_plan_title("", "abc")
    assert is_placeholder_plan_title("未命名计划", "abc")
    assert is_placeholder_plan_title("Plan abc12345", "abc12345")
    assert not is_placeholder_plan_title("订单模块文档整理", "abc12345")


def test_normalize_session_title_truncates() -> None:
    long_text = "这是一段明显超过二十四个汉字限制的首条用户提问内容需要被截断"
    out = normalize_session_title(long_text)
    assert len(out) <= 24
    assert out.endswith("…")


def test_extract_title_from_curl() -> None:
    msg = "curl 'https://api.example.com/api/external/app/getUser' -H 'Auth: x'"
    assert extract_title_candidate(msg) == "getUser"
    assert suggest_title_from_text(msg) == "getUser"


def test_extract_title_first_sentence() -> None:
    msg = "报表服务报错是什么原因？请帮我看一下日志"
    assert suggest_title_from_text(msg) == "报表服务报错是什么原因"


def test_disambiguate_duplicate_titles() -> None:
    entries = [
        ("cli-aaa11111", "报表服务报错问题"),
        ("cli-bbb22222", "报表服务报错问题"),
        ("cli-ccc33333", "其它问题"),
    ]
    titles = disambiguate_session_titles(entries)
    assert titles[0].endswith("aaa11111")
    assert titles[1].endswith("bbb22222")
    assert titles[2] == "其它问题"


def test_resolve_full_title_from_messages_jsonl(tmp_path: Path) -> None:
    from llgraph.session.user_storage import session_messages_path, session_thread_dir

    workspace = tmp_path
    thread_id = "cli-feedface"
    thread_dir = session_thread_dir(workspace, thread_id)
    thread_dir.mkdir(parents=True)
    save_session_meta(
        workspace,
        thread_id,
        {"session_kind": "agent", "title": "demo-query-service、query…", "title_source": "auto"},
    )
    msg_path = session_messages_path(workspace, thread_id)
    long_msg = (
        "demo-query-service、demo-backend-service 里 ReportService 报错，"
        "请帮我看连接池配置"
    )
    msg_path.write_text(
        json.dumps({"role": "user", "content": long_msg}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    full = resolve_session_full_title(workspace, thread_id)
    assert len(full) <= 30
    assert full.startswith("demo-query-service")
    assert "…" not in full
    assert peek_title_from_messages_jsonl(workspace, thread_id) == full


def test_title_skips_workspace_context_injection(tmp_path: Path) -> None:
    from llgraph.session.user_storage import session_messages_path, session_thread_dir

    workspace = tmp_path
    thread_id = "cli-context01"
    thread_dir = session_thread_dir(workspace, thread_id)
    thread_dir.mkdir(parents=True)
    save_session_meta(
        workspace,
        thread_id,
        {"session_kind": "agent", "title": "<workspace-c…", "title_source": "auto"},
    )
    injected = (
        "<workspace-context>\nSome rules here\n</workspace-context>\n\n"
        "报表服务报错是什么原因？请帮我看一下"
    )
    msg_path = session_messages_path(workspace, thread_id)
    msg_path.write_text(
        json.dumps({"role": "user", "content": injected}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    full = resolve_session_full_title(workspace, thread_id)
    assert full == "报表服务报错是什么原因"
    assert "<workspace" not in full.lower()


def test_suggest_full_title_longer_than_display() -> None:
    msg = "demo-query-service、demo-backend-service 里 ReportService 报错"
    full = suggest_full_title_from_text(msg)
    short = suggest_title_from_text(msg)
    assert len(full) <= 30
    assert len(short) <= 24
    assert len(full) >= len(short)
    assert full.startswith("demo-query-service")
