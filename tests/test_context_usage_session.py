"""上下文用量：会话识别与 messages.jsonl 回退。"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from llgraph.context.context_stats import collect_context_usage
from llgraph.context.context_session import ContextSession
from llgraph.core.agent_session import AgentSessionContext

# Web Console 属可选依赖：pip install 'llgraph[web]'
pytest.importorskip("fastapi", reason="需要可选依赖 fastapi")

from llgraph.web.server.app import _resolve_session_kind  # noqa: E402


def test_resolve_session_kind_from_cli_prefix(tmp_path: Path) -> None:
    ws = tmp_path
    tid = "cli-abc12345"
    assert _resolve_session_kind(ws, tid) == "agent"


def test_collect_context_usage_falls_back_to_jsonl(tmp_path: Path, monkeypatch) -> None:
    from llgraph.session.session_file_store import save_session_messages
    from llgraph.session.user_storage import session_thread_dir

    ws = tmp_path
    tid = "cli-fallback1"
    session_thread_dir(ws, tid).mkdir(parents=True, exist_ok=True)
    long_reply = "分析结果：" + ("x" * 9000)
    save_session_messages(
        ws,
        tid,
        [
            HumanMessage(content="数据分账需求"),
            AIMessage(content=long_reply),
        ],
    )

    class _FakeAgent:
        def get_state(self, _config):
            class _State:
                values = {"messages": []}

            return _State()

    ctx = AgentSessionContext(
        agent=_FakeAgent(),
        workspace=ws,
        thread_id=tid,
        trace_session=None,  # type: ignore[arg-type]
        context_session=ContextSession(),
        with_memory=True,
        allow_write=True,
    )
    breakdown = collect_context_usage(
        ws,
        context_session=ContextSession(),
        allow_write=True,
        agent_session=ctx,
    )
    assert breakdown.message_count >= 2
    assert breakdown.conversation > 1000


def test_collect_context_usage_from_disk_without_agent(tmp_path: Path) -> None:
    from llgraph.session.session_file_store import save_session_messages
    from llgraph.session.user_storage import session_thread_dir

    ws = tmp_path
    tid = "cli-diskctx"
    session_thread_dir(ws, tid).mkdir(parents=True, exist_ok=True)
    save_session_messages(
        ws,
        tid,
        [
            HumanMessage(content="hello"),
            AIMessage(content="world " * 500),
        ],
    )
    breakdown = collect_context_usage(
        ws,
        context_session=ContextSession(),
        allow_write=False,
        thread_id=tid,
        mcp_tools=[],
    )
    assert breakdown.message_count >= 2
    assert breakdown.conversation > 100
