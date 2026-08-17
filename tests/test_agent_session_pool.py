"""Agent 会话 LRU 保活池测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from llgraph.core.agent_session_pool import (
    AgentSessionPool,
    _make_cache_key,
)
from llgraph.core.session_bootstrap import AgentRuntimeBundle
from llgraph.display.trace_display import TraceSession


def _bundle(workspace: Path, *, allow_write: bool = False) -> AgentRuntimeBundle:
    return AgentRuntimeBundle(
        workspace=workspace,
        trace_session=TraceSession(),
        context_session=MagicMock(),
        allow_write=allow_write,
        mcp_tools=[],
        mcp_registry=None,
        watch_service=None,
        web_search_enabled=False,
        sandbox_policy=None,
        sandbox_cli_enabled=None,
        no_spill=True,
        memory_kind="memory",
        mcp_summary="",
        watch_active=False,
    )


def test_lru_evicts_oldest_when_exceeds_maxsize(tmp_path: Path) -> None:
    pool = AgentSessionPool(maxsize=2)
    built: list[str] = []

    def fake_build(bundle: AgentRuntimeBundle, thread_id: str):
        built.append(thread_id)
        agent = MagicMock(name=f"agent-{thread_id}")
        from llgraph.core.agent_session import AgentSessionContext

        return AgentSessionContext(
            agent=agent,
            workspace=bundle.workspace,
            thread_id=thread_id,
            trace_session=bundle.trace_session,
            context_session=bundle.context_session,
        )

    bundle = _bundle(tmp_path)
    with patch(
        "llgraph.core.agent_session_pool.build_agent_session_for_thread",
        side_effect=fake_build,
    ), patch(
        "llgraph.core.agent_session_pool._messages_mtime",
        return_value=1.0,
    ), patch(
        "llgraph.core.agent_session_pool.release_checkpointer",
    ) as release_mock, patch(
        "llgraph.core.agent_session_pool.log_react_phase",
    ):
        ctx_a = pool.get_or_build(bundle, "cli-a")
        ctx_b = pool.get_or_build(bundle, "cli-b")
        ctx_c = pool.get_or_build(bundle, "cli-c")

    assert ctx_a.agent is not ctx_c.agent
    assert built == ["cli-a", "cli-b", "cli-c"]
    release_mock.assert_called_once_with(tmp_path, "cli-a")

    with patch(
        "llgraph.core.agent_session_pool.build_agent_session_for_thread",
        side_effect=fake_build,
    ), patch(
        "llgraph.core.agent_session_pool._messages_mtime",
        return_value=1.0,
    ), patch(
        "llgraph.core.agent_session_pool.log_react_phase",
    ):
        ctx_a2 = pool.get_or_build(bundle, "cli-a")

    assert ctx_a2.agent is not ctx_a.agent
    assert "cli-a" in built


def test_cache_hit_reuses_same_agent(tmp_path: Path) -> None:
    pool = AgentSessionPool(maxsize=5)
    agent = MagicMock(name="agent")

    def fake_build(bundle: AgentRuntimeBundle, thread_id: str):
        from llgraph.core.agent_session import AgentSessionContext

        return AgentSessionContext(
            agent=agent,
            workspace=bundle.workspace,
            thread_id=thread_id,
            trace_session=bundle.trace_session,
            context_session=bundle.context_session,
        )

    bundle = _bundle(tmp_path)
    with patch(
        "llgraph.core.agent_session_pool.build_agent_session_for_thread",
        side_effect=fake_build,
    ), patch(
        "llgraph.core.agent_session_pool._messages_mtime",
        return_value=2.0,
    ), patch(
        "llgraph.core.agent_session_pool.log_react_phase",
    ) as log_mock:
        first = pool.get_or_build(bundle, "cli-hit")
        second = pool.get_or_build(bundle, "cli-hit")

    assert first.agent is second.agent is agent


def test_disk_mtime_newer_triggers_restore(tmp_path: Path) -> None:
    pool = AgentSessionPool(maxsize=5)
    agent = MagicMock(name="agent")
    def fake_build(bundle: AgentRuntimeBundle, thread_id: str):
        from llgraph.core.agent_session import AgentSessionContext

        return AgentSessionContext(
            agent=agent,
            workspace=bundle.workspace,
            thread_id=thread_id,
            trace_session=bundle.trace_session,
            context_session=bundle.context_session,
        )

    bundle = _bundle(tmp_path)
    with patch(
        "llgraph.core.agent_session_pool.build_agent_session_for_thread",
        side_effect=fake_build,
    ), patch(
        "llgraph.core.agent_session_pool._messages_mtime",
        side_effect=[1.0, 3.0],
    ), patch(
        "llgraph.core.agent_session_pool.restore_session_to_agent",
    ) as restore_mock, patch(
        "llgraph.core.agent_session_pool.log_react_phase",
    ):
        pool.get_or_build(bundle, "cli-r")
        pool.get_or_build(bundle, "cli-r")

    restore_mock.assert_called_once_with(agent, tmp_path, "cli-r")


def test_save_agent_session_messages_syncs_pool_mtime(tmp_path: Path) -> None:
    from langchain_core.messages import HumanMessage

    from llgraph.core.agent_session_pool import AgentSessionPool
    from llgraph.session.session_file_store import save_agent_session_messages

    pool = AgentSessionPool(maxsize=5)
    agent = MagicMock(name="agent")
    tid = "cli-sync"

    def fake_build(bundle: AgentRuntimeBundle, thread_id: str):
        from llgraph.core.agent_session import AgentSessionContext

        return AgentSessionContext(
            agent=agent,
            workspace=bundle.workspace,
            thread_id=thread_id,
            trace_session=bundle.trace_session,
            context_session=bundle.context_session,
        )

    bundle = _bundle(tmp_path)
    with patch(
        "llgraph.core.agent_session_pool._POOL",
        pool,
    ), patch(
        "llgraph.core.agent_session_pool.build_agent_session_for_thread",
        side_effect=fake_build,
    ), patch(
        "llgraph.core.agent_session_pool._messages_mtime",
        side_effect=[1.0, 5.0, 5.0],
    ), patch(
        "llgraph.core.agent_session_pool.restore_session_to_agent",
    ) as restore_mock, patch(
        "llgraph.core.agent_session_pool.log_react_phase",
    ):
        pool.get_or_build(bundle, tid)
        save_agent_session_messages(
            tmp_path,
            tid,
            [HumanMessage(content="hello")],
            sync_pool=True,
        )
        pool.get_or_build(bundle, tid)

    restore_mock.assert_not_called()


def test_cache_key_changes_when_allow_write_changes(tmp_path: Path) -> None:
    read_bundle = _bundle(tmp_path, allow_write=False)
    write_bundle = _bundle(tmp_path, allow_write=True)
    with patch("llgraph.core.agent_session_pool.resolve_effective_model", return_value="m1"):
        k1 = _make_cache_key(read_bundle, "cli-x")
        k2 = _make_cache_key(write_bundle, "cli-x")
    assert k1 != k2
    assert k1.allow_write is False
    assert k2.allow_write is True
