"""进程内 Agent 会话 LRU 保活池（预热最近使用的 N 个 thread）。"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llgraph.core.agent_session import AgentSessionContext
from llgraph.core.checkpointer_factory import release_checkpointer
from llgraph.core.llm_settings import resolve_effective_model
from llgraph.core.session_bootstrap import AgentRuntimeBundle, build_agent_session_for_thread
from llgraph.session.session_file_store import restore_session_to_agent
from llgraph.session.session_run_log import log_react_phase
from llgraph.session.user_storage import session_messages_path

from llgraph.config.agent_pool_settings import (
    DEFAULT_POOL_SIZE,
    MAX_POOL_SIZE,
    resolve_agent_pool_settings,
)


@dataclass(frozen=True)
class AgentSessionCacheKey:
    """保活池键：配置变化时视为不同会话。"""

    workspace: str
    thread_id: str
    allow_write: bool
    model_id: str
    web_search_enabled: bool
    sandbox_enabled: bool
    mcp_fingerprint: str


@dataclass
class PooledAgentSession:
    """池内条目。"""

    ctx: AgentSessionContext
    messages_mtime: float
    warmed_at: float


def _workspace_key(workspace: Path) -> str:
    return str(workspace.expanduser().resolve())


def _messages_mtime(workspace: Path, thread_id: str) -> float:
    path = session_messages_path(workspace, thread_id)
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _mcp_fingerprint(mcp_tools: list) -> str:
    if not mcp_tools:
        return ""
    names: list[str] = []
    for tool in mcp_tools:
        name = getattr(tool, "name", None)
        names.append(str(name) if name else str(tool))
    return ",".join(sorted(names))


def _make_cache_key(bundle: AgentRuntimeBundle, thread_id: str) -> AgentSessionCacheKey:
    workspace = bundle.workspace
    sandbox = bundle.sandbox_policy
    return AgentSessionCacheKey(
        workspace=_workspace_key(workspace),
        thread_id=thread_id.strip(),
        allow_write=bool(bundle.allow_write),
        model_id=resolve_effective_model(workspace),
        web_search_enabled=bool(bundle.web_search_enabled),
        sandbox_enabled=bool(sandbox is not None and sandbox.enabled),
        mcp_fingerprint=_mcp_fingerprint(bundle.mcp_tools or []),
    )


def _refresh_turn_handles(ctx: AgentSessionContext, bundle: AgentRuntimeBundle) -> None:
    """每轮刷新 trace / 共享运行时句柄（不重建 Agent）。"""
    ctx.trace_session = bundle.trace_session
    ctx.context_session = bundle.context_session
    ctx.mcp_tools = bundle.mcp_tools
    ctx.mcp_registry = bundle.mcp_registry
    ctx.watch_service = bundle.watch_service
    ctx.web_search_enabled = bundle.web_search_enabled
    ctx.sandbox_policy = bundle.sandbox_policy
    ctx.sandbox_cli_enabled = bundle.sandbox_cli_enabled


class AgentSessionPool:
    """线程安全 LRU；默认保留最近 5 个已构建 Agent。"""

    def __init__(self, *, maxsize: int = DEFAULT_POOL_SIZE) -> None:
        self._maxsize = max(1, maxsize)
        self._entries: OrderedDict[AgentSessionCacheKey, PooledAgentSession] = OrderedDict()
        self._lock = threading.Lock()

    def set_maxsize(self, maxsize: int) -> None:
        """按 agent.json agent_pool.pool_size 调整容量并淘汰溢出条目。"""
        with self._lock:
            self._maxsize = max(1, min(int(maxsize), MAX_POOL_SIZE))
            while len(self._entries) >= self._maxsize:
                self._evict_oldest()

    def get_or_build(self, bundle: AgentRuntimeBundle, thread_id: str) -> AgentSessionContext:
        """
        命中则复用已编译 Agent；磁盘 messages 更新则仅 restore。

        @param bundle 运行时依赖
        @param thread_id cli-* thread
        @return AgentSessionContext
        """
        tid = thread_id.strip()
        key = _make_cache_key(bundle, tid)
        workspace = bundle.workspace
        _sync_pool_for_workspace(workspace)

        with self._lock:
            pooled = self._entries.get(key)
            if pooled is not None:
                self._entries.move_to_end(key)
                ctx = pooled.ctx
                _refresh_turn_handles(ctx, bundle)
                current_mtime = _messages_mtime(workspace, tid)
                if current_mtime > pooled.messages_mtime + 1e-6:
                    t0 = time.monotonic()
                    restore_session_to_agent(ctx.agent, workspace, tid)
                    pooled.messages_mtime = current_mtime
                    log_react_phase(
                        workspace,
                        tid,
                        phase="agent_pool_restore",
                        detail={"pool_size": len(self._entries)},
                        duration_sec=time.monotonic() - t0,
                    )
                else:
                    pass
                return ctx

        t0 = time.monotonic()
        ctx = build_agent_session_for_thread(bundle, tid)
        mtime = _messages_mtime(workspace, tid)
        build_sec = time.monotonic() - t0

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                _refresh_turn_handles(existing.ctx, bundle)
                return existing.ctx

            while len(self._entries) >= self._maxsize:
                self._evict_oldest()

            self._entries[key] = PooledAgentSession(
                ctx=ctx,
                messages_mtime=mtime,
                warmed_at=time.time(),
            )
            log_react_phase(
                workspace,
                tid,
                phase="agent_pool_miss",
                detail={"pool_size": len(self._entries), "evicted": False},
                duration_sec=build_sec,
            )
            return ctx

    def notify_messages_persisted(self, workspace: Path, thread_id: str) -> None:
        """persist 后同步 mtime，避免下一轮误判需 restore。"""
        ws = _workspace_key(workspace)
        tid = thread_id.strip()
        mtime = _messages_mtime(workspace, tid)
        with self._lock:
            for key, pooled in self._entries.items():
                if key.workspace == ws and key.thread_id == tid:
                    pooled.messages_mtime = mtime
                    return

    def invalidate_thread(self, workspace: Path, thread_id: str) -> None:
        """显式淘汰某 thread（如删会话）。"""
        ws = _workspace_key(workspace)
        tid = thread_id.strip()
        with self._lock:
            to_remove = [k for k in self._entries if k.workspace == ws and k.thread_id == tid]
            for key in to_remove:
                self._entries.pop(key, None)
                release_checkpointer(workspace, tid)

    def stats(self) -> dict[str, Any]:
        """调试：当前池大小与 thread 列表。"""
        with self._lock:
            return {
                "maxsize": self._maxsize,
                "size": len(self._entries),
                "threads": [
                    {
                        "workspace": k.workspace,
                        "thread_id": k.thread_id,
                        "warmed_at": e.warmed_at,
                    }
                    for k, e in self._entries.items()
                ],
            }

    def _evict_oldest(self) -> None:
        key, pooled = self._entries.popitem(last=False)
        release_checkpointer(Path(key.workspace), key.thread_id)
        log_react_phase(
            Path(key.workspace),
            key.thread_id,
            phase="agent_pool_evict",
            detail={"pool_size": len(self._entries)},
        )
        del pooled


def _sync_pool_for_workspace(workspace: Path) -> None:
    """读取工作区 agent.json 并同步 LRU 容量。"""
    settings = resolve_agent_pool_settings(workspace)
    _POOL.set_maxsize(settings.pool_size)


_POOL = AgentSessionPool(maxsize=DEFAULT_POOL_SIZE)


def warm_agent_session_for_thread(
    bundle: AgentRuntimeBundle,
    thread_id: str,
) -> AgentSessionContext:
    """
    主动预热 Agent（与 get_or_build 相同，供新建会话 / 批量 warm 使用）。

    @param bundle 运行时依赖
    @param thread_id cli-* thread
    @return AgentSessionContext
    """
    return _POOL.get_or_build(bundle, thread_id.strip())


def warm_recent_agent_sessions(
    bundle: AgentRuntimeBundle,
    thread_ids: list[str],
    *,
    limit: int | None = None,
) -> list[str]:
    """
    按顺序预热若干 thread（超出 LRU 容量时由池自动淘汰最旧条目）。

    @param bundle 运行时依赖
    @param thread_ids 候选 thread 列表（通常按 updated_at 降序）
    @param limit 最多预热条数；None 时读 agent.json agent_pool.warm_recent_limit
    @return 实际尝试预热的 thread_id 列表
    """
    settings = resolve_agent_pool_settings(bundle.workspace)
    cap = settings.warm_recent_limit if limit is None else max(0, min(limit, MAX_POOL_SIZE))
    if cap <= 0:
        return []
    warmed: list[str] = []
    for tid in thread_ids:
        if len(warmed) >= cap:
            break
        stripped = tid.strip()
        if not stripped:
            continue
        warm_agent_session_for_thread(bundle, stripped)
        warmed.append(stripped)
    return warmed


def get_or_build_agent_session_for_thread(
    bundle: AgentRuntimeBundle,
    thread_id: str,
) -> AgentSessionContext:
    """
    Web 热路径：LRU 保活 + 按需 restore。

    @param bundle 运行时依赖
    @param thread_id cli-* thread
    @return AgentSessionContext
    """
    return _POOL.get_or_build(bundle, thread_id)


def notify_agent_session_persisted(workspace: Path, thread_id: str) -> None:
    """
    messages.jsonl 落盘后更新池内 mtime。

    @param workspace 工作区根
    @param thread_id 会话 thread
    """
    _POOL.notify_messages_persisted(workspace, thread_id)


def invalidate_agent_session_thread(workspace: Path, thread_id: str) -> None:
    """
    从保活池移除指定 thread。

    @param workspace 工作区根
    @param thread_id 会话 thread
    """
    _POOL.invalidate_thread(workspace, thread_id)


def agent_session_pool_stats() -> dict[str, Any]:
    """@return 保活池快照"""
    return _POOL.stats()
