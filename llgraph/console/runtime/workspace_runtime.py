"""工作区 Runtime：MCP、Trace 配置等进程级资源。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llgraph.config.config import load_llgraph_env
from llgraph.context.context_session import ContextSession
from llgraph.core.tools import load_mcp_tool_bundle
from llgraph.display.trace_display import TraceMode, TraceSession
from llgraph.runtime.shutdown import shutdown_agent_resources
from llgraph.sandbox.policy import build_sandbox_policy
from llgraph.config.sandbox_settings import resolve_sandbox_settings
from llgraph.session.session_web_search import resolve_initial_web_search_enabled


def _index_lock_held_by_self(workspace: Path) -> bool:
    """索引锁文件是否由当前进程持有（Watch 启动成功会写入 pid）。"""
    import os

    from llgraph.code_index.paths import index_root

    path = index_root(workspace) / ".index.lock"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return text.isdigit() and int(text) == os.getpid()


@dataclass
class WorkspaceRuntime:
    """单工作区运行时资源。"""

    workspace: Path
    allow_write: bool = False
    mcp_tools: list = field(default_factory=list)
    mcp_registry: Any = None
    mcp_summary: str = ""
    mcp_ready: threading.Event = field(default_factory=threading.Event)
    sandbox_policy: Any = None
    sandbox_cli_enabled: bool | None = None
    web_search_enabled: bool = False
    trace_session: TraceSession = field(default_factory=TraceSession)
    context_session: ContextSession = field(default_factory=ContextSession)
    watch_service: Any = None


class WorkspaceRuntimeManager:
    """按工作区路径缓存 Runtime。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtimes: dict[str, WorkspaceRuntime] = {}
        self._index_jobs_lock = threading.Lock()
        self._index_jobs: dict[str, threading.Thread] = {}
        load_llgraph_env()

    def get(self, workspace: Path, *, allow_write: bool = False) -> WorkspaceRuntime:
        """
        获取或创建工作区 Runtime。

        MCP 在后台线程加载，避免 HTTP 请求被 npx/连库卡住，进而在重启时触发
        uvicorn graceful shutdown CancelledError。

        @param workspace 工作区根
        @param allow_write 是否允许写（影响 MCP 加载与沙箱）
        @return WorkspaceRuntime
        """
        key = str(workspace.expanduser().resolve())
        with self._lock:
            if key in self._runtimes:
                rt = self._runtimes[key]
                if rt.allow_write != allow_write:
                    self._refresh_write_mode(rt, allow_write=allow_write)
                return rt
            sandbox_settings = resolve_sandbox_settings(workspace)
            sandbox_policy = build_sandbox_policy(
                workspace,
                sandbox_settings,
                cli_enabled=None,
                allow_write=allow_write,
            )
            runtime = WorkspaceRuntime(
                workspace=workspace,
                allow_write=allow_write,
                mcp_tools=[],
                mcp_registry=None,
                mcp_summary="MCP: 后台加载中…",
                sandbox_policy=sandbox_policy,
                sandbox_cli_enabled=None,
                web_search_enabled=resolve_initial_web_search_enabled(workspace),
            )
            from llgraph.code_index.index_watch import (
                attach_watch_shutdown,
                start_index_watch_with_agent,
            )

            runtime.watch_service = start_index_watch_with_agent(workspace, no_watch=False)
            attach_watch_shutdown(runtime.watch_service)
            self._runtimes[key] = runtime

        self._start_mcp_load(runtime, allow_write=allow_write)
        return runtime

    def _start_mcp_load(self, runtime: WorkspaceRuntime, *, allow_write: bool) -> None:
        """后台加载 MCP，不阻塞 get()/HTTP。"""

        def _run() -> None:
            tools: list = []
            registry = None
            summary = "MCP: 未加载"
            try:
                tools, registry, summary = load_mcp_tool_bundle(
                    runtime.workspace,
                    allow_write=allow_write,
                )
            except Exception as exc:
                tools, registry, summary = [], None, f"MCP: 加载异常 {exc}"
            key = str(runtime.workspace.expanduser().resolve())
            with self._lock:
                current = self._runtimes.get(key)
                if current is not runtime:
                    if registry is not None:
                        try:
                            registry.stop()
                        except Exception:
                            pass
                    return
                if (
                    registry is not None
                    and current.allow_write != allow_write
                ):
                    try:
                        tools = registry.rebuild_for_allow_write(
                            current.workspace, current.allow_write
                        )
                        summary = registry.summary()
                    except Exception:
                        pass
                current.mcp_tools = list(tools or [])
                current.mcp_registry = registry
                current.mcp_summary = summary or "MCP: 未加载"
                current.mcp_ready.set()

        threading.Thread(
            target=_run,
            name=f"llgraph-mcp-{runtime.workspace.name}",
            daemon=True,
        ).start()

    def wait_mcp_ready(self, workspace: Path, *, timeout: float = 15.0) -> bool:
        """
        等待 MCP 后台加载完成（开聊前可短等）。

        @param workspace 工作区根
        @param timeout 最长等待秒数
        @return 是否已就绪（超时仍可能部分可用）
        """
        key = str(workspace.expanduser().resolve())
        with self._lock:
            rt = self._runtimes.get(key)
        if rt is None:
            return False
        return rt.mcp_ready.wait(timeout=timeout)

    def ensure_index_watch(self, workspace: Path) -> bool:
        """
        按配置确保索引 Watch 在运行（Web 打开工作区 / 查 index-status 时调用）。

        @param workspace 工作区根
        @return 是否正在监听
        """
        from llgraph.code_index.index_settings import resolve_index_settings
        from llgraph.code_index.index_watch import attach_watch_shutdown, ensure_index_watch

        settings = resolve_index_settings(workspace)
        if not settings.watch_enabled or not settings.watch_with_agent:
            return False
        rt = self.get(workspace)
        if rt.watch_service is not None and getattr(rt.watch_service, "active", False):
            return True
        service, _err = ensure_index_watch(workspace, rt.watch_service)
        rt.watch_service = service
        if service is not None and service.active:
            attach_watch_shutdown(service)
            return True
        # 本进程已占索引锁（Watch 持有）：勿再抢锁失败后误报「启动失败」
        return _index_lock_held_by_self(workspace)

    def index_watch_active(self, workspace: Path) -> bool:
        """
        当前进程是否已在监听该工作区索引变更（不创建 Runtime）。

        @param workspace 工作区根
        @return 是否监听中
        """
        key = str(workspace.expanduser().resolve())
        with self._lock:
            rt = self._runtimes.get(key)
        if rt is not None and rt.watch_service is not None:
            if getattr(rt.watch_service, "active", False):
                return True
        # Watch 占着索引锁但 service 引用异常时，仍视为本进程已在监听
        return _index_lock_held_by_self(workspace)

    def pause_index_watch(self, workspace: Path) -> bool:
        """
        暂停本工作区 index-watch 并释放索引锁（供手动全量/增量抢锁）。

        @param workspace 工作区根
        @return 是否原先在监听（调用方完成后应 resume）
        """
        key = str(workspace.expanduser().resolve())
        with self._lock:
            rt = self._runtimes.get(key)
        if rt is None or rt.watch_service is None:
            return False
        if not getattr(rt.watch_service, "active", False):
            return False
        try:
            rt.watch_service.stop()
        except Exception:
            pass
        return True

    def resume_index_watch(self, workspace: Path) -> bool:
        """
        按配置重新拉起 index-watch。

        @param workspace 工作区根
        @return 是否正在监听
        """
        return self.ensure_index_watch(workspace)

    def start_index_job(self, workspace: Path, action: str) -> dict[str, Any]:
        """
        在后台线程启动索引任务（Web 用）；立即返回，进度见 live_progress。

        @param workspace 工作区根
        @param action incremental|full|rebuild|dry-run
        @return started / busy / error 结果
        """
        import threading

        from llgraph.code_index.index_dispatch import dispatch_index
        from llgraph.code_index.index_progress import read_live_progress, write_live_progress

        key = str(workspace.expanduser().resolve())
        action = action.strip().lower()
        argv_map = {
            "full": ["full"],
            "incremental": ["incremental"],
            "rebuild": ["rebuild"],
            "dry-run": ["dry-run"],
        }
        argv = argv_map.get(action)
        if argv is None:
            return {"ok": False, "started": False, "error": f"未知操作: {action}"}

        with self._index_jobs_lock:
            job = self._index_jobs.get(key)
            if job is not None and job.is_alive():
                return {
                    "ok": False,
                    "started": False,
                    "busy": True,
                    "action": action,
                    "error": "已有索引任务在运行",
                }
            # 清理已结束线程
            self._index_jobs.pop(key, None)

        live = read_live_progress(workspace)
        if live and live.get("running"):
            return {
                "ok": False,
                "started": False,
                "busy": True,
                "action": str(live.get("action") or action),
                "error": "已有索引任务在运行",
            }

        write_live_progress(
            workspace,
            {
                "running": True,
                "action": action,
                "phase": "prepare",
                "files_scanned": 0,
                "files_skipped": 0,
                "files_updated": 0,
                "chunks_written": 0,
                "files_total": None,
                "percent": 0.0,
                "elapsed_sec": 0.0,
                "error": None,
            },
        )

        def _run() -> None:
            paused = False
            try:
                paused = self.pause_index_watch(workspace)
                result = dispatch_index(
                    workspace,
                    argv,
                    prog="/index",
                    bare_means_status=False,
                )
                live_now = read_live_progress(workspace) or {}
                if result.exit_code != 0 and live_now.get("running"):
                    write_live_progress(
                        workspace,
                        {
                            **live_now,
                            "running": False,
                            "phase": "done",
                            "ok": False,
                            "error": live_now.get("error")
                            or "索引失败（见日志）",
                            "percent": live_now.get("percent"),
                        },
                    )
            except Exception as exc:
                write_live_progress(
                    workspace,
                    {
                        "running": False,
                        "action": action,
                        "phase": "done",
                        "ok": False,
                        "error": str(exc),
                        "files_scanned": 0,
                        "files_skipped": 0,
                        "files_updated": 0,
                        "chunks_written": 0,
                        "percent": None,
                        "elapsed_sec": 0.0,
                    },
                )
            finally:
                if paused:
                    try:
                        self.resume_index_watch(workspace)
                    except Exception:
                        pass
                with self._index_jobs_lock:
                    self._index_jobs.pop(key, None)

        thread = threading.Thread(
            target=_run,
            name=f"llgraph-index-{action}",
            daemon=True,
        )
        with self._index_jobs_lock:
            self._index_jobs[key] = thread
        thread.start()
        return {"ok": True, "started": True, "action": action, "exit_code": None, "log_path": None}

    def index_job_running(self, workspace: Path) -> bool:
        """
        本进程是否有后台索引任务。

        @param workspace 工作区根
        @return 是否运行中
        """
        key = str(workspace.expanduser().resolve())
        with self._index_jobs_lock:
            job = self._index_jobs.get(key)
            return job is not None and job.is_alive()

    def _refresh_write_mode(self, rt: WorkspaceRuntime, *, allow_write: bool) -> None:
        """
        Web「允许写」切换时刷新 MCP 与沙箱（不重建整个 Runtime）。

        @param rt 工作区 Runtime
        @param allow_write 目标写权限
        """
        workspace = rt.workspace
        sandbox_settings = resolve_sandbox_settings(workspace)
        sandbox_policy = build_sandbox_policy(
            workspace,
            sandbox_settings,
            cli_enabled=rt.sandbox_cli_enabled,
            allow_write=allow_write,
        )
        rt.allow_write = allow_write
        rt.sandbox_policy = sandbox_policy
        # MCP 仍在加载：只改 allow_write，后台线程结束时会按当前模式过滤
        if rt.mcp_registry is None:
            return
        try:
            rt.mcp_tools = rt.mcp_registry.rebuild_for_allow_write(
                workspace, allow_write
            )
            rt.mcp_summary = rt.mcp_registry.summary()
        except Exception:
            pass

    def _rebuild_sandbox_policy(self, rt: WorkspaceRuntime) -> None:
        """
        按当前 cli 覆盖与工作区写模式重建沙箱策略。

        @param rt 工作区 Runtime
        """
        sandbox_settings = resolve_sandbox_settings(rt.workspace)
        rt.sandbox_policy = build_sandbox_policy(
            rt.workspace,
            sandbox_settings,
            cli_enabled=rt.sandbox_cli_enabled,
            allow_write=rt.allow_write,
        )

    def set_sandbox_enabled(
        self,
        workspace: Path,
        *,
        enabled: bool,
        allow_write: bool | None = None,
    ) -> Any:
        """
        切换工作区 Runtime 的 OS 沙箱开关（覆盖 sandbox.json）。

        @param workspace 工作区根
        @param enabled 是否启用
        @param allow_write 当前写模式；None 时沿用 Runtime 缓存
        @return 更新后的 SandboxPolicy
        """
        if allow_write is None:
            key = str(workspace.expanduser().resolve())
            with self._lock:
                cached = self._runtimes.get(key)
            allow_write = cached.allow_write if cached is not None else False
        rt = self.get(workspace, allow_write=allow_write)
        rt.sandbox_cli_enabled = enabled
        self._rebuild_sandbox_policy(rt)
        policy = rt.sandbox_policy
        if enabled and not policy.enabled:
            rt.sandbox_cli_enabled = False
            self._rebuild_sandbox_policy(rt)
            warning = policy.startup_warning()
            raise ValueError(warning or "沙箱后端不可用，无法启用")
        return policy

    def set_web_search_enabled(self, workspace: Path, *, enabled: bool) -> bool:
        """
        切换工作区 Runtime 的联网搜索开关（影响后续 Agent 与能力清单）。

        @param workspace 工作区根
        @param enabled 是否启用
        @return 当前是否启用
        """
        if enabled:
            from llgraph.config.web_search_settings import validate_web_search_ready

            ok, err = validate_web_search_ready(workspace)
            if not ok:
                raise ValueError(err or "Web 搜索未就绪")
        rt = self.get(workspace)
        rt.web_search_enabled = enabled
        return enabled

    def set_trace_mode(self, workspace: Path, mode: str) -> TraceMode:
        """
        设置 trace 模式。

        @param workspace 工作区根
        @param mode all/steps/reply/none
        @return TraceMode
        """
        from llgraph.display.trace_display import parse_trace_mode

        parsed = parse_trace_mode(mode) or TraceMode.STEPS
        rt = self.get(workspace)
        rt.trace_session.mode = parsed
        return parsed

    def shutdown_all(self) -> None:
        """释放 MCP、index watch 等资源。"""
        with self._lock:
            for rt in self._runtimes.values():
                if rt.watch_service is not None:
                    try:
                        rt.watch_service.stop()
                    except Exception:
                        pass
                shutdown_agent_resources(mcp_registry=rt.mcp_registry)
            self._runtimes.clear()


RUNTIME_MANAGER = WorkspaceRuntimeManager()
