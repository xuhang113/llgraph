"""MCP 客户端：stdio 连接、工具列表与调用。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from llgraph.config.mcp_config import McpServerConfig, McpSettings, format_mcp_summary
from llgraph.permissions.mcp import is_write_mcp_tool

logger = logging.getLogger(__name__)


class _McpServerRuntime:
    """单 MCP Server 长连接（独立线程 + asyncio loop）。"""

    def __init__(self, config: McpServerConfig, *, timeout_sec: float) -> None:
        self.config = config
        self.timeout_sec = timeout_sec
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._tools: list[Any] = []
        self._session = None
        self._stdio_ctx = None

    def start(self) -> bool:
        """
        启动 MCP 子进程与会话。

        @return 是否成功
        """
        self._thread = threading.Thread(target=self._thread_main, name=f"mcp-{self.config.name}", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self.timeout_sec + 10):
            self._error = "连接超时"
            return False
        return self._error is None

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as exc:
            self._error = str(exc)
            logger.exception("MCP %s 连接失败", self.config.name)
        finally:
            self._ready.set()

    async def _connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env or None,
            cwd=self.config.cwd,
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        listed = await self._session.list_tools()
        self._tools = list(listed.tools)

    def list_tools(self) -> list[Any]:
        """返回 MCP 工具定义列表。"""
        return list(self._tools)

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        同步调用 MCP 工具。

        @param tool_name 工具名
        @param arguments 参数
        @return 文本结果
        """
        if self._loop is None or self._session is None:
            return f"MCP {self.config.name} 未连接"
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(tool_name, arguments),
            self._loop,
        )
        try:
            return future.result(timeout=self.timeout_sec)
        except Exception as exc:
            return f"MCP 调用失败 ({self.config.name}/{tool_name}): {exc}"

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(tool_name, arguments)
        parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(str(block.text))
            else:
                parts.append(str(block))
        if result.isError:
            return f"MCP 错误: {''.join(parts)}"
        return "".join(parts) or "(空结果)"

    def stop(self) -> None:
        """关闭会话与子进程。"""
        if self._loop is None:
            return

        async def _shutdown() -> None:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
            if self._stdio_ctx is not None:
                await self._stdio_ctx.__aexit__(None, None, None)

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            fut.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3)


class McpToolRegistry:
    """管理多个 MCP Server 并生成 LangChain 工具。"""

    def __init__(self, settings: McpSettings) -> None:
        self.settings = settings
        self._runtimes: dict[str, _McpServerRuntime] = {}
        self._langchain_tools: list[StructuredTool] = []
        self._load_errors: list[str] = []

    def start(self) -> None:
        """连接所有已启用 Server 并注册工具。"""
        for cfg in self.settings.servers:
            runtime = _McpServerRuntime(cfg, timeout_sec=self.settings.timeout_sec)
            if not runtime.start():
                msg = runtime._error or "未知错误"
                self._load_errors.append(f"{cfg.name}: {msg}")
                runtime.stop()
                continue
            self._runtimes[cfg.name] = runtime
            self._langchain_tools.extend(
                self._build_tools_for_server(cfg.name, runtime)
            )

    def stop(self) -> None:
        """关闭所有 MCP 连接。"""
        for runtime in self._runtimes.values():
            runtime.stop()
        self._runtimes.clear()
        self._langchain_tools.clear()

    def get_tools(self) -> list[StructuredTool]:
        """已注册的 LangChain 工具。"""
        return list(self._langchain_tools)

    def summary(self) -> str:
        """加载摘要。"""
        base = format_mcp_summary(self.settings)
        if self._langchain_tools:
            base += f"\n  工具数: {len(self._langchain_tools)}"
        for err in self._load_errors:
            base += f"\n  [失败] {err}"
        return base

    def _build_tools_for_server(
        self,
        server_name: str,
        runtime: _McpServerRuntime,
        *,
        allow_write_tools: bool | None = None,
    ) -> list[StructuredTool]:
        permit_write = (
            self.settings.allow_write_tools
            if allow_write_tools is None
            else allow_write_tools
        )
        tools: list[StructuredTool] = []
        for mcp_tool in runtime.list_tools():
            name = mcp_tool.name
            desc = mcp_tool.description or name
            if not permit_write and is_write_mcp_tool(name, desc):
                continue
            lc_name = f"mcp__{server_name}__{name}"
            input_schema = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}

            schema_text = ""
            if isinstance(input_schema, dict) and input_schema:
                try:
                    schema_text = json.dumps(input_schema, ensure_ascii=False)[:1500]
                except TypeError:
                    schema_text = str(input_schema)[:1500]
            full_desc = f"[MCP:{server_name}] {desc}"
            if schema_text:
                full_desc += f"\n参数 JSON Schema: {schema_text}"

            def make_func(tname: str, rt: _McpServerRuntime):
                def _invoke(arguments_json: str = "{}") -> str:
                    """
                    调用 MCP 工具。

                    @param arguments_json 工具参数的 JSON 对象字符串
                    """
                    try:
                        args = json.loads(arguments_json) if arguments_json.strip() else {}
                    except json.JSONDecodeError as exc:
                        return f"arguments_json 不是合法 JSON: {exc}"
                    if not isinstance(args, dict):
                        return "arguments_json 须为 JSON 对象"
                    return rt.call_tool_sync(tname, args)

                return _invoke

            func = make_func(name, runtime)
            tool = StructuredTool.from_function(
                func=func,
                name=lc_name[:64],
                description=full_desc[:4000],
            )
            tools.append(tool)
        return tools

    def rebuild_for_allow_write(self, workspace: Path, allow_write: bool) -> list[StructuredTool]:
        """
        按只读/可写重新过滤 MCP 工具（连接保持不变）。

        @param workspace 工作区根
        @param allow_write 是否允许写类 MCP 工具
        @return 更新后的工具列表
        """
        from llgraph.config.mcp_config import resolve_mcp_settings

        settings = resolve_mcp_settings(workspace, allow_write=allow_write)
        self._langchain_tools.clear()
        for server_name, runtime in self._runtimes.items():
            self._langchain_tools.extend(
                self._build_tools_for_server(
                    server_name,
                    runtime,
                    allow_write_tools=settings.allow_write_tools,
                )
            )
        return list(self._langchain_tools)


def create_mcp_tools(
    settings: McpSettings,
) -> tuple[list[StructuredTool], McpToolRegistry | None]:
    """
    启动 MCP 并返回 LangChain 工具。

    @param settings MCP 配置
    @return (tools, registry) registry 用于退出时 stop
    """
    if not settings.servers:
        return [], None
    registry = McpToolRegistry(settings)
    registry.start()
    return registry.get_tools(), registry
