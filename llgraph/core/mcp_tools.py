"""MCP 客户端：stdio 连接、工具列表与调用。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from llgraph.config.mcp_config import McpServerConfig, McpSettings, format_mcp_summary
from llgraph.permissions.mcp import is_write_mcp_tool

logger = logging.getLogger(__name__)

# Server / 工具中文备注（展示给 Agent 与 Web「工具」页）
_MCP_SERVER_ZH: dict[str, str] = {
    "mysql-biz": "业务库测试环境（内网 MySQL，只读）",
    "mysql-bigdata": "大数据平台测试库（PolarDB MySQL，只读）",
    "mysql": "MySQL 数据库（只读）",
    "postgres": "PostgreSQL 数据库",
    "filesystem": "工作区文件系统",
}

_MCP_TOOL_ZH: dict[str, str] = {
    "mysql_query": "执行 SQL 查询；多库模式请用「库名.表名」，勿写库",
    "query": "执行 SQL 查询",
    "read_query": "执行只读 SQL 查询",
    "read_file": "读取文件内容",
    "read_text_file": "读取文本文件",
    "list_directory": "列出目录下的文件与子目录",
    "directory_tree": "查看目录树结构",
    "search_files": "按名称搜索文件",
    "get_file_info": "查看文件元信息",
    "list_allowed_directories": "列出允许访问的根目录",
}


def mcp_zh_note(server_name: str, tool_name: str) -> str:
    """
    生成 MCP 工具中文备注。

    @param server_name MCP Server 名
    @param tool_name 原始工具名
    @return 中文说明一行
    """
    srv = _MCP_SERVER_ZH.get(server_name, f"外部 MCP「{server_name}」")
    tool = _MCP_TOOL_ZH.get(tool_name, f"调用工具「{tool_name}」")
    return f"{srv} · {tool}"


def _json_type_to_python(type_name: object) -> type:
    if type_name == "integer":
        return int
    if type_name == "number":
        return float
    if type_name == "boolean":
        return bool
    if type_name == "object":
        return dict
    if type_name == "array":
        return list
    return str


def _mcp_input_schema_to_model(
    lc_name: str, input_schema: dict[str, Any]
) -> type[BaseModel] | None:
    """
    将 MCP inputSchema 转为 Pydantic 入参模型，便于模型直接传 sql 等字段。

    @param lc_name LangChain 工具名
    @param input_schema MCP JSON Schema
    @return 模型类；无法解析时返回 None（回退 arguments_json）
    """
    props = input_schema.get("properties")
    if not isinstance(props, dict) or not props:
        return None
    required = {
        str(x) for x in (input_schema.get("required") or []) if isinstance(x, str)
    }
    fields: dict[str, Any] = {}
    for key, spec in props.items():
        key_s = str(key)
        if not isinstance(spec, dict):
            spec = {}
        typ = _json_type_to_python(spec.get("type"))
        desc = str(spec.get("description") or key_s)
        if key_s in required:
            fields[key_s] = (typ, Field(description=desc))
        else:
            fields[key_s] = (typ | None, Field(default=None, description=desc))
    if not fields:
        return None
    safe = "".join(c if c.isalnum() else "_" for c in lc_name)[:48] or "McpTool"
    return create_model(f"McpArgs_{safe}", **fields)  # type: ignore[call-overload]


class _McpArgumentsJson(BaseModel):
    """无结构化 schema 时的回退入参。"""

    arguments_json: str = Field(
        default="{}",
        description="工具参数的 JSON 对象字符串，例如 {\"sql\":\"SELECT 1\"}",
    )


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
        self._thread = threading.Thread(
            target=self._thread_main, name=f"mcp-{self.config.name}", daemon=True
        )
        self._thread.start()
        # 连接等待与 call_tool 超时分离：避免内网不通时卡满 timeout_sec（如 90s）
        connect_wait = min(max(5.0, self.timeout_sec), 25.0) + 5.0
        if not self._ready.wait(timeout=connect_wait):
            self._error = f"连接超时（>{connect_wait:.0f}s）"
            try:
                self.stop()
            except Exception:
                pass
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
            self._ready.set()
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None
            return

        # 连接成功后必须 run_forever，否则 call_tool 无法投递协程
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.run_until_complete(self._shutdown_async())
            except Exception:
                pass
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    async def _connect(self) -> None:
        import os

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # 合并系统环境，避免只传 WORKSPACE_ROOT 时丢掉 PATH 导致 npx 失败
        env = {**os.environ, **(self.config.env or {})}
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=env,
            cwd=self.config.cwd,
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        listed = await self._session.list_tools()
        self._tools = list(listed.tools)

    async def _shutdown_async(self) -> None:
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._stdio_ctx is not None:
            try:
                await self._stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_ctx = None

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
        loop = self._loop
        thread = self._thread
        if loop is None:
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.5)
            self._thread = None
            return

        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None
        self._loop = None
        self._session = None
        self._stdio_ctx = None


class McpToolRegistry:
    """管理多个 MCP Server 并生成 LangChain 工具。"""

    def __init__(self, settings: McpSettings) -> None:
        self.settings = settings
        self._runtimes: dict[str, _McpServerRuntime] = {}
        self._langchain_tools: list[StructuredTool] = []
        self._load_errors: list[str] = []

    @property
    def load_errors(self) -> list[str]:
        """各 Server 加载失败原因。"""
        return list(self._load_errors)

    def start(self) -> None:
        """
        连接所有已启用 Server 并注册工具。

        单个 Server 失败只跳过该 Server，不影响其它 Server 与主流程。
        """
        for cfg in self.settings.servers:
            runtime: _McpServerRuntime | None = None
            try:
                runtime = _McpServerRuntime(cfg, timeout_sec=self.settings.timeout_sec)
                if not runtime.start():
                    msg = runtime._error or "未知错误"
                    self._load_errors.append(f"{cfg.name}: {msg}")
                    runtime.stop()
                    continue
                tools = self._build_tools_for_server(cfg.name, runtime)
                self._runtimes[cfg.name] = runtime
                self._langchain_tools.extend(tools)
                runtime = None  # 所有权已转入 registry
            except Exception as exc:
                logger.warning("MCP Server %s 加载失败（已跳过）: %s", cfg.name, exc)
                self._load_errors.append(f"{cfg.name}: {exc}")
                if runtime is not None:
                    try:
                        runtime.stop()
                    except Exception:
                        pass

    def stop(self) -> None:
        """关闭所有 MCP 连接。"""
        for runtime in self._runtimes.values():
            try:
                runtime.stop()
            except Exception:
                pass
        self._runtimes.clear()
        self._langchain_tools.clear()

    def get_tools(self) -> list[StructuredTool]:
        """已注册的 LangChain 工具（仅加载成功的 Server）。"""
        return list(self._langchain_tools)

    def summary(self) -> str:
        """加载摘要。"""
        ok = len(self._runtimes)
        total = len(self.settings.servers)
        base = format_mcp_summary(self.settings)
        if self._langchain_tools:
            base += f"\n  已加载: {ok}/{total} Server，工具数 {len(self._langchain_tools)}"
        elif total:
            base += f"\n  已加载: 0/{total} Server（失败已跳过，不影响其它功能）"
        for err in self._load_errors:
            base += f"\n  [跳过] {err}"
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
            full_desc = (
                f"【中文说明】{mcp_zh_note(server_name, name)}\n"
                f"[MCP:{server_name}] {desc}"
            )
            if schema_text:
                full_desc += f"\n参数 JSON Schema: {schema_text}"

            args_model = (
                _mcp_input_schema_to_model(lc_name, input_schema)
                if isinstance(input_schema, dict)
                else None
            )

            def make_structured(tname: str, rt: _McpServerRuntime):
                def _invoke(**kwargs: Any) -> str:
                    """按 MCP schema 字段调用工具。"""
                    args = {
                        k: v
                        for k, v in kwargs.items()
                        if v is not None and k != "arguments_json"
                    }
                    # 兼容旧调用：仅传 arguments_json
                    raw = kwargs.get("arguments_json")
                    if not args and isinstance(raw, str):
                        try:
                            parsed = json.loads(raw.strip() or "{}")
                        except json.JSONDecodeError as exc:
                            return f"arguments_json 不是合法 JSON: {exc}"
                        if not isinstance(parsed, dict):
                            return "arguments_json 须为 JSON 对象"
                        args = parsed
                    return rt.call_tool_sync(tname, args)

                return _invoke

            def make_json_blob(tname: str, rt: _McpServerRuntime):
                def _invoke(arguments_json: str = "{}") -> str:
                    """
                    调用 MCP 工具。

                    @param arguments_json 工具参数的 JSON 对象字符串
                    """
                    try:
                        args = (
                            json.loads(arguments_json)
                            if arguments_json and arguments_json.strip()
                            else {}
                        )
                    except json.JSONDecodeError as exc:
                        return f"arguments_json 不是合法 JSON: {exc}"
                    if not isinstance(args, dict):
                        return "arguments_json 须为 JSON 对象"
                    return rt.call_tool_sync(tname, args)

                return _invoke

            if args_model is not None:
                tool = StructuredTool.from_function(
                    func=make_structured(name, runtime),
                    name=lc_name[:64],
                    description=full_desc[:4000],
                    args_schema=args_model,
                )
            else:
                tool = StructuredTool.from_function(
                    func=make_json_blob(name, runtime),
                    name=lc_name[:64],
                    description=full_desc[:4000],
                    args_schema=_McpArgumentsJson,
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

    加载失败不抛异常：返回已成功的工具；全部失败则工具列表为空。

    @param settings MCP 配置
    @return (tools, registry) registry 用于退出时 stop；无 server 时为 None
    """
    if not settings.servers:
        return [], None
    registry = McpToolRegistry(settings)
    try:
        registry.start()
    except Exception as exc:
        logger.exception("MCP 整体加载异常（已降级为空工具）: %s", exc)
        registry._load_errors.append(f"(整体) {exc}")
        try:
            registry.stop()
        except Exception:
            pass
        return [], registry
    return registry.get_tools(), registry
