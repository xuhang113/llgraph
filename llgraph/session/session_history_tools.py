"""会话历史检索工具（模型按需拉取压缩前的细节）。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.context.context_settings import resolve_context_settings
from llgraph.context.runtime_context import require_active_thread_id
from llgraph.session.session_history_search import search_session_history


def create_session_history_tools(workspace_root: Path) -> list:
    """
    创建 search_session_history 工具。

    @param workspace_root 工作区根
    @return Tool 列表；未启用时返回空
    """
    root = workspace_root.expanduser().resolve()
    settings = resolve_context_settings(root)
    if not settings.session_history_search_enabled:
        return []

    def search_session_history_tool(
        query: str,
        top_k: int = 8,
        include_tool_results: bool = False,
    ) -> str:
        """
        按关键词检索**本会话多轮历史**（归档、messages.jsonl、锚点章节）。

        **仅当**需要早先轮次细节且当前上下文里没有时再调用：
        - 用户明确提「之前/刚才/上次/延续/你说过」或引用未在当前窗口出现的结论/命令；
        - 或存在 conversation-anchor / 压缩归档，且要核对压缩前细节。

        **禁止**用于：
        - 找回「当前用户消息在问什么」（当前 Human 消息即目标，应直接读对话，勿搜历史）；
        - 纯代码/实现排查（用 grep_files / search_code_* / read_file）；
        - 臆造「用户异议」「上一轮结论」等当前会话并不存在的情节。

        工作流：本工具 → 不足再用返回的 read_file 行段；禁止 cat 全量 messages.jsonl。

        @param query 检索关键词（具体业务词；勿填「用户异议」「上一轮结论」等元叙事）
        @param top_k 返回条数，默认 8，最大 20
        @param include_tool_results 是否含历史 tool 长输出
        """
        try:
            thread_id = require_active_thread_id()
        except ValueError as exc:
            return str(exc)

        try:
            k = max(1, min(20, int(top_k)))
        except (TypeError, ValueError):
            k = settings.session_history_search_top_k

        from llgraph.session.session_history_search import guard_session_history_query

        blocked = guard_session_history_query(root, thread_id, query)
        if blocked:
            return blocked

        return search_session_history(
            root,
            thread_id,
            query,
            top_k=k,
            include_tool_results=bool(include_tool_results),
        )

    return [
        StructuredTool.from_function(
            func=search_session_history_tool,
            name="search_session_history",
            description=search_session_history_tool.__doc__ or "",
        ),
    ]
