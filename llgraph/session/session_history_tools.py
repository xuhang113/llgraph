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
        按用户问题或关键词检索本会话全历史（归档 jsonl、messages.jsonl、结构化锚点章节）。

        应在以下情况主动调用（勿假设上下文里仍有完整旧对话）：
        - 用户提到「之前/刚才/上次/延续」或引用未在当前窗口出现的决策、文件、报错、shell 命令；
        - 上下文已压缩、切换模型、或你对任务背景不确定（防失忆）；
        - 置顶 <conversation-anchor> 不足以回答细节。

        工作流程：先本工具 → 命中不足时用返回的 read_file 行段（start_line/end_line）精读。
        禁止：search_files 扫 messages.jsonl、cat 整文件、或绕过本工具直接读全量历史。

        检索 shell/find/git/命令 类问题时，会自动纳入 tool 输出与 AI 的 tool_calls 参数。

        不要用于：当前轮已可见的最近几轮内容；纯代码检索请用 search_code_hybrid/grep_files。

        @param query 检索问句或 3～12 个关键词（中英文、文件路径片段、服务名等）
        @param top_k 返回条数，默认 8，最大 20
        @param include_tool_results 是否包含历史 tool 长输出（shell/命令类 query 会自动 true）
        """
        try:
            thread_id = require_active_thread_id()
        except ValueError as exc:
            return str(exc)

        try:
            k = max(1, min(20, int(top_k)))
        except (TypeError, ValueError):
            k = settings.session_history_search_top_k

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
