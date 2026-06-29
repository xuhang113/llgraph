"""Agent 代码索引检索工具。"""

from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.code_index.parallel_search import search_parallel
from llgraph.code_index.paths import DEFAULT_SEARCH_TOP_K
from llgraph.code_index.search import search_semantic
from llgraph.core.tool_execution_context import (
    count_tool_results_since_user,
    get_tool_execution_messages,
)

_DUPLICATE_PARALLEL_MSG = (
    "【llgraph 拦截】本用户问题已调用过 search_code_parallel，禁止再次调用（换 query 仍算同一问题）。\n"
    "请改用（不要再 parallel）：\n"
    "1. grep_files(pattern=\"业务词|字段名|类名\", path=\".\") — 不确定仓库时用 path=\".\"；\n"
    "2. read_files(paths=[...]) 或 read_file(path, start_line, end_line) 宽段精读；\n"
    "3. path 须来自 list_directory/parallel/glob 结果，禁止猜目录名。"
)


def create_code_index_tools(workspace_root: Path) -> list:
    """
    创建语义 / 并行检索工具。

    @param workspace_root 工作区根
    @return Tool 列表
    """
    root = workspace_root.expanduser().resolve()

    def search_code_semantic(
        query: str,
        top_k: int = DEFAULT_SEARCH_TOP_K,
        path_prefix: str = ".",
    ) -> str:
        """
        语义向量检索（需先 llgraph index）。用于自然语言描述、业务概念、脚本职责反查代码。

        适用：不知道精确类名/文件名，问「XX 定时任务在哪实现」「谁同步 OA 组织」等。
        不适用：已知精确符号请用 grep_files。

        @param query 自然语言或概念描述（可含中英文、服务名、业务词）
        @param top_k 返回条数，默认 8
        @param path_prefix 限定相对子目录，默认 .
        """
        return search_semantic(
            root,
            query,
            top_k=top_k,
            path_prefix=path_prefix,
            source="agent",
            tool="search_code_semantic",
        )

    def search_code_parallel(
        query: str,
        top_k: int = DEFAULT_SEARCH_TOP_K,
        path_prefix: str = ".",
    ) -> str:
        """
        并行代码检索（字面量 grep + 向量，无内嵌 LLM）。

        **每个用户问题最多 1 次**；用户问题已含字段名/类名时应先用 grep_files(path=".")，勿调用本工具。
        之后必须用 grep_files + read_files/read_file；换 query 再调会被拦截。

        @param query 检索词（类名 + 业务描述，空格分隔，建议 5+ 词）
        @param top_k 最终返回条数，默认 8
        @param path_prefix 限定相对子目录，默认 .；不确定时用 .
        """
        prior = get_tool_execution_messages()
        if count_tool_results_since_user(prior, "search_code_parallel") >= 1:
            return _DUPLICATE_PARALLEL_MSG
        return search_parallel(
            root,
            query,
            top_k=top_k,
            path_prefix=path_prefix,
            source="agent",
            tool="search_code_parallel",
        )

    return [
        StructuredTool.from_function(
            func=search_code_semantic,
            name="search_code_semantic",
            description=search_code_semantic.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=search_code_parallel,
            name="search_code_parallel",
            description=search_code_parallel.__doc__ or "",
        ),
    ]
