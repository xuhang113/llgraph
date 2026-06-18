"""Agent 代码索引检索工具。"""

from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.code_index.parallel_search import search_parallel
from llgraph.code_index.search import search_semantic


def create_code_index_tools(workspace_root: Path) -> list:
    """
    创建语义 / 并行检索工具。

    @param workspace_root 工作区根
    @return Tool 列表
    """
    root = workspace_root.expanduser().resolve()

    def search_code_semantic(
        query: str,
        top_k: int = 15,
        path_prefix: str = ".",
    ) -> str:
        """
        语义向量检索（需先 llgraph index）。用于自然语言描述、业务概念、脚本职责反查代码。

        适用：不知道精确类名/文件名，问「XX 定时任务在哪实现」「谁同步 OA 组织」等。
        不适用：已知精确符号请用 grep_files。

        @param query 自然语言或概念描述（可含中英文、服务名、业务词）
        @param top_k 返回条数，默认 15
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
        top_k: int = 15,
        path_prefix: str = ".",
    ) -> str:
        """
        并行代码检索（字面量 grep + 向量，无内嵌 LLM）。

        意图不清时**可选调用一次**；请在 query 里自行扩展类名/关键字（空格分隔）。
        同一轮后续用 read_files + grep_files 追上下游，勿再调用本工具。

        @param query 主 Agent 扩展后的检索词（类名 + 业务描述）
        @param top_k 最终返回条数，默认 15
        @param path_prefix 限定相对子目录（建议仓库名），默认 .
        """
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
