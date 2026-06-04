"""Agent 代码索引检索工具。"""

from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.code_index.hybrid import search_hybrid
from llgraph.code_index.search import search_semantic


def create_code_index_tools(workspace_root: Path) -> list:
    """
    创建语义 / Hybrid 检索工具。

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
        按语义在工作区代码索引中检索（需先 llgraph index）。

        @param query 自然语言或概念描述
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

    def search_code_hybrid(
        query: str,
        top_k: int = 10,
        path_prefix: str = ".",
    ) -> str:
        """
        Hybrid 检索：ripgrep 字面匹配 + 向量语义，RRF 融合（推荐）。

        @param query 问题、类名、错误信息或概念描述
        @param top_k 最终返回条数，默认 10
        @param path_prefix 限定相对子目录
        """
        return search_hybrid(
            root,
            query,
            top_k=top_k,
            path_prefix=path_prefix,
            source="agent",
            tool="search_code_hybrid",
        )

    return [
        StructuredTool.from_function(
            func=search_code_semantic,
            name="search_code_semantic",
            description=search_code_semantic.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=search_code_hybrid,
            name="search_code_hybrid",
            description=search_code_hybrid.__doc__ or "",
        ),
    ]
