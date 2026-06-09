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

    def search_code_hybrid(
        query: str,
        top_k: int = 10,
        path_prefix: str = ".",
    ) -> str:
        """
        Hybrid 检索（索引已启用时**首选**）：路径/文件名 + 内容 grep + 向量语义，RRF 融合。

        已内聚 search_files 的路径匹配能力；找 crontab 脚本、Java Job、Python 实现时直接用本工具，
        勿再单独调用 search_files。

        @param query 脚本名、业务描述、类名、错误信息或概念（可中英文组合）
        @param top_k 最终返回条数，默认 10
        @param path_prefix 限定相对子目录，如 queryplatform-backend-service
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
