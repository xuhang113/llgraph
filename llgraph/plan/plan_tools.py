"""Agent 模式 Plan 只读查询工具。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.plan.plan_query import query_plans_text


def create_plan_tools(workspace_root: Path) -> list:
    """
    创建 Plan 查询工具（Agent 模式）。

    @param workspace_root 工作区根
    @return Tool 列表
    """
    root = workspace_root.expanduser().resolve()

    def query_plans(query: str = "", limit: int = 10) -> str:
        """
        查询本工作区 Plan 会话与执行结果（只读，同用户 ~/.llgraph）。

        何时使用：用户问 Plan 执行结果、某 plan-* 输出了什么、历史规划任务等；
        不要用于搜业务代码（用 search_code_parallel / grep_files）。

        query 规则：
        - 空：最近 Plan 列表（默认最多 10 条）
        - plan-xxxxxxxx（thread_id）或 8 位 plan_id：该会话详情（tasks 摘要、final_report、产物路径）
        - 其它：按标题/ id 关键词筛选列表

        注意 thread_id（plan-*）≠ plan_id（8 hex）；产物目录 .llgraph/plans/{plan_id}/。

        @param query 筛选或指定 Plan；默认空=列表
        @param limit 列表条数上限 1～20
        @return 文本摘要与 read_file 路径
        """
        return query_plans_text(root, query=query, limit=limit)

    return [
        StructuredTool.from_function(
            func=query_plans,
            name="query_plans",
            description=(
                "List or look up Plan sessions (plan-*) and task results for this workspace. "
                "Empty query lists recent plans; plan-* or 8-char plan_id returns detail."
            ),
        ),
    ]
