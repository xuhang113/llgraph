"""Tavily Web 搜索内置工具。"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from llgraph.config.web_search_settings import (
    infer_tavily_search_extras,
    resolve_tavily_api_key,
    resolve_web_search_settings,
    validate_web_search_ready,
)

_log = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """web_search 入参。"""

    query: str = Field(description="搜索查询，使用自然语言描述要查的内容")
    max_results: int | None = Field(
        default=None,
        description="返回条数上限（默认由配置决定，通常 5）",
    )


def _format_tavily_response(payload: dict) -> str:
    """
    将 Tavily 响应格式化为模型可读文本。

    @param payload Tavily search 响应
    @return 多行文本
    """
    lines: list[str] = []
    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        lines.append("## 摘要")
        lines.append(answer.strip())
        lines.append("")

    results = payload.get("results") or []
    if results:
        lines.append("## 结果")
        for idx, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip() or "(无标题)"
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            score = item.get("score")
            header = f"{idx}. {title}"
            if url:
                header += f"\n   {url}"
            if score is not None:
                header += f"\n   relevance: {score}"
            lines.append(header)
            if content:
                lines.append(f"   {content}")
            lines.append("")

    if not lines:
        return "未找到相关网页结果。"
    return "\n".join(lines).strip()


def _run_web_search(workspace: Path, query: str, max_results: int | None) -> str:
    """
    执行 Tavily 搜索。

    @param workspace 工作区根
    @param query 搜索词
    @param max_results 条数上限
    @return 格式化结果或错误说明
    """
    stripped = query.strip()
    if not stripped:
        _log.error("web_search 校验失败: query 为空")
        return "错误: query 不能为空。"

    ok, err = validate_web_search_ready(workspace)
    if not ok:
        _log.error("web_search 不可用: %s", err)
        return f"错误: {err}"

    settings = resolve_web_search_settings(workspace)
    limit = max_results if max_results is not None else settings.max_results
    try:
        limit = max(1, min(20, int(limit)))
    except (TypeError, ValueError):
        limit = settings.max_results

    api_key = resolve_tavily_api_key(workspace)
    if not api_key:
        return "错误: 未配置 Tavily API Key。"

    from tavily import TavilyClient
    from tavily.errors import TimeoutError as TavilyTimeoutError

    client = TavilyClient(api_key=api_key)
    extras = infer_tavily_search_extras(stripped)
    if sys.stderr.isatty():
        hint = f" depth={settings.search_depth}"
        if extras:
            hint += f" {extras}"
        print(
            f"  · web_search 请求 Tavily（通常 5～20s）{hint}…",
            flush=True,
            file=sys.stderr,
        )
    started = time.perf_counter()
    try:
        payload = client.search(
            query=stripped,
            max_results=limit,
            search_depth=settings.search_depth,
            include_answer=settings.include_answer,
            timeout=settings.timeout_sec,
            **extras,
        )
    except TavilyTimeoutError:
        _log.error(
            "web_search 超时 query=%r timeout=%ss depth=%s",
            stripped,
            settings.timeout_sec,
            settings.search_depth,
        )
        return (
            f"错误: Tavily 搜索超时（{settings.timeout_sec:g}s）。"
            "可缩短 query、去掉具体日期重试，或在 ~/.llgraph/agent.json 的 web_search 中"
            "调大 timeout_sec / 将 search_depth 设为 ultra-fast、include_answer 设为 false。"
        )
    except Exception as exc:
        _log.error("web_search 调用失败 query=%r: %s", stripped, exc)
        return f"错误: Tavily 搜索失败 ({exc})。"

    if not isinstance(payload, dict):
        return "错误: Tavily 返回格式异常。"
    elapsed = time.perf_counter() - started
    _log.info(
        "web_search 完成 query=%r elapsed=%.2fs depth=%s extras=%s",
        stripped,
        elapsed,
        settings.search_depth,
        extras,
    )
    return _format_tavily_response(payload)


def create_web_search_tools(workspace: Path) -> list[StructuredTool]:
    """
    创建 web_search 工具。

    @param workspace 工作区根
    @return 工具列表（0 或 1 项）
    """
    root = workspace.expanduser().resolve()

    def _search(query: str, max_results: int | None = None) -> str:
        return _run_web_search(root, query, max_results)

    tool = StructuredTool.from_function(
        func=_search,
        name="web_search",
        description=(
            "搜索互联网获取最新信息（文档、版本、新闻、公开 API 说明等）。"
            "工作区代码与本地文档无法回答时再使用；勿用于已能通过 read_file/grep 解决的问题。"
            "问「今日/现在/最新」时先调用 get_current_utc_time 确认日期，query 勿编造未来日期。"
        ),
        args_schema=WebSearchInput,
    )
    return [tool]
