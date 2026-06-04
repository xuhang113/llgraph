"""Web 搜索配置（Tavily；仅 ~/.llgraph/agent.json 的 web_search 段）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from llgraph.core.agent_config import load_user_agent_config
from llgraph.config.config import load_llgraph_env

DEFAULT_API_KEY_ENV = "TAVILY_API_KEY"
VALID_SEARCH_DEPTHS = frozenset({"basic", "advanced", "fast", "ultra-fast"})
VALID_TOPICS = frozenset({"general", "news", "finance"})


@dataclass(frozen=True)
class WebSearchSettings:
    """Tavily 搜索参数。"""

    default_enabled: bool
    api_key_env: str
    max_results: int
    search_depth: str
    include_answer: bool
    timeout_sec: float


def _parse_bool(value: object, default: bool) -> bool:
    """
    解析布尔配置。

    @param value 配置值
    @param default 默认值
    @return 布尔结果
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    return bool(value)


def resolve_web_search_settings(workspace: Path | None = None) -> WebSearchSettings:
    """
    解析 web_search 配置（仅用户级 ~/.llgraph/agent.json，工作区不覆盖）。

    @param workspace 保留以兼容调用方；不参与配置读取
    @return WebSearchSettings
    """
    _ = workspace
    cfg = load_user_agent_config()
    raw = cfg.get("web_search") if isinstance(cfg.get("web_search"), dict) else {}

    default_enabled = _parse_bool(raw.get("default_enabled"), False)

    api_key_env = str(raw.get("api_key_env", DEFAULT_API_KEY_ENV)).strip()
    if not api_key_env:
        api_key_env = DEFAULT_API_KEY_ENV

    max_results = raw.get("max_results", 3)
    try:
        max_results = max(1, min(20, int(max_results)))
    except (TypeError, ValueError):
        max_results = 3

    depth = str(raw.get("search_depth", "ultra-fast")).strip().lower()
    if depth not in VALID_SEARCH_DEPTHS:
        depth = "ultra-fast"

    include_answer = _parse_bool(raw.get("include_answer"), False)

    timeout_sec = raw.get("timeout_sec", 25)
    try:
        timeout_sec = float(timeout_sec)
        timeout_sec = max(5.0, min(120.0, timeout_sec))
    except (TypeError, ValueError):
        timeout_sec = 25.0

    return WebSearchSettings(
        default_enabled=default_enabled,
        api_key_env=api_key_env,
        max_results=max_results,
        search_depth=depth,
        include_answer=include_answer,
        timeout_sec=timeout_sec,
    )


def resolve_tavily_api_key(workspace: Path) -> str | None:
    """
    读取 Tavily API Key。

    @param workspace 工作区根
    @return API Key 或 None
    """
    settings = resolve_web_search_settings(workspace)
    load_llgraph_env()
    key = os.getenv(settings.api_key_env, "").strip()
    return key or None


def check_tavily_dependency() -> str | None:
    """
    检查 tavily-python 是否已安装。

    @return 缺失时的安装提示；已安装则 None
    """
    try:
        import tavily  # noqa: F401
    except ImportError:
        return (
            "未安装 tavily-python。请执行: pip install tavily-python"
            " 或 pip install -e '.[search]'"
        )
    return None


def validate_web_search_ready(workspace: Path) -> tuple[bool, str]:
    """
    启用 web 搜索前的依赖与凭据检查。

    @param workspace 工作区根
    @return (是否就绪, 错误说明；就绪时为空串)
    """
    dep_err = check_tavily_dependency()
    if dep_err:
        return False, dep_err

    settings = resolve_web_search_settings(workspace)
    key = resolve_tavily_api_key(workspace)
    if not key:
        return (
            False,
            f"未配置 Tavily API Key。请在 ~/.config/llgraph/llgraph.env 设置 "
            f"{settings.api_key_env}=tvly-...（免费额度见 https://docs.tavily.com/documentation/api-credits）",
        )
    return True, ""


_FINANCE_HINTS = (
    "a股",
    "股市",
    "股票",
    "个股",
    "上证",
    "深证",
    "创业板",
    "沪指",
    "深指",
    "大盘",
    "行情",
    "涨停",
    "跌停",
)
_TIME_SENSITIVE_HINTS = ("今日", "今天", "最新", "实时", "当前", "now", "today")


def infer_tavily_search_extras(query: str) -> dict[str, str]:
    """
    根据 query 推断 Tavily topic / time_range，缩短财经与时效类检索耗时。

    @param query 搜索词
    @return 传给 client.search 的额外参数（可能为空）
    """
    normalized = query.strip().lower()
    extras: dict[str, str] = {}
    if any(hint in normalized for hint in _FINANCE_HINTS):
        extras["topic"] = "finance"
    elif any(hint in query for hint in ("新闻", "快讯")):
        extras["topic"] = "news"
    if any(hint in query for hint in _TIME_SENSITIVE_HINTS) or any(
        hint in normalized for hint in ("today", "latest")
    ):
        extras["time_range"] = "day"
    topic = extras.get("topic")
    if topic is not None and topic not in VALID_TOPICS:
        extras.pop("topic", None)
    return extras
