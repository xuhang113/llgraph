"""llgraph 日志级别：CLI、环境变量、agent.json、会话 /log。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from llgraph.config.edit_settings import load_agent_config

SEARCH_LOGGER_NAME = "llgraph.search"

_LEVEL_ALIASES: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "err": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "w": logging.WARNING,
    "info": logging.INFO,
    "i": logging.INFO,
    "debug": logging.DEBUG,
    "d": logging.DEBUG,
    "notset": logging.NOTSET,
}

_LEVEL_NAMES: dict[int, str] = {
    logging.CRITICAL: "CRITICAL",
    logging.ERROR: "ERROR",
    logging.WARNING: "WARNING",
    logging.INFO: "INFO",
    logging.DEBUG: "DEBUG",
    logging.NOTSET: "NOTSET",
}

_runtime_level: int | None = None
_runtime_search_file: bool | None = None


def parse_log_level(name: str | None, *, default: int = logging.WARNING) -> int:
    """
    解析日志级别名。

    @param name 级别字符串（如 debug、INFO）
    @param default 无法识别时的默认值
    @return logging 级别常量
    """
    if name is None or not str(name).strip():
        return default
    key = str(name).strip().lower()
    return _LEVEL_ALIASES.get(key, default)


def level_name(level: int) -> str:
    """
    级别常量转可读名。

    @param level logging 级别
    @return 大写级别名
    """
    return _LEVEL_NAMES.get(level, str(level))


def resolve_log_level(
    workspace: Path | None = None,
    *,
    cli_override: str | None = None,
    default: int = logging.WARNING,
) -> int:
    """
    解析向量检索日志级别（优先级：运行时 /log > CLI > 环境变量 > agent.json）。

    @param workspace 工作区根（读取 agent.json）
    @param cli_override 命令行 --log-level
    @param default 最终兜底
    @return logging 级别
    """
    if _runtime_level is not None:
        return _runtime_level

    if cli_override:
        return parse_log_level(cli_override, default=default)

    env = os.environ.get("LLGRAPH_LOG_LEVEL", "").strip()
    if env:
        return parse_log_level(env, default=default)

    if workspace is not None:
        cfg = load_agent_config(workspace)
        logging_cfg = cfg.get("logging") if isinstance(cfg.get("logging"), dict) else {}
        level_raw = logging_cfg.get("level")
        if isinstance(level_raw, str) and level_raw.strip():
            return parse_log_level(level_raw, default=default)
        search_cfg = cfg.get("search") if isinstance(cfg.get("search"), dict) else {}
        level_raw = search_cfg.get("log_level")
        if isinstance(level_raw, str) and level_raw.strip():
            return parse_log_level(level_raw, default=default)

    return default


def resolve_search_console_enabled(
    workspace: Path | None,
    *,
    cli_override: str | None = None,
) -> bool:
    """
    是否将向量检索日志打印到终端（默认关，仅写 search.log）。

    @param workspace 工作区根
    @param cli_override 启动时 --log-level（显式开启终端日志）
    @return 是否添加 StreamHandler
    """
    if cli_override and str(cli_override).strip():
        return True

    env = os.environ.get("LLGRAPH_LOG_CONSOLE", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False

    if workspace is not None:
        cfg = load_agent_config(workspace)
        logging_cfg = cfg.get("logging") if isinstance(cfg.get("logging"), dict) else {}
        if "search_console" in logging_cfg:
            return bool(logging_cfg.get("search_console"))
        search_cfg = cfg.get("search") if isinstance(cfg.get("search"), dict) else {}
        if "log_console" in search_cfg:
            return bool(search_cfg.get("log_console"))

    return False


def resolve_search_file_enabled(workspace: Path | None, *, level: int) -> bool:
    """
    是否将向量检索日志落盘到 search.log。

    @param workspace 工作区根
    @param level 当前日志级别
    @return 是否写文件
    """
    if _runtime_search_file is not None:
        return _runtime_search_file

    if workspace is not None:
        cfg = load_agent_config(workspace)
        logging_cfg = cfg.get("logging") if isinstance(cfg.get("logging"), dict) else {}
        if "search_file" in logging_cfg:
            return bool(logging_cfg.get("search_file"))
        search_cfg = cfg.get("search") if isinstance(cfg.get("search"), dict) else {}
        if "log_file" in search_cfg:
            return bool(search_cfg.get("log_file"))

    # DEBUG/INFO 默认落盘，便于事后核对是否走向量检索
    return level <= logging.INFO


def search_log_path(workspace: Path) -> Path:
    """
    向量检索日志路径。

    @param workspace 工作区根
    @return .llgraph/index/logs/search.log
    """
    return workspace / ".llgraph" / "index" / "logs" / "search.log"


def setup_search_logging(
    workspace: Path | None,
    level: int | str | None = None,
    *,
    cli_override: str | None = None,
    search_file: bool | None = None,
) -> int:
    """
    配置 llgraph.search logger（控制台 + 可选 search.log）。

    @param workspace 工作区根
    @param level 显式级别；为 None 时走 resolve_log_level
    @param cli_override 命令行 --log-level
    @param search_file 是否落盘；None 时按配置与级别推断
    @return 生效的 logging 级别
    """
    if level is None:
        effective = resolve_log_level(workspace, cli_override=cli_override)
    elif isinstance(level, str):
        effective = parse_log_level(level)
    else:
        effective = int(level)

    file_enabled = (
        search_file
        if search_file is not None
        else resolve_search_file_enabled(workspace, level=effective)
    )

    logger = logging.getLogger(SEARCH_LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(effective)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_enabled = resolve_search_console_enabled(
        workspace,
        cli_override=cli_override,
    )
    if console_enabled:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(effective)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    if file_enabled and workspace is not None:
        log_path = search_log_path(workspace)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
        file_handler.setLevel(effective)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return effective


def set_runtime_log_level(
    workspace: Path,
    level_name: str,
    *,
    search_file: bool | None = None,
) -> int:
    """
    会话内 /log 动态调整级别。

    @param workspace 工作区根
    @param level_name 级别名
    @param search_file 是否落盘；None 保持原配置
    @return 生效级别
    """
    global _runtime_level, _runtime_search_file
    effective = parse_log_level(level_name)
    _runtime_level = effective
    if search_file is not None:
        _runtime_search_file = search_file
    setup_search_logging(workspace, effective, search_file=search_file)
    return effective


def get_search_logger() -> logging.Logger:
    """获取向量检索 logger。"""
    return logging.getLogger(SEARCH_LOGGER_NAME)


def resolve_retention_days(workspace: Path | None = None) -> int:
    """
    日志保留天数（委托 log_retention，供本模块展示）。

    @param workspace 工作区根
    @return 天数
    """
    from llgraph.display.log_retention import resolve_retention_days as _resolve

    return _resolve(workspace)


def format_log_status(workspace: Path, *, cli_override: str | None = None) -> str:
    """
    当前日志配置摘要（/log 无参数时展示）。

    @param workspace 工作区根
    @param cli_override 启动时 CLI 级别（仅展示用）
    @return 多行说明
    """
    level = resolve_log_level(workspace, cli_override=cli_override)
    file_on = resolve_search_file_enabled(workspace, level=level)
    from llgraph.display.execution_log import execution_log_path, resolve_execution_log_enabled

    retention = resolve_retention_days(workspace)
    exec_on = resolve_execution_log_enabled(workspace)
    lines = [
        f"向量检索日志级别: {level_name(level)}",
        f"终端输出 [vector]: {'开' if resolve_search_console_enabled(workspace) else '关（默认，不刷对话区）'}",
        f"落盘 search.log: {'开' if file_on else '关'}",
        f"日志文件: {search_log_path(workspace)}",
        f"执行日志 JSONL: {'开' if exec_on else '关'} → {execution_log_path(workspace)}",
        f"日志保留: {retention} 天（过期自动清理，含 index/search/execution/terminals）",
        "调整: /log debug | /log info | /log warning",
        "查看: /log tail  手动清理: /log purge",
        "启动: llgraph --log-level debug -C <工作区>  （默认仅 search.log）",
        "终端日志: llgraph --log-console  或  LLGRAPH_LOG_CONSOLE=1",
        "运维提示: LLGRAPH_VERBOSE_CONTEXT=1  显示裁剪/压缩/修链一行",
        "环境: LLGRAPH_LOG_LEVEL=debug",
        "配置: logging.retention_days / logging.execution_log（agent.json）",
    ]
    if cli_override:
        lines.insert(1, f"启动参数 --log-level: {cli_override}")
    return "\n".join(lines)
