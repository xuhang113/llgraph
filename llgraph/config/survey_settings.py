"""Survey 交互配置（agent.json / CLI / 会话内开关）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llgraph.config.edit_settings import load_agent_config

if TYPE_CHECKING:
    from llgraph.context.context_session import ContextSession

_CLI_DISABLED = False


def set_survey_cli_disabled(disabled: bool) -> None:
    """
    启动参数 --no-survey 设置进程级禁用。

    @param disabled 是否禁用
    """
    global _CLI_DISABLED
    _CLI_DISABLED = disabled


def survey_cli_disabled() -> bool:
    """
    是否由 CLI/环境变量禁用 survey。

    @return 是否禁用
    """
    if _CLI_DISABLED:
        return True
    raw = os.environ.get("LLGRAPH_SURVEY", "").strip().lower()
    return raw in ("0", "false", "no", "off", "disable", "disabled")


@dataclass(frozen=True)
class SurveySettings:
    """Survey 默认配置（agent.json → survey）。"""

    enabled: bool
    preflight: bool
    followup: bool
    command: bool


def resolve_survey_settings(workspace: Path | None) -> SurveySettings:
    """
    解析 survey 配置。

    @param workspace 工作区根
    @return SurveySettings
    """
    enabled = True
    preflight = True
    followup = True
    command = True
    if workspace is None:
        return SurveySettings(
            enabled=enabled,
            preflight=preflight,
            followup=followup,
            command=command,
        )

    cfg = load_agent_config(workspace)
    raw = cfg.get("survey") if isinstance(cfg.get("survey"), dict) else {}
    if isinstance(raw, dict):
        if "enabled" in raw:
            val = raw.get("enabled")
            if isinstance(val, str):
                enabled = val.strip().lower() not in ("0", "false", "no", "off")
            else:
                enabled = bool(val)
        if "preflight" in raw:
            preflight = bool(raw.get("preflight"))
        if "followup" in raw:
            followup = bool(raw.get("followup"))
        if "command" in raw:
            command = bool(raw.get("command"))

    if not enabled:
        preflight = False
        followup = False
        command = False

    return SurveySettings(
        enabled=enabled,
        preflight=preflight,
        followup=followup,
        command=command,
    )


def survey_interactive_enabled(
    workspace: Path | None,
    context_session: ContextSession | None = None,
) -> bool:
    """
    当前是否启用交互式 survey（前置向导 / 助手 followup / 弹窗）。

    优先级：CLI/环境变量 > 会话 /survey off|on > agent.json。

    @param workspace 工作区根
    @param context_session 会话状态
    @return 是否启用
    """
    if survey_cli_disabled():
        return False
    if context_session is not None and context_session.survey_enabled is not None:
        return context_session.survey_enabled
    return resolve_survey_settings(workspace).enabled


def survey_preflight_enabled(
    workspace: Path | None,
    context_session: ContextSession | None = None,
) -> bool:
    """
    是否启用「梳理/整理」类请求的前置 survey。

    @param workspace 工作区根
    @param context_session 会话状态
    @return 是否启用
    """
    if not survey_interactive_enabled(workspace, context_session):
        return False
    return resolve_survey_settings(workspace).preflight


def survey_followup_enabled(
    workspace: Path | None,
    context_session: ContextSession | None = None,
) -> bool:
    """
    是否启用助手回复后的 survey followup。

    @param workspace 工作区根
    @param context_session 会话状态
    @return 是否启用
    """
    if not survey_interactive_enabled(workspace, context_session):
        return False
    return resolve_survey_settings(workspace).followup


def survey_command_enabled(
    workspace: Path | None,
    context_session: ContextSession | None = None,
) -> bool:
    """
    是否允许 /survey 命令打开向导。

    @param workspace 工作区根
    @param context_session 会话状态
    @return 是否启用
    """
    if not survey_interactive_enabled(workspace, context_session):
        return False
    return resolve_survey_settings(workspace).command


def format_survey_status(
    workspace: Path,
    context_session: ContextSession | None = None,
) -> str:
    """
    格式化 survey 开关状态说明。

    @param workspace 工作区根
    @param context_session 会话状态
    @return 多行文本
    """
    settings = resolve_survey_settings(workspace)
    effective = survey_interactive_enabled(workspace, context_session)
    lines = [
        f"Survey 交互: {'开' if effective else '关'}",
        f"  agent.json survey.enabled: {'开' if settings.enabled else '关'}",
        f"  CLI/环境 LLGRAPH_SURVEY: {'关' if survey_cli_disabled() else '未禁用'}",
    ]
    if context_session is not None and context_session.survey_enabled is not None:
        lines.append(
            f"  本会话覆盖: {'开' if context_session.survey_enabled else '关'}",
        )
    else:
        lines.append("  本会话覆盖: （无，跟随配置）")
    if effective:
        lines.append(
            f"  子项: preflight={'开' if settings.preflight else '关'}"
            f" · followup={'开' if settings.followup else '关'}"
            f" · command={'开' if settings.command else '关'}",
        )
    lines.append("  命令: /survey on | off | status")
    lines.append("  启动: llgraph --no-survey  或  LLGRAPH_SURVEY=0")
    return "\n".join(lines)
