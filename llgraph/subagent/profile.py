"""Subagent Profile：explore / worker / planner / general。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class SubagentProfile:
    """
    子 Agent 配置档。

    @param kind 唯一 kind
    @param title 默认展示名
    @param allow_write 默认是否可写
    @param include_mcp 是否继承父 MCP 工具
    @param max_turns 默认步数；None 用父/全局
    @param thread_suffix 子 thread 后缀模板，含 {sub_id}
    @param role_loader 返回角色块；None 用内置 explore 文案
    """

    kind: str
    title: str
    allow_write: bool = False
    include_mcp: bool = True
    max_turns: int | None = None
    thread_suffix: str = ":{kind}:{sub_id}"
    role_loader: Callable[[Path], str] | None = None

    def format_sub_thread(self, parent_thread_id: str, sub_id: str) -> str:
        suffix = self.thread_suffix.format(kind=self.kind, sub_id=sub_id)
        if not suffix.startswith(":"):
            suffix = f":{suffix}"
        return f"{parent_thread_id}{suffix}"


def _load_explore_role(workspace: Path) -> str:
    from llgraph.loaders.prompt_loader import prompt_text, render_prompt

    raw = prompt_text("subagent", "explore", "role")
    if not raw:
        return (
            "\n\n--- Explore Subagent ---\n"
            "你是只读代码库探索子 Agent。广搜定位文件与调用链，用尽可能少的步骤给出结论。\n"
            "禁止修改文件。最终回复须为简洁摘要：相关路径、关键发现、未决问题。\n"
        )
    body = render_prompt(raw, workspace=str(workspace))
    return body if body.startswith("\n") else f"\n\n{body}\n"


def _default_max_turns_explore(workspace: Path | None = None) -> int:
    from llgraph.core.agent_config import load_agent_config
    from llgraph.core.react_limits import parse_react_max_turns

    cfg = load_agent_config(workspace) if workspace else {}
    section = cfg.get("subagent") if isinstance(cfg.get("subagent"), dict) else {}
    explore = section.get("explore") if isinstance(section.get("explore"), dict) else {}
    return parse_react_max_turns(explore.get("max_turns"), default=40)


_PROFILES: dict[str, SubagentProfile] = {
    "explore": SubagentProfile(
        kind="explore",
        title="Explore",
        allow_write=False,
        include_mcp=False,
        max_turns=40,
        thread_suffix=":explore:{sub_id}",
        role_loader=_load_explore_role,
    ),
    "general": SubagentProfile(
        kind="general",
        title="Subagent",
        allow_write=False,
        include_mcp=True,
        max_turns=None,
        thread_suffix=":subagent:{sub_id}",
    ),
    "worker": SubagentProfile(
        kind="worker",
        title="Worker",
        allow_write=True,
        include_mcp=True,
        max_turns=None,
        thread_suffix=":worker:{sub_id}",
    ),
    "planner": SubagentProfile(
        kind="planner",
        title="Planner",
        allow_write=False,
        include_mcp=True,
        max_turns=None,
        thread_suffix=":planner:{sub_id}",
    ),
}


def get_subagent_profile(kind: str, *, workspace: Path | None = None) -> SubagentProfile:
    """
    按 kind 取 Profile；explore 的 max_turns 可读 agent.json。

    @param kind explore | worker | planner | general
    @param workspace 可选，用于覆盖 explore max_turns
    """
    key = (kind or "explore").strip().lower()
    base = _PROFILES.get(key) or _PROFILES["general"]
    if key == "explore" and workspace is not None:
        return SubagentProfile(
            kind=base.kind,
            title=base.title,
            allow_write=base.allow_write,
            include_mcp=base.include_mcp,
            max_turns=_default_max_turns_explore(workspace),
            thread_suffix=base.thread_suffix,
            role_loader=base.role_loader,
        )
    return base


def list_subagent_profiles() -> list[SubagentProfile]:
    """已注册 Profile 列表。"""
    return list(_PROFILES.values())


def resolve_role_block(profile: SubagentProfile, workspace: Path, override: str | None = None) -> str:
    """角色说明块：显式 override 优先。"""
    if override and override.strip():
        return override if override.startswith("\n") else f"\n\n{override.strip()}\n"
    if profile.role_loader is not None:
        return profile.role_loader(workspace)
    return f"\n\n--- {profile.title} Subagent ---\n完成指派任务后给出简洁摘要。\n"
