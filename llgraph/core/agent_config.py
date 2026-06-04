"""agent.json 分层加载：用户 ~/.llgraph 为底，工作区 .llgraph 覆盖。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

LLGRAPH_DIR = ".llgraph"
USER_ONLY_AGENT_SECTIONS = frozenset({"web_search"})
AGENT_CONFIG_FILENAME = "agent.json"
USER_LLGRAPH_HOME = Path.home() / ".llgraph"


def user_agent_config_path() -> Path:
    """
    用户级 agent.json 路径。

    @return ~/.llgraph/agent.json
    """
    return USER_LLGRAPH_HOME / AGENT_CONFIG_FILENAME


def workspace_agent_config_path(workspace: Path) -> Path:
    """
    工作区级 agent.json 路径。

    @param workspace 工作区根
    @return <workspace>/.llgraph/agent.json
    """
    return workspace.expanduser().resolve() / LLGRAPH_DIR / AGENT_CONFIG_FILENAME


def _load_json_dict(path: Path) -> dict:
    """
    读取 JSON 对象为 dict。

    @param path 文件路径
    @return 配置字典；失败或不存在返回 {}
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def deep_merge_config(base: dict, override: dict) -> dict:
    """
    深度合并配置：override 覆盖 base。

    - 同为 dict 的键：递归合并
    - 标量、列表等：override 整段替换

    @param base 底层配置（用户级）
    @param override 覆盖配置（工作区级）
    @return 合并结果
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_user_agent_config() -> dict:
    """
    仅加载用户级 ~/.llgraph/agent.json。

    @return 配置字典
    """
    return _load_json_dict(user_agent_config_path())


def load_workspace_agent_config(workspace: Path) -> dict:
    """
    仅加载工作区 .llgraph/agent.json。

    @param workspace 工作区根
    @return 配置字典
    """
    return _load_json_dict(workspace_agent_config_path(workspace))


def load_agent_config(workspace: Path | None = None) -> dict:
    """
    合并加载 agent.json（工作区覆盖用户）。

    优先级（后者覆盖前者）：
    1. ~/.llgraph/agent.json（用户默认）
    2. <workspace>/.llgraph/agent.json（项目覆盖）

    会话内 /model、/log 等运行时覆盖不在此函数内，由各模块单独处理。

    @param workspace 工作区根；None 时仅返回用户配置
    @return 合并后的配置字典
    """
    user_cfg = load_user_agent_config()
    if workspace is None:
        return user_cfg

    ws_cfg = load_workspace_agent_config(workspace)
    if not user_cfg:
        return ws_cfg
    if not ws_cfg:
        return user_cfg
    return deep_merge_config(user_cfg, ws_cfg)


def format_agent_config_sources(workspace: Path) -> str:
    """
    配置来源摘要（/config 用）。

    @param workspace 工作区根
    @return 多行说明
    """
    user_path = user_agent_config_path()
    ws_path = workspace_agent_config_path(workspace)
    user_exists = user_path.is_file()
    ws_exists = ws_path.is_file()

    lines = [
        "agent.json 配置层级（类似 Cursor：用户默认 + 工作区覆盖）",
        "",
        f"用户: {user_path}  {'(已加载)' if user_exists else '(不存在)'}",
        f"工作区: {ws_path}  {'(已加载)' if ws_exists else '(不存在)'}",
        "",
        "合并规则: 工作区同名字段覆盖用户；嵌套对象递归合并；数组整段替换。",
        f"仅用户级（工作区不生效）: {', '.join(sorted(USER_ONLY_AGENT_SECTIONS))}",
        "运行时覆盖: /model、/log 仅当前会话，不写盘。",
        "API 凭据: ~/.config/llgraph/llgraph.env（LLGRAPH_*，不在 agent.json）",
    ]

    if not user_exists and not ws_exists:
        lines.append("")
        lines.append(
            "提示: 可复制 examples/user-llgraph/agent.json 到 ~/.llgraph/agent.json；"
            "工作区用 llgraph --init-config -C <目录>"
        )
    return "\n".join(lines)
