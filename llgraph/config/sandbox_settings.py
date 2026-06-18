"""沙箱配置加载（~/.llgraph/sandbox.json + 工作区 .llgraph/sandbox.json）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llgraph.core.agent_config import LLGRAPH_DIR, USER_LLGRAPH_HOME, deep_merge_config

SANDBOX_CONFIG_FILENAME = "sandbox.json"

VALID_MODES = frozenset({"workspace_readwrite", "workspace_readonly"})
VALID_NETWORK = frozenset({"deny", "allow"})


@dataclass(frozen=True)
class SandboxSettings:
    """合并后的沙箱配置。"""

    enabled: bool
    mode: str
    network: str
    additional_readonly_paths: tuple[str, ...]
    additional_readwrite_paths: tuple[str, ...]
    allow_tmp_write: bool
    bind_write_mode: bool
    auto_enable_on_readonly: bool
    user_config_path: str
    workspace_config_path: str
    config_source: str


def _sandbox_config_path(root: Path) -> Path:
    return root.expanduser().resolve() / LLGRAPH_DIR / SANDBOX_CONFIG_FILENAME


def user_sandbox_config_path() -> Path:
    """
    用户级 sandbox.json 路径。

    @return ~/.llgraph/sandbox.json
    """
    return USER_LLGRAPH_HOME / SANDBOX_CONFIG_FILENAME


def workspace_sandbox_config_path(workspace: Path) -> Path:
    """
    工作区级 sandbox.json 路径。

    @param workspace 工作区根
    @return <workspace>/.llgraph/sandbox.json
    """
    return _sandbox_config_path(workspace)


def _load_json_dict(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_bool(value: object, default: bool) -> bool:
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


def _parse_path_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return tuple(items)


def resolve_sandbox_settings(workspace: Path) -> SandboxSettings:
    """
    解析沙箱配置（工作区覆盖用户）。

    @param workspace 工作区根
    @return SandboxSettings
    """
    user_path = user_sandbox_config_path()
    ws_path = workspace_sandbox_config_path(workspace)
    user_raw = _load_json_dict(user_path)
    ws_raw = _load_json_dict(ws_path)
    merged = deep_merge_config(user_raw, ws_raw) if user_raw or ws_raw else {}

    enabled = _parse_bool(merged.get("enabled"), False)
    mode_raw = str(merged.get("mode", "workspace_readwrite")).strip()
    mode = mode_raw if mode_raw in VALID_MODES else "workspace_readwrite"
    network_raw = str(merged.get("network", "deny")).strip()
    network = network_raw if network_raw in VALID_NETWORK else "deny"
    readonly = _parse_path_list(merged.get("additionalReadonlyPaths"))
    readwrite = _parse_path_list(merged.get("additionalReadwritePaths"))
    allow_tmp = _parse_bool(merged.get("allowTmpWrite"), True)
    bind_write_mode = _parse_bool(merged.get("bindWriteMode"), True)
    auto_enable_on_readonly = _parse_bool(merged.get("autoEnableOnReadonly"), False)

    sources: list[str] = []
    if user_path.is_file():
        sources.append(str(user_path))
    if ws_path.is_file():
        sources.append(str(ws_path))
    source = " + ".join(sources) if sources else "(默认，未找到 sandbox.json)"

    return SandboxSettings(
        enabled=enabled,
        mode=mode,
        network=network,
        additional_readonly_paths=readonly,
        additional_readwrite_paths=readwrite,
        allow_tmp_write=allow_tmp,
        bind_write_mode=bind_write_mode,
        auto_enable_on_readonly=auto_enable_on_readonly,
        user_config_path=str(user_path),
        workspace_config_path=str(ws_path),
        config_source=source,
    )


def format_sandbox_config_hint(workspace: Path) -> str:
    """
    沙箱权限调整提示（工具被拒绝时附加）。

    @param workspace 工作区根
    @return 多行说明
    """
    user_path = user_sandbox_config_path()
    ws_path = workspace_sandbox_config_path(workspace)
    return (
        "可在以下 sandbox.json 中调整沙箱权限（工作区覆盖用户）：\n"
        f"  用户: {user_path}\n"
        f"  工作区: {ws_path}\n"
        "常用字段: enabled, bindWriteMode(默认 true，mode 随只读/-w 联动), "
        "autoEnableOnReadonly(默认 false，true 时只读启动自动 OS 沙箱), "
        "mode, network(deny|allow), additionalReadonlyPaths, "
        "additionalReadwritePaths, allowTmpWrite"
    )
