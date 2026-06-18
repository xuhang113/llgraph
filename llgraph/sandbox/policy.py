"""沙箱路径策略：与 sandbox.json 一致的内置工具权限检查。"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from llgraph.config.sandbox_settings import (
    SandboxSettings,
    format_sandbox_config_hint,
)
from llgraph.sandbox.platform import detect_sandbox_backend, sandbox_backend_unavailable_message


@dataclass(frozen=True)
class SandboxPolicy:
    """
    运行时沙箱策略（配置 + 工作区 + 是否生效）。

    内置文件工具走本策略；Shell 子进程另经 OS 沙箱包装。
    """

    active: bool
    mode: str
    network: str
    workspace: Path
    readonly_roots: tuple[Path, ...]
    readwrite_roots: tuple[Path, ...]
    allow_tmp_write: bool
    backend: str | None
    user_config_path: str
    workspace_config_path: str

    @property
    def enabled(self) -> bool:
        """是否启用沙箱（active 且后端可用）。"""
        return self.active and self.backend is not None

    def startup_warning(self) -> str | None:
        """
        启动时沙箱不可用警告。

        @return 警告文本；无则 None
        """
        if not self.active:
            return None
        if self.backend is not None:
            return None
        return sandbox_backend_unavailable_message()

    def _path_in_roots(self, path: Path, roots: tuple[Path, ...]) -> bool:
        resolved = path.expanduser().resolve()
        for root in roots:
            if not root.exists():
                continue
            try:
                resolved.relative_to(root.expanduser().resolve())
                return True
            except ValueError:
                continue
        return False

    def check_read(self, path: Path) -> str | None:
        """
        检查路径是否允许读取。

        @param path 绝对路径
        @return 拒绝原因；允许则 None
        """
        if not self.enabled:
            return None
        if self._path_in_roots(path, self.readonly_roots):
            return None
        if self._path_in_roots(path, self.readwrite_roots):
            return None
        return f"沙箱拒绝读取: {path}"

    def check_write(self, path: Path) -> str | None:
        """
        检查路径是否允许写入。

        @param path 绝对路径
        @return 拒绝原因；允许则 None
        """
        if not self.enabled:
            return None
        if self.mode == "workspace_readonly":
            return f"沙箱为只读模式，禁止写入: {path}"
        if self._path_in_roots(path, self.readwrite_roots):
            return None
        return f"沙箱拒绝写入: {path}"

    def format_denial(self, reason: str) -> str:
        """
        格式化拒绝信息（含配置路径提示）。

        @param reason 拒绝原因
        @return 多行错误说明
        """
        hint = format_sandbox_config_hint(self.workspace)
        return f"错误: {reason}\n无权限执行该操作。\n{hint}"

    def create_seatbelt_profile_file(self) -> Path:
        """
        生成 macOS Seatbelt profile 临时文件。

        @return profile 文件路径
        """
        from llgraph.sandbox.macos import build_seatbelt_profile

        content = build_seatbelt_profile(self)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sb",
            delete=False,
            encoding="utf-8",
        )
        handle.write(content)
        handle.close()
        return Path(handle.name)


def _resolve_sandbox_active(
    settings: SandboxSettings,
    *,
    cli_enabled: bool | None,
    allow_write: bool,
) -> bool:
    """
    是否启用 OS 沙箱（CLI 优先，其次只读自动启用，最后 sandbox.json enabled）。

    @param settings 沙箱配置
    @param cli_enabled True=--sandbox，False=--no-sandbox，None=沿用规则
    @param allow_write 当前是否可写（-w / /write on）
    @return 是否 active
    """
    if cli_enabled is True:
        return True
    if cli_enabled is False:
        return False
    if settings.auto_enable_on_readonly and not allow_write:
        return True
    return settings.enabled


def _resolve_effective_sandbox_mode(
    settings: SandboxSettings,
    *,
    active: bool,
    allow_write: bool,
) -> str:
    """
    解析运行时沙箱 mode（bindWriteMode 时随 allow_write 切换）。

    @param settings 沙箱配置
    @param active 沙箱是否启用
    @param allow_write 当前是否可写
    @return workspace_readonly 或 workspace_readwrite
    """
    if active and settings.bind_write_mode:
        if allow_write:
            return "workspace_readwrite"
        return "workspace_readonly"
    return settings.mode


def build_sandbox_policy(
    workspace: Path,
    settings: SandboxSettings,
    *,
    cli_enabled: bool | None = None,
    allow_write: bool = False,
) -> SandboxPolicy:
    """
    根据配置与 CLI 构建运行时沙箱策略。

    @param workspace 工作区根
    @param settings 合并后的 sandbox.json
    @param cli_enabled True=--sandbox，False=--no-sandbox，None=沿用配置
    @param allow_write 当前是否可写；bindWriteMode 时决定 workspace_readonly/readwrite
    @return SandboxPolicy
    """
    ws = workspace.expanduser().resolve()
    user_home = Path.home().expanduser().resolve()
    active = _resolve_sandbox_active(
        settings,
        cli_enabled=cli_enabled,
        allow_write=allow_write,
    )
    effective_mode = _resolve_effective_sandbox_mode(
        settings,
        active=active,
        allow_write=allow_write,
    )

    # 工作区 + 用户主目录（~/.llgraph/skills 等）始终可读，与无沙箱时 read_file 白名单对齐
    readonly: list[Path] = [ws, user_home]
    readwrite: list[Path] = []

    for raw in settings.additional_readonly_paths:
        readonly.append(Path(raw).expanduser().resolve())

    if effective_mode == "workspace_readwrite":
        readwrite.append(ws)
        for raw in settings.additional_readwrite_paths:
            readwrite.append(Path(raw).expanduser().resolve())
    else:
        for raw in settings.additional_readwrite_paths:
            readonly.append(Path(raw).expanduser().resolve())

    if settings.allow_tmp_write:
        readwrite.extend([
            Path("/tmp"),
            Path("/private/tmp"),
        ])

    backend = detect_sandbox_backend() if active else None

    return SandboxPolicy(
        active=active,
        mode=effective_mode,
        network=settings.network,
        workspace=ws,
        readonly_roots=tuple(dict.fromkeys(readonly)),
        readwrite_roots=tuple(dict.fromkeys(readwrite)),
        allow_tmp_write=settings.allow_tmp_write,
        backend=backend,
        user_config_path=settings.user_config_path,
        workspace_config_path=settings.workspace_config_path,
    )
