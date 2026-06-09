"""沙箱后端探测（macOS / Linux）。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MACOS_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")


def detect_sandbox_backend() -> str | None:
    """
    当前平台可用的 OS 沙箱后端。

    @return macos_seatbelt | linux_bwrap | None
    """
    if sys.platform == "darwin" and MACOS_SANDBOX_EXEC.is_file():
        return "macos_seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "linux_bwrap"
    return None


def sandbox_backend_unavailable_message() -> str:
    """
    沙箱后端不可用时的说明。

    @return 多行文本
    """
    if sys.platform == "darwin":
        return "macOS 未找到 /usr/bin/sandbox-exec，无法启用 OS 沙箱。"
    if sys.platform.startswith("linux"):
        return (
            "Linux 未找到 bubblewrap (bwrap)。请安装后重试，例如: "
            "apt install bubblewrap / dnf install bubblewrap"
        )
    return f"当前平台 ({sys.platform}) 暂不支持 OS 沙箱（仅 macOS / Linux）。"
