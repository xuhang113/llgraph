"""macOS sandbox-exec Seatbelt profile 生成。"""

from __future__ import annotations

from llgraph.sandbox.policy import SandboxPolicy


def _quote_subpath(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def build_seatbelt_profile(policy: SandboxPolicy) -> str:
    """
    构建 Seatbelt profile 文本。

    @param policy 沙箱策略
    @return .sb profile 内容
    """
    lines = [
        "(version 1)",
        "(deny default)",
    ]

    for root in policy.readonly_roots:
        if root.exists():
            lines.append(f'(allow file-read* (subpath "{_quote_subpath(str(root))}"))')

    if policy.mode == "workspace_readwrite":
        for root in policy.readwrite_roots:
            if root.exists():
                sub = _quote_subpath(str(root))
                lines.append(f'(allow file-read* file-write* (subpath "{sub}"))')

    lines.extend([
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
    ])

    if policy.network == "allow":
        lines.append("(allow network*)")

    return "\n".join(lines) + "\n"
