"""索引 manifest：文件 path → sha256。"""

import json
from pathlib import Path

from llgraph.code_index.paths import ensure_index_dirs, manifest_path


def load_manifest(workspace: Path) -> dict[str, str]:
    """
    加载 manifest。

    @param workspace 工作区根
    @return rel_path -> file_sha256
    """
    path = manifest_path(workspace)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        files = data.get("files", {})
        if isinstance(files, dict):
            return {str(k): str(v) for k, v in files.items()}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def clear_manifest(workspace: Path) -> None:
    """清空 manifest（重建索引前）。"""
    save_manifest(workspace, {})


def save_manifest(workspace: Path, files: dict[str, str]) -> None:
    """
    保存 manifest。

    @param workspace 工作区根
    @param files rel_path -> sha256
    """
    ensure_index_dirs(workspace)
    payload = {"files": dict(sorted(files.items()))}
    manifest_path(workspace).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
