"""工作区 .llgraph 配置目录：初始化模板（与 Cursor 独立）。"""

import shutil
from pathlib import Path

LLGRAPH_DIR_NAME = ".llgraph"
SKILLS_DIR_NAME = "skills"
RULES_DIR_NAME = "rules"


def _package_examples_root() -> Path:
    """
    仓库 examples/ 根目录（与 Python 包 llgraph/ 同级）。

    workspace_config 位于 llgraph/config/，向上三级到仓库根。
    """
    return Path(__file__).resolve().parent.parent.parent / "examples"


def package_default_config_dir() -> Path:
    """
    包内默认配置模板目录。

    @return examples/default-workspace/.llgraph 路径
    """
    return _package_examples_root() / "default-workspace" / LLGRAPH_DIR_NAME


def package_user_config_dir() -> Path:
    """
    包内用户级配置模板目录。

    @return examples/user-llgraph 路径
    """
    return _package_examples_root() / "user-llgraph"


def is_packaged_example_workspace(resolved: str | Path) -> bool:
    """
    是否为 llgraph 包内 examples 模板目录（不应出现在 Web 最近工作区）。

    @param resolved 工作区绝对路径
    @return 是模板目录返回 True
    """
    try:
        path = Path(resolved).expanduser().resolve()
    except OSError:
        return False
    examples_root = _package_examples_root().resolve()
    for name in ("user-llgraph", "default-workspace"):
        try:
            if path == (examples_root / name).resolve():
                return True
        except OSError:
            continue
    return False


def init_user_llgraph(*, force: bool = False) -> list[str]:
    """
    将默认用户配置复制到 ~/.llgraph/（不覆盖已有文件，除非 force）。

    @param force 为 True 时覆盖已存在的同名文件
    @return 已复制/更新的相对路径列表
    """
    from llgraph.core.agent_config import USER_LLGRAPH_HOME

    src = package_user_config_dir()
    if not src.is_dir():
        raise FileNotFoundError(f"缺少用户配置模板: {src}")

    dest_root = USER_LLGRAPH_HOME
    copied: list[str] = []

    for src_file in sorted(src.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dest_file = dest_root / rel
        if dest_file.exists() and not force:
            continue
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied.append(f"~/.llgraph/{rel.as_posix()}")

    return copied


def init_workspace_llgraph(
    workspace: Path,
    *,
    force: bool = False,
) -> list[str]:
    """
    将包内默认 .llgraph 复制到工作区（不覆盖已有文件，除非 force）。

    @param workspace 工作区根
    @param force 为 True 时覆盖已存在的同名文件
    @return 已复制/更新的相对路径列表
    """
    src = package_default_config_dir()
    if not src.is_dir():
        raise FileNotFoundError(f"缺少默认配置模板: {src}")

    dest_root = workspace / LLGRAPH_DIR_NAME
    copied: list[str] = []

    for src_file in sorted(src.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dest_file = dest_root / rel
        if dest_file.exists() and not force:
            continue
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied.append(f"{LLGRAPH_DIR_NAME}/{rel.as_posix()}")

    return copied
