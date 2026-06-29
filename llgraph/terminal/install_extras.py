"""可选 pip extra 检测与 /help deps 安装建议。"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtraSpec:
    """
    pyproject optional-dependencies 条目说明。

    @param extra pip extra 名（terminal / index / …）
    @param title 简短标题
    @param features 能力说明
    @param modules 探测用 import 名（全部存在视为已安装）
    @param pip_hint 单独安装示例
    """

    extra: str
    title: str
    features: str
    modules: tuple[str, ...]
    pip_hint: str


EXTRA_SPECS: tuple[ExtraSpec, ...] = (
    ExtraSpec(
        extra="terminal",
        title="终端 Rich 渲染",
        features="/trace rich on、Markdown 终端高亮（默认关，需手动开）",
        modules=("rich",),
        pip_hint="pip install -e '.[terminal]'",
    ),
    ExtraSpec(
        extra="index",
        title="代码向量索引",
        features="/index、llgraph index/search、上下文检索（lancedb + 向量模型）",
        modules=("lancedb", "sentence_transformers"),
        pip_hint="pip install -e '.[index]'",
    ),
    ExtraSpec(
        extra="ast",
        title="AST 分块",
        features="索引时按语法树切分（Python/Java/JS 等）",
        modules=("tree_sitter",),
        pip_hint="pip install -e '.[ast]'",
    ),
    ExtraSpec(
        extra="watch",
        title="索引文件监听",
        features="/watch on、保存文件后自动增量索引",
        modules=("watchdog",),
        pip_hint="pip install -e '.[watch]'",
    ),
    ExtraSpec(
        extra="mcp",
        title="MCP 工具",
        features="agent.json 中 MCP 服务器、/tools 展示外部工具",
        modules=("mcp",),
        pip_hint="pip install -e '.[mcp]'",
    ),
    ExtraSpec(
        extra="search",
        title="Web 搜索",
        features="/web on、web_search 工具（Tavily）",
        modules=("tavily",),
        pip_hint="pip install -e '.[search]'",
    ),
    ExtraSpec(
        extra="web",
        title="Web Console",
        features="llgraph web、图片 multipart 上传（fastapi + uvicorn + python-multipart）",
        modules=("fastapi", "uvicorn", "multipart"),
        pip_hint="pip install -e '.[web]'  或  uv sync --extra web",
    ),
)

BASE_INSTALL = "pip install -e .  或  uv sync"
FULL_DEV_HINT = "uv sync --extra web --extra terminal --extra index --extra watch --extra mcp --extra search --extra ast --extra dev"
SETUP_SCRIPT_HINT = "./scripts/setup.sh dev"


def _module_available(name: str) -> bool:
    """
    模块是否可导入。

    @param name import 名
    @return 是否已安装
    """
    return importlib.util.find_spec(name) is not None


def extra_installed(spec: ExtraSpec) -> bool:
    """
    可选 extra 是否已满足（其 modules 均可 import）。

    @param spec ExtraSpec
    @return 是否已安装
    """
    return all(_module_available(mod) for mod in spec.modules)


def missing_extras() -> list[ExtraSpec]:
    """
    当前环境未安装的 optional extra 列表。

    @return 缺失项
    """
    return [spec for spec in EXTRA_SPECS if not extra_installed(spec)]


def suggest_pip_install(*, include_terminal: bool = True) -> str:
    """
    根据缺失项生成一条 pip 建议（可组合 extra）。

    @param include_terminal 是否把 terminal 列入建议
    @return 安装命令
    """
    missing = [spec.extra for spec in missing_extras()]
    if not include_terminal:
        missing = [name for name in missing if name != "terminal"]
    if not missing:
        return BASE_INSTALL + "  # 常用 optional 已就绪"
    if command_has_uv():
        parts = " ".join(f"--extra {name}" for name in missing)
        return f"uv sync {parts}  或  {SETUP_SCRIPT_HINT}"
    return f"pip install -e '.[{','.join(missing)}]'"


def command_has_uv() -> bool:
    """PATH 中是否可用 uv。"""
    import shutil

    return shutil.which("uv") is not None


def format_install_extras_report(*, missing_only: bool = False) -> str:
    """
    /help deps 正文：各 extra 状态与安装建议。

    @param missing_only 仅列出未安装项
    @return 多行 Markdown/纯文本
    """
    lines: list[str] = [
        "llgraph 可选依赖（pip extras）",
        "=============================",
        "",
        "说明:",
        f"  {BASE_INSTALL}  只装核心依赖（langgraph、prompt-toolkit 等）",
        "  下列能力需额外 extra；可按需组合，例如:",
        f"  {FULL_DEV_HINT}",
        "",
    ]

    missing_names: list[str] = []
    for spec in EXTRA_SPECS:
        ok = extra_installed(spec)
        if missing_only and ok:
            continue
        mark = "✓ 已安装" if ok else "✗ 未安装"
        lines.append(f"[{mark}] [{spec.extra}] {spec.title}")
        lines.append(f"      {spec.features}")
        lines.append(f"      → {spec.pip_hint}")
        if not ok:
            missing_names.append(spec.extra)
        lines.append("")

    lines.append("【推荐】")
    if missing_names:
        lines.append(f"  一键安装: {SETUP_SCRIPT_HINT}")
        lines.append(f"  或补装: {suggest_pip_install()}")
        lines.append("")
        lines.append("  对照当前用法:")
        need: list[str] = []
        if "index" in missing_names or "watch" in missing_names:
            need.append("要做代码索引/检索 → [index]；自动增量 → 再加 [watch]")
        if "mcp" in missing_names:
            need.append("要用 MCP 工具 → [mcp]")
        if "search" in missing_names:
            need.append("要 /web on → [search]（另配 Tavily API Key）")
        if "terminal" in missing_names:
            need.append("要 Rich 终端渲染 → [terminal]，会话内 /trace rich on")
        if "ast" in missing_names:
            need.append("索引要语法树分块 → [ast]（通常与 [index] 一起装）")
        if need:
            for item in need:
                lines.append(f"  · {item}")
        else:
            lines.append("  · 按上表 [未安装] 项选择 extra 即可")
    else:
        lines.append("  常用 optional 均已安装，无需额外 pip 操作。")

    lines.append("")
    lines.append("查看已装包: pip show llgraph  |  pip list | rg 'rich|lancedb|watchdog|mcp|tavily'")
    return "\n".join(lines)
