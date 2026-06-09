"""斜杠补全目录：Skills、自定义 Commands、常用内置元命令。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.config.catalog_paths import scope_label
from llgraph.loaders.commands_loader import discover_commands
from llgraph.loaders.skills_loader import discover_skills

# 内置元命令（/help 等）；Skills 与 .llgraph/commands 优先同名项
_SLASH_META_ITEMS: tuple[tuple[str, str], ...] = (
    ("help", "交互帮助与快捷键"),
    ("skill", "列出 / 启用技能"),
    ("trace", "过程展示档位（steps / all / reply）"),
    ("rule", "规则列表与开关"),
    ("compress", "压缩会话历史"),
    ("context", "上下文 token 用量"),
    ("session", "会话切换与管理"),
    ("index", "代码索引构建与状态"),
    ("survey", "梳理确认向导"),
    ("paste", "多行粘贴模式"),
    ("model", "切换模型"),
    ("config", "配置文件路径"),
    ("commands", "列出自定义命令"),
    ("review", "代码评审（Claude CLI）"),
    ("log", "执行日志"),
    ("tools", "工具列表"),
    ("web", "联网搜索开关"),
    ("write", "写权限说明"),
)


@dataclass(frozen=True)
class SlashCatalogItem:
    """斜杠补全条目。"""

    name: str
    description: str
    category: str
    insert_text: str
    origin: str = ""


def build_slash_catalog(workspace: Path) -> list[SlashCatalogItem]:
    """
    构建当前工作区可用的斜杠补全列表。

    @param workspace 工作区根
    @return Skills → Commands → 内置 顺序；同名仅保留先出现的
    """
    seen: set[str] = set()
    items: list[SlashCatalogItem] = []

    def _add(item: SlashCatalogItem) -> None:
        key = item.name.lower()
        if key in seen:
            return
        seen.add(key)
        items.append(item)

    for skill in discover_skills(workspace):
        origin = scope_label(skill.scope)
        desc = skill.description.strip() or skill.name
        if origin:
            desc = f"{desc} ({origin})"
        _add(
            SlashCatalogItem(
                name=skill.name,
                description=desc,
                category="Skills",
                insert_text=f"/{skill.name} ",
                origin=origin,
            )
        )

    for cmd in discover_commands(workspace):
        _add(
            SlashCatalogItem(
                name=cmd.name,
                description=cmd.description.strip() or cmd.name,
                category="Commands",
                insert_text=f"/{cmd.name} ",
            )
        )

    for name, desc in _SLASH_META_ITEMS:
        _add(
            SlashCatalogItem(
                name=name,
                description=desc,
                category="内置",
                insert_text=f"/{name} ",
            )
        )

    return items


def slash_category_badge(category: str) -> str:
    """
    斜杠补全类型标志（展示在说明前）。

    @param category Skills | Commands | 内置
    @return 如 [skill]、[command]、[meta]
    """
    mapping = {
        "Skills": "[skill]",
        "Commands": "[command]",
        "内置": "[meta]",
    }
    return mapping.get(category, f"[{category.lower()}]")


def filter_slash_catalog(
    catalog: list[SlashCatalogItem],
    partial: str,
    *,
    limit: int = 24,
) -> list[SlashCatalogItem]:
    """
    按 / 后首 token 前缀过滤（不含空格时生效）。

    @param catalog 全量目录
    @param partial / 之后、空格之前的片段
    @param limit 最多返回条数
    @return 过滤结果
    """
    key = partial.lstrip("/").lower()
    if not key:
        matched = catalog
    else:
        matched = [item for item in catalog if item.name.lower().startswith(key)]
    order = {"Skills": 0, "Commands": 1, "内置": 2}
    matched = sorted(
        matched,
        key=lambda item: (order.get(item.category, 9), item.name.lower()),
    )
    return matched[:limit]


def parse_slash_partial(line: str, *, cursor_at_end: bool = True) -> str | None:
    """
    解析可补全的斜杠前缀；已有空格则视为在写任务描述，不再补全。

    @param line 当前输入行
    @param cursor_at_end 光标是否在行尾（终端默认真）
    @return partial 或 None（不可补全）
    """
    if not line.startswith("/"):
        return None
    if not cursor_at_end:
        return None
    rest = line[1:]
    if " " in rest:
        return None
    return rest
