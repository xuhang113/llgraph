"""工作区 markdowns/ 目录轻量索引（供模型优先查阅文档）。"""

import re
from pathlib import Path

_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_MAX_FILES = 80
_MAX_CHARS = 3500


def build_markdowns_index(workspace: Path) -> str:
    """
    生成 markdowns/*.md 的紧凑索引。

    @param workspace 工作区根
    @return 索引文本，无 markdowns 目录时返回空串
    """
    md_dir = workspace / "markdowns"
    if not md_dir.is_dir():
        return ""

    lines: list[str] = [
        "## 工作区文档索引（markdowns/）",
        "业务归属、项目对照类问题请先 read_file 相关文档，再扫代码。",
        "",
    ]
    count = 0
    for path in sorted(md_dir.rglob("*.md")):
        if count >= _MAX_FILES:
            lines.append(f"- … 另有更多文档未列出（共>{_MAX_FILES}）")
            break
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(workspace).as_posix()
            head = path.read_text(encoding="utf-8")[:2000]
        except OSError:
            continue
        title_match = _TITLE_RE.search(head)
        title = title_match.group(1).strip() if title_match else path.stem
        lines.append(f"- `{rel}` — {title}")
        count += 1

    if count == 0:
        return ""

    text = "\n".join(lines)
    if len(text) > _MAX_CHARS:
        return text[: _MAX_CHARS - 20] + "\n…（索引已截断）"
    return text
