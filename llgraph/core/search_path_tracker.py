"""从用户消息提取检索词并生成批量 grep 的 harness 提示（不拦截工具）。"""

from __future__ import annotations

import re
from urllib.parse import urlparse

_LITERAL_STOP = frozenset(
    {
        "com",
        "cn",
        "net",
        "org",
        "http",
        "https",
        "www",
        "api",
        "get",
        "post",
        "put",
        "delete",
        "json",
        "html",
        "the",
        "and",
        "for",
        "from",
        "with",
        "path",
        "code",
        "list",
        "data",
        "info",
        "design",
        "chart",
        "这个",
        "什么",
        "有没有",
        "影响",
        "重复",
    }
)

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_TB_TABLE_RE = re.compile(r"\btb_[a-z][\w]*\b", re.IGNORECASE)
_FIELD_RE = re.compile(
    r"[a-z][\w]*_(?:id|ids|type|code|name|no|num)",
    re.IGNORECASE,
)


def _add_term(seen: set[str], out: list[str], raw: str) -> None:
    token = raw.strip().strip("\"'`").lower()
    if len(token) < 2 or token in _LITERAL_STOP or token in seen:
        return
    if token.isdigit():
        return
    seen.add(token)
    out.append(token)


def expand_table_search_terms(table: str) -> list[str]:
    """
    由表名 tb_* 展开 DO/Mapper 等常见检索词。

    @param table 表名
    @return 检索词列表（含表名本身）
    """
    name = table.strip().lower()
    terms = [name]
    if not name.startswith("tb_"):
        return terms
    rest = name[3:]
    parts = [p for p in rest.split("_") if p]
    if not parts:
        return terms
    camel = "".join(p.capitalize() for p in parts)
    for suffix in ("", "DO", "Mapper", "Entity", "Service"):
        candidate = f"{camel}{suffix}"
        if candidate.lower() not in {t.lower() for t in terms}:
            terms.append(candidate)
    return terms


def extract_grep_batch_terms(text: str) -> tuple[str, ...]:
    """
    从用户消息收集应合并进单次 grep 的检索词。

    @param text 用户输入
    @return 去重后的词列表（保持顺序）
    """
    if not text or not text.strip():
        return ()

    seen: set[str] = set()
    out: list[str] = []

    for table in _TB_TABLE_RE.findall(text):
        for term in expand_table_search_terms(table):
            _add_term(seen, out, term)

    for field in _FIELD_RE.findall(text):
        _add_term(seen, out, field)

    for url in _URL_RE.findall(text):
        parsed = urlparse(url)
        if parsed.hostname:
            for part in parsed.hostname.split("."):
                _add_term(seen, out, part)
        for seg in parsed.path.split("/"):
            if seg and re.fullmatch(r"[\w.-]+", seg):
                _add_term(seen, out, seg)

    for match in re.finditer(r"[A-Za-z][\w.-]{2,}", text):
        _add_term(seen, out, match.group())

    return tuple(out)


def format_retrieval_batch_hint(user_message: str) -> str:
    """
    生成批量检索 harness 提示（注入 workspace-context）。

    @param user_message 用户输入
    @return Markdown 块；词不足时为空
    """
    terms = extract_grep_batch_terms(user_message)
    if len(terms) < 2:
        return ""

    grep_pat = "|".join(terms[:14])
    quoted = "、".join(f"`{t}`" for t in terms[:10])
    if len(terms) > 10:
        quoted += f" 等 {len(terms)} 项"

    return "\n".join(
        [
            "## 检索批量化（减少 LLM 往返）",
            f"用户消息含多个可合并检索词：{quoted}",
            f"**首轮建议一条 grep**（勿拆多轮）：`grep_files(pattern=\"{grep_pat}\", path=\".\")`",
            "禁止：先 grep 表名、下一轮再 grep 类名/字段名；应一次 `pattern=\"表名|类名|字段|…\"`。",
            "grep 命中后，**同轮或下一轮**用 `read_files(paths=[...])` 一次读齐 DO/Mapper/Service（≤8 个 path），勿分多轮 read。",
            "目标：单表/单点问题 **2～4 次模型决策**内够答就停。",
        ]
    )


# 兼容旧名
extract_user_search_literals = extract_grep_batch_terms
format_search_anchor_hint = format_retrieval_batch_hint
