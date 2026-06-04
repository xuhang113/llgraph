"""检索词解析（通用，无业务词典）。"""

import re

_KEYWORD_SPLIT = re.compile(r"[,，\s、/|]+")
_ENGLISH_TOKEN = re.compile(r"[a-zA-Z][\w-]*")
# 从 topic 末尾剥离的通用中文后缀（非业务映射）
_TOPIC_SUFFIXES = ("业务", "服务", "系统", "模块", "平台", "中心", "管理", "功能")


def parse_keyword_list(keywords: str) -> list[str]:
    """
    解析逗号/空格/顿号分隔的关键词列表。

    @param keywords 原始字符串
    @return 去重保序后的词列表
    """
    if not keywords or not keywords.strip():
        return []
    seen: set[str] = set()
    result: list[str] = []
    for part in _KEYWORD_SPLIT.split(keywords.strip()):
        token = part.strip()
        if not token:
            continue
        key = token.lower()
        if key not in seen:
            seen.add(key)
            result.append(token)
    return result


def _topic_token_variants(topic: str) -> list[str]:
    """从 topic 做通用切分：整句、去通用后缀、英文片段。"""
    variants: list[str] = []
    text = topic.strip()
    if not text:
        return variants

    variants.append(text)

    for suffix in _TOPIC_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            stem = text[: -len(suffix)].strip()
            if len(stem) >= 2:
                variants.append(stem)

    for part in _KEYWORD_SPLIT.split(text):
        part = part.strip()
        if part and part not in variants:
            variants.append(part)

    for match in _ENGLISH_TOKEN.finditer(text):
        token = match.group()
        if token not in variants:
            variants.append(token)

    return variants


def build_search_terms(topic: str = "", keywords: str = "") -> list[str]:
    """
    合并检索词：以调用方（模型）传入的 keywords 为主，topic 仅做通用切分补充。

    同义词扩展由模型在调用 search_workspace 时自行填入 keywords，
    例如 keywords=\"live,livestream,broadcast,acme-live\"，不在代码里维护业务词典。

    @param topic 用户问题中的主题描述（可选）
    @param keywords 多个检索词，逗号/空格分隔（建议 5～12 个）
    @return 去重后的检索词列表
    """
    merged: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        t = term.strip()
        if not t or len(t) < 2:
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        merged.append(t)

    for token in parse_keyword_list(keywords):
        add(token)

    for variant in _topic_token_variants(topic):
        add(variant)

    return merged
