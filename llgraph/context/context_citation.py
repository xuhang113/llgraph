"""下游引用信号：识别历史工具结果是否被后续 AI 回复引用。

用途：
- 裁剪保护：被后续结论/推理引用过的 ToolMessage 优先保留全文（非纯 recency）。
- 预览增强：真被裁剪时，指针里附「被引用行 ± 上下文」，减少误导。

只用确定性信号（path:line / 文件路径），不调 LLM。
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

_CODE_EXT = r"(?:vue|ts|tsx|js|jsx|java|kt|go|py|rb|rs|c|cc|cpp|h|hpp|cs|php|scala|xml|yml|yaml|json|sql|proto)"
# path:line 锚点，如 ChartDataController.java:167、a/b/Foo.vue:12
_PATH_LINE_RE = re.compile(rf"([\w./\\-]+\.{_CODE_EXT}):(\d{{1,6}})\b", re.IGNORECASE)
# 文件路径（含目录或单文件名），无行号
_FILE_PATH_RE = re.compile(rf"\b((?:[\w.-]+[/\\])*[\w.-]+\.{_CODE_EXT})\b", re.IGNORECASE)
# read 行标记：`   167| ...`
_READ_LINE_MARK_RE = re.compile(r"^\s*(\d{1,6})\s*\|", re.MULTILINE)
# read 段头：`--- path (行 90-175 / 共 ...) ---`
_READ_PATH_HDR_RE = re.compile(r"^---\s+(.+?)\s+\(行\s+(\d+)", re.MULTILINE)
# grep 命中头：`--- path:167 ---`
_GREP_HIT_HDR_RE = re.compile(rf"^---\s+([\w./\\-]+\.{_CODE_EXT}):(\d{{1,6}})", re.IGNORECASE | re.MULTILINE)


def _basename(path: str) -> str:
    return re.split(r"[/\\]", path.strip())[-1]


def _visible_text(msg: BaseMessage) -> str:
    """AIMessage 的用户可见正文 + 规划行（不含超长 thinking）。"""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def extract_reference_anchors(text: str) -> set[str]:
    """
    从文本抽取可用于引用比对的锚点集合。

    锚点两类（归一化小写）：
    - `pathline:<basename>:<line>`（强信号）
    - `file:<basename>`（弱信号，文件名去目录）

    @param text 任意文本
    @return 锚点集合
    """
    anchors: set[str] = set()
    sample = str(text or "")
    if not sample:
        return anchors
    for m in _PATH_LINE_RE.finditer(sample):
        base = _basename(m.group(1)).lower()
        anchors.add(f"pathline:{base}:{m.group(2)}")
        anchors.add(f"file:{base}")
    for m in _FILE_PATH_RE.finditer(sample):
        base = _basename(m.group(1)).lower()
        anchors.add(f"file:{base}")
    return anchors


def _downstream_ai_anchors(messages: list[BaseMessage], after_idx: int) -> set[str]:
    """after_idx 之后所有 AIMessage 可见正文里的引用锚点并集。"""
    anchors: set[str] = set()
    for msg in messages[after_idx + 1 :]:
        if isinstance(msg, AIMessage):
            anchors |= extract_reference_anchors(_visible_text(msg))
    return anchors


def _tool_content(msg: BaseMessage) -> str:
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content or "")


def cited_tool_indices(messages: list[BaseMessage]) -> set[int]:
    """
    哪些历史 ToolMessage 的产出被后续 AI 回复引用过。

    命中判定：ToolMessage 内容锚点 ∩ 其后任一 AI 可见正文锚点 ≠ ∅。
    优先按 path:line 强信号；只共享文件名（file:）也算（弱信号，宁多勿漏）。

    @param messages 消息列表
    @return 被引用的 ToolMessage 下标集合
    """
    tool_idxs = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if not tool_idxs:
        return set()
    # 预先按位置切分：对每个 tool，取其后的 AI 锚点。用后缀并集降复杂度。
    suffix_anchors: dict[int, set[str]] = {}
    acc: set[str] = set()
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            acc = acc | extract_reference_anchors(_visible_text(messages[i]))
        suffix_anchors[i] = acc
    cited: set[int] = set()
    for idx in tool_idxs:
        downstream = suffix_anchors.get(idx + 1, set())
        if not downstream:
            continue
        tool_anchors = extract_reference_anchors(_tool_content(messages[idx]))
        if tool_anchors & downstream:
            cited.add(idx)
    return cited


def cited_line_pairs_for_tool(
    messages: list[BaseMessage],
    idx: int,
) -> list[tuple[str, int]]:
    """
    该 ToolMessage 中、被下游 AI 以 path:line 精确引用到的 (basename, line)。

    @param messages 消息列表
    @param idx ToolMessage 下标
    @return (文件名, 行号) 列表（去重、升序）
    """
    if not (0 <= idx < len(messages)) or not isinstance(messages[idx], ToolMessage):
        return []
    downstream = _downstream_ai_anchors(messages, idx)
    if not downstream:
        return []
    tool_anchors = extract_reference_anchors(_tool_content(messages[idx]))
    tool_files = {a[len("file:") :] for a in tool_anchors if a.startswith("file:")}
    if not tool_files:
        return []
    pairs: set[tuple[str, int]] = set()
    for anchor in downstream:
        if not anchor.startswith("pathline:"):
            continue
        _, base, line = anchor.split(":", 2)
        if base not in tool_files:
            continue
        try:
            pairs.add((base, int(line)))
        except ValueError:
            continue
    return sorted(pairs, key=lambda p: (p[0], p[1]))


def build_cited_line_preview(
    content: str,
    pairs: list[tuple[str, int]],
    *,
    radius: int = 6,
    max_lines: int = 40,
) -> str:
    """
    从工具结果里抽出被引用行 ± radius 的预览（read 行标 / grep 命中通用）。

    @param content 工具原始输出
    @param pairs 被引用的 (文件名, 行号)
    @param radius 上下文半径
    @param max_lines 预览最多行数
    @return 预览文本；无命中时空串
    """
    if not pairs or not content:
        return ""
    lines = content.splitlines()
    want: set[int] = set()
    # read：`  167| ...` 行标即真实行号
    marked: dict[int, int] = {}
    for i, ln in enumerate(lines):
        m = _READ_LINE_MARK_RE.match(ln)
        if m:
            marked[int(m.group(1))] = i
    target_lines = {line for _base, line in pairs}
    for line in target_lines:
        anchor_i = marked.get(line)
        if anchor_i is not None:
            for j in range(max(0, anchor_i - radius), min(len(lines), anchor_i + radius + 1)):
                want.add(j)
    # grep：`--- path:167 ---` 命中头，取其后若干行
    for m in _GREP_HIT_HDR_RE.finditer(content):
        try:
            hit_line = int(m.group(2))
        except ValueError:
            continue
        if hit_line not in target_lines:
            continue
        hdr_i = content[: m.start()].count("\n")
        for j in range(hdr_i, min(len(lines), hdr_i + radius * 2 + 1)):
            want.add(j)
    if not want:
        return ""
    ordered = sorted(want)[:max_lines]
    out: list[str] = []
    prev = None
    for i in ordered:
        if prev is not None and i > prev + 1:
            out.append("…")
        raw = lines[i]
        out.append(raw[:200] + "…" if len(raw) > 200 else raw)
        prev = i
    return "\n".join(out).strip()
