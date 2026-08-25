"""Shell 工具结果整形：头尾截断（对齐 Cursor / Claude Code）。

pytest/mvn/npm 失败信息几乎总在输出末尾。只截开头会让模型误判成功，
再空转一轮重跑。超长输出保留开头 + 末尾，并标明省略了多少字符。
"""

from __future__ import annotations

DEFAULT_HEAD_RATIO = 0.4
_OMIT_MARKER = "…(省略 {n} 字符；保留开头与末尾。失败信息通常在末尾。)"


def combine_stdio(stdout: str, stderr: str) -> str:
    """
    合并 stdout / stderr（先 stdout 后 stderr，与历史行为一致）。

    @param stdout 标准输出
    @param stderr 标准错误
    @return 合并文本
    """
    out = stdout or ""
    err = stderr or ""
    if not err:
        return out
    if not out:
        return err
    if not out.endswith("\n"):
        out += "\n"
    return out + err


def clip_shell_output(
    text: str,
    max_chars: int,
    *,
    head_ratio: float = DEFAULT_HEAD_RATIO,
) -> tuple[str, int]:
    """
    超长输出保留开头与末尾。

    @param text 原始合并输出
    @param max_chars 模型可见上限
    @param head_ratio 开头占比（其余给末尾；须为 (0, 1)）
    @return (可能截断后的文本, 省略字符数)
    """
    body = text or ""
    cap = max(64, int(max_chars))
    if len(body) <= cap:
        return body, 0

    ratio = min(0.85, max(0.15, float(head_ratio)))
    omitted = len(body) - cap
    marker = _OMIT_MARKER.format(n=omitted)
    # 标记本身占预算，避免截完又超长
    budget = cap - len(marker) - 2
    if budget < 32:
        return body[: cap - 1] + "…", omitted

    head_len = max(16, int(budget * ratio))
    tail_len = max(16, budget - head_len)
    head = body[:head_len]
    tail = body[-tail_len:]
    if not head.endswith("\n"):
        head += "\n"
    if not tail.startswith("\n"):
        tail = "\n" + tail
    return head + marker + tail, omitted
