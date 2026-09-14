"""扩散式检索治理：同一 pattern 只换目录反复 grep 时，改成一次扩根搜索。

模型常见的反模式是「同一个 pattern 按目录逐个搜」：先 `llgraph/core`，
下一轮 `llgraph/context`，再下一轮 `llgraph/session`……每一次都要一个 LLM 往返，
而这些结果拼起来还不如在共同祖先上搜一次完整（本仓实测 5 次 = 10523 字符 / 5 次往返，
扩根 1 次 = 5188 字符 / 1 次往返，且一次就给出全部 14 个命中文件）。

本模块只做决策（纯路径运算，不碰磁盘、不发工具调用）：
攒够 N 个互不覆盖的搜索根之后，把这一次调用的 `path` 改写成它们的最近公共祖先，
并把「已扩到哪」写进结果尾部——`tool_loop_guard` 据此把后续同 pattern 的子目录搜索
交给已有的覆盖判定短路掉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

WIDENED_MARKER = "[llgraph] 检索已扩根"
MERGED_MARKER = "[llgraph] 检索已合并"

_WIDENED_ROOT_RE = re.compile(re.escape(WIDENED_MARKER) + r":\s*path=\"([^\"]*)\"")
_MAX_SOURCES_IN_NOTE = 4


@dataclass(frozen=True)
class GrepScope:
    """一次已完成 grep 的检索范围（只取判定需要的三项）。"""

    pattern: str
    file_glob: str
    path: str


@dataclass(frozen=True)
class PendingGrep:
    """本批待执行的一次 grep。"""

    call_id: str
    pattern: str
    file_glob: str
    path: str


@dataclass(frozen=True)
class WidenDecision:
    """把某次调用的搜索根扩到 root。"""

    root: str
    pattern: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class WidenPlan:
    """本批的扩根决策。"""

    widen: dict[str, WidenDecision]
    merged: dict[str, str]

    def is_empty(self) -> bool:
        """@return 是否无任何决策"""
        return not self.widen and not self.merged


def normalize_root(raw: str) -> str:
    """
    规范化搜索根（仅路径运算）。

    @param raw 原始 path 参数
    @return 规范根；工作区根统一为 "."
    """
    text = str(raw or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = text.rstrip("/")
    return text or "."


def root_covers(prior: str, new: str) -> bool:
    """
    prior 是否已覆盖 new（与 tool_loop_guard 的 grep 覆盖判定同口径）。

    @param prior 已搜过的根
    @param new 本次根
    @return 是否覆盖
    """
    if prior == ".":
        return True
    if new == prior:
        return True
    return new.startswith(prior + "/")


def common_root(paths: list[str]) -> str:
    """
    最近公共祖先目录。

    @param paths 规范化后的搜索根
    @return 公共祖先；无共同前缀时为 "."
    """
    if not paths:
        return "."
    if any(p == "." for p in paths):
        return "."
    split = [p.split("/") for p in paths]
    common: list[str] = []
    for parts in zip(*split):
        first = parts[0]
        if any(part != first for part in parts):
            break
        common.append(first)
    return "/".join(common) if common else "."


def _maximal(roots: list[str]) -> list[str]:
    """去掉被其它根覆盖的根，只留互不覆盖的极大根。"""
    out: list[str] = []
    for root in roots:
        if any(other != root and root_covers(other, root) for other in roots):
            continue
        if root not in out:
            out.append(root)
    return out


def _scope_key(pattern: str, file_glob: str) -> tuple[str, str]:
    return (pattern.strip(), file_glob.strip())


def plan_grep_widenings(
    prior: list[GrepScope],
    pending: list[PendingGrep],
    *,
    widen_after: int,
    workspace: Path | None = None,
) -> WidenPlan:
    """
    计算本批 grep 的扩根 / 合并决策。

    只在「同 pattern + 同 file_glob」已经攒够 `widen_after` 个互不覆盖的搜索根，
    且本次又换到一个新的、谁也不覆盖谁的根时才扩根。已被历史根覆盖的调用不在此处理
    （由 `tool_loop_guard` 的覆盖短路径负责）。

    @param prior 本问内已完成的 grep 范围（按时间顺序）
    @param pending 本批待执行 grep（按调用顺序）
    @param widen_after 触发所需的历史互不覆盖根数；≤0 关闭
    @param workspace 工作区根；给出时校验扩根目标确实是目录
    @return 决策
    """
    widen: dict[str, WidenDecision] = {}
    merged: dict[str, str] = {}
    if widen_after <= 0 or not pending:
        return WidenPlan(widen=widen, merged=merged)

    known: dict[tuple[str, str], list[str]] = {}
    for scope in prior:
        if not scope.pattern.strip():
            continue
        key = _scope_key(scope.pattern, scope.file_glob)
        known.setdefault(key, []).append(normalize_root(scope.path))
    for key, roots in known.items():
        known[key] = _maximal(roots)

    # 本批内由扩根引入的根 → 引入它的 call_id
    batch_roots: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for item in pending:
        if not item.pattern.strip() or not item.call_id:
            continue
        key = _scope_key(item.pattern, item.file_glob)
        new = normalize_root(item.path)
        roots = known.setdefault(key, [])

        owner = next(
            (cid for root, cid in batch_roots.get(key, []) if root_covers(root, new)),
            None,
        )
        if owner is not None:
            merged[item.call_id] = owner
            continue
        if any(root_covers(root, new) for root in roots):
            continue

        disjoint = [root for root in roots if not root_covers(new, root)]
        if len(disjoint) >= widen_after:
            root = common_root([*disjoint, new])
            if root != new and _root_usable(root, workspace):
                widen[item.call_id] = WidenDecision(
                    root=root,
                    pattern=item.pattern,
                    sources=tuple(_maximal([*disjoint, new])),
                )
                known[key] = _maximal([*roots, root])
                batch_roots.setdefault(key, []).append((root, item.call_id))
                continue

        known[key] = _maximal([*roots, new])

    return WidenPlan(widen=widen, merged=merged)


def _root_usable(root: str, workspace: Path | None) -> bool:
    if root == ".":
        return True
    if workspace is None:
        return True
    try:
        return (workspace / root).is_dir()
    except OSError:
        return False


def format_widened_note(decision: WidenDecision) -> str:
    """
    扩根结果尾部追加的说明（同时是后续覆盖判定的依据，勿改 `path="..."` 形态）。

    @param decision 扩根决策
    @return 追加文案
    """
    sources = [s for s in decision.sources if s != decision.root]
    shown = "、".join(sources[:_MAX_SOURCES_IN_NOTE]) or "多个子目录"
    if len(sources) > _MAX_SOURCES_IN_NOTE:
        shown += f" 等 {len(sources)} 处"
    return "\n".join(
        [
            "",
            f'{WIDENED_MARKER}: path="{decision.root}"',
            f"本问同 pattern（{decision.pattern!r}）已按 {shown} 逐目录搜过，"
            f"本次直接搜 {decision.root}，一次覆盖上述全部范围。",
            f"禁止再对 {decision.root} 的子目录用同一 pattern grep_files；"
            "要细节请对上表路径 read_files，或换 pattern / file_glob。",
        ]
    )


def format_merged_note(
    *,
    requested: str,
    root: str,
    pattern: str,
    owner_call_id: str,
) -> str:
    """
    同批内被扩根覆盖的调用的占位正文（以 `[llgraph]` 开头，不进历史索引）。

    @param requested 本次请求的搜索根
    @param root 已扩到的根
    @param pattern 检索模式
    @param owner_call_id 执行扩根搜索的 tool_call_id
    @return 占位正文
    """
    return "\n".join(
        [
            MERGED_MARKER,
            f"本批已用 path=\"{root}\" 搜过同一 pattern（{pattern!r}），"
            f"覆盖你请求的 {requested}（见 tool_call_id={owner_call_id or '(无 id)'}）。",
            "同一 pattern 勿再按子目录逐个 grep_files；请基于那条结果继续。",
        ]
    )


def widened_root_from_result(content: str) -> str:
    """
    从 grep 结果尾部解析出实际搜过的根。

    @param content 工具返回正文
    @return 扩根后的根；没有扩根时为空串
    """
    text = str(content or "")
    if WIDENED_MARKER not in text:
        return ""
    match = _WIDENED_ROOT_RE.search(text)
    if not match:
        return ""
    return normalize_root(match.group(1))


def apply_widened_note(content: str, decision: WidenDecision) -> str:
    """
    把扩根说明追加到 grep 结果末尾（刻意追加而非前置：前置会被
    `is_llgraph_placeholder` 判成占位，整条结果就不进历史索引，覆盖短路也就没了）。

    @param content 原结果
    @param decision 扩根决策
    @return 追加后的正文
    """
    body = str(content or "")
    if WIDENED_MARKER in body:
        return body
    return body.rstrip("\n") + "\n" + format_widened_note(decision)
