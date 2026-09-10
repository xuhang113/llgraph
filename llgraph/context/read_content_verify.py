"""历史 read 结果与磁盘逐行核对：跨轮 read 去重的唯一可信前提。

`read_file` / `read_files` 的输出本身就带够核对的信息：每个文件块有
`--- path (行 s-e / 共 N 行) ---` 头，正文每行是 `行号| 原文`。因此不需要在读的时候
额外记指纹，事后拿输出正文与磁盘比一遍就能确定「这段内容有没有变过」——
它同时覆盖了本进程的写入、别的进程的写入、以及用户在编辑器里的手改。

只做**精确**判定：行数不同、任一行不同、文件不存在、路径越界，一律算「变了」，
调用方应放行真实 read。宁可白读一次，不能让模型拿着过期正文去拼 old_string。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_READ_HDR = re.compile(
    r"^---\s+(.+?)\s+\(行\s+(\d+)-(\d+)\s+/ 共\s+(\d+)\s+行\)",
)
_NUMBERED_LINE = re.compile(r"^(\d+)\| (.*)$")

# 单条 read 结果最多核对的行数 / 单文件最大核对字节：都是防御性上限，
# 超过就当「无法核对」放行，不为省 token 去读一个几 MB 的文件。
MAX_VERIFY_LINES = 8000
MAX_VERIFY_BYTES = 2_000_000


@dataclass(frozen=True)
class ReadBlock:
    """read 输出里的一个文件块。"""

    path: str
    start: int
    end: int
    total: int
    lines: tuple[tuple[int, str], ...]

    def covers(self, start: int, end: int) -> bool:
        """
        本块是否完整覆盖请求行段。

        请求的 end 超出文件末尾时按末尾算——read 工具自己也是这么 clamp 的，
        且调用方核对过「总行数没变」，所以这里的 clamp 是精确的，不是放松。

        @param start 请求起始行（>=1）
        @param end 请求结束行；<=0 表示到文件末尾
        @return 是否完整覆盖
        """
        want_end = self.total if end <= 0 else min(end, self.total)
        return self.start <= max(1, start) and self.end >= want_end


def parse_read_blocks(content: str, *, max_lines: int = MAX_VERIFY_LINES) -> list[ReadBlock]:
    """
    解析 read_file / read_files 输出为文件块。

    大纲行（`  120| def foo`，行首有缩进）不会被当成正文行，因此
    `read_focus` 折叠结果也能安全解析：只拿到真正的编号正文段。

    @param content 工具输出正文
    @param max_lines 最多解析的正文行数；超出返回空（视为无法核对）
    @return 文件块列表；无法解析时为空
    """
    blocks: list[ReadBlock] = []
    cur_path = ""
    cur_start = 0
    cur_end = 0
    cur_total = 0
    cur_lines: list[tuple[int, str]] = []
    parsed = 0

    def flush() -> None:
        if cur_path and cur_lines:
            blocks.append(
                ReadBlock(
                    path=cur_path,
                    start=cur_start,
                    end=cur_end,
                    total=cur_total,
                    lines=tuple(cur_lines),
                )
            )

    for raw in content.splitlines():
        header = _READ_HDR.match(raw)
        if header is not None:
            flush()
            cur_path = header.group(1).strip()
            try:
                cur_start = int(header.group(2))
                cur_end = int(header.group(3))
                cur_total = int(header.group(4))
            except ValueError:
                cur_path = ""
            cur_lines = []
            continue
        if not cur_path:
            continue
        numbered = _NUMBERED_LINE.match(raw)
        if numbered is None:
            continue
        try:
            line_no = int(numbered.group(1))
        except ValueError:
            continue
        cur_lines.append((line_no, numbered.group(2)))
        parsed += 1
        if parsed > max_lines:
            return []
    flush()
    return blocks


def read_blocks_by_path(blocks: list[ReadBlock]) -> dict[str, list[ReadBlock]]:
    """@param blocks 文件块 @return 路径 → 该路径的块"""
    out: dict[str, list[ReadBlock]] = {}
    for block in blocks:
        out.setdefault(block.path, []).append(block)
    return out


def resolve_verify_target(path: str, workspace: Path) -> Path | None:
    """
    把 read 输出里的展示路径还原成可核对的工作区内文件。

    绝对路径（Skills / Rules）与越界路径一律返回 None：那些不在本模块职责内，
    调用方据此放行真实 read。

    @param path 展示路径
    @param workspace 工作区根
    @return 文件路径；不可核对时 None
    """
    rel = (path or "").strip()
    if not rel or rel.startswith("~"):
        return None
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    try:
        target = (workspace / candidate).resolve()
        root = workspace.resolve()
    except (OSError, RuntimeError):
        return None
    if target != root and root not in target.parents:
        return None
    return target


def path_content_unchanged(
    path: str,
    blocks: list[ReadBlock],
    workspace: Path,
    *,
    max_bytes: int = MAX_VERIFY_BYTES,
) -> bool:
    """
    该路径上的历史 read 正文是否与磁盘逐行一致。

    行数必须也一致：只比覆盖到的行会漏掉「文件尾部被追加/删除」，
    那时 `end_line=0`（读到末尾）的旧结果其实已经不完整了。

    @param path 展示路径
    @param blocks 该路径的历史块
    @param workspace 工作区根
    @param max_bytes 单文件核对上限
    @return 是否完全一致
    """
    if not blocks:
        return False
    target = resolve_verify_target(path, workspace)
    if target is None:
        return False
    try:
        if not target.is_file():
            return False
        if target.stat().st_size > max_bytes:
            return False
        current = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False

    total = len(current)
    for block in blocks:
        if block.total != total:
            return False
        if not block.lines:
            return False
        for line_no, text in block.lines:
            if line_no < 1 or line_no > total:
                return False
            if current[line_no - 1] != text:
                return False
    return True
