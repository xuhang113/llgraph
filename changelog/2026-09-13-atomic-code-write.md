# 2026-09-13 改码写入：不毁文件、不篡改正文

选题：**稳定性**。接上一轮「会话状态原子落盘」的方向往下做一层——
会话状态早就安全了，用户自己的源码写入却还是最脆的一环。
下手前先在写路径上实测，抓到两个都会直接坑到真实改码的问题。

## 做了什么

### 1. 写入原子化（数据丢失）

`write_file` / `append_file` / `search_replace` / `/undo` 还原全都是 `Path.write_text()`：
先把目标文件截断成 0 字节，再往里写。用户 Ctrl-C 掐掉跑飞的 Agent、SIGTERM、OOM
落在这中间，磁盘上剩的就是空文件或半截源码，而这份改动往往还没进 git。

新增 `llgraph/core/atomic_write.py`：同目录临时文件 → `os.replace`。
和会话落盘不同，这里换的是用户自己的文件，额外保住文件语义：

- 目标是符号链接时写穿到真实文件，不把链接换成普通文件
- 保留权限位（`collect.sh` 0755 不会被写成 0644 而跑不起来）
- 硬链接（`st_nlink > 1`）退回就地覆盖，`os.replace` 会切断链接关系
- 目录只读之类建不出临时文件时退回就地覆盖，写工具不因此失去能力
- 磁盘写满（ENOSPC）直接报错，不退回就地覆盖——那只会先截断再失败
- 不 fsync：这里防的是进程被打断，不为每次编辑付一次磁盘同步

### 2. 入参纠偏不再改写正文（工具直接不可用）

纠偏层对每个字符串参数都跑一遍 `maybe_parse_json`，副作用有两个：

- 正文长得像 JSON 就被解析成 dict，pydantic 当场拒。**`content` 是 `{...}` 的
  `write_file` 一次都成功不了**：`package.json`、`tsconfig.json`、`mcp.json` 全中招，
  模型看不出哪错了只会原样重试。
- 不像 JSON 的字符串被 `strip()`。写出的文件一律丢末尾换行（每个 diff 都带
  `\ No newline at end of file`），`old_string` / `new_string` 的首尾空行也被吃掉，
  等于模型没法用空行表达位置。

正文字段（`content`、`old_string`、`new_string` 及其商用别名）改为原样透传；
`replacements` / `todos` 的 JSON 解析与路径类参数的行为保持不变。

## 改了哪些路径

- `llgraph/core/atomic_write.py`（新增）
- `llgraph/core/filesystem_tools.py`
- `llgraph/core/tool_arg_coerce.py`
- `llgraph/session/session_edits.py`
- `tests/test_atomic_workspace_write.py`（新增）
- `tests/test_tool_arg_coerce.py`
- `docs/项目结构.md`

## 怎么验收

- `python3 -m pytest tests -q` 全绿（662 passed / 4 skipped），`ruff check` 干净
- 打断保原文：`tests/test_atomic_workspace_write.py` 里把 `os.replace` 换成抛
  `KeyboardInterrupt`，`search_replace` 之后源文件仍是原内容且目录没有 tmp 残留；
  改动前同一场景会留下空文件
- 文件语义：0755 脚本编辑后仍 0755；符号链接编辑后仍是链接、真实文件被更新；
  硬链接编辑后 `st_nlink` 仍为 2 且两个路径都看到新内容
- 正文保真：`write_file(path="package.json", content='{...}')` 返回「已写入」且
  落盘内容逐字节相同；`content="x = 1\n"` 落盘保留末尾换行
- 入口没坏：`llgraph --help`、`python -m llgraph --help` 正常

## 未做 / 下一步不要做

- 没有加 fsync 开关、没有做备份/回收站；`/undo` 已有会话快照，不要再叠一层
- 没有碰 `read` / `grep` / 索引路径，这轮只动写入
- 不要把 `atomic_write` 和 `session/atomic_store` 合并：一个要保用户文件的权限与
  链接语义并允许就地回退，另一个只管 `~/.llgraph` 下的状态文件，合了就得互相妥协
- 下一轮建议：写入之外，`maybe_parse_json` 对路径类参数的 strip 仍是隐式行为，
  可以顺着「工具入参保真」把纠偏层的副作用收敛成显式规则；或转去查
  工具循环里的无效调用（同一 grep 反复打）
