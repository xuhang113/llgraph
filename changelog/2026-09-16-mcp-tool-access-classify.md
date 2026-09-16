# 2026-09-16 MCP 只读模式：写工具别再漏放，读工具别再被悄悄摘掉

选题：**稳定性（只读模式越权）**，兼顾商用体验（判错可见、可改）。
`2026-09-15` 结尾点名的第一件事：

> 「`llgraph/permissions/mcp.py` 的写类 MCP 工具判定还是关键词命中
> （`name + description` 里出现 write/edit/delete 就算写），误判面大且零测试。」

老实现是 26 行：把 `name + description` 拼一起，出现 8 个关键词之一就算写。
两头都不准，而只读是 llgraph 的默认模式。

## 先量：真 Server 的 tools/list

本机装了 7 个真实 MCP Server（`mcp-server-git`、`mcp-server-time`、`mcp-server-fetch`、
`@modelcontextprotocol/server-filesystem` / `-memory` / `-everything` /
`-sequential-thinking`），抓下 `tools/list`，拿服务端自己声明的 `readOnlyHint` 当标签：

| 51 个有标注的真实工具 | 老实现 | 本轮 |
|---|---|---|
| 漏放（写工具在只读会话里可调用） | **9** | 0（标注在）/ 1（纯启发式） |
| 误伤（只读工具被隐藏） | 1 | 0 |

漏放的那 9 个里有 `git_commit`、`git_add`、`git_reset`、`git_checkout`、`move_file`。
`git_reset` / `git_checkout` 会直接冲掉用户未提交的改动——只读会话本来就是
「我先让它看看，别动我代码」的场景，这是本轮真正要修的东西。
原因很直白：这些工具的描述是 `Records changes to the repository`、
`Adds file contents to the staging area`、`Unstages all staged changes`、
`Switches branches`，一个关键词都不含。

误伤那 1 个是 `trigger-long-running-operation`，描述里的 `progress updates`
命中了 `update`。同一条链上更要命的是本仓自己在用的 MySQL / PolarDB Server：
它们的主工具 `mysql_query` 一旦描述里写了「支持 SELECT/INSERT/UPDATE/DELETE」，
只读模式下整台库就哑了，而用户唯一的补救是把这台 Server 的写工具**全部**打开。

端到端（真 Server 走生产加载链路）：只读模式下 `mcp-server-git` 原本暴露 11/12 个工具，
现在暴露 7 个——正好是 7 个 `readOnlyHint: true` 的那几个。

## 做了什么

### 1. 判定分层，按可靠性排序（`permissions/mcp.py` 重写）

`classify_mcp_tool(name, description, annotations=, read_tools=, write_tools=)`
返回 `McpToolAccess(is_write, source, detail)`，优先级：

1. **用户覆盖**：该 Server 的 `write_tools` / `read_tools`（glob，大小写不敏感）。
   两张表都命中时按写处理。
2. **服务端标注**：`readOnlyHint`（camel / snake 两套命名，dict 与 pydantic 模型都认）；
   只给了 `destructiveHint: true` 也算写。字段不是布尔（服务端乱填）就当没给。
3. **工具名动词位**：名字切词（`_` `-` `.` 与驼峰都算边界），前两个词是动词位
   （MCP 命名约定是动词打头，但常带一层 Server 前缀：`git_commit`、`slack_post_message`）。
   动词位里**先出现**的那个词说话：`get_post` / `get_commit` 的第二个词是名词，
   `git_commit` / `merge_pull_request` 的才是动作。动词位没结论再扫剩下的词。
4. **描述的起始动词**（只看第一句、只看前几个词、跳过 `Recursively` 这类副词，
   带轻量词干还原）：这才是 `git_commit` 那批不带标注的工具被认出来的原因。
5. 都判不出来 → **默认按读**。

刻意**不再**整段扫描描述：`progress updates` 里的 `update`、
`get_pull_request` 里的 `pull` 都会误命中，那是老实现误伤的唯一来源。
词干还原也只用在描述上：工具名里的 `unstaged`（`git_diff_unstaged`）
一旦被还原成 `unstage`，一个只读 diff 工具就会被判成写。

写动词表分两档：`create` / `commit` / `reset` / `move` 这类**硬动词**出现即判写；
`run` / `execute` / `manage` / `request` 这类**软动词**只在动词位、且同位置
没有读动词时才判写（`run_query` 可能真的只是查）。

### 2. 判错有出路：per-Server 覆盖（`config/mcp_config.py`）

```json
{ "servers": { "git": { "read_tools": ["git_stash_list"], "write_tools": ["run_*"] } } }
```

没有这两个口子时，一次误伤的代价是「把整台 Server 的写工具全打开」。

### 3. 隐藏必须可见（`core/mcp_tools.py`）

以前被过滤的工具是静默 `continue`：模型看不见它，用户也不知道少了什么，
只能猜为什么模型说这个能力不存在。现在启动摘要多一行：

```
[只读] git 已隐藏写类工具: git_commit、git_add、git_reset、git_create_branch、git_checkout（需要就用 -w，或在 mcp.json 该 Server 下写 read_tools）
```

每条隐藏同时按 `logger.info` 打出判据（`annotation:readOnlyHint=false`、`name:commit`），
判错时能一眼看出是哪一层命中了什么。

### 4. 过滤与重放对齐

`_McpServerRuntime` 连接期把标注和描述一起缓存下来（重连后照样能判），
过滤工具表和「重连后能不能自动重放」（`mcp_health.replay_allowed`）走同一个
`runtime.tool_access()`。两处不一致的话，只读模式下被隐藏的写工具反而会被当成读工具重放。

## 改了哪些路径

- `llgraph/permissions/mcp.py`（重写；删掉 `is_write_mcp_tool`，改用 `classify_mcp_tool`）
- `llgraph/permissions/__init__.py`
- `llgraph/config/mcp_config.py`（`read_tools` / `write_tools` 解析）
- `llgraph/core/mcp_compat.py`（`tool_annotations`）
- `llgraph/core/mcp_tools.py`（判定接线、标注缓存、隐藏清单进摘要）
- `tests/test_mcp_tool_access.py`（新，151 例）
- `docs/操作手册.md`、`docs/模块说明.md`

## 怎么验收

- `python3 -m pytest tests -q` → **916 passed, 3 skipped**（本轮前 765 + 新增 151，
  本机未装 `[index]` 可选依赖，3 skip 与基线一致）
- `python3 -m ruff check llgraph tests` → All checks passed；`compileall` 通过
- `pip install -e .` 后 `llgraph --help`、`python3 -m llgraph --help` 正常
- 回归用例断的是**判定结果**而不是文案：
  - 45 条真实工具（名 + 描述首句，期望取 `readOnlyHint`）在「没有标注」时也要判对
  - 同 45 条在「有标注」时以标注为准，且 `source == annotation`
  - 36 条市面常见命名形态：`get_post` / `get_commit` / `list_pull_requests` 判读，
    `merge_pull_request` / `slack_reply_to_thread` / `sub_issue_add` / `execute_sql` 判写
  - 覆盖压过标注、两表都命中按写、glob 与大小写、配置写成字符串 / 数字 / None 不崩
  - 描述只看起始动词：「后面的句子在讲另一个写工具怎么用」不算写
  - 名字不做词干还原（`git_diff_unstaged` 仍判读）
  - 加载期：只读模式隐藏哪些、摘要里点名、`-w` 一个不隐藏、`read_tools` 能要回来、
    `write_tools` 能多隐藏、标注压过名字、过滤与重放判定一致
- 端到端（真 `mcp-server-git` / `server-filesystem` 走 `create_mcp_tools`）：
  只读模式 git 从 11 个工具收到 7 个（漏放的 4 个写工具消失），
  `read_tools: ["git_checkout"]` 能把指定工具要回来，`-w` 下 12 个全在

## 未做 / 下一步不要做

- **不要**把「判不出来」的默认从读改成写。`fetch`、`sequentialthinking`、`mysql_query`
  这类工具全靠默认按读活着；默认按写会把只读会话的外部能力大面积摘掉，
  而用户看到的只是「模型说它没有这个工具」。漏放那一档交给上面三层收敛。
- **不要**继续往关键词表里加词来「提高召回」。`pull`（`get_pull_request`）、
  `trigger`（`trigger-long-running-operation` 是只读的）都是收进来就立刻误伤的词，
  它们留在表外是量出来的结果，不是遗漏。
- **不要**给工具名加词干还原。上面 `git_diff_unstaged` 那条已经固化成用例。
- 已知会判错且**刻意**不救的：`search_replace` 这种「读动词打头 + 第二个词才是真动作」
  的命名会被判成读。救它就要牺牲 `get_post` / `get_commit` 那一批，
  而 `get_*` / `list_*` 的只读工具在市面 Server 里多一个量级。真撞上了用 `write_tools` 覆盖。
- 没有做「按 `inputSchema` 判定」（例如出现 `content` / `body` 字段就算写）。
  看起来很准，其实 `query` / `sql` 也是入参，而且 schema 比名字长得多、误伤面更大。
  真要做请先像本轮一样先量。
- 没有碰 `allow_write_tools` 的语义，也没有做「调用时二次确认」。
  MCP 的写工具目前是「加载期过滤」一刀切，不像 shell 那样有 `-w` 之外的闸门；
  真要做交互式授权，那是 `permissions/` 与 CLI / Web 一起动的独立一轮。
- 下一轮建议（二选一）：
  - **性能**：`2026-09-14` 结尾点名过的 `format_duplicate_block` 「上次返回摘录」
    上限 900 字符，检索被短路后模型看到的占位里大半是重复摘录。仍未做。
  - **商用体验 / 改码命中率**：`search_replace` 前是否该强制带目标段真实行号，
    从 `2026-09-09` 起连续点名五轮仍未做。动手前先量「`edit_apply` 六层容错之后
    剩下的失败里有多少是行号能救的」。
