# 2026-09-19 编辑器里能真改代码：写入前弹 ACP 授权框

选题：**排队第 2 件（编辑器里干活）**。`2026-09-18` ACP 骨架那篇末尾写的下一件就是这个：

> 「没做 `session/request_permission`：写权限现在靠 `llgraph acp --write` 一刀切开，
> 编辑器里不会弹授权框。**下一轮该攻这个**——ACP Agent 不能改代码，「在编辑器里干活」就只做了一半。」

那篇还留了道岔：「要么在工具层加一个可注入的确认钩子，要么在 ACP 侧包一层工具代理」。
选了**前者**——工具代理要在 `session_bootstrap` → `build_agent` → `get_agent_tools` 这条链上
截住已经建好的工具再包一层，而 Agent 会话是按 thread 缓存复用的，代理包一次就跟着缓存留下来了；
钩子是每轮 invoke 外面登记、跑完就摘，谁登记谁负责。

## 做了什么

### 1. `permissions/approval.py`：策略之外多一层「交互」

`file_write.py` / `shell.py` 回答的是「这个模式允不允许这类操作」，是**策略**；
这里回答「策略放行之后，还要不要问一下坐在前面的人」，是**交互**。两件事分开，因为前者
在 CLI / Web / ACP 三个入口下是同一套，后者只有编辑器有地方弹框。

闸门放在 **ContextVar** 上，不是全局变量也不是 `WorkspaceContext` 字段：

- 全局变量：同进程多会话时会串台（ACP 一个连接可以开多个 session）
- `WorkspaceContext` 字段：ctx 跟着 Agent 会话缓存活着，闸门却是每轮换一个
- ContextVar：langchain 的执行器在提交任务时复制调用方 context，所以工具即便被扔进
  LangGraph 的线程池跑，也读得到本轮闸门（`runtime_context.py` 的 thread_id 已经在吃这条保证）

**没登记闸门时一律放行**——CLI 与 Web Console 一行代码没动，行为完全不变。

闸门自己抛异常、或回了个读不懂的东西，都按**拒绝**算：授权链路出问题时宁可这一刀不落地。

### 2. 四个真正会写的地方去问闸门

`write_file` / `append_file` / `search_replace` 在 `write_workspace_text` 之前问，
`run_shell_command` 在 spawn 之前问。两个细节：

- **问的时机在「算完改动之后」**：`search_replace` 要先把 hunk 应用到内存里，才有
  改前 / 改后全文交给编辑器渲染 diff。放在入参校验旁边省事，但弹窗里就只有一个文件名。
- **被拒不计入 `WriteFailureTracker`**：那是「写法不对」的连续失败计数器，到阈值会给模型
  发分块重试提示。用户按了拒绝而收到「把内容分块再试」，是最糟的组合。拒绝语也特意避开了
  那个计数器扫 ToolMessage 时认的失败特征词（`错误:`、`文件不存在:` …），有回归测试钉住。
- 只读模式的 shell **不问**：能过只读策略闸门的命令本身不改工作区，再弹框只是噪音。

### 3. `session/request_permission`：本实现里第一条反向请求

`jsonrpc.py` 原来只会应答，回包一律忽略（注释写着「本轮不主动发请求」）。现在加了 `request()`：
自增 id、pending 表、写出去后阻塞等读循环唤醒。三处是「不这么写就会挂」的：

- **等待必须在工作线程**，读循环不能被占住——否则等授权期间收不到 `session/cancel`，
  用户既点不了按钮也停不下来（和上一轮 `session/prompt` 不能同步执行是同一个道理）
- **等待可取消**：`cancel_check` 轮询而不是死等，`session/cancel` 一来就放弃这次授权，
  返回 cancelled，工具拿到「已停止本轮」而不是一直挂着
- **连接断了要把 pending 一起放掉**（`_fail_all_pending`）：编辑器被关掉时，
  正等授权的那个工作线程会永远挂在 Event 上，进程也就退不干净

弹窗载荷用 ACP 的 ToolCall：编辑改动带 `diff` 块（编辑器唯一会渲染成改动预览的块型），
shell 带命令原文，`locations` 给绝对路径。四个选项按 ACP 的 kind 给全：
允许这一次 / 本会话都允许 / 拒绝 / 本会话都拒绝。**「都允许」记在会话上**，
所以闸门按会话建而不是按轮建——用户点过一次就不该下一轮再被问。

### 4. `llgraph acp` 的三档

| 命令 | 写工具 | 弹框 |
|------|--------|------|
| `llgraph acp` | 有 | **每次写 / 执行都弹**（新默认） |
| `llgraph acp --write` | 有 | 不弹（等于上一轮的 `--write`） |
| `llgraph acp --read-only` | 无 | 不弹 |

默认从「只读」改成「可写但逐次确认」是刻意的：编辑器里的 Agent 改不了代码就只做了一半，
而 ACP 客户端本来就都实现了授权弹窗，把决定权交回给人比一刀切成只读有用。
老配置 `"args": ["acp", "--write"]` 行为一字不变。

## 改了哪些路径

- `llgraph/permissions/approval.py`（新）、`llgraph/permissions/__init__.py`
- `llgraph/editor/acp/permission.py`（新）
- `llgraph/editor/acp/jsonrpc.py`（反向请求、回包匹配、断线放行）
- `llgraph/editor/acp/server.py`（`ask_permission`、会话级闸门）、`llgraph/editor/acp/turn.py`（登记闸门）
- `llgraph/core/filesystem_tools.py`（三个写工具）、`llgraph/core/shell_tools.py`（执行前）
- `llgraph/cli/acp_cli.py`（`--write` 语义 + `--read-only`）
- `tests/test_acp_permission.py`（新，25 例）、`tests/test_acp_end_to_end.py`（stub 的工具可换 + 2 例写入验收）
- `README.md`、`docs/模块说明.md`、`docs/项目结构.md`、`AGENTS.md`、`.cursor/rules/cloud-agent.mdc`

## 怎么验收

- `python3 -m pytest tests -q` → **983 passed, 10 skipped**（本轮前 956；新增 27 例。
  skip 全是本机没装的可选依赖：`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` / `langchain-google-genai`）；
  `ruff check llgraph tests` 通过
- 闸门语义（`tests/test_acp_permission.py`）：没闸门一律放行、`with` 块外自动摘掉、
  闸门抛异常 / 回包读不懂都按拒绝、拒绝语不污染写失败计数器
- 工具层：拒绝 `write_file` 后文件**不存在**、拒绝 `search_replace` / `append_file` 后
  内容**逐字节不变**、允许后照常落地；闸门拿到的是 path + 改前/改后全文；
  shell 拒绝后命令**没跑**（拿「写文件的 python -c」当探针，靠文件是否出现判断）；只读 shell 不问
- ACP 侧：弹窗载荷带 diff / 命令原文 / 绝对路径 / 四个选项，
  `allow_always` 之后同类不再问（shell 仍单独问）、`reject_always` 之后一直拒、
  `cancelled` 与等待期间被取消、编辑器回错误或选项认不出 → 拒绝
- 协议层（真 pipe，测试侧扮编辑器）：收到 `session/request_permission` → 回 `allow_once` →
  决定传到工具；回 `reject_once` → 这一刀不落地但**对话继续**（`stopReason: end_turn`）；
  等授权时发 `session/cancel` → 回 `cancelled`；等授权时关掉连接 → 拒绝而不是挂死；
  `--write`（免确认）下一条授权请求都不发
- 端到端（stub server 说 Anthropic 协议，**真跑 ReAct**）：模型要求 `search_replace` →
  允许后 `app.py` 真被改成 `return 2`；拒绝后文件不变，且**回灌给模型的工具结果里有「拒绝」**
  （模型不能以为自己改成了）
- 真子进程（像编辑器那样 `python -m llgraph acp -C <ws>`，stdin/stdout 交互）：
  弹窗标题 `执行 search_replace(app.py)`、kind `edit`、diff 块里两份全文都在、
  回 `allow_once` 后磁盘上的文件变了，回 `reject_once` 后没变；两次都 `stopReason: end_turn`，
  watchdog 缺失的警告仍只在 stderr
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常

## 未做 / 下一步不要做

- **不要**把闸门从 ContextVar 挪回全局变量或 `WorkspaceContext` 字段。
  前者多会话串台，后者跟着 Agent 会话缓存活得比一轮长。
- **不要**把授权点挪到工具入参校验旁边。看着更早拦住，代价是弹窗里没有 diff——
  用户要看的是「这一刀改了什么」，不是「哪个文件要被改」。
- **不要**在读循环里等授权回包。和上一轮 `session/prompt` 一样：等待期间必须还能收 `session/cancel`。
- **不要**让授权失败默认放行（包括超时）。现在超时（10 分钟）算拒绝。
- 没做 `session/load` 续聊：能力仍声明为 `false`，编辑器重启后拿旧 sessionId 接不回去。
  **下一轮就是这件**——要把 `messages.jsonl` 回放成 `session/update`（用户 / 助手 / 工具三类都要映射），
  注意工具那类回放不出耗时与输出，得决定是折成文本还是只发标题。
- 没给 MCP 写类工具接授权：MCP 工具是外部对象，拦不到「落盘前」那一刻，
  只能在调用前按 `permissions/mcp.py` 的判定整体问一次。要做得先想清楚问的粒度。
- 没做「记住某个具体文件 / 某条命令」的粒度：现在「都允许」是按 kind（编辑 / 执行）记的。
  再细就要引入一张会话级规则表，而 ACP 没有把规则回传编辑器的口子，用户看不到自己批过什么。
- 没有 `tool_call` 的 pending → in_progress 中间态（授权弹窗用的是独立的 `perm_*` id）：
  真要串起来，得让 trace 在工具**开始前**也发一次 sink 事件，那是动 `trace_display` 的公共路径。
