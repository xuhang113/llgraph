# 2026-09-25 命令在编辑器里滚起来：ACP 终端（`terminal/*`）

选题：**排队第 2 件（编辑器里干活）**。`2026-09-24` diff 那篇末尾点名的就是这件：

> 「下一件按 `AGENTS.md` 排队走：**ACP 终端**（`terminal/create` + `terminal/output`）——
> `run_shell_command` 现在要等命令跑完才一次性回填输出，编辑器里看不到滚动中的日志；
> 难点是客户端不一定声明 `terminal` 能力（没有时必须回落现有写法），
> 且输出要边跑边推，得先决定从工具的哪一层拿到流式 stdout。」

在此之前，编辑器里一条 `pytest -q` 的几分钟内那一行只写着「进行中」，跑完才「唰」地出现
一整段截断过的文本。`2026-09-22` 的三段状态解决了「是不是卡死了」，但「它现在跑到哪了」
仍然只能切到终端去看——而这是改代码时最常盯的一块屏幕。

## 做了什么

### 1. 流式 stdout 从哪拿：根本不用拿

上一篇留的难点（「从工具的哪一层拿到流式 stdout」）问错了方向。ACP 的终端**不是**
「我们把输出推给编辑器」，而是 `terminal/create` 让**编辑器自己起进程**、自己渲染输出；
我们只在 `tool_call` 的 `content` 里挂一个 `{"type": "terminal", "terminalId": ...}` 块，
那一行就变成一个活的终端。所以这轮没有任何「边跑边推输出」的代码：
推一条 id 过去，实时性归编辑器。

代价是命令换了执行方：

- **沙箱开着时不交出去。** 编辑器那条路没有 seatbelt / bwrap 包装，交过去等于悄悄
  把隔离关了。`sandbox.enabled` 为真时一律本地起进程（沙箱默认关，所以这条路默认生效）。
- **能不能跑仍在我们这边判。** `permissions.shell` 的拦截与可写模式下的授权弹窗都排在
  `terminal/create` **之前**，一行没动：编辑器只负责跑和显示，不负责「这条命令该不该跑」。
- **环境变量不传。** `terminal/create` 的 `env` 留空，命令就跑在编辑器自己的环境里——
  那正是用户按 ⌘` 打开终端会得到的环境，比把 Agent 进程的 env 拓过去更接近预期。

### 2. 不动结果格式化、后台 job 与 `await_shell`：换的是进程，不是流程

`EditorTerminalProcess` 长得和 `sandbox/exec.py` 的 `LiveShellProcess` 一样
（`snapshot_stdio` / `returncode` / `elapsed_sec` / `wait` / `kill` / `error` / `sandboxed`），
`run_shell_command` 里只有一处 `spawn` 换成了「先问编辑器、不行再本地起」。
于是头尾截断、`cd` 记忆、相同命令去重、`max_jobs` 上限、`await_shell`、硬超时回收**一行都没改**——
后台跑一条 `npm run dev` 再 `await_shell` 看一眼，与本地进程走的是同一段代码。

两个取舍在这层：

- **退出状态走挂着的 `terminal/wait_for_exit`**（后台线程等，不设超时）。命令可能跑一小时，
  设超时只会把它误判成失联；取消与硬超时改走 `terminal/kill`，杀掉命令，编辑器自然回退出状态。
- **输出按需取，不轮询。** 编辑器已经在渲染实时输出，我们再每秒 `terminal/output` 拉一遍
  只是把同一段文本反复搬过 stdio。只在真要文本时取：跑完那一次，或 `await_shell` 看一眼那一次。
  跑完取到的那份留下来（`terminal/release` 之后不能再问），所以同一条 job 反复 `await_shell`
  也不会多问编辑器一次。

`outputByteLimit` 给得比模型可见的 `max_output_chars` 宽（4 倍，128KB~1MB）：
这个上限同时管着**编辑器里能往上翻多少**，人翻日志比模型读得多；但也不能无上限，跑完要整段过 stdio。

### 3. 发在哪：终端块当场发，收尾那条重发它、不再附文本

与 `2026-09-24` 的 diff 相反——diff 攒到收尾发，终端**必须马上发**：
实时输出就靠编辑器拿着这个 id 自己渲染，攒到收尾再发就只剩一份跑完的文本，等于这轮没做。
所以 `terminal/create` 一拿到 id 就经 `core/tool_progress.py` 的观察者报给 Sink
（与起步通知、编辑通知同一处 ContextVar，归属同样按当前 `tool_call_id` 认），
Sink 当场发一条只带 `content` 的 `tool_call_update`。

收尾那条**重发同一个终端块**（ACP 的 `content` 是整块替换的，不重发就被文本顶掉），
并且**不再回填工具的文本输出**：终端里已经是同一段输出的全文，还带着退出状态，
再在下面附一份截断到 40 行的副本，只是同样的东西看两遍。
这和 `2026-09-24` 那篇「不要因为有了 diff 就删掉文本输出」不冲突：diff 旁边的文本是
**另一种信息**（语法诊断、分块提示），终端旁边的文本是**同一段输出的副本**。
模型那边照旧拿到完整的文本结果——这轮改的只是编辑器里画什么。

### 4. 四个「不交出去」

- **沙箱开着**（见上）。
- **客户端没声明 `clientCapabilities.terminal`。** 拿不到来源，`spawn_editor_terminal`
  返回 None，命令本地起——CLI / Web Console 一行行为都没变。
- **编辑器开不出来 / 连着失败 3 次。** 与 `fs_bridge` 同一个熔断：每次失败都要等一个超时，
  一轮里几条命令就能把对话拖死。熔断后本会话的命令只在本地跑。
- **纯 `cd` 那条与被拦下的命令。** 前者不起进程，后者在闸门就返回了。

## 改了哪些路径

- `llgraph/core/shell_terminal.py`（新：来源协议与 ContextVar、`EditorTerminalProcess`、`spawn_editor_terminal`）
- `llgraph/core/tool_progress.py`（终端观察者与 `notify_terminal_created`）
- `llgraph/core/shell_tools.py`（`_spawn`：先问编辑器，再回落 `spawn_sandboxed_shell`）
- `llgraph/core/shell_jobs.py`（`ShellProcess` 两种进程；`running_count` 去掉对 `.proc` 的直取）
- `llgraph/editor/acp/terminal_bridge.py`（新：`terminal/*` 五条反向请求、回包解析、熔断）
- `llgraph/editor/acp/updates.py`（`terminal_content`、`tool_call_terminal`、`tool_call_from_step(terminals=)`）
- `llgraph/editor/acp/sink.py`（`tool_terminal` 当场发、收尾重发）、`turn.py`（登记来源与观察者）、`server.py`（能力协商 + 按会话建桥）
- `tests/test_acp_terminal.py`（新，44 例）、`tests/test_acp_end_to_end.py`（+2 例）
- `README.md`、`docs/模块说明.md`、`docs/项目结构.md`、`AGENTS.md`、`.cursor/rules/cloud-agent.mdc`

## 怎么验收

- `python3 -m pytest tests -q`（除 `test_prompt_cache_breakpoints.py`）→ **1124 passed, 10 skipped**
  （本轮前 1078；新增 46 例）；`ruff check llgraph tests` 通过。
  `tests/test_prompt_cache_breakpoints.py` 单跑仍是 **11 failed**，与本轮无关且与上两轮同因：
  本机装的 `langchain-anthropic` 里 `_format_messages()` 多了必填关键字参数 `model`。
  skip 全是本机没装的可选依赖（`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` /
  `langchain-google-genai`）
- 来源语义：没登记来源时 `spawn_editor_terminal` 返回 None、作用域随 with 块结束、
  编辑器开不出来 / 抛异常一律回落；建好报一次（带当前 `tool_call_id`），
  没有当前调用 id / 空终端 id 不报、观察者抛异常不外溢
- 同形进程：跑完报退出码与输出、最终输出只取一次且随后 `terminal/release`、
  还在跑的不放也不缓存、`truncated` 加一句注记、只有信号没有码时报 -1、
  取消与硬超时都走 `terminal/kill`（`error` 分别记 cancelled / timeout）、`kill` 幂等、
  编辑器失联时这条 job 仍收得掉（退出码 -1 + 输出里说明失联）
- `run_shell_command`：有编辑器终端时**不许**本地起进程（本地那条路被换成会抛的桩件），
  `command` / `args` 是 `/bin/sh -c <命令串>`、`cwd` 是绝对路径、`outputByteLimit` 按配置算；
  结果头尾与 `[exit N]` 一行没变；`working_directory` 补成绝对路径；
  后台起 + `await_shell` 走通；沙箱开着时一条 `terminal/create` 都不发；
  被拦下的命令与纯 `cd` 不开终端
- 桥：`terminal/create` / `output` / `wait_for_exit` / `kill` / `release` 载荷按协议，
  退出状态顶层与 `exitStatus` 两种形状都认（认错一种就会把跑完的命令当成还在跑），
  脏回包不炸、`wait_for_exit` 不设超时、连着失败 3 次熔断、取消时不建终端、
  **取消之后仍取得到输出**（模型要知道停之前跑出了什么）
- 载荷与 Sink：终端块在命令跑完**之前**就发出去（只带 content，不带 status），
  收尾那条重发它且不再附文本副本；失败态也保留终端块；
  没报过 pending 的调用不挂终端；重复通知只发一次；并行两条命令各归各位
- 端到端（stub server 说 Anthropic 协议，**真跑 ReAct**）：模型调 `run_shell_command` 那轮，
  `terminal/create` 收到 `["-c", "echo hi"]` 与工作区绝对路径，更新序列是
  `pending` → `in_progress` → **终端块** → `completed`（content 仍是终端块），
  跑完 `terminal/release`，模型拿到的工具结果里是终端的输出；
  不给 `editor_terminal` 那轮命令本地跑、收尾照旧是文本块、一个 terminal 块都没有
- 协议层（真 pipe）：声明 `terminal` 能力的客户端收到带 `sessionId` 的 `terminal/create`；
  没声明的一条 `terminal/*` 都收不到
- 真子进程（像编辑器那样 `python -m llgraph acp -C <ws>`，stdin/stdout 交互）：
  带 `clientCapabilities.terminal` 的 `initialize` 与 `session/new` 正常回包
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常

## 未做 / 下一步不要做

- **不要**改成「我们自己起进程，再把输出推给编辑器」。ACP 没有这个方向的方法；
  终端块的实时性来自编辑器自己持有那个进程，自己起就只能退回本轮之前的一次性回填。
- **不要**为了省一次往返去轮询 `terminal/output` 当进度条。编辑器已经在渲染实时输出，
  轮询只是把同一段文本反复搬过 stdio；真要在 llgraph 侧看实时输出，那是终端 TUI 的事。
- **不要**沙箱开着也交给编辑器。那条路没有 seatbelt / bwrap 包装，交过去等于把隔离悄悄关了；
  真要做得先让编辑器跑我们拼好的 `sandbox-exec` / `bwrap` argv，并想清楚 profile 文件的生命周期。
- **不要**因为有了终端就把工具的文本输出也从**模型**那边删掉。编辑器里不附副本是界面取舍，
  模型仍然只能从工具结果读到命令跑出了什么。
- **不要**给 `terminal/create` 补 `env`。命令跑在编辑器的环境里才是用户预期（与他按 ⌘` 得到的一样）；
  把 Agent 进程的 env 拓过去还有把凭据写进编辑器日志的风险。
- **不要**给别的工具也挂终端块。只有 `run_shell_command` 有「一条命令的输出流」这个东西，
  别的工具挂过去就是一个空终端。
- 后台 job 跨轮仍只在 llgraph 这边记着：编辑器那个终端在下一轮的 `tool_call` 里不会重新挂出来
  （`toolCallId` 是按轮发的）。要做得先决定「上一轮起的命令」在编辑器里画在哪一行。
- `await_shell` 看一眼时取回的那份输出只回给模型，没有重发终端块：那一行的终端本来就还活着，
  重发只是多一次刷新。
- 下一件按 `AGENTS.md` 排队走：**ACP 计划**（`session/update` 的 `plan`）——`todo_write` 的待办
  现在只有 CLI / Web Console 看得到，编辑器里一条都不显示；难点是 `plan` 是全量快照，
  得先决定每次 `todo_write` 之后整份发、还是只在状态真的变了时发，以及 `todo_store` 的状态
  怎么映射成 ACP 的 `pending` / `in_progress` / `completed`。
