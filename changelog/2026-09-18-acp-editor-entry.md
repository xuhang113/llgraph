# 2026-09-18 编辑器里干活：llgraph 变成 ACP Agent，Zed / Neovim 能直接拉起来

选题：**排队第 2 件（编辑器里干活）**。第 1 件（模型入口）2026-09-17 已完成，那篇末尾写的下一件就是这个。

两条路里选了 **ACP（Agent Client Protocol）**，没做 VS Code 扩展：

- ACP 是 stdio 上的 JSON-RPC，纯 Python 就能实现，**不引入 npm / TypeScript 构建链**，也不用为了验收去装一个编辑器
- 一份实现同时喂 Zed / Neovim / Emacs 这些 ACP 客户端；VS Code 扩展只服务一家，而且真正的难点（会话、工具过程、取消）是同一套，先把这套跑通更划算
- 能在 CI 里端到端验收：协议两端都是进程内可控的流

## 做了什么

### 1. `llgraph/editor/acp/`：协议这一层

| 模块 | 干什么 |
|------|--------|
| `jsonrpc.py` | ndjson JSON-RPC 2.0：读循环、错误码、并发写锁 |
| `server.py` | `initialize` / `authenticate` / `session/new` / `session/prompt` / `session/cancel` |
| `turn.py` | 真跑一轮 ReAct |
| `sink.py` | TraceSink 实现：把过程转成 `session/update` |
| `updates.py` | ACP 载荷与映射（含工具名 → ToolKind） |

`llgraph acp` 是子命令，编辑器以子进程方式拉起（`main.py` 顶部按 `sys.argv[1]` 分流，和 `web` / `index` 一样，不付 langchain import 的钱）。

### 2. 三个「不这么写就会炸」的地方

**stdout 归协议独占。** `serve_stdio` 把进程内 `sys.stdout` 改指 stderr，协议写端握住真正的 stdout。
Agent 链路里有不少 `print`（`ops_notice`、watchdog 缺失提示……），随便一行插进 ndjson，编辑器侧的解析就断了。
子进程实测里 "未安装 watchdog" 那行确实出现在 stderr，协议流干净。

**`session/prompt` 不能占住读循环。** 一轮 ReAct 可能跑几分钟，编辑器的「停止」按钮得在这期间送达。
所以 prompt 丢给工作线程，读循环继续收 `session/cancel`，回包由工作线程自己发（`DEFERRED` 哨兵）。

**取消要走两条路。** 光把会话上的 cancelled 置位，只能在 ReAct 步与步之间生效；
正卡在模型返回上的那一轮不认这个——`core/react_invoke.py` 读的是 Console 的取消登记表。
所以 `session/cancel` 同时调 `request_agent_cancel()`，而 `run_acp_turn` 必须先 `try_register_agent_chat()` 把自己登记进去（否则那个调用直接返回 False，什么都不会发生）。

### 3. 会话与 Web Console 共用，不是另起一套

`run_acp_turn` 复用 `RUNTIME_MANAGER`（MCP、沙箱、index watch）、会话保活池、session 锁与 `messages.jsonl`。
于是编辑器里的 sessionId 就是 `cli-xxxxxxxx`，聊到一半可以 `llgraph --thread-id <sessionId>` 切回 CLI 接着聊；
同一时刻只能一边操作（锁 owner 记成 `acp`，与 CLI / Web 互斥的提示语一致）。

### 4. trace → ACP 的映射

- 正文流 → `agent_message_chunk`；模型没走流式时，收尾再把全文发一条，保证编辑器一定拿得到答复
- 思考 → `agent_thought_chunk`。trace 的 `thinking_update` 每次给**全文**，只发新增后缀，否则编辑器里会看到一段话被重复刷出来
- 工具步骤 → `tool_call`。llgraph 的步骤是工具跑完才登记（带耗时和输出），所以直接发 completed，没有 pending → in_progress 的中间态
- 工具名 → ToolKind：llgraph 自带工具查表（`search_replace` 打头是 search，其实是编辑，靠表纠正）；MCP 工具按动词位判，分词直接复用 `permissions/mcp.py` 的 `split_tool_name`，免得同一批工具名在两处被切成不同的词
- trace 行（`line()`）**不外发**：ACP 没有对应更新类型，塞进思考里只会吵

## 改了哪些路径

- `llgraph/editor/__init__.py`、`llgraph/editor/acp/{__init__,jsonrpc,updates,sink,turn,server}.py`（新）
- `llgraph/cli/acp_cli.py`（新）、`llgraph/main.py`（子命令分流）
- `tests/test_acp_server.py`（新，18 例）、`tests/test_acp_end_to_end.py`（新，2 例）
- `README.md`（「在编辑器里用（ACP）」+ 常用 CLI + 项目结构）
- `docs/项目结构.md`（3.11 节）、`docs/模块说明.md`（`editor/acp/` 一节）
- `AGENTS.md`、`.cursor/rules/cloud-agent.mdc`（排队推进）

## 怎么验收

- `python3 -m pytest tests -q` → **956 passed, 10 skipped**（本轮前 936；新增 20 例。skip 全是本机没装的可选依赖：`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` / `langchain-google-genai`）；`ruff check llgraph tests` 通过
- 协议层（`tests/test_acp_server.py`，turn_runner 注入桩件、不打模型）：
  握手与版本协商、`session/new` 取 cwd / 拒相对路径 / 回落到 `-C`、
  过程事件**排在回包之前**送达、跑一轮时能收 `session/cancel` 并回 `stopReason: cancelled`、
  同会话第二次提问被拒、执行失败回 JSON-RPC 错误且会话不卡在 busy、
  畸形 JSON 回 `-32700` 之后连接仍可用、通知不回包
- 端到端（`tests/test_acp_end_to_end.py`，本机 stub server 说 Anthropic 协议的 SSE，**真跑 ReAct**）：
  提问 → 模型要求 `read_file` → 工具真读到文件 → `tool_call` 更新里带文件内容且 kind 判成 read →
  正文分两个 `agent_message_chunk` 流出来 → `stopReason: end_turn`；另一例断言默认只读时
  发给模型的工具定义里没有 `write_file` / `search_replace`
- 真子进程（像编辑器那样跑 `python -m llgraph acp -C <ws>`，stdin/stdout 交互）：
  三个请求依次拿到 `protocolVersion: 1` → `sessionId: cli-xxxxxxxx` → `stopReason: end_turn`，
  中间收到 `tool_call` 与两条正文 chunk；watchdog 缺失的警告落在 stderr，没污染协议流
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常

## 未做 / 下一步不要做

- **不要**重写这层骨架去换 VS Code 扩展。要做 VS Code，是在这套之上再加一个薄客户端，不是推倒。
- **不要**把 `session/prompt` 改回在读循环里同步执行。看着简洁，代价是编辑器点「停止」没反应。
- **不要**只靠会话上的 cancelled 标记做取消，也不要绕开 `try_register_agent_chat` ——
  这两件是同一个机制的两半，少一半，长模型调用就停不下来。
- 没做 `session/request_permission`：写权限现在靠 `llgraph acp --write` 一刀切开，
  编辑器里不会弹授权框。**下一轮该攻这个**——ACP Agent 不能改代码，「在编辑器里干活」就只做了一半。
  动手前先想清楚：llgraph 的写工具没有回调式授权点（`permissions/` 是策略判定，不是交互），
  要么在工具层加一个可注入的确认钩子，要么在 ACP 侧包一层工具代理。
- 没做 `session/load`：能力声明为 `false`。编辑器重启后拿着旧 sessionId 接不回去，
  要补得把 `messages.jsonl` 回放成 `session/update`（用户 / 助手 / 工具三类都要映射）。
- 没做 `fs/read_text_file` / `fs/write_text_file` 反向请求。接了能读到编辑器里**未保存的缓冲区**，
  比我们自己读磁盘准；但那会让文件工具多一条来源，先把授权那件做完再说。
- 没做图片 / 音频 prompt（`promptCapabilities` 里声明为 false），`resource` 块目前折成文本塞进正文。
- 工具步骤没有 in_progress 中间态——要有，得在 trace 侧加「工具开始」的 sink 事件，
  那是动 `trace_display` 的公共路径，为编辑器一个消费者改它不值当。
