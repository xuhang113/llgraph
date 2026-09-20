# 2026-09-20 编辑器重启后接得回去：ACP `session/load` 续聊

选题：**排队第 2 件（编辑器里干活）**。`2026-09-19` 授权弹窗那篇末尾点名的就是这件：

> 「没做 `session/load` 续聊：能力仍声明为 `false`，编辑器重启后拿旧 sessionId 接不回去。
> **下一轮就是这件**——要把 `messages.jsonl` 回放成 `session/update`（用户 / 助手 / 工具三类都要映射），
> 注意工具那类回放不出耗时与输出，得决定是折成文本还是只发标题。」

之前 `loadSession: false`，编辑器一关，那个会话在编辑器里就没了：历史还在
`~/.llgraph/context/.../messages.jsonl`（`llgraph --thread-id` 还接得回来），但编辑器不认。

## 做了什么

### 1. `editor/acp/replay.py`：存储里的消息 → 编辑器里的更新

ACP 的 `session/load` 是**由 Agent 侧把整段对话重新推一遍**，客户端照单渲染，
所以这层做的是映射，不是「恢复状态」。三类消息的落点：

| 存的 | 推的 | 说明 |
|------|------|------|
| HumanMessage | `user_message_chunk` | 先剥掉注入块（`strip_injected_context_from_user_message`），编辑器里显示的是用户真打过的字 |
| AIMessage | `agent_message_chunk` + `tool_call` | 正文与它发起的工具调用按原顺序 |
| ToolMessage | 并进对应的 `tool_call` | 按 `tool_call_id` 配对 |

三个决定：

- **工具那类折成文本，不是只发标题。** 回放确实拿不到耗时（trace 那份没落盘），
  但输出本身就在 ToolMessage 里。只发标题的话，用户重启后看到的是一排「执行 read_file(a.txt)」
  却点不开，比实时那轮退化得太多。折行上限取 20（实时是 40）：回放的是整段历史，不是一轮。
- **`system` 与注入的 manifest / anchor / 摘要不回放。** 那些是给模型看的上下文，
  在聊天区里画出来，用户会以为自己发过这些话。判定直接用
  `message_normalize._is_business_user_message`，与 Web 聊天区同一套口径。
- **toolCallId 用 `load_N` 前缀。** 实时那轮的 sink 发的是 `call_1`、`call_2`，
  回放完接着聊就会撞号，编辑器会把两次不同的调用画成同一次。

历史太长时从最近一端截断（默认 200 条），并在最前面补一条 `agent_thought_chunk` 说明省略了多少。
ACP 没有「系统提示」这类更新，折进思考块是唯一不假装成用户 / 助手发言的位置。
被截断处剩下的孤立 ToolMessage 仍会单独发一条——否则那次调用凭空消失。
**模型侧不受截断影响**：它的历史是下一轮 prompt 时会话保活池自己从 jsonl 读的，与回放两回事。

### 2. `server.py`：`session/load` 与 `loadSession: true`

- 会话存在性按**工作区**判（`session_is_resumable`）：历史本来就按工作区分目录存，
  别的工作区的 sessionId 接不回来。只有 `meta.json`（开了会话没说话就重启）也算可续，回放为空。
- **回放交给工作线程**，和 `session/prompt` 同一个理由：读循环不能被占住。
  顺带用同一个 `busy` 位占住会话——编辑器紧接着发 prompt 会被挡回去，不至于两边同时写历史。
  失败时回 JSON-RPC 错误并把 `busy` 放掉，会话不卡死。
- `cwd` 解析与 `session/new` 共用一份（顺手抽出 `_resolve_workspace` / `_make_session`），
  因此 `llgraph acp -C <目录>` 那条兜底对 load 一样生效。
- 回包给 `{}` 而不是 `null`：ACP 后续版本在这个响应里放可选字段，对端按对象解析更稳
  （与既有的 `authenticate` 一致）。
- `AcpServer(history_loader=...)` 可注入，协议层测试不必真读盘。

## 改了哪些路径

- `llgraph/editor/acp/replay.py`（新）
- `llgraph/editor/acp/server.py`（`session/load`、能力声明、工作区/会话构造抽取）
- `tests/test_acp_session_load.py`（新，19 例）、`tests/test_acp_end_to_end.py`（+1 例，
  顺带让 `clean_runtime` 清掉进程级会话保活池：留着不清，别的用例一调 `get_or_build`
  就会触发淘汰，把 `release_checkpointer` 记到人家的 mock 上）
- `README.md`、`docs/模块说明.md`、`docs/项目结构.md`、`AGENTS.md`、`.cursor/rules/cloud-agent.mdc`

## 怎么验收

- `python3 -m pytest tests -q` → **1003 passed, 10 skipped**（本轮前 983；新增 20 例。
  skip 全是本机没装的可选依赖：`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` / `langchain-google-genai`）；
  `ruff check llgraph tests` 通过
- 回放映射：三类消息落到三种更新且顺序不变；system 与 manifest / anchor 不出现；
  工具结果并进 `tool_call`（不再单独出现一条）、`status=error` 映射成 `failed`、输出按行折断；
  toolCallId 是 `load_N`；被截断时有省略提示且孤立工具结果仍在；空历史回空
- 读盘：`messages.jsonl` 回放得出用户 / 助手两类；另一个工作区的同名 sessionId 判为不可续；
  只有 `meta.json` 的会话可续且回放为空
- 协议层（真 pipe，测试侧扮编辑器）：`initialize` 声明 `loadSession: true`；
  回放事件**排在回包之前**；未知 sessionId / 缺 sessionId / 相对 cwd 回 `-32602`；
  加载中发 `session/prompt` 回 `-32600` 且加载照常完成；加载器抛错回 JSON-RPC 错误且会话不卡在 busy；
  没给 cwd 时回落到 `-C` 指定的工作区
- 续聊的实质（stub server 说 Anthropic 协议，**真跑 ReAct**）：给会话预置一段历史 →
  `session/load` 拿回它 → 再提一轮问 → **发给模型的请求体里有上一轮的内容**
- 真子进程（像编辑器那样 `python -m llgraph acp -C <ws>`，stdin/stdout 交互）：
  预置历史的会话 `session/load` 回 `{}`，随后收到 user / assistant / `执行 read_file(hello.txt)`（kind `read`、
  带文件内容）/ assistant 四条更新；未知 sessionId 报「该工作区下没有会话」；就绪提示仍只在 stderr
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常

## 未做 / 下一步不要做

- **不要**在读循环里同步回放。历史可以很长，读循环被占住期间编辑器的其他消息全都排队。
- **不要**去掉 `load_` 前缀改用存储里的 tool_call id。那些 id 是模型给的，
  与实时那轮的 `call_N` 不在一个命名空间，撞号了编辑器会把两次调用合成一次。
- **不要**把 manifest / anchor 也回放出来「让用户看全」。它们是注入给模型的上下文，
  不是用户发言；真要展示，得先有一种编辑器认的「系统备注」更新类型。
- **不要**在 `session/load` 里提前把 Agent 内存状态恢复出来。保活池在下一轮 prompt 时
  自己从 jsonl 读，提前做等于把冷构建的代价挪到加载这一步，而用户可能只是想看看历史。
- 没有回放思考（`agent_thought_chunk` 只用在省略提示上）：思考没有单独落盘，
  想回放得先决定要不要把它写进 `messages.jsonl`，那是动存储格式。
- 没有回放工具耗时与 `tool_call` 的中间态：耗时在 trace 侧，没进 jsonl。
- 截断上限（200 条 / 每条工具 20 行）是拍的，没有按字节量自适应。
  真要调，先量一遍长会话回放出去的总字节，别凭感觉加。
- 下一件按 `AGENTS.md` 排队走：**`fs/read_text_file` / `fs/write_text_file` 反向请求**——
  接了就能读到编辑器里**未保存的缓冲区**，比我们自己读磁盘准；难点是文件工具会多一条来源，
  得想清楚哪些路径走编辑器、哪些仍走磁盘（沙箱与权限判定都在磁盘那条上）。
