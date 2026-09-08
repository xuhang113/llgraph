# 2026-09-08 会话落盘：写被打断不毁会话，坏一行不丢整段历史

选题：**稳定性（会话损坏）**。上一轮 `2026-09-07-cold-start-latency.md` 结尾点名
「速度低垂果实已摘完，下一轮换稳定性或商用体验」，本轮接这条走稳定性。

挑这个切口的理由是它的失败形态最狠：不是慢一点，是**用户的整段对话没了**，
而且没有任何提示。llgraph 的会话正文、meta、manifest、锚点、编辑账本、spill、
压缩前归档，全部是 `open("w")` 就地截断再逐行写：

- 写到一半被 Ctrl-C / SIGTERM / OOM / 磁盘写满打断，磁盘上留下的就是半截文件；
- 读侧 `load_session_messages` 一遇 `JSONDecodeError` 就 `return []`，
  等于**一个坏字节报废整个会话**；
- 更糟的是坏文件被读成空之后，本轮结束 `persist_agent_session` 会用 Agent
  状态全量覆盖它，坏字节被冲掉，人工也救不回来。

先量再改，量的是「会不会真发生」而不是墙钟。用旧写法起 4 个写者 + 1 个读者
（每条消息约 24KB，真实会话动辄几百 KB，超过文本缓冲，`close` 之前就会多次
`write`）：**1466 次读里 484 次读到坏行或不完整版本**（33%）。这不是理论风险，
CLI 落盘的同时 Web Console 轮询、会话列表统计条数，撞上就是这个概率。
小消息反而撞不上（一次 `write` 就写完），所以这个 bug 平时只在长会话上现形——
正是最不该丢的那种会话。

## 做了什么

**1. 新增 `llgraph/session/atomic_store.py`：同目录唯一 tmp + `os.replace`**

`os.replace` 在 POSIX / Windows 上都是原子替换，读侧任何时刻看到的要么是完整的旧版、
要么是完整的新版，不存在中间态。三个入口 `atomic_write_text` / `atomic_write_json` /
`atomic_write_jsonl`（jsonl 逐行流式写 tmp，不在内存里拼整份）。

几个刻意的取舍：

- tmp **必须同目录**：`os.replace` 跨文件系统会抛 `OSError`，`/tmp` 与 `~/.llgraph`
  很可能不在同一挂载点，所以不用 `tempfile` 的默认目录。
- tmp 名带 `pid + 线程 id + 计数`，且是 `.` 开头 + `.tmp` 结尾的隐藏名。
  原来 `todo_store` / `index_progress` 用的是**固定 tmp 名**（`.json.tmp` / `.tmp`），
  两个写者会往同一个 tmp 里对写，原子写反而变成新的竞态源；隐藏名则保证
  `session_registry` 的 `*.jsonl` glob、`load_session_messages` 都不会误捡到它。
- 任何异常（含序列化失败）都清掉 tmp 再原样抛出，替换没发生，目标文件仍是上一版。
- 本进程首次写某目录时顺手清一遍**超过 1 小时**的残留 tmp（进程被 kill 在 replace
  之前会留下）。只扫一层、只删够老的，不会误删同机其它进程正在写的 tmp。

**2. 会话正文读侧改成容错恢复**

`read_session_message_rows` 逐行解析，坏行只丢那一行；整批 `messages_from_dict`
失败时退化成逐行转换，坏行单独丢弃。有丢弃就顺手改写回干净的一份，下次读不再有坏行。

**3. 无法解析的原始字节先留一份再继续（`quarantine_corrupt_messages`）**

坏行跳过之后本轮结束仍会全量覆盖 `messages.jsonl`。所以读到坏行时先把原始字节
另存 `messages.jsonl.corrupt-<UTC 时间戳>`（最多留 3 份，避免无限堆积），
并按 `logging.warning` 告知跳过条数与副本路径。整份都读不出来时同样留副本再返回空，
不再静默清空。

**4. 把同一类写法一次改干净**

只改 llgraph 自己的状态文件，**没有**碰 `filesystem_tools` 写用户源码的三处
`write_text`——那涉及符号链接、硬链接、文件权限与属主语义，原子替换会改变行为，
不属于本轮。

顺带修的两个真 bug（都在这一类里）：

- 编辑快照 `snapshots/*.txt` 是 `/undo` 的唯一原文来源，半截快照会把「回滚」
  变成「毁文件」；
- `context_spill` 落盘的工具正文在上下文里已被指针替换，正文只剩这一份，
  半截 spill 会被模型当成完整工具输出。

## 改了哪些路径

- `llgraph/session/atomic_store.py`（新）
- `llgraph/session/session_file_store.py`（原子写 + 容错读 + 坏行留档）
- `llgraph/session/session_meta.py`、`session_manifest.py`、`session_run_log.py`、
  `web_trace_store.py`、`session_edits.py`（meta / 账本 / 编辑快照）
- `llgraph/context/conversation_anchor.py`、`context_compressor.py`（压缩前全量归档）、
  `context_spill.py`
- `llgraph/core/todo_store.py`、`llgraph/code_index/index_progress.py`（固定 tmp 名 → 唯一 tmp）
- `llgraph/memory/paths.py`、`llgraph/subagent/persist.py`、`llgraph/subagent/registry.py`
- `tests/test_session_store_durability.py`（新，10 例）

## 怎么验收

回归测试断的是**耐久性契约**，不是墙钟：

- 写失败（`OSError` / 序列化失败）后目标文件仍是完整旧版，且不留 tmp
- 4 写者 + 1 读者并发跑大消息，读侧一次都不该看到坏行或不完整版本，
  收尾落盘必须是某个写者的完整版本
- 尾行被截断时保留前面的历史（不再整段清空）；中间坏一行只丢那一行
- 整份读不出来时原始字节已留档，副本内容逐字节等于坏文件
- 副本最多 3 份，最新一份必须在
- 残留 tmp 不影响读，够老的能被清掉，够新的（别的进程可能正在写）不许动

三处**变异验证**（把改动逐个退回，对应用例必挂）：

- 原子写退回就地截断写 → 并发用例挂在「读到不完整版本：1 行 / 读到 2 条坏行」
- 容错读退回「一遇坏行 return []」→ 截断用例挂在 `assert 0 >= 7`
- 旧写法 + 旧读法的独立复现脚本：并发 1466 次读里 484 次拿到坏文件；
  尾行截断时旧读侧的结果是「整份丢弃」

手工：

- `python -m pytest -q` → **558 passed, 3 skipped**（本轮前 548 + 本轮新增 10；
  3 skip 仍是 lancedb / fastapi 可选依赖）
- `python -m ruff check llgraph tests` → All checks passed
- `pip install -e ".[web,terminal]"` 后 `llgraph --help` 50~54ms（与上一轮 35~75ms
  同档，`atomic_store` 纯标准库，没把启动预算吃回去）；
  子解释器验真 `import llgraph.main` 后 `sys.modules` 里仍无
  `anthropic` / `langchain_anthropic` / `langgraph`
- `llgraph --list-sessions` 正常；save → load 往返消息逐条一致；
  落盘行格式与改前一致（每行 `{"type","data"}`、结尾换行）；
  `meta.json` 的 `message_count` / `store` / `messages_format` 不变
- 构造截断会话跑一次 load：stderr 打出「会话历史有 1 条无法解析，已跳过并保留 5 条；
  原始副本: …/messages.jsonl.corrupt-<ts>」，目录里副本在、正文已被改写干净

## 未做 / 下一步不要做

- **不要**把 `filesystem_tools` 写用户源码的 `write_text`（`llgraph/core/filesystem_tools.py`
  三处）也改成原子替换。`os.replace` 会断开硬链接、改变 inode，也可能丢掉原文件的权限
  与属主；用户源码不是我们的状态文件，要动得先把权限/符号链接语义想清楚，本轮刻意不碰。
- **不要**为了「更保险」给每次写都加 `fsync`。`atomic_store` 留了 `fsync` 开关但默认关：
  会话每轮都要落盘，逐次同步磁盘会把落盘从微秒级拖到毫秒级，而 `os.replace` 已经解决
  了「读到半截」这个真问题；`fsync` 只多防「掉电」，代价与收益不成比例。
- **不要**把 tmp 名改回固定名（`.json.tmp` 之类），也不要去掉 `.` 前缀：
  前者是并发对写的根源，后者会被 `*.jsonl` / `*.json` 的 glob 捡到。
- **不要**把残留 tmp 的清理阈值调小到分钟级或改成「每次写都扫」：
  同机可能有别的 llgraph 进程正在写同一目录，扫太勤既有误删风险又白付 `iterdir`。
- 没有做「按 mtime/内容哈希做跨轮 read 缓存」那条性能线（`2026-09-06` 提到的
  `tool_loop_guard` 只在最近一条真实 user 之后去重）。它是真切口，但要先解决
  「指针指向的历史结果会被 `tool_prune` / 压缩剪掉」的问题，否则省 token 换来丢上下文，
  是拿稳定性换性能。下一轮要做它，**先**把「被引用的工具结果不许被剪」这个约束做出来。
- 下一轮建议：**性能**（上面那条跨轮重复读，按上述顺序做），或继续**稳定性**里
  尚未碰过的「工具失败恢复 / 死循环」——`tool_loop_guard` 目前只拦重复调用，
  连续失败（同一路径改不动、search_replace 老是 old_string 不唯一）还没有升级策略。
