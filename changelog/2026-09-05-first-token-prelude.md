# 2026-09-05 首 token 前奏提速：空记忆库不冷启动，工具历史不二次方回溯

选题：速度。接上一轮 prompt cache 留下的第二条线——「首 token 前的同步前奏（workspace context / memory recall / manifest 同步）能否并发或延迟到首次真需要时再做」。

结论：前奏里真正贵的两块都不是「该做但慢」，而是**根本不该做**。全新工作区首轮花约 2.3s 加载 embedding 模型和 lancedb，只为确认记忆库里一行都没有；`_recent_read_paths` 的路径正则在无空格长行上二次方回溯，60 条工具结果的历史要 2.78s。修完首轮前奏合计从 **1503ms 降到 34ms**（历史含 lock/压缩 JSON 时 4262ms → 30ms），真实 CLI `--once` 墙钟 5.9~6.2s → 3.9~4.1s。

## 做了什么

先量后改。在真实仓库工作区上把前奏逐步打表（20 轮 / 160 条消息 / 362k 字符历史），耗时分布极度偏斜：`tool_prune` 3.6ms、`compress` 2.2ms、`manifest_sync` 8.7ms、`sanitize` 12.6ms 都无话可说，但 `memory_recall` 1453.7ms、`workspace_context` 在特定正文下 2761.8ms。所以本轮没有去做「前奏并发化」——把 30ms 的东西并发起来毫无意义，两个大头拆掉之后前奏已经不在预算里了。

**1. 空记忆库首轮白付约 2.3s（`memory/recall.py`、`memory/store.py`、`memory/paths.py`）**

`recall_memories` 原来先 `embed_memory_text` 再查表。本地 `sentence-transformers` 首次加载实测 1562ms，而新工作区的记忆库是空的——这 1.5s 换来的是「在 0 行的表上做向量检索」。

改成先取候选行再决定要不要 embed。这不是启发式：向量检索与关键词过滤读**同一张表、同一组条件**（user / workspace / status=active / kinds，见 `search_memory_vectors` 的 where 子句），候选为空时向量必然也为空，短路可证等价。有候选行时照常走向量召回，一步不少。

短路之后还剩 819ms，全在 `import lancedb`——`list_memory_rows` 第一件事就是 `connect_memory_db`，而新工作区的 lance 目录压根还没建。加 `memory_store_is_definitely_empty` 做纯文件系统前置判断：目录缺失、或存在但一个条目都没有，直接返回空。刻意**不猜 lancedb 的落盘布局**（不去找 `agent_memories.lance`），目录里有任何内容都交回 lancedb 正常读，这样将来 lancedb 换布局也不会把有数据的库误判成空。副作用是好的：读路径不再 `connect`，也就不会被读操作创建空目录，跨进程持续有效。后台整理走 `ensure_memory_dirs` 建出的空目录由「空目录」那一支兜住。

**2. `_recent_read_paths` 的二次方回溯（`context/context_continuity.py`）**

`_PATH_IN_TOOL_RE` 是 `[^\s`'"><|]+\.(?:md|json|…)` 这种形状：起点不锚定，于是在每个位置贪婪吃到行尾、再一路回溯找扩展名。正文里只要有**长的无空格串**就退化成 O(n²)——package-lock.json、压缩 JSON、base64 数据 URI、CSV 单行、`uv.lock` 全是这个形态，单条 4000 字符正文 45ms，60 条工具结果的历史 2.78s。

起点锚定到分隔符（`(?:\A|[\s`'"><|])`）后，同样正文 0.077ms（600x），真源码 / Markdown 也快 3~12x。`group(1)` 与旧写法逐个相同，4000 例随机模糊比对零差异——纯性能修复。

触发条件比想象中常见：`build_continuity_context_hint` 的门是「继续 / 接着 / 刚才 / 上一轮 / 重写 / 那份」，是改代码会话里最普通的开场白。

**3. 顺带修掉 embedder 的并发重复加载（`code_index/local_embedder.py`、`memory/scheduler.py`）**

库非空时首轮仍要付一次约 1.5s 的模型加载。会话建好后（CLI 与 Web/Console 两条 bootstrap）起 daemon 线程预热，用户敲第一条消息的时间就把它盖掉；空库或 remote provider 不预热，避免白占内存。

`_get_model` 原来是无锁双查 dict。加上预热线程后这个洞会立刻暴露：后台预热与前台首次调用同时进来，各加载一份权重。改成双检加锁，并发只加载一次、后到者拿同一实例。这个洞在 Web 并发请求 / 子 Agent 场景里本来就存在。

## 改了哪些路径

- `llgraph/memory/recall.py` 候选为空时短路，不再无谓 embed
- `llgraph/memory/paths.py` 新增 `memory_store_is_definitely_empty`
- `llgraph/memory/store.py` `list_memory_rows` / `search_memory_vectors` 前置空库判断
- `llgraph/memory/scheduler.py` 新增 `schedule_memory_embedder_prewarm`
- `llgraph/code_index/local_embedder.py` `_get_model` 双检加锁；新增 `local_embedder_is_loaded` / `prewarm_local_embedder`
- `llgraph/core/session_bootstrap.py`、`llgraph/main.py` 接预热
- `llgraph/context/context_continuity.py` `_PATH_IN_TOOL_RE` 锚定起点
- 新增 `tests/test_invoke_prelude_latency.py`、`tests/test_memory_recall_cold_start.py`、`tests/test_memory_embedder_prewarm.py`

## 怎么验收

- 首轮前奏合计（真实仓库工作区，20 轮 / 160 条消息 / 362k 字符历史，每步只跑一次）：

  | 工具正文形态 | 修复前 | 修复后 |
  | --- | --- | --- |
  | 真源码 | 1503.0 ms | 34.3 ms |
  | 无空格长行（lock / 压缩 JSON / base64） | 4262.3 ms | 29.8 ms |

- 分项：`memory_recall` 1453.7 → 0.3ms；`workspace_context` 22.1 → 6.8ms（真源码）、2761.8 → 5.3ms（无空格长行）
- 真实 CLI 链路（stub OpenAI 兼容网关，`llgraph --once`，全新工作区）墙钟 5.92 / 6.21s → 4.07 / 3.85s；出站 payload 里 `<workspace-context>`、`<user_query>`、16 个工具定义都在
- 子进程验真：全新工作区跑完一次召回，`lancedb` / `torch` / `sentence_transformers` 都不在 `sys.modules`
- `python -m pytest`：487 passed / 42 failed。42 个与基线 `f134164` 逐条相同（缺网关凭据、langchain-core 版本导致的工具 invoke 签名、execution_log 时区等），**零新增**；`ruff --select F821,F823` 也是基线那 2 个
- 四个修复全部做过变异验证（把改动逐个回退，对应用例必挂）：`_recent_read_paths` 退回后测出 2776ms、空库短路退回后 `embed_memory_text` 被调用、lancedb 前置判断退回后 `sys.modules` 里出现 lancedb、去掉锁后 4 线程加载了 4 次模型
- `uv pip install -e .` 与 `llgraph --help` 正常

## 未做 / 下一步不要做

- 不要为了「保险」把读路径上的 `connect_memory_db` 改回无条件调用，那 0.8s 会原样回来
- 不要把 `memory_store_is_definitely_empty` 扩展成「按 `agent_memories.lance` 判断表在不在」，那是在猜 lancedb 的落盘布局，换版本就可能把有数据的库判成空；只在「目录不存在 / 目录里一个条目都没有」这两种无歧义情形下短路
- 不要去掉 `_PATH_IN_TOOL_RE` 的起点锚点，也不要在这条正则里加新的贪婪负字符类；要加扩展名就加在 `(?:md|mdc|…)` 里
- 不要在空库或 remote provider 上预热模型，那只是把浪费从「同步」挪到「后台」
- 没有做「前奏并发化」：拆掉两个大头后整段只剩约 34ms，再拿线程去切它得不偿失
- 下一步可攻：上一轮留下的另一条线仍然开着——`dispatch_full_tool_budget_tokens` 默认 12% 窗口是在「缓存无效」假设下定的保守值，现在缓存真的命中了，缓存读只要 0.1x 价钱，多留全文几乎不涨 TTFT，值得实测重定默认值。另外本轮量到 `import llgraph.core.agent` 单独就要 2.2s、`build_agent` 冷启动 0.8s（主要花在 19 个工具的 pydantic schema 生成），这是 CLI 敲下回车到出提示符之间的固定成本，可考虑按需延迟绑定工具 schema
