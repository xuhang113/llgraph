# 2026-09-05 Prompt Cache 前缀稳定化（工具循环不再每步全价重算）

选题：速度 / 性能。上一轮拆掉 Plan / Survey 后主 Agent 路径变干净，本轮攻工具循环的首 token 与 token 浪费。

结论：原实现里**可读回的缓存前缀冻结在对话开头**——不管工具循环跑多久，每次请求只能读回 system 那一块，历史全程全价重算。修好后前缀随对话增长（锯齿上升），20 步工具循环的未缓存字符量降到原来的 **26%**（缓存覆盖 3.9% → 54.0%）。

## 做了什么

Anthropic prompt cache 按**精确前缀**命中，前缀一旦在中途被改写，改写点之后的全部内容都要重新计费、重新算 TTFT。llgraph 默认开着 prompt cache，但出站路径上有两处结构性问题让它几乎完全失效：

**1. 出站 tool 链是 recency 滑窗，每步都改写历史（`dispatch_compaction.py`）**

原实现「永远只保留最近 N 条重工具结果全文」。每走一步新增一条重结果，就恰好有一条历史 ToolMessage 由全文变指针——历史中段每步都被改写，缓存每步击穿。

改成**压缩纪元**：全文量在高水位以下时一条都不新压，出站字节与上一步逐字节相同；跨过高水位才一次压到低水位（纪元 +1）。压缩点按纪元跳变而非每步平移，于是「每步一次缓存击穿」变成「每 3~5 步一次」。

量化边界保证纪元内稳定：`epoch = floor((total - low) / (high - low))`，`total` 落在同一纪元区间内时压缩量恒定。单调性上，本 thread 压过的 `tool_call_id` 记进水位，不会被晚到的「引用保护」救回全文——否则一次晚到的引用就让前缀回退，缓存和上下文双输。

默认 `dispatch_full_tool_hysteresis=2.5`、`dispatch_full_tool_budget_tokens` 按模型窗口 12%（夹在 8k~48k）、`dispatch_compact_low_ratio=0.4`。滞回设 1.0 即退回旧滑窗行为。

**2. 对话断点打在出站最后一块，而那是 ephemeral 提醒（`prompt_cache.py`）**

原来走 `llm.bind(cache_control=...)`，top-level 形式让 Anthropic 把断点自动放在**最后一个内容块**上。但出站尾部恰好是 `<system-reminder>` / 预算提醒这类 ephemeral 内容：本步有、下步没有。写进缓存的前缀下一步永远匹配不上，等于每步付 1.25x 写入费再全价重算。

改成显式断点，打在**最后一条稳定消息**上（跳过 ephemeral 尾巴），这才是 Anthropic 多轮缓存的标准用法——断点随对话前移，写入的前缀正好是下一步请求的真前缀。第二个断点放在本轮 user 消息上：压缩纪元推进的那一步会在本轮工具链中途分叉，这个断点至少兜住此前所有轮次的历史（压缩步因此还能读回 8~14% 而不是 0）。

配额上 system 静态前缀与 tools 定义各占一个，消息级最多再用 2 个，不超 Anthropic 单请求 4 个断点上限。同时从 `react_graph` 摘掉 top-level 绑定，否则会多打一个块直接超限报错。

一个易踩的坑：断点只能挂在**块**上。若一条消息「本步带断点＝块形式、下步不带＝字符串形式」，序列化就会变，恰好在想缓存的那一块上把前缀打断。所以凡可能成为断点的消息，无论本步是否带断点都统一成块形式。

**3. `/trace stats` 的 cache 指标从估算改成实测**

原来是 `cacheable_prefix_estimate` 启发式，看不出上面两个问题。改成 `record_dispatch_prefix` 实测相邻两次出站与**上次写进缓存的前缀**的公共前缀（ephemeral 尾部不算进去，否则指标会偏乐观）。

度量口径上踩过一次坑，值得记下来：Anthropic 不是「按最长公共前缀命中」，而是「断点位置 B 写下的条目覆盖块 [0, B]，下一次请求只有自己的 [0, B] 与之**逐字节全等**才能读回该条目，否则回退到更早的断点」。按 LCP 算会高估旧实现的命中率，正好掩盖了「断点落在 ephemeral 尾巴上」这个 bug——因为 LCP 模型下那条无用的写入不影响读回量，而真实语义下整个条目直接作废。

**4. 顺带修掉自己引入的性能回归**

加纪元时把整段工具正文当「消息列表」传给了 `estimate_tokens`，字符串被逐字符迭代走 isinstance 分支：30 轮工具链（约 36 万字符）出站组装从 3.8ms 涨到 61.3ms，一个 30 步 turn 白送近 2 秒。新增 `estimate_text_tokens(text)` 作为单段文本估算的唯一入口，`_heavy_tool_slots` 改用它。误用版返回值与正确版完全相同，是纯性能 bug，所以补的是耗时回归测试。

## 改了哪些路径

- 新增 `llgraph/context/dispatch_compaction.py`（纪元规划 + 前缀稳定度量）
- `llgraph/core/prompt_cache.py` 显式消息级断点；`llgraph/core/react_graph.py` 摘掉 top-level 绑定
- `llgraph/context/incremental_context.py` 出站裁剪改调纪元规划；`llgraph/context/message_normalize.py` 串 thread_id 并在 redact 后打断点
- `llgraph/context/context_settings.py` 新增 3 个 dispatch 配置项（含 `/context` 帮助与当前值）
- `llgraph/context/context_compressor.py` 新增 `estimate_text_tokens`
- `llgraph/commands/meta_commands.py` cache 指标改实测
- 新增 `tests/test_dispatch_cache_epoch.py`、`tests/test_prompt_cache_breakpoints.py`

## 怎么验收

- `python -m pytest tests/test_dispatch_cache_epoch.py tests/test_prompt_cache_breakpoints.py tests/test_dispatch_tool_chain.py`（48 passed）
- 断点测试用 `langchain_anthropic.chat_models._format_messages` 真序列化器比对 block 序列，不是自己模拟的形状
- 全量 `python -m pytest`：490 passed / 21 failed，21 个在基线 commit `1cc369e` 上一模一样（execution_log 时区、memory 需 lancedb、thinking/parallel dispatch 需真网关、prompt_cache_tool_bind 需凭据），零新增
- 关键回归 `test_readable_cache_prefix_grows_with_conversation` 走完整出站管线，按真实语义算每步可读回块数。三种形态泾渭分明，且两种失效都已用变异测试确认能卡住：
  - 断点落在 ephemeral 尾巴：`[0,2,2,2,…]` 永远冻结在 2 块
  - 断点修好但仍是滑窗：`[0,2,5,8,2,2,2,…]` 头几步有效后塌回
  - 现在：`[0,2,5,8,11,14,17,20,2,26,29,32]` 锯齿上升，总覆盖约 71%
- 三配置对比（20 步工具循环，真实语义）：缓存覆盖 3.9% → 17.4% → 54.0%，未缓存字符 169 万 → 146 万 → 44 万（26.1%）。旧实现每步只读回 **1 块**（system），只修断点那档爬到 17 块后塌回 **2 块**——两个修复必须一起上，缺一个都会重新冻结
- 真实 CLI 链路（stub Anthropic 网关，4 轮 read_file，检查落盘的真 payload）：可读前缀从**恒定 23,303 字符**变成随对话增长（23,301 → 24,614 → 34,787 → 37,134），覆盖 60.4% → 73.3%；断点数每次请求 3~4 个，未超 Anthropic 上限
- 出站组装耗时 `prepare_messages_for_llm_dispatch`：3.8ms → 4.6ms（多出的 0.8ms 是指纹哈希与块形状归一，换回约 4 万 token 的缓存）
- `pip install -e .` 与 `llgraph --help` 正常；`/context` 能看到 3 个新配置项

## 未做 / 下一步不要做

- 不要再把 `cache_control` 通过 `llm.bind()` top-level 传（`apply_prompt_cache_to_llm` 只留给诊断脚本），会把断点又打到 ephemeral 尾巴上并可能超 4 断点上限
- 不要给「引用保护」加复活已压缩消息的能力，会破坏单调性、直接废掉纪元
- 不要用「最长公共前缀」评估缓存效果，会高估旧实现、掩盖断点位置的 bug；要按「断点处前缀全等才能读回」算
- 压缩那一步的覆盖仍只有 8~14%，这是有界上下文的固有代价，不是 bug；想再降压缩频率应调 `dispatch_full_tool_budget_tokens` 而不是加断点（实测在已压缩前缀上加断点省不到字节）
- 下一步可攻：现在缓存真的命中了，`dispatch_full_tool_budget_tokens` 默认 12% 窗口这个值是在「缓存无效」假设下定的偏保守值；缓存读只要 0.1x 价钱，多留全文几乎不涨 TTFT，值得实测重定默认值。另一条线是首 token 前的同步前奏（workspace context / memory recall / manifest 同步）能否并发或延迟到首次真需要时再做
