# 2026-09-17 模型入口接四家：官方 Key 或本地 Ollama 都能直接跑，网关仍是默认路径

选题：**排队第 1 件（模型入口）**。`2026-09-16` 把这件事排到队首时写的是：

> 「没有改 `llm.py` 接多家厂商（交给定时轮第 1 件）」

改造前 `llgraph` 只有一条路：`LLGRAPH_API_BASE_URL` + `LLGRAPH_API_KEY` 的 OpenAI 兼容网关。
手上只有 Anthropic / OpenAI / Gemini 官方 Key、或只想拿本机 Ollama 跑的人装完就卡在
「缺少环境变量: LLGRAPH_API_BASE_URL」，而报错里根本没提还有别的路可走。

## 做了什么

### 1. 新增 `config/providers.py`：先回答「走哪家」

支持 `gateway`（不动的默认）、`anthropic`、`openai`、`gemini`、`ollama`。选择顺序：

1. `LLGRAPH_PROVIDER`（或工作区 `agent.json` → `llm.provider`，后者更靠前）
2. **网关凭据齐就走网关**——这条在第 2 位是刻意的：老用户机器上同时 export 着
   `ANTHROPIC_API_KEY`（Claude CLI 用）时，不能因为本轮改造把请求悄悄从网关挪到官方 API 上
3. 模型 id 指向、且那家配了 Key（于是 `/model gemini-2.5-flash` 能顺带切入口）
4. 唯一配了 Key 的官方厂商
5. 本机 `ollama serve` 在听（TCP 探一下 11434，结果缓存 30s；`LLGRAPH_OLLAMA_AUTODETECT=0` 关掉）

Key 变量：`LLGRAPH_<厂商>_API_KEY` 优先，其次官方名（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
`GEMINI_API_KEY` / `GOOGLE_API_KEY`）——「开箱」就是指这一层回退。
但 base_url **只**认 `LLGRAPH_<厂商>_BASE_URL`：官方 `*_BASE_URL` 常被别的工具指到自建代理上，
自动继承会让「我配的是官方 Key」和「请求实际去了哪」对不上。`OLLAMA_HOST` 是例外，它本来就是地址变量。

`LLGRAPH_MODEL` 没配时用该入口的默认模型（`claude-sonnet-4-5` / `gpt-4.1` /
`gemini-2.5-flash` / `qwen3:8b`），否则只 export 一个 `OPENAI_API_KEY` 的人会拿着
`claude-opus-4-6` 去问 OpenAI。

### 2. `core/llm.py` 改成 `create_chat_llm` 工厂

按 provider 建 `ChatAnthropic` / `ChatOpenAI` / `ChatGoogleGenerativeAI` / `ChatOllama`；
`create_gateway_llm` 保留为旧名别名（8 处调用点与脚本不用改）。
网关那两个补丁（`usage.cache_creation=null`、kimi reasoning payload）只在网关路径打，
官方 Anthropic 不受影响。langchain 包没装时报「pip install langchain-xxx」而不是 ImportError 栈。

### 3. 厂商专属参数分家（这是真正会炸的地方）

- `parallel_tool_calls` / `tool_choice="auto"`：只发 Anthropic / OpenAI 协议。
  Gemini、Ollama 的 `bind_tools` 不会因为多个未知 kwarg 抛 `TypeError`，
  而是把它原样塞进请求体，等服务端报错——原来的 `except TypeError` 兜不住这种。
- `cache_control` 断点：只在网关 / 官方 Anthropic 打。别家请求体里没这个字段，
  打上去等于往工具定义里塞一段没人认的 JSON。
- thinking：Anthropic 协议照旧；OpenAI 侧只在配置点名了 effort 时映射成 `reasoning_effort`
  （非推理模型收到它会 400，所以不按模型族猜）。
- provider 挂在模型实例上（`llgraph_provider`），`bind_tools` 与 prompt cache 读它决定发什么。

### 4. 顺带修掉两个「换了入口就露出来」的问题

- `/model list` 原来无条件调 `fetch_gateway_models()` → 没网关凭据时直接抛异常，
  走官方 Key / Ollama 的人一执行就崩。现在没凭据返回空列表。
- 本地模型（`qwen3:8b` 这类带 `:` 标签的 id）上下文窗口按 32K 算，不再落到 200K 默认值。
  200K 意味着自动压缩阈值 170K，本地模型第一轮长上下文就会被服务端截断，而用户看不到原因。
- 启动前的凭据预检从 `get_llgraph_settings()` 换成 `verify_model_credentials(workspace)`，
  一家都没配时把四条路一起列出来，不再只报缺哪两个网关变量。

## 改了哪些路径

- `llgraph/config/providers.py`（新）
- `llgraph/config/config.py`（`configured_model_or_none` / `gateway_credentials_present`）
- `llgraph/core/llm.py`（重写为多 provider 工厂）
- `llgraph/core/llm_settings.py`（`resolve_explicit_model` / `resolve_effective_provider`；status 与 banner 显示入口）
- `llgraph/core/react_graph.py`（`_bind_tools_if_needed` 按 provider 传参）
- `llgraph/core/prompt_cache_settings.py`（非 Anthropic 协议不打断点）
- `llgraph/core/model_context_window.py`（本地标签模型 32K）
- `llgraph/core/gateway_models.py`（无网关凭据不抛错）
- `llgraph/main.py`（启动预检）
- `pyproject.toml`（`openai` / `gemini` / `ollama` / `models` extras）、`scripts/install.sh`（默认多带 `models`）
- `tests/test_model_provider_entry.py`（新，26 例）、`tests/test_provider_tool_call_wire.py`（新，3 例）、`tests/conftest.py`
- `examples/llgraph.env.example`、`README.md`、`docs/操作手册.md`、`docs/模块说明.md`、`docs/项目结构.md`

## 怎么验收

- `python3 -m pytest tests -q` → **942 passed, 4 skipped**（本轮前 913 passed；4 skip 全是本机缺
  `fastapi` / `lancedb` 可选依赖）；`ruff check llgraph tests` 通过；`bash -n scripts/install.sh` 通过
- `pip install -e .` 后 `llgraph --help`、`python3 -m llgraph --help` 正常
- 协议层（stub server，不打真 API，本地端口起 HTTP）：三条路各跑通「请求带工具定义 →
  返回 tool_calls → 回灌 tool 结果 → 拿到正文」
  - 网关（`/v1/messages`，Anthropic 协议）：**这条是回归重点**，证明改造后默认路径没变
  - `provider=openai`（`/v1/chat/completions`）：请求里有 `parallel_tool_calls: true`
  - `provider=ollama`（`/api/chat`）：请求里**没有** `parallel_tool_calls` / `tool_choice`
- 选择逻辑回归：网关凭据在时即便同时 export 了 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` 也仍走网关；
  只配一家 Key 时用那家的默认模型；两家 Key 都在时按模型 id 走；
  `agent.json llm.provider` 压过环境变量；一家都没配时模型名仍可解析、只在建客户端时报错
- 真本地 Ollama（本机 `ollama serve` + `qwen3:1.7b` / `qwen3:4b`，**未配任何 `LLGRAPH_API_*`**）：
  - 直接建客户端跑一轮：模型发起 `city_time` 工具调用 → 回灌结果 → 给出正文
  - 真 CLI（`llgraph --once -C <ws>`）：纯对话一轮 26s 完成；带工具那轮 `read_file(hello.txt)`
    被正确调用并读到文件内容（1.7b / 4b 之后会重复调工具，是小模型能力问题，不是入口问题）

## 未做 / 下一步不要做

- **不要**把「网关凭据齐 → 走网关」这条从第 2 位挪走，也不要让官方 `*_BASE_URL` 参与解析。
  这两条都是为了同一件事：老用户升级后请求去向不能变。
- Gemini 没有协议层 stub 测试：`langchain-google-genai` 走 google-genai SDK，
  地址不经 llgraph 配置，起 stub 要么改环境变量硬钩 SDK、要么 mock 掉客户端（就测不到协议了）。
  只测到了客户端参数。真要补，先想清楚测的是协议还是我们的参数。
- **不要**顺手把 `ollama` 入口改成走它的 OpenAI 兼容层（`/v1`）。现在用的是原生 `/api/chat`，
  `num_predict`、tools 都对得上；换 `/v1` 会多一层它自己的转换，出问题更难定位。
- 没有做「一个会话里混用多家」（比如主 Agent 用 Claude、子 Agent 用本地模型）。
  provider 现在是进程/工作区级的，`subagent/runner.py` 与 `session_title_llm.py`
  都复用同一个工厂。真要做，得先决定 `/model` 的语义是「换模型」还是「换入口+模型」。
- 没有给 `/model list` 接官方厂商的模型列表（Ollama 的 `/api/tags`、OpenAI 的 `/v1/models`）。
  现在非网关入口列表为空，只能靠 `agent.json llm.models` 或直接 `/model <名>`。
- 本地模型上下文窗口是按 `:` 标签一律 32K 的保守值，没有去读 ollama 的 `/api/show`。
  真要准就读它，但那是一次网络往返 + 缓存，别塞进模型名解析这条热路径。
- 下一件按 `AGENTS.md` 排队走：**编辑器里干活（VS Code 扩展或 ACP 插件）**。
  终端 TUI 仍然后置。
