# 2026-09-06 恢复测试信号：缺凭据不崩、单测离线可跑

选题：**稳定性**。上一轮 changelog 说「接工具层方向做深一件」，但落地前先撞到一个更前置的问题：
在干净机器上 `pip install -e . && pytest` 是 **43 red**，其中一半只因为没配网关凭据。
测试红成一片，后续每一轮（含定时 Agent、CI）都拿不到可信信号，改代码等于盲改。所以本轮先把信号修好。

## 做了什么

**1. 模型名解析与网关凭据解耦（运行期收益，不只是测试）**

`resolve_effective_model()` 原来最后一步走 `get_llgraph_settings()`，而后者缺 `LLGRAPH_API_BASE_URL` /
`LLGRAPH_API_KEY` 就抛「缺少环境变量」。但模型名只决定上下文窗口、dispatch profile、Agent 缓存键、
trace 展示——这些只读路径不该因为没配 key 就崩（`/model` 状态、启动横幅、Web Console 用量统计都会中招）。

- `config.py` 新增 `resolve_configured_model()`：只读 `LLGRAPH_MODEL`，缺则回退 `DEFAULT_MODEL`，永不抛
- `llm_settings.py` 的 `resolve_effective_model` / `format_model_status` / `format_model_banner_suffix` 改用它
- 真正建 `ChatAnthropic`（`llm.py`）与拉网关模型表（`gateway_models.py`）仍强校验凭据，`llgraph --once`
  缺凭据照旧打印「配置错误: 缺少环境变量…」后退出 1

**2. 单测不再依赖开发机配置（`tests/conftest.py`）**

- `config.py` 新增 `LLGRAPH_IGNORE_ENV_FILES=1` 开关：置位后 `load_llgraph_env()` 不读
  `~/.config/llgraph/llgraph.env` 与项目 `.env`
- conftest autouse 夹具置位该开关 + 灌固定假凭据（base_url 指向不可路由的 `127.0.0.1:9`）。
  以前没配就整片红、配了则可能真打网关烧真 token，现在两种机器结果一致
- 本地时间展示类断言固定 `TZ=Asia/Shanghai`（原来换时区就红）

**3. 修红/修脆的用例**

- `test_thinking_dispatch`（12 个）：出站尾部现在会挂 `<system-reminder>` 工具往返预算，`prepared[-1]`
  已不是 assistant。改成取最后一条 `AIMessage`
- `test_parallel_tool_dispatch`（3 个）：HTTP 层 reasoning 注入按 `llm.model` 判定，而用例的 llm 建在
  环境默认模型上，参数化形同虚设。改为 `set_runtime_model(model_id)` 后再建客户端，并显式开 thinking
  （deepseek/glm thinking 关闭时按 dispatch 矩阵本就不注入）
- `test_prompt_cache_tool_bind`（2 个）：原来靠真 `invoke()` 打网关才能跑。改为 payload 构造完即抛
  哨兵异常中断，纯离线、快 3 倍，断言不变
- `lancedb`（记忆落盘）、`fastapi`（Web Console）属可选 extra，缺失时 `importorskip` 跳过而非报错
- 修掉仓库 ruff 配置本就选中的 2 个 F821（`agent.py` 的 `BaseMessage`、`session_file_store.py` 的
  `ContextSession`），`ruff check` 恢复全绿

**4. 加 CI（`.github/workflows/tests.yml`）**

push（main / cursor/auto_upgrade）与 PR 上跑 `ruff check` + `pytest` + CLI 冒烟，防止再悄悄变红。

## 改了哪些路径

- `llgraph/config/config.py`、`llgraph/core/llm_settings.py`
- `llgraph/core/agent.py`、`llgraph/session/session_file_store.py`（仅补 import，无行为变化）
- `tests/conftest.py`、`tests/test_model_id_no_credentials.py`（新增）
- `tests/test_thinking_dispatch.py`、`tests/test_parallel_tool_dispatch.py`、
  `tests/test_prompt_cache_tool_bind.py`、`tests/test_memory_store.py`、
  `tests/test_memory_web_search.py`、`tests/test_context_usage_session.py`
- `.github/workflows/tests.yml`（新增）

## 怎么验收

干净机器（不配任何 `LLGRAPH_*`、不装 lancedb）：

- `pip install -e ".[web]" pytest ruff`
- `python -m pytest -q` → **526 passed, 3 skipped**（本轮前：43 failed）
- `python -m ruff check llgraph tests` → All checks passed
- `python -m llgraph --help`、`python -c "import llgraph.main"` 正常
- 缺凭据时 `llgraph --once "hi"` 仍打印「配置错误: 缺少环境变量: LLGRAPH_API_BASE_URL, LLGRAPH_API_KEY」
- 缺凭据时 `format_model_status(ws)` 能正常输出模型与上下文预算（回归用例
  `tests/test_model_id_no_credentials.py`）

## 未做 / 下一步不要做

- 没动 Agent 主链路行为：ReAct 图、工具、压缩、出站修链一行没改。本轮只改「配置耦合 + 测试可信度」
- 3 个 skip 是可选依赖（lancedb / fastapi 缺一即跳），不要为了「全绿数字」把重依赖塞进主 dependencies
- `test_parallel_tool_dispatch` 暴露的一点：deepseek/glm 在 thinking 关闭时不注入 reasoning 是
  dispatch 矩阵的既定行为。**不要**把它改成无条件注入，先确认网关是否真的要求
- 下一轮建议接回上一条 changelog 的工具层方向做深一件（性能：跨轮重复读文件仍会重发全文——
  `tool_loop_guard` 只在「最近一条真实 user 之后」去重；或速度：`import llgraph.main` 冷启约 1.2s，
  其中 ~0.35s 花在 `langchain_anthropic`/`anthropic`，可评估懒加载）。现在测试是绿的，可以放心改
