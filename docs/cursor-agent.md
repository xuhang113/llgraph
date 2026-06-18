# 基于 Cursor 的 Agent 实现

llgraph 是 **Cursor Agent 工作流在终端侧的独立实现**：复用相近的产品概念与使用习惯，但运行在 **CLI + LangGraph + 自建 AI Gateway**，不依赖 Cursor IDE。

适合场景：已在 Cursor 里习惯 Agent 排查/改代码，希望在 **终端、脚本、无 IDE 环境** 下用同一套心智模型操作 monorepo 工作区。

---

## 1. 定位对比

| 维度 | Cursor Agent | llgraph |
|------|----------------|---------|
| 运行形态 | IDE 内嵌（Chat / Agent 面板） | 终端 `llgraph` 交互或 `--once` |
| 模型接入 | Cursor 托管 / 自选模型 | Anthropic 兼容 API（`LLGRAPH_*`） |
| Agent 引擎 | Cursor 专有编排 | LangGraph `create_react_agent`（ReAct 循环） |
| 工作区配置 | `.cursor/`、`.cursorrules` | `.llgraph/`（**独立目录，互不读取**） |
| 代码补全 Tab | ✅ | ❌（仅 Agent，无补全） |
| 图形化 diff / Apply | ✅ | ❌（用 `/diff`、git diff） |

**结论**：llgraph 不是 Cursor 插件，而是 **「Cursor 式 Agent」的 CLI 复刻 + 自建网关适配**；配置与运行时完全分离，避免两套系统绑死。

---

## 2. 能力对照（Cursor → llgraph）

以下能力按 Cursor Agent 常见用法对齐实现（详见各模块）：

| Cursor 概念 / 能力 | llgraph 实现 | 配置 / 命令 |
|-------------------|--------------|-------------|
| **Rules**（项目规则） | `.llgraph/rules/*.mdc`，`alwaysApply` / `globs` 注入 system | `/rule` |
| **Skills**（按需技能） | `.llgraph/skills/<name>/SKILL.md` | `/skill <name>` |
| **User Rules / 思考规范** | `.llgraph/thought/*.md` + `agent.json` | 自动注入 |
| **@codebase / 语义搜代码** | LanceDB 向量索引 + `search_code_hybrid` | `llgraph index`、`/index` |
| **Grep / 读文件** | `grep_files`、`read_file`、`search_workspace` | 内置工具 |
| **局部改代码**（Apply / Edit） | `search_replace`（优先于整文件 `write_file`） | `llgraph -w` |
| **会话内「改了哪些文件」** | `SessionEditTracker` | `/changes`、`/diff` |
| **MCP 工具** | stdio MCP 客户端，工具名 `mcp__<server>__<tool>` | `.llgraph/mcp.json` |
| **自定义 Slash 命令** | `.llgraph/commands/*.md`（frontmatter + prompt） | `/commands` |
| **长上下文 / 压缩** | `/compress` + `compress_strategy: auto`（滚动 anchor + 按 token 自动扩展出站 user 轮） | `agent.json` → `context`；`/context` 查看 |
| **续写 / 重写连续性** | `<workspace-context>` pin（已改路径、近期 read、上轮摘要） | 自动（`context_continuity.py`） |
| **动态上下文发现** | 大工具结果落盘 + 指针，按需 `read_file` | P6，`/trace stats` |
| **代码评审** | `/review`，落盘 `~/llgraph-review/` | `agent.json` → `review` |
| **保存后索引更新** | `watchdog` debounce 增量索引 | 随 Agent 启动，`--no-watch-index` 关闭 |
| **过程展示**（工具链可见） | `/trace all\|steps\|reply\|none` | 类似 IDE 里展开工具调用 |

参考 Cursor 官方博客：[Dynamic context discovery](https://cursor.com/cn/blog/dynamic-context-discovery) — llgraph P6（工具结果落盘 + 指针）与 P3（历史压缩归档）对应该思路的 CLI 侧落地。

---

## 3. 架构差异（简图）

```
Cursor IDE Agent                    llgraph CLI Agent
─────────────────                   ─────────────────
用户 ↔ Cursor UI                    用户 ↔ 终端 (llgraph)
        │                                   │
        ▼                                   ▼
  Cursor 编排层                      LangGraph ReAct
        │                                   │
   ├─ 内置工具                          ├─ filesystem_tools
   ├─ MCP (.cursor/mcp.json)           ├─ code_index_tools
   ├─ 索引 / @codebase                 ├─ MCP (.llgraph/mcp.json)
   └─ 模型 API                         └─ ChatAnthropic → LLGRAPH_API_BASE_URL
```

相同点：**ReAct 循环**（规划 → 调工具 → 再规划 → 回复）、**工作区沙箱**、**Rules/Skills 增强 system prompt**。

不同点：llgraph 把 **索引、watch、落盘、评审** 做成可配置的 `.llgraph/` 子系统；不接入 Cursor 的 Tab、Composer UI、Cloud Agent。

---

## 4. 配置为何独立（`.llgraph/` vs `.cursor/`）

| 原则 | 说明 |
|------|------|
| **不自动同步** | llgraph **不读取** `.cursorrules`、`.cursor/mcp.json`、Cursor Rules |
| **避免双写耦合** | 两套产品各自演进；MCP、规则需在 `.llgraph/` **单独维护** |
| **可对照迁移** | `llgraph --init-config` 提供模板；可把 Cursor 规则**手动**抄到 `.llgraph/rules/` |
| **同仓并行** | 同一 monorepo 可同时存在 `.cursor/`（IDE）与 `.llgraph/`（CLI），互不影响 |

若团队希望「规则一致」，应通过 **文档或共享 markdown** 约定内容，而不是让 llgraph 隐式读 Cursor 配置。

---

## 5. 典型迁移：从 Cursor 到 llgraph

1. **工作区根目录**  
   `llgraph -C /path/to/your/workspace`（与 Cursor 打开同一仓库即可）。

2. **初始化**  
   `llgraph --init-config -C .` → 生成 `.llgraph/`。

3. **规则**  
   将 `.cursor/rules/*.mdc` 中关键规则复制到 `.llgraph/rules/`（按需删减 IDE 专用项）。

4. **索引**  
   `llgraph index -C .` → Agent 才能 `search_code_hybrid`（类似 @codebase）。

5. **MCP**  
   在 `.llgraph/mcp.json` 的 `servers` 中**自行填写**与 Cursor 等价的 MCP（命令、args、env）；格式见 [操作手册.md](操作手册.md)。

6. **改代码**  
   Cursor 里 Agent Apply → llgraph 里 `llgraph -w` + `search_replace` + `/changes`。

7. **评审**  
   Cursor 侧可用 `claude-cli-delegate` → llgraph 侧用 `/review`（内置 Gateway LLM，落盘 `~/llgraph-review/`）。

---

## 6. 未实现 / 不计划对齐的 Cursor 能力

- IDE 内联补全（Tab）
- 多文件 Composer UI、可视化 patch 应用
- Cursor Cloud Agents、Background Agent
- 自动读取 Cursor `terminals/*.txt`（P6 V2 可选；当前仅 llgraph 自身工具输出落盘）
- Cursor Hooks（`.cursor/hooks`）— 需自行在 shell 侧集成

**已实现**（与旧版文档表述不同）：网关支持时，出站可对稳定 system / tools / 对话尾打 `cache_control` 断点（见 [会话上下文与历史.md](会话上下文与历史.md) §10）。

---

## 7. 相关文档

- [操作手册.md](操作手册.md) — 日常命令与配置
- [模块说明.md](模块说明.md) — 源码模块
- [README.md](../README.md) — 项目概览
