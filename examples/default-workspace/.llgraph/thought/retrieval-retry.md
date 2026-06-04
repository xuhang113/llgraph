---
description: 检索无结果时的扩词与重试（对齐 Cursor Agent 习惯）
enabled: true
priority: 100
---

## 检索无结果时

1. **先假设可能是检索词不对**，不要第一轮就告诉用户「仓库里没有」。
2. **扩词**：中英文、缩写、驼峰与 snake_case、常见笔误（例：graphify → llgraph、ll-graph、code index、`.llgraph`）。
3. **search_workspace**：`keywords` 一次给 **5～12 个**词；换 `path`（子服务目录、`.llgraph`、`docs`、`markdowns`）。
4. **grep_files**：用 `词A|词B|词C` 做 OR；精确符号/类名优先 grep。
5. **search_code_hybrid**：概念/「类似逻辑」且已索引时使用。
6. **read_file**：配置与 README（如 `embedding.json`、`manifest.json`）先读后总结。

至少采用 **两种不同的工具或参数组合** 仍无命中后，再说明「当前工作区未找到相关内容」，并列出已尝试的检索词。
