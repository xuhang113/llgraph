# 2026-09-05 补上自动开发 changelog 留痕

选题：工作流（对齐折芯海）。此前定时轮只写 git log / 总结，下一轮接不住方向。

## 做了什么

- 新增本目录与 `AGENTS.md`、`.cursor/rules/cloud-agent.mdc`
- 定时工作流改为：开工读最近 3 条 changelog，收工必写一条；没有高价值切口只写下一步
- 下面是 auto_upgrade 上已有、但还没写成 changelog 的方向，供下一轮接着做，不要推倒重来

已有方向（从近提交归纳，本轮未再改这些代码）：

- 工具入参纠偏、路径唯一解析、会话内 todo_write
- 大文件 read 折叠、shell 超时保输出、grep 过宽折叠
- 出站 tool 链 recency 压缩、search_replace 容错、重复工具拦截

## 改了哪些路径

- AGENTS.md
- .cursor/rules/cloud-agent.mdc
- changelog/
- .github/workflows/main.yml

## 怎么验收

- 下一轮 Cloud Agent 先读本条和 AGENTS.md，再选一件北星下手
- 收工仓库里必须多一条新的 changelog，不能只改代码

## 未做 / 下一步不要做

- 下一轮建议：在速度 / 性能 / 稳定性 / 商用体验里，接上面已有工具层方向做深一件
- 不要为了留痕去大范围格式化或重写文档
