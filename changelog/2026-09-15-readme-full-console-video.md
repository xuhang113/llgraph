# 2026-09-15 README 介绍视频改成整屏 Console

选题：商用体验（首页视频不能只裁 Trace，要能看见完整控制台）。

## 做了什么

- 重录多轮 ReAct 介绍视频：侧栏 + 会话 + Trace 整屏，不再只截右侧 Trace
- 上传 GitHub 附件并替换 README 内嵌链接
- 源文件仍是 `docs/assets/react-intro.mp4`

## 改了哪些路径

- `README.md`
- `docs/assets/react-intro.mp4`

## 怎么验收

- GitHub README 播放器里能同时看到会话区和 Trace 步骤，而不是 Trace 单独一块

## 未做 / 下一步不要做

- 没有现场再跑一轮新 Agent；画面来自本仓库已有真实会话
- 不要再把 Trace 裁条当成首页主视频
