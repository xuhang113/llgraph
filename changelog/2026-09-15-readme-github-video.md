# 2026-09-15 README 用 GitHub 附件让视频能播

选题：商用体验（首页相对路径 `<video>` 被 GitHub 剥掉，访客看不到播放器）。

## 做了什么

- 把首页介绍视频改成 GitHub user-attachments 直链，README 里单独一行，走官方内嵌播放器
- 仓库里仍保留 `docs/assets/react-intro.mp4` 作为源文件

## 改了哪些路径

- `README.md`

## 怎么验收

- GitHub README 顶部出现可播的多轮 ReAct 视频，而不是空白

## 未做 / 下一步不要做

- 不要再写相对路径 `<video src="docs/assets/...">`，GitHub 会丢掉
