# r2v 参考音频原声输出

r2v 的「用原声」原先被后端静默回退为 `generate`，前端也隐藏声音下拉，输出永远是模型生成的声音。

## 根因

| 层 | 问题 |
|---|---|
| 后端 | `VIDEO_EDIT_AUDIO_TASKS` 只认 `v2v` / `rv2v`，`resolve_audio_mode()` 把 r2v 的 `source` 强制改成 `generate` |
| 前端 | 声音下拉只对 v2v / rv2v 显示 |
| 抽取 | r2v 没有源视频时间轴，即便开了 `source` 也抽不到音 |
| 格式 | 参考音频采样率不一致时，mux 后按错误采样率播放（16kHz 当 44.1kHz → 加速约 2.75 倍） |

## 修复

- r2v 加入 source 白名单，前端显示声音下拉
- `source` 且抽不到源视频音时，用该段第一条可用参考音频 mux
- mux 前归一化到 44.1kHz 立体声（失败打 warning，按原始格式 mux）
- 全部导出：每段用各自参考音频按时间轴拼接；该段没挂参考音频则为静音（不回退其它段）

## 双通道

| 模式 | 输出音频 | 参考音频还干什么 |
|---|---|---|
| `generate`（默认，未改） | 模型生成（AV latent 解码） | 条件，驱动口型 |
| `source` | 参考音频 wav 原样 mux | 仍作条件，画面贴原声 |

```
ref_audios → MiniMaxH3ReferenceToVideo._encode_ref_audio
           → AudioVAE(32kHz) → AV latent → 驱动画面口型/节律
```

## 使用

- **用原声**：输出 = 该段参考音频；prompt 的 audio 标签写 `fully_copy`；台词须与音频一致，否则口型错位
- **生成声音**：输出 = 模型生成（可改词）；prompt 写 `reference`。此路径与合入前一致
- 参考音频格式随意（自动归一化）；每段挂各自的。没挂的段在 source 下为静音
