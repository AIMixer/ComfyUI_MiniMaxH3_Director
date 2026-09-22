# 上游集成验收记录（2026-09-22）

## 合并基线

- 集成分支：`codex/merge-upstream-20260922`。
- 功能分支远端：`9c4ff0d`；上游主线：`576424fb252b2b4a538e792027f7db1a5ef376e4`。
- 原始本地状态：`8ebeb5e`，已保存为 `backup/addguide-before-upstream-20260922`。
- `feat/addguide-p1` 已快进到 `9c4ff0d`；集成分支用 merge 保留两边历史。

## 主线优先处理

完整合入上游 12 个提交，保留 FaceRefine、SelfLift、Semantic Bridge、Refine 独立种子和画布比例，以及外接组缓存、参考图尺寸、导演包提示词和参考视频真实时长修复。

唯一文本冲突位于时间轴 import，保留 AddGuide 与 external witness 两组依赖。

`director/executor_core.py` 处理两处语义兼容问题：

1. 先应用 Semantic Bridge，再保存 AddGuide 二采条件，最后向一采加入定时图音引导，防止二采漏掉语义增强。
2. 仅 AddGuide 二采使用独立条件快照；普通任务继续使用上游最终 positive，保留后续 motion context 修改，避免段间引导在二采丢失。

新增功能的专用实现与节点沿用上游代码；不因 AddGuide 修改其默认参数、输入输出或采样实现。

## 已完成验证

| 检查 | 结果与范围 |
| --- | --- |
| Python 完整回归 | 75 passed，5 subtests passed |
| 主线 CPU 行为对照 | 同一套 19 项测试在集成版、原版上游 576424f 均通过 |
| 主线前端行为对照 | 同一套 4 项测试在集成版、原版上游均通过 |
| 原有前端回归 | AddGuide 与外接 Group 两套均通过 |
| 语法检查 | Python 源码与 14 个前端 JS 模块通过 |
| 真实 ComfyUI 宿主检查 | 使用已有 torch 2.9.1+cu130 环境；11 个节点类型加载及 INPUT_TYPES 通过 |
| 官方 AddGuide + SelfLift | 真实宿主 AddGuide 接口、NestedTensor 与 SelfLift 空间缩放通过；使用小型合成张量和替代 VAE，验证帧序/原条件不变/音频保留 |

主线专项覆盖：Refine 种子与 SIGMAS/模型保留、放大画布比例；Semantic Bridge 条件张量改写、元数据与幂等；SelfLift 空间缩放、时间和音频保留；FaceRefine 配置；导包提示词保护；外接组时长与 witness 失效。

条件生命周期测试覆盖 Bridge 开关与普通任务/AddGuide，并验证普通任务最终 motion-context 条件仍传入二采。这是局部执行路径验证，不替代完整 executor GPU 测试。

宿主探测未启动 PromptServer，因此出现 HTTP routes 未注册提示属探测环境限制；HTTP/浏览器运行路径未据此宣称通过。即使设置 CPU 模式，宿主导入仍报告 CUDA 已初始化；探测没有加载完整 H3 模型或运行成片采样。

## 未验收项目

- 完整 GPU 一采/二采、SelfLift 放大、人脸检测及贴回画质。
- AddGuide 图音引导与 SelfLift、Refine、Semantic Bridge 联用的真实成片。
- 完整浏览器中的节点接线、队列与导演包交互。
- 本地默认 `models/semantic_bridge/` 未发现可用权重目录；没有下载权重或改动已有 ComfyUI 安装。

这些项目需在有可用 Semantic Bridge 权重的完整工作流中抽查，不能用当前单元/接口检查代替。

## 复现

```powershell
python -m pytest -q
node --test tests/test_upstream_features_ui.mjs
node tests/test_addguide_ui.mjs
node tests/test_external_groups_ui.mjs
```

将 `MINIMAX_UPSTREAM_SOURCE_ROOT`、`UPSTREAM_UI_SOURCE_ROOT` 分别指向干净的上游源码目录，可对照运行两个主线专项测试文件；普通全套测试不设置这两个变量。

## 回退与发布状态

集成结果仅提交到本地集成分支。`feat/addguide-p1` 保持在同步后的 `9c4ff0d`，远端未推送。

需要恢复同步前代码时可切换 `backup/addguide-before-upstream-20260922`；需要恢复同步后未集成代码时可切换 `feat/addguide-p1`。切换前先保存后续未提交改动，无需重写或删除历史。
