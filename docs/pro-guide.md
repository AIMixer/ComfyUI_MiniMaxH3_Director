# MiniMax H3 导演台 Pro

本维护版基于 AIMixer，保留原节点内部标识和工作流端口。安装时只保留一份导演台插件；同时启用原版和 Pro 会造成注册冲突。替换安装前保存工作流与本地修改。

## 生成模式

顶部先选 **H3 / Pika-H3**，再选任务。H3 保留原功能；Pika 的 FL2VA 适配器用于纯文与首尾帧，Ref2VA 用于多模态参考。

| Pika 任务 | 使用方式 |
| --- | --- |
| 纯文 | 选文生视频，仅填写提示词 |
| 首帧 | 图生视频，或首尾帧模式仅填首帧 |
| 尾帧 | 首尾帧模式仅填尾帧 |
| 首尾帧 | 首尾帧模式填入两张图片 |
| 多模态参考 | 选参考生成，按顺序添加图片、视频和音频 |

Pika 固定 CFG1、lcm、六次生成前向、视频／音频时钟 12/3；噪声节点为 1000/750/500/250/100/40/0 经视频 shift12 转换。固定参数在面板给出摘要，原有 H3 参数保留，切回 H3 时继续使用。

[完整模型下载与安装](models.md) 是模型说明的统一入口，包含底模、文本编码器、双 VAE、两份 Pika LoRA 的逐文件下载链接和目录，以及官方教程组合与本机已验证社区组合的区别。

Pika 选项会在分段采样前自动加载匹配任务的 LoRA，强度固定 1；缺文件时会报错，插件不自动下载。不要在外部 LoRA Loader 重复加载同一适配器。原始 Pika FP32 适配器不能改名当作 Comfy LoRA；不同兼容文件名的显式选择规则见模型入口。

官方本地模式为 24 fps、480p／768p，时长请求 5–15 秒并按 17n+5 帧网格对齐；5 秒对应 124 帧。Ref2VA 至少需要一张图片或一段视频，最多 9 张图片、3 段视频、3 段独立音频，总数不超过 12；视频及独立音频各自合计不超过 15 秒，单条为 2–15 秒。视频自带音轨与该视频配对，不另占独立音频槽。

导演台保留的短样、自定义画布、混合时间线、续接、二采、SelfLift 和修脸属于本地调度或社区扩展，不据此承诺属于官方 Pika 发布能力。参考[官方说明](https://github.com/Pika-Labs/Pika-H3-Flex)。本仓库不提供托管 API 的 4K 或分轨服务。

[五类可执行 API 示例与调用说明](../example_workflows/pika_pro/README.md)。示例素材为占位文件，使用前换成自己的输入。

## 语言、声音与成品

- **自动／中文／English**：自动优先跟随 ComfyUI 的语言设置，再回退浏览器语言；手动选择持久化。语言切换不会改提示词或任务参数。
- **使用原声**决定导出音轨；独立的**锁定原声**决定是否将该音轨作为生成条件，默认不锁定。音频锁定的干净 latent 随缓存保存，不依赖进程内短期缓存。
- 声明现成视频的片段不进入采样。缺文件、解码或写缓存失败会报错；不会改为生成新片段。空选择进入纯拼接时，缺有效缓存同样报错。
- 分段成品、最终拼接、临时预览和分段缓存是不同对象。应检查正式 MP4 的帧数、音轨及完整解码。

## 兼容与维护

旧 `pika_flex=true` 工作流继续启用 Pika；显式切回 H3 会清除该旧开关。导演包携带生成系列、解码和声音条件选项。改变 LoRA 文件或强度会使相关缓存失效。

缓存采用稳定身份槽，片段排序变化不应读到邻居成品；音频锁定使用全局音轨时，缓存还会核对片段在音轨中的时间位置。一采等待确认的结果只写一采缓存，不充当二采成品。

SelfLift 3D 放大失败会明确报错，不静默改为双线性插值。大时间轴的磁盘合并遵守硬切语义，临时映射文件随最后一个持有者释放；不会按文件年龄删除仍可能使用的映射。

上游处理依据、未采用实现与待复现问题见[逐项审查记录](upstream-review.md)。性能和媒体验收以本维护版实际运行记录为准；历史 101.29 秒不构成当前硬件环境的速度保证。

## 现有 SelfLift 与脸修工作流

`MiniMaxH3DirectorAvatarSelfLift` 将导演台的一采交给已安装的 [selflift-Avatar](https://github.com/slmonker/selflift-Avatar)；需要其 `SelfLiftAvatarH3Sampler` 和 Euler 采样器。保留原节点标识与端口，缺依赖会明确报错。动态续写遮罩不适用于这条桥接路径，需使用段间引导。

`MiniMaxH3FaceRefineVideo` 对已有 IMAGE 视频帧复用脸修流程；原音轨由原生视频节点接回，不在该节点重新生成。`MiniMaxH3DirectorFaceRefine` 增加可选独立模型、参考图和提示词，支持按切镜跟踪、掉检帧保留与自适应强度；旧输入顺序不变。相关脸修实现参考 [Carasibana/ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)，保留其 [MIT 许可](../director/face_refine/UPSTREAM_LICENSE)。

The Avatar bridge retains `MiniMaxH3DirectorAvatarSelfLift` and requires selflift-Avatar's `SelfLiftAvatarH3Sampler` plus Euler. Use guide continuity rather than dynamic continuation masks. `MiniMaxH3FaceRefineVideo` refines existing IMAGE batches; reconnect the original audio through native video nodes. FaceRefine's optional independent model, reference image, prompt, cut tracking and strength controls preserve the existing input order. The adapted face code retains its MIT attribution.
