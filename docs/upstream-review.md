# 上游跟进（2026-11-15）

本次检查 AIMixer 上游从 `4c6245e518dd8a494f3afc4f3cccd53f22d82d10` 到 `85c66f59d6b13bec71324271258fcae07a2a5898` 的四个提交，选择性适配 JiuChenTian 的 [31149e2](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/commit/31149e285f7556f77d5cd9786c289ab28cae8374) 与 [f0cb69c](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/commit/f0cb69cfb4fd1dfb07a10101935730d25e94bbd4)：

- 导演包构建移到工作线程；媒体免重复压缩，JSON 继续压缩。大包由新版页面使用浏览器原生下载，保留旧页面 POST ZIP 与小包 base64 路径；消耗式下载退出后清理临时包。
- 本地 batch 素材组增加“删除所选”；运行中或外接组控制时禁用，全选时保留第一组。保留 Pro 的 Pika、音频选项和按名称恢复参数逻辑，不改变节点输入输出或工作流格式。
- 暂不引入全局素材去重：现有组内 `genImage` / `startImage` 可能共用目标路径，直接叠加全局映射存在引用冲突风险；这是静态审查结论，未对上游完整复现。种子展示及逐段落盘已被上游后续提交撤回，按最终净变化不引入。

验证：Python 编译与 JavaScript 语法、真实 PNG 导演包导出/导入、Pro 参数保留、aiohttp 小包和 49 MiB 合成载荷下载（ZIP 哈希/CRC、清理、旧 POST 兼容）、工作线程执行，以及实际 JavaScript 函数在模拟 DOM/API/编辑器中的下载和批删边界检查通过。合成载荷不是可播放视频；未做实际浏览器下载落盘、GPU 生成回归或实际大时间线性能测量，不据此宣称全部功能或画质验收。

# 上游 PR／Issue 处理记录（2026-11-02）

上游基线：[4c6245e518dd](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/commit/4c6245e518dd8a494f3afc4f3cccd53f22d82d10)。已检查全部 **71 个 PR（20 个未关闭）及 205 个 Issue** 的正文、差异和讨论；原始API快照与哈希留在本机维护记录，不包含私人运行资料到本仓库。状态以本次快照为界。

“已整合”说明代码来源已进入维护版；不等同于每种硬件、所有长视频及画质均验收通过。源修复采用原生加载/推理机制，保留原作者历史；明显过时、冲突或改变生成语义的实现分别处理，不机械合并。

## 关键整合与边界

- 缓存：稳定身份槽、迁移占用唯一、选段不裁历史槽；LoRA文件/强度、成品视频来源与锁定音轨切片参与指纹。
- 导出：保留veryfast预设，短音轨补尾，不用`-shortest`截掉末帧；硬切不调亮度，内存与磁盘路径共用规则。
- 内存：批次转帧、磁盘映射及别名释放；不遍历其他进程清工作集，不在单条结束后无条件卸载所有模型。
- 原声：锁定生成是独立可选功能，默认关闭；干净音频latent随缓存持久化，不依赖8项进程缓存。
- 成品片段：声明成品视频和纯拼接时不隐式采样；缺素材或缓存写入失败明确报错，支持全部/分段导出。
- 分块：mask与PrefixRemask按tile裁剪，共享条件只裁一次，异常还原；不把失败回退全幅推理作为省显存方案。
- Pika：配方与任务分开，原提示词进入原生条件节点；内部阶段按原生执行器处理预取、转换缓冲及动态显存水位。

## 验证范围

Python编译与JS语法通过。真实torch的边界断言覆盖音频缓存重载、超过8段、分块条件恢复和身份槽冲突；CPU整合另验证124帧静音/短音轨完整解码、硬切逐像素一致、memmap引用释放、成品/纯拼接零采样、一采hold及新模式路由。部分媒体/模型边界使用stub，不能代替GPU画质与音画同步验收。真实浏览器验证通过系列切换、旧26项工作流迁移、按名称保存重开、旧Pika开关兼容、H3参数保留及自动/中/英语切换。性能结果另列，不将历史单点耗时作为速度保证。

## PR 逐项处置

| PR | 作者 | 上游状态 | 处置 |
| --- | --- | --- | --- |
| [#276](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/276) Fix prompt group wheel scrolling on canvas overlay | chrislingxiao | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#268](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/268) fix: preserve external Director writes and baseline handoff | qwert20020408-web | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#267](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/267) feat: 支持「现成视频当某一段」，并在空选择时直接拼接 | erbierbier | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#263](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/263) feat: 复用缓存续跑 + 修复首段一采缓存永不落盘 + prev加载diff诊断日志 | a437409255 | closed | 部分覆盖：保留基线的首段缓存、codec与诊断，吸收221/225的累计运行修复；不默认复用未确认的一采改变重抽语义，也不采祖先241/259的目录与裁尾改动。 |
| [#261](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/261) feat(examples): 新增 AMD/GGUF 本地适配工作流归档 example_workflows/AMD-7900xtx/ | happymy | closed | 不采用AMD专用示例归档：本轮硬件无法验证该GGUF/AMD组合；保留原PR来源，不将NVIDIA验收推广为AMD通过。 |
| [#259](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/259) 段缓存尾部裁剪：.av.pt/.pre.low.pt 只存续接尾部（省约50%磁盘）+ 合并 upstream | a437409255 | closed | 不采用latent只存尾部的缓存格式变更：可能丢失完整二采/重跑所需数据，当前保留完整latent及稳定身份槽。 |
| [#258](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/258) 修复原声和参考音频采样率被误标为 44100 Hz 导致变速、变调的问题 | RyouMon | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#254](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/254) 支持 2X 视频 VAE（打包 PixelShuffle 解码头） | tuanlai1234 | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#253](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/253) Freeze source-audio tokens into the AV latent audio stream (audioMode=source) for tighter lipsync | bananasss00 | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#251](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/251) feat: 段缓存目录支持自定义命名，避免多工作流共用节点 id 互相覆盖 | ljxdn | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#250](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/250) feat: 出片预览与导出统一取源（预览即产物），预览面板支持下载整片 | JiangXinYu113 | closed | 不采用该分支对正式导出/预览的整体重构；本轮明确区分成品与预览，使用一次正式导出验收，不据预览声称成品通过。 |
| [#245](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/245) latent 放大权重列表不再用 "3d" 子串误筛同类新权重 | tuanlai1234 | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#241](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/241) 新增延迟合并成片、项目管理与合并进度条（含内存水位保护） | a437409255 | open | 部分吸收：采用197/221/225解决拼接内存、累计缓存与身份问题，并补Map接线；不叠加该分支的项目目录、latent裁尾及清理体系，避免两套缓存身份并存。 |
| [#231](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/231) fix: make Director cache status inspection media-lightweight | NeverLookBackAgain | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#225](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/225) feat: 段缓存改按身份槽寻址，任意位置插入段不再整体失效 | ljxdn | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#221](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/221) Fix Director memory cleanup and partial-run stability | cGtv0SH | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#218](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/218) fix: 分段导出 +「选择运行」时输出只剩 1 帧占位 | reatang | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#216](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/216) perf: 段缓存帧改无损 FFV1 存储，1080p 段 1452MB 降到 451MB | ljxdn | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#215](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/215) fix: fl2v 仅首帧的组被误锁尾帧，导致段间引导下一段回到首帧 | reatang | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#214](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/214) fix: fl2v「选择运行」复选框缺失，及「引用上段」被重新勾选 | reatang | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#212](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/212) Keep aspect ratio (follow picture) output size | Kristijan1001 | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#211](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/211) fix: 二采画布变化时重对齐 keyframe/cond 与 PackedLayout，修复 i2v 二采形状不匹配崩溃 | TheBestJie | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#207](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/207) fix: 段间补调 cleanup_prefetch_queues()，修复多段生成时第二段起上下文编码恶化/卡死（另含阶段精确卸载） | 34262315716 | closed | 已吸收阶段边界目标：内部原生节点结束时同步清预取/转换缓冲及动态水位；不引入强制全模型卸载。 |
| [#206](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/206) fix: spatial tile 同步 tile 化 5D denoise_mask | yangala | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#204](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/204) feat: 集成 T8 语义桥（Semantic Bridge） | Andrew2020Yuan | closed | 保留基线已有实现；旧分支不重复合入。维护者评论：语义桥已支持；不要重新引入旧T8分支。 |
| [#197](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/197) fix: 大时间轴改为磁盘流式拼接，避免四段并驻内存导致二次 OOM | 972426472 | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#187](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/187) fix: 先确认一采（hold）时不再写入最终缓存/最终mp4，避免「全部导出」误合并未二采画面 | randyjtw | closed | 已吸收核心修复：等待确认的一采不写最终缓存；phase-align重存与预览后缀依据前一段自身hold状态。 |
| [#182](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/182) perf: 段间引导低频对齐改批量 GPU 计算，1080p 段 427s 降到 0.37s | ljxdn | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#180](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/180) Feature/refine0913 | zhangpannb | open | 不采用强制卸载全部模型的旧实现：保留已有显式二采前清理选项，并采用原生阶段资源边界。 |
| [#175](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/175) fix: leave seam untouched for hard-cut segments (per-segment continuity off) | ahui58 | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#174](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/174) feat: 一采缓存检查识别外接组（接线指纹 + 逐段独立判定） | ljxdn | closed | 保留基线已有实现；旧分支不重复合入。维护者列出346e0ca/7390adb/5a8e380并明确不要Merge旧main；核对新槽位/LoRA见证，不重复合包。 |
| [#171](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/171) feat: 内存与磁盘开关、提示词库、每组存视频、完成后重跑 / RAM & disk toggles, prompt library, per-group video saving, re-run loop | Kristijan1001 | open | 部分吸收：跟随图片由212整合，原有缓存codec/分段保存继续保留；不引入跨进程内存清理。提示词库与自动循环运行暂不采用，避免改变队列与编辑状态语义。 |
| [#169](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/169) 在refine节点添加一采后清理VRAM、是否一采解码和保存一采视频的选项，可保证二采不爆显存 | youngbaby777 | closed | 保留基线已有实现；旧分支不重复合入。维护者只采二采前清VRAM；明确不采decode_first_pass/stage_first_pass_images与fp16放大，避免破坏对比片和re-pin。 |
| [#161](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/161) feat: 每个片段可单独挂 LoRA / per-segment LoRA stacks in the Director timeline | Kristijan1001 | closed | 保留基线已有实现；旧分支不重复合入。分段LoRA由63834b9落地且171作者正文确认；不重复合入旧分支。 |
| [#154](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/154) feat: 增加Refine节点分块支持(enable_chunking)，优化8G显存体验 | SuitThug3977 | closed | 保留基线已有实现；旧分支不重复合入。维护者声明仅取核心分块功能；上游h3_latent_upscale已有enable_chunking。不要复写旧算法，另核控件和指纹。 |
| [#152](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/152) fix(director): 引导+重绘叠加官方Guide关键帧，修复开启重绘后跨段断续 | luckday888 | closed | 不采用把软重绘与另一套Guide强行叠加的旧方案；保留当前独立guide/continue机制，质量变化需独立跨段验收。 |
| [#151](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/151) feat: 采样中新增「音频预览」，可逐段试听当前步生成音频 | luckday888 | closed | 不启用采样中音频预览：增加每步解码与传输，和单条性能目标相冲突；正式音轨仍正常输出。 |
| [#150](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/150) feat: 新增「选择导出」模式，连续勾选段合并成片、不连续段拆分输出 | luckday888 | closed | 不采用新的选择导出模式；本轮保留选择运行、正式分段及全部导出契约，避免混淆采样选择与导出选择。 |
| [#149](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/149) feat: 新增「音频接续」开关，关闭后跨段音频硬切 | luckday888 | closed | 不增加独立跨段音频开关；保留现有连续性与原声/静音控制，需画音组合场景再决定拆分语义。 |
| [#146](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/146) fix: preserve two-pass continuity across segments | cosmosno2 | open | 不重复合入：当前基线已分离一采/最终latent与SelfLift低清流；旧实现会重复引入并覆盖后续修复。 |
| [#145](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/145) fix: release UNet before first-pass preview decode | cosmosno2 | closed | 采用原生采样/解码阶段边界释放转换缓冲；不原样引入强制卸载全部UNet的分支。 |
| [#141](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/141) feat: port grade suite, luma-only drift compensation, cache resume & … | ahui58 | open | 部分吸收其流式导出与续跑问题目标，由197/221/225实现；不合入会删除当前约1.6万行功能的旧分支。调色/亮度默认处理暂不引入，需独立质量验收。 |
| [#137](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/137) fix(director): 修复自定义帧率未同步生成帧数 | MagicPwn | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#133](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/133) Feat/增加 MiniMax H3 AddGuide 时间轴支持 | babysorry | open | 本轮不采用：AddGuide是额外的时间轴任务与界面体系，超出Pika发布的五类任务；旧分支同时改写现有分组/缓存/语义桥，缺多Guide组合验收，不能作为普通修复合入。 |
| [#129](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/129) fix: 修复分段导出最终视频仅1帧及分段mp4写出失败 | luckday888 | closed | 保留基线已有实现；旧分支不重复合入。维护者明确说问题已修复；218/143为相关后续证据，仍核114对IMAGE契约的更广范围。 |
| [#128](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/128) fix: 修复参考音频音色丢失/错位（<Audio N> 标签与音频槽位严格对齐） | luckday888 | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#126](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/126) fix: 段间引导保留模型生成的完整尾部，修复句尾读音/收音被吞音 | luckday888 | closed | 保留基线已有实现；旧分支不重复合入。保留连续段完整尾部并改帧预算；main已有continuity_keep_tail，先比较与137/255帧数音频对齐，勿重复扩时。 |
| [#125](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/125) fix: 曝光锚定改用 gamma 端点钉死校正，修复提亮导致高光过曝 | luckday888 | closed | 不改变默认曝光算法：gamma端点钉死属于画质取舍，不能仅凭单场景报告替换当前接缝处理。 |
| [#120](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/120) 混合模式分段生成 + 逐组审核 + 分段缓存管理 | dabaicai001 | open | 部分覆盖：保留当前mixed、缓存检查、选择运行及最新释放逻辑；不叠加旧执行器与二采界面重写，逐组暂停审核暂不引入。 |
| [#118](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/118) feat: add a safe in-app updater | breadtitor | open | 已整合（保留原作者历史）；额外冲突与正确性修复见本页验证范围。 |
| [#113](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/113) fix: r2v/v2v/rv2v 报错 TypeError: unsupported operand type(s) for //: 'str' and 'int'（#111） | randyjtw | closed | 保留基线已有实现；旧分支不重复合入。维护者已合并；main提交74a8409将官方ReferenceToVideo改为关键字参数，旧参数位置补丁不重复。 |
| [#107](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/107) fix: 段间引导接入过期上游素材时不再悄悄发生，报告中明确提示 | randyjtw | closed | 部分覆盖：保留现有来源变更拒绝与缓存诊断；本轮补成品视频身份、LoRA及音频切片指纹，不覆盖新版状态检查为旧分支。 |
| [#105](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/105) fix: 重抽某段后二采报错、检测面板显示错误状态 | randyjtw | closed | 部分覆盖：保留当前逐段一采检查与confirm流程，本轮同步Pika/LoRA/原声锁定指纹；不以旧界面重写替代231的新请求去重。 |
| [#94](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/94) Add input and output media chooser | kenlcj | closed | 保留基线已有实现；旧分支不重复合入。维护者以87cd5e9实现输入缩略图懒加载；有意未采Output页签/路径，不能把输入输出全链路当已支持。 |
| [#93](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/93) fix: 「选择运行」重抽时 images_pre_refine 未勾选段优先用一采缓存填充 | randyjtw | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#92](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/92) r2v 原声选项 + SolAttn v5 兼容 + 音频归一化 | li771667481 | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#86](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/86) fix: 工作流重载后高级采样参数恢复为默认值 | hcwhan | closed | 保留基线已有实现；旧分支不重复合入。维护者以9007d9a替代：按命名恢复，80ms settle后才快照；原PR早期snapshot会固化错误默认值，不照搬。 |
| [#83](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/83) 性能优化：降低导演节点内存占用并刷新缓存状态 | cb4y | closed | 保留基线已有实现；旧分支不重复合入。维护者以a2fea4f重放音频PCM/懒加载/工作集/状态文字四块并保92原声；不合dirty fork。 |
| [#82](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/82) 修复和优化几个小bug | luckday888 | closed | 保留基线已有实现；旧分支不重复合入。维护者59a718a取chip剪贴板与时长输入；明确未取r2v旁白前缀和gitignore改变，勿原样合入。 |
| [#81](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/81) fix: 同步 R2V 删除按钮状态与术语 | breadtitor | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#74](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/74) feat: 导演包导入导出（时间轴 JSON + 素材 zip） | smallops | closed | 保留基线已有实现；旧分支不重复合入。维护者9e6b4fb squash且修Windows空zip、路由范围/TTL；旧head仍有空zip问题，不合原分支。 |
| [#72](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/72) 改进导演台素材编辑与一采确认流程 | cb4y | closed | 保留基线已有实现；旧分支不重复合入。作者给三组等价patch ID映射，维护者确认main；3929193/8882c00/5de4b8b为核对入口。 |
| [#71](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/71) 尝试进行内存泄漏修复 | luckday888 | closed | 保留基线已有实现；旧分支不重复合入。维护者只取优化代码，因冲突不全合；剩余项需逐行比较，不能把全部内存问题标已解。 |
| [#63](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/63) 新增/修改功能说明 | a437409255 | closed | 不合入旧功能汇总包：已按对应原始PR逐项整合，避免覆盖新版分组、二采与缓存结构。 |
| [#59](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/59) Fix continuity compatibility with SolAttn | QQWWLOOK | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#56](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/56) fix: add drag-and-drop to r2v video/audio and i2v image slots | K3NK3 | closed | 保留基线已有实现；旧分支不重复合入。维护者01afc49替代；旧PR写死0–2槽会写错组，main按vidOffset/audOffset与上传辅助函数处理。 |
| [#49](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/49) feat: 为 MiniMaxH3Director 增加已上传媒体选择与预览，并统一“选择/上传”交互文案 | brodra | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#44](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/44) feat: 支持调整 R2V 提示词框高度 | breadtitor | merged | 上游基线已合并，保留；不重复叠加旧PR。 |
| [#38](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/38) Sync fork features with upstream (split first/second pass, stream export, VRAM optimization, refine/export-only) | RealDeanZhao | closed | 不合入旧fork同步包：包含多项已覆盖或相互冲突功能，按各原始PR处理并保留归属。 |
| [#35](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/35) feat: 新增「流式导出」开关，长片导出不再 OOM 崩溃 (#32) | richfugui001-netizen | closed | 部分吸收目标：由197磁盘拼接与221分块转帧降低内存峰值，保留IMAGE接口；不引入另一套流式开关/执行器。 |
| [#14](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/14) fix(r2v): 按组过滤公共参考图并同步预览状态 | nlwy | closed | 不采用自动丢弃未被局部提示词点名的公共参考图：它会改变已有素材输入语义，Pika保留用户明确提供的有序参考。 |

## Issue 逐项处置

同一根因的重复报告保留独立来源，不因合入关联PR就关闭或宣称解决上游Issue。

| Issue | 分类 | 处置 |
| --- | --- | --- |
| [#277](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/277) [Bug] Director r2v process disappears silently between TE staged and UNET staged (ComfyUI 0.41.0, 5 runs / 4 fresh instances, no traceback) | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#275](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/275) 用SelfLift后，两段之间会闪屏（段间引导已开启） | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#274](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/274) 内存溢出，10个片段每个10秒，在最后合并时报错。日志如下 | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#273](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/273) 选中的分组在生成后会取消选中 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#272](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/272) SelfLift + 段间引导：多段时第 2 段起高清阶段失效，画面明显变糊 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#271](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/271) [功能请求] 在 H3 Director 中原生接入 RefMod，复用角色参考包 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#270](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/270) 【提个建议】不知道能不能跟MinimaxH3Mod这个主体制作插件联动 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#269](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/269) 添加参考音频后模型胡乱说话 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#266](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/266) 有计划在外接参数的节点支持外接公共参数吗 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#265](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/265) 希望增加全局音频平铺功能 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#264](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/264) 选择静音导致最后视频合成报错 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#262](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/262) 段间引导+引用上段时音频参考触发张量形状错误：RuntimeError: The size of tensor a (54) must match the size of tensor b (76) at non-singleton dimension 3 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#260](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/260) 能不能加入Veda加速节点 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#257](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/257) selflift单采，段间引导越来越模糊 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#256](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/256) Is Custom Sampler Input possible? (For use with MiniMax-H3 Turbo Sampler by Larryvrh) | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#255](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/255) Director 插件部分输出少一帧，修复建议 | 明确修复候选 | 已修复；真实FFmpeg/PyAV验证静音与短音轨均完整解码124帧。 |
| [#252](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/252) BUG：SelfLift采样开启分块+开启段间引导时会发生空间分块采样与 denoise_mask 形状不匹配异常（Spatial tiles × denoise_mask cond） | 明确修复候选 | 已修复分块mask、共享cond重复裁剪、PrefixRemask回灌及异常恢复；真实torch边界断言通过，RoPE/融合画质方案未引入。 |
| [#249](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/249) 【Bug】⭐⭐修复了 group combine bug⭐⭐ | 明确修复候选 | 已修复数值排序、Map接线读取与target_slot校正；语法/静态契约通过；模式与语言已做GUI验证，12组Combine重排仍未完整GUI复现。 |
| [#248](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/248) 希望加入分段controlnet功能 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#247](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/247) [Bug] SelfLift 的 3D lift 失败会静默降级为 bilinear 插值，与 Refine 的硬失败行为不一致 | 明确修复候选 | 已修复：3D放大失败明确抛错，不再静默双线性回退；GPU故障场景待验。 |
| [#246](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/246) ref模型提示词编写建议 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#244](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/244) 使用自定义sigma后，视频无法预览 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#243](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/243) Getting out of ,memory while creating the video | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#242](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/242) 有没有人遇到过风扇狂转但是突然无DP信号输入的情况？只能强制重启解决 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#240](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/240) FaceRefine：修脸结果未写入导出视频（输出与无 FaceRefine 基线逐像素一致）；export_pre_face_refine + comfy-aimdo 0.5.3 下修脸采样进程崩溃 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#239](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/239) Replace reference image still showing original image. | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#238](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/238) 【建议】关于分镜一键克隆 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#237](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/237) 公共参数图片最多只能为9张 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#236](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/236) 【BUG】开启段间引导会报错 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#235](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/235) 导入导演台报错提示“crypto.randomUUID is not a function”(已提交BUG修改PR，新增方法createPackUploadId() ) | 明确修复候选 | 已修复四处上传ID生成；原生UUID、getRandomValues、无crypto三条路径内联通过。 |
| [#234](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/234) fl2v 多段视频拼接时，总会出现镜头放大，推进 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#233](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/233) 使用原声只能引用一段音频！ | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#232](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/232) 使用SelfLift的时候多段不起作用 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#230](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/230) Bug: latent upscaler runs on CPU on Intel XPU systems (device detection only checks CUDA) | 明确修复候选 | 已改用ComfyUI设备抽象；未在XPU硬件验收。 |
| [#229](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/229) 导演台的 rv2v 的分段没有公共参数且 v2v 的分段没有片段提示词输入（仅有全局） | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#228](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/228) 参考 生视频卡住不动 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#227](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/227) 【建議】face refine 可否能設定，超過某比率的臉就不需修 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#226](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/226) upscale-h3_latent，ValueError: fl2v plan builder received task_key=r2v | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#224](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/224) 显存占用异常 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#223](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/223) 是否支持多卡，模型切分推理 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#222](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/222) don't see selflift after git pull and restart | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#220](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/220) 全部导出报错，建议大佬改为流式导出模式，而非一次性加载，而且这个问题已经反复出现，一直未修复。 | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#219](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/219) [Bug] Refine enable_tiling 在分段连续性(rv2v)第 2 段起必然失败：denoise_mask 未按 tile 裁剪（out[0]*denoise_mask 形状不匹配） | 明确修复候选 | 已修复分块mask、共享cond重复裁剪、PrefixRemask回灌及异常恢复；真实torch边界断言通过，RoPE/融合画质方案未引入。 |
| [#217](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/217) SELFLIFT 似乎每一段视频后也需要清理VRAM | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#213](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/213) [强烈建议]加入这个语义桥，提高提示词遵从度 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#210](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/210) 二采会大幅降低音频质量，甚至导致爆音 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#209](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/209) MiniMax H3 Director Refine i2v二采放大问题 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#208](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/208) 多段生成时段 1 正常而段 2 起上下文编码恶化至卡死（已定位根因，修复见 PR #207） | 明确修复候选 | 已补与原生一致的内部阶段资源边界；正在做同条件性能验证，多段原报告环境未逐一复现。 |
| [#205](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/205) 想问下可以把现在外接的节点整合到导演台内部吗？现在时间感觉随着外接的节点越多，又变成普通的工作流模式了 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#203](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/203) 请问可以把vae解码这块独立出来吗？ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#202](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/202) 希望加入audio lock功能，方便做数字人 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#200](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/200) 【建议】face refine目前只能单人修脸，是否可以加入这个多人修脸功能 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#199](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/199) 二采完成后再次运行，仍会重新采样（一采缓存文件已写入但似乎未复用） | 明确修复候选 | 保留明确的一采确认/二采续跑入口；不默认把普通重抽改成缓存命中。 |
| [#198](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/198) [Bug] 外接组同步会清空每段的 continuityFromPrev（「引用上段」自动复原 + 二采缓存连带失效） | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#196](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/196) 【p0】全部导出模式报内存OOM | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#195](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/195) 每段生成完毕后，点击生成好的片段下方的预览，播放没反应 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#194](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/194) Feature Request: Allow per-scene LoRA/model loading in addition to the base model (ref2v) model | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#193](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/193) [Bug] fl2v（首尾帧生视频）模式下单镜头仅传首帧时，尾帧被错误赋予首帧导致强制首尾同图（单组与多组均受影响） | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#192](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/192) 参考图片生成视频时，口型跟声音对不上，看起来音频延迟了几秒的样子 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#191](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/191) SelfLift实测这个真不错~~坐等大佬适配 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#190](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/190) 关于设置二采latent放大模型的precision参数 | 明确修复候选 | 不把fp32默认改成fp16；保留精度稳定性，需特定硬件画质验收。 |
| [#189](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/189) 参考音频生成的时候音频有时候有问题 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#188](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/188) 希望增加一次性导入分段txt和一次性设置时长的功能 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#186](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/186) 希望在混合模式下，嵌套公共参数功能，这样的话不同分段还可引用不同的公共参数 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#185](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/185) 选择分段导出只有最后一个视频导出，选择只运行一个镜头时得到的只是一帧的视频。 | 明确修复候选 | 已整合218/221选段输出与累计缓存修复；本轮source/assemble分段路径CPU通过，长段累计需实机。 |
| [#184](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/184) 建议增加缓存功能 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#183](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/183) 内核升级至v0.35.0以上时，导演台节点会运行崩溃，报错RuntimeError: aimdo memory compile error | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#181](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/181) Refine 二采空间分块导致 shape mismatch（AV 联合采样音频 token 对不上） | 明确修复候选 | 已修复分块mask、共享cond重复裁剪、PrefixRemask回灌及异常恢复；真实torch边界断言通过，RoPE/融合画质方案未引入。 |
| [#179](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/179) 感觉问题越来越多了，现在视频分段都会提示保存失败，完全不可用状态，只会生成第一段，只有多有视频生成完都看不到，音画对不上 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#178](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/178) 上传音频原声模式，不同步生成的视频，，什么时候解决啊？ | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#177](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/177) 希望在多段生成的时候，每段结算的时候不要跳来跳去 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#176](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/176) 生成多段视频最后拼接时爆内存不足not enough memory | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#173](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/173) r2v参考生成分段会导致分段部分衔接不上 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#172](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/172) 加入音频参考使用原声对口型的时候，导出的视频口型没有对上。其他工作流同提示词可以对上 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#170](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/170) [建议]期待能在二采节点上加入优先提示词选项。 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#168](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/168) 单独导出片段报错 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#167](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/167) **缺少单段克隆功能**：目前只能手动点击“添加一组/切分”，然后再逐项复制粘贴提示词等参数，分段较多时操作繁琐。 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#166](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/166) 混合模式中可否增加组拷贝功能，相同的参考图生视频，每次都载入参考图太繁琐。 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#165](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/165) [建议] 可参考 MiniMaxRefDirector-ComfyUI 补齐：主体/素材资产层、提示词模板可编辑、导演向快捷控件 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#164](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/164) 插件全部改版了, 界面乱七八糟的, 比原来的难用多了, | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#163](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/163) 多个分段随机种不一致导致无法二采 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#162](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/162) 開啟段接引導+重繪會狂跳記憶體洩漏~只能強制關閉 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#160](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/160) 选择运行部分分镜时，Segment continuity异常 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#159](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/159) 在runninghub上使用插件报错 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#158](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/158) 我尝试用VDN加速，好像采样总是进度不了，几个小时不动，不知道什么原因？ | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#157](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/157) Your social media bilibili account cannot be followed or commented on. What's going on? | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#156](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/156) 多分段二采识别出错/段间随机种不一致 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#155](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/155) 内存溢出的异常 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#153](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/153) 大佬又有加速新方案了SelfLift，爱屋大佬也测了效果特别不错，速度提升幅度很高而且质量降低很少，你可以看看能不能兼容 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#148](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/148) 最新版本OOM(commit id:7de4a95) | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#147](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/147) 作者还有什么新功能要准备更新的哇 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#144](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/144) 开了段间引导，二采过渡点明显 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#143](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/143) [Bug] 分段导出时 CreateVideo → SaveVideo 输出 0 秒视频，但 seg_XXXX.mp4 正常 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#142](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/142) [Feature] MiniMaxH3Director 增加 latent / latent_pre_refine 输出 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#140](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/140) 提个小建议, 不要新增功能了,先确保现在 所有的功能都是正确的 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#139](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/139) 二采加速没有实际起效，成品全是噪点 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#138](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/138) 分段导出报错 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#136](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/136) 修改帧率参数不生效，生成总帧数仍是24*duration不变,只是生成的视频播放速率修改为了目标帧率，导致播放时长变成了2*duration | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#135](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/135) 分段导出，一共三段只导出了最后一段 警告信息为Segment #1 mp4 export failed (generation continues): [Errno 22] Invalid argument | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#134](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/134) 能否加强一下导演台提示词这一块 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#132](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/132) 超多分段视频comfyui参考生视频最后阶段进程直接中断并强制退出comfyui | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#131](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/131) 想问下一采有没有单段重抽？ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#130](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/130) 在导入以前的导演包后，在这基础上编辑其他提示词会出现之前的视频和新生成的视频错乱 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#127](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/127) 即使安裝 h3 motion context，director也能正常接續 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#124](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/124) 分段导出，连续运行多段 采样速度直线下降 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#123](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/123) 功能建议：公共提示词 增加 公共后缀，每个素材 添加 “使用前缀”，“使用后缀” 单独勾选开关，可以独立控制是否使用 前缀或者后缀提示词。 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#122](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/122) Refine upscale 二采 shape mismatch | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#121](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/121) 音频下拉框选择原声，生成视频没有声音 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#119](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/119) rv2v模式参考素材生视频，上传源视频，素材上传一张图片，图像人物替换视频中人物，运行报类型转换错误； | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#117](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/117) t2v文生视频模式全部导出内存溢出 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#116](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/116) 能否在二采增加ComfyUI Minimax H3 Latent Upscaler这个节点？ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#115](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/115) 是否考虑增加多卡模式 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#114](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/114) 分段导出释放像素后，images 输出静默降级为 1 帧，API 调用方无法察觉 | 明确修复候选 | 保留分段释放契约与已有218过滤；完整成品以正式MP4核查，不声称所有释放后的IMAGE槽完整。 |
| [#112](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/112) 批量提交任务排队执行段间缓存问题 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#111](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/111) [bug]  TypeError: unsupported operand type(s) for //: 'str' and 'int'] | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#110](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/110) 新版本的图生视频，二次采样失败 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#109](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/109) 请问下这个是什么问啊 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#108](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/108) This plugin has serious issues,  aside from the prompt generation function, nothing else works | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#106](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/106) 修改了一下导演台兼容图图电影感工作流采样 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#104](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/104) 我不太確定有沒有這個功能，視頻延長 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#103](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/103) 关键帧参考模式 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#102](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/102) 确实，一开始说话还正常，后面说话就乱七八糟 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#101](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/101) Add OrcaRouter as an optional AI provider | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#100](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/100) 按下 ctrl+z 编辑窗口就会消失，有人遇到过一样的问题吗？ | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#99](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/99) 请求大佬增加ComfyUI-H3-FaceRefine的支持，将面部修复做成可独立开关的后处理模块（类似二采），与一采/二采自由组合 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#98](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/98) 二采 MMH3 分割放大技术 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#97](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/97) 二采高清放大后画面模糊 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#96](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/96) I2V upscale refine fails because first-frame conditioning is not rebuilt for target canvas | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#95](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/95) 请求增加对ComfyUI-H3VAE_TRT的支持 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#91](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/91) 添加素材，选择已有，弹窗文件选择器优化建议 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#90](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/90) 您好~提示詞建議功能~ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#89](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/89) 能不能增加一个音频总轨道平铺的功能？ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#88](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/88) 崩溃位置：`concat_continuous_chunks` → `_additive_opening_luma` | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#87](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/87) 开启段间引导后，长对话内容总是说不完整 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#85](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/85) 刷新comfyui界面后，之前设置好的参数直接还原到默认值 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#84](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/84) 没一次能运行成功，什么问题？ | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#80](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/80) 二采结束后内存错误 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#79](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/79) [Feature/Bug] 素材组生成视频预览不持久化，保存工作流重开后丢失 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#78](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/78) No description provided. | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#77](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/77) 作者大大，我有个疑问啊！希望你解答一下！ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#76](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/76) 最新版本，选采样其中一组视频后，会出现死机现象 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#75](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/75) CPU 内存峰值过高触发了 PyTorch/C++ 原生崩溃 | 明确修复候选 | 已整合221分批转帧，避免整段uint8/float瞬时副本；大分辨率峰值待测。 |
| [#73](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/73) 参考音频问题 | 明确修复候选 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#70](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/70) 大佬感觉可以把一采的调度跟采样也能自由拆分出来，毕竟minimaxh3现在好多玩法都是在那换调度或者采样sigma啥的像二采那样单独的就好了。 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#69](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/69) 大佬，为啥我选哪个标签都是FL2VA模式 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#68](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/68) 竖屏二采失效，一采0.5，二采试了1.0，1.5，2.0都不行 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#67](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/67) 多段loop的时候 可以加一下可指定不同lora的功能吗？ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#66](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/66) 任务类型：r2v，多个素材组 引用公共音频失效问题可以修复下吗？ | 明确修复候选 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#65](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/65) lib/audio_io.py: _load_full_audio 将采样率错误覆盖为 44100Hz，导致非 44.1kHz 音色参考变调 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#64](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/64) 可以添加个自定义一采sigma吗？ | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#62](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/62) 能在r2v里增加lipSync以解决H3原生语音质量不佳的问题吗？ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#61](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/61) 20步跑完后，合并成视频时报异常了，不知道哪个阶段出问题了 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#60](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/60) 长视频生成——分段导出后合并 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#58](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/58) 生成長視頻用段間引導後面會越來越劣化 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#57](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/57) 每个分镜能不能增加保存功能。 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#55](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/55) Bug提交：二采分辨率选择跟随导演台会导致画面扭曲和模糊 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#54](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/54) 之前用可以，更新出錯 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#53](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/53) 靠，牛逼，好用。内存的使用优化下就完美了。 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#52](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/52) 功能需求：段落收起功能 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#51](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/51) 使用参考视频转视频，视频转视频时，使用段间引导或全部导出时会一直使用旧的素材无法刷新，重新上传素材也是一样。 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#50](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/50) [Bug] fl2v 段 + 引用上段(pin←prev) 必现报错: cannot mix unmarked stock keyframes with director context anchors when references shift the target origin | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#48](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/48) 功能小建议: 二采流程先合成输出一采视频, 检查是否抽卡失败 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#47](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/47) 如果是用I2V二採或放大latent都會失敗 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#46](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/46) 工具很好用，希望能优化一下内存占用 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#45](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/45) Feature Request：R2V 分段生成的实时下游输出（Incremental / Streaming Segment Output） | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#43](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/43) 建议增加一个单纯制作refine用的工作流 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#42](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/42) Is it possible to rerun certain parts in the settings? | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#41](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/41) 【需求】支持单次 I2V / FL2V Sampling 内的多时间点 Guide / Intermediate Keyframes | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#40](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/40) 是否能設定一個開關 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#39](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/39) rv2v在编辑几次视频和长提示词后，提示词和参考控制力下降 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#37](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/37) 功能需求：增加公共参数输入点 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#36](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/36) 添加收起展开素材组的功能 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#34](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/34) Could you add a feature for a custom sigma value? That would resolve the pixelation issue. | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#33](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/33) 功能建议：「选择运行」支持全不选 = 纯导出模式（从缓存导出，不重新采样） | 明确修复候选 | 已修复空选择纯拼接与缺缓存硬失败；CPU整合验证两种导出模式零采样。 |
| [#32](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/32) 多段长片「全部导出」阶段 OOM，进程被内核 SIGKILL（cleanup_segment_vram 卸载模型未生效） | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#31](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/31) 图生视频分段,无法接其他分段 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#30](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/30) 官方ExtendIntermediateSigmas节点修复噪点劣化 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#29](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/29) Segment Continuity Frames does not work in text to video | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#28](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/28) cpu莫名2000%占用，多参导演台无法播放视频 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#27](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/27) 需要數字人對口型功能及音效修復，感謝大大 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#26](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/26) 首尾帧fl2v + 段间引导 报错修复办法 | 明确修复候选 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#25](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/25) 相同提示词参考图参数时重新生成新视频的操作问题。 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#24](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/24) 不知道是bug还是需要优化？内存占用严重 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#23](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/23) 這個工具很好用，但目前碰到的問題 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#22](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/22) 希望添加资源快速插入功能 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#21](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/21) 请问导演台可否增加"积极情感(positive)"信号的输出？ | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#20](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/20) 分段輸出時，seg_xxxx.mp4   之類的影片都完成了，但UI畫面還會卡住很久 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#19](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/19) runninghub上跑两个片段报错Director continuity: another pack already patched MiniMax H3 PackedLayout.__init__. Disable the other pack and restart ComfyUI. | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#18](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/18) ComfyUI-DaSiWa-Nodes 导致页面假死。 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#17](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/17) 图生视频模式下，无法拖动上传源图 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#16](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/16) 关于避免”内存不足异常“使得8G显存32G内存和其他能成功跑单段视频的配置就可以无限时长生成视频的优化想法 | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#15](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/15) 生成时间被限定了，怎么放开 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#13](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/13) 一些优化建议 | 明确修复候选 | 部分修复：磁盘拼接、分块写帧与引用释放已整合，小张量内存/磁盘硬切及映射生命周期验证通过；不据此宣称所有长片或下游编码OOM已解决。 |
| [#12](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/12) v2v-視訊轉視訊，分段匯出發生錯誤 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#11](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/11) 希望可以增加一些功能 | 功能建议 | 功能建议已审查，本轮不扩展；关联PR按上述决定处理。 |
| [#10](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/10) 段間引導時會與ComfyUI-SolAttn_triton起衝突 | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#9](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/9) bug | 需复现 | 待复现：保留原报告与环境信息；没有足够证据认定已解决。 |
| [#8](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/8) 选中已经有的提示词再粘贴新的提示词进去，本来会被覆盖掉的提示词还是存在，必须先把原来的删掉再粘贴 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#7](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/7) 11日新版本的主node的窗口大小无法保存，拖动修改大小后，点击运行窗口大小会还原 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#6](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/6) r2v的提示词，复制进去的内容在修改后，点了运行会还原成修改前的内容，随机触发 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#5](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/5) 外部 Group 时间线 UI 无法透传 ComfyUI Reroute 节点 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#4](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/4) 遇到一个问题，使用导演台生成6段视频，如果最后一段生成失败了，前5段视频也会丢失 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#3](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/3) 请问什么时候能支持lora哈 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#2](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/2) 提示词过长时好像会内存溢出 | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
| [#1](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/1) English Global version please!? | 已覆盖候选 | 基线已有对应修复，保留；原报告环境未逐一复现。 |
