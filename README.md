# ComfyUI MiniMax H3 Director

基于 **ComfyUI 官方 MiniMax-H3** 的多段音视频导演台插件。仓库地址：[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)

**English** → [README_EN.md](README_EN.md)

![MiniMaxH3Director 工作流截图](docs/screenshot.png)

## 功能介绍

**MiniMaxH3Director** 是面向长视频、多段生成的 MiniMax H3 导演台节点，把分段计划、条件编码、采样解码和导出整合在一个节点里。底层走官方 `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` + `MiniMaxH3SigmaShift` + `KSampler` + AV 分离解码链路，原生输出立体声音频。

### 核心能力

| 功能 | 说明 |
|------|------|
| **多段时间轴** | 节点内上传视频，支持切分、均分、智能分镜分割（PySceneDetect）、追加；分割点可选中删除；可视化时间轴预览每段范围与缩略图 |
| **多任务模式** | `task_type`：`t2v`（文生视频）、`i2v`（图生视频）、`fl2v`（首尾帧生视频）、`r2v`（参考主体生视频 / 素材组）、`v2v`（视频转视频）、`rv2v`（参考素材改视频） |
| **首尾帧 (fl2v)** | 独立首尾帧时间轴：多组关键帧、「添加一组」上传首帧和/或尾帧（官方支持只传尾帧）；拖缘调时长；提示词写中间运动；支持「选择运行」只跑部分组 |
| **参考素材组 (r2v)** | fl2v 式分组 UI：上方「公共参数」共享参考图/音频与公共提示词（与每组提示词拼接）；每组可再挂图片1–9 / 音频1–3 / 视频1–3；提示词用 `<Picture N>` / `<Video K>` / `<Audio J>`（或 `@` 引用）；时间轴预览与选中状态同步 |
| **源视频编辑 (v2v / rv2v)** | Bernini 风格源视频时间轴；每段源画面自动绑定 `<Video 1>`；`rv2v` 另可挂参考图（图片1–9）与参考音频（音频1–3） |
| **选择运行** | 开启后只采样勾选的片段/素材组；未勾选段可用缓存或源画面填充（全部导出时） |
| **外部多组接线** | `Director Group (Image to Video)` / `(Reference to Video)` + `Groups Combine`；连入导演台 `i2v_groups` / `r2v_groups` 后外部优先覆盖 UI 素材，仍支持跑批与选择运行 |
| **原生立体声音频** | 与画面同次采样生成；`v2v`/`rv2v` 可选生成声音 / 使用原声 / 静音 |
| **段间引导** | 默认关闭；多段 `t2v` / `i2v` / `fl2v` / `r2v` / `v2v` / `rv2v` 时可开启，将上一段生成结果的末尾运动（及生成音频）钉入下一段采样再裁掉前缀。上下文帧数：5 / 22 / 39 / 56，**默认推荐为 22**。**感谢 [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) 提供的实现思路** |
| **二采 / 放大 (Refine)** | 外接 **MiniMax H3 Director Refine** 到导演台 `refine` 口。未接线 = 原来的单次采样。`refine` = 同分辨率精修；`upscale` = 先放大到目标画布再低 denoise 二采。`passes` 可多次精修（upscale 只放大一次）。可选接 `refine_model` 换二采 UNET。`images` 为二采后成片，`images_pre_refine` 为一采（放大前）画面 |
| **运行报告** | `report` 口输出分段计划、每段任务摘要 |

### 运行按钮：一采 / 二采 / 流式导出 / 正常导出

「性能」组提供 4 个可自由组合的运行按钮，取代旧的 `stream_export` / `rerun_mode`：

| 按钮 | 默认 | 说明 |
|------|------|------|
| **一采** | 开 | 正常首次采样（每段必跑或已有缓存）。不勾 = 复用上次缓存 latent |
| **二采** | 开 | 在一采基础上做第二次采样/放大精修。需连接 Refine 节点；不勾则即使接了 Refine 也不跑二采 |
| **流式导出** | 关 | 从分段缓存流式渲染成片并写 mp4（防 OOM），峰值内存恒定（≈6GB）；开启后 images/audio 输出为空，成片路径见 `video_path` |
| **正常导出** | 开 | 内存拼接输出 images/audio（连接 CreateVideo 使用）。两者都不勾时默认按正常导出处理 |

- **流式导出 / 正常导出互斥**：勾选其一自动取消另一个（后端另有兜底校验）；
- **一采 + 二采 + 导出** = 完整流程（默认）；只勾「二采」= 只跑二采；只勾导出（不勾一采/二采）= 只跑导出。

> **升级提示**：旧工作流里的 `stream_export` / `rerun_mode` 控件已移除，加载旧图后这两个值会被丢弃、
> 按上表默认值重置（流式导出默认关、重跑默认完整流程），请按需重新勾选运行按钮。

**流式导出（防 OOM）**：导出阶段**跳过全量拼接**，从分段缓存**流式渲染成片并写 mp4 文件**——
解决长片「全部导出」全量拼接（约 3× 帧体积）叠加模型驻留导致的内核 OOM 击杀（issue #32）。流式导出步骤本身
峰值内存约 2 段（≈6GB）、不随总时长增长；整体峰值仍包含采样期累积的各段帧与模型权重，长片上限
由这两者决定（本改动消除的是导出阶段的最大尖峰）。逐边界修复 + 逐段写入 ffmpeg + 逐段释放，成片
已含音轨（视频+音频），路径见 `video_path`。**用法**：勾「流式导出」→ 正常排队（无需连接 CreateVideo）→
成片在 `output/video/MiniMaxH3_Director_stream_<时间戳>.mp4`。

> **流式导出 × 导出模式**：「全部导出」= 整片合并成一个 mp4；**「分段导出」+ 流式导出 = 每段渲染成
> 独立 mp4**（`output/minimax_seg_export_stream/<时间戳>/seg_XXXX.mp4`，峰值内存约 1 段，同样低内存）——
> 此时跳过采样期增量导出，统一由缓存逐段渲染。也正因如此，「只跑导出」+「分段导出」可勾「流式导出」
> 从缓存直接补出各段 mp4（此前分段导出只跑导出会报错）。

**只跑二采**：已完整跑过一遍（分段缓存已生成）后，若只想对已生成结果做二采/放大精修，**不勾「一采」、
勾「二采」**，配合「选择运行」勾选要精修的组，再连上 Refine 节点排队：

- 勾选段**跳过一采采样**，直接复用磁盘缓存的一采 latent 跑二采，省掉最耗时的一采（省约一采/（一采+二采）的时间）；
- 一采是确定性的：若一采参数不变，缓存 latent 与你现有画面一致，二采在其上精修，内容不会跑飞；
- 未勾选段仍走缓存填充，「全部导出」照常拼片；
- **前提**：需先完整跑过一次（缓存含 `seg_XXXX.av.pt`），且必须连接 Refine 节点（只跑二采时未接 Refine 会报错）；
- 缓存 latent 是上次的最终结果：若上次已开二采，这次再只跑二采等于叠加一次二采（相当于 passes+1），
  如需“只精修一次”，请先以完整流程（不接 Refine）跑一遍，再只勾「二采」。
- 与「流式导出」天然搭配：二采后缓存被覆盖，流式导出直接产出精修后的成片。

**只跑导出**：已完整跑过一遍后，**不勾「一采」「二采」，只勾导出**（流式或正常），即不采样、不二采，
仅从分段缓存重新渲染成片——适合调整导出参数（如换「流式导出」开关、改导出模式）后直接出片，零采样耗时。
未接 Refine 时用此模式即可；接了 Refine 也不会跑二采。

> **只跑导出不加载模型**：`model / video_vae / audio_vae / clip` 四口已改为可选。只跑导出时
> **断开这四个口的加载节点**（UNETLoader / VAELoader / CLIPLoader），ComfyUI 不会执行它们，
> 不再把大模型读进内存，出片几乎零等待。若四个口保持连接，模型仍会被加载（ComfyUI 为求输出会执行加载节点）。
> 在完整流程 / 只跑二采模式下仍必须连接模型，缺失会报错。

### 输入 / 输出

**输入：** `model` → `video_vae` → `audio_vae` → `clip`  
**可选：** `i2v_groups`（Image to Video 多组）/ `r2v_groups`（Reference to Video 多组）/ `refine`（`MiniMax H3 Director Refine`）

**输出：** `images` → `audio` → `fps` → `frame_count` → `source_images` → `report` → `images_pre_refine`

> CLIP Loader 的 **type 必须选 `minimax`**（Qwen3-VL）。  
> `t2v` / `i2v` / `fl2v` 用 **fl2va** UNET；`r2v` / `v2v` / `rv2v` 用 **ref2va** UNET。

## 依赖

请将 **ComfyUI** 升级到 **v0.30.0** 及以上（含官方 MiniMax H3 节点：[PR #15224](https://github.com/comfyanonymous/ComfyUI/pull/15224)、[PR #15228](https://github.com/comfyanonymous/ComfyUI/pull/15228)）。

可选：`scenedetect`（智能分割）、`opencv-python-headless`（源视频解码）、`imageio-ffmpeg`（原声抽取）——见 `requirements.txt`。  
Refine 的 `nvidia_rtx_vsr` 另需 NVIDIA GPU，可 `pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com`（不是硬依赖）。

## 安装

### 方法一：手动安装（标准方式）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git

pip install -r ComfyUI_MiniMaxH3_Director/requirements.txt
```

重启 ComfyUI。

### 方法二：ComfyUI Manager

1. 打开 **ComfyUI Manager**
2. 选择 **Install via Git URL**
3. 填入 `https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git` 并安装
4. 重启 ComfyUI

## 模型与工作流下载

完整资源包（**MiniMax H3 模型权重** + **示例 JSON 工作流**）见：

**[Comfyit 搅拌站 · 文章 506：MiniMax H3 模型和工作流](https://comfyit.cn/article/506)**

下载后将 `models/` 合并到 `ComfyUI/models/`，JSON 工作流拖入 ComfyUI 即可。

也可参考：

- **Hugging Face：** [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- **ComfyUI 文档：** [MiniMax H3 工作流示例](https://docs.comfy.org/zh/tutorials/video/minimax/minimax-h3)

本仓库自带示例：`example_workflows/`

| 工作流 | task_type | UNET | 说明 |
|--------|-----------|------|------|
| `minimax_h3_director_t2v.json` | t2v | fl2va | 文生音视频 |
| `minimax_h3_director_fl2v.json` | fl2v | fl2va | 首尾帧（「添加一组」） |
| `minimax_h3_director_r2v.json` | r2v | **ref2va** | 参考改视频素材组 |
| `minimax_h3_director_v2v.json` | v2v | **ref2va** | 源视频时间轴编辑 |
| `minimax_h3_director_rv2v.json` | rv2v | **ref2va** | 源视频 + 参考图/音频 |
| `minimax_h3_director_external_groups_i2v.json` | fl2v | fl2va | 外部 Group×2 → Combine → `i2v_groups` |
| `minimax_h3_director_external_groups_r2v.json` | r2v | **ref2va** | 外部 Group×N → Combine → `r2v_groups` |
| `minimax_h3_director_refine.json` | r2v | **ref2va** | 外接 Refine 二采；`images` 与 `images_pre_refine` 各出一路成片 |

### 推荐模型文件

| 用途 | 文件名 | 目录 |
|------|--------|------|
| UNET (t2v / i2v / fl2v) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| UNET (r2v / v2v / rv2v) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

## 快速开始

1. 确认 ComfyUI ≥ **0.30.0**，已能加载官方 MiniMax H3 节点
2. 从 [文章 506](https://comfyit.cn/article/506) 或本仓库 `example_workflows/` 加载示例
3. 连接 UNET / CLIP / video_vae / audio_vae，在导演台 UI 内编辑时间轴与提示词后 Queue

**视频教程：** [B 站合集 · 插件使用教程](https://space.bilibili.com/1997403556/lists/8357740)

### 默认采样参数

- 画布默认 **0.4MP 16:9（864×480）**，**5 秒 / 124** 帧 @ **24 fps**（17k+5 网格）
- **25** steps，`res_multistep` + `simple`，CFG **1.0**
- Sigma shift：video **12** / audio **3**

### 首尾帧 fl2v 用法摘要

1. 任务类型选 **「首尾帧生视频 (fl2v)」**
2. 点击「添加一组」，上传首帧和/或尾帧（可只传尾帧）
3. 在镜卡片或时间轴上调整时长；提示词写中间运动 / 镜头 / 过渡
4. Queue 生成；多组可勾选「选择运行」只跑部分组

### 参考主体 r2v 用法摘要

1. 任务类型选 **「参考主体生视频 (r2v)」**（需 **ref2va** UNET + audio_vae）
2. 点击 **「启用公共参数」** 展开面板（默认折叠/关闭）；上传共用参考图/音频并写公共提示词（如角色锁定 / `subject_definitions`）；启用后会与每组提示词拼接
3. 点击「添加素材组」；每组写分镜提示词，可按需再挂本组独有素材（同槽位覆盖公共素材）
4. 提示词中用 `<Picture N>` / `<Video K>` / `<Audio J>`，或输入 `@`（启用公共参数时可引用公共 + 本组素材）
5. 时间轴可预览各组时长与缩略图；「选择运行」与素材组勾选同步

### 源视频 v2v / rv2v 用法摘要

1. 选 **v2v** 或 **rv2v**，上传源视频并分段（切分 / 均分 / 智能分割）
2. 每段写提示词；系统自动将源片段绑定为 `<Video 1>`
3. `rv2v` 可额外上传参考图 / 参考音频；声音模式可选生成 / 原声 / 静音

### 二采 / 放大 Refine 用法摘要

1. 添加 **MiniMax H3 Director Refine**，把 `refine` 接到导演台 `refine` 口。不接则仍是原来的一采
2. `mode=refine`：同分辨率再采一遍（精修）。`mode=upscale`：先放大到目标画布再二采；分辨率控件仅在 `upscale` 时显示（可跟随导演台、按比例+百万像素，或自定义宽高）
3. `passes`：精修次数，默认 1、最多 9999。`upscale` 只在第 1 次放大，后面都是同分辨率精修
4. 可选接 `refine_model`（二采 UNET）；不接则用导演台主模型。适合一采挂 Turbo LoRA、二采卸掉或换另一套
5. 导演台 `images` 是二采后成片；`images_pre_refine` 是一采、放大前的画面，便于对比。`source_images` 仍是时间轴原片，不是一采结果
6. `denoise` 常用 0.2–0.35，越大改动越大。`steps=0` 时约为一采步数的 40%
7. fl2v 默认跳过二采（保护钉死的首尾帧）；关掉 Refine 上的 `skip_fl2v` 才会采
8. `upscale` 的放大方式：`lanczos`（可另接 `UPSCALE_MODEL`，如 RealESRGAN）；`nvidia_rtx_vsr` 用 RTX Video Super Resolution（不接放大模型）
9. 「分段导出」且 `passes>1` 时，每轮会另落 `seg_XXXX_pN.mp4`；「全部导出」只出一采和终稿

示例：`example_workflows/minimax_h3_director_refine.json`

### 外部多组接线（第三方节点接入）

对齐官网两个 conditioning 节点，把扩写 / 抠图 / Load Video 等处理结果以**多组**形式送进导演台：

1. 添加 **`MiniMax H3 Director Group (Image to Video)`** 或 **`(Reference to Video)`**
2. 按组接线：`prompt` / `duration_sec`；I2V 系接 `first_frame` / `last_frame`（无帧=t2v，仅首=i2v，仅尾或首+尾=fl2v）；R2V 为 Autogrow（同官方 Reference to Video）：接图/视频/音频会自动多出空口（图≤9、视频≤3、音频≤3）。输出宽高在**导演台**统一设置
3. 多组：用 **`Director Groups Combine`**（Autogrow：接满最后一个口会自动多出新口，同官方 Reference to Video）→ 导演台 `i2v_groups` / `r2v_groups`；单组可直接把 `group` 连到导演台
4. 导演台 `task_type` 与口一致（t2v/i2v/fl2v ↔ `i2v_groups`；r2v ↔ `r2v_groups`）；**不要两口同时连接**
5. 连接后执行以图中接线为准（外部优先）；UI 卡片变淡，仍可用「选择运行」按组序勾选

## 配套生态 · [Comfyit 搅拌站](https://comfyit.cn/)

[Comfyit](https://comfyit.cn/) 提供环境、模型、工作流与教程配套：

| 栏目 | 链接 |
|------|------|
| 模型 / 工作流包 | [comfyit.cn/article/506](https://comfyit.cn/article/506) |
| 官方 MiniMax H3 文档 | [docs.comfy.org · MiniMax H3](https://docs.comfy.org/zh/tutorials/video/minimax/minimax-h3) |
| 插件视频教程 | [B 站合集](https://space.bilibili.com/1997403556/lists/8357740) |
| 产品中心 | [comfyit.cn/products](https://comfyit.cn/products) |
| 插件广场 | [comfyit.cn/plugins](https://comfyit.cn/plugins) |
| 模型广场 | [comfyit.cn/resources/models](https://comfyit.cn/resources/models) |
| 工作流广场 | [comfyit.cn/workflows](https://comfyit.cn/workflows) |

## 作者与交流

| | |
|---|---|
| **维护者** | [AI搅拌手 / AIMixer](https://github.com/AIMixer) |
| **本仓库** | [github.com/AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) |
| **姊妹插件** | [ComfyUI_Bernini_Director](https://github.com/AIMixer/ComfyUI_Bernini_Director) |
| **作者 QQ** | **3697688140** |
| **B 站** | [space.bilibili.com/1997403556](https://space.bilibili.com/1997403556) |
| **插件教程** | [B 站合集 · 使用教程](https://space.bilibili.com/1997403556/lists/8357740) |
| **QQ 交流群** | **551482703** · **425064221** · **559826331** |
| **Comfyit 搅拌站** | [comfyit.cn](https://comfyit.cn/) |

## 致谢

- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — 官方 MiniMax H3 支持
- [MiniMax-AI](https://github.com/MiniMax-AI) — MiniMax H3 模型
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) — 权重与文档
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) — 段间运动/音频续拍思路参考

## 许可证

Apache-2.0
