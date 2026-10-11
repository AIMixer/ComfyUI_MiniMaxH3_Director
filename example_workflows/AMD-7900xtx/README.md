# MiniMax H3 Director — AMD / GGUF 本地适配版（RX 7900 XTX）

本目录是 [ComfyUI_MiniMaxH3_Director](../../README.md) 的 **AMD Radeon（RX 7900 XTX 24GB / gfx1100 / ROCm）本地适配产物**。
与仓库 `example_workflows/` 下的官方模板**并存**，官方文件未做任何改动（本目录为独立的归档目录）。

> [!CAUTION]
> ## ⚠️ CK 使用警告（生图禁用）
>
> `--use-ck-attention` **只能用于 MiniMax H3 生视频，绝不能用于生图（Qwen-Image 2.1 等）**。
> 该 flag 会把扩散模型自注意力**全局无条件覆盖**为 comfy_kitchen INT8；在此后端下 Qwen-Image 2.1 只要文本流跨过
> **64 token** 即**输出严重损坏**（绿/紫伪影、纹理破碎、结构崩坏），**且不抛任何异常**，极易误判为模型/提示词问题。
> 这是 comfy-kitchen **0.2.36** 的回归（#207/#208 重写 masked attention），**PyPI 无修复版本**（0.2.37 不存在）。
> H3 生视频已双向确认安全（文本编码器 + DiT 主干），可继续用 CK。详见《CK 注意力回归问题调查报告》。
>
> | 用途 | 启动脚本 | CK |
> |---|---|---|
> | **生图**（Qwen-Image 2.1 等） | `run_amd_gpu_no_ck_attention.bat` | ❌ 必须关 |
> | **生视频**（MiniMax H3） | `run_amd_gpu_enable_dynamic_vram.bat` | ✅ 开 |
>
> 两个脚本**除 ck 开关外完全一致**。CK 是**进程级全局**开关，切换用途**必须先完全关掉旧实例再重启**。
> 启动日志自查：出现 `Using Comfy Kitchen attention` 才是 CK 生效；**不应**出现 `Using sage attention`。

> [!WARNING]
> ## ⚠️ 连续生成会越跑越慢（**不是 H3 特有**：生视频 / 生图 / TTS 生音频 **都有**）
>
> `--enable-dynamic-vram` 下，**残留物理页随任务累积**：同一条工作流**跑得越多越慢**，十几分钟的活能被拖成几小时。
> **经本机实测，这不是生视频独有的问题——生图（Qwen-Image 2.1）、TTS 生音频一样会逐次退化**，凡是连续出活都要防。
>
> **缓解办法：调 ComfyUI 的 `/free`**（卸载模型 + 清 Dynamic VRAM 残留）。
> - **任务之间**：多段脚本已在段间内置 `POST /free`；段外手动清理、以及**批跑期间的周期清理**用
>   [`plan/bat/free_vram`](https://github.com/happymy/MinimaxH3-7900xtx/tree/master/plan/bat/free_vram)（`free_vram.bat` / `python free_vram.py`；`--status` 只看状态、`--interval 60` 挂旁边每 60s 清一次、`--no-verify` 发完即走）。
> - **`/free` 是异步 flag 端点**：**HTTP 200 ≠ 已释放**，真正卸载由 prompt worker 在**当前任务结束后**执行；要确认释放，须看 `/system_stats` 的 `vram_free` 上升（工具默认会轮询验证）。
> - **生成中调用是安全的**（只挂 flag，不打断当前 prompt），但它**救不了「单个任务内部越跑越慢」**——那种只能**重启 ComfyUI**。

## 与官方模板的差异

| 项 | 官方模板 | 本目录（`*_gguf.json`） |
|---|---|---|
| UNET | `UNETLoader` + `*_int8_convrot.safetensors` | `UnetLoaderGGUF` + `*_Q4_K_M.gguf` |
| 文本编码器 | `CLIPLoader`（`qwen3vl_32b_minimax_h3_nvfp4_awq`，type `minimax`） | `CCTechClipProjLoader`（`qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn` + `mmh3-8b-ClipProj-v3.1`，type `boogu`） |
| SageAttention | 加速版 / external_groups_i2v / external_groups_r2v 内含 Sage 节点 | 同位置节点 **已旁路**（`mode=4`） |
| 其他（VAE / Director / 连线 / 采样参数） | — | 原样保留 |

## 工作流清单（10 个）

| 文件 | 任务 | UNET | Sage | 额外依赖 |
|---|---|---|---|---|
| `minimax_h3_director_t2v_gguf.json` | t2v | fl2va | — | — |
| `minimax_h3_director_i2v_gguf.json` | i2v | fl2va | — | — |
| `minimax_h3_director_fl2v_gguf.json` | fl2v | fl2va | — | — |
| `minimax_h3_director_multiseg_t2v_gguf.json` | 多段 t2v | fl2va | — | — |
| `minimax_h3_director_r2v_gguf.json` | r2v | ref2va | — | — |
| `minimax_h3_director_v2v_gguf.json` | v2v | ref2va | — | — |
| `minimax_h3_director_rv2v_gguf.json` | rv2v | ref2va | — | — |
| `minimax_h3_director_加速版_gguf.json` | 加速版（243 帧） | ref2va | 旁路 ×2 | — |
| `minimax_h3_director_external_groups_i2v_gguf.json` | external i2v（734 帧） | fl2va | 旁路 ×2 | — |
| `minimax_h3_director_external_groups_r2v_gguf.json` | external r2v（1210 帧） | ref2va | 旁路 ×2 | ComfyUI-VideoHelperSuite + **自备参考视频** |

命名规则 = 官方 `<原名>` + `_gguf.json`；逐一对应 `../` 下的同名官方文件。

## 本地模型路径

| 用途 | 文件名 | 目录 |
|---|---|---|
| UNET (fl2va) | `minimax_h3_fl2va_pruned-Q4_K_M.gguf` | `models/diffusion_models/` |
| UNET (ref2va) | `minimax_h3_ref2va_pruned-Q4_K_M.gguf` | `models/diffusion_models/` |
| 文本编码器 | `qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors` | `models/text_encoders/` |
| ClipProj | `mmh3-8b-ClipProj-v3.1.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

以上模型文件本机齐全。

## 已移除：二采_加速（显存不足）

`minimax_h3_director_二采_加速_gguf.json` 曾在本目录内，因**显存占用超出本机 24GB** 已移除，不再纳入归档。

它是两阶段管线（一采 `fl2va` + 二采 `ref2va` + 3D latent 放大），采样峰值显存需求在所有工作流里最高；而 AMD 用不了 SageAttention——该节点在 NVIDIA 上的作用正是**压低自注意力峰值显存**，旁路后失去这层保护，峰值进一步抬高。RX 7900 XTX 的 24GB 承载不了。

（它同时也是唯一需要额外 2 个 turbo LoRA + 1 个 latent upscaler 的工作流，那些文件本机也没有。）

## 平台差异（NVIDIA vs AMD）

官方模板面向 NVIDIA；本适配面向 AMD RDNA3（gfx1100）。差异与对应处置：

| 能力 | NVIDIA | AMD RDNA3（本机） | 对本适配的影响 |
|---|---|---|---|
| SageAttention（KJ 两个节点） | 支持（SM 75/80/86/89/90/120/121），降自注意力峰值显存 + 提速 | **无实现**（无 gfx 分支） | 必须旁路（加速版 / external_groups_i2v / external_groups_r2v，共 6 个节点） |
| FP8 运算（E4M3/E5M2） | Ada/Hopper/Blackwell 原生 | **不支持**（RDNA3 无 float8 计算，本机 `SUPPORT_FP8_OPS=False`） | fp8 权重能加载但按 bf16 计算 → 更慢更占显存 |
| NVFP4 量化 | Blackwell/Ada 专有 4-bit 浮点块量化 | **不可用** | 官方 `qwen3vl_32b_..._nvfp4_awq` 编码器换成本地 `heretic-8B` |
| INT8（`int8_convrot` / comfy_kitchen） | 支持 | **支持**（RDNA3 原生 int8） | `--use-ck-attention` 的 INT8 注意力在 AMD 可用 |
| FlashAttention | CUDA 支持 | **不支持** | 注意力无 flash 核，采样更慢 |
| 扩散权重格式 | 官方 `int8_convrot` / `w4a8_mixed` | 本地用 GGUF Q4_K_M（dequant → bf16） | 全部换 `UnetLoaderGGUF` |

一句话：AMD 上**能跑**的是「GGUF 权重 + bf16/INT8 计算 + CK INT8 注意力」；**不能跑**的是「SageAttention」「FP8 计算」「NVFP4 编码器」「FlashAttention」。

依据：ComfyUI-KJNodes 源码（Sage 内核仅 NVIDIA SM）；AMD ROCm《Data types and precision support》（RDNA3：int8 ✅、无 float8 计算）；ComfyUI issue #14777（gfx1100 INT8 ConvRot 可用）；本机实测。

## 依赖与注意

- 必装 **ComfyUI-GGUF-Loader**（提供 `UnetLoaderGGUF` 与 `CCTechClipProjLoader`）。
- `external_groups_r2v_gguf.json` 需 **ComfyUI-VideoHelperSuite**（`VHS_LoadVideo` 节点），并把参考视频换成自己的文件放进 `ComfyUI/input/`。
- **SageAttention 节点一律旁路**：本机未装 sageattention；其内核只支持 NVIDIA SM 架构，AMD 无实现，装了也无法运行。该节点在此类长片里的作用是**降低自注意力峰值显存**（非提速），旁路后请靠 `clear_vram_between_segments` + 减帧 / 降分辨率 / 减参考组防 OOM。
- CK 使用见**顶部警告**：生视频用 `run_amd_gpu_enable_dynamic_vram.bat`（含 `--use-ck-attention`）；**生图必须**用 `run_amd_gpu_no_ck_attention.bat`（CK 是进程级全局开关，切换须重启）。
- 单段 5s 采样在本机约 8.5–9 分钟（ROCm + Q4 GGUF，无 flash-attn）。

## 采样参数 / 输出

与官方模板一致，详见 [`../README.md`](../README.md)：默认 0.4MP 16:9（864×480）、5 秒 / 124 帧 @ 24fps、25 steps（`res_multistep` + `simple`，CFG 1.0），Sigma shift video 12 / audio 3。

## 推荐配套工具：生成中实时帧预览（`preview_watch`）

> 🔗 <https://github.com/happymy/MinimaxH3-7900xtx/tree/master/plan/bat/preview_watch>

**在本机跑 H3 长片时强烈推荐**。它 = 一个 ComfyUI 内核补丁 + 一个旁路监视脚本：

- **内核补丁**（`frame_preview_kernel.patch`，3 个文件 / 约 7KB）：采样中途用**工作流自己的 video VAE** 解码 latent，把**成片分辨率的真帧**（不是 34×60 的 `latent2rgb` 小图）落盘到 `%TEMP%\iw-preview-*`，并自动弹出资源管理器。触发点：`step 0` 首批，之后每 `total_steps // 4` 步一批，每批均匀取 6 帧。
- **监视脚本**（`watch_h3_preview.py`，零侵入）：轮询 `GET /queue` 与 `%TEMP%\iw-preview-*`，实时打印帧；可 `--collect <dir>` 归档、`--once` 只盯当前队列。

### 本机（RX 7900 XTX）实测结论

1. **大部分情况下，采样到 `step 5` 的预览图就已足够判断是否符合生成意图** —— 不必等 8–9 分钟出片，几十秒就能看出跑没跑偏、要不要中断重跑。
2. **附带效果：提前卸载 DiT，显著降低显存压力。** `step 0` 首批预览要做一次 video VAE decode，这次 decode 自带的 `free_memory` 会在采样最早期就把 DiT 卸到低水位、之后按需换回；在 `--enable-dynamic-vram` 下，这一步让峰值显存明显下降。
3. **由此让长片成为可能**：`0.4 Mpx / 20 step` 下约 **2 分钟生成 1 秒视频**，**20 秒长片**可以跑通，且**耗时仅为原来的 1/3**（不提前卸载时显存被顶爆，数据退到内存 / 页面文件走，那一段渲染会非常慢）。

### 使用要点

- 补丁是对着 **ComfyUI v0.39.2** 生成的，需 `git apply` 到**同版本**的官方源码树（版本不同会因行偏移失败，按补丁手工落 3 处改动亦可）。
- 脚本为 Windows 路径（`os.startfile` / 资源管理器 / `%TEMP%`）。
- 全局开关：`ComfyUI\user\iw-preview-flag.txt`（由 `watch_h3_preview.py --preview on/off` 写入）；置 `off` 时既不落帧、也不提前卸载 DiT。
- 内核逻辑与计算后端**无关**：CUDA / NVIDIA 同样适用（那边 DynamicVRAM 默认就是开启的）；本节数据为本机 7900 XTX 的实测结果。

## 来源与致谢

本目录（以及本 README）的**部分数据、代码与脚本来自
[`happymy/MinimaxH3-7900xtx`](https://github.com/happymy/MinimaxH3-7900xtx)**，借用 / 参考的部分包括：

| 内容 | 出处 |
|---|---|
| 显存清理工具 `/free`（`free_vram.py` / `free_vram.bat`） | [`plan/bat/free_vram`](https://github.com/happymy/MinimaxH3-7900xtx/tree/master/plan/bat/free_vram) |
| 生成中实时帧预览（内核补丁 + 监视脚本） | [`plan/bat/preview_watch`](https://github.com/happymy/MinimaxH3-7900xtx/tree/master/plan/bat/preview_watch) |
| 连续生成性能下降、`step 0` 预览提前卸载 DiT 等**实测数据与结论** | 该仓库 `plan/README.md`、[`plan/bat`](https://github.com/happymy/MinimaxH3-7900xtx/tree/master/plan/bat) |
| AMD / ROCm 平台适配经验（启动脚本、`--enable-dynamic-vram` 等） | 该仓库 [`README.md`](https://github.com/happymy/MinimaxH3-7900xtx/blob/master/README.md) |

本目录 `example_workflows/AMD-7900xtx/` 是在上述基础上做的 **GGUF / RX 7900 XTX 适配归档**；官方模板未做改动。
