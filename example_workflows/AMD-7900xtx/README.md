# MiniMax H3 Director — AMD / GGUF 本地适配版（RX 7900 XTX）

本目录是 [ComfyUI_MiniMaxH3_Director](../../README.md) 的 **AMD Radeon（RX 7900 XTX 24GB / gfx1100 / ROCm）本地适配产物**。
与仓库 `example_workflows/` 下的官方模板**并存**，官方文件未做任何改动（本目录为独立的归档目录）。

## 与官方模板的差异

| 项 | 官方模板 | 本目录（`*_gguf.json`） |
|---|---|---|
| UNET | `UNETLoader` + `*_int8_convrot.safetensors` | `UnetLoaderGGUF` + `*_Q4_K_M.gguf` |
| 文本编码器 | `CLIPLoader`（`qwen3vl_32b_minimax_h3_nvfp4_awq`，type `minimax`） | `CCTechClipProjLoader`（`qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn` + `mmh3-8b-ClipProj-v3.1`，type `boogu`） |
| SageAttention | 加速版 / external_groups_* / 二采_加速 内含 Sage 节点 | 同位置节点 **已旁路**（`mode=4`） |
| 其他（VAE / Director / 连线 / 采样参数） | — | 原样保留 |

## 工作流清单（11 个）

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
| `minimax_h3_director_二采_加速_gguf.json` | 两阶段二采（367 帧） | fl2va + ref2va | 旁路 ×4 | 2 个 turbo LoRA + latent upscaler（**本机缺**） |

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

`minimax_h3_director_二采_加速_gguf.json` 另需以下 3 个文件（**本机当前缺失，须先下载放入**，否则 Queue 报错）：

- `models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`
- `models/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`
- `models/latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors`

其余 10 个工作流引用的模型文件本机齐全。

## 平台差异（NVIDIA vs AMD）

官方模板面向 NVIDIA；本适配面向 AMD RDNA3（gfx1100）。差异与对应处置：

| 能力 | NVIDIA | AMD RDNA3（本机） | 对本适配的影响 |
|---|---|---|---|
| SageAttention（KJ 两个节点） | 支持（SM 75/80/86/89/90/120/121），降自注意力峰值显存 + 提速 | **无实现**（无 gfx 分支） | 必须旁路（长片 3 个 ×2 + 二采 ×4，共 10 个节点） |
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
- H3 生视频用带 `--use-ck-attention` 的启动脚本（`run_amd_gpu_enable_dynamic_vram.bat`，扩散模型自注意力默认走 comfy_kitchen INT8）；CK 是进程级全局且同时影响生图，**生图请换无 CK 的脚本**（`run_amd_gpu_no_ck_attention.bat`）。
- 单段 5s 采样在本机约 8.5–9 分钟（ROCm + Q4 GGUF，无 flash-attn）。

## 采样参数 / 输出

与官方模板一致，详见 [`../README.md`](../README.md)：默认 0.4MP 16:9（864×480）、5 秒 / 124 帧 @ 24fps、25 steps（`res_multistep` + `simple`，CFG 1.0），Sigma shift video 12 / audio 3。
