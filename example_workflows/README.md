# MiniMax H3 Director — 示例工作流

拖入 ComfyUI 画布即可使用。需已安装本插件，且 ComfyUI 主干含 MiniMax H3（v0.30.0+）。

| 文件 | 任务 | UNET | 说明 |
|------|------|------|------|
| `minimax_h3_director_t2v.json` | t2v | fl2va | 文生音视频，可直接 Queue |
| `minimax_h3_director_fl2v.json` | fl2v | fl2va | 首尾帧；「添加一组」后上传首帧和/或尾帧（可只传尾帧） |
| `minimax_h3_director_r2v.json` | r2v | **ref2va** | 参考改视频；素材组：图片1–9 / 音频1–3 / 视频1–3 |
| `minimax_h3_director_v2v.json` | v2v | **ref2va** | 源视频编辑；导演台上传视频并分段（同 Bernini v2v） |
| `minimax_h3_director_rv2v.json` | rv2v | **ref2va** | 参考改视频；源视频 + 图片1–9 |
| `minimax_h3_director_external_groups_i2v.json` | fl2v | fl2va | 外部 Group（Image to Video）→ Combine → Director.`i2v_groups`；时长/素材以接线为准 |
| `minimax_h3_director_external_groups_r2v.json` | r2v | **ref2va** | 外部 Group（Reference to Video）→ Combine → Director.`r2v_groups`；可用「选择运行」勾选组序 |
| `minimax_h3_director_二采_加速.json` | r2v | **ref2va** | 外接 **MiniMax H3 Director Refine** → Director.`refine`（SIGMAS + H3 latent）。`images` 为二采后成片，`images_pre_refine` 为一采对比片 |

## 模型路径（与官方模板一致）

| 用途 | 文件名 | 目录 |
|------|--------|------|
| UNET (t2v/i2v/fl2v) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| UNET (r2v / v2v / rv2v) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

CLIP Loader 的 **type 必须选 `minimax`**。

## 默认采样参数

- 画布默认 **0.4MP 16:9（864×480）**，**5 秒 / 124** 帧 @ **24 fps**（17k+5 网格）
- **25** steps，`res_multistep` + `simple`，CFG **1.0**
- Sigma shift：video **12** / audio **3**

## 输出

导演台 → `CreateVideo` → `SaveVideo`（前缀 `video/MiniMaxH3_Director_*`），报告接 `PreviewAny`。

Refine 示例另把 `images_pre_refine` 接到第二路 `CreateVideo` / `SaveVideo`，方便和二采后成片对比。未接 Refine 时该口与 `images` 相同。

## Refine 二采

- 二采一律按 SIGMAS：把 `BasicScheduler` 或 `ManualSigmas` 接到 Refine 的 `sigmas` 口。`BasicScheduler` 请接和二采相同的 MODEL
- 不接 Refine 节点 = 原来的单次采样
- `mode=refine`：同分辨率精修；`mode=upscale`：放大到目标画布再二采；`mode=latent_upscale`：只放大 H3 latent
- 导演台分辨率是一采；Refine 画布（比例+百万像素 / 自定义）是放大目标
- `passes`：精修次数（默认 1）；`upscale` 只在第 1 次放大；`latent_upscale` 不二采
- 可选接 `refine_model`（二采 UNET）；不接则用导演台主模型
- `upscale` 默认 `h3_latent`：在 Refine 节点 `upscale_method` 下方下拉选 3D 权重（`mode=latent_upscale` 时同样出现）。权重放 `ComfyUI/models/latent_upscale_models/`。`lanczos` 可另接 `upscale_model`（RealESRGAN 等），不接则纯插值；也可改 `nvidia_rtx_vsr`
- fl2v 默认跳过二采；关掉 `skip_fl2v` 才会采首尾帧镜头

## 本地 GGUF / AMD 适配版（`*_gguf.json`，10 个）

面向 AMD Radeon（RX 7900 XTX 24GB，gfx1100，ROCm）的本地适配产物，与上面的官方模板**并存**（官方文件未改动）。差异：

| 项 | 官方模板 | `*_gguf.json` |
|---|---|---|
| UNET | `UNETLoader` + `*_int8_convrot.safetensors` | `UnetLoaderGGUF` + `*_Q4_K_M.gguf` |
| 文本编码器 | `CLIPLoader`（`qwen3vl_32b_..._nvfp4_awq`，type `minimax`） | `CCTechClipProjLoader`（`qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn` + `mmh3-8b-ClipProj-v3.1`，type `boogu`） |
| SageAttention | 加速版 / external_groups_* / 二采_加速 含 Sage 节点 | 同位置节点 **已旁路**（mode=4） |

对应 10 个文件：`t2v`、`i2v`、`r2v`、`multiseg_t2v`、`fl2v`、`v2v`、`rv2v`、`加速版`、`external_groups_i2v`、`external_groups_r2v`，命名均为 `<原名>_gguf.json`。`二采_加速` 未适配（24GB 易 OOM）。

本地模型路径：

| 用途 | 文件名 | 目录 |
|------|--------|------|
| UNET (fl2va) | `minimax_h3_fl2va_pruned-Q4_K_M.gguf` | `models/diffusion_models/` |
| UNET (ref2va) | `minimax_h3_ref2va_pruned-Q4_K_M.gguf` | `models/diffusion_models/` |
| 文本编码器 | `qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors` | `models/text_encoders/` |
| ClipProj | `mmh3-8b-ClipProj-v3.1.safetensors` | `models/text_encoders/` |
| Video / Audio VAE | 同官方 | `models/vae/` |

依赖与注意：

- 必装 **ComfyUI-GGUF-Loader**（提供 `UnetLoaderGGUF` 与 `CCTechClipProjLoader`）。
- `external_groups_r2v` 需 **ComfyUI-VideoHelperSuite**（`VHS_LoadVideo`）。
- **SageAttention 节点一律旁路**：本机未装 sageattention；且其内核只支持 NVIDIA SM 架构（sm75/80/86/89/90/120/121），AMD 无实现，装了也无法运行。该节点在此类长片里的作用是**降低自注意力峰值显存**（非提速），旁路后请靠 `clear_vram_between_segments` + 减帧/降分辨率/减参考组防 OOM。
- H3 生视频用带 `--use-ck-attention` 的启动脚本（扩散模型自注意力默认走 comfy_kitchen INT8）；CK 是进程级全局且同时影响生图，**生图请换无 CK 的脚本**。
- 单段 5s 采样在本机约 8.5–9 分钟（ROCm + Q4 GGUF，无 flash-attn）。
