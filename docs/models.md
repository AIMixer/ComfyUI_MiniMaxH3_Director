# 模型下载与安装 / Models and installation

核验日期 / Links checked: **2026-11-02**。

导演台不分发或自动下载模型。下表下载链接固定到已核验的 Hugging Face 版本；将文件放到实际运行的 ComfyUI 的 `models/` 下，刷新模型列表或重启后再选择。已有兼容 H3 基础模型、编码器及双 VAE 可复用，Pika 只新增任务 LoRA。

Director does not bundle or download weights. Links below pin verified Hugging Face revisions. Place files under the running ComfyUI installation's `models/` directory, then refresh the model list or restart. Compatible H3 base models, text encoder and both VAEs can be reused; Pika adds task LoRAs.

## 官方教程组合 / Comfy tutorial configuration

依据 [Comfy 官方 Pika 教程 / Comfy Pika tutorial](https://docs.comfy.org/tutorials/video/pika/pika-h3-flex) 和 [Comfy-Org 模型仓库 / model repository](https://huggingface.co/Comfy-Org/MiniMax-H3)。这些是 ComfyUI 格式的转换／量化文件，原始 H3 权重来自 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)。按任务选择一个底模和一个 LoRA；运行全部五类任务才需要两套。

These are ComfyUI-format converted/quantized files. Choose one matching base model and one LoRA per task; download both pairs to cover all five tasks.

| 用途 / Role | 文件下载 / File download | 安装目录 / Directory |
| --- | --- | --- |
| 纯文、首／尾帧底模 / Text and endpoint base | [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/912b6e38b2092ec4d2e86d24183e0d59ea059d48/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | `models/diffusion_models/` |
| 多模态底模 / Multimodal base | [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/912b6e38b2092ec4d2e86d24183e0d59ea059d48/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `models/diffusion_models/` |
| 纯文、首／尾帧适配器 / Text and endpoint LoRA | [pika_h3_flex_fl2va_v1_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/912b6e38b2092ec4d2e86d24183e0d59ea059d48/loras/pika_h3_flex_fl2va_v1_bf16.safetensors) | `models/loras/` |
| 多模态适配器 / Multimodal LoRA | [pika_h3_flex_ref2va_v1_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/912b6e38b2092ec4d2e86d24183e0d59ea059d48/loras/pika_h3_flex_ref2va_v1_bf16.safetensors) | `models/loras/` |
| 文本编码器 / Text encoder | [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/912b6e38b2092ec4d2e86d24183e0d59ea059d48/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `models/text_encoders/` |
| 视频 VAE / Video VAE | [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/912b6e38b2092ec4d2e86d24183e0d59ea059d48/vae/minimax_h3_video_vae_int8_convrot.safetensors) | `models/vae/` |
| 音频 VAE / Audio VAE | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/912b6e38b2092ec4d2e86d24183e0d59ea059d48/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/vae/` |

`UNETLoader` 接 `model`，`CLIPLoader` 接 `clip`，**type 选 `minimax`**；两个 `VAELoader` 分别接 `video_vae`、`audio_vae`。NVFP4 文本编码器不要求 Blackwell GPU；INT8 ConvRot 等量化需对应 ComfyUI／PyTorch 加载支持，见[模型仓库说明](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/README.md)。文件扩展名相同不代表所有量化格式都可互换。

Connect `UNETLoader` to `model`, `CLIPLoader` to `clip` with **type `minimax`**, and separate `VAELoader` nodes to `video_vae` and `audio_vae`. The NVFP4 text encoder does not require Blackwell. Quantized weights require compatible ComfyUI/PyTorch loading support; a shared extension does not make all formats interchangeable.

## 五任务配对 / Five task pairs

| 任务 / Task | 导演台 / Director | Pika CLI | 底模路径与 LoRA / Base path and LoRA |
| --- | --- | --- | --- |
| 纯文 / Text | `t2v` | `t2va` | FL2VA + `pika_h3_flex_fl2va_v1_bf16.safetensors` |
| 仅首帧 / First frame | `i2v`，或 / or `fl2v` 仅首帧 / first only | `i2va` | FL2VA + 同上 / same |
| 仅尾帧 / Last frame | `fl2v` 仅尾帧 / last only | `l2va` | FL2VA + 同上 / same |
| 首尾帧 / First and last | `fl2v` 两帧 / both frames | `fl2va` | FL2VA + 同上 / same |
| 多模态参考 / Multimodal references | `r2v` | `ref2va` | Ref2VA + `pika_h3_flex_ref2va_v1_bf16.safetensors` |

选择 `generation_family=Pika-H3` 后，导演台会在每段采样前自动配对并加载任务 LoRA，固定强度 **1.0**；缺文件会报错。不要在外部 LoRA Loader 再加载同一适配器，避免重复叠加。若使用兼容的不同文件名，在片段 LoRA 中明确选择，名称保留 `pika_h3_flex_fl2va_`／`pika_h3_flex_ref2va_` 前缀；切换任务会配对另一变体，因此对应文件也必须存在。Pika 固定 CFG 1、LCM、六步和视频／音频 shift 12/3；不要另叠 Turbo／Lightning 加速 LoRA，也不要直接沿用 H3 的 25 步采样说明。

With `generation_family=Pika-H3`, Director loads the matching task LoRA internally at **strength 1.0** before each segment. Missing files fail before sampling. Do not load the same adapter again in an external LoRA Loader. For a compatible alternative filename, select it in the segment LoRA list and retain the `pika_h3_flex_fl2va_` / `pika_h3_flex_ref2va_` prefix; switching tasks requires the matching counterpart filename. Pika fixes CFG 1, LCM, six steps and video/audio shifts 12/3. Do not stack Turbo/Lightning or use the H3 25-step recipe.

H3 模式不自动添加 Pika LoRA：`t2v/i2v/fl2v` 使用 FL2VA，`r2v/v2v/rv2v` 使用 Ref2VA；原有 H3 示例也可能选择 [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/912b6e38b2092ec4d2e86d24183e0d59ea059d48/vae/minimax_h3_video_vae_fp16.safetensors)，同样放在 `models/vae/`。2X VAE、SelfLift 和 Refine 属于可选扩展，不是运行 Pika 五任务的必需模型。

H3 mode does not add Pika LoRAs: use FL2VA for `t2v/i2v/fl2v`, Ref2VA for `r2v/v2v/rv2v`. Existing H3 examples may use `minimax_h3_video_vae_fp16.safetensors`. The 2X VAE, SelfLift and Refine are optional extensions, not requirements for the five Pika tasks.

## 本机已验证／API 示例组合 / Locally verified API configuration

仓库的 [Pika API 示例](../example_workflows/pika_pro/README.md) 使用以下社区文件替代上表的底模和编码器；双 VAE、两份 Pika LoRA 与上表相同。此组合在本机已有运行证据，不代表 Pika／MiniMax 官方推荐，也不保证任意量化组合的效果或兼容性。底模为融合权重，不能据此认定它与两份官方独立底模输出等价。

The repository's Pika API examples use these community files instead of the separate bases and text encoder above, with the same two VAEs and Pika LoRAs. This combination has local execution evidence; it is not an official Pika/MiniMax recommendation or a guarantee for other quantizations. The merged base is not output-equivalent to the two separate bases.

| 用途 / Role | 文件下载 / File download | 安装目录 / Directory |
| --- | --- | --- |
| 五任务共用社区融合底模 / Shared community hybrid base | [minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/a36feb17fbd1f20ff4bdd509ccd07e2b7b585a38/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors) | `models/diffusion_models/` |
| 社区 INT4 编码器 / Community INT4 text encoder | [qwen3vl_32b_minimax_h3_int4_convrot.safetensors](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot/resolve/3c167bc916cab6cf3b85b9e3769952757ebb2e5d/qwen3vl_32b_minimax_h3_int4_convrot.safetensors) | `models/text_encoders/` |

本机记录与上述公开文件的 SHA256 一致 / Recorded local SHA256 matches these published files:

- Hybrid: `a629cfea8d89a071b140c6e1935dc9a23e72de6badc18975a2bb9e6d1423d76d`
- INT4 encoder: `4389571ab5db4180bcae33d13d07855319530f60a85ab0f1061379401a8ef66c`

示例中的模型名称必须按自己的模型列表调整；不必为了套用官方教程组合重新下载已有兼容权重。单独底模组合中，前四份 API 图选 FL2VA，多模态图选 Ref2VA。

Update example loader names to match your model list. Existing compatible files need not be downloaded again. With separate base models, use FL2VA in the first four API examples and Ref2VA in the multimodal example.

## 原始与 Comfy 格式边界 / Original versus Comfy formats

- **Pika 官方原始适配器 / Original Pika adapters:** [官方推理仓库](https://github.com/Pika-Labs/Pika-H3-Flex) 的 [release.json](https://github.com/Pika-Labs/Pika-H3-Flex/blob/main/release.json) 固定 HF 版本 `0ddf55ad7067ca1e76812253bd7b4600df87f759`，使用 [FL2VA 原始文件](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/fl2va/adapter.safetensors) 和 [Ref2VA 原始文件](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/ref2va/adapter.safetensors)。它们是官方 Diffusers／PEFT 推理包使用的 FP32 格式，**不能改名后直接放进当前导演台替代 Comfy LoRA**。These FP32 files target the official Diffusers/PEFT runtime; renaming them does not convert them into Comfy LoRAs.
- **当前默认 Comfy 文件 / Current defaults:** 本页首表的 Comfy-Org `*_v1_bf16.safetensors`，对应导演台默认文件名和本机验证组合。The first table contains the defaults used by Director and the locally verified configuration.
- **Pika 另行发布的 Comfy 文件 / Pika's separate Comfy release:** [官方格式说明](https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/main/comfy/README.md) 另提供 `comfy/loras/*_comfy_bf16.safetensors`。转换包含模块重命名、QKV 融合及 SwiGLU 半部重排，文件名和哈希与 Comfy-Org v1 不同。该上传说明只验证完整性和转换张量，未验 ComfyUI 成片；本项目未验证其替换结果。如自行使用，需在片段 LoRA 明确选择兼容文件名，不能期待默认 v1 名自动命中。Conversion changes module names, fuses QKV and reorders SwiGLU halves. Names and hashes differ from Comfy-Org v1. The upload reports tensor/integrity validation, not a ComfyUI render; this project has not verified substitution. Select compatible alternative names explicitly in segment LoRAs.

所有模型及其派生权重遵循各来源标注的 MiniMax H3 模型许可；插件代码的 Apache-2.0 许可不替代模型许可。

Weights and derivatives retain the model licenses stated by their sources. The plugin's Apache-2.0 code license does not replace model licenses.
