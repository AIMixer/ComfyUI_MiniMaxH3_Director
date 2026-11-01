# Pika-H3 Director Pro API 示例

五份示例由导演台自身保存正式分段视频，通过原生 `PreviewAny` 显示执行报告，不连接额外的 `CreateVideo` / `SaveVideo`，避免重复导出。不依赖 VHS。这里的 JSON 是 **API 图**，用于 `POST /prompt`，不是可拖入画布的 UI 工作流。导演台包含自定义控件及保存恢复逻辑，未提供未经 UI 导入验证的转换图。

| 示例 | 任务 | 需要自行选择或上传的素材 |
| --- | --- | --- |
| [纯文生视频](pika_pro_t2v_api.json) | t2v | 无 |
| [首帧](pika_pro_first_api.json) | fl2v，仅首帧 | `Pika_reference.png` |
| [尾帧](pika_pro_last_api.json) | fl2v，仅尾帧 | `Pika_last.png` |
| [首尾帧](pika_pro_first_last_api.json) | fl2v，首尾锚点 | `Pika_reference.png`、`Pika_last.png` |
| [多模态参考](pika_pro_multimodal_api.json) | r2v | `Pika_reference.png`、`Pika_reference.mp4`、`Pika_reference.wav` |

这些文件名都是占位符，目录中不包含素材。使用前请在自己的 ComfyUI 中上传或选择素材，并将 JSON 中的素材记录改成实际文件名；不要期待模板自动下载素材。API 中素材保存在节点 `200` 的 `inputs.timeline_data` JSON 字符串内，图片字段为 `imageFile` / `fileName`，视频为 `videoFile` / `fileName`，音频为 `audioFile` / `fileName`。请同时更新同一记录的两个文件名字段；有子目录时填写 `subfolder`，上传输入素材使用 `type: "input"`。首帧在 `startImage`，尾帧在 `endImage`，多模态参考在 `refs` / `refVideos` / `refAudios`。提示词引用 `<Picture 1>` / `<Video 1>` / `<Audio 1>` 对应这些数组中 `index: 0` 的素材。

多模态示例使用一张图片、一段参考视频和一段参考音频；视频和独立音频请各选 2–15 秒的有效素材。当前 Pika 适配还检查最多 9 张图、3 段视频、3 段独立音频，参考总数最多 12；多段参考视频总时长最多 15 秒，独立参考音频总时长最多 15 秒。参考视频提供运动线索，属于生成条件；它不是“成品视频直接拼接”。示例使用模型生成音轨，不复制参考 WAV 原声，也不启用音频冻结。首尾素材和演示提示词应与自己的场景相符，必要时一并修改提示词。

## 模型与统一设置

节点 `2`、`4`、`5`、`6` 分别加载扩散模型、文本编码器、视频 VAE 和音频 VAE。模板中的名称是可替换示例：

- `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`
- `qwen3vl_32b_minimax_h3_int4_convrot.safetensors`
- `minimax_h3_video_vae_int8_convrot.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`

请从自己的 ComfyUI 模型列表选择兼容权重，并更新对应 loader 输入。模板不分发模型，也不保证其他量化格式或所有权重组合均兼容；部分量化权重需要各自的加载支持。示例扩散权重名称代表可覆盖 FL2VA / Ref2VA 路径的组合；若使用独立模型，前四项选择兼容 FL2VA 的权重，多模态选择兼容 Ref2VA 的权重。

导演台在 `generation_family: "Pika-H3"` 下按任务自动加对应 Pika 适配器，默认名称为 `pika_h3_flex_fl2va_v1_bf16.safetensors` 或 `pika_h3_flex_ref2va_v1_bf16.safetensors`。请预先准备匹配的适配器并使其出现在 LoRA 列表中。若自己的兼容适配器文件名不同，可在该段的 `loras` 中写入对应名称（保留 `pika_h3_flex_fl2va_` / `pika_h3_flex_ref2va_` 命名规则），并保持强度 1；避免在 loader 外另加同一适配器导致叠加。

全部模板固定为：`generation_family=Pika-H3`、旧开关 `pika_flex=false`、`video_decode=tiled`，24 fps、124 帧、832×480、seed 42。124 帧在 24 fps 下约为 5.167 秒。Pika 路径采用 CFG 1、LCM、六步及视频/音频时钟 12/3。实时 TAE 预览、段间续接、音频冻结均关闭；没有连接 SelfLift、Refine、FaceRefine 或额外 SIGMAS。`PreviewAny` 仅显示执行报告文本，不开启实时视频预览。

## 提交 API 图

1. 安装并启用 H3 Director Pro，启动自己的 ComfyUI。
2. 选择模型，替换所需素材占位符，检查提示词。
3. 用以下标准库 Python 代码提交一份 API 图。按自己的服务器设置修改地址；成功提交会开始生成。

```python
import json
import urllib.error
import urllib.request
from pathlib import Path

api_file = Path("pika_pro_t2v_api.json")
graph = json.loads(api_file.read_text(encoding="utf-8"))
request = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": graph}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request) as response:
        print(response.read().decode("utf-8"))
except urllib.error.HTTPError as error:
    print(error.read().decode("utf-8"))
    raise
```

在该示例目录执行代码，或把 `api_file` 改为自己选定的 API 文件位置。也可以由现有客户端将图对象放到请求体的 `prompt` 字段。返回 `prompt_id` 只表示入队，完成与错误请查看 ComfyUI 界面或 `/history/{prompt_id}`；不应把入队成功当作成片已生成。

每份图只有一个输出触发节点：`201` 的 `PreviewAny` 连接导演台 `200` 的 `report`（端口 5）。导演台内部以 `exportMode: "segments"` 保存正式视频，示例不再额外保存第二份。正式视频与缓存目录遵循导演台节点既有约定；`cache_name` 仅为各示例隔离缓存，不修改正式输出规则。

生成完成后，在报告预览中查看 `Segment mp4 export dir` 和逐段 `saved →` 路径。API 客户端可读取 `GET /history/{prompt_id}`，报告文本位于该任务的 `outputs["201"]["text"]` 中（通常为字符串数组）；先确认任务已完成且无错误。仅请求已成功入队时，history 中可能还没有该输出。

## 验证范围

五份 JSON 已做解析、节点类型、必填输入、连接端口类型、枚举值和 `PreviewAny` 输出触发链的静态核对，依据当前运行时导出的节点定义。也核对了统一采样设置、首尾素材槽、多模态标签和占位符；不包含私人提示词、私人素材名或绝对文件路径。

这些公开模板没有提交生成，没有做 UI 图导入或使用占位素材的成片验收。节点定义和模型列表可能随 ComfyUI 版本变化；若服务器返回输入验证错误，请根据本机节点定义调整模型名称及相应输入。
