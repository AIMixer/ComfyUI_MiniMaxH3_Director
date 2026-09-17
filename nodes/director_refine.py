"""Graph packer: Refine / upscale config for MiniMax H3 Director.refine."""

from __future__ import annotations

import comfy.samplers

from ..director.h3_latent_upscale import list_h3_latent_upscale_models
from ..director.refine_pack import (
    ASPECT_RATIO_CHOICES,
    DEFAULT_REFINE_SIGMA_SAMPLER,
    DEFAULT_UPSCALE_MEGAPIXELS,
    FOLLOW_DIRECTOR_ASPECT,
    MAX_REFINE_PASSES,
    MMX_DIR_REFINE,
    REFINE_MODES,
    SEED_MODES,
    UPSCALE_METHODS,
    infer_upscale_target,
    pack_refine,
)

_CATEGORY = "MiniMaxH3"


class MiniMaxH3DirectorRefine:
    """Pack refine/upscale settings. Connect ``refine`` to Director.refine.

    ``refine``: same-resolution second sample.
    ``upscale``: enlarge to target canvas then second-sample.
    ``latent_upscale``: H3 latent enlarge only, no second sample.
    Second sample uses SIGMAS from BasicScheduler / ManualSigmas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    list(REFINE_MODES),
                    {
                        "default": "refine",
                        "tooltip": (
                            "refine = 同分辨率二采（精修）。"
                            "upscale = 先放大到目标画布再二采。"
                            "latent_upscale = 只放大 H3 latent，不再二采。"
                        ),
                    },
                ),
                "upscale_method": (
                    list(UPSCALE_METHODS),
                    {
                        "default": "h3_latent",
                        "tooltip": (
                            "仅 mode=upscale。"
                            "h3_latent = 先按目标画布放大 H3 视频 latent，再二采"
                            "（下方选 3D 权重）。"
                            "lanczos = 像素插值；可另接 upscale_model（RealESRGAN 等）。"
                            "nvidia_rtx_vsr = NVIDIA RTX Video Super Resolution"
                            "（需 nvidia-vfx + NVIDIA GPU）。"
                        ),
                    },
                ),
                "latent_upscale_model": (
                    list_h3_latent_upscale_models(),
                    {
                        "tooltip": (
                            "H3 3D latent 放大权重。"
                            "放到 ComfyUI/models/latent_upscale_models/，"
                            "文件名含 3d（如 minimax_h3_latent_upscaler_3d_*.safetensors）。"
                            "mode=latent_upscale，或 upscale + h3_latent 时使用。"
                        ),
                    },
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": DEFAULT_REFINE_SIGMA_SAMPLER,
                        "tooltip": (
                            "二采采样器。海螺案例用 euler；"
                            "BasicScheduler 高质量二采常用 res_multistep。"
                        ),
                    },
                ),
                "passes": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_REFINE_PASSES,
                        "tooltip": (
                            "精修次数。1 = 一次二采。"
                            "upscale 时只有第 1 次放大，之后都是同分辨率精修。"
                            "latent_upscale 不二采，此值无效。"
                        ),
                    },
                ),
            },
            "optional": {
                "refine_model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Second-pass UNET (二采模型)。"
                            "不接则用导演台主模型。"
                            "适合一采挂 Turbo LoRA、二采卸掉或换另一套。"
                        ),
                    },
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "二采噪声表。接 Comfy 自带 BasicScheduler 或 ManualSigmas。"
                            "mode=refine / upscale 时必须接线。"
                            "BasicScheduler 请接和二采相同的 MODEL（导演台主模型或 refine_model）。"
                            "H3 的 SigmaShift 仍由 Refine 内部套上。"
                        ),
                    },
                ),
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {
                        "tooltip": (
                            "可选。用「加载放大模型」接入，例如 RealESRGAN_x2plus。"
                            "仅 mode=upscale 且 upscale_method=lanczos 时使用。"
                            "不接则纯 lanczos 插值。选 nvidia_rtx_vsr / h3_latent 时忽略此口。"
                        ),
                    },
                ),
                "seed_mode": (
                    list(SEED_MODES),
                    {
                        "default": "inherit",
                        "tooltip": "inherit = 用导演台 seed；offset = 每轮 seed+1、+2…。",
                    },
                ),
                "aspect_ratio": (
                    list(ASPECT_RATIO_CHOICES),
                    {
                        "default": FOLLOW_DIRECTOR_ASPECT,
                        "tooltip": (
                            "放大目标画布，算法同导演台「输出分辨率」。"
                            "导演台是一采分辨率（例如 0.4 MP），这里是放大后的目标"
                            "（例如 1.0 MP）。跟随导演台：按导演台画布比例推 720P 档。"
                            "比例预设：配合百万像素。"
                            "自定义：直接填宽高（对齐 ×32）。"
                        ),
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": DEFAULT_UPSCALE_MEGAPIXELS,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": (
                            "百万像素，同导演台 ResolutionSelector。"
                            "1.0 MP 在 16:9 约为 1376×768（对齐 32）。仅比例预设时生效。"
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 0,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "自定义宽度（×32）。仅「自定义」时生效。",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 720,
                        "min": 0,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "自定义高度（×32）。仅「自定义」时生效。",
                    },
                ),
                "skip_fl2v": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "跳过首尾帧（fl2v）镜头的二采/放大。"
                            "二采会改画面，容易把钉死的首尾帧画飘；默认跳过以保护关键帧。"
                            "关掉则 fl2v 也走精修 / latent 放大。"
                        ),
                    },
                ),
                "confirm_first_pass": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "先确认一采再二采。默认关：一采完立刻二采（与现在相同）。"
                            "开：没有一采缓存时只跑一采并写出缓存/_pre.mp4；"
                            "已有精确匹配的一采缓存（同一 seed 及一采参数）则跳过一采只跑二采。"
                            "seed 请用 fixed，或第二次 Queue 前改回写出缓存时的 seed。"
                        ),
                    },
                ),
                "enable_latent_chunking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "H3 latent 放大的时间分块（省显存，默认关）。"
                            "开：latent 超过 24 帧时按 24 帧切块、重叠区加权融合，"
                            "放大网峰值更低，长段不易在这一步 OOM；接缝可能和整段前向不同。"
                            "≤24 帧仍走整段。关则与现在完全一致。"
                        ),
                    },
                ),
                "enable_tiling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "二采空间分块（省显存，默认关）。"
                            "开：沿画面长边切开视频 latent，每步分块前向再叠回；"
                            "音频整段参与，不切空间。"
                            "关：整幅一次采样，与现在完全一致。"
                            "mode=latent_upscale（不二采）时无效。"
                        ),
                    },
                ),
                "tile_count": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "分块数量。越大单块显存越低，但前向次数更多、更慢。"
                            "1 等同不分块。"
                        ),
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 2048,
                        "step": 64,
                        "tooltip": (
                            "块间重叠，单位为输出像素（步长 64）。"
                            "越大接缝越轻，单块也越大，省显存越少。"
                        ),
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        # Skip combo/min checks so old workflows (target_width=0 → aspect_ratio) can load.
        return True

    RETURN_TYPES = (MMX_DIR_REFINE, "INT", "INT")
    RETURN_NAMES = ("refine", "width", "height")
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "MiniMax H3 Director Refine: connect to Director.refine. "
        "Director.images is the refined / upscaled result; "
        "Director.images_pre_refine is the first-pass video (before second sample). "
        "Second sample uses SIGMAS from BasicScheduler / ManualSigmas. "
        "Upscale / latent_upscale canvas uses the same aspect + megapixels / custom W×H as Director. "
        "Director first-pass stays at its own resolution; Refine target is the enlarge size. "
        "width / height are the resolved target canvas (×32). "
        "Does not sample by itself — no IMAGE output. "
        "confirm_first_pass: first Queue writes first-pass cache; "
        "second Queue with the same seed runs refine only."
    )

    def pack(
        self,
        mode="refine",
        upscale_method="h3_latent",
        sampler="",
        passes=1,
        seed_mode="inherit",
        aspect_ratio=FOLLOW_DIRECTOR_ASPECT,
        megapixels=DEFAULT_UPSCALE_MEGAPIXELS,
        width=1280,
        height=720,
        skip_fl2v=True,
        confirm_first_pass=False,
        enable_latent_chunking=False,
        enable_tiling=False,
        tile_count=2,
        tile_overlap=128,
        latent_upscale_model=None,
        upscale_model=None,
        h3_latent_model="",
        sigmas=None,
        refine_model=None,
        model=None,
        target_width=0,
        target_height=0,
        **kwargs,
    ):
        del kwargs
        try:
            mp = float(megapixels)
        except (TypeError, ValueError):
            mp = DEFAULT_UPSCALE_MEGAPIXELS
        if mp < 0.1:
            mp = DEFAULT_UPSCALE_MEGAPIXELS
        try:
            w = int(width or 0)
        except (TypeError, ValueError):
            w = 1280
        try:
            h = int(height or 0)
        except (TypeError, ValueError):
            h = 720
        if w < 32:
            w = 1280
        if h < 32:
            h = 720
        try:
            n_passes = int(passes or 1)
        except (TypeError, ValueError):
            n_passes = 1
        if n_passes < 1:
            n_passes = 1
        pack = pack_refine(
            mode=mode,
            passes=n_passes,
            seed_mode=seed_mode,
            aspect_ratio=aspect_ratio,
            megapixels=mp,
            width=w,
            height=h,
            target_width=target_width,
            target_height=target_height,
            skip_fl2v=skip_fl2v,
            confirm_first_pass=bool(confirm_first_pass),
            enable_latent_chunking=bool(enable_latent_chunking),
            enable_tiling=bool(enable_tiling),
            tile_count=tile_count,
            tile_overlap=tile_overlap,
            upscale_method=upscale_method,
            sample_model=refine_model if refine_model is not None else model,
            latent_upscale_model=latent_upscale_model if latent_upscale_model is not None else h3_latent_model,
            upscale_model=upscale_model,
            sampler=sampler,
            sigmas=sigmas,
        )
        out_w = int(pack.get("target_width") or 0)
        out_h = int(pack.get("target_height") or 0)
        if out_w <= 0 or out_h <= 0:
            out_w, out_h = infer_upscale_target(0, 0)
        return (pack, int(out_w), int(out_h))
