"""Graph packer: SelfLift (progressive first-pass) for MiniMax H3 Director.selflift."""

from __future__ import annotations

from ..director.h3_latent_upscale import list_h3_latent_upscale_models
from ..director.refine_pack import DEFAULT_SPATIAL_TILES, DEFAULT_TILE_OVERLAP, MAX_SPATIAL_TILES
from ..director.selflift.pack import (
    DEFAULT_HIGHRES_STEPS,
    DEFAULT_LOWRES_SCALE,
    DEFAULT_RHO,
    DEFAULT_TRANSITION_STEP,
    DEFAULT_W_MAX,
    DEFAULT_W_MIN,
    MMX_DIR_SELFLIFT,
    SPLIT_MODES,
    UPSAMPLE_MODES,
    pack_selflift,
)

_CATEGORY = "MiniMaxH3"


class MiniMaxH3DirectorSelfLift:
    """Pack SelfLift settings. Connect ``selflift`` to Director.selflift.

    Unconnected Director is unchanged (same first-pass as today). When wired,
    each segment's first sample is low-res prefix + 3D lift + high-res tail
    on the Director canvas. Timeline continuity keeps a native low-res tail
    plus the existing high-res pin so segment seams stay stable.

    Does not sample by itself. TST (temporal stability) is not bundled —
    patch Director.model before this graph if you use a TST node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = list_h3_latent_upscale_models()
        default_model = models[0] if models else ""
        return {
            "required": {
                "bd_grp_selflift_sample": ("BDGROUP", {"default": "渐进采样"}),
                "split_mode": (
                    list(SPLIT_MODES),
                    {
                        "default": "highres_steps",
                        "tooltip": (
                            "highres_steps = 高清收尾步数（8 步默认 2，即低清 6 + 高清 2）。"
                            "transition_step = SelfLift 论文的 k（8 步默认 6）。"
                        ),
                    },
                ),
                "highres_steps": (
                    "INT",
                    {
                        "default": DEFAULT_HIGHRES_STEPS,
                        "min": 1,
                        "max": 64,
                        "tooltip": "仅 split_mode=highres_steps。高清阶段步数，建议约为总步数的 25%。",
                    },
                ),
                "transition_step": (
                    "INT",
                    {
                        "default": DEFAULT_TRANSITION_STEP,
                        "min": 1,
                        "max": 200,
                        "tooltip": "仅 split_mode=transition_step。低清前缀步数 k。8 步 turbo 用 6。",
                    },
                ),
                "lowres_scale": (
                    "FLOAT",
                    {
                        "default": DEFAULT_LOWRES_SCALE,
                        "min": 0.25,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "低清画布 = 导演台画布 × 该倍率，再对齐 ×32。"
                            "1.0 = 不渐进，走原来的一采。"
                        ),
                    },
                ),
                "sampler_mode": (
                    ["euler", "follow_director"],
                    {
                        "default": "euler",
                        "tooltip": (
                            "默认 euler：SelfLift 两阶段都用 Euler（论文路径，s_churn=0）。"
                            "不改导演台采样器。follow_director = 跟导演台，但必须也是 euler，否则直接报错。"
                        ),
                    },
                ),
                "native_low_carry": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "段间把上一段 native 低清尾写入当前低清前缀（与高清 pin 一起）。"
                            "关掉则低清前缀只是高清尾再降格，接缝容易闪/糊。"
                            "导演台未开段间引导时无效。"
                        ),
                    },
                ),
                "bd_grp_selflift_lift": ("BDGROUP", {"default": "提升 / 3D"}),
                "latent_upscale_model": (
                    models,
                    {
                        "default": default_model,
                        "tooltip": (
                            "H3 3D latent 放大权重，与 Refine 同一目录："
                            "ComfyUI/models/latent_upscale_models/。"
                            "rho=0（默认）时必选。"
                        ),
                    },
                ),
                "latent_upsample": (
                    list(UPSAMPLE_MODES),
                    {
                        "default": "bilinear",
                        "tooltip": "降格 cond / 升采样过渡态 x 的插值。干净端点 x0 走 3D 网。",
                    },
                ),
                "rho": (
                    "FLOAT",
                    {
                        "default": DEFAULT_RHO,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "论文像素锚混合。0 = 只用 3D lift（推荐，避免 VAE 往返）。"
                            ">0 时解码低清 x0、lanczos 放大像素再编码，按 w_min/w_max 混进 3D 结果。"
                            "真正的 3D 权重在上面的 latent_upscale_model。"
                        ),
                    },
                ),
                "w_min": (
                    "FLOAT",
                    {
                        "default": DEFAULT_W_MIN,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "仅 rho>0。低频区域的像素锚权重。",
                    },
                ),
                "w_max": (
                    "FLOAT",
                    {
                        "default": DEFAULT_W_MAX,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "仅 rho>0。高频残差区域的像素锚权重。",
                    },
                ),
                "enable_latent_chunking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "3D lift 时间分块（省显存，默认关）。接缝可能和整段前向不同。",
                    },
                ),
                "bd_grp_selflift_tile": ("BDGROUP", {"default": "高清分块"}),
                "enable_tiling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "仅高清收尾空间分块（默认关）。低清阶段不分块。"
                            "音频不切空间。"
                        ),
                    },
                ),
                "tile_count": (
                    "INT",
                    {
                        "default": DEFAULT_SPATIAL_TILES,
                        "min": 1,
                        "max": MAX_SPATIAL_TILES,
                        "tooltip": "高清分块数量。1 等同不分块。",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": DEFAULT_TILE_OVERLAP,
                        "min": 0,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "块间重叠，单位为输出像素。",
                    },
                ),
            },
            "optional": {
                "model_hires": (
                    "MODEL",
                    {
                        "tooltip": (
                            "可选高清阶段 UNET。不接则低清/高清都用导演台主模型。"
                            "适合低清挂 Turbo、高清换一套。"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (MMX_DIR_SELFLIFT,)
    RETURN_NAMES = ("selflift",)
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Connect to Director.selflift (above refine). Director first-pass becomes "
        "low-res prefix + 3D lift + high-res tail on the timeline canvas. "
        "Unconnected = current single-stage sample. "
        "Refine may still enlarge afterward (upscale / latent_upscale) or stay "
        "same-canvas. FaceRefine still runs after decode. "
        "Timeline continuity keeps native low-res carry + high-res pin. "
        "Euler only. TST is not bundled; patch Director.model if needed."
    )

    def pack(
        self,
        split_mode="highres_steps",
        highres_steps=DEFAULT_HIGHRES_STEPS,
        transition_step=DEFAULT_TRANSITION_STEP,
        lowres_scale=DEFAULT_LOWRES_SCALE,
        sampler_mode="euler",
        native_low_carry=True,
        latent_upscale_model=None,
        latent_upsample="bilinear",
        rho=DEFAULT_RHO,
        w_min=DEFAULT_W_MIN,
        w_max=DEFAULT_W_MAX,
        enable_latent_chunking=False,
        enable_tiling=False,
        tile_count=DEFAULT_SPATIAL_TILES,
        tile_overlap=DEFAULT_TILE_OVERLAP,
        model_hires=None,
        **kwargs,
    ):
        del kwargs
        return (
            pack_selflift(
                split_mode=split_mode,
                highres_steps=highres_steps,
                transition_step=transition_step,
                lowres_scale=lowres_scale,
                latent_upscale_model=latent_upscale_model,
                sampler_mode=sampler_mode,
                native_low_carry=native_low_carry,
                rho=rho,
                w_min=w_min,
                w_max=w_max,
                latent_upsample=latent_upsample,
                enable_latent_chunking=enable_latent_chunking,
                enable_tiling=enable_tiling,
                tile_count=tile_count,
                tile_overlap=tile_overlap,
                model_hires=model_hires,
            ),
        )
