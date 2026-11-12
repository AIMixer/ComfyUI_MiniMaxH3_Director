"""Configure the original selflift-Avatar sampler for Director segments."""

from __future__ import annotations

from ..director.selflift.pack import MMX_DIR_SELFLIFT


class MiniMaxH3DirectorAvatarSelfLift:
    @classmethod
    def INPUT_TYPES(cls):
        from nodes import NODE_CLASS_MAPPINGS

        sampler = NODE_CLASS_MAPPINGS.get("SelfLiftAvatarH3Sampler")
        if sampler is None:
            raise RuntimeError("需要安装并启用 selflift-Avatar 的 SelfLiftAvatarH3Sampler。")
        original = sampler.INPUT_TYPES()
        required = original["required"]
        optional = original["optional"]
        return {
            "required": {name: required[name] for name in (
                "sampler", "transition_step", "lowres_scale", "rho", "w_min", "w_max", "upscaler_model"
            )},
            "optional": {name: optional[name] for name in (
                "high_res_model", "highres_tiling", "tiling_mode", "tiling_tiles", "tiling_axis",
                "highres_tiling_mode", "highres_token_chunk", "highres_weight_cache_gb"
            )},
        }

    RETURN_TYPES = (MMX_DIR_SELFLIFT,)
    RETURN_NAMES = ("selflift",)
    FUNCTION = "pack"
    CATEGORY = "MiniMaxH3"
    DESCRIPTION = (
        "Connect to Director.selflift. Each segment calls the installed original "
        "SelfLiftAvatarH3Sampler with the Director model, conditioning, latent, seed and sigmas. "
        "Use guide continuity; dynamic continue masks are not supported by the original sampler."
    )

    def pack(self, sampler, transition_step, lowres_scale, rho, w_min, w_max, upscaler_model, **options):
        return ({
            "enabled": True,
            "backend": "avatar",
            "sampler": sampler,
            "transition_step": transition_step,
            "lowres_scale": lowres_scale,
            "rho": rho,
            "w_min": w_min,
            "w_max": w_max,
            "upscaler_model": upscaler_model,
            **options,
        },)
