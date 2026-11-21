"""Graph packer: RelayVSR upscale settings for MiniMax H3 Director.vsr."""

from __future__ import annotations

import comfy.samplers

from ..director.relay_vsr.pack import (
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    MMX_DIR_VSR,
    pack_relay_vsr,
)

_CATEGORY = "MiniMaxH3"


class MiniMaxH3DirectorRelayVSR:
    """Pack the official RelayVSR upscale graph into one node.

    Connect ``vsr`` to Director.vsr. Unconnected Director is unchanged.
    When wired, Director runs this pass after every other sample
    (after Refine if that is connected, and after FaceRefine if that is connected).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "RelayVSR UNET。示例：relayvsr_x2_lightx2v_int8_convrot.safetensors。"
                            "4 倍换成 relayvsr_x4_int8_convrot，并配合同倍数的 Flash VAE。"
                        ),
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "RelayVSR Flash 解码器。2 倍用 relayvsr_flash_vae_x2_fp32.safetensors，"
                            "4 倍用 relayvsr_flash_vae_x4_fp32.safetensors。倍数必须和 UNET 一致。"
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "放大采样种子。官方示例是 KSampler 自己的种子，不跟随导演台。",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "tooltip": "官方 RelayVSR 是 1 步。",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "官方示例 cfg=1，正负条件接的是同一路。",
                    },
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": DEFAULT_SAMPLER, "tooltip": "官方示例 euler。"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": DEFAULT_SCHEDULER, "tooltip": "官方示例 simple。"},
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "官方示例 denoise=1。",
                    },
                ),
                "detail": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "VSR Decode 的 detail。1 是模型完整细节，"
                            "0 接近低清画面的双三次放大。"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (MMX_DIR_VSR,)
    RETURN_NAMES = ("vsr",)
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Connect to Director.vsr. Director then upscales each segment with RelayVSR "
        "after Refine and FaceRefine, using the official 1-step graph "
        "(RelayVSR Conditioning → KSampler → VSR Decode). "
        "Audio and frame count pass through. Unconnected Director is unchanged. "
        "Pair relayvsr_x2 with flash_vae_x2, or x4 with flash_vae_x4."
    )

    def pack(
        self,
        model,
        vae,
        seed=0,
        steps=1,
        cfg=1.0,
        sampler=DEFAULT_SAMPLER,
        scheduler=DEFAULT_SCHEDULER,
        denoise=1.0,
        detail=1.0,
        **kwargs,
    ):
        del kwargs
        return (
            pack_relay_vsr(
                model=model,
                vae=vae,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                denoise=denoise,
                detail=detail,
            ),
        )
