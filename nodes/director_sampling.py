"""Graph packer: sampling config for MiniMax H3 Director.sampling."""

from __future__ import annotations

import comfy.samplers

from ..director.sampling_pack import (
    DEFAULT_CFG,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    DEFAULT_SIGMA_TAIL_STEPS,
    DEFAULT_STEPS,
    MMX_DIR_SAMPLING,
    pack_sampling,
)

_CATEGORY = "MiniMaxH3"


class MiniMaxH3DirectorSampling:
    """Pack all sampling settings; connect ``sampling`` to Director.sampling.

    Unconnected Director keeps using its own widgets (backward compatible).
    Optional ``sampler`` accepts ComfyUI ``SAMPLER`` (KSamplerSelect / any
    third-party sampler node); optional ``sigmas`` accepts ``SIGMAS``
    (BasicScheduler / any third-party scheduler node). Either overrides the
    sampler / scheduler combo when connected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": DEFAULT_STEPS,
                        "min": 1,
                        "max": 200,
                        "tooltip": "Sampling steps — official template: 25. Ignored when a SIGMAS node is wired into the sigmas port (steps = len(sigmas)-1).",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": DEFAULT_SAMPLER,
                        "tooltip": "Sampler. Overridden when a SAMPLER node is wired into the sampler port.",
                    },
                ),
                "scheduler_name": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": DEFAULT_SCHEDULER,
                        "tooltip": "Scheduler. Overridden when a SIGMAS node is wired into the sigmas port.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": DEFAULT_CFG,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.01,
                        "tooltip": "CFG for sampling.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Random seed for sampling.",
                    },
                ),
                "shift_video": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": (
                            "视频流 sigma 曲线的弧度。不改动起始 sigma(≈1.0 纯噪声)和结束 "
                            "sigma(=0 干净画面)，只改变中间分布：shift 越大，前段步数留在高噪声区，"
                            "更多步数压进低 sigma 尾部（细节区）。H3 官方默认 12，配合 sigma_refine "
                            "效果更好。 / Curvature of the video sigma schedule. Endpoints "
                            "(start ≈1.0 noise / end 0 clean) stay fixed; larger shift pushes "
                            "more steps into the low-sigma detail tail. Official default 12.",
                        ),
                    },
                ),
                "shift_audio": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": (
                            "音频流 sigma 曲线的弧度。模型内部用 shift_video / shift_audio 把音频 "
                            "latent 对齐到视频时间轴（默认 12/3=4），一般保持 3 即可。"
                            " / Curvature of the audio sigma schedule. The model aligns audio "
                            "to the video timeline via shift_video/shift_audio (default 12/3=4).",
                        ),
                    },
                ),
                "bd_grp_enhance": ("BDGROUP", {"default": "一键增强"}),
                "sigma_refine": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Sigma 精修：低 sigma 尾部加密（粗凿部分不变）。"
                            "治运动边缘的像素颗粒和闪烁，几乎不增加时间。"
                            " / Sigma refine: densify the low-sigma tail (coarse head "
                            "unchanged). Reduces motion-edge grain/flicker at almost "
                            "no extra cost.",
                        ),
                    },
                ),
                "sigma_tail_steps": (
                    "INT",
                    {
                        "default": DEFAULT_SIGMA_TAIL_STEPS,
                        "min": 0,
                        "max": 50,
                        "tooltip": "低 sigma 尾部额外插入的步数。默认 1，越多越稳但更慢。",
                    },
                ),
            },
            "optional": {
                "sampler": (
                    "SAMPLER",
                    {
                        "tooltip": (
                            "可选。接 KSamplerSelect 或任意第三方 sampler 节点，"
                            "覆盖 sampler_name。"
                            " / Optional. Any ComfyUI SAMPLER node (KSamplerSelect etc.) "
                            "overrides sampler_name.",
                        ),
                    },
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "可选。接任意 SIGMAS 节点（BasicScheduler / KarrasScheduler / "
                            "第三方调度器）覆盖 scheduler_name。\n"
                            "· 实际采样步数 = len(sigmas) - 1（由外部节点决定，本节点 steps "
                            "在接上 sigmas 后不参与）。\n"
                            "· 建议 sigmas 以 0 结尾（完整去噪）；若结尾非 0，输出会残留噪声。\n"
                            "· sigma 值会自动按 shift_video 重映射到 SigmaShift 后的模型，"
                            "无需手动对齐。\n"
                            "· 接上后 sigma_refine 不生效（外部调度完全由你控制）。\n"
                            " / Optional. Any ComfyUI SIGMAS node (BasicScheduler / "
                            "KarrasScheduler / third-party scheduler) overrides scheduler_name. "
                            "Actual steps = len(sigmas)-1. Should end at 0 for a fully denoised "
                            "result. Values are auto re-mapped through shift_video; sigma_refine "
                            "is ignored while a custom schedule is connected.",
                        ),
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        return True

    RETURN_TYPES = (MMX_DIR_SAMPLING,)
    RETURN_NAMES = ("sampling",)
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "MiniMax H3 Director Sampling: pack steps / sampler / scheduler / cfg / "
        "seed / sigma shift / sigma refine, connect to Director.sampling. "
        "Optional sampler (SAMPLER) and sigmas (SIGMAS) ports accept ComfyUI "
        "KSamplerSelect / BasicScheduler or any third-party sampler / scheduler node. "
        "Does not sample by itself — no IMAGE output."
    )

    def pack(
        self,
        steps=DEFAULT_STEPS,
        sampler_name=DEFAULT_SAMPLER,
        scheduler_name=DEFAULT_SCHEDULER,
        cfg=DEFAULT_CFG,
        seed=0,
        shift_video=12.0,
        shift_audio=3.0,
        sigma_refine=True,
        sigma_tail_steps=DEFAULT_SIGMA_TAIL_STEPS,
        sampler=None,
        sigmas=None,
        **kwargs,
    ):
        del kwargs
        pack = pack_sampling(
            steps=steps,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            cfg=cfg,
            seed=seed,
            shift_video=shift_video,
            shift_audio=shift_audio,
            sigma_refine=sigma_refine,
            sigma_tail_steps=sigma_tail_steps,
            sampler=sampler,
            sigmas=sigmas,
        )
        return (pack,)