"""Graph packer: face-refine config for MiniMax H3 Director.face_refine."""

from __future__ import annotations

import comfy.samplers

from ..director.face_refine.pack import (
    CANVAS_MODES,
    DEFAULT_DETECTOR,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    MMX_DIR_FACE_REFINE,
    PASTE_REGIONS,
    SEED_MODES,
    SELECT_MODES,
    detector_choices,
    pack_face_refine,
)

_CATEGORY = "MiniMaxH3"


class MiniMaxH3DirectorFaceRefine:
    """Pack face-refine settings. Connect ``face_refine`` to Director.face_refine.

    Unconnected Director skips this entirely. When wired, Director tracks the
    face on the final decoded frames (after Refine if that node is also wired),
    re-samples the crop with MiniMax H3, and pastes only the face back.
    """

    @classmethod
    def INPUT_TYPES(cls):
        detectors = detector_choices()
        return {
            "required": {
                "bd_grp_face_detect": ("BDGROUP", {"default": "脸部检测设置"}),
                "detector": (
                    detectors,
                    {
                        "default": DEFAULT_DETECTOR if DEFAULT_DETECTOR in detectors else detectors[0],
                        "tooltip": (
                            "人脸检测权重，放到 models/ultralytics/bbox/（如 face_yolov8m.pt）。"
                        ),
                    },
                ),
                "confidence": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.05,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": "检测阈值。更低更容易抓住侧脸和小脸。",
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 2.5,
                        "min": 1.2,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "裁剪边长 = 脸高 × 该倍数。2.0–3.0 常用。",
                    },
                ),
                "canvas_width": (
                    "INT",
                    {
                        "default": 768,
                        "min": 128,
                        "max": 1344,
                        "step": 32,
                        "tooltip": "H3 生成裁剪的宽度。manual 模式按此值。",
                    },
                ),
                "canvas_height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 128,
                        "max": 1344,
                        "step": 32,
                        "tooltip": "H3 生成裁剪的高度。manual 模式按此值。",
                    },
                ),
                "canvas_mode": (
                    list(CANVAS_MODES),
                    {
                        "default": "manual",
                        "tooltip": (
                            "manual = 使用上面宽高。"
                            "auto_capped_768 = 按最大裁剪自适应，上限 768。"
                        ),
                    },
                ),
                "select": (
                    list(SELECT_MODES),
                    {
                        "default": "largest_face",
                        "tooltip": "锁定对象：最大脸，或最靠近画面中心的脸。锁定后按邻近框跟踪。",
                    },
                ),
                "bd_grp_face_sample": ("BDGROUP", {"default": "采样设置"}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.40,
                        "min": 0.02,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "裁剪再采的 denoise（BasicScheduler）。H3 不要用 FaceDetailer 的 0.25；"
                            "模板约 0.40。接了 sigmas 则忽略此项。"
                        ),
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 50,
                        "tooltip": "脸修采样步数。配合 turbo LoRA 常用 8。接了 sigmas 则忽略。",
                    },
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": DEFAULT_SAMPLER,
                        "tooltip": "脸修采样器。示例工作流常用 euler。",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": DEFAULT_SCHEDULER,
                        "tooltip": "脸修调度器。接了 sigmas 则忽略。",
                    },
                ),
            },
            "optional": {
                "seed_mode": (
                    list(SEED_MODES),
                    {
                        "default": "inherit",
                        "tooltip": "inherit = 用导演台 seed；offset = seed+1+段号。",
                    },
                ),
                "bd_grp_face_paste": ("BDGROUP", {"default": "贴回设置"}),
                "paste_region": (
                    list(PASTE_REGIONS),
                    {
                        "default": "face_only",
                        "tooltip": "只贴检测脸框（推荐）。full_crop 会贴整块裁剪，容易露方块。",
                    },
                ),
                "mask_dilation": (
                    "INT",
                    {"default": 16, "min": 0, "max": 256, "step": 2},
                ),
                "feather": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,
                        "max": 256,
                        "step": 2,
                        "tooltip": "贴回羽化半径，单位为成片像素。矩形遮罩建议约 24。",
                    },
                ),
                "colour_match": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "blend": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": "可选。接线后覆盖步数 / 调度器 / denoise。",
                    },
                ),
                "refine_model": ("MODEL", {"tooltip": "独立修脸模型；不接时沿用片段模型。Pika主生成可另接未叠加LoRA的H3底模修脸。"}),
                "reference_image": ("IMAGE", {"tooltip": "可选清晰人物参考；不接时沿用片段参考图。不会自动挑选生成帧。"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "可选修脸专用提示词；留空沿用片段提示词。"}),
                "cut_detection": (["none", "auto"], {"default": "none", "tooltip": "auto按真实硬切镜独立跟踪、平滑与修复。"}),
                "cut_threshold": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "max_gap_frames": ("INT", {"default": 6, "min": 0, "max": 240}),
                "adaptive_strength": ("BOOLEAN", {"default": False, "tooltip": "小脸加强、大脸减弱；同时修正H3视频遮罩的MODEL采样行为。"}),
                "large_face_multiplier": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_px_small": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 800.0}),
                "face_px_large": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 1600.0}),
            },
        }

    RETURN_TYPES = (MMX_DIR_FACE_REFINE,)
    RETURN_NAMES = ("face_refine",)
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Connect to Director.face_refine. Director then face-refines the final decoded "
        "segment (after Director Refine if that is also connected). "
        "images is the stitched result; images_pre_face_refine is the video before stitch "
        "when Director「输出修脸前」is on. "
        "Unconnected Director is unchanged. Requires ultralytics + a face YOLO weight."
    )

    def pack(
        self,
        detector=DEFAULT_DETECTOR,
        confidence=0.35,
        crop_factor=2.5,
        canvas_width=768,
        canvas_height=768,
        canvas_mode="manual",
        select="largest_face",
        denoise=0.40,
        steps=8,
        sampler=DEFAULT_SAMPLER,
        scheduler=DEFAULT_SCHEDULER,
        seed_mode="inherit",
        paste_region="face_only",
        mask_dilation=16,
        feather=24,
        colour_match=1.0,
        blend=1.0,
        sigmas=None,
        **kwargs,
    ):
        return (
            pack_face_refine(
                detector=detector,
                confidence=confidence,
                crop_factor=crop_factor,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                canvas_mode=canvas_mode,
                select=select,
                denoise=denoise,
                steps=steps,
                sampler=sampler,
                scheduler=scheduler,
                seed_mode=seed_mode,
                paste_region=paste_region,
                mask_dilation=mask_dilation,
                feather=feather,
                colour_match=colour_match,
                blend=blend,
                sigmas=sigmas,
                **{key: kwargs[key] for key in (
                    "refine_model", "reference_image", "prompt", "cut_detection", "cut_threshold",
                    "max_gap_frames", "adaptive_strength", "large_face_multiplier", "face_px_small",
                    "face_px_large",
                ) if key in kwargs},
            ),
        )
