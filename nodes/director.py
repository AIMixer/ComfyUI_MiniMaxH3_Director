"""MiniMax H3 Director — timeline UI + official MiniMax H3 AV execution."""

from __future__ import annotations

import comfy.samplers

from ..director.executor_core import execute_director_plan_core
from .director_common import (
    finalize_director_outputs,
    prepare_director_plan,
    timeline_required_inputs,
    director_perf_inputs,
)

_CATEGORY = "MiniMaxH3"

_DEFAULT_GLOBAL_PROMPT = "A cinematic scene with natural motion and synchronized ambience"


def director_timeline_required_inputs() -> dict:
    """Timeline widgets — defaults aligned with official MiniMax H3 workflow templates."""
    inputs = timeline_required_inputs()
    combo_options, combo_meta = inputs["task_type"]

    gp_meta = dict(inputs["global_prompt"][1])
    gp_meta["default"] = _DEFAULT_GLOBAL_PROMPT
    gp_meta["tooltip"] = (
        "User prompt — sent directly to MiniMaxH3ImageToVideo / ReferenceToVideo. "
        "r2v: <Picture 1>. v2v: source-timeline edit (<Video 1>). "
        "rv2v: source timeline + reference images (<Video 1> + <Picture N>)."
    )

    frames_meta = dict(inputs["total_frames"][1])
    frames_meta["default"] = 124
    frames_meta["tooltip"] = (
        "Frame count at 24 fps; snapped to MiniMax 17k+5 grid (124 ≈ 5s)."
    )

    return {
        **inputs,
        "task_type": (combo_options, combo_meta),
        "global_prompt": ("STRING", gp_meta),
        "total_frames": ("INT", frames_meta),
    }


class MiniMaxH3Director:
    """In-node timeline Director using ComfyUI official MiniMax H3 pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "MiniMax H3 UNET (UNETLoader)."},
                ),
                "video_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 video VAE (minimax_h3_video_vae)."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 audio VAE (minimax_h3_audio_vae). Required for r2v / v2v / rv2v."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "CLIPLoader type=minimax (qwen3vl)."},
                ),
                **director_timeline_required_inputs(),
            },
            "optional": {
                "i2v_groups": (
                    "MMX_DIR_GROUP",
                    {
                        "tooltip": (
                            "External Image to Video group(s) (t2v / i2v / fl2v). "
                            "When connected, overrides UI cards for execution (external priority). "
                            "Connect Group (Image to Video).group, or Groups Combine."
                        ),
                    },
                ),
                "r2v_groups": (
                    "MMX_DIR_GROUP",
                    {
                        "tooltip": (
                            "External Reference to Video group(s). "
                            "When connected, overrides UI cards for execution (external priority). "
                            "Connect Group (Reference to Video).group, or Groups Combine."
                        ),
                    },
                ),
                "semantic_bridge": (
                    "MMX_DIR_SEMANTIC_BRIDGE",
                    {
                        "tooltip": (
                            "Optional Semantic Bridge node (above SelfLift). When connected, "
                            "official cond tokens are rewritten with the student MLP "
                            "(RMS-norm → residual mix). Unconnected = identical. "
                            "Distilled on FL2VA; r2v / v2v / rv2v is forced-compat — use with care."
                        ),
                    },
                ),
                "selflift": (
                    "MMX_DIR_SELFLIFT",
                    {
                        "tooltip": (
                            "Optional SelfLift node (above Refine). When connected, first-pass "
                            "is low-res prefix + 3D lift + high-res tail on this Director canvas. "
                            "Unconnected = current single-stage sample. "
                            "Refine may still upscale afterward (e.g. 1.0MP first pass → 2.0MP). "
                            "Timeline continuity keeps native low-res carry + high-res pin. "
                            "Euler only."
                        ),
                    },
                ),
                "refine": (
                    "MMX_DIR_REFINE",
                    {
                        "tooltip": (
                            "Optional Refine node. When connected, each segment runs a second "
                            "sample pass (same-size refine, or upscale then sample). "
                            "Wire a MODEL into Refine.refine_model to use a different UNET for that pass; "
                            "unwired uses this Director model. "
                            "images is the refined result; images_pre_refine is the first pass. "
                            "Unconnected = single-pass (current behavior)."
                        ),
                    },
                ),
                "face_refine": (
                    "MMX_DIR_FACE_REFINE",
                    {
                        "tooltip": (
                            "Optional FaceRefine node. When connected, Director tracks the face "
                            "on the final decoded frames (after Refine if that is also wired), "
                            "re-samples the crop, and pastes the face back. "
                            "images is after stitch; images_pre_face_refine is before stitch "
                            "when「输出修脸前」is on (otherwise that output is blocked). "
                            "Unconnected = no face pass (that output stays blocked)."
                        ),
                    },
                ),
                "bd_grp_advanced": ("BDGROUP", {"default": "高级采样"}),
                "steps": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 200,
                        "tooltip": (
                            "一采步数（官方模板 25）。"
                            "接了 sigmas 口后忽略此项，改用外接噪声表。"
                        ),
                    },
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "res_multistep",
                        "tooltip": "Official template: KSamplerSelect res_multistep.",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "simple",
                        "tooltip": (
                            "一采调度器（官方模板 simple）。"
                            "接了 sigmas 口后忽略此项，改用外接噪声表。"
                        ),
                    },
                ),
                "shift_video": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "MiniMaxH3SigmaShift shift_video."},
                ),
                "shift_audio": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "MiniMaxH3SigmaShift shift_audio."},
                ),
                **director_perf_inputs(),
                "sigmas": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "可选。一采噪声表，接 BasicScheduler 或 ManualSigmas。"
                            "接线后覆盖导演台「步数」和「调度器」（采样器下拉仍有效）。"
                            "BasicScheduler 请接 SigmaShift 之后的同一套 H3 MODEL。"
                            "不接则仍用步数 + 调度器、denoise=1 自动算表。"
                        ),
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        if input_types is not None:
            expected = {
                "model": "MODEL",
                "video_vae": "VAE",
                "audio_vae": "VAE",
                "clip": "CLIP",
            }
            for name, want in expected.items():
                got = input_types.get(name)
                if got is not None and got != want:
                    return f"{name}: expected {want}, linked node returns {got}."
            got_sigmas = input_types.get("sigmas")
            if got_sigmas is not None and got_sigmas != "SIGMAS":
                return f"sigmas: expected SIGMAS, linked node returns {got_sigmas}."
            got_bridge = input_types.get("semantic_bridge")
            if got_bridge is not None and got_bridge != "MMX_DIR_SEMANTIC_BRIDGE":
                return (
                    "semantic_bridge: expected MiniMax H3 Director Semantic Bridge "
                    f"(MMX_DIR_SEMANTIC_BRIDGE), linked node returns {got_bridge}."
                )
            got_selflift = input_types.get("selflift")
            if got_selflift is not None and got_selflift != "MMX_DIR_SELFLIFT":
                return (
                    "selflift: expected MiniMax H3 Director SelfLift "
                    f"(MMX_DIR_SELFLIFT), linked node returns {got_selflift}."
                )
            got_face = input_types.get("face_refine")
            if got_face is not None and got_face != "MMX_DIR_FACE_REFINE":
                return (
                    "face_refine: expected MiniMax H3 Director FaceRefine "
                    f"(MMX_DIR_FACE_REFINE), linked node returns {got_face}."
                )
        return True

    @classmethod
    def IS_CHANGED(cls, unique_id=None, **kwargs):
        # Do not return NaN: that would re-run every Director queue even when
        # confirm_first_pass is off. Linked Refine is None here, so fingerprint
        # the .pre cache files that only the confirmation hold writes.
        del kwargs
        from ..director.segment_cache import first_pass_cache_disk_signature

        return first_pass_cache_disk_signature(unique_id)

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "IMAGE", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "images",
        "audio",
        "fps",
        "frame_count",
        "source_images",
        "report",
        "images_pre_refine",
        "images_pre_face_refine",
    )
    OUTPUT_IS_LIST = (True, True, False, False, True, False, True, True)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "MiniMax H3 Director: MiniMaxH3ImageToVideo / ReferenceToVideo conditioning, "
        "single-stage KSampler + MiniMaxH3SigmaShift, LTXVSeparateAVLatent decode. "
        "Supports t2v / i2v / fl2v / r2v / v2v / rv2v. "
        "Optional i2v_groups / r2v_groups accept multi-group packs from Director Group nodes "
        "(external priority over UI cards). Optional semantic_bridge accepts "
        "MiniMax H3 Director Semantic Bridge (cond-token student; Ref2VA forced-compat). "
        "Optional selflift accepts MiniMax H3 Director SelfLift "
        "(progressive first-pass on this canvas). Optional refine accepts MiniMax H3 Director Refine "
        "(second sample / upscale). Optional face_refine accepts MiniMax H3 Director FaceRefine "
        "(crop / re-sample / stitch). images_pre_refine is the first-pass video before refine. "
        "images_pre_face_refine is the video before face stitch "
        "(blocked unless FaceRefine is connected and「输出修脸前」is on). "
        "Defaults: 0.4MP 16:9 (864×480), 5s / 124 frames @ 24 fps."
    )

    def execute(
        self,
        model,
        video_vae,
        audio_vae,
        clip,
        task_type,
        global_prompt,
        frame_rate,
        width,
        height,
        ref_max_size,
        total_frames,
        timeline_data,
        unique_id=None,
        i2v_groups=None,
        r2v_groups=None,
        semantic_bridge=None,
        selflift=None,
        refine=None,
        face_refine=None,
        sigmas=None,
        steps=25,
        sampler="res_multistep",
        scheduler="simple",
        cfg=1.0,
        seed=0,
        shift_video=12.0,
        shift_audio=3.0,
        clear_vram_between_segments=True,
        clear_vram_before_refine=False,
        clear_vram_before_face_refine=False,
        cache_frames_codec="raw",
        export_source_images=False,
        export_pre_face_refine=False,
        **kwargs,
    ):
        del kwargs

        plan = prepare_director_plan(
            timeline_data=timeline_data,
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            unique_id=unique_id,
            i2v_groups=i2v_groups,
            r2v_groups=r2v_groups,
            semantic_bridge=semantic_bridge,
            selflift=selflift,
            refine=refine,
            face_refine=face_refine,
        )
        codec = str(cache_frames_codec or "raw").strip().lower()
        plan.cache_frames_codec = "ffv1" if codec == "ffv1" else "raw"

        try:
            combined, segment_outputs, segment_audios, report, export_frame_counts, pre_combined, pre_segments, held_for_confirmation, pre_face_combined, pre_face_segments = (
                execute_director_plan_core(
                    plan,
                    node_id=unique_id,
                    model=model,
                    vae=video_vae,
                    audio_vae=audio_vae,
                    clip=clip,
                    cfg=cfg,
                    seed=seed,
                    steps=steps,
                    sampler=sampler,
                    scheduler=scheduler,
                    sigmas=sigmas,
                    shift_video=shift_video,
                    shift_audio=shift_audio,
                    clear_vram_between_segments=clear_vram_between_segments,
                    clear_vram_before_refine=clear_vram_before_refine,
                    clear_vram_before_face_refine=clear_vram_before_face_refine,
                    export_pre_face_refine=export_pre_face_refine,
                )
            )

            return finalize_director_outputs(
                plan,
                combined,
                segment_outputs,
                report,
                export_source_images=export_source_images,
                segment_audios=segment_audios,
                segment_frame_counts=export_frame_counts,
                pre_refine_combined=pre_combined,
                pre_refine_segments=pre_segments,
                pre_face_combined=pre_face_combined,
                pre_face_segments=pre_face_segments,
                export_pre_face_refine=export_pre_face_refine,
                block_final_images=held_for_confirmation,
            )
        finally:
            # Full source/reference PCM is execution-scoped.
            cache = getattr(plan, "audio_decode_cache", None)
            if isinstance(cache, dict):
                cache.clear()
            for item in getattr(plan, "global_ref_audios", None) or []:
                if getattr(item, "audio_path", ""):
                    item.audio = None
