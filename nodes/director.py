"""MiniMax H3 Director — timeline UI + official MiniMax H3 AV execution."""

from __future__ import annotations

import logging

import comfy.samplers

from ..director.executor_core import execute_director_plan_core
from .director_common import (
    finalize_director_outputs,
    prepare_director_plan,
    timeline_required_inputs,
    director_perf_inputs,
)

_CATEGORY = "MiniMaxH3"

log = logging.getLogger("ComfyUI-MiniMaxH3-Director")

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
                **director_timeline_required_inputs(),
            },
            "optional": {
                "model": (
                    "MODEL",
                    {"tooltip": "MiniMax H3 UNET (UNETLoader). export_only（只跑导出）模式下可断开不连，跳过模型加载。"},
                ),
                "video_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 video VAE (minimax_h3_video_vae). export_only 模式下可断开不连。"},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 audio VAE (minimax_h3_audio_vae). Required for r2v / v2v / rv2v. export_only 模式下可断开不连。"},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "CLIPLoader type=minimax (qwen3vl). export_only 模式下可断开不连。"},
                ),
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
                "sampling": (
                    "MMX_DIR_SAMPLING",
                    {
                        "tooltip": (
                            "Optional Sampling node (MiniMax H3 Director Sampling). When connected, "
                            "it overrides steps / sampler / scheduler / cfg / seed / shift / sigma-refine. "
                            "Its optional SAMPLER / SIGMAS ports accept KSamplerSelect / BasicScheduler "
                            "or any third-party sampler / scheduler node. Unconnected = this node's widgets."
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
                        "tooltip": "Sampling steps — official template: 25.",
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
                        "tooltip": "Official template: BasicScheduler simple.",
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
        return True

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "source_images", "report", "images_pre_refine", "video_path")
    OUTPUT_IS_LIST = (True, True, False, False, True, False, True, False)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "MiniMax H3 Director: MiniMaxH3ImageToVideo / ReferenceToVideo conditioning, "
        "single-stage KSampler + MiniMaxH3SigmaShift, LTXVSeparateAVLatent decode. "
        "Supports t2v / i2v / fl2v / r2v / v2v / rv2v. "
        "Optional i2v_groups / r2v_groups accept multi-group packs from Director Group nodes "
        "(external priority over UI cards). Optional refine accepts MiniMax H3 Director Refine "
        "(second sample / upscale). images_pre_refine is the first-pass video before refine. "
        "Defaults: 0.4MP 16:9 (864×480), 5s / 124 frames @ 24 fps."
    )

    def execute(
        self,
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
        refine=None,
        sampling=None,
        steps=25,
        sampler="res_multistep",
        scheduler="simple",
        cfg=1.0,
        seed=0,
        shift_video=12.0,
        shift_audio=3.0,
        clear_vram_between_segments="unload_models",
        export_source_images=False,
        run_first_pass=True,
        run_refine=True,
        run_stream_export=False,
        run_normal_export=True,
        model=None,
        video_vae=None,
        audio_vae=None,
        clip=None,
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
            refine=refine,
            sampling=sampling,
        )
        plan.refine_only = (not bool(run_first_pass)) and bool(run_refine)
        plan.export_only = (not bool(run_first_pass)) and (not bool(run_refine))

        if bool(run_stream_export) and bool(run_normal_export):
            raise ValueError(
                "MiniMax H3 Director: 「流式导出」与「正常导出」互斥，不能同时勾选；"
                "请只保留其一。"
                " / Stream export and normal export are mutually exclusive; keep only one."
            )
        if not bool(run_stream_export) and not bool(run_normal_export):
            log.warning("Neither export checked — defaulting to normal export.")
            run_normal_export = True

        from ..director.sampling_pack import (
            sampling_cfg_for,
            sampling_sampler_name_for,
            sampling_sampler_obj_for,
            sampling_scheduler_for,
            sampling_seed_for,
            sampling_shift_for,
            sampling_sigma_refine_for,
            sampling_sigma_tail_for,
            sampling_sigmas_for,
            sampling_steps_for,
        )

        s_pack = getattr(plan, "sampling", None)
        eff_steps = sampling_steps_for(s_pack, steps)
        eff_sampler = sampling_sampler_name_for(s_pack, sampler)
        eff_scheduler = sampling_scheduler_for(s_pack, scheduler)
        eff_cfg = sampling_cfg_for(s_pack, cfg)
        eff_seed = sampling_seed_for(s_pack, seed)
        eff_shift_video, eff_shift_audio = sampling_shift_for(
            s_pack, shift_video, shift_audio
        )
        eff_sampler_obj = sampling_sampler_obj_for(s_pack)
        eff_sigmas = sampling_sigmas_for(s_pack)
        eff_sigma_refine = sampling_sigma_refine_for(s_pack, False)
        eff_sigma_tail = sampling_sigma_tail_for(s_pack, 1)

        stream_export = bool(run_stream_export)

        combined, segment_outputs, segment_audios, report, export_frame_counts, pre_combined, pre_segments, video_path = (
            execute_director_plan_core(
                plan,
                node_id=unique_id,
                model=model,
                vae=video_vae,
                audio_vae=audio_vae,
                clip=clip,
                cfg=eff_cfg,
                seed=eff_seed,
                steps=eff_steps,
                sampler=eff_sampler,
                scheduler=eff_scheduler,
                shift_video=eff_shift_video,
                shift_audio=eff_shift_audio,
                sampler_obj=eff_sampler_obj,
                sigmas=eff_sigmas,
                sigma_refine=eff_sigma_refine,
                sigma_tail_steps=eff_sigma_tail,
                clear_vram_between_segments=clear_vram_between_segments,
                stream_export=stream_export,
            )
        )

        if stream_export:
            fps_out = float(plan.frame_rate or 24.0)
            total_frames = sum(int(c) for c in export_frame_counts)
            note = "\n\n流式导出已开启：成片已写入 mp4（见 video_path 输出），images/audio 输出为空。"
            return ([], [], fps_out, total_frames, [], report + note, [], video_path)

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
        ) + (video_path,)
