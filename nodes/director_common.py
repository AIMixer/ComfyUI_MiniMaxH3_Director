"""Shared helpers for the MiniMax H3 Director timeline node."""

from __future__ import annotations

import json
import logging

import torch
from comfy_execution.graph_utils import ExecutionBlocker

from ..director.audio_export import (
    AUDIO_MODE_GENERATE,
    build_director_audio_outputs,
    resolve_audio_mode,
    source_audio_report_note,
)
from ..director.frame_align import pad_or_trim_frames
from ..director.gen_timeline import is_prompt_batch_timeline, is_video_batch_task_key
from ..director.plan import build_director_plan, count_all_timeline_segments, count_timeline_segments, plan_summary
from ..director.progress import report_director_planning
from ..lib.image_prep import fit_canvas, fit_video_long_edge
from ..lib.video_io import load_timeline_segment
from ..lib.task_prompts import task_type_combo_options

log = logging.getLogger("ComfyUI-MiniMaxH3-Director")


def timeline_required_inputs() -> dict:
    """Timeline + prompt widgets shared by Director nodes."""
    combo_options, combo_meta = task_type_combo_options()
    return {
        "task_type": (combo_options, combo_meta),
        "global_prompt": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "tooltip": "Synced from in-node UI (global mode).",
            },
        ),
        "bd_grp_sample": ("BDGROUP", {"default": "采样设置"}),
        "cfg": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "CFG for KSampler."},
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
        "frame_rate": (
            "FLOAT",
            {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Timeline / output FPS (H3 trained at 24)."},
        ),
        "width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
        "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
        "ref_max_size": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
        "total_frames": (
            "INT",
            {
                "default": 124,
                "min": 5,
                "max": 100000,
                "tooltip": "Timeline total frames (fl2v = sum of shots). Per-shot generation still capped near 512.",
            },
        ),
        "timeline_data": (
            "STRING",
            {"default": "", "multiline": True, "tooltip": "Internal — video, segments, refs (populated by UI)."},
        ),
    }


def director_perf_inputs() -> dict:
    """Performance widgets shared by Director nodes."""
    return {
        "bd_grp_perf": ("BDGROUP", {"default": "性能"}),
        "clear_vram_between_segments": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "段间清理显存：每段结束后卸载模型并清空 CUDA 缓存。",
            },
        ),
        "export_source_images": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "将时间轴原片解码到独立的 source_images 输出口；"
                    "需将 source_images 另接预览/合成节点才能查看，不会改变主 images。"
                    "默认关以节省内存。"
                ),
            },
        ),
    }


def default_timeline_json(
    *,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> str:
    return json.dumps(
        {
            "version": 4,
            "editMode": "global",
            "totalFrames": total_frames,
            "frameRate": frame_rate,
            "width": width,
            "height": height,
            "refMaxSize": ref_max_size,
            "output": {
                "mode": "fixed",
                "longEdge": ref_max_size,
                "width": width,
                "height": height,
                "maxExportFrames": 0,
                "exportMode": "all",
                "audioMode": "generate",
                "refImageSize": "match",
            },
            "videoClips": [],
            "video": {
                "fileName": "",
                "videoFile": "",
                "subfolder": "",
                "type": "input",
                "frames": [],
                "frameMap": [],
            },
            "global": {"taskType": task_type, "prompt": global_prompt, "refs": [], "referenceVideo": {}, "continuousReference": False},
            "segments": [
                {
                    "id": "s0",
                    "start": 0,
                    "length": total_frames,
                    "prompt": "",
                    "taskType": "",
                    "refs": [],
                    "referenceVideo": {},
                }
            ],
        },
        ensure_ascii=False,
    )


def prepare_director_plan(
    *,
    timeline_data: str,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
    unique_id: str | None,
    i2v_groups=None,
    r2v_groups=None,
    refine=None,
):
    from ..director.external_groups import (
        build_plan_from_external_groups,
        validate_external_group_inputs,
    )

    if not timeline_data or not timeline_data.strip():
        timeline_data = default_timeline_json(
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )

    task_key, ext_groups, family = validate_external_group_inputs(
        task_type=task_type,
        i2v_groups=i2v_groups,
        r2v_groups=r2v_groups,
    )

    if ext_groups is not None and family is not None:
        report_director_planning(
            unique_id,
            len(ext_groups),
            timeline_segment_total=len(ext_groups),
        )
        plan = build_plan_from_external_groups(
            ext_groups,
            family=family,
            timeline_data=timeline_data,
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )
        plan = _attach_refine(plan, refine)
        log.info(
            "MiniMax H3 Director: external %s groups × %d (task=%s) | %s",
            family,
            len(ext_groups),
            task_key,
            plan_summary(plan).replace("\n", " | "),
        )
        return plan

    report_director_planning(
        unique_id,
        count_timeline_segments(timeline_data),
        timeline_segment_total=count_all_timeline_segments(timeline_data),
    )

    plan = build_director_plan(
        timeline_data,
        global_task_type=task_type,
        global_prompt=global_prompt,
        total_frames=total_frames,
        frame_rate=frame_rate,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
    )
    plan = _attach_refine(plan, refine)
    log.info(plan_summary(plan).replace("\n", " | "))
    return plan


def _attach_refine(plan, refine):
    from ..director.refine_pack import normalize_refine_pack

    plan.refine = normalize_refine_pack(
        refine,
        base_width=int(getattr(plan, "width", 0) or 0),
        base_height=int(getattr(plan, "height", 0) or 0),
    )
    return plan


def _fit_source_clip_to_plan(plan, raw_clip: torch.Tensor) -> torch.Tensor:
    if plan.output_mode == "fixed":
        return fit_canvas(raw_clip, plan.width, plan.height)
    return fit_video_long_edge(raw_clip, plan.ref_max_size)


def build_source_images_output(
    plan,
    images_out: list[torch.Tensor],
    *,
    split_outputs: bool,
) -> list[torch.Tensor]:
    if split_outputs:
        chunks: list[torch.Tensor] = []
        # images_out is run-order (选择运行), not full timeline order.
        segs = plan.segments
        run_idx = getattr(plan, "run_indices", None)
        if run_idx is not None:
            segs = [
                plan.segments[i]
                for i in sorted(run_idx)
                if 0 <= i < len(plan.segments)
            ]
        for seg, generated in zip(segs, images_out):
            target_len = int(generated.shape[0])
            raw = load_timeline_segment(plan.raw, seg.start_frame, seg.end_frame)
            fitted = _fit_source_clip_to_plan(plan, raw)
            chunks.append(pad_or_trim_frames(fitted, target_len).cpu().float())
        return chunks

    target_len = int(images_out[0].shape[0]) if images_out else int(plan.total_frames or 0)
    raw = load_timeline_segment(plan.raw, 0, target_len)
    fitted = _fit_source_clip_to_plan(plan, raw)
    return [pad_or_trim_frames(fitted, target_len).cpu().float()]


def _empty_source_images_for(images_out: list[torch.Tensor]) -> list[torch.Tensor]:
    if not images_out:
        return [torch.full((1, 1, 1, 3), 0.5)]
    placeholders: list[torch.Tensor] = []
    for img in images_out:
        if isinstance(img, torch.Tensor) and img.ndim == 4:
            h, w, c = int(img.shape[1]), int(img.shape[2]), int(img.shape[3])
        else:
            h, w, c = 1, 1, 3
        placeholders.append(torch.full((1, h, w, c), 0.5))
    return placeholders


def _ensure_nonempty_image_batches(images_out: list[torch.Tensor], *, label: str) -> list[torch.Tensor]:
    fixed: list[torch.Tensor] = []
    for i, img in enumerate(images_out):
        if not isinstance(img, torch.Tensor) or img.ndim != 4:
            raise ValueError(f"Director {label}[{i}] is not a valid IMAGE tensor.")
        if int(img.shape[0]) <= 0:
            h, w, c = int(img.shape[1]), int(img.shape[2]), int(img.shape[3])
            log.warning("Director %s[%d] has 0 frames; emitting 1-frame placeholder.", label, i)
            fixed.append(torch.full((1, max(1, h), max(1, w), max(1, c)), 0.5))
        else:
            fixed.append(img)
    return fixed


def _layout_image_batches(
    plan,
    combined,
    segment_outputs,
    *,
    export_segments: bool,
    is_batch: bool,
    video_batch: bool,
) -> tuple[list[torch.Tensor], int]:
    if export_segments or (is_batch and not video_batch):
        images_out = segment_outputs
        frame_count = sum(int(s.shape[0]) for s in segment_outputs)
        return images_out, frame_count
    combined = pad_or_trim_frames(combined, plan.total_frames).cpu().float()
    return [combined], int(combined.shape[0])


def finalize_director_outputs(
    plan,
    combined,
    segment_outputs,
    report,
    *,
    export_source_images: bool = False,
    segment_audios: list | None = None,
    segment_frame_counts: list[int] | None = None,
    pre_refine_combined=None,
    pre_refine_segments: list | None = None,
    block_final_images: bool = False,
):
    is_batch = is_prompt_batch_timeline(plan.raw, plan.global_task_key)
    export_segments = plan.export_mode == "segments"
    video_batch = is_video_batch_task_key(plan.global_task_key)
    split_layout = export_segments or (is_batch and not video_batch)

    images_out, frame_count = _layout_image_batches(
        plan,
        combined,
        segment_outputs,
        export_segments=export_segments,
        is_batch=is_batch,
        video_batch=video_batch,
    )
    if export_segments and len(segment_outputs) > 1:
        report = (
            report
            + f"\n\nExport mode: segments — {len(segment_outputs)} clip(s) on images output."
        )
    if plan.run_indices is not None and split_layout:
        report = (
            report
            + f"\n\nPartial run: output contains {len(segment_outputs)} re-generated clip(s) only."
        )
    if not split_layout:
        if video_batch and is_batch and len(segment_outputs) > 1:
            report = report + f"\n\nExport mode: all — merged {frame_count} frame(s) on images output."
        if plan.run_indices is not None and video_batch:
            report = report + f"\n\nPartial run: re-generated {len(segment_outputs)} video group(s)."

    pre_segs = pre_refine_segments if pre_refine_segments else segment_outputs
    pre_comb = pre_refine_combined if pre_refine_combined is not None else combined
    share_pre = pre_comb is combined and (
        pre_segs is segment_outputs
        or (
            len(pre_segs) == len(segment_outputs)
            and all(a is b for a, b in zip(pre_segs, segment_outputs))
        )
    )
    if share_pre:
        pre_refine_out = images_out
    else:
        try:
            pre_refine_out, _ = _layout_image_batches(
                plan,
                pre_comb,
                pre_segs,
                export_segments=export_segments,
                is_batch=is_batch,
                video_batch=video_batch,
            )
        except Exception as exc:
            log.warning("images_pre_refine layout failed: %s", exc)
            pre_refine_out = images_out
            report = report + f"\n\nimages_pre_refine: fallback to images ({exc})."

    split_for_audio = split_layout
    audio_frame_end = frame_count if not split_for_audio else None
    audio_mode = resolve_audio_mode(plan)
    use_generated = audio_mode == AUDIO_MODE_GENERATE
    # Prefer caller-provided export lengths (post continuity trim); else match IMAGE batches.
    if segment_frame_counts is None and segment_audios and split_for_audio:
        segment_frame_counts = [int(s.shape[0]) for s in segment_outputs]
    audio_out, source_fallback = build_director_audio_outputs(
        plan,
        images_out,
        export_segments=split_for_audio,
        output_frame_end=audio_frame_end,
        segment_audios=segment_audios if use_generated else None,
        segment_frame_counts=segment_frame_counts if use_generated else None,
        audio_mode=audio_mode,
    )
    report = report + source_audio_report_note(
        plan,
        audio_out,
        export_segments=split_for_audio,
        output_frame_end=audio_frame_end,
        used_generated_audio=bool(use_generated and segment_audios),
        audio_mode=audio_mode,
        source_fallback=source_fallback,
    )

    split_source_outputs = export_segments or (is_batch and not video_batch)
    if export_source_images:
        try:
            source_images_out = build_source_images_output(
                plan,
                images_out,
                split_outputs=split_source_outputs,
            )
            source_frames = sum(int(batch.shape[0]) for batch in source_images_out)
            report = report + (
                f"\n\nSource images: decoded {source_frames} timeline frame(s) "
                f"on {len(source_images_out)} source_images batch(es)."
            )
        except Exception as exc:
            log.warning("Source images output failed: %s", exc)
            # Never disguise generated frames as the source comparison. A neutral
            # placeholder makes the failure visible while preserving the expensive run.
            source_images_out = _empty_source_images_for(images_out)
            report = report + (
                "\n\nSource images: FAILED — emitted neutral placeholder(s), not generated "
                f"frames. Check the timeline source path/decode ({type(exc).__name__}: {exc})."
            )
    else:
        source_images_out = _empty_source_images_for(images_out)

    images_out = _ensure_nonempty_image_batches(images_out, label="images")
    source_images_out = _ensure_nonempty_image_batches(source_images_out, label="source_images")
    pre_refine_out = _ensure_nonempty_image_batches(pre_refine_out, label="images_pre_refine")

    refine_pack = getattr(plan, "refine", None)
    if isinstance(refine_pack, dict) and refine_pack.get("enabled"):
        report = report + (
            "\n\nimages_pre_refine: first-pass video (before second sample / upscale). "
            "Cached or passthrough slots reuse the stored final frames."
        )
    else:
        report = report + (
            "\n\nimages_pre_refine: same as images (Refine node not connected)."
        )

    report = report + "\n\n有问题联系作者：AI搅拌手  QQ交流群：551482703"

    fps_out = float(plan.frame_rate or 24.0)
    if block_final_images:
        report = report + (
            "\n\n本轮仅确认一采：images（最终/二采输出）已阻断，"
            "请从 images_pre_refine 查看或保存一采；再次 Queue 完成二采后 images 才会输出。"
        )
        images_out = ExecutionBlocker(None)
    return images_out, audio_out, fps_out, frame_count, source_images_out, report, pre_refine_out
