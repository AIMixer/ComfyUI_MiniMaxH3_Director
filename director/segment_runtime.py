"""Per-segment helpers shared by the Director executor."""

from __future__ import annotations

import base64
import io

import torch
from PIL import Image

from ..lib.image_prep import fit_canvas, fit_video_long_edge
from ..lib.video_io import load_timeline_segment
from .frame_align import pad_or_trim_frames
from .plan import DirectorPlan


def needs_source_video(task_key: str) -> bool:
    return task_key in {"i2v", "fl2v", "v2v", "rv2v"}


def is_gen_timeline_plan(plan: DirectorPlan) -> bool:
    mode = str((plan.raw or {}).get("timelineMode") or "").lower()
    return mode in ("gen_blank", "gen_image", "prompt_batch", "image_batch", "fl2v")


def resolve_segment_raw_clip(plan: DirectorPlan, seg) -> torch.Tensor:
    """Prefer in-memory gen canvas / segment clip; fall back to timeline video decode."""
    if seg.source_clip is not None and seg.source_clip.shape[0] > 0:
        return seg.source_clip.clone()

    # Pure t2v (incl. external groups) has no source frames.
    # r2v/r2i: real data comes from refs / ref_videos / ref_audios, not gray source canvas.
    #   When lazy_source_clips is ON (or external pack path), skip slicing source_video.
    task_key = getattr(seg, "task_key", "")
    if task_key in ("t2v", "r2v", "r2i"):
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    # fl2v end-only: plan leaves source_clip=None on purpose. Do not slice the
    # placeholder gen source_video (len=segment_count gray frames) — that was
    # incorrectly becoming first_frame in the single-node path.
    if getattr(seg, "task_key", "") == "fl2v" and is_gen_timeline_plan(plan):
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    sv = plan.source_video
    if is_gen_timeline_plan(plan) and sv is not None and int(sv.shape[0]) > 0:
        start = max(0, int(seg.start_frame))
        end = min(int(seg.end_frame), int(sv.shape[0]))
        if end > start:
            return sv[start:end].clone()

    if (plan.raw or {}).get("externalGroups", {}).get("active"):
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    return load_timeline_segment(plan.raw, seg.start_frame, seg.end_frame)


def resolve_segment_raw_clip_with_lookahead(
    plan: DirectorPlan,
    seg,
    *,
    end_extra: int = 0,
) -> torch.Tensor:
    """Like ``resolve_segment_raw_clip``, but may pull frames past ``seg.end_frame``.

    Extra frames are conditioning-only (continuity gen length matching); they are
    not kept in the exported segment after trim.
    """
    extra = max(0, int(end_extra))
    if extra <= 0:
        return resolve_segment_raw_clip(plan, seg)

    if seg.source_clip is not None and seg.source_clip.shape[0] > 0:
        # Gen canvases have no timeline lookahead beyond the clip itself.
        return seg.source_clip.clone()

    task_key = getattr(seg, "task_key", "")
    if task_key in ("t2v", "r2v", "r2i", "fl2v") and is_gen_timeline_plan(plan):
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    end = int(seg.end_frame) + extra
    sv = plan.source_video
    if is_gen_timeline_plan(plan) and sv is not None and int(sv.shape[0]) > 0:
        start = max(0, int(seg.start_frame))
        end = min(end, int(sv.shape[0]))
        if end > start:
            return sv[start:end].clone()

    from ..lib.video_io import logical_frame_count

    total = logical_frame_count(plan.raw)
    end = min(end, total)
    start = max(0, int(seg.start_frame))
    if end <= start:
        return resolve_segment_raw_clip(plan, seg)
    return load_timeline_segment(plan.raw, start, end)


def source_passthrough_chunk(plan: DirectorPlan, seg) -> torch.Tensor:
    """Scaled source frames for skipped v2v segments with no generation cache yet."""
    raw_clip = resolve_segment_raw_clip(plan, seg)
    target_len = raw_clip.shape[0]
    if plan.output_mode == "fixed":
        clip = fit_canvas(raw_clip, plan.width, plan.height)
    else:
        clip = fit_video_long_edge(raw_clip, plan.ref_max_size)
    return pad_or_trim_frames(clip, target_len).cpu().float()


def segment_passthrough_chunk(plan: DirectorPlan, seg) -> torch.Tensor | None:
    """Best-effort fill for skipped segments from **real source video** only.

    Gen timelines (t2v/r2v/i2v/fl2v batch) use gray canvases or held refs as
    ``source_clip`` — never splice those into「全部导出」or the merge goes gray.
    Unselected gen segments must come from disk cache instead.
    """
    if is_gen_timeline_plan(plan):
        return None
    if (plan.raw or {}).get("externalGroups", {}).get("active"):
        # External i2v/r2v packs also lack real camera footage for fill.
        if not needs_source_video(seg.task_key) or seg.task_key in {"i2v", "fl2v"}:
            return None
    if needs_source_video(seg.task_key):
        try:
            return source_passthrough_chunk(plan, seg)
        except Exception:
            return None
    return None


def tensor_frame_to_jpeg_b64(frame: torch.Tensor) -> str:
    arr = (frame.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def frames_label(seg) -> str:
    return f"帧 {seg.start_frame}–{seg.end_frame} ({seg.frame_count}f)"
