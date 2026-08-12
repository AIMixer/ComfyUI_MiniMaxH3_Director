"""Cross-segment continuity helpers for MiniMax H3 Director.

Active path (opt-in「段间引导」): motion-context pin via
``director.h3_motion_context`` — previous segment AV tail → next segment
conditioning, then trim the pinned prefix.
Tasks: t2v / i2v / fl2v / r2v / v2v / rv2v.

Legacy Wan/SCAIL helpers below are retained for concat seam utilities and
historical APIs; ``apply_scail_continuity_core`` is a no-op on MiniMax H3.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from ..lib.image_prep import cat_frames_variable_size, fit_canvas, fit_long_edge
from .h3_motion_context import (
    CONTINUITY_TASK_KEYS,
    DEFAULT_CONTEXT_FRAMES as DEFAULT_CONTINUITY_OVERLAP,
    snap_context_frames,
)
from .plan import DirectorPlan, SegmentPlan, wan_align_frame_count
from .segment_cache import load_segment_cache

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.continuity")

MIN_CONTINUITY_OVERLAP = 5
MAX_CONTINUITY_OVERLAP = 56
# Last N frames of prev as appearance refs (hair/outfit pop at joins needs >1).
MAX_CONTINUITY_REF_FRAMES = 2
# Leading-body MAD skip disabled: on successful rv2v/SCAIL handoff the body
# start *should* resemble prev tail; trimming it removes continuity.
CONTINUITY_SEAM_ECHO_BUDGET = 0
CONTINUITY_SEAM_ECHO_MAD = 8.0
CONTINUITY_SEAM_JOIN_MAX_SKIP = 0
CONTINUITY_SEAM_JOIN_MAD = 8.0
# Source-only lookahead past segment end for wan-align coverage (not an export skip).
# 8f ≈ 2 latent groups — reduces freeze tails used as the next SCAIL guide.
CONTINUITY_SOURCE_LOOKAHEAD = 8
# Burn-in after SCAIL lock, discarded on export (timeline-safe anti hold→pop).
# 00035 still pulsed in exported opening — extend burn-in so pulse stays discarded.
CONTINUITY_SETTLING_FRAMES = 12
# Conditioning: mild RGB body bridge (clear-image path; mask ramp caused 花屏).
CONTINUITY_SOURCE_BODY_BRIDGE = 8
CONTINUITY_SOURCE_BODY_BRIDGE_WEIGHT = 0.40
CONTINUITY_SOURCE_BODY_BRIDGE_BLUR = 0  # 0 = full RGB lerp in prepend (not lowfreq)
# Warm-start first free latents from locked; mask MUST stay 1.0 (full denoise).
CONTINUITY_FREE_LATENT_WARMSTART = 0.55
CONTINUITY_FREE_LATENT_WARMSTART_2 = 0.34
CONTINUITY_FREE_LATENT_WARMSTART_3 = 0.16
# v13 mask ramp / lock feather → under-denoise 花屏 (user screenshot). Hard lock only.
CONTINUITY_FREE_LATENT_MASK_RAMP = (1.0, 1.0, 1.0)
CONTINUITY_LOCK_FEATHER_LATENT = 0
CONTINUITY_LOCK_FEATHER_MASK = 0.0
CONTINUITY_UNLOCK_FEATHER_LATENT = 0
CONTINUITY_UNLOCK_START_MASK = 1.0
CONTINUITY_UNLOCK_INIT_BLEND = 0.0
# No multiplicative gain / long RGB blend (画面花 / 幻影).
CONTINUITY_SEAM_SOFTEN_MAD = 99.0
CONTINUITY_SEAM_SOFTEN_FRAMES = 0
CONTINUITY_SEAM_SOFTEN_WEIGHT = 0.0
CONTINUITY_OPENING_LAST_FRAME_BLEND = 0
CONTINUITY_OPENING_LAST_FRAME_WEIGHT = 0.0
CONTINUITY_FIRST_HANDOFF_BLEND = 0
CONTINUITY_FIRST_HANDOFF_WEIGHT = 0.0
CONTINUITY_TAIL_LUMA_BLEND = 0
CONTINUITY_OPENING_EXPOSURE_SMOOTH = 0
CONTINUITY_OPENING_EXPOSURE_SPIKE = 0.008
CONTINUITY_OPENING_LUMA_BLEND = 0
CONTINUITY_OPENING_LUMA_MIN_RATIO = 0.55
CONTINUITY_OPENING_LUMA_MAX_RATIO = 1.8
CONTINUITY_OPENING_LUMA_EPSILON = 0.015
# Concat: additive luma ONLY — body0/hold→pop RGB caused 拖影+一顿一顿 (00035).
CONTINUITY_SEAM_ADD_LUMA_FRAMES = 12
CONTINUITY_SEAM_ADD_LUMA_MAX = 0.10
CONTINUITY_BODY0_SEAM_WEIGHT = 0.0
CONTINUITY_BODY1_SEAM_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_MAD = 99.0
CONTINUITY_MICRO_SEAM_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_WEIGHT_MAX = 0.0
CONTINUITY_MICRO_SEAM_SECOND_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_THIRD_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_MAD_SPAN = 8.0
CONTINUITY_OPENING_Y_MAP_BLEND = 0
CONTINUITY_OPENING_Y_MAP_WEIGHT = 0.0
CONTINUITY_TAIL_SOFTEN_FRAMES = 0
CONTINUITY_TAIL_SOFTEN_WEIGHT = 0.0
CONTINUITY_HOLD_MAX_FRAMES = 0
CONTINUITY_HOLD_MAD = 4.5
CONTINUITY_HOLD_POP_JUMP_MAD = 10.0
CONTINUITY_HOLD_POP_BRIDGE_WEIGHT = 0.0
CONTINUITY_HOLD_POP_BRIDGE_WEIGHT_MAX = 0.0
CONTINUITY_HOLD_POP_LAND_WEIGHT = 0.0
CONTINUITY_HOLD_POP_LOOKAHEAD = 2
CONTINUITY_HOLD_POP_SCAN = 6
CONTINUITY_HOLD_POP_LOWFREQ = False
CONTINUITY_HOLD_POP_BLUR = 32
CONTINUITY_SPIKE_PREV_MAD = 6.5
CONTINUITY_SPIKE_JUMP_MAD = 12.0
CONTINUITY_SPIKE_WEIGHT = 0.0
CONTINUITY_SPIKE_LAND_WEIGHT = 0.0
CONTINUITY_SPIKE_SCAN = 5
CONTINUITY_HOLD_POP_ON_TAIL = False


def _truthy_continuity_flag(value) -> bool:
    """Accept bool/int/common string forms from UI or reloaded workflows."""
    if value is True or value == 1:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return False


def resolve_continuity_settings(timeline: dict, *, segment_count: int) -> tuple[bool, int]:
    """Read segment continuity flags from timeline JSON (output only; default off)."""
    if segment_count < 2:
        return False, 0
    output = timeline.get("output") or {}
    enabled = _truthy_continuity_flag(
        output.get("continuityEnabled", output.get("continuity_enabled"))
    )
    if not enabled:
        return False, 0
    raw = (
        output.get("continuityOverlapFrames")
        or output.get("continuity_overlap_frames")
        or DEFAULT_CONTINUITY_OVERLAP
    )
    return True, snap_context_frames(raw)


def resolve_segment_continuity_from_prev(
    seg_data: dict | None,
    *,
    segment_index: int,
) -> bool:
    """Per-segment「引用上段」flag.

    Master「段间引导」must also be on (checked by ``is_continuity_active``).
    Missing field defaults to True so existing workflows keep pinning every segment.
    Segment index 0 never pins.
    """
    if int(segment_index) <= 0:
        return False
    if not isinstance(seg_data, dict):
        return True
    if "continuityFromPrev" in seg_data:
        raw = seg_data.get("continuityFromPrev")
    elif "continuity_from_prev" in seg_data:
        raw = seg_data.get("continuity_from_prev")
    else:
        return True
    return _truthy_continuity_flag(raw)


def timeline_row_for_index(timeline: dict | None, index: int) -> dict:
    """Best-effort segment/shot/group row from timeline for per-segment flags."""
    if not isinstance(timeline, dict) or int(index) < 0:
        return {}
    for key in ("segments", "shots", "r2vGroups", "fl2vGroups"):
        rows = timeline.get(key)
        if isinstance(rows, list) and int(index) < len(rows):
            row = rows[int(index)]
            if isinstance(row, dict):
                return row
    return {}


def resolve_continuity_lock_pixels(overlap_frames: int) -> int:
    """SCAIL prefix length in pixels (Wan 4n+1, for clean VAE round-trip)."""
    ov = max(MIN_CONTINUITY_OVERLAP, min(MAX_CONTINUITY_OVERLAP, int(overlap_frames)))
    return wan_align_frame_count(ov)


def resolve_continuity_guide_frames(overlap_frames: int) -> tuple[int, int, int, int, int]:
    """Map UI overlap → (context_px, tail_refs, seam_blend, opening_blend, color_match)."""
    lock = resolve_continuity_lock_pixels(overlap_frames)
    refs = min(MAX_CONTINUITY_REF_FRAMES, max(1, lock))
    return lock, refs, 0, 0, 0


def resolve_segment_generation_frames(
    *,
    segment_frame_count: int,
    segment_index: int,
    continuity_enabled: bool,
    continuity_overlap: int,
) -> tuple[int, int]:
    """Return (gen_frames, prefix_trim_after_decode).

    Continuity segments generate
    ``lock + settling + body + source_lookahead`` (Wan-aligned).
    After decode we drop ``lock + settling`` and keep the body in source order.
    Settling is a discarded burn-in so hold→pop is not exported.
    """
    body = max(1, int(segment_frame_count))
    if not continuity_enabled or segment_index <= 0:
        return wan_align_frame_count(body), 0
    lock_px = resolve_continuity_lock_pixels(continuity_overlap)
    if lock_px <= 0:
        return wan_align_frame_count(body), 0
    settling = max(0, int(CONTINUITY_SETTLING_FRAMES))
    raw = lock_px + settling + body + CONTINUITY_SOURCE_LOOKAHEAD
    return wan_align_frame_count(raw), lock_px + settling


def resolve_continuity_settling_frames() -> int:
    """Discarded burn-in frames after the SCAIL lock (0 when disabled)."""
    return max(0, int(CONTINUITY_SETTLING_FRAMES))


def _proxy_mad(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cheap mean-abs diff on a spatial proxy (RGB only)."""
    if a.shape[0] != b.shape[0]:
        return 1e9
    aa = a[..., :3].float()
    bb = b[..., :3].float()
    # Downsample for speed; values are 0..1 tensors from ComfyUI.
    step_h = max(1, int(aa.shape[1]) // 64)
    step_w = max(1, int(aa.shape[2]) // 64)
    aa = aa[:, ::step_h, ::step_w, :]
    bb = bb[:, ::step_h, ::step_w, :]
    # Scale to ~0..255 so thresholds match the offline video diagnostics.
    return float((aa - bb).abs().mean().item() * 255.0)


def count_leading_seam_echo_frames(
    body: torch.Tensor,
    prev_tail: torch.Tensor | None,
    *,
    max_skip: int = 4,
    mad_threshold: float = CONTINUITY_SEAM_ECHO_MAD,
) -> int:
    """Diagnostic: count leading body frames that replay ``prev_tail``.

    Not used to shift export/merge — dropping these removes the SCAIL handoff
    (result looks like continuity is off, especially on successful rv2v).
    """
    if (
        prev_tail is None
        or max_skip <= 0
        or int(body.shape[0]) <= 0
        or int(prev_tail.shape[0]) <= 0
    ):
        return 0
    limit = min(int(max_skip), int(body.shape[0]), int(prev_tail.shape[0]))
    if limit <= 0:
        return 0
    for k in range(limit, 0, -1):
        if _proxy_mad(body[:k], prev_tail[-k:]) <= mad_threshold:
            return k
    return 0


def match_clip_to_gen_length(clip: torch.Tensor, gen_frames: int) -> torch.Tensor:
    """Make source length equal ``gen_frames`` without freezing the last frame.

    Prefer truncating; if short, mirror-extend from existing motion (same idea as
    ``encode_tail_clip``) so Wan is not asked to invent hollow tail frames.
    """
    gen = max(1, int(gen_frames))
    n = int(clip.shape[0])
    if n == gen:
        return clip
    if n > gen:
        return clip[:gen]
    need = gen - n
    pad = clip[: min(need, n)].flip(0)
    while int(pad.shape[0]) < need:
        pad = torch.cat([pad, clip.flip(0)], dim=0)
    return torch.cat([clip, pad[:need]], dim=0)


def is_continuity_active(plan: DirectorPlan, seg: SegmentPlan) -> bool:
    """True only when UI「段间引导」is ON and this segment should pin the previous.

    When False, the executor must stay on the official MiniMax H3 path
    (stock ImageToVideo / ReferenceToVideo, no motion-context pin/patch/trim).
    Supported tasks: t2v / i2v / fl2v / r2v / v2v / rv2v.
    Per-segment ``continuity_from_prev`` (default True) can opt out while master is on.
    """
    return (
        plan.continuity_enabled
        and plan.segment_count >= 2
        and seg.index > 0
        and seg.task_key in CONTINUITY_TASK_KEYS
        and bool(getattr(seg, "continuity_from_prev", True))
    )


def is_continuity_lead_segment(plan: DirectorPlan, seg: SegmentPlan) -> bool:
    """First segment while continuity is on: no SCAIL, but avoid freeze-tail handoff."""
    return (
        plan.continuity_enabled
        and plan.segment_count >= 2
        and int(seg.index) == 0
        and seg.task_key in CONTINUITY_TASK_KEYS
    )


def resolve_prev_segment_output(
    plan: DirectorPlan,
    all_segments: list[SegmentPlan],
    seg_index: int,
    completed: dict[int, torch.Tensor],
    node_id: str | None,
) -> torch.Tensor | None:
    prev_idx = seg_index - 1
    if prev_idx < 0:
        return None
    if prev_idx in completed:
        return completed[prev_idx]
    prev_seg = all_segments[prev_idx]
    # Stale-ok: partial re-run after pipeline/fingerprint churn still needs the
    # previous render for motion context (better than failing or pinning gray).
    cached = load_segment_cache(node_id, prev_seg, plan, allow_stale=True)
    if cached is not None:
        return cached
    if not plan.continuity_enabled:
        return None
    raise ValueError(
        f"段间连贯：片段 #{seg_index + 1} 需要上一段 #{prev_idx + 1} 的生成结果。"
        "请先运行上一段，或开启「全部运行」以生成完整序列；"
        "若使用「选择运行」，请确保上一段已有有效缓存。"
    )


def _frame_mean_luminance(frame: torch.Tensor) -> float:
    """Mean Rec.601 luma for an HWC tensor in [0, 1] (from 8d2e27c)."""
    f = frame.float()
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return float(luma.mean().item())


def _tensor_mean_luminance(frames: torch.Tensor) -> float:
    """Mean Rec.601 luma over a frame batch (from 8d2e27c)."""
    f = frames.float()
    if f.dim() == 3:
        f = f.unsqueeze(0)
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return float(luma.mean().item())


def normalize_guide_luma_to_source(
    guide: torch.Tensor,
    source_frame: torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """Match guide batch mean luma to a source frame (8d2e27c prepend path)."""
    if guide is None or int(guide.shape[0]) <= 0:
        return guide
    ref = source_frame
    if ref.dim() == 4:
        ref = ref[0]
    ref = fit_canvas(ref.unsqueeze(0), width, height)[0]
    out = fit_canvas(guide, width, height).float()
    ref_mean = max(_frame_mean_luminance(ref), 1e-6)
    guide_mean = max(_tensor_mean_luminance(out), 1e-6)
    ratio = ref_mean / guide_mean
    ratio = max(0.55, min(1.8, ratio))
    if abs(ratio - 1.0) < 0.01:
        return out.to(dtype=guide.dtype)
    log.info(
        "Segment continuity: normalize guide luma ×%.3f (guide=%.3f → source=%.3f)",
        ratio,
        guide_mean,
        ref_mean,
    )
    return (out * ratio).clamp(0.0, 1.0).to(dtype=guide.dtype)


def _match_opening_luma_to_guide(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    blend_frames: int = CONTINUITY_OPENING_LUMA_BLEND,
) -> torch.Tensor:
    """Deprecated path: fading gain on body opening caused post-seam brightness pump.

    Kept for call-site compat; ``CONTINUITY_OPENING_LUMA_BLEND`` is 0. Prefer
    ``_match_tail_luma_to_next``.
    """
    if (
        body is None
        or guide is None
        or int(body.shape[0]) <= 0
        or int(guide.shape[0]) <= 0
        or int(blend_frames) <= 0
    ):
        return body
    last = guide[-1]
    if last.dim() == 4:
        last = last[0]
    if tuple(last.shape) != tuple(body[0].shape):
        last = fit_canvas(last.unsqueeze(0), int(body.shape[2]), int(body.shape[1]))[0]
    last_f = last.float()
    body0 = body[0].float()
    ref_mean = last_f.mean(dim=(0, 1)).clamp_min(1e-6)
    body_mean = body0.mean(dim=(0, 1)).clamp_min(1e-6)
    gain = (ref_mean / body_mean).clamp(
        CONTINUITY_OPENING_LUMA_MIN_RATIO,
        CONTINUITY_OPENING_LUMA_MAX_RATIO,
    )
    if float((gain - 1.0).abs().max().item()) < CONTINUITY_OPENING_LUMA_EPSILON:
        return body
    n = min(int(blend_frames), int(body.shape[0]))
    out = body.clone()
    for i in range(n):
        t = float(i) / float(n - 1) if n > 1 else 0.0
        g = gain * (1.0 - t) + torch.ones_like(gain) * t
        out[i] = (out[i].float() * g).clamp(0.0, 1.0).to(dtype=out.dtype)
    return out


def _match_tail_luma_to_next(
    prev: torch.Tensor,
    nxt: torch.Tensor,
    *,
    blend_frames: int = CONTINUITY_TAIL_LUMA_BLEND,
) -> torch.Tensor:
    """Fade prev[-n:] per-channel mean toward next[0] (exposure only, no pose mix)."""
    if (
        prev is None
        or nxt is None
        or int(prev.shape[0]) <= 0
        or int(nxt.shape[0]) <= 0
        or int(blend_frames) <= 0
    ):
        return prev
    first = nxt[0]
    if first.dim() == 4:
        first = first[0]
    if tuple(first.shape) != tuple(prev[-1].shape):
        first = fit_canvas(first.unsqueeze(0), int(prev.shape[2]), int(prev.shape[1]))[0]
    ref_mean = first.float().mean(dim=(0, 1)).clamp_min(1e-6)
    prev_mean = prev[-1].float().mean(dim=(0, 1)).clamp_min(1e-6)
    gain = (ref_mean / prev_mean).clamp(
        CONTINUITY_OPENING_LUMA_MIN_RATIO,
        CONTINUITY_OPENING_LUMA_MAX_RATIO,
    )
    if float((gain - 1.0).abs().max().item()) < CONTINUITY_OPENING_LUMA_EPSILON:
        return prev
    n = min(int(blend_frames), int(prev.shape[0]))
    out = prev.clone()
    for i in range(n):
        t = float(i) / float(n - 1) if n > 1 else 1.0
        g = torch.ones_like(gain) * (1.0 - t) + gain * t
        idx = -(n - i)
        out[idx] = (out[idx].float() * g).clamp(0.0, 1.0).to(dtype=out.dtype)
    log.info(
        "Segment continuity: tail RGB gain (%.3f,%.3f,%.3f) over %df → next[0]",
        float(gain[0]),
        float(gain[1]),
        float(gain[2]),
        n,
    )
    return out


def _smooth_opening_exposure(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    blend_frames: int = CONTINUITY_OPENING_EXPOSURE_SMOOTH,
) -> torch.Tensor:
    """Rescale body[:n] so per-channel means follow guide[-1] → body[n] linearly.

    SCAIL unlock often leaves body[0] matched then pops bright in 1–2 frames
    (00011 dY +3..+6). A single gain on body[0] cannot fix that trajectory.
    """
    if (
        body is None
        or guide is None
        or int(body.shape[0]) < 3
        or int(guide.shape[0]) <= 0
        or int(blend_frames) < 2
    ):
        return body
    n = min(int(blend_frames), int(body.shape[0]) - 1)
    if n < 2:
        return body
    last = guide[-1]
    if last.dim() == 4:
        last = last[0]
    if tuple(last.shape) != tuple(body[0].shape):
        last = fit_canvas(last.unsqueeze(0), int(body.shape[2]), int(body.shape[1]))[0]
    start = last.float().mean(dim=(0, 1)).clamp_min(1e-6)
    end = body[n].float().mean(dim=(0, 1)).clamp_min(1e-6)
    # Skip if opening is already smooth (no post-seam brightness spike).
    b0 = body[0].float().mean(dim=(0, 1))
    b1 = body[min(1, n)].float().mean(dim=(0, 1))
    b2 = body[min(2, n)].float().mean(dim=(0, 1))
    spike = float(((b1 - b0).abs() + (b2 - b1).abs()).mean().item())
    if spike < CONTINUITY_OPENING_EXPOSURE_SPIKE:
        return body
    out = body.clone()
    for i in range(n):
        t = float(i) / float(n)
        target = start * (1.0 - t) + end * t
        cur = out[i].float().mean(dim=(0, 1)).clamp_min(1e-6)
        gain = (target / cur).clamp(
            CONTINUITY_OPENING_LUMA_MIN_RATIO,
            CONTINUITY_OPENING_LUMA_MAX_RATIO,
        )
        out[i] = (out[i].float() * gain).clamp(0.0, 1.0).to(dtype=out.dtype)
    log.info(
        "Segment continuity: opening exposure ramp %df (spike=%.3f)",
        n,
        spike,
    )
    return out


def _adaptive_seam_soften(
    body: torch.Tensor,
    guide: torch.Tensor,
) -> torch.Tensor:
    """Disabled by default — RGB mix toward prev[-1] smeared detail (画面花)."""
    if (
        body is None
        or guide is None
        or int(body.shape[0]) <= 0
        or int(guide.shape[0]) <= 0
        or CONTINUITY_SEAM_SOFTEN_FRAMES <= 0
        or CONTINUITY_SEAM_SOFTEN_WEIGHT <= 0
    ):
        return body
    seam_mad = _frame_mad(guide[-1], body[0])
    if seam_mad < CONTINUITY_SEAM_SOFTEN_MAD:
        return body
    extra = (seam_mad - CONTINUITY_SEAM_SOFTEN_MAD) / 50.0
    weight = min(0.32, float(CONTINUITY_SEAM_SOFTEN_WEIGHT) + extra * 0.10)
    return _blend_opening_toward_last_frame(
        body,
        guide,
        blend_frames=int(CONTINUITY_SEAM_SOFTEN_FRAMES),
        max_weight=weight,
    )


def _additive_opening_luma(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    blend_frames: int = CONTINUITY_SEAM_ADD_LUMA_FRAMES,
    max_delta: float = CONTINUITY_SEAM_ADD_LUMA_MAX,
) -> torch.Tensor:
    """Add the same RGB delta so mean luma eases from guide[-1] → body[n].

    Unlike multiplicative gain, this keeps local contrast (less 画面花).
    Triggers when body[0] disagrees with prev *or* the opening pops, even if
    prev and body[n] already share a similar mean (common SCAIL dark dip).
    """
    if (
        body is None
        or guide is None
        or int(body.shape[0]) < 3
        or int(guide.shape[0]) <= 0
        or int(blend_frames) < 2
        or max_delta <= 0
    ):
        return body
    n = min(int(blend_frames), int(body.shape[0]) - 1)
    if n < 2:
        return body
    last = guide[-1]
    if last.dim() == 4:
        last = last[0]
    if tuple(last.shape) != tuple(body[0].shape):
        last = fit_canvas(last.unsqueeze(0), int(body.shape[2]), int(body.shape[1]))[0]
    coeffs = body.new_tensor([0.299, 0.587, 0.114])

    def _y_mean(frame: torch.Tensor) -> float:
        return float((frame.float() * coeffs).sum(dim=-1).mean().item())

    y_start = _y_mean(last)
    y0 = _y_mean(body[0])
    y1 = _y_mean(body[min(1, n)])
    y2 = _y_mean(body[min(2, n)])
    y_end = _y_mean(body[n])
    seam_gap = abs(y0 - y_start)
    open_spike = abs(y1 - y0) + abs(y2 - y1)
    if (
        seam_gap < CONTINUITY_OPENING_EXPOSURE_SPIKE
        and open_spike < CONTINUITY_OPENING_EXPOSURE_SPIKE
        and abs(y_end - y_start) < CONTINUITY_OPENING_EXPOSURE_SPIKE
    ):
        return body

    out = body.clone()
    for i in range(n):
        t = float(i) / float(n)
        target = y_start * (1.0 - t) + y_end * t
        cur = _y_mean(out[i])
        delta = max(-float(max_delta), min(float(max_delta), target - cur))
        if abs(delta) < 1e-5:
            continue
        out[i] = (out[i].float() + delta).clamp(0.0, 1.0).to(dtype=out.dtype)
    log.info(
        "Segment continuity: additive opening luma %df "
        "(seam_gap=%.3f spike=%.3f, %.3f→%.3f, cap±%.3f)",
        n,
        seam_gap,
        open_spike,
        y_start,
        y_end,
        max_delta,
    )
    return out


def _blur_hwc(frame: torch.Tensor, kernel: int) -> torch.Tensor:
    """Box-blur an HWC frame (reflect pad). kernel forced odd >=3."""
    k = int(kernel)
    if k < 3:
        return frame.float()
    if k % 2 == 0:
        k += 1
    x = frame.detach().float()
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    t = x.permute(2, 0, 1).unsqueeze(0)
    pad = k // 2
    t = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="reflect")
    t = torch.nn.functional.avg_pool2d(t, kernel_size=k, stride=1)
    return t.squeeze(0).permute(1, 2, 0)


def _lowfreq_appearance_pull(
    src: torch.Tensor,
    guide: torch.Tensor,
    *,
    weight: float,
    blur: int,
) -> torch.Tensor:
    """Keep src detail; absorb guide's low-frequency grade/lighting only.

    Full RGB lerp copies pose edges → 重影 + hold→pop. Low-freq residual
    transfer matches appearance without a second silhouette.
    """
    w = float(weight)
    if w <= 0:
        return src
    g = guide
    if g.dim() == 4:
        g = g[0]
    if tuple(g.shape[:2]) != tuple(src.shape[:2]):
        g = fit_canvas(g.unsqueeze(0), int(src.shape[1]), int(src.shape[0]))[0]
    b_src = _blur_hwc(src, blur)
    b_guide = _blur_hwc(g, blur)
    out = src.float() + w * (b_guide - b_src)
    return out.clamp(0.0, 1.0).to(dtype=src.dtype)


def _soften_body0_toward_prev(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    weight0: float = CONTINUITY_BODY0_SEAM_WEIGHT,
    weight1: float = CONTINUITY_BODY1_SEAM_WEIGHT,
) -> torch.Tensor:
    """Unilateral ease of body opening toward prev[-1] — no prev-tail rewrite.

    00033 顿感: hard cut / opening brake. Tiny pull on body[0]/[1] restores join
    softness without painting the next pose into the previous ending.
    Weight scales up slightly when the cut MAD is high.
    """
    if (
        body is None
        or guide is None
        or int(body.shape[0]) <= 0
        or int(guide.shape[0]) <= 0
        or (weight0 <= 0 and weight1 <= 0)
    ):
        return body
    last = guide[-1]
    if last.dim() == 4:
        last = last[0]
    if tuple(last.shape) != tuple(body[0].shape):
        last = fit_canvas(last.unsqueeze(0), int(body.shape[2]), int(body.shape[1]))[0]
    cut = _frame_mad(last, body[0])
    # Soft boost when cut is harsh; keep mild (v16 boost→ghost).
    boost = 1.0 + max(0.0, min(0.25, (cut - 10.0) / 16.0))
    w0 = min(0.10, float(weight0) * boost)
    w1 = min(0.05, float(weight1) * boost)
    out = body.clone()
    last_f = last.float()
    if w0 > 0:
        out[0] = (
            (out[0].float() * (1.0 - w0) + last_f * w0)
            .clamp(0.0, 1.0)
            .to(dtype=out.dtype)
        )
    if w1 > 0 and int(out.shape[0]) > 1:
        out[1] = (
            (out[1].float() * (1.0 - w1) + last_f * w1)
            .clamp(0.0, 1.0)
            .to(dtype=out.dtype)
        )
    log.info(
        "Segment continuity: unilateral opening seam ease "
        "w=%.2f/%.2f (cutMAD=%.1f boost=%.2f)",
        w0,
        w1,
        cut,
        boost,
    )
    return out


def _frame_mad(a: torch.Tensor, b: torch.Tensor) -> float:
    """Single-frame MAD on ~0..255 scale (matches offline diagnostics / SEAM_*)."""
    aa = a.detach().float()
    bb = b.detach().float()
    if aa.dim() == 3:
        aa = aa.unsqueeze(0)
        bb = bb.unsqueeze(0)
    return _proxy_mad(aa, bb)


def _soft_pull_opening_luma_map(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    blend_frames: int = CONTINUITY_OPENING_Y_MAP_BLEND,
    max_weight: float = CONTINUITY_OPENING_Y_MAP_WEIGHT,
) -> torch.Tensor:
    """Pull body opening luma toward guide[-1]; keep body chroma (less pose ghost)."""
    n = min(int(blend_frames), int(body.shape[0]) if body is not None else 0)
    if n <= 0 or guide is None or int(guide.shape[0]) <= 0 or max_weight <= 0:
        return body
    out = body.clone()
    last = guide[-1]
    if last.dim() == 4:
        last = last[0]
    if tuple(last.shape) != tuple(out[0].shape):
        last = fit_canvas(last.unsqueeze(0), int(out.shape[2]), int(out.shape[1]))[0]
    coeffs = out.new_tensor([0.299, 0.587, 0.114])
    y_last = (last.float() * coeffs).sum(dim=-1)
    for i in range(n):
        w = float(max_weight) * (1.0 - float(i) / float(n))
        pix = out[i].float()
        y_b = (pix * coeffs).sum(dim=-1)
        y_t = y_b * (1.0 - w) + y_last * w
        scale = (y_t / y_b.clamp_min(1e-6)).unsqueeze(-1)
        out[i] = (pix * scale).clamp(0.0, 1.0).to(dtype=out.dtype)
    return out


def _unfreeze_held_tail(
    left: torch.Tensor,
    *,
    max_frames: int = CONTINUITY_HOLD_MAX_FRAMES,
    hold_mad: float = CONTINUITY_HOLD_MAD,
) -> torch.Tensor:
    """Spread a near-cut freeze across the tail using *only* left's own last frame.

    Morphing toward ``next[0]`` (00009) paints the next pose into the previous
    ending — visible 幻影. Intra-lerp removes the hold without cross-segment mix.
    """
    if left is None or int(left.shape[0]) < 3 or int(max_frames) <= 0:
        return left

    hold_n = 0
    limit = min(int(max_frames), int(left.shape[0]) - 1)
    for i in range(1, limit + 1):
        if _frame_mad(left[-i], left[-(i + 1)]) <= hold_mad:
            hold_n = i
            break
    if hold_n <= 0:
        return left

    out = left.clone()
    anchor = out[-(hold_n + 1)].float()
    end = out[-1].float()
    # If the freeze is the last pair, end≈anchor — nothing to spread.
    if _frame_mad(anchor, end) <= hold_mad:
        return left
    for j in range(hold_n):
        t = float(j + 1) / float(hold_n + 1)
        idx = -(hold_n - j)
        out[idx] = (anchor * (1.0 - t) + end * t).clamp(0.0, 1.0).to(dtype=out.dtype)
    log.info(
        "Segment continuity: unfreeze held tail %df (hold_mad≤%.1f, intra only)",
        hold_n,
        hold_mad,
    )
    return out


def _micro_seam_bridge(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    mad_threshold: float = CONTINUITY_MICRO_SEAM_MAD,
    weight: float = CONTINUITY_MICRO_SEAM_WEIGHT,
    second_weight: float = CONTINUITY_MICRO_SEAM_SECOND_WEIGHT,
    third_weight: float = CONTINUITY_MICRO_SEAM_THIRD_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilateral 1–2 frame soften; weight scales with seam MAD (capped)."""
    if (
        left is None
        or right is None
        or int(left.shape[0]) <= 0
        or int(right.shape[0]) <= 0
        or weight <= 0
    ):
        return left, right
    a = left[-1]
    b = right[0]
    if b.dim() == 4:
        b = b[0]
    if a.dim() == 4:
        a = a[0]
    if tuple(a.shape) != tuple(b.shape):
        b = fit_canvas(b.unsqueeze(0), int(left.shape[2]), int(left.shape[1]))[0]
    seam_mad = _frame_mad(a, b)
    if seam_mad < mad_threshold:
        return left, right

    # Harder cut → stronger bridge, but never above WEIGHT_MAX (00020 ghost zone).
    span = max(1.0, float(CONTINUITY_MICRO_SEAM_MAD_SPAN))
    t = max(0.0, min(1.0, (seam_mad - mad_threshold) / span))
    w0 = float(weight) + (float(CONTINUITY_MICRO_SEAM_WEIGHT_MAX) - float(weight)) * t
    w1 = w0 * float(second_weight) if 0 < float(second_weight) < 1.0 else float(second_weight)
    w2 = float(third_weight)
    if 0 < w2 < 1.0:
        w2 = w0 * w2

    out_l = left.clone()
    out_r = right.clone()
    a_f = a.float()
    b_f = b.float()
    weights = (w0, w1, w2)
    for k, wk in enumerate(weights):
        if wk <= 0:
            continue
        if int(out_l.shape[0]) > k:
            idx = -(k + 1)
            out_l[idx] = (
                (out_l[idx].float() * (1.0 - wk) + b_f * wk)
                .clamp(0.0, 1.0)
                .to(dtype=out_l.dtype)
            )
        if int(out_r.shape[0]) > k:
            out_r[k] = (
                (out_r[k].float() * (1.0 - wk) + a_f * wk)
                .clamp(0.0, 1.0)
                .to(dtype=out_r.dtype)
            )
    log.info(
        "Segment continuity: micro seam bridge MAD=%.1f → w=%.2f/%.2f/%.2f",
        seam_mad,
        weights[0],
        weights[1],
        weights[2],
    )
    return out_l, out_r


def _hold_pop_weight(jump: float, *, base: float, jump_mad: float) -> float:
    """Scale bridge weight with jump size; cap below ghost zone."""
    span = 10.0
    t = max(0.0, min(1.0, (float(jump) - float(jump_mad)) / span))
    w_max = float(CONTINUITY_HOLD_POP_BRIDGE_WEIGHT_MAX)
    return float(base) + (w_max - float(base)) * t


def _find_hold_pop_jump(
    frames: torch.Tensor,
    *,
    hold_start: int,
    hold_end: int,
    jump_mad: float,
    lookahead: int,
) -> tuple[int, float] | None:
    """After a hold run, find the pop within ``lookahead`` pairs (incl. delayed)."""
    n = int(frames.shape[0])
    # hold_end is first index where pair mad may exceed hold; search from there.
    last = min(hold_end + max(0, int(lookahead)), n - 2)
    for k in range(hold_end, last + 1):
        jump = _frame_mad(frames[k], frames[k + 1])
        if jump >= jump_mad:
            return k, jump
    return None


def _apply_hold_pop_rewrite(
    out: torch.Tensor,
    *,
    i: int,
    jump_at: int,
    jump: float,
    bridge_weight: float,
    jump_mad: float,
    land_weight: float,
) -> float:
    """Unstick the held frame before a pop via low-freq pull (anti-ghost)."""
    del i, land_weight
    w = _hold_pop_weight(jump, base=bridge_weight, jump_mad=jump_mad)
    target = out[jump_at + 1]
    idx = jump_at
    t = 0.70 * w
    if CONTINUITY_HOLD_POP_LOWFREQ:
        out[idx] = _lowfreq_appearance_pull(
            out[idx],
            target,
            weight=t,
            blur=int(CONTINUITY_HOLD_POP_BLUR),
        )
    else:
        out[idx] = (
            (out[idx].float() * (1.0 - t) + target.float() * t)
            .clamp(0.0, 1.0)
            .to(dtype=out.dtype)
        )
    return w


def _break_hold_pop_window(
    frames: torch.Tensor,
    *,
    scan: int = CONTINUITY_HOLD_POP_SCAN,
    hold_mad: float = CONTINUITY_HOLD_MAD,
    jump_mad: float = CONTINUITY_HOLD_POP_JUMP_MAD,
    bridge_weight: float = CONTINUITY_HOLD_POP_BRIDGE_WEIGHT,
    land_weight: float = CONTINUITY_HOLD_POP_LAND_WEIGHT,
    lookahead: int = CONTINUITY_HOLD_POP_LOOKAHEAD,
    from_end: bool = False,
) -> torch.Tensor:
    """Rewrite hold→(mid)→pop runs with adaptive intra bridge + landing ease.

    00025 often holds, drifts for 1 mid frame (MAD≈5–8), then pops — looking
    only at the pair right after hold misses those. No cross-segment mix.
    """
    if (
        frames is None
        or int(frames.shape[0]) < 4
        or int(scan) < 3
        or bridge_weight <= 0
    ):
        return frames
    n = int(frames.shape[0])
    limit = min(int(scan), n - 2)
    if limit < 1:
        return frames

    def _scan_forward(src: torch.Tensor, start: int, end: int) -> torch.Tensor:
        out = src
        touched = False
        i = start
        while i < end:
            if _frame_mad(out[i], out[i + 1]) > hold_mad:
                i += 1
                continue
            j = i
            while (
                j + 1 < n - 1
                and j - i < 3
                and _frame_mad(out[j], out[j + 1]) <= hold_mad
            ):
                j += 1
            found = _find_hold_pop_jump(
                out,
                hold_start=i,
                hold_end=j,
                jump_mad=jump_mad,
                lookahead=lookahead,
            )
            if found is None:
                i = j + 1
                continue
            jump_at, jump = found
            if not touched:
                out = src.clone()
                touched = True
            w = _apply_hold_pop_rewrite(
                out,
                i=i,
                jump_at=jump_at,
                jump=jump,
                bridge_weight=bridge_weight,
                jump_mad=jump_mad,
                land_weight=land_weight,
            )
            log.info(
                "Segment continuity: hold→pop break %s @%d→%d jump=%.1f "
                "(hold≤%.1f w=%.2f)",
                "tail" if from_end else "opening",
                i,
                jump_at,
                jump,
                hold_mad,
                w,
            )
            i = jump_at + 1
        return out

    if from_end:
        start = max(0, n - 1 - limit)
        best = None  # (i, jump_at, jump)
        i = start
        while i < n - 2:
            if _frame_mad(frames[i], frames[i + 1]) > hold_mad:
                i += 1
                continue
            j = i
            while (
                j + 1 < n - 1
                and j - i < 3
                and _frame_mad(frames[j], frames[j + 1]) <= hold_mad
            ):
                j += 1
            found = _find_hold_pop_jump(
                frames,
                hold_start=i,
                hold_end=j,
                jump_mad=jump_mad,
                lookahead=lookahead,
            )
            if found is not None:
                jump_at, jump = found
                best = (i, jump_at, jump)
            i = j + 1
        if best is None:
            return frames
        best_i, best_jump_at, best_jump = best
        out = frames.clone()
        w = _apply_hold_pop_rewrite(
            out,
            i=best_i,
            jump_at=best_jump_at,
            jump=best_jump,
            bridge_weight=bridge_weight,
            jump_mad=jump_mad,
            land_weight=land_weight,
        )
        log.info(
            "Segment continuity: hold→pop break tail @%d→%d jump=%.1f "
            "(hold≤%.1f w=%.2f)",
            best_i,
            best_jump_at,
            best_jump,
            hold_mad,
            w,
        )
        return out

    return _scan_forward(frames, 0, limit)


def _ease_opening_spikes(
    frames: torch.Tensor,
    *,
    scan: int = CONTINUITY_SPIKE_SCAN,
    prev_mad: float = CONTINUITY_SPIKE_PREV_MAD,
    jump_mad: float = CONTINUITY_SPIKE_JUMP_MAD,
    weight: float = CONTINUITY_SPIKE_WEIGHT,
    land_weight: float = CONTINUITY_SPIKE_LAND_WEIGHT,
) -> torch.Tensor:
    """Ease soft-hesitation→hard-pop in the first few body frames (00024 seam100)."""
    if (
        frames is None
        or int(frames.shape[0]) < 3
        or int(scan) < 1
        or weight <= 0
    ):
        return frames
    n = int(frames.shape[0])
    limit = min(int(scan), n - 2)
    out = frames
    touched = False
    for i in range(limit):
        m0 = _frame_mad(out[i], out[i + 1])
        m1 = _frame_mad(out[i + 1], out[i + 2])
        if m0 > prev_mad or m1 < jump_mad:
            continue
        # Skip if already a classic hold (handled by hold→pop); only soft band.
        if m0 <= float(CONTINUITY_HOLD_MAD):
            continue
        if not touched:
            out = frames.clone()
            touched = True
        t = float(weight)
        out[i + 1] = (
            (out[i + 1].float() * (1.0 - t) + out[i + 2].float() * t)
            .clamp(0.0, 1.0)
            .to(dtype=out.dtype)
        )
        lw = float(land_weight)
        if lw > 0:
            out[i + 2] = (
                (out[i + 2].float() * (1.0 - lw) + out[i + 1].float() * lw)
                .clamp(0.0, 1.0)
                .to(dtype=out.dtype)
            )
        log.info(
            "Segment continuity: opening spike ease @%d (prev=%.1f jump=%.1f w=%.2f)",
            i,
            m0,
            m1,
            t,
        )
    return out


def _blend_tail_toward_next_first(
    prev: torch.Tensor,
    nxt: torch.Tensor,
    *,
    blend_frames: int = CONTINUITY_TAIL_SOFTEN_FRAMES,
    max_weight: float = CONTINUITY_TAIL_SOFTEN_WEIGHT,
) -> torch.Tensor:
    """Nudge prev[-n:] toward next[0] only — never lerp prev[-n:] into next[:n]."""
    if (
        prev is None
        or nxt is None
        or int(prev.shape[0]) <= 0
        or int(nxt.shape[0]) <= 0
        or int(blend_frames) <= 0
        or max_weight <= 0
    ):
        return prev
    first = nxt[0]
    if first.dim() == 4:
        first = first[0]
    if tuple(first.shape) != tuple(prev[-1].shape):
        first = fit_canvas(first.unsqueeze(0), int(prev.shape[2]), int(prev.shape[1]))[0]
    n = min(int(blend_frames), int(prev.shape[0]))
    out = prev.clone()
    first = first.to(device=out.device, dtype=out.dtype)
    for i in range(n):
        w = float(max_weight) * (1.0 - float(i) / float(n))
        out[-(i + 1)] = first * w + out[-(i + 1)] * (1.0 - w)
    return out


def prepend_continuity_source(
    clip_frames: torch.Tensor,
    prev_output: torch.Tensor | None,
    *,
    lock_px: int,
    width: int,
    height: int,
    settling_frames: int = 0,
) -> torch.Tensor:
    """Prepend prev-tail (+ optional settling burn-in) for SCAIL-aligned canvas.

    Layout: ``[lock from prev | settling ease | body | …]``.
    Settling eases prev→body[0] and is trimmed after decode — it absorbs the
    hold→pop that used to sit on exported body[0], without shifting the timeline.
    """
    if prev_output is None or lock_px <= 0 or int(clip_frames.shape[0]) <= 0:
        return clip_frames
    n = min(int(lock_px), int(prev_output.shape[0]))
    if n <= 0:
        return clip_frames
    body = fit_canvas(clip_frames, width, height)
    prefix = normalize_guide_luma_to_source(
        prev_output[-n:],
        body[0],
        width=width,
        height=height,
    )
    prefix = prefix.to(device=body.device, dtype=body.dtype)

    settle_n = max(0, int(settling_frames))
    settling = None
    if settle_n > 0:
        last = prefix[-1].to(device=body.device, dtype=body.dtype)
        first = body[0].to(device=body.device, dtype=body.dtype)
        frames = []
        for i in range(settle_n):
            # u: 0→1 across burn-in; start near prev, end near body[0].
            u = float(i + 1) / float(settle_n + 1)
            w_prev = 1.0 - u
            frames.append((last * w_prev + first * (1.0 - w_prev)).clamp(0.0, 1.0))
        settling = torch.stack(frames, dim=0).to(dtype=body.dtype)
        log.info(
            "Segment continuity: settling burn-in %df (prev→body[0], discarded on export)",
            settle_n,
        )

    n_bridge = min(int(CONTINUITY_SOURCE_BODY_BRIDGE), int(body.shape[0]))
    w0 = float(CONTINUITY_SOURCE_BODY_BRIDGE_WEIGHT)
    blur = int(CONTINUITY_SOURCE_BODY_BRIDGE_BLUR)
    if n_bridge > 0 and w0 > 0:
        body = body.clone()
        # Bridge from end of settling (or prefix) so body opening is soft.
        guide = settling[-1] if settling is not None else prefix[-1]
        guide = guide.to(device=body.device, dtype=body.dtype)
        for i in range(n_bridge):
            u = float(i) / float(max(1, n_bridge - 1)) if n_bridge > 1 else 0.0
            w = w0 * 0.5 * (1.0 + math.cos(math.pi * u))
            if blur >= 3:
                body[i] = _lowfreq_appearance_pull(body[i], guide, weight=w, blur=blur)
            else:
                body[i] = (guide * w + body[i] * (1.0 - w)).clamp(0.0, 1.0).to(
                    dtype=body.dtype
                )
        log.info(
            "Segment continuity: source body bridge %df (w0=%.2f blur=%d)",
            n_bridge,
            w0,
            blur,
        )

    parts = [prefix]
    if settling is not None:
        parts.append(settling)
    parts.append(body)
    out = torch.cat(parts, dim=0)
    log.info(
        "Segment continuity: prepended %d lock + %d settling + %d body source frame(s)",
        n,
        settle_n,
        int(body.shape[0]),
    )
    return out


def _normalize_context_latent_5d(latent: torch.Tensor) -> torch.Tensor:
    """Match official BerniniConditioning: keep VAE encode as ``[1, C, F, H, W]``."""
    if latent.ndim == 3:
        latent = latent.unsqueeze(0).unsqueeze(2)
    elif latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(
            f"Context latent must be 5D [1,C,F,H,W] (got {tuple(latent.shape)})"
        )
    if int(latent.shape[0]) != 1:
        raise ValueError(f"Context latent batch must be 1, got {tuple(latent.shape)}")
    return latent


def _latent_frame_count(pixel_frames: int) -> int:
    return max(1, (max(1, int(pixel_frames)) - 1) // 4 + 1)


def _encode_video_latent(vae, frames: torch.Tensor) -> torch.Tensor:
    """Encode [F,H,W,C] frames with ComfyUI native VAE → 5D [1,C,F,H,W]."""
    lat = vae.encode(frames[..., :3])
    return _normalize_context_latent_5d(lat)


def _encode_reference_latent(vae, frame: torch.Tensor, ref_max_size: int) -> torch.Tensor:
    """Encode a single reference frame (long-edge resize) with native VAE."""
    resized = fit_long_edge(frame[..., :3], int(ref_max_size))
    lat = vae.encode(resized)
    return _normalize_context_latent_5d(lat)


def encode_tail_clip(
    tail_clip: torch.Tensor,
    *,
    vae,
    width: int,
    height: int,
) -> torch.Tensor:
    """VAE-encode prev-tail clip for SCAIL lock (must already be Wan 4n+1 length)."""
    clip = fit_canvas(tail_clip, width, height)
    aligned = wan_align_frame_count(int(clip.shape[0]))
    if int(clip.shape[0]) > aligned:
        clip = clip[:aligned]
    elif int(clip.shape[0]) < aligned:
        # Prefer mirror from existing motion over freezing last frame.
        need = aligned - int(clip.shape[0])
        pad = clip[: min(need, int(clip.shape[0]))].flip(0)
        while int(pad.shape[0]) < need:
            pad = torch.cat([pad, clip.flip(0)], dim=0)
        clip = torch.cat([clip, pad[:need]], dim=0)
    return _encode_video_latent(vae, clip)


def apply_scail_prefix_to_latent(
    latent: dict[str, Any],
    tail_latent: torch.Tensor,
    overlap_pixel_frames: int,
) -> tuple[dict[str, Any], bool]:
    """Write prev-tail prefix into latent + noise_mask=0 (official KSampler path).

    Returns ``(latent, applied)``. Spatial mismatch is resized instead of silently
    skipping (silent skip made continuity look identical to OFF).
    """
    tail_latent = _normalize_context_latent_5d(tail_latent)
    samples = latent["samples"]
    if samples.ndim == 4:
        samples = samples.unsqueeze(0)
    _, _c, t_total, h, w = samples.shape
    th, tw = int(tail_latent.shape[3]), int(tail_latent.shape[4])
    if th != h or tw != w:
        log.warning(
            "Segment continuity: SCAIL spatial mismatch tail %dx%d vs latent %dx%d "
            "— resizing tail latent (was a silent skip)",
            th,
            tw,
            h,
            w,
        )
        # [1,C,F,H,W] → interpolate over H,W per temporal slice.
        b, c, f, _, _ = tail_latent.shape
        flat = tail_latent.reshape(b * c * f, 1, th, tw)
        flat = torch.nn.functional.interpolate(
            flat.float(), size=(h, w), mode="bilinear", align_corners=False
        )
        tail_latent = flat.to(dtype=tail_latent.dtype).reshape(b, c, f, h, w)
    aligned_pixels = wan_align_frame_count(int(overlap_pixel_frames))
    t_tail = min(
        int(tail_latent.shape[2]),
        _latent_frame_count(aligned_pixels),
        t_total,
    )
    if t_tail <= 0:
        return latent, False

    out = dict(latent)
    patched = samples.clone()
    patched[:, :, :t_tail] = tail_latent[:, :, :t_tail].to(
        device=patched.device, dtype=patched.dtype
    )
    noise_mask = torch.ones(
        (1, 1, t_total, h, w),
        dtype=patched.dtype,
        device=patched.device,
    )
    noise_mask[:, :, :t_tail] = 0.0
    # Feather inside the locked prefix (still mostly frozen).
    feather = min(int(CONTINUITY_LOCK_FEATHER_LATENT), max(0, t_tail - 1))
    if feather > 0:
        for i in range(feather):
            t = t_tail - feather + i
            alpha = float(CONTINUITY_LOCK_FEATHER_MASK) * float(i + 1) / float(feather)
            noise_mask[:, :, t] = alpha

    # Soft-unlock on free latents disabled (CONTINUITY_UNLOCK_* = 0): mixing prev
    # latent into free init + mask<1 left body under-denoised → 画面花.
    unlock_n = min(
        int(CONTINUITY_UNLOCK_FEATHER_LATENT),
        max(0, t_total - t_tail),
    )
    if unlock_n > 0 and t_tail > 0 and float(CONTINUITY_UNLOCK_INIT_BLEND) > 0:
        last_locked = patched[:, :, t_tail - 1 : t_tail]
        init_w = float(CONTINUITY_UNLOCK_INIT_BLEND)
        start_m = float(CONTINUITY_UNLOCK_START_MASK)
        for i in range(unlock_n):
            t = t_tail + i
            if unlock_n == 1:
                mask_i = start_m
                blend_i = init_w
            else:
                u = float(i) / float(unlock_n - 1)
                mask_i = start_m + (1.0 - start_m) * u
                blend_i = init_w * (1.0 - u)
            if blend_i > 0:
                patched[:, :, t] = (
                    patched[:, :, t] * (1.0 - blend_i)
                    + last_locked[:, :, 0] * blend_i
                )
            noise_mask[:, :, t] = min(1.0, max(0.0, mask_i))

    # Warm-start first free latents from last locked; mask stays 1.0 (full denoise).
    # Partial mask ramp (v13) under-denoised → 花屏; never revisit without A/B.
    warms = (
        float(CONTINUITY_FREE_LATENT_WARMSTART),
        float(CONTINUITY_FREE_LATENT_WARMSTART_2),
        float(CONTINUITY_FREE_LATENT_WARMSTART_3),
    )
    applied_w = []
    for offset, warm in enumerate(warms):
        t = t_tail + offset
        if t <= 0 or t >= t_total or warm <= 0:
            applied_w.append(0.0)
            continue
        patched[:, :, t] = (
            patched[:, :, t] * (1.0 - warm) + patched[:, :, t - 1] * warm
        )
        noise_mask[:, :, t] = 1.0
        applied_w.append(warm)

    out["samples"] = patched
    out["noise_mask"] = noise_mask
    log.info(
        "Segment continuity: SCAIL prefix lock %d latent frame(s) (%d px, "
        "lock_feather=%d, warmstart=%.2f/%.2f/%.2f, free_mask=1.0)",
        t_tail,
        aligned_pixels,
        feather,
        applied_w[0] if len(applied_w) > 0 else 0.0,
        applied_w[1] if len(applied_w) > 1 else 0.0,
        applied_w[2] if len(applied_w) > 2 else 0.0,
    )
    return out, True


def _context_latents_from_conditioning(conditioning) -> list[torch.Tensor]:
    for _tensor, payload in conditioning or []:
        if isinstance(payload, dict):
            streams = payload.get("context_latents")
            if streams:
                return list(streams)
    return []


def append_tail_reference_latent(
    streams: list[torch.Tensor],
    prev_output: torch.Tensor,
    *,
    vae,
    width: int,
    height: int,
    ref_max_size: int,
    n_frames: int = 1,
) -> list[torch.Tensor]:
    """Append last ``n_frames`` of prev as appearance reference latents."""
    if int(prev_output.shape[0]) <= 0:
        return streams
    n = min(max(1, int(n_frames)), int(prev_output.shape[0]))
    out = list(streams)
    for frame in fit_canvas(prev_output[-n:], width, height):
        lat = _encode_reference_latent(vae, frame.unsqueeze(0), ref_max_size)
        out.append(lat)
    log.info("Segment continuity: appended %d appearance reference latent(s)", n)
    return out


def apply_continuity_to_core_conditioning(
    positive,
    negative,
    *,
    prev_output: torch.Tensor,
    vae,
    width: int,
    height: int,
    ref_max_size: int,
    n_ref_frames: int = 1,
):
    """Append appearance ref only — motion handoff lives in prepended source + SCAIL."""
    import node_helpers

    if int(n_ref_frames) <= 0:
        return positive, negative
    streams = _context_latents_from_conditioning(positive)
    streams = append_tail_reference_latent(
        streams,
        prev_output,
        vae=vae,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
        n_frames=n_ref_frames,
    )
    payload = {"context_latents": streams}
    positive = node_helpers.conditioning_set_values(positive, payload)
    negative = node_helpers.conditioning_set_values(negative, payload)
    return positive, negative


def apply_scail_continuity_core(
    *,
    plan: DirectorPlan,
    seg: SegmentPlan,
    prev_output: torch.Tensor | None,
    positive,
    negative,
    vae,
    width: int,
    height: int,
    ref_max_size: int = 848,
    latent: dict[str, Any] | None = None,
    source_luma_ref: torch.Tensor | None = None,
) -> tuple[Any, Any, dict[str, Any] | None, str | None]:
    """SCAIL is Wan-only; MiniMax H3 uses last-frame handoff in executor."""
    del plan, seg, prev_output, vae, width, height, ref_max_size, source_luma_ref
    return positive, negative, latent, None


def _blend_opening_toward_last_frame(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    blend_frames: int = CONTINUITY_OPENING_LAST_FRAME_BLEND,
    max_weight: float = CONTINUITY_OPENING_LAST_FRAME_WEIGHT,
) -> torch.Tensor:
    """Nudge body head toward prev's last frame only — never replay prev[-n:]."""
    n = min(int(blend_frames), int(body.shape[0]))
    if n <= 0 or guide is None or int(guide.shape[0]) <= 0 or max_weight <= 0:
        return body
    out = body.clone()
    last = guide[-1:].to(device=out.device, dtype=out.dtype)
    if last.shape[1:] != out.shape[1:]:
        last = fit_canvas(last, int(out.shape[2]), int(out.shape[1]))
    for i in range(n):
        w = float(max_weight) * (1.0 - float(i) / float(n))
        out[i] = last[0] * w + out[i] * (1.0 - w)
    return out


def trim_decoded_for_continuity(
    decoded: torch.Tensor,
    *,
    prefix_trim: int,
    target_len: int,
    prev_tail: torch.Tensor | None = None,
    max_echo_skip: int = 0,
    segment_index: int | None = None,
) -> torch.Tensor:
    """Drop SCAIL overlap prefix and keep body length.

    Exposure is graded on the previous tail at concat (not on body opening —
    fading opening gain caused post-seam brightness pump / 一闪一闪 on 00010).
    Sequence crossfade against ``prev[-n:]`` is intentionally NOT used.
    """
    # Call-site compat; exposure / soften happen in concat_continuous_chunks.
    del max_echo_skip, prev_tail, segment_index
    prefix_trim = max(0, int(prefix_trim))
    out = decoded
    if prefix_trim > 0 and int(decoded.shape[0]) > prefix_trim:
        out = decoded[prefix_trim:]
    elif prefix_trim > 0 and int(decoded.shape[0]) == prefix_trim:
        out = decoded[:0]

    if target_len > 0 and int(out.shape[0]) > target_len:
        out = out[:target_len]
    return out


def continuity_merged_frame_count(plan: DirectorPlan) -> int:
    return int(plan.total_frames)


def concat_continuous_chunks(
    chunks: list[torch.Tensor],
    segments: list[SegmentPlan],
    plan: DirectorPlan,
) -> torch.Tensor:
    """Concatenate with exposure-only seam fix.

    Generation settling burn-in handles flash/pulse. Concat RGB morphs are OFF
    (00035: body0/hold→pop caused 拖影 and stutter pulses).
    """
    del segments
    if not chunks:
        raise ValueError("concat_continuous_chunks: no chunks")
    if not getattr(plan, "continuity_enabled", False) or len(chunks) < 2:
        return cat_frames_variable_size(chunks)
    fixed: list[torch.Tensor] = [chunks[0]]
    for i in range(1, len(chunks)):
        left = _unfreeze_held_tail(fixed[-1])
        if CONTINUITY_HOLD_POP_ON_TAIL:
            left = _break_hold_pop_window(left, from_end=True)
        body = _break_hold_pop_window(chunks[i], from_end=False)
        if float(CONTINUITY_SPIKE_WEIGHT) > 0:
            body = _ease_opening_spikes(body)
        body = _soften_body0_toward_prev(body, left)
        body = _additive_opening_luma(body, left)
        left, body = _micro_seam_bridge(left, body)
        fixed[-1] = left
        fixed.append(body)
    return cat_frames_variable_size(fixed)


def apply_cached_segment_continuity(
    chunk: torch.Tensor,
    seg: SegmentPlan,
    plan: DirectorPlan,
    completed_outputs: dict[int, torch.Tensor],
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    del seg, plan, completed_outputs, width, height
    return chunk
