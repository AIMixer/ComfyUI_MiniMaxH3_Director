"""Director-owned MiniMax H3 segment motion/audio continuation helpers.

Pins the previous segment's tail into the next segment as never-denoised
conditioning, then trims that prefix from decoded output. Inspired by the
community Motion Context approach; original Apache-2.0 code for this Director.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .h3_context_patches import (
    CTX_AUDIO_END_KEY,
    CTX_FRAME_KEY,
    ensure_layout_patch,
    ensure_payload_patch,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.h3_motion_context")

FPS = 24.0
AUDIO_HZ = 40.0
FRAME_RESCALE = 5.0 / 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

# Pixel windows that map to a whole number of VAE latent steps from cycle 0.
CONTEXT_FRAME_CHOICES = (5, 22, 39, 56)
DEFAULT_CONTEXT_FRAMES = 22
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)

CONTINUITY_TASK_KEYS = frozenset({"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"})
# v8: v7 + export audio cache + fps in fingerprint + trim hydrate on partial re-run.
# Single source of truth — imported by segment_cache.segment_cache_fingerprint.
CONTINUITY_PIPELINE_ID = "minimax_h3_motion_context_v8"
# Example workflow tested value (NikoDemon80): audio_context_length=24 with video=22.
DEFAULT_AUDIO_CONTEXT_FRAMES = 24


def snap_context_frames(raw: int | float | None) -> int:
    """Snap UI/plan overlap to a supported context window (official baseline 22)."""
    try:
        n = int(raw or DEFAULT_CONTEXT_FRAMES)
    except (TypeError, ValueError):
        n = DEFAULT_CONTEXT_FRAMES
    chosen = min(CONTEXT_FRAME_CHOICES, key=lambda g: (abs(g - n), -g))
    return int(chosen)


def recommended_context_frames(task_key: str | None = None) -> int:
    """Official Motion Context baseline (22) for all continuity tasks."""
    del task_key
    return DEFAULT_CONTEXT_FRAMES


def pixel_frames_for_latent_t(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def steps_for_frames(n: int) -> int | None:
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def step_offsets(latent_t: int) -> list[int]:
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _streams_from_latent(latent: dict) -> list[torch.Tensor]:
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            f"Director continuity: expected MiniMax H3 AV NestedTensor, got {type(samples)!r}"
        )
    if not parts:
        raise ValueError("Director continuity: AV latent has no streams")
    return parts


def video_from_latent(latent: dict) -> torch.Tensor:
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            f"Director continuity: expected video latent [B,C,T,H,W], got {tuple(video.shape)}"
        )
    return video


def _resize_frames(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import comfy.utils

    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def _phase_aligned_tail_start(
    total_steps: int, n_steps: int, end_frame: int | None
) -> tuple[int, int, int]:
    """Pick a 5-cycle-aligned step start whose pixel window ends at/before ``end_frame``.

    Returns ``(start_step, pin_end_px, gap_after_pin)``.

    When ``end_frame`` is None, use the absolute latent end (official Motion Context).
    Director passes the *exported* end so align() overshoot beyond the visible
    segment is never pinned into the next clip. ``gap_after_pin`` is how many
    exported frames sit *after* the pin window — those must be dropped from the
    previous export before concat, or the next clip's opening will echo them.
    """
    if n_steps > total_steps:
        raise ValueError(
            f"Director continuity: need {n_steps} latent steps, context has {total_steps}."
        )
    if end_frame is None:
        start = total_steps - n_steps
        if start % 5 != 0:
            raise RuntimeError(
                f"Director continuity: tail start cycle {start % 5} != 0; refusing shifted join."
            )
        pin_end = pixel_frames_for_latent_t(total_steps)
        return start, pin_end, 0

    end_limit = int(end_frame)
    best_start = None
    best_end_px = -1
    for start in range(0, total_steps - n_steps + 1, 5):
        start_px = pixel_frames_for_latent_t(start)
        end_px = start_px + pixel_frames_for_latent_t(n_steps)
        if end_px <= end_limit and end_px >= best_end_px:
            best_start = start
            best_end_px = end_px
    if best_start is None:
        raise RuntimeError(
            f"Director continuity: no phase-aligned {n_steps}-step window ending "
            f"at or before frame {end_limit}."
        )
    gap = max(0, end_limit - best_end_px)
    if gap > 0:
        log.info(
            "Director continuity: pin window ends %df before export end "
            "(phase align; export_end=%d, pin_end=%d) — prev export tail will be trimmed",
            gap,
            end_limit,
            best_end_px,
        )
    return best_start, best_end_px, gap


def _video_tail_blocks(
    latent: dict,
    n: int,
    *,
    end_frame: int | None = None,
) -> tuple[list[torch.Tensor], list[int], int, int, int]:
    """Return ``(blocks, offsets, covered, pin_end_px, gap_after_pin)``."""
    video = video_from_latent(latent)
    total = int(video.shape[2])
    steps = steps_for_frames(n)
    if steps is None:
        raise ValueError(
            f"Director continuity: {n} frames is not a whole number of latent steps "
            f"(use {', '.join(str(x) for x in CONTEXT_FRAME_CHOICES)})."
        )
    start, pin_end_px, gap = _phase_aligned_tail_start(total, steps, end_frame)
    covered = pixel_frames_for_latent_t(steps)
    if covered != n:
        raise RuntimeError(
            f"Director continuity: {steps} steps cover {covered} frames, expected {n}."
        )
    blocks = [video[:1, :, start + k : start + k + 1].clone() for k in range(steps)]
    return blocks, step_offsets(steps), covered, pin_end_px, gap


def _audio_tail_from_latent(
    latent: dict,
    a_frames: int,
    *,
    end_frame: int | None = None,
) -> tuple[torch.Tensor, int, float]:
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError("Director continuity: context latent has no audio stream.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            f"Director continuity: expected audio latent [B,C,2,T], got {tuple(audio.shape)}"
        )
    total_t = int(audio.shape[-1])
    frames = pixel_frames_for_latent_t(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (0.0 <= overhang < 1.0):
        log.warning(
            "Director continuity: unexpected audio grid (%d steps / %d frames); "
            "assuming no overhang.",
            total_t,
            frames,
        )
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        log.warning(
            "Director continuity: asked for %d audio steps, latent has %d; pinning all.",
            rt,
            total_t,
        )
        rt = total_t
    if rt < 1:
        raise ValueError("Director continuity: empty audio window")
    if end_frame is None:
        audio_end = total_t
    else:
        # Match the video pin window end (export end), not the sample overshoot.
        audio_end = int(round(float(end_frame) / float(FPS) * AUDIO_HZ))
        audio_end = max(rt, min(total_t, audio_end))
    audio_start = audio_end - rt
    if audio_start < 0:
        audio_start = 0
        rt = audio_end
    return audio[:1, ..., audio_start:audio_end].clone(), rt, float(overhang)


def _usable_context_audio(audio: dict | None) -> dict | None:
    """Return ``audio`` only when it has a non-empty waveform (resample-safe)."""
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor) or waveform.numel() <= 0:
        return None
    if waveform.ndim < 1 or int(waveform.shape[-1]) <= 0:
        return None
    return audio


def _encode_tail_audio(audio_vae, audio: dict, seconds: float) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio
    except ImportError:
        torchaudio = None
    usable = _usable_context_audio(audio)
    if usable is None:
        raise ValueError("Director continuity: empty context audio waveform")
    waveform = usable["waveform"]
    sr = int(usable.get("sample_rate") or getattr(audio_vae, "audio_sample_rate", 32000) or 32000)
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                f"Director continuity: context audio is {sr} Hz, VAE wants {vae_sr} Hz, "
                "and torchaudio is unavailable."
            )
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have >= want:
        waveform = waveform[..., have - want :]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def _existing_keyframes(positive) -> list[dict]:
    """Best-effort read of keyframes already on conditioning (e.g. fl2v last frame)."""
    try:
        if not positive:
            return []
        meta = positive[0][1] if isinstance(positive[0], (list, tuple)) else None
        if not isinstance(meta, dict):
            return []
        kfs = meta.get("minimax_keyframes") or []
        return [dict(kf) for kf in kfs if isinstance(kf, dict)]
    except Exception:
        return []


def apply_motion_context(
    positive,
    latent: dict,
    *,
    vae,
    context_length: int,
    context_latent: dict | None = None,
    context_frames: torch.Tensor | None = None,
    context_audio: dict | None = None,
    audio_vae=None,
    continue_audio: bool = True,
    keep_existing_keyframes: bool = True,
    context_end_frame: int | None = None,
    audio_context_length: int | None = None,
) -> tuple[Any, int, int]:
    """Inject previous-segment motion (and optional audio) into conditioning.

    Returns ``(positive, trim_frames, prev_export_trim_tail)``.

    ``trim_frames`` is the pinned head length to remove after decode.
    ``prev_export_trim_tail`` is how many frames to drop from the *previous*
    segment's export before concat (phase-align pin often ends a few frames
    before the export end; leaving them causes a visible ~5f echo at the seam).

    ``context_end_frame``: exclusive pixel index on the previous *sample*
    timeline to pin up to. With official sample semantics (no post-trim crop)
    this is usually ``None`` (absolute latent end). Kept for legacy overshoot
    caches.

    ``audio_context_length``: official example uses 24 with video context 22.
    """
    import node_helpers

    ensure_layout_patch()
    # r2v/v2v/rv2v already carry minimax_refs; stock payload overwrites keyframe
    # video latents unless coexistence merge is installed.
    ensure_payload_patch()
    context_length = snap_context_frames(context_length)
    if audio_context_length is None:
        audio_ctx = DEFAULT_AUDIO_CONTEXT_FRAMES
    else:
        try:
            audio_ctx = max(0, int(audio_context_length))
        except (TypeError, ValueError):
            audio_ctx = DEFAULT_AUDIO_CONTEXT_FRAMES

    video = video_from_latent(latent)
    width = int(video.shape[4]) * 16
    height = int(video.shape[3]) * 16
    frame_count = pixel_frames_for_latent_t(int(video.shape[2]))

    pin_audio_latent = context_latent
    if context_latent is not None:
        src = video_from_latent(context_latent)
        src_w, src_h = int(src.shape[4]) * 16, int(src.shape[3]) * 16
        if src_w == width and src_h == height:
            available = pixel_frames_for_latent_t(int(src.shape[2]))
            if context_end_frame is not None:
                available = min(available, max(0, int(context_end_frame)))
            video_src = "latent"
        elif context_frames is not None and int(context_frames.shape[0]) >= 1:
            # Refine upscale stores a larger AV latent; next segment's first pass
            # is still the Director canvas. Pin from the decoded export instead
            # of crashing — that export is what concat actually uses.
            log.warning(
                "Director continuity: context latent is %dx%d but this segment "
                "is %dx%d — pin from decoded frames (typical after Refine upscale).",
                src_w,
                src_h,
                width,
                height,
            )
            context_end_frame = None
            pin_audio_latent = None
            available = int(context_frames.shape[0])
            video_src = "pixels"
        else:
            raise ValueError(
                f"Director continuity: context latent is {src_w}x{src_h} but this "
                f"segment is {width}x{height}. Regenerate the previous segment at "
                "this resolution, or keep a decoded export for pixel pin."
            )
    else:
        if context_frames is None or int(context_frames.shape[0]) < 1:
            raise ValueError(
                "Director continuity: need previous segment latent or decoded frames."
            )
        available = int(context_frames.shape[0])
        video_src = "pixels"

    n = min(int(context_length), available)
    if n < 1:
        raise ValueError("Director continuity: no frames available to pin")
    run = next(g for g in VIDEO_RUN_GRID if g <= n)
    if run != n:
        log.warning(
            "Director continuity: %d frames off VAE grid; pinning last %d.", n, run
        )
        n = run
    if n >= frame_count:
        raise ValueError(
            f"Director continuity: cannot pin {n} frames into a {frame_count}-frame clip."
        )

    pin_end_px: int | None = None
    prev_export_trim_tail = 0
    if video_src == "latent":
        blocks, offsets, covered, pin_end_px, prev_export_trim_tail = _video_tail_blocks(
            context_latent, n, end_frame=context_end_frame
        )
        span = covered
    else:
        # Decoded frames are already the export — absolute tail is correct.
        tail = _resize_frames(context_frames[available - n :], width, height)
        enc = vae.encode(tail)
        if getattr(enc, "ndim", 0) != 5:
            raise ValueError(
                f"Director continuity: VAE encode returned shape "
                f"{tuple(getattr(enc, 'shape', ()))}, expected [B,C,T,H,W]."
            )
        steps = int(enc.shape[2])
        offsets = step_offsets(steps)
        covered = pixel_frames_for_latent_t(steps)
        if covered != n:
            raise RuntimeError(
                f"Director continuity: {n} frames encoded to {steps} steps covering "
                f"{covered}; VAE grid mismatch."
            )
        blocks = [enc[:, :, k : k + 1] for k in range(steps)]
        span = covered
        pin_end_px = available
        prev_export_trim_tail = 0

    ctx_keyframes = [
        {
            "resolved_frame_index": 0,
            CTX_FRAME_KEY: int(p),
            "latent": blk,
        }
        for p, blk in zip(offsets, blocks)
    ]

    merged = list(ctx_keyframes)
    if keep_existing_keyframes:
        for kf in _existing_keyframes(positive):
            # Drop stock first-frame at 0 — replaced by context head.
            if int(kf.get("resolved_frame_index", -1)) == 0 and CTX_FRAME_KEY not in kf:
                continue
            # Avoid duplicating director context markers.
            if CTX_FRAME_KEY in kf:
                continue
            # fl2v keeps the stock last_frame keyframe. Mark it with its own
            # resolved_frame_index (same pixel-index space as CTX_FRAME_KEY) so
            # _rewrite_keyframe_times can re-time it when references shift the
            # target origin, instead of skipping it and tripping the guard.
            rfi = int(kf.get("resolved_frame_index", -1))
            if rfi >= 0:
                kf = dict(kf)
                kf[CTX_FRAME_KEY] = rfi
            merged.append(kf)

    values: dict[str, Any] = {
        "minimax_keyframes": merged,
        "minimax_frame_count": frame_count,
    }
    out = node_helpers.conditioning_set_values(positive, values)

    context_audio = _usable_context_audio(context_audio)
    if continue_audio and pin_audio_latent is None and context_audio is None:
        log.warning(
            "Director continuity: previous export audio is empty; pinning video only."
        )
    if continue_audio and (pin_audio_latent is not None or context_audio is not None):
        # Official: audio window independent; 0 follows video span. Example WF uses 24.
        a_frames = int(audio_ctx) if audio_ctx > 0 else int(span)
        # Align audio pin end with the video pin window (not export overshoot).
        audio_end_limit = pin_end_px if pin_end_px is not None else context_end_frame
        if pin_audio_latent is not None:
            audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                pin_audio_latent, a_frames, end_frame=audio_end_limit
            )
        else:
            if audio_vae is None:
                raise ValueError(
                    "Director continuity: context_audio requires audio_vae "
                    "(or pass previous AV latent)."
                )
            audio_latent, ref_audio_t = _encode_tail_audio(
                audio_vae, context_audio, a_frames / float(FPS)
            )
            overhang = 0.0
        end_frame = float(span) + float(overhang) / FRAME_RESCALE
        end_coord = round(FRAME_RESCALE * end_frame)
        end_frame = end_coord / FRAME_RESCALE
        audio_ref = {
            "kind": "audio",
            "ref_audio_t": ref_audio_t,
            "audio_latent": audio_latent,
            CTX_AUDIO_END_KEY: end_frame,
        }
        out = node_helpers.conditioning_set_values(
            out, {"minimax_refs": [audio_ref]}, append=True
        )
        log.info(
            "Director continuity: pinned %d video frames (%s) + %d audio steps"
            "%s%s",
            span,
            video_src,
            ref_audio_t,
            f" (context_end={context_end_frame})" if context_end_frame is not None else "",
            f", trim_prev_export={prev_export_trim_tail}f" if prev_export_trim_tail else "",
        )
    else:
        log.info(
            "Director continuity: pinned %d video frames (%s), audio off%s%s",
            span,
            video_src,
            f" (context_end={context_end_frame})" if context_end_frame is not None else "",
            f", trim_prev_export={prev_export_trim_tail}f" if prev_export_trim_tail else "",
        )

    return out, int(span), int(prev_export_trim_tail)


def trim_context_prefix(
    images: torch.Tensor,
    audio: dict | None,
    trim_frames: int,
    *,
    fps: float = FPS,
    match_tail: bool = True,
) -> tuple[torch.Tensor, dict | None]:
    """Remove pinned head from decoded images/audio; optionally match audio duration."""
    trim = max(0, int(trim_frames))
    if trim > 0:
        if int(images.shape[0]) <= trim:
            raise ValueError(
                f"Director continuity: cannot trim {trim} frames from "
                f"{int(images.shape[0])}-frame decode."
            )
        images = images[trim:]
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return images, audio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 32000)
    drop = int(round((trim / float(fps)) * sr)) if trim > 0 else 0
    if drop > 0 and int(waveform.shape[-1]) > drop:
        waveform = waveform[..., drop:]
    if match_tail:
        want = int(round((int(images.shape[0]) / float(fps)) * sr))
        if int(waveform.shape[-1]) > want:
            waveform = waveform[..., :want]
    return images, {"waveform": waveform, "sample_rate": sr}


def trim_export_tail(
    images: torch.Tensor,
    audio: dict | None,
    trim_frames: int,
    *,
    fps: float = FPS,
) -> tuple[torch.Tensor, dict | None]:
    """Drop trailing frames from a previous export so it ends at the pin window."""
    trim = max(0, int(trim_frames))
    if trim <= 0:
        return images, audio
    keep = int(images.shape[0]) - trim
    if keep < 1:
        raise ValueError(
            f"Director continuity: cannot drop {trim} tail frames from "
            f"{int(images.shape[0])}-frame export."
        )
    images = images[:keep]
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return images, audio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 32000)
    want = int(round((keep / float(fps)) * sr))
    if int(waveform.shape[-1]) > want:
        waveform = waveform[..., :want]
    return images, {"waveform": waveform, "sample_rate": sr}


def generation_frame_budget(visible_frames: int, context_frames: int) -> tuple[int, int]:
    """Return ``(sample_length, trim_frames)`` for Director continuity.

    Director contract: UI segment duration == exported frames.

    Standalone Motion Context sets ``length`` to the sample and delivers
    ``length - context`` (shorter than the UI seconds). That produced the
    27s-vs-30s result. Here we instead:

    1. ``sample = align(visible + context)`` so the pin fits in the head
    2. Trim ``context`` frames after decode
    3. Keep exactly ``visible`` frames for export
    4. Next pin uses ``context_end_frame = trim + visible`` (not the sample
       absolute end, which includes align overshoot beyond the export)
    5. If phase-align places the pin a few frames before that export end,
       drop those frames from the previous export before concat (v7)
    """
    from .frame_align import minimax_align_frame_count

    visible = minimax_align_frame_count(max(5, int(visible_frames)))
    ctx = snap_context_frames(context_frames) if context_frames else 0
    if ctx <= 0:
        return visible, 0
    sample = minimax_align_frame_count(visible + ctx)
    if ctx >= sample:
        raise ValueError(
            f"Director continuity: context {ctx}f must be smaller than sample "
            f"length {sample}f."
        )
    return sample, ctx


def handoff_end_frame(*, trim_frames: int, export_frames: int) -> int:
    """Sample-timeline pixel index where the exported segment ends (exclusive)."""
    return max(0, int(trim_frames)) + max(0, int(export_frames))
