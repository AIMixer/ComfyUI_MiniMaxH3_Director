"""Seed Director segment caches from a finished video (video-as-segment). Apache-2.0.

Purpose
-------
Let a segment whose pixels already exist -- an mp4 exported by an earlier run,
another project's output, or footage edited elsewhere -- be used AS that
segment, so that later segments continue from it instead of it being
re-rendered.

Why this is only a *seeding* problem
------------------------------------
The Director already has both halves of "continue a finished video":

1. A segment whose disk cache exists is treated as finished and is **not
   sampled** -- ``executor_core`` loads it into ``completed_outputs[seg.index]``
   and its AV latent into ``completed_av_latents[seg.index]``.
2. The AV latent of the previous segment is carried into the next one as the
   continuation seed -- ``h3_latent_continue.apply_latent_continue``.

What is missing is only a way to *fill* that cache from a video rather than
from the sampler. This module does exactly that, using the same video/audio
VAEs the run is already using.

Timeline field
--------------
Per segment, in ``timeline_data``::

    "sourceVideo": {"videoFile": "out.mp4", "type": "input",
                    "subfolder": "minimax_seg_export/20261016_004547"}

``videoFile`` may also be given as ``fileName``. A finished segment must not
also be selected for sampling; leave it out of ``runSelection``.

Never raises: any segment that cannot be seeded is simply left uncached, so the
Director falls back to sampling it as usual.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from ..lib.audio_io import frames_to_audio_samples, load_reference_audio
from ..lib.video_io import load_video_resampled, resolve_video_path
from .h3_motion_context import _repack_av_streams
from .segment_cache import save_segment_cache

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.seed_from_video")

log.info("[Director] seed_from_video module loaded (video-as-segment enabled)")

class _SkipLatent(Exception):
    """Internal: raised to skip the VAE encode, caught immediately below."""


SOURCE_VIDEO_KEYS = ("sourceVideo", "source_video")
DEFAULT_AUDIO_SR = 32000


def _read_source_video(seg) -> dict | None:
    """Return the segment's source-video dict, or None when it has none."""
    for key in SOURCE_VIDEO_KEYS:
        raw = getattr(seg, key, None)
        if isinstance(raw, dict) and (raw.get("videoFile") or raw.get("fileName")):
            return raw
    return None


def _resample_audio_to_vae(audio: dict, vae_sr: int, want_samples: int) -> torch.Tensor | None:
    """Fit a decoded AUDIO dict to ``want_samples`` at the audio VAE's rate."""
    if not isinstance(audio, dict):
        return None
    wave = audio.get("waveform")
    if not torch.is_tensor(wave) or wave.numel() == 0:
        return None
    sr = int(audio.get("sample_rate") or DEFAULT_AUDIO_SR)
    if sr != vae_sr:
        try:
            import torchaudio.functional as AF

            wave = AF.resample(wave, sr, vae_sr)
        except Exception as exc:  # noqa: BLE001
            log.warning("seed-from-video: audio resample failed (%s); using raw rate", exc)
    have = int(wave.shape[-1])
    if want_samples > 0 and have > want_samples:
        wave = wave[..., :want_samples]
    elif want_samples > 0 and have < want_samples:
        pad = want_samples - have
        wave = torch.nn.functional.pad(wave, (0, pad))
    return wave


def seed_segment_from_video(seg, plan, node_id: str, vae, audio_vae, need_latent: bool = True) -> bool:
    """Decode one segment's source video and write its cache. Returns success."""
    src = _read_source_video(seg)
    if src is None:
        return False

    n = int(seg.frame_count)
    if n <= 0:
        log.warning("seed-from-video: segment %d has no frames; skipped", seg.index + 1)
        return False

    try:
        path = resolve_video_path(src)
    except Exception as exc:  # noqa: BLE001
        log.warning("seed-from-video: segment %d cannot resolve %r (%s)", seg.index + 1, src, exc)
        return False
    if not path or not os.path.exists(path):
        log.warning("seed-from-video: segment %d file not found: %r", seg.index + 1, path)
        return False

    fps = float(getattr(plan, "frame_rate", 24.0) or 24.0)
    out_w = int(getattr(plan, "width", 0) or 0)
    out_h = int(getattr(plan, "height", 0) or 0)

    try:
        frames = load_video_resampled(
            path,
            fps,
            list(range(n)),
            storage_width=out_w or None,
            storage_height=out_h or None,
            hold_past_eof=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("seed-from-video: segment %d decode failed (%s)", seg.index + 1, exc)
        return False

    if not torch.is_tensor(frames) or frames.numel() == 0:
        log.warning("seed-from-video: segment %d decoded nothing", seg.index + 1)
        return False

    frames = frames[:n, ..., :3].contiguous().float()
    if frames.shape[0] < n:  # top up short decodes with the last frame
        tail = frames[-1:].repeat(n - frames.shape[0], 1, 1, 1)
        frames = torch.cat([frames, tail], dim=0)
    if out_w and out_h and (frames.shape[2] != out_h or frames.shape[1] != out_w):
        try:
            moved = frames.movedim(-1, 1)
            moved = torch.nn.functional.interpolate(
                moved, size=(out_h, out_w), mode="bilinear", align_corners=False
            )
            frames = moved.movedim(1, -1).contiguous()
        except Exception as exc:  # noqa: BLE001
            log.warning("seed-from-video: segment %d resize failed (%s)", seg.index + 1, exc)

    # ---- AV latent (video stream + audio stream) ----
    # Only needed when a later segment will be *sampled* and must continue from
    # this one. In assemble mode nothing is sampled, so skip the (expensive) VAE
    # encode entirely -- the pixels and the export audio are all that's required.
    av_latent: dict[str, Any] | None = None
    if not need_latent:
        log.info(
            "Seed-from-video: segment %d needs no continuity seed -> skipping VAE encode",
            seg.index + 1,
        )
    try:
        if not need_latent:
            raise _SkipLatent()
        v_lat = vae.encode(frames)
        streams: list[Any] = []
        if isinstance(v_lat, dict) and "samples" in v_lat:
            streams.append(v_lat["samples"])
        elif torch.is_tensor(v_lat):
            streams.append(v_lat)

        if audio_vae is not None:
            vae_sr = int(getattr(audio_vae, "audio_sample_rate", DEFAULT_AUDIO_SR) or DEFAULT_AUDIO_SR)
            audio = load_reference_audio(path)
            want = frames_to_audio_samples(n, fps, vae_sr)
            wave = _resample_audio_to_vae(audio, vae_sr, want)
            if wave is not None:
                a_lat = audio_vae.encode(wave[:1].movedim(1, -1))
                samples = a_lat.get("samples") if isinstance(a_lat, dict) else a_lat
                if torch.is_tensor(samples):
                    streams.append(samples)

        if streams:
            av_latent = {"samples": _repack_av_streams(streams)}
    except _SkipLatent:
        av_latent = None
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "seed-from-video: segment %d latent encode failed (%s); "
            "seeding pixels only (continuity will fall back)",
            seg.index + 1,
            exc,
        )
        av_latent = None

    # ---- export audio (decoded, not latent) ----
    export_audio: dict | None = None
    try:
        export_audio = load_reference_audio(path)
    except Exception:  # noqa: BLE001
        export_audio = None

    handoff = {
        "export_frames": n,
        "official_mc_length": False,
        "sample_frames": n,
        "trim_frames": 0,
    }

    try:
        save_segment_cache(
            node_id,
            seg,
            plan,
            frames,
            av_latent=av_latent,
            handoff=handoff,
            audio=export_audio,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("seed-from-video: segment %d cache write failed (%s)", seg.index + 1, exc)
        return False

    log.info(
        "Seed-from-video: segment %d/%d seeded from %s (%d frames%s)",
        seg.index + 1,
        int(getattr(plan, "segment_count", 0) or 0),
        os.path.basename(path),
        n,
        ", +av_latent" if av_latent is not None else ", pixels only",
    )
    return True


def seed_segments_from_source_videos(plan, node_id: str | None, vae, audio_vae) -> int:
    """Seed every segment that declares a source video. Returns how many were seeded."""
    if not node_id or vae is None:
        log.info("Seed-from-video: skipped (node_id=%r vae=%r)", node_id, vae is not None)
        return 0
    segs = getattr(plan, "segments", None) or []
    with_src = sum(1 for s in segs if _read_source_video(s) is not None)
    log.info(
        "Seed-from-video: scanning %d segment(s), %d declare sourceVideo",
        len(segs),
        with_src,
    )
    print(f"[Director] 视频播种扫描：{len(segs)} 段，其中 {with_src} 段带 sourceVideo", flush=True)
    segs_all = list(getattr(plan, "segments", None) or [])
    run_indices = getattr(plan, "run_indices", None)   # None = 全跑
    assemble = bool(getattr(plan, "assemble_only", False)) or (
        run_indices is not None and len(run_indices) == 0
    )

    def _will_sample(idx: int) -> bool:
        """True when segment ``idx`` is going to be generated (not from cache)."""
        if run_indices is None:
            return True
        return idx in run_indices

    seeded = 0
    for pos, seg in enumerate(segs_all):
        try:
            nxt = pos + 1
            need_latent = (not assemble) and nxt < len(segs_all) and _will_sample(nxt)
            if seed_segment_from_video(seg, plan, node_id, vae, audio_vae, need_latent=need_latent):
                seeded += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Seed-from-video: segment %s skipped (%s)", getattr(seg, "index", "?"), exc)
    return seeded


ASSEMBLE_KEYS = ("assembleOnly", "assemble_only", "assemble")


def _seg_has_source(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    for key in SOURCE_VIDEO_KEYS:
        v = raw.get(key)
        if isinstance(v, dict) and (v.get("videoFile") or v.get("fileName")):
            return True
    return False


def enable_assemble_mode_when_nothing_selected(timeline_data: str) -> tuple[str, bool]:
    """Empty run selection == "concatenate what already exists".

    ``runSelectEnabled`` on with an **empty** ``runSelection`` means the user does
    not want anything sampled: they want the finished material stitched together.
    Every segment is then served from cache (the ones that carry a
    ``sourceVideo`` are seeded by this module), the sampler never fires, and the
    export is a straight concatenation.

    A **non-empty** selection keeps the normal behaviour, so selecting a segment
    still re-runs it -- which is also how a segment's seed gets refreshed from its
    (possibly changed) source video.

    An explicit ``assembleOnly`` flag forces it regardless of the selection.
    Returns ``(timeline_data, assemble)``. Never raises.
    """
    try:
        tl = json.loads(timeline_data) if isinstance(timeline_data, str) else dict(timeline_data or {})
    except Exception:  # noqa: BLE001
        return timeline_data, False
    if not isinstance(tl, dict):
        return timeline_data, False

    segs = tl.get("segments") or []
    if not segs:
        return timeline_data, False

    enabled = bool(tl.get("runSelectEnabled") or tl.get("run_select_enabled"))
    raw = tl.get("runSelection")
    if raw is None:
        raw = tl.get("run_selection")
    selected = [i for i in (raw or []) if isinstance(i, int) or str(i).isdigit()]

    explicit = any(bool(tl.get(k)) for k in ASSEMBLE_KEYS)
    nothing_selected = enabled and not selected
    if not (explicit or nothing_selected):
        return timeline_data, False

    # 关键：runSelection 是「要生成哪几段」，不是「要保留哪几段」。
    # 拼接模式必须保持它为空 —— 一段都不采样，全部从缓存取。
    tl["runSelectEnabled"] = True
    tl["runSelection"] = []
    tl["assembleOnly"] = True
    log.info(
        "Assemble mode (%s): %d segment(s), nothing selected -> concatenating, no sampling",
        "explicit flag" if explicit else "empty run selection",
        len(segs),
    )
    return json.dumps(tl, ensure_ascii=False), True
