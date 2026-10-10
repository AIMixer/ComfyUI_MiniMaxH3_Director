"""Freeze the AV latent's audio stream to the source track (audioMode=source).

``audioMode=source`` previously only changed the **export** path: the exported
mp4 muxes the original timeline / reference audio instead of decoding the
model-generated audio, but generation still denoised the audio stream freely —
the model invented its own track, so lip movement drifted off the real song
(the r2v music-video use case).

With the freeze, source mode also drives **generation** with that same audio:

1. the segment's audio is encoded with the audio VAE (timeline slice first,
   then ``<Audio J>`` reference cards — the same priority the source-mode mux
   uses, so the frozen audio always matches what ends up in the mp4);
2. the encoded latent is written into the AV latent's audio stream;
3. a per-stream ``noise_mask`` (video = 1 → generate, audio = 0 → keep clean)
   pins those tokens at every sampler step. The joint audio-video attention
   then aligns mouth motion to the real track.

The frozen latent carries a small marker dict
(:data:`AUDIO_FREEZE_MARK_KEY`) that survives disk cache, refine AV re-joins
and SelfLift repacks; the sampler rebuilds the keep-mask from it on every
sampling stage (see ``refresh_frozen_audio_mask``).

Opt-out: set the environment variable ``MINIMAX_H3_AUDIO_FREEZE=0``. The
freeze state is part of the first-pass cache fingerprint, so toggling it
re-samples the affected segments exactly once.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.audio_freeze")

# Marker stored inside the latent dict. Plain dict of ints/strs: survives
# torch.save (segment cache), dict copies (refine joins) and NestedTensor
# repacks. Contains no tensors on purpose.
AUDIO_FREEZE_MARK_KEY = "h3_audio_freeze_mark"

FREEZE_VERSION = 2

# Audio-latent ticks per video frame when comfy's constant is unavailable
# (40 ticks/s at 24 fps = 5/3 — matches FRAME_PER_TOKEN=4 * FRAME_RESCALE=5/3
# in comfy.ldm.minimax.model).
DEFAULT_FRAME_RESCALE = 5.0 / 3.0

# Encoded segment audio by PCM signature, kept so the sampler can RESTORE the
# frozen window content, not just its mask. Why the restore is needed: comfy's
# noise_mask machinery treats ``latent["samples"]`` as the clean x0 — it
# overrides the model's prediction with it (``out = out*mask + latent*(1-mask)``)
# and reinjects it via ``scale_latent_inpaint`` at every step. The SelfLift /
# refine resume path rewrites the whole AV latent into the sigma-resume
# representation (``inverse_noise_scaling`` = ``x / (1 - sigma)`` ≈ x13 at
# sigma≈0.92, with the low-res pass noise mixed in), so a mask-only refresh
# would pin that mangled audio as the "clean" conditioning of every high-res
# pass. Restoring the window content keeps the frozen track bit-identical
# through every sampling stage. Small (≈0.4 MB per segment), CPU-resident.
_Z_CACHE: dict[str, Any] = {}
_Z_CACHE_MAX = 8


def _cache_z(sig: str, z) -> None:
    """Remember the encoded window for refresh-time content restore (LRU-ish)."""
    try:
        _Z_CACHE.pop(str(sig), None)
        _Z_CACHE[str(sig)] = z
        while len(_Z_CACHE) > _Z_CACHE_MAX:
            _Z_CACHE.pop(next(iter(_Z_CACHE)))
    except Exception:
        pass


def _cached_z(sig: str):
    try:
        return _Z_CACHE.get(str(sig))
    except Exception:
        return None


def _env_enabled() -> bool:
    raw = str(os.environ.get("MINIMAX_H3_AUDIO_FREEZE", "")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def freeze_requested(plan) -> tuple[bool, str]:
    """Whether the freeze should act on this plan: ``(active, detail)``.

    Active only when the plan's audio mode resolves to ``source`` (the mux
    keeps the original audio). Every other mode is untouched.
    """
    if not _env_enabled():
        return False, "disabled by MINIMAX_H3_AUDIO_FREEZE=0"
    from .audio_export import AUDIO_MODE_SOURCE, resolve_audio_mode

    try:
        mode = str(resolve_audio_mode(plan) or "")
    except Exception as exc:
        return False, f"resolve_audio_mode failed: {exc}"
    if mode != AUDIO_MODE_SOURCE:
        return False, f"audioMode={mode or 'generate'} (needs source)"
    return True, "source mode"


def fingerprint_flag(plan) -> str | None:
    """Fingerprint token for the freeze state (None = freeze inactive)."""
    ok, _detail = freeze_requested(plan)
    return f"source-v{FREEZE_VERSION}" if ok else None


def frame_rescale() -> float:
    """Audio-latent ticks per video frame (comfy FRAME_RESCALE, fallback 5/3)."""
    try:
        from comfy.ldm.minimax.model import FRAME_RESCALE

        value = float(FRAME_RESCALE)
        if value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_FRAME_RESCALE


# ---------------------------------------------------------------------------
# audio source selection (mirrors the source-mode mux priority)
# ---------------------------------------------------------------------------


def timeline_pcm_for_segment(plan, seg) -> dict[str, Any] | None:
    """Timeline song slice for [seg.start_frame, seg.end_frame) — the mux source."""
    start = int(getattr(seg, "start_frame", 0) or 0)
    end = int(getattr(seg, "end_frame", 0) or 0)
    if end <= start:
        return None
    try:
        from ..lib.audio_io import extract_timeline_audio

        return extract_timeline_audio(
            getattr(plan, "raw", None) or {},
            start,
            end,
            float(getattr(plan, "frame_rate", 24.0) or 24.0),
            audio_cache=getattr(plan, "audio_decode_cache", None),
        )
    except Exception as exc:
        log.debug("Audio freeze: timeline audio extraction failed: %s", exc)
        return None


def _first_ref_pcm(ref_audios, ref_video_audios):
    """First usable PCM: standalone ref audios first, then ref-video soundtracks."""
    for source in (ref_audios, ref_video_audios):
        if not isinstance(source, dict):
            continue
        try:
            keys = sorted(
                source.keys(),
                key=lambda k: (
                    int(str(k).rsplit("_", 1)[-1])
                    if str(k).rsplit("_", 1)[-1].isdigit()
                    else 1 << 30
                ),
            )
        except Exception:
            keys = list(source.keys())
        for key in keys:
            audio = source.get(key)
            if isinstance(audio, dict) and audio.get("waveform") is not None:
                return audio
    return None


def segment_freeze_pcm(plan, seg, ref_audios, ref_video_audios):
    """Pick the audio to freeze: timeline slice → ref audios → ref-video audio.

    Same priority as ``prepare_segment_audio_for_file_export`` uses for the
    source-mode mux, so the frozen tokens always match the muxed track.
    Returns ``(pcm_dict, source_label)``.
    """
    pcm = timeline_pcm_for_segment(plan, seg)
    if isinstance(pcm, dict) and pcm.get("waveform") is not None:
        return pcm, "timeline audio"
    pcm = _first_ref_pcm(ref_audios, ref_video_audios)
    if pcm is not None:
        return pcm, "reference audio"
    return None, ""


def pcm_signature(audio: dict) -> str:
    """Short content hash of the PCM (goes into the freeze marker)."""
    try:
        wave = audio["waveform"]
        if not torch.is_tensor(wave):
            return ""
        digest = hashlib.sha1()
        digest.update(
            wave.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
        )
        digest.update(str(int(audio.get("sample_rate") or 0)).encode())
        return digest.hexdigest()[:16]
    except Exception:
        return ""


def encode_freeze_audio(audio_vae, audio: dict):
    """Official MiniMax H3 encode: resample → audio VAE → ``[1, 32, 2, T]``."""
    import comfy.audio

    waveform = audio["waveform"]
    if not torch.is_tensor(waveform):
        return None, 0
    sr = int(audio.get("sample_rate") or 0)
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000) or 32000)
    if sr and sr != vae_sr:
        waveform = comfy.audio.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    if not torch.is_tensor(z):
        z = getattr(z, "samples", z)
    if not torch.is_tensor(z) or z.ndim != 4:
        return None, 0
    return z.detach().to("cpu"), int(z.shape[-1])


# ---------------------------------------------------------------------------
# AV latent helpers (NestedTensor of 5D video + 4D audio streams)
# ---------------------------------------------------------------------------


def _av_streams(latent: dict):
    """Return (video, audio, nested, video_squeezed, audio_squeezed) or None.

    Batch dims are normalized to 5D video / 4D audio; the squeeze flags say
    whether an ``unsqueeze`` was applied so the streams can be packed back
    with their original dims.
    """
    samples = latent.get("samples")
    if samples is None or torch.is_tensor(samples):
        return None
    parts = None
    nested = bool(getattr(samples, "is_nested", False))
    if nested and hasattr(samples, "unbind"):
        try:
            parts = list(samples.unbind())
        except Exception:
            return None
    elif isinstance(samples, (tuple, list)) and len(samples) >= 2:
        parts = list(samples[:2])
    if not parts or len(parts) < 2:
        return None
    video, audio = parts[0], parts[1]
    if not torch.is_tensor(video) or not torch.is_tensor(audio):
        return None
    video_squeezed = video.ndim == 4
    audio_squeezed = audio.ndim == 3
    if video_squeezed:
        video = video.unsqueeze(0)
    if audio_squeezed:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        return None
    return video, audio, nested, video_squeezed, audio_squeezed


def _nested_pack(video, audio, template):
    try:
        import comfy.nested_tensor

        return comfy.nested_tensor.NestedTensor((video, audio))
    except Exception:
        if template is not None and hasattr(template, "tensors"):
            try:
                return type(template)((video, audio))
            except Exception:
                pass
        return (video, audio)


def _repack_samples(
    latent: dict, video, audio, *, video_squeezed: bool, audio_squeezed: bool
) -> None:
    if video_squeezed:
        video = video.squeeze(0)
    if audio_squeezed:
        audio = audio.squeeze(0)
    latent["samples"] = _nested_pack(video, audio, latent.get("samples"))


def _unbind_noise_mask(mask):
    """Split a latent noise_mask into (video_part, audio_part); either may be None.

    A plain (non-nested) mask applies to the video stream only.
    """
    if mask is None:
        return None, None
    if torch.is_tensor(mask):
        return mask, None
    parts = None
    if getattr(mask, "is_nested", False) and hasattr(mask, "unbind"):
        try:
            parts = list(mask.unbind())
        except Exception:
            parts = None
    elif isinstance(mask, (tuple, list)):
        parts = list(mask)
    if not parts:
        return None, None
    video = parts[0] if len(parts) > 0 else None
    audio = parts[1] if len(parts) > 1 else None
    return video, audio


def _install_noise_mask(latent: dict, video, audio, *, offset: int, covered: int) -> None:
    """Per-stream keep mask: video untouched, audio [offset, offset+covered) pinned.

    Composes with any existing mask (e.g. a continuity prefix lock): the audio
    part is multiplied, the video part is kept as-is.
    """
    audio_t = int(audio.shape[-1])
    ours = torch.ones((1, 1, 1, audio_t), dtype=torch.float32, device=audio.device)
    end = min(audio_t, offset + covered)
    if end > offset:
        ours[..., offset:end] = 0.0

    exist_video, exist_audio = _unbind_noise_mask(latent.get("noise_mask"))
    if torch.is_tensor(exist_audio):
        try:
            audio_mask = exist_audio.to(torch.float32) * ours
        except Exception:
            audio_mask = ours
    else:
        audio_mask = ours
    audio_mask = audio_mask.to(device=audio.device, dtype=torch.float32).contiguous()

    if torch.is_tensor(exist_video):
        video_mask = exist_video.to(device=audio.device, dtype=torch.float32).contiguous()
    else:
        video_mask = torch.ones(
            (1, 1, int(video.shape[2]), 1, 1),
            dtype=torch.float32,
            device=audio.device,
        )
    latent["noise_mask"] = _nested_pack(video_mask, audio_mask, latent.get("samples"))


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def apply_first_pass_audio_freeze(
    plan,
    seg,
    *,
    latent,
    audio_vae,
    ref_audios=None,
    ref_video_audios=None,
    trim_frames: int = 0,
) -> str | None:
    """Freeze this segment's audio tokens onto the source track. First pass.

    Called after conditioning + continuity pinning, right before sampling.
    ``trim_frames`` (pinned context prefix) shifts the audio window so the
    frozen song position matches the video position. Returns a short report
    note, or None when nothing was frozen (wrong mode, no audio, unexpected
    latent — always a graceful no-op, never breaks a run).
    """
    ok, _detail = freeze_requested(plan)
    if not ok:
        return None
    if not isinstance(latent, dict) or audio_vae is None:
        return None
    av = _av_streams(latent)
    if av is None:
        log.info(
            "Audio freeze: sampling latent is not an AV pair (%s); segment left unfrozen.",
            type(latent.get("samples")).__name__,
        )
        return None
    video, audio, _nested, video_squeezed, audio_squeezed = av

    # Already frozen (first-pass cache hit / re-entry): the sampler refreshes
    # the mask from the marker, no re-encode needed.
    mark = latent.get(AUDIO_FREEZE_MARK_KEY)
    if isinstance(mark, dict) and int(mark.get("covered_ticks") or 0) > 0:
        return None

    pcm, source_label = segment_freeze_pcm(plan, seg, ref_audios, ref_video_audios)
    if pcm is None:
        log.info(
            "Audio freeze: no usable segment audio (no timeline song, no ref audio); "
            "segment left unfrozen."
        )
        return None
    z, z_t = encode_freeze_audio(audio_vae, pcm)
    if z is None or z_t <= 0:
        log.warning(
            "Audio freeze: %s could not be encoded by the audio VAE; segment left unfrozen.",
            source_label,
        )
        return None

    audio_t = int(audio.shape[-1])
    offset = int(round(max(0.0, float(trim_frames or 0)) * frame_rescale()))
    if offset >= audio_t:
        log.warning(
            "Audio freeze: offset %d >= audio length %d ticks; segment left unfrozen.",
            offset,
            audio_t,
        )
        return None
    covered = max(0, min(int(z.shape[-1]), audio_t - offset))
    if covered <= 0:
        log.warning(
            "Audio freeze: zero covered ticks (audio %d, offset %d); segment left unfrozen.",
            audio_t,
            offset,
        )
        return None

    fresh = audio.clone()
    fresh[..., offset : offset + covered] = z[..., :covered].to(
        device=fresh.device, dtype=fresh.dtype
    )
    _repack_samples(
        latent, video, fresh,
        video_squeezed=video_squeezed, audio_squeezed=audio_squeezed,
    )
    _install_noise_mask(latent, video, audio, offset=offset, covered=covered)
    sig = pcm_signature(pcm)
    _cache_z(sig, z.detach().to("cpu"))
    latent[AUDIO_FREEZE_MARK_KEY] = {
        "offset_ticks": offset,
        "covered_ticks": covered,
        "source": source_label,
        "sig": sig,
        "version": FREEZE_VERSION,
    }
    log.info(
        "Audio freeze: %s encoded (%d ticks, sig=%s); pinned %d/%d ticks @+%d.",
        source_label, z_t, sig, covered, audio_t, offset,
    )
    return f"pinned {covered}/{audio_t} audio ticks to {source_label} (@+{offset})"


def refresh_frozen_audio_mask(latent) -> str | None:
    """Rebuild keep-mask + frozen content on a marked latent. Sampler entry.

    Called at the top of every ``sample_single_stage``: refine AV re-joins,
    continue-mode locks and SelfLift repacks legitimately drop or rewrite the
    mask, but the freeze marker survives in the latent dict — rebuild from it
    so the frozen audio tokens stay clean through every sampling stage
    (first pass, refine passes, SelfLift, FaceRefine).

    v2: also RESTORES the frozen window content. The SelfLift / refine resume
    math re-expresses the whole AV latent in the sigma-resume representation
    (``inverse_noise_scaling`` = ``x / (1 - sigma)``), which rewrites the audio
    window too. comfy's mask machinery would then treat that mangled audio as
    the clean x0 to preserve (prediction override + ``scale_latent_inpaint``
    reinjection), so the high-res passes would condition on noise-dominated
    audio. The encoded window is cached by PCM signature at apply time and
    written back here whenever it drifted. No-op when the content already
    matches (first entry, cache hits), and falls back to a mask-only refresh
    when the cache is cold (e.g. latent loaded from disk cache — those carry
    the clean window since v2).
    """
    if not isinstance(latent, dict):
        return None
    mark = latent.get(AUDIO_FREEZE_MARK_KEY)
    if not isinstance(mark, dict):
        return None
    av = _av_streams(latent)
    if av is None:
        return None
    video, audio, _nested, video_squeezed, audio_squeezed = av
    audio_t = int(audio.shape[-1])
    offset = max(0, int(mark.get("offset_ticks") or 0))
    covered = max(0, int(mark.get("covered_ticks") or 0))
    if covered <= 0 or offset >= audio_t:
        return None
    end = min(audio_t, offset + covered)
    restored = False
    if end > offset:
        z = _cached_z(str(mark.get("sig") or ""))
        if z is not None and int(z.shape[-1]) >= end - offset:
            want = z[..., : end - offset].to(device=audio.device, dtype=audio.dtype)
            if not torch.equal(audio[..., offset:end], want):
                fresh = audio.clone()
                fresh[..., offset:end] = want
                _repack_samples(
                    latent, video, fresh,
                    video_squeezed=video_squeezed, audio_squeezed=audio_squeezed,
                )
                restored = True
    _install_noise_mask(latent, video, audio, offset=offset, covered=covered)
    if restored:
        log.info(
            "Audio freeze: restored %d frozen ticks @+%d (resume transform "
            "rewrote the audio stream).",
            end - offset, offset,
        )
        return f"content restore + mask refresh {covered}/{audio_t} ticks @+{offset}"
    return f"mask refresh {covered}/{audio_t} ticks @+{offset}"


def timeline_audio_identity(plan) -> list[str]:
    """Timeline clip identity behind frozen audio slices (path + size + mtime).

    Mirrors ``source_video_identity``: swapping or re-rendering a timeline
    clip changes the audio slice the freeze would encode, so its identity is
    folded into the first-pass fingerprint. Empty for gen timelines (r2v card
    packs) — those are covered by the ref-audio stamps.
    """
    if plan is None:
        return []
    try:
        from ..lib.video_io import resolve_video_path, video_clips_from_timeline

        clips = video_clips_from_timeline(getattr(plan, "raw", None) or {})
    except Exception:
        return []
    tokens: list[str] = []
    for clip in clips or []:
        if not isinstance(clip, dict):
            continue
        rel = str(clip.get("videoFile") or clip.get("fileName") or "").strip().replace("\\", "/")
        if not rel:
            continue
        try:
            path = resolve_video_path(clip)
            st = os.stat(path)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
            tokens.append(f"{rel}:{st.st_size}:{mtime_ns}")
        except Exception:
            tokens.append(f"{rel}:missing")
    return sorted(tokens)
