"""Director-owned latent-continue handoff (Apache-2.0).

Writes the previous sampled MiniMax H3 AV tail into the next empty latent and
builds a NestedTensor noise mask. Optional per-step prefix remask lives on a
cloned MODEL after SigmaShift. Does not install h3_context_patches and must
not be stacked with apply_motion_context.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from .h3_motion_context import (
    DEFAULT_AUDIO_CONTEXT_FRAMES,
    FPS,
    _audio_tail_from_latent,
    _encode_tail_audio,
    _resize_frames,
    _streams_from_latent,
    _usable_context_audio,
    _video_tail_blocks,
    pixel_frames_for_latent_t,
    snap_context_frames,
    video_from_latent,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.h3_latent_continue")

CONTINUE_PIPELINE_ID = "minimax_h3_latent_continue_v3"
PREFIX_STEPS_KEY = "_director_continue_prefix_steps"
CONTINUE_SEAM_KEY = "_director_continue_seam_min"
SEAM_TAPER_TOKENS = 4
# H3 pins mask=0 rows at VISUAL_COND_TIMESTEP (official Guide). Sampler also
# copies latent_image 100% where mask=0. Continue must stay above that floor.
SEAM_MIN_MASK = 0.65
SEAM_FLOOR_MIN = 0.40
SEAM_FLOOR_MAX = 0.95
AUDIO_SOFT_RELEASE_TICKS = 8
_WRAPPER_KEY = "director_h3_continue_prefix_remask"


def clamp_seam_min_mask(value) -> float:
    """User-facing 重绘幅度. Higher = more redraw / less copy of the previous tail."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = SEAM_MIN_MASK
    if not math.isfinite(n):
        n = SEAM_MIN_MASK
    return max(SEAM_FLOOR_MIN, min(SEAM_FLOOR_MAX, n))


def prefix_token_weights(
    prefix_steps: int,
    taper_steps: int = SEAM_TAPER_TOKENS,
    seam_min: float | None = None,
) -> tuple[float, ...]:
    """1.0 on the disposable head, taper toward seam_min — never 0."""
    n = int(prefix_steps)
    if n < 1:
        return ()
    taper = max(1, min(int(taper_steps), n))
    head = n - taper
    floor = clamp_seam_min_mask(SEAM_MIN_MASK if seam_min is None else seam_min)
    weights = [1.0] * head
    weights.extend(1.0 + (floor - 1.0) * (float(i + 1) / float(taper)) for i in range(taper))
    return tuple(weights)


def _nested(video: torch.Tensor, audio: torch.Tensor, template=None):
    try:
        from comfy.nested_tensor import NestedTensor

        return NestedTensor((video, audio))
    except Exception:
        cls = type(template) if template is not None else None
        if cls is not None:
            return cls((video, audio))
        raise


def _spatial_video_mask(
    t_steps: int,
    prefix_steps: int,
    *,
    height: int,
    width: int,
    device,
    dtype,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """H3 ``mask_row_values`` needs [B,1,T,H,W] so ``mask[0,0]`` is [T,H,W]."""
    t = int(t_steps)
    h = max(1, int(height))
    w = max(1, int(width))
    mask = torch.ones((1, 1, t, h, w), device=device, dtype=dtype)
    n = max(0, min(int(prefix_steps), t))
    if n < 1:
        return mask
    if weights is None:
        ramp = torch.tensor(prefix_token_weights(n), device=device, dtype=dtype)
    else:
        ramp = weights[:n].to(device=device, dtype=dtype)
    mask[:, :, :n] = ramp.view(1, 1, n, 1, 1)
    return mask


def _soft_av_audio_mask(audio_t: int, pin_t: int, *, device, dtype) -> torch.Tensor:
    mask = torch.ones((1, 1, 1, int(audio_t)), device=device, dtype=dtype)
    n = max(0, min(int(pin_t), int(audio_t)))
    if n < 1:
        return mask
    mask[..., :n] = 0.0
    release = min(AUDIO_SOFT_RELEASE_TICKS, n)
    if release < 1:
        return mask
    idx = torch.arange(1, release + 1, device=device, dtype=dtype)
    ramp = 0.5 - 0.5 * torch.cos(math.pi * idx / float(release))
    mask[..., n - release : n] = ramp.reshape(1, 1, 1, -1)
    return mask


def apply_latent_continue(
    latent: dict,
    *,
    prev_av: dict | None,
    prev_tail: torch.Tensor | None,
    vae,
    context_length: int,
    context_end_frame: int | None = None,
    pin_audio: bool = True,
    context_audio: dict | None = None,
    audio_vae=None,
    audio_context_length: int | None = None,
    seam_min_mask: float | None = None,
) -> tuple[dict, int, int]:
    """Copy prev AV tail into ``latent`` samples + noise_mask. No conditioning edits.

    Returns ``(latent, trim_frames, prev_export_trim)``.
    """
    context_length = snap_context_frames(context_length)
    video = video_from_latent(latent)
    streams = _streams_from_latent(latent)
    if len(streams) < 2:
        raise ValueError("Director continue: target latent has no audio stream.")
    audio = streams[1]
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    width = int(video.shape[4]) * 16
    height = int(video.shape[3]) * 16
    frame_count = pixel_frames_for_latent_t(int(video.shape[2]))

    pin_audio_latent = prev_av
    if prev_av is not None:
        src = video_from_latent(prev_av)
        src_w, src_h = int(src.shape[4]) * 16, int(src.shape[3]) * 16
        if src_w == width and src_h == height:
            available = pixel_frames_for_latent_t(int(src.shape[2]))
            if context_end_frame is not None:
                available = min(available, max(0, int(context_end_frame)))
            video_src = "latent"
        elif prev_tail is not None and int(prev_tail.shape[0]) >= 1:
            log.warning(
                "Director continue: context latent is %dx%d but this segment is "
                "%dx%d — write from decoded frames.",
                src_w,
                src_h,
                width,
                height,
            )
            context_end_frame = None
            pin_audio_latent = None
            available = int(prev_tail.shape[0])
            video_src = "pixels"
        else:
            raise ValueError(
                f"Director continue: context latent is {src_w}x{src_h} but this "
                f"segment is {width}x{height}."
            )
    else:
        if prev_tail is None or int(prev_tail.shape[0]) < 1:
            raise ValueError("Director continue: need previous AV latent or decoded frames.")
        available = int(prev_tail.shape[0])
        video_src = "pixels"

    n = min(int(context_length), available)
    if n < 1:
        raise ValueError("Director continue: no frames available to write.")
    from .h3_motion_context import VIDEO_RUN_GRID

    run = next(g for g in VIDEO_RUN_GRID if g <= n)
    if run != n:
        log.warning("Director continue: %d frames off VAE grid; using last %d.", n, run)
        n = run
    if n >= frame_count:
        raise ValueError(
            f"Director continue: cannot write {n} frames into a {frame_count}-frame clip."
        )

    pin_end_px: int | None = None
    prev_export_trim = 0
    if video_src == "latent":
        blocks, _offsets, covered, pin_end_px, prev_export_trim = _video_tail_blocks(
            prev_av, n, end_frame=context_end_frame
        )
        tail_video = torch.cat(blocks, dim=2)
        span = covered
    else:
        tail = _resize_frames(prev_tail[available - n :], width, height)
        enc = vae.encode(tail)
        if getattr(enc, "ndim", 0) != 5:
            raise ValueError(
                f"Director continue: VAE encode returned shape "
                f"{tuple(getattr(enc, 'shape', ()))}, expected [B,C,T,H,W]."
            )
        covered = pixel_frames_for_latent_t(int(enc.shape[2]))
        if covered != n:
            raise RuntimeError(
                f"Director continue: {n} frames encoded to {int(enc.shape[2])} steps "
                f"covering {covered}; VAE grid mismatch."
            )
        tail_video = enc
        span = covered
        pin_end_px = available
        prev_export_trim = 0

    t_tail = min(int(tail_video.shape[2]), int(video.shape[2]) - 1)
    if t_tail < 1:
        raise ValueError("Director continue: empty video prefix.")
    patched_video = video.clone()
    patched_video[:, :, :t_tail] = tail_video[:, :, :t_tail].to(
        device=patched_video.device, dtype=patched_video.dtype
    )

    patched_audio = audio.clone()
    audio_pin_t = 0
    if pin_audio and (pin_audio_latent is not None or _usable_context_audio(context_audio) is not None):
        a_frames = int(audio_context_length) if audio_context_length else DEFAULT_AUDIO_CONTEXT_FRAMES
        if a_frames <= 0:
            a_frames = int(span)
        audio_end_limit = pin_end_px if pin_end_px is not None else context_end_frame
        if pin_audio_latent is not None:
            audio_tail, audio_pin_t, _overhang = _audio_tail_from_latent(
                pin_audio_latent, a_frames, end_frame=audio_end_limit
            )
        else:
            if audio_vae is None:
                raise ValueError("Director continue: context_audio requires audio_vae.")
            audio_tail, audio_pin_t = _encode_tail_audio(
                audio_vae, context_audio, a_frames / float(FPS)
            )
        audio_pin_t = min(int(audio_pin_t), int(patched_audio.shape[-1]))
        if audio_pin_t > 0:
            patched_audio[..., :audio_pin_t] = audio_tail[..., :audio_pin_t].to(
                device=patched_audio.device, dtype=patched_audio.dtype
            )
    elif pin_audio:
        log.warning("Director continue: previous export audio is empty; video prefix only.")

    out = dict(latent)
    template = latent.get("samples")
    out["samples"] = _nested(patched_video, patched_audio, template)
    seam = clamp_seam_min_mask(SEAM_MIN_MASK if seam_min_mask is None else seam_min_mask)
    weights = prefix_token_weights(t_tail, seam_min=seam)
    video_mask = _spatial_video_mask(
        int(patched_video.shape[2]),
        t_tail,
        height=int(patched_video.shape[3]),
        width=int(patched_video.shape[4]),
        device=patched_video.device,
        dtype=torch.float32,
        weights=torch.tensor(weights, dtype=torch.float32),
    )
    audio_mask = _soft_av_audio_mask(
        int(patched_audio.shape[-1]),
        audio_pin_t,
        device=patched_audio.device,
        dtype=torch.float32,
    )
    out["noise_mask"] = _nested(video_mask, audio_mask, template)
    out[PREFIX_STEPS_KEY] = int(t_tail)
    out[CONTINUE_SEAM_KEY] = float(seam)
    log.info(
        "Director continue: wrote %d video tokens (%s, %df) + %d audio ticks; "
        "prefix mask head=%.2f seam=%.2f (floor=%.2f, no cond-pin); "
        "trim=%df prev_export_trim=%df",
        t_tail,
        video_src,
        span,
        audio_pin_t,
        weights[0] if weights else 0.0,
        weights[-1] if weights else 0.0,
        seam,
        span,
        prev_export_trim,
    )
    return out, int(span), int(prev_export_trim)


def _schedule_values(sigmas: Any) -> tuple[float, ...]:
    if torch.is_tensor(sigmas):
        raw = sigmas.detach().float().reshape(-1).cpu().tolist()
    else:
        raw = list(sigmas or ())
    values = [float(v) for v in raw if math.isfinite(float(v)) and float(v) >= 0.0]
    return tuple(sorted(set(values), reverse=True))


def _next_sigma_ratio(current: float, sigmas: Any) -> float:
    cur = float(current)
    if not math.isfinite(cur) or cur <= 0.0:
        return 0.0
    tol = max(1e-7, abs(cur) * 1e-6)
    for cand in _schedule_values(sigmas):
        if cand < cur - tol:
            return max(0.0, min(1.0, cand / cur))
    return 0.0


def _apply_video_prefix_weights(video_mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Write taper weights onto the temporal prefix of a video-shaped mask."""
    n = int(weights.numel())
    if n < 1 or not torch.is_tensor(video_mask):
        return video_mask
    w = weights.to(device=video_mask.device, dtype=video_mask.dtype)
    out = video_mask.clone()
    if video_mask.ndim == 5 and n <= int(video_mask.shape[2]):
        # [B,C,T,H,W] or compact [1,1,T,1,1]
        view = [1] * 5
        view[2] = n
        out[:, :, :n] = w.view(*view)
        return out
    if video_mask.ndim == 3 and n <= int(video_mask.shape[-1]):
        # [B,1,T] temporal token mask (H3 often packs this, not C*T*H*W).
        out[..., :n] = w.view(1, 1, n)
        return out
    if video_mask.ndim == 4 and n <= int(video_mask.shape[2]):
        view = [1] * 4
        view[2] = n
        out[:, :, :n] = w.view(*view)
        return out
    return video_mask


def _unbind_mask(mask):
    # torch.Tensor.unbind exists but splits batch, not NestedTensor streams.
    if torch.is_tensor(mask):
        return [mask]
    if hasattr(mask, "unbind"):
        return list(mask.unbind())
    if hasattr(mask, "tensors"):
        return list(mask.tensors)
    if isinstance(mask, (tuple, list)):
        return list(mask)
    return None


def _prefix_steps_from_latent(latent: dict) -> int:
    raw = latent.pop(PREFIX_STEPS_KEY, 0) if isinstance(latent, dict) else 0
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _seam_min_from_latent(latent: dict) -> float:
    raw = None
    if isinstance(latent, dict):
        raw = latent.get(CONTINUE_SEAM_KEY)
    return clamp_seam_min_mask(SEAM_MIN_MASK if raw is None else raw)


class _PrefixRemask:
    def __init__(
        self,
        prefix_steps: int,
        sigmas: Any,
        video_shape: tuple[int, ...],
        seam_min: float | None = None,
    ):
        self.prefix_steps = int(prefix_steps)
        self.sigmas = _schedule_values(sigmas)
        self.video_shape = tuple(int(x) for x in video_shape)
        self.seam_min = clamp_seam_min_mask(SEAM_MIN_MASK if seam_min is None else seam_min)
        self.current_video_mask: torch.Tensor | None = None

    def _live_weights(self, sigma, extra_options=None) -> torch.Tensor:
        current = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
        schedule = self.sigmas or _schedule_values((extra_options or {}).get("sigmas", ()))
        ratio = _next_sigma_ratio(current, schedule)
        # Head stays fully open. Seam follows sigma but never drops to cond-pin.
        live = []
        for base in prefix_token_weights(self.prefix_steps, seam_min=self.seam_min):
            if base >= 0.999:
                live.append(1.0)
            else:
                live.append(max(self.seam_min, float(base) * max(float(ratio), 0.5)))
        return torch.tensor(live, dtype=torch.float32)

    def denoise_mask_function(self, sigma, denoise_mask, extra_options=None):
        weights = self._live_weights(sigma, extra_options)
        _b, _c, t, h, w = self.video_shape
        device = denoise_mask.device if torch.is_tensor(denoise_mask) else "cpu"
        dtype = denoise_mask.dtype if torch.is_tensor(denoise_mask) else torch.float32
        streams = _unbind_mask(denoise_mask)
        if streams and torch.is_tensor(streams[0]):
            device = streams[0].device
            dtype = streams[0].dtype
        spatial = torch.ceil(
            _spatial_video_mask(
                t,
                self.prefix_steps,
                height=h,
                width=w,
                device=device,
                dtype=torch.float32,
                weights=weights,
            )
            * 256.0
        ) / 256.0
        # H3 apply_model must see [B,1,T,H,W]; never a 1D/3D token strip.
        self.current_video_mask = spatial

        if streams and torch.is_tensor(streams[0]) and streams[0].ndim == 5:
            video = _apply_video_prefix_weights(streams[0].float(), weights)
            video = torch.ceil(video * 256.0) / 256.0
            rest = [s for s in streams[1:]]
            if rest:
                return _nested(video.to(dtype=streams[0].dtype), rest[0], denoise_mask)
            return video.to(dtype=streams[0].dtype)

        if torch.is_tensor(denoise_mask) and denoise_mask.ndim == 3 and len(self.video_shape) == 5:
            elems = int(math.prod(self.video_shape[1:]))
            last = int(denoise_mask.shape[-1])
            # Packed AV is [B,1,video_flat+audio_flat]; video-only pack is == elems.
            if last >= elems:
                packed = denoise_mask.clone()
                video = packed[..., :elems].reshape(self.video_shape)
                video = _apply_video_prefix_weights(video.float(), weights)
                video = torch.ceil(video * 256.0) / 256.0
                packed[..., :elems] = video.reshape(packed.shape[0], 1, elems).to(
                    dtype=packed.dtype
                )
                return packed
            video = _apply_video_prefix_weights(denoise_mask.float(), weights)
            return torch.ceil(video * 256.0).div(256.0).to(dtype=denoise_mask.dtype)

        if torch.is_tensor(denoise_mask) and denoise_mask.ndim == 5:
            video = _apply_video_prefix_weights(denoise_mask.float(), weights)
            return torch.ceil(video * 256.0).div(256.0).to(dtype=denoise_mask.dtype)
        return denoise_mask

    def apply_model_wrapper(self, executor, *args, **kwargs):
        mask = self.current_video_mask
        if torch.is_tensor(mask) and mask.ndim == 5 and mask.shape[-2] >= 2 and mask.shape[-1] >= 2:
            kwargs["denoise_mask"] = mask
        return executor(*args, **kwargs)


def install_continue_prefix_remask(model, latent: dict, sigmas) -> Any:
    """Clone ``model`` and attach schedule-matched video-prefix remask. Guide never calls this."""
    if model is None or not callable(getattr(model, "clone", None)):
        return model
    prefix_steps = _prefix_steps_from_latent(latent)
    if prefix_steps < 1:
        return model
    try:
        samples = latent.get("samples") if isinstance(latent, dict) else None
        streams = _unbind_mask(samples) if samples is not None else None
        if not streams or not torch.is_tensor(streams[0]) or streams[0].ndim != 5:
            return model
        if prefix_steps >= int(streams[0].shape[2]):
            return model
        patched = model.clone()
        if not callable(getattr(patched, "set_model_denoise_mask_function", None)):
            log.warning("Director continue: MODEL has no denoise-mask hook; static mask only.")
            return model
        state = _PrefixRemask(
            prefix_steps, sigmas, tuple(streams[0].shape), seam_min=_seam_min_from_latent(latent)
        )
        patched.set_model_denoise_mask_function(state.denoise_mask_function)
        try:
            from comfy.patcher_extension import WrappersMP

            if callable(getattr(patched, "add_wrapper_with_key", None)):
                patched.add_wrapper_with_key(
                    WrappersMP.APPLY_MODEL,
                    _WRAPPER_KEY,
                    state.apply_model_wrapper,
                )
        except Exception as exc:
            log.debug("Director continue: APPLY_MODEL wrapper skipped (%s).", exc)
        return patched
    except Exception as exc:
        log.warning("Director continue: prefix remask not installed (%s); static mask only.", exc)
        return model
