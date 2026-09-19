"""Dual-res continuity: native low-res carry + high-res pin after lift."""

from __future__ import annotations

import logging

import torch

from ..h3_latent_continue import CONTINUE_SEAM_KEY, PREFIX_STEPS_KEY
from ..h3_motion_context import (
    AUDIO_HZ,
    FPS,
    _repack_av_streams,
    _streams_from_latent,
    copy_av_tail_into_prefix,
    pixel_frames_for_latent_t,
    steps_for_frames,
    video_from_latent,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.selflift.carry")


def prefix_steps_from_pin_frames(n_frames: int) -> int:
    n = int(n_frames or 0)
    if n <= 0:
        return 0
    steps = steps_for_frames(n)
    return int(steps) if steps else 0


def copy_low_tail(
    low_latent: dict,
    prev_low_carry: dict | None,
    n_frames: int,
    *,
    end_frame: int | None = None,
) -> dict:
    """Paste the previous segment's native low-res tail into the current low prefix."""
    if prev_low_carry is None or int(n_frames or 0) <= 0:
        return low_latent
    try:
        src = video_from_latent(prev_low_carry)
        dst = video_from_latent(low_latent)
        if tuple(src.shape[-2:]) != tuple(dst.shape[-2:]):
            log.warning(
                "SelfLift low-carry spatial mismatch %s vs %s; skip native splice.",
                tuple(src.shape[-2:]),
                tuple(dst.shape[-2:]),
            )
            return low_latent
        mask = low_latent.get("noise_mask")
        out = copy_av_tail_into_prefix(
            low_latent, prev_low_carry, int(n_frames), end_frame=end_frame
        )
        if mask is not None:
            out["noise_mask"] = mask
        return out
    except Exception as exc:
        log.warning("SelfLift native low-carry splice skipped (%s).", exc)
        return low_latent


def _audio_pin_ticks(source: dict, prefix_steps: int) -> int:
    """How many audio latent ticks the continue pin occupies."""
    mask = source.get("noise_mask") if isinstance(source, dict) else None
    if mask is not None and not torch.is_tensor(mask) and getattr(mask, "is_nested", False):
        try:
            parts = list(mask.unbind())
            if len(parts) > 1 and torch.is_tensor(parts[1]):
                seq = parts[1].detach().float().reshape(-1, parts[1].shape[-1]).mean(dim=0)
                pinned = int((seq < 0.99).sum().item())
                if pinned > 0:
                    return pinned
        except Exception:
            pass
    n = max(0, int(prefix_steps or 0))
    if n < 1:
        return 0
    frames = pixel_frames_for_latent_t(n)
    return max(0, int(round(float(frames) / float(FPS) * AUDIO_HZ)))


_LIFT_DC_CLAMP = 0.05


def match_lift_prefix_dc(lifted: torch.Tensor, previous: torch.Tensor, prefix_steps: int) -> torch.Tensor:
    """Remove per-channel lift bias on the disposable prefix only.

    Compare the continue prefix in the 3D-lifted volume against the true
    high-res predecessor. Subtract a clamped per-channel median delta from
    those prefix tokens. Do not extend the shift into newly generated frames
    (that rebound shows up after overlap trim).
    """
    if not torch.is_tensor(lifted) or not torch.is_tensor(previous):
        return lifted
    a = lifted.unsqueeze(0) if lifted.ndim == 4 else lifted
    b = previous.unsqueeze(0) if previous.ndim == 4 else previous
    squeezed = lifted.ndim == 4
    if a.ndim != 5 or b.ndim != 5:
        return lifted
    if a.shape[:2] != b.shape[:2] or tuple(a.shape[-2:]) != tuple(b.shape[-2:]):
        return lifted
    n = min(max(0, int(prefix_steps)), int(a.shape[2]), int(b.shape[2]))
    if n < 1:
        return lifted
    b = b.to(device=a.device)
    deltas = []
    for ch in range(int(a.shape[1])):
        diff = a[:, ch, :n].float() - b[:, ch, :n].float()
        deltas.append(diff.reshape(-1).median())
        del diff
    dc = torch.stack(deltas).clamp(-float(_LIFT_DC_CLAMP), float(_LIFT_DC_CLAMP))
    dc = dc.to(device=a.device, dtype=a.dtype).view(1, -1, 1, 1, 1)
    out = a.clone()
    out[:, :, :n] = out[:, :, :n] - dc
    if squeezed:
        out = out.squeeze(0)
    log.info(
        "SelfLift seam DC: prefix=%d clamp=%.3f max_abs=%.5f",
        n,
        _LIFT_DC_CLAMP,
        float(dc.detach().float().abs().max().item()),
    )
    return out


def restore_high_pin(target: dict, source: dict, prefix_steps: int) -> dict:
    """Write the sharp continue prefix back after 3D lift.

    Timeline Director / latent-continue paste the previous sampled AV tail into
    the next opening. SelfLift then replaces that head with a 3D-upscaled
    low-res estimate; remask would lock the blurry head and the visible seam
    (after overlap trim) goes soft. Copy the original high-res pin back so the
    high-res Euler attends to the real predecessor.
    """
    n = int(prefix_steps or 0)
    if n < 1 or not isinstance(target, dict) or not isinstance(source, dict):
        return target
    try:
        src_streams = list(_streams_from_latent(source))
        dst_streams = list(_streams_from_latent(target))
    except Exception as exc:
        log.warning("SelfLift high pin skipped (%s).", exc)
        return target
    if not src_streams or not dst_streams:
        return target
    out = []
    video_copied = 0
    audio_copied = 0
    for i, dst in enumerate(dst_streams):
        src = src_streams[i] if i < len(src_streams) else None
        if not torch.is_tensor(dst) or not torch.is_tensor(src):
            out.append(dst)
            continue
        work = dst.clone()
        if i == 0:
            src_v = src.unsqueeze(0) if src.ndim == 4 else src
            dst_v = work.unsqueeze(0) if work.ndim == 4 else work
            squeezed = work.ndim == 4
            if src_v.ndim != 5 or dst_v.ndim != 5:
                out.append(dst)
                continue
            if tuple(src_v.shape[-2:]) != tuple(dst_v.shape[-2:]) or int(src_v.shape[1]) != int(
                dst_v.shape[1]
            ):
                log.warning(
                    "SelfLift high pin spatial mismatch %s vs %s; skip.",
                    tuple(src_v.shape),
                    tuple(dst_v.shape),
                )
                out.append(dst)
                continue
            t = min(n, int(src_v.shape[2]), int(dst_v.shape[2]))
            if t < 1:
                out.append(dst)
                continue
            dst_v = dst_v.clone()
            dst_v[:, :, :t] = src_v[:, :, :t].to(device=dst_v.device, dtype=dst_v.dtype)
            work = dst_v.squeeze(0) if squeezed else dst_v
            video_copied = t
        else:
            pin_t = min(
                _audio_pin_ticks(source, n),
                int(src.shape[-1]),
                int(work.shape[-1]),
            )
            if pin_t > 0:
                work = work.clone()
                work[..., :pin_t] = src[..., :pin_t].to(device=work.device, dtype=work.dtype)
                audio_copied = pin_t
        out.append(work.contiguous())
    packed = dict(target)
    packed["samples"] = _repack_av_streams(out, source)
    if source.get("noise_mask") is not None:
        packed["noise_mask"] = source["noise_mask"]
    log.info(
        "SelfLift high pin: restored %d video tokens + %d audio ticks from continue latent.",
        video_copied,
        audio_copied,
    )
    return packed


def stamp_continue_keys(latent: dict, source: dict | None) -> dict:
    """Copy remask metadata onto a resized latent so after_shift still fires."""
    if not isinstance(latent, dict):
        return latent
    src = source if isinstance(source, dict) else {}
    if PREFIX_STEPS_KEY in src:
        latent[PREFIX_STEPS_KEY] = src[PREFIX_STEPS_KEY]
    if CONTINUE_SEAM_KEY in src:
        latent[CONTINUE_SEAM_KEY] = src[CONTINUE_SEAM_KEY]
    return latent


def lock_prefix_keep_mask(latent: dict, prefix_steps: int) -> dict:
    """Force the video prefix to mask=0 so native inpaint keeps latent_image.

    Timeline continuation writes a hard keep-mask on the overlap, then a
    schedule-matched remask opens those tokens by next/current sigma. A
    generate (1.0) prefix never becomes visual-cond, so the first frame after
    overlap trim stays soft.
    """
    n = max(0, int(prefix_steps or 0))
    if n < 1 or not isinstance(latent, dict):
        return latent
    out = dict(latent)
    out[PREFIX_STEPS_KEY] = n
    out[CONTINUE_SEAM_KEY] = 0.0
    mask = out.get("noise_mask")
    if mask is None:
        return out

    def _zero_video(video: torch.Tensor) -> torch.Tensor:
        work = video.clone()
        if work.ndim == 5:
            t = min(n, int(work.shape[2]))
            if t > 0:
                work[:, :, :t] = 0
            return work
        if work.ndim == 3:
            t = min(n, int(work.shape[-1]))
            if t > 0:
                work[..., :t] = 0
            return work
        if work.ndim == 4:
            t = min(n, int(work.shape[2]))
            if t > 0:
                work[:, :, :t] = 0
            return work
        return video

    try:
        if torch.is_tensor(mask):
            out["noise_mask"] = _zero_video(mask)
            return out
        if getattr(mask, "is_nested", False) and hasattr(mask, "unbind"):
            parts = list(mask.unbind())
        elif isinstance(mask, (tuple, list)):
            parts = list(mask)
        else:
            return out
        if parts and torch.is_tensor(parts[0]):
            parts[0] = _zero_video(parts[0])
            out["noise_mask"] = _repack_av_streams(parts, {"samples": mask})
        log.info("SelfLift continue: keep-mask on %d video prefix token(s), seam=0", n)
    except Exception as exc:
        log.warning("SelfLift keep-mask skipped (%s).", exc)
    return out
