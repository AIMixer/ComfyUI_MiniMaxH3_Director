"""First-pass progressive sample: low-res prefix → lift → high-res tail (Euler)."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from ..core_sampling import ShiftedModelCache, sample_single_stage
from ..h3_latent_continue import PREFIX_STEPS_KEY
from ..h3_motion_context import _repack_av_streams, av_pixel_size
from .carry import (
    copy_low_tail,
    lock_prefix_keep_mask,
    match_lift_prefix_dc,
    prefix_steps_from_pin_frames,
    stamp_continue_keys,
)
from .cond import interpolate_video_5d, resize_av_video, resize_positive_spatial
from .grid import is_same_canvas, lowres_canvas, pixel_to_latent_hw
from .lift import lift_video_tensor, mix_rho, pixel_anchor_latent
from .pack import require_euler, resolve_transition_k

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.selflift.sample")

PhaseCallback = Callable[[str, float], None]
StepPreviewCallback = Callable[[int, int, Any], None]


def _sigma_vec(sigmas) -> torch.Tensor:
    if torch.is_tensor(sigmas):
        return sigmas.detach().float().reshape(-1).cpu()
    return torch.tensor([float(x) for x in sigmas], dtype=torch.float32)


def _n_steps(sigmas: torch.Tensor) -> int:
    return max(1, int(sigmas.numel()) - 1)


def _is_nested(samples) -> bool:
    # torch.Tensor also has unbind() (batch axis) and may report is_nested=False.
    if samples is None or torch.is_tensor(samples):
        return False
    return bool(getattr(samples, "is_nested", False))


def _streams(samples):
    """Split AV streams. Never Tensor.unbind — that is the batch axis."""
    if _is_nested(samples):
        return list(samples.unbind()), True
    if isinstance(samples, (tuple, list)):
        return list(samples), True
    if torch.is_tensor(samples):
        return [samples], False
    raise ValueError(f"SelfLift expected NestedTensor AV samples, got {type(samples)!r}")


def _pack(streams, nested, template=None):
    if nested:
        return _repack_av_streams(list(streams), {"samples": template} if template is not None else None)
    return streams[0]


def _isfinite_av(samples) -> bool:
    try:
        for part in _streams(samples)[0]:
            if torch.is_tensor(part) and not torch.isfinite(part).all():
                return False
        return True
    except Exception:
        return True


def _from_callback(value, template_latent: dict | None):
    """Callback x0/x may be NestedTensor, packed [B,1,flat], or a latent dict."""
    if value is None:
        return None
    if isinstance(value, dict) and "samples" in value:
        return value["samples"]
    if _is_nested(value) or isinstance(value, (tuple, list)):
        return value
    if torch.is_tensor(value) and isinstance(template_latent, dict):
        tpl = template_latent.get("samples")
        if _is_nested(tpl):
            try:
                import comfy.utils

                shapes = [tuple(p.shape) for p in tpl.unbind()]
                parts = list(comfy.utils.unpack_latents(value, shapes))
                return _repack_av_streams(parts, template_latent)
            except Exception as exc:
                log.debug("SelfLift packed callback unpack skipped: %s", exc)
    return value


def _patcher_inner(model):
    return getattr(model, "model", model)


def _model_sampling(model):
    if hasattr(model, "get_model_object"):
        try:
            return model.get_model_object("model_sampling")
        except Exception:
            pass
    inner = _patcher_inner(model)
    ms = getattr(inner, "model_sampling", None)
    if ms is not None:
        return ms
    deeper = getattr(inner, "model", None)
    return getattr(deeper, "model_sampling", None) if deeper is not None else None


def _latent_format(model):
    if hasattr(model, "get_model_object"):
        try:
            return model.get_model_object("latent_format")
        except Exception:
            pass
    inner = _patcher_inner(model)
    fmt = getattr(inner, "latent_format", None)
    if fmt is not None:
        return fmt
    deeper = getattr(inner, "model", None)
    return getattr(deeper, "latent_format", None) if deeper is not None else None


def _process_video_format(fmt, method: str, video: torch.Tensor) -> torch.Tensor:
    fn = getattr(fmt, method, None) if fmt is not None else None
    if not callable(fn):
        return video
    out = fn(video.float())
    if not torch.is_tensor(out):
        return video
    return out.to(device=video.device, dtype=video.dtype)


def _process_latent_in(model, samples):
    inner = _patcher_inner(model)
    fn = getattr(inner, "process_latent_in", None)
    if callable(fn):
        try:
            return fn(samples)
        except Exception as exc:
            log.debug("SelfLift process_latent_in on NestedTensor failed (%s); per-stream.", exc)
    fmt = _latent_format(model)
    streams, nested = _streams(samples)
    out = [_process_video_format(fmt, "process_in", s) if torch.is_tensor(s) else s for s in streams]
    return _pack(out, nested, samples)


def _process_latent_out(model, samples):
    """Storage-space NestedTensor so SamplerCustomAdvanced process_latent_in is not a double-in."""
    inner = _patcher_inner(model)
    fn = getattr(inner, "process_latent_out", None)
    if callable(fn):
        try:
            return fn(samples)
        except Exception as exc:
            log.debug("SelfLift process_latent_out on NestedTensor failed (%s); per-stream.", exc)
    fmt = _latent_format(model)
    streams, nested = _streams(samples)
    out = [_process_video_format(fmt, "process_out", s) if torch.is_tensor(s) else s for s in streams]
    return _pack(out, nested, samples)


def _euler_step(state, denoised, sigma, sigma_next):
    """Euler update: x + (x - x0) * (σ_next-σ)/σ."""
    sig = torch.as_tensor(sigma, device=state.device, dtype=state.dtype).reshape(-1)[0]
    sig_n = torch.as_tensor(sigma_next, device=state.device, dtype=state.dtype).reshape(-1)[0]
    if float(sig.abs()) < 1e-12:
        return state
    step = ((sig_n - sig) / sig).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(device=state.device, dtype=state.dtype)) * step


def _force_spatial(video, dst_h: int, dst_w: int, mode: str):
    if not torch.is_tensor(video) or video.ndim < 4:
        return video
    if int(video.shape[-2]) == int(dst_h) and int(video.shape[-1]) == int(dst_w):
        return video
    return interpolate_video_5d(video, int(dst_h), int(dst_w), mode)


def sample_selflift_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    pack: dict,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    sigmas=None,
    on_phase: PhaseCallback | None = None,
    on_step_preview: StepPreviewCallback | None = None,
    preview_every: int = 1,
    after_shift=None,
    shift_cache: ShiftedModelCache | None = None,
    prev_low_carry: dict | None = None,
    pin_frames: int = 0,
    prev_end_frame: int | None = None,
    vae=None,
    canvas_width: int = 0,
    canvas_height: int = 0,
):
    """Run low-res prefix + 3D lift + high-res tail. Returns (high_av, low_carry)."""
    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    sampler_use = require_euler(sampler_name, pack)
    high_model = pack.get("sample_model") if pack.get("sample_model") is not None else model

    # Resolve canvas from the (already pinned) high-res latent when unset.
    size = av_pixel_size(latent)
    high_w = int(canvas_width or 0) or (size[0] if size else 0)
    high_h = int(canvas_height or 0) or (size[1] if size else 0)
    if high_w <= 0 or high_h <= 0:
        raise ValueError("SelfLift: cannot read target canvas from AV latent.")

    scale = float(pack.get("lowres_scale") or 0.5)
    low_w, low_h = lowres_canvas(high_w, high_h, scale)
    if is_same_canvas((low_w, low_h), (high_w, high_h)):
        log.info("SelfLift scale=%.3f is a no-op at %dx%d; using single-stage sample.", scale, high_w, high_h)
        out = sample_single_stage(
            model=high_model,
            positive=positive,
            negative=negative,
            latent=latent,
            seed=seed,
            cfg=cfg,
            steps=steps,
            sampler_name=sampler_name,
            scheduler=scheduler,
            shift_video=shift_video,
            shift_audio=shift_audio,
            sigmas=sigmas,
            on_phase=on_phase,
            on_step_preview=on_step_preview,
            preview_every=preview_every,
            after_shift=after_shift,
            shift_cache=shift_cache,
        )
        return out, None

    captured: dict[str, Any] = {}

    def _capture_preview(step: int, total_steps: int, x0, x=None) -> None:
        captured["step"] = int(step)
        captured["total"] = int(total_steps)
        captured["x0"] = x0
        if x is not None:
            captured["x"] = x
        if on_step_preview is not None:
            on_step_preview(step, total_steps, x0)

    def _on_state(step, total, x0, x):
        _capture_preview(int(step), int(total), x0, x)

    upsample = str(pack.get("latent_upsample") or "bilinear")
    src_lat_h, src_lat_w = pixel_to_latent_hw(high_w, high_h)
    dst_lat_h, dst_lat_w = pixel_to_latent_hw(low_w, low_h)

    prefix_steps = 0
    if isinstance(latent, dict):
        try:
            prefix_steps = int(latent.get(PREFIX_STEPS_KEY) or 0)
        except (TypeError, ValueError):
            prefix_steps = 0
    if prefix_steps < 1 and int(pin_frames or 0) > 0:
        prefix_steps = prefix_steps_from_pin_frames(pin_frames)

    low_latent = resize_av_video(latent, dst_lat_h, dst_lat_w, upsample)
    low_latent = stamp_continue_keys(low_latent, latent)
    if pack.get("native_low_carry", True) and prev_low_carry is not None and int(pin_frames or 0) > 0:
        low_latent = copy_low_tail(
            low_latent, prev_low_carry, int(pin_frames), end_frame=prev_end_frame
        )
        low_latent = stamp_continue_keys(low_latent, latent)
    if prefix_steps > 0:
        low_latent = lock_prefix_keep_mask(low_latent, prefix_steps)

    low_positive = resize_positive_spatial(
        positive, src_lat_h, src_lat_w, dst_lat_h, dst_lat_w, upsample
    )

    if shift_cache is not None:
        shifted_hi = shift_cache.get(high_model, shift_video, shift_audio)
    else:
        shifted_hi = high_model

    if sigmas is not None:
        sigma_t = _sigma_vec(sigmas)
    else:
        from comfy_extras.nodes_custom_sampler import BasicScheduler
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        def _unpack(out):
            if hasattr(out, "args") and out.args:
                return out.args
            if isinstance(out, (tuple, list)):
                return out
            raise RuntimeError(type(out))

        sched_model = shifted_hi
        if shift_cache is None:
            sched_model = _unpack(
                MiniMaxH3SigmaShift.execute(high_model, float(shift_video), float(shift_audio))
            )[0]
            shifted_hi = sched_model
        sigma_out = BasicScheduler.execute(sched_model, str(scheduler), int(steps), 1.0)
        sigma_t = _sigma_vec(_unpack(sigma_out)[0])

    n_steps = _n_steps(sigma_t)
    k = resolve_transition_k(pack, n_steps)
    if k < 1 or k >= n_steps:
        raise ValueError(
            f"SelfLift transition k={k} invalid for {n_steps} steps "
            "(need at least 1 low-res and 1 high-res step)."
        )
    low_sigmas = sigma_t[: k + 1]
    high_sigmas = sigma_t[k:]
    log.info(
        "SelfLift: %dx%d → %dx%d, k=%d/%d (low %d + high %d), euler, rho=%.2f",
        high_w,
        high_h,
        low_w,
        low_h,
        k,
        n_steps,
        k,
        n_steps - k,
        float(pack.get("rho") or 0),
    )

    notify("selflift_low", 0)
    _ = sample_single_stage(
        model=model,
        positive=low_positive,
        negative=negative,
        latent=low_latent,
        seed=seed,
        cfg=cfg,
        steps=k,
        sampler_name=sampler_use,
        scheduler=scheduler,
        shift_video=shift_video,
        shift_audio=shift_audio,
        sigmas=low_sigmas,
        on_phase=None,
        on_step_preview=None,
        on_step_state=_on_state,
        preview_every=-1,
        after_shift=after_shift,
        shift_cache=shift_cache,
        enable_tiling=False,
    )
    notify("selflift_low", 1)

    x_low = _from_callback(captured.get("x"), low_latent)
    x0_low = _from_callback(captured.get("x0"), low_latent)
    if x0_low is None:
        raise RuntimeError("SelfLift: low-res sampler did not report x0 at the transition step.")

    x0_streams, _ = _streams(x0_low)
    x_streams, _ = _streams(x_low if x_low is not None else x0_low)
    high_streams, high_nested = _streams(latent["samples"])
    if high_nested and len(x0_streams) != len(high_streams):
        raise RuntimeError(
            f"SelfLift: callback x0 has {len(x0_streams)} AV stream(s), "
            f"high latent has {len(high_streams)}. Packed callback was not unpacked."
        )
    target_lat_h = int(high_streams[0].shape[-2])
    target_lat_w = int(high_streams[0].shape[-1])

    # Native low-res carry in storage/VAE space (matches resized low_latent).
    try:
        low_carry = {"samples": _process_latent_out(shifted_hi, x0_low)}
    except Exception:
        low_carry = {"samples": x0_low}

    if not pack.get("h3_latent_model") and pack.get("latent_upscale_ref") is None:
        raise ValueError(
            "SelfLift 需要 H3 3D latent 放大权重。"
            "请把 minimax_h3_latent_upscaler_3d_*.safetensors 放到 "
            "ComfyUI/models/latent_upscale_models/ 并在节点里选中。"
        )

    notify("selflift_lift", 0)
    fmt = _latent_format(shifted_hi)
    video_x0 = x0_streams[0]
    try:
        z0_low_vae = _process_video_format(fmt, "process_out", video_x0)
    except Exception as exc:
        log.warning("SelfLift process_out skipped (%s); lifting in callback space.", exc)
        z0_low_vae = video_x0
        fmt = None

    lifted_vae = lift_video_tensor(
        z0_low_vae,
        src_width=low_w,
        src_height=low_h,
        dst_width=high_w,
        dst_height=high_h,
        model_name=str(pack.get("h3_latent_model") or ""),
        model=pack.get("latent_upscale_ref"),
        enable_chunking=bool(pack.get("enable_latent_chunking")),
        fallback=upsample,
        prefix_steps=prefix_steps,
    )
    lifted_vae = _force_spatial(lifted_vae, target_lat_h, target_lat_w, upsample)

    rho = float(pack.get("rho") or 0)
    if rho > 1e-8:
        pix = pixel_anchor_latent(
            z0_low_vae,
            vae=vae,
            src_width=low_w,
            src_height=low_h,
            dst_width=high_w,
            dst_height=high_h,
        )
        if pix is not None:
            pix = _force_spatial(pix, target_lat_h, target_lat_w, upsample)
            lifted_vae = mix_rho(
                lifted_vae,
                pix,
                rho,
                float(pack.get("w_min") or 0.5),
                float(pack.get("w_max") or 1.0),
            )

    z0_high = _process_video_format(fmt, "process_in", lifted_vae) if fmt is not None else lifted_vae
    z0_high = _force_spatial(z0_high, target_lat_h, target_lat_w, upsample)
    z0_high = z0_high.to(device=video_x0.device, dtype=video_x0.dtype)
    if prefix_steps > 0:
        z0_vae = _process_video_format(fmt, "process_out", z0_high) if fmt is not None else z0_high
        z0_vae = match_lift_prefix_dc(z0_vae, high_streams[0], prefix_steps)
        z0_high = _process_video_format(fmt, "process_in", z0_vae) if fmt is not None else z0_vae
        z0_high = z0_high.to(device=video_x0.device, dtype=video_x0.dtype)
    notify("selflift_lift", 1)

    sigma_km1 = low_sigmas[-2] if low_sigmas.numel() >= 2 else low_sigmas[0]
    sigma_k = low_sigmas[-1]
    ms = _model_sampling(shifted_hi)
    if ms is None or not callable(getattr(ms, "noise_scaling", None)):
        raise RuntimeError("SelfLift: model_sampling.noise_scaling is required for the transition.")
    if not callable(getattr(ms, "inverse_noise_scaling", None)):
        raise RuntimeError("SelfLift: model_sampling.inverse_noise_scaling is required for the transition.")

    import comfy.sample

    batch_index = latent.get("batch_index") if isinstance(latent, dict) else None
    video_noise = comfy.sample.prepare_noise(
        z0_high, (int(seed) + 1) % (1 << 64), batch_index
    ).to(device=z0_high.device, dtype=z0_high.dtype)
    sigma_km1_t = torch.as_tensor(sigma_km1, device=z0_high.device, dtype=z0_high.dtype)
    video_state = ms.noise_scaling(sigma_km1_t, video_noise, z0_high)
    next_streams = [_euler_step(video_state, z0_high, sigma_km1, sigma_k)]
    for state, denoised in zip(x_streams[1:], x0_streams[1:]):
        next_streams.append(_euler_step(state, denoised, sigma_km1, sigma_k))
    while len(next_streams) < len(high_streams):
        next_streams.append(high_streams[len(next_streams)])

    resume_streams = []
    for part in next_streams:
        sig = torch.as_tensor(sigma_k, device=part.device, dtype=part.dtype)
        resume_streams.append(ms.inverse_noise_scaling(sig, part))

    resume_packed = _pack(resume_streams, high_nested, latent.get("samples"))
    resume_samples = _process_latent_out(shifted_hi, resume_packed)
    if high_nested and not _is_nested(resume_samples):
        if torch.is_tensor(resume_samples):
            try:
                import comfy.utils

                shapes = [tuple(p.shape) for p in high_streams]
                resume_samples = _repack_av_streams(
                    list(comfy.utils.unpack_latents(resume_samples, shapes)), latent
                )
            except Exception as exc:
                raise RuntimeError(
                    "SelfLift: process_latent_out flattened AV NestedTensor and unpack failed."
                ) from exc
        else:
            resume_samples = _repack_av_streams(_streams(resume_samples)[0], latent)

    if not _isfinite_av(resume_samples):
        raise RuntimeError("SelfLift: NaN after transition Euler. Check 3D weights / Euler sampler.")

    log.info(
        "SelfLift transition: video %s → %s (target %dx%d), nested=%s, re-noise σ=%.6g→%.6g",
        tuple(video_x0.shape),
        tuple(z0_high.shape),
        target_lat_h,
        target_lat_w,
        high_nested,
        float(torch.as_tensor(sigma_km1).reshape(-1)[0]),
        float(torch.as_tensor(sigma_k).reshape(-1)[0]),
    )

    high_noise = None
    high_zero_noise = True
    if prefix_steps > 0:
        # Timeline-style HQ inpaint resume: latent_image stays the continue
        # pin (empty generate region). Encode the lifted Euler state into noise
        # so locked tokens remain the true high-res predecessor at the model
        # input, not a 3D-upscaled copy.
        clean_in = _process_latent_in(shifted_hi, latent["samples"])
        clean_streams, _ = _streams(clean_in)
        sigma_resume = float(torch.as_tensor(sigma_k).detach().float().reshape(-1)[0])
        noise_scale = float(getattr(ms, "noise_scale", 1.0) or 1.0)
        denom = max(sigma_resume * noise_scale, 1e-6)
        noise_parts = []
        for idx, (state, clean) in enumerate(zip(next_streams, clean_streams)):
            clean_t = clean.to(device=state.device, dtype=state.dtype)
            noise = (state - (1.0 - sigma_resume) * clean_t) / denom
            if idx == 0:
                unit = comfy.sample.prepare_noise(
                    clean_t, (int(seed) + 1) % (1 << 64), batch_index
                ).to(device=clean_t.device, dtype=clean_t.dtype)
                t_pin = min(prefix_steps, int(noise.shape[2]), int(unit.shape[2]))
                if t_pin > 0:
                    noise = noise.clone()
                    noise[:, :, :t_pin] = unit[:, :, :t_pin]
            noise_parts.append(noise)
        high_noise = _pack(noise_parts, high_nested, latent.get("samples"))
        high_zero_noise = False
        high_latent = dict(latent)
        high_latent = stamp_continue_keys(high_latent, latent)
        high_latent = lock_prefix_keep_mask(high_latent, prefix_steps)
        log.info(
            "SelfLift HQ resume: native inpaint keep-mask, prefix=%d tokens, σ=%.6g",
            prefix_steps,
            sigma_resume,
        )
    else:
        high_latent = dict(latent)
        high_latent["samples"] = resume_samples
        high_latent = stamp_continue_keys(high_latent, latent)

    last_ok = {"samples": resume_samples}

    def _high_state(step, total, x0, x):
        if x0 is not None and _isfinite_av(x0):
            last_ok["samples"] = x0
        _capture_preview(int(step), int(total), x0, x)

    notify("selflift_high", 0)
    try:
        high_out = sample_single_stage(
            model=high_model,
            positive=positive,
            negative=negative,
            latent=high_latent,
            seed=seed,
            cfg=cfg,
            steps=max(1, n_steps - k),
            sampler_name=sampler_use,
            scheduler=scheduler,
            shift_video=shift_video,
            shift_audio=shift_audio,
            sigmas=high_sigmas,
            on_phase=None,
            on_step_preview=None,
            on_step_state=_high_state,
            preview_every=-1 if on_step_preview is None else preview_every,
            after_shift=after_shift,
            shift_cache=shift_cache,
            enable_tiling=bool(pack.get("enable_tiling")),
            tile_count=int(pack.get("tile_count") or 2),
            tile_overlap=int(pack.get("tile_overlap") or 128),
            zero_noise=high_zero_noise,
            noise_override=high_noise,
        )
    except Exception:
        raise
    notify("selflift_high", 1)

    if isinstance(high_out, dict) and "samples" in high_out:
        samples = high_out["samples"]
    else:
        samples = high_out
        high_out = {"samples": samples}
    if not _isfinite_av(samples):
        log.warning("SelfLift high-res output has NaN; falling back to last finite x0.")
        high_out = dict(high_out)
        recovered = _from_callback(last_ok["samples"], high_latent)
        try:
            high_out["samples"] = _process_latent_out(shifted_hi, recovered)
        except Exception:
            high_out["samples"] = recovered

    return high_out, low_carry
