"""Second-sample (refine) and optional pixel upscale for Director segments."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from ..lib.image_prep import ensure_minimax_canvas
from .core_sampling import sample_single_stage
from .refine_pack import (
    latent_upscale_model_name,
    refine_model_for,
    refine_needs_canvas,
    refine_passes_for,
    refine_seed_for,
    refine_sigmas_override,
    refine_uses_h3_latent,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.refine")

PhaseCallback = Callable[[str, float], None]
StepPreviewCallback = Callable[[int, int, Any], None]
RefinePassCallback = Callable[[int, int, dict], None]


def _make_upscale_pbar(total: int):
    try:
        import comfy.utils

        if getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True):
            return comfy.utils.ProgressBar(max(1, int(total)))
    except Exception:
        pass
    return None


def _report_upscale_frames(done: int, total: int, *, on_phase=None, pbar=None) -> None:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    if pbar is not None:
        pbar.update_absolute(done)
    if on_phase is not None and (done == 0 or done == total or done % 4 == 0):
        on_phase("upscale", done / total)
    if done == 0 or done == total or done % 8 == 0:
        log.info("upscale %d/%d (%.0f%%)", done, total, 100.0 * done / total)


def _unpack(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    return (out,)


def _latent_without_mask(latent: dict) -> dict:
    out = dict(latent)
    out.pop("noise_mask", None)
    return out


def _split_av(samples: dict):
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent

    sep = LTXVSeparateAVLatent.execute(samples)
    video_latent, audio_latent = _unpack(sep)[:2]
    return video_latent, audio_latent


def _decode_video(vae, video_latent):
    from nodes import VAEDecode

    images, = VAEDecode().decode(vae, video_latent)
    return images


def _encode_video(vae, images) -> dict:
    from nodes import VAEEncode

    encoded = VAEEncode().encode(vae, images)
    latent = _unpack(encoded)[0]
    if not isinstance(latent, dict):
        latent = {"samples": latent}
    return latent


def _join_av(video_latent: dict, audio_latent, template: dict) -> dict:
    v = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
    a = audio_latent.get("samples") if isinstance(audio_latent, dict) else audio_latent
    out = dict(template)
    out.pop("noise_mask", None)
    try:
        from comfy_extras.nodes_lt import LTXVConcatAVLatent

        joined = LTXVConcatAVLatent.execute(video_latent, audio_latent)
        packed = _unpack(joined)[0]
        if isinstance(packed, dict) and "samples" in packed:
            return packed
        out["samples"] = packed
        return out
    except Exception:
        pass
    try:
        import comfy.nested_tensor

        if a is not None:
            out["samples"] = comfy.nested_tensor.NestedTensor((v, a))
        else:
            out["samples"] = v
        return out
    except Exception:
        if a is not None:
            out["samples"] = (v, a)
        else:
            out["samples"] = v
        return out


def _scale_images(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    from comfy.utils import common_upscale

    rgb = images[..., :3]
    return common_upscale(
        rgb.movedim(-1, 1), int(width), int(height), "lanczos", "disabled"
    ).movedim(1, -1)


def _upscale_with_rtx_vsr(
    images: torch.Tensor,
    width: int,
    height: int,
    *,
    on_phase=None,
    pbar=None,
) -> torch.Tensor:
    """NVIDIA RTX Video Super Resolution to an explicit canvas (same idea as KJNodes)."""
    try:
        import nvvfx
    except ImportError as exc:
        raise ImportError(
            "nvidia_rtx_vsr 需要 nvidia-vfx，并使用兼容的 NVIDIA GPU。"
            "可 pip install nvidia-vfx，或把 upscale_method 改回 lanczos。"
        ) from exc

    quality = getattr(getattr(nvvfx, "effects", None), "QualityLevel", None)
    level = getattr(quality, "ULTRA", None) if quality is not None else None
    ctx = nvvfx.VideoSuperRes(level) if level is not None else nvvfx.VideoSuperRes()
    nvvfx_sr = ctx.__enter__()
    try:
        nvvfx_sr.output_width = max(8, round(int(width) / 8) * 8)
        nvvfx_sr.output_height = max(8, round(int(height) / 8) * 8)
        if hasattr(nvvfx_sr, "load"):
            nvvfx_sr.load()
        frames_chw = images[..., :3].movedim(-1, 1).contiguous()
        if frames_chw.device.type != "cuda":
            frames_chw = frames_chw.cuda()
        upscaled = []
        n = int(frames_chw.shape[0])
        for i in range(n):
            dlpack_out = nvvfx_sr.run(frames_chw[i]).image
            upscaled.append(torch.from_dlpack(dlpack_out).clone())
            _report_upscale_frames(i + 1, n, on_phase=on_phase, pbar=pbar)
        return torch.stack(upscaled, dim=0).movedim(1, -1)
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass


def _upscale_with_model(
    upscale_model,
    images: torch.Tensor,
    *,
    width: int,
    height: int,
    chunk: int = 1,
    on_phase=None,
    pbar=None,
) -> torch.Tensor:
    """Load the pixel upscaler once, then snap each chunk to the target canvas.

    Calling ImageUpscaleWithModel per chunk re-runs load_models_gpu and floods
    the log with ``Requested to load RRDBNet`` / dynamic-VRAM prepare lines.
    4× output is downscaled immediately so a 10s clip never sits at 4K.
    """
    import comfy.model_management as model_management
    import comfy.utils

    n = int(images.shape[0])
    step = max(1, int(chunk))
    device = upscale_model.patcher.load_device
    scale = float(max(getattr(upscale_model, "scale", 1.0) or 1.0, 1.0))
    sample = images[: min(step, n)]
    # Official ImageUpscaleWithModel asks for ~4–5GB and evicts H3, which then
    # re-prepares RRDBNet every chunk. Ask only for the upscaler + one tile.
    memory_required = (
        128 * 1024 * 1024
        + int(sample.nelement() * sample.element_size())
    )
    model_management.load_models_gpu(
        [upscale_model.patcher],
        memory_required=memory_required,
        force_full_load=True,
    )
    log.info(
        "upscale model: %d frame(s) → %d×%d, chunk=%d (loaded once)",
        n,
        int(width),
        int(height),
        step,
    )

    output_device = model_management.intermediate_device()
    overlap = 32
    parts = []
    for i in range(0, n, step):
        batch = images[i : i + step]
        in_img = batch.movedim(-1, -3).to(device)
        oom = True
        tile = 512
        while oom:
            try:
                s = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a.float()),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=scale,
                    pbar=None,
                    output_device=output_device,
                )
                oom = False
            except Exception as exc:
                model_management.raise_non_oom(exc)
                tile //= 2
                if tile < 128:
                    raise
        frame = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        try:
            frame = frame.to(model_management.intermediate_dtype())
        except Exception:
            pass
        fh, fw = int(frame.shape[1]), int(frame.shape[2])
        if fw != int(width) or fh != int(height):
            frame = _scale_images(frame, int(width), int(height))
        parts.append(frame.cpu())
        del in_img, s, frame
        _report_upscale_frames(min(i + int(batch.shape[0]), n), n, on_phase=on_phase, pbar=pbar)
    return torch.cat(parts, dim=0)


def upscale_image_batch(
    images: torch.Tensor,
    *,
    width: int,
    height: int,
    upscale_model=None,
    upscale_method: str = "lanczos",
    on_phase=None,
) -> torch.Tensor:
    width, height = ensure_minimax_canvas(width, height)
    method = str(upscale_method or "lanczos").strip().lower()
    n = int(images.shape[0])
    pbar = _make_upscale_pbar(n)
    _report_upscale_frames(0, n, on_phase=on_phase, pbar=pbar)
    work = images
    if method == "nvidia_rtx_vsr":
        try:
            work = _upscale_with_rtx_vsr(
                images, width, height, on_phase=on_phase, pbar=pbar,
            )
        except Exception as exc:
            log.warning("nvidia_rtx_vsr failed (%s); falling back to interpolate.", exc)
            work = images
    elif upscale_model is not None:
        try:
            work = _upscale_with_model(
                upscale_model,
                images,
                width=width,
                height=height,
                on_phase=on_phase,
                pbar=pbar,
            )
        except Exception as exc:
            log.warning("Upscale model failed (%s); falling back to interpolate.", exc)
            work = images
    h, w = int(work.shape[1]), int(work.shape[2])
    if w != width or h != height:
        work = _scale_images(work, width, height)
    _report_upscale_frames(n, n, on_phase=on_phase, pbar=pbar)
    return work


def _source_canvas(plan, first_pass_images: torch.Tensor | None) -> tuple[int, int]:
    if first_pass_images is not None and getattr(first_pass_images, "ndim", 0) >= 3:
        return int(first_pass_images.shape[2]), int(first_pass_images.shape[1])
    return (
        max(int(getattr(plan, "width", 1280) or 1280), 32),
        max(int(getattr(plan, "height", 720) or 720), 32),
    )


def _resolve_refine_canvas(plan, pack: dict) -> tuple[int, int]:
    tw = int(pack.get("target_width") or 0)
    th = int(pack.get("target_height") or 0)
    if tw > 0 and th > 0:
        return ensure_minimax_canvas(tw, th)
    return ensure_minimax_canvas(
        max(int(getattr(plan, "width", 1280) or 1280), 32),
        max(int(getattr(plan, "height", 720) or 720), 32),
    )


def _apply_h3_latent_upscale(
    work: dict,
    pack: dict,
    *,
    plan,
    tw: int,
    th: int,
    first_pass_images: torch.Tensor | None,
    pin_frames: int,
    task_key: str,
    vae,
    refine_positive,
    on_phase=None,
) -> tuple[dict, Any, list[str]]:
    from .h3_latent_upscale import upscale_h3_video_latent

    src_w, src_h = _source_canvas(plan, first_pass_images)
    if on_phase:
        on_phase("upscale", 0)
    video_latent, audio_latent = _split_av(work)
    model_name = latent_upscale_model_name(pack)
    latent_mod = pack.get("latent_upscale_module")
    log.info(
        "Director H3 latent upscale: %s %d×%d → %d×%d",
        model_name or ("(wired)" if latent_mod is not None else "(missing)"),
        src_w,
        src_h,
        tw,
        th,
    )
    encoded = upscale_h3_video_latent(
        video_latent,
        target_width=tw,
        target_height=th,
        source_width=src_w,
        source_height=src_h,
        model_name=model_name,
        model=latent_mod,
    )
    if isinstance(encoded, dict):
        encoded.pop("noise_mask", None)
    if isinstance(audio_latent, dict):
        audio_latent = dict(audio_latent)
        audio_latent.pop("noise_mask", None)
    work = _join_av(encoded, audio_latent, work)
    notes = [f"{tw}×{th}", "h3_latent"]
    if pin_frames > 0 and first_pass_images is not None:
        try:
            prefix = first_pass_images[:pin_frames]
            ph, pw = int(prefix.shape[1]), int(prefix.shape[2])
            if pw != tw or ph != th:
                prefix = _scale_images(prefix, tw, th)
            refine_positive, pinned = _repin_after_upscale(
                refine_positive,
                work,
                vae=vae,
                prefix_frames=prefix,
                trim_frames=pin_frames,
                task_key=task_key,
            )
            if pinned:
                notes.append(f"re-pin {pin_frames}f")
        except Exception as exc:
            log.warning("H3 latent upscale re-pin failed (%s); continuing.", exc)
    if on_phase:
        on_phase("upscale", 1)
    return work, refine_positive, notes


def _repin_after_upscale(
    positive,
    latent: dict,
    *,
    vae,
    prefix_frames: torch.Tensor,
    trim_frames: int,
    task_key: str,
):
    """Rebuild motion-context keyframes at the new canvas. Does not touch first-pass."""
    from .h3_motion_context import apply_motion_context

    n = min(int(trim_frames), int(prefix_frames.shape[0]))
    if n < 1:
        return positive, False
    new_positive, _, _ = apply_motion_context(
        positive,
        latent,
        vae=vae,
        context_length=n,
        context_frames=prefix_frames[:n],
        continue_audio=False,
        keep_existing_keyframes=(task_key == "fl2v"),
    )
    return new_positive, True


def apply_segment_refine(
    plan,
    seg,
    *,
    samples: dict,
    model,
    vae,
    audio_vae=None,
    positive,
    negative,
    seed: int,
    cfg: float,
    first_steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float,
    shift_audio: float,
    on_phase: PhaseCallback | None = None,
    on_step_preview: StepPreviewCallback | None = None,
    first_pass_images: torch.Tensor | None = None,
    trim_frames: int = 0,
    on_pass: RefinePassCallback | None = None,
) -> tuple[dict, str]:
    """Run optional refine/upscale second sample. Never raises — returns first-pass on failure.

    ``first_pass_images``: already-decoded first-pass frames (skips a second VAE
    decode in upscale mode). Includes motion-context prefix when continuity is on.
    ``trim_frames``: pinned prefix length from first pass (0 = no continuity).
    ``passes``: sample this many times after first-pass. Upscale (if any) runs
    once before pass 1; later passes are same-canvas refine only.
    ``on_pass(pass_index, n_passes, latent)``: after each sample (1-based).
    First-pass sampling is unchanged — this only runs after it.
    """
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return samples, ""
    if pack.get("skip_fl2v", True) and getattr(seg, "task_key", "") == "fl2v":
        return samples, "refine skipped (fl2v)"

    mode = pack.get("mode") or "refine"
    n_passes = refine_passes_for(pack)
    refine_model = refine_model_for(pack, model)
    note_parts = [mode]
    if mode != "latent_upscale":
        if n_passes > 1:
            note_parts.append(f"passes={n_passes}")
        if refine_model is not model:
            note_parts.append("custom model")
    pin_frames = max(0, int(trim_frames or 0))
    task_key = str(getattr(seg, "task_key", "") or "")

    # Same-size refine keeps any first-pass mask so a continuity lock still holds.
    # No continuity → drop stray masks so refine can touch the whole clip.
    work = dict(samples) if pin_frames > 0 else _latent_without_mask(samples)
    refine_positive = positive
    last_ok = samples
    try:
        if refine_needs_canvas(pack):
            tw, th = _resolve_refine_canvas(plan, pack)
            if refine_uses_h3_latent(pack):
                work, refine_positive, extra = _apply_h3_latent_upscale(
                    work,
                    pack,
                    plan=plan,
                    tw=tw,
                    th=th,
                    first_pass_images=first_pass_images,
                    pin_frames=pin_frames,
                    task_key=task_key,
                    vae=vae,
                    refine_positive=refine_positive,
                    on_phase=on_phase,
                )
                note_parts.extend(extra)
                last_ok = work
            else:
                if on_phase:
                    on_phase("upscale", 0)
                video_latent, audio_latent = _split_av(work)
                if first_pass_images is not None:
                    frames = first_pass_images
                else:
                    frames = _decode_video(vae, video_latent)
                method = pack.get("upscale_method") or "h3_latent"
                pixel_model = pack.get("upscale_model")
                if method == "nvidia_rtx_vsr":
                    how = "nvidia_rtx_vsr"
                elif pixel_model is not None:
                    how = "upscale_model"
                else:
                    how = "lanczos"
                log.info(
                    "Director upscale: %d frames via %s → %d×%d",
                    int(frames.shape[0]),
                    how,
                    tw,
                    th,
                )
                frames = upscale_image_batch(
                    frames,
                    width=tw,
                    height=th,
                    upscale_model=pixel_model,
                    upscale_method=method,
                    on_phase=on_phase,
                )
                encoded = _encode_video(vae, frames)
                work = _join_av(encoded, audio_latent, work)
                if pin_frames > 0:
                    try:
                        refine_positive, pinned = _repin_after_upscale(
                            refine_positive,
                            work,
                            vae=vae,
                            prefix_frames=frames,
                            trim_frames=pin_frames,
                            task_key=task_key,
                        )
                        if pinned:
                            note_parts.append(f"re-pin {pin_frames}f")
                    except Exception as exc:
                        log.warning(
                            "Segment %s refine upscale re-pin failed (%s); "
                            "second sample continues without a new pin.",
                            int(getattr(seg, "index", 0)) + 1,
                            exc,
                        )
                note_parts.append(f"{tw}×{th}")
                note_parts.append(how)
                if on_phase:
                    on_phase("upscale", 1)
                last_ok = work

        if mode == "latent_upscale":
            if on_phase:
                on_phase("refine", 1)
            return work, "refine " + ", ".join(note_parts)

        wired_sigmas = bool(pack.get("has_sigmas_tensor") or pack.get("sigmas_tensor") is not None)
        sigma_list = refine_sigmas_override(pack)
        if sigma_list is None:
            raise ValueError(
                "Refine 二采需要把 BasicScheduler 或 ManualSigmas 接到 sigmas 口。"
            )
        sigma_sampler = str(pack.get("sampler") or "euler")
        sigma_steps = max(1, len(sigma_list) - 1)
        how = "sigmas wired" if wired_sigmas else f"sigma {sigma_sampler}"
        note_parts.append(f"{how} {sigma_steps}-step")
        if refine_model is not model and sigma_steps <= 4:
            log.warning(
                "Refine sigma pass is short. "
                "A custom refine_model without Turbo can look like noise. "
                "Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, "
                "or use a matching second-pass UNET."
            )
        # Pass 1 samples after optional upscale; later passes are same-canvas refine only.
        for i in range(n_passes):
            log.info(
                "Director refine pass %d/%d (%s %s %d-step%s)",
                i + 1,
                n_passes,
                "sigmas wired" if wired_sigmas else "sigma",
                sigma_sampler,
                sigma_steps,
                ", custom model" if refine_model is not model else "",
            )
            if on_phase:
                on_phase("refine", (i + 0.5) / n_passes)
            work = sample_single_stage(
                model=refine_model,
                positive=refine_positive,
                negative=negative,
                latent=work,
                seed=refine_seed_for(pack, seed, pass_index=i),
                cfg=cfg,
                steps=sigma_steps,
                sampler_name=sigma_sampler,
                scheduler=scheduler,
                shift_video=shift_video,
                shift_audio=shift_audio,
                denoise=1.0,
                on_phase=None,
                on_step_preview=on_step_preview,
                preview_every=-1,
                phase_name="refine",
                sigmas=sigma_list,
                apply_shift=True,
            )
            last_ok = work
            if on_pass is not None:
                try:
                    on_pass(i + 1, n_passes, work)
                except Exception as exc:
                    log.warning(
                        "Segment %s refine pass %d hook failed (%s).",
                        int(getattr(seg, "index", 0)) + 1,
                        i + 1,
                        exc,
                    )
        if on_phase:
            on_phase("refine", 1)
        return work, "refine " + ", ".join(note_parts)
    except Exception as exc:
        log.warning(
            "Segment %s refine failed (%s); keeping %s.",
            int(getattr(seg, "index", 0)) + 1,
            exc,
            "last successful pass" if last_ok is not samples else "first-pass latent",
        )
        if last_ok is not samples:
            return last_ok, f"refine FAILED ({exc}); kept last good pass"
        return samples, f"refine FAILED ({exc}); used first pass"
