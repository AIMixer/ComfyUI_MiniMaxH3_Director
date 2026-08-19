"""Second-sample (refine) and optional pixel upscale for Director segments."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from ..lib.image_prep import ensure_minimax_canvas
from .core_sampling import sample_single_stage
from .refine_pack import refine_model_for, refine_passes_for, refine_seed_for, refine_steps_for

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


def _latent_upscale_video(video_latent: dict, width: int, height: int) -> dict:
    """Upscale the video latent in latent space (no VAE round trip).

    Audio latent is intentionally left untouched by the caller. Reuses ComfyUI's
    LatentUpscale (bilinear, spatial-only; H3 video latent is (B, C, T, H, W)).
    """
    from nodes import LatentUpscale

    tw, th = ensure_minimax_canvas(int(width), int(height))
    out = LatentUpscale().upscale(video_latent, "bilinear", tw, th, "disabled")
    if isinstance(out, (tuple, list)):
        return out[0]
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
    chunk: int = 4,
    *,
    on_phase=None,
    pbar=None,
) -> torch.Tensor:
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

    n = int(images.shape[0])
    parts = []
    node = ImageUpscaleWithModel()
    for i in range(0, n, max(1, chunk)):
        batch = images[i : i + chunk]
        out = node.upscale(upscale_model, batch)
        frame = _unpack(out)[0]
        parts.append(frame)
        _report_upscale_frames(min(i + int(frame.shape[0]), n), n, on_phase=on_phase, pbar=pbar)
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
                upscale_model, images, on_phase=on_phase, pbar=pbar,
            )
        except Exception as exc:
            log.warning("Upscale model failed (%s); falling back to interpolate.", exc)
            work = images
    h, w = int(work.shape[1]), int(work.shape[2])
    if w != width or h != height:
        work = _scale_images(work, width, height)
    _report_upscale_frames(n, n, on_phase=on_phase, pbar=pbar)
    return work


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
    preview_every: int | None = None,
    first_pass_images: torch.Tensor | None = None,
    trim_frames: int = 0,
    on_pass: RefinePassCallback | None = None,
    sampler_obj=None,
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
    denoise = float(pack.get("denoise") or 0.25)
    r_steps = refine_steps_for(pack, first_steps)
    n_passes = refine_passes_for(pack)
    refine_model = refine_model_for(pack, model)
    note_parts = [f"{mode} denoise={denoise:.2f} steps={r_steps}"]
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
        if mode == "upscale":
            tw = int(pack.get("target_width") or 0)
            th = int(pack.get("target_height") or 0)
            if tw <= 0 or th <= 0:
                tw, th = ensure_minimax_canvas(
                    max(int(getattr(plan, "width", 1280) or 1280), 32),
                    max(int(getattr(plan, "height", 720) or 720), 32),
                )
            if on_phase:
                on_phase("upscale", 0)
            video_latent, audio_latent = _split_av(work)
            method = pack.get("upscale_method") or "lanczos"
            if method == "latent":
                log.info(
                    "Director upscale: latent-space × → %d×%d",
                    int(tw),
                    int(th),
                )
                up_video = _latent_upscale_video(video_latent, tw, th)
                work = _join_av(up_video, audio_latent, work)
                if pin_frames > 0:
                    try:
                        frames = _decode_video(vae, up_video)
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
                            "Segment %s refine latent upscale re-pin failed (%s); "
                            "second sample continues without a new pin.",
                            int(getattr(seg, "index", 0)) + 1,
                            exc,
                        )
                note_parts.append(f"{tw}×{th} latent")
            else:
                if first_pass_images is not None:
                    frames = first_pass_images
                else:
                    frames = _decode_video(vae, video_latent)
                if method == "nvidia_rtx_vsr":
                    how = "nvidia_rtx_vsr"
                elif pack.get("has_upscale_model"):
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
                    upscale_model=pack.get("upscale_model"),
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
                if method == "nvidia_rtx_vsr":
                    note_parts.append("nvidia_rtx_vsr")
                elif pack.get("has_upscale_model"):
                    note_parts.append("upscale_model")
                else:
                    note_parts.append("lanczos")
            if on_phase:
                on_phase("upscale", 1)

        # Pass 1 samples after optional upscale; later passes are same-canvas refine only.
        for i in range(n_passes):
            log.info(
                "Director refine pass %d/%d (steps=%d%s)",
                i + 1,
                n_passes,
                r_steps,
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
                steps=r_steps,
                sampler_name=sampler_name,
                scheduler=scheduler,
                shift_video=shift_video,
                shift_audio=shift_audio,
                denoise=denoise,
                sampler_obj=sampler_obj,
                on_phase=None,
                on_step_preview=on_step_preview,
                preview_every=(
                    preview_every if preview_every is not None
                    else max(1, (int(r_steps) + 11) // 12)
                ),
                phase_name="refine",
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
