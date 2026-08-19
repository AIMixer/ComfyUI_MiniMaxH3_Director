"""Single-stage sampling for MiniMax H3 (SigmaShift + KSampler).

Supports the official ComfyUI pipeline plus the flexible external-sampling path:
- ``sampler_obj`` (SAMPLER from KSamplerSelect / third-party nodes) and ``sigmas``
  (SIGMAS from BasicScheduler / third-party scheduler nodes).
- ``sigma_refine``: densify the low-sigma tail of the first pass (Sigma 精修).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]
StepPreviewCallback = Callable[[int, int, Any], None]


def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output type: {type(out)!r}")


def _tail_slice(sigmas: torch.Tensor, steps: int, denoise: float) -> torch.Tensor:
    """Keep only the low-sigma tail (mirrors KSampler.set_steps denoise logic)."""
    if denoise is None or denoise >= 0.9999:
        return sigmas
    n = int(int(steps) * float(denoise))
    if n <= 0:
        return torch.FloatTensor([])
    return sigmas[-(n + 1):]


def _remap_sigmas_through_shift(model, model_shifted, sigmas: torch.Tensor) -> torch.Tensor:
    """Map external SIGMAS (computed from the unshifted schedule) onto the shifted one.

    MiniMaxH3SigmaShift swaps the model's ``model_sampling``; a scheduler node
    wired to the raw model produces unshifted sigmas. Convert each sigma to its
    timestep on the original schedule, then back to a sigma on the shifted one.
    Best-effort: on any failure the sigmas are used as-is.
    """
    if sigmas is None or len(sigmas) == 0:
        return sigmas
    try:
        s = sigmas.float()
        orig_ms = model.get_model_object("model_sampling")
        shift_ms = model_shifted.get_model_object("model_sampling")
        t = orig_ms.timestep(s)
        mapped = shift_ms.sigma(t)
        if int(s[-1]) == 0:
            mapped = torch.cat([mapped[:-1], torch.zeros_like(mapped[-1:])])
        return mapped.to(dtype=sigmas.dtype, device=sigmas.device)
    except Exception as exc:
        log.debug("External sigma remap skipped (%s); using sigmas as-is.", exc)
        return sigmas


def _densify_low_sigma_tail(
    sigmas: torch.Tensor,
    tail_extra: int,
    tail_frac: float,
) -> torch.Tensor:
    """Insert extra steps into the low-sigma tail, keeping the coarse head intact.

    ``sigmas`` has ``steps + 1`` entries ending in 0. The last ``tail_frac`` of
    the denoising steps are re-interpolated geometrically with ``tail_extra``
    extra points so the fine-detail phase gets more steps. Total entries become
    ``steps + tail_extra + 1`` (one extra transition per extra point).
    """
    tail_extra = int(tail_extra)
    if tail_extra <= 0:
        return sigmas
    steps = len(sigmas) - 1
    if steps < 2:
        return sigmas
    frac = max(0.05, min(0.9, float(tail_frac)))
    head_steps = int(round(steps * (1.0 - frac)))
    head_steps = max(1, min(steps - 1, head_steps))
    head = sigmas[: head_steps + 1]
    tail_nonzero = sigmas[head_steps:-1]
    if tail_nonzero.numel() < 1:
        return sigmas
    n_tail = int(tail_nonzero.numel()) + tail_extra
    lo = float(tail_nonzero[0].clamp_min(1e-12).log())
    hi = float(tail_nonzero[-1].clamp_min(1e-12).log())
    new_tail = torch.exp(torch.linspace(lo, hi, n_tail)).to(dtype=sigmas.dtype)
    return torch.cat([head[:-1], new_tail, sigmas[-1:]])


def _build_sigmas(
    model_shifted,
    *,
    scheduler: str,
    steps: int,
    denoise: float,
    sigma_refine: bool,
    sigma_tail_steps: int,
    sigma_tail_frac: float,
) -> torch.Tensor:
    """Compute the first-pass sigma schedule (shifted), densifying the tail when asked."""
    import comfy.samplers

    shift_ms = model_shifted.get_model_object("model_sampling")
    base = comfy.samplers.calculate_sigmas(shift_ms, scheduler, int(steps)).float()
    if sigma_refine:
        base = _densify_low_sigma_tail(base, int(sigma_tail_steps), sigma_tail_frac)
    return _tail_slice(base, steps, denoise)


def sample_single_stage(
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
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    on_phase: PhaseCallback | None = None,
    on_step_preview: StepPreviewCallback | None = None,
    preview_every: int = 1,
    denoise: float = 1.0,
    phase_name: str = "sample",
    sampler_obj=None,
    sigmas=None,
    sigma_refine: bool = False,
    sigma_tail_steps: int = 1,
    sigma_tail_frac: float = 0.3,
):
    import comfy.sample
    import comfy.utils
    import latent_preview
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    notify(phase_name, 0)
    shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
    model_shifted = _unpack_node_output(shifted)[0]

    neg = negative if negative else []
    steps = int(steps)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model_shifted,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    noise = comfy.sample.prepare_noise(
        latent_image,
        int(seed),
        latent.get("batch_index", None),
    )
    noise_mask = latent.get("noise_mask", None)

    # Resolve the sigma schedule: external SIGMAS > sigma-refine / sampler path.
    run_sigmas = None
    if sigmas is not None:
        run_sigmas = _remap_sigmas_through_shift(model, model_shifted, sigmas)
    elif sampler_obj is not None or sigma_refine:
        run_sigmas = _build_sigmas(
            model_shifted,
            scheduler=scheduler,
            steps=steps,
            denoise=denoise,
            sigma_refine=sigma_refine,
            sigma_tail_steps=sigma_tail_steps,
            sigma_tail_frac=sigma_tail_frac,
        )
    if run_sigmas is not None:
        run_sigmas = run_sigmas.to(device=latent_image.device) if len(run_sigmas) else run_sigmas
        run_sigmas = run_sigmas.to(dtype=latent_image.dtype) if len(run_sigmas) else run_sigmas

    n_steps = int(len(run_sigmas) - 1) if run_sigmas is not None and len(run_sigmas) > 0 else steps
    base_cb = latent_preview.prepare_callback(model_shifted, n_steps)
    every = max(1, int(preview_every))

    def callback(step, x0, x, total_steps):
        if on_step_preview is not None:
            try:
                if step % every == 0 or step >= max(0, int(total_steps) - 1):
                    on_step_preview(int(step), int(total_steps), x0)
            except Exception as exc:
                log.debug("Step preview callback skipped: %s", exc)
        if base_cb is not None:
            base_cb(step, x0, x, total_steps)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    if sampler_obj is not None:
        if run_sigmas is None:
            run_sigmas = _build_sigmas(
                model_shifted,
                scheduler=scheduler,
                steps=steps,
                denoise=denoise,
                sigma_refine=False,
                sigma_tail_steps=0,
                sigma_tail_frac=0.3,
            )
            run_sigmas = run_sigmas.to(device=latent_image.device).to(dtype=latent_image.dtype)
        samples = comfy.sample.sample_custom(
            model_shifted,
            noise,
            float(cfg),
            sampler_obj,
            run_sigmas,
            positive,
            neg,
            latent_image,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=int(seed),
        )
    else:
        samples = comfy.sample.sample(
            model_shifted,
            noise,
            steps,
            float(cfg),
            sampler_name,
            scheduler,
            positive,
            neg,
            latent_image,
            denoise=float(max(0.0, min(1.0, denoise))),
            noise_mask=noise_mask,
            sigmas=run_sigmas,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=int(seed),
        )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    notify(phase_name, 1)
    return out