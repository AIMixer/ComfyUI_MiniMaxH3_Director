"""One RelayVSR pass on a decoded segment. Matches the official Comfy graph:

GetVideoComponents images → RelayVSRConditioning → KSampler
(positive wired to both positive and negative, steps 1, cfg 1, euler/simple)
→ VSRDecode(Flash VAE, detail) → frames. Audio is left untouched.
"""

from __future__ import annotations

import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.relay_vsr")


def _unpack(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output type: {type(out)!r}")


def apply_segment_relay_vsr(frames, pack: dict):
    import comfy.model_management as mm
    import torch
    from comfy_extras.nodes_wan import RelayVSRConditioning, VSRDecode
    from nodes import common_ksampler

    if not torch.is_tensor(frames) or frames.ndim != 4 or int(frames.shape[0]) < 1:
        return frames
    model = pack.get("model")
    vae = pack.get("vae")
    if model is None or vae is None:
        return frames
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion is None or "RelayVSR" not in type(diffusion).__name__:
        raise RuntimeError(
            "RelayVSR 的 model 需要 RelayVSR UNET"
            "（例如 relayvsr_x2_lightx2v_int8_convrot），不能接 MiniMax H3 UNET。"
        )

    images = frames.detach().float().contiguous().to(mm.intermediate_device())
    positive, latent = _unpack(RelayVSRConditioning.execute(model, images))
    sampled = common_ksampler(
        model,
        int(pack.get("seed") or 0),
        max(1, int(pack.get("steps") or 1)),
        float(pack.get("cfg") if pack.get("cfg") is not None else 1.0),
        str(pack.get("sampler") or "euler"),
        str(pack.get("scheduler") or "simple"),
        positive,
        positive,
        latent,
        denoise=float(pack.get("denoise") if pack.get("denoise") is not None else 1.0),
    )
    decoded = _unpack(
        VSRDecode.execute(
            vae,
            sampled[0],
            images,
            detail=float(pack.get("detail") if pack.get("detail") is not None else 1.0),
        )
    )[0]
    out = decoded.detach().cpu().float().contiguous()
    if int(out.shape[0]) != int(frames.shape[0]):
        raise RuntimeError(
            f"RelayVSR 输出 {int(out.shape[0])} 帧，输入 {int(frames.shape[0])} 帧，无法和音频对齐。"
        )
    log.info(
        "RelayVSR %s → %dx%d (%d frames, detail=%.2f)",
        tuple(frames.shape[1:3]),
        int(out.shape[1]),
        int(out.shape[2]),
        int(out.shape[0]),
        float(pack.get("detail") or 1.0),
    )
    return out
