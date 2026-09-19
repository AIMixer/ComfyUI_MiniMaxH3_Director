"""Encode crop frames into the video stream of an H3 joint AV latent.

Adapted from ComfyUI-H3-FaceRefine H3InjectVideoLatent (MIT).
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.face_refine")


def inject_video_latent(av_latent: dict, images: torch.Tensor, vae) -> dict:
    samples = av_latent.get("samples")
    if samples is None:
        raise KeyError('LATENT is missing "samples".')
    import comfy.nested_tensor

    is_nested = isinstance(samples, comfy.nested_tensor.NestedTensor) or getattr(
        samples, "is_nested", False
    )
    if not is_nested:
        raise ValueError(
            "FaceRefine 需要 MiniMax H3 联合 AV latent（NestedTensor）。"
        )
    members = list(samples.unbind())
    video_tmpl = members[0]
    encoded = vae.encode(images[..., :3])
    if encoded.ndim == 4:
        encoded = encoded.unsqueeze(0).movedim(1, 2)
    tgt_t, tgt_h, tgt_w = video_tmpl.shape[-3], video_tmpl.shape[-2], video_tmpl.shape[-1]
    got_t, got_h, got_w = encoded.shape[-3], encoded.shape[-2], encoded.shape[-1]
    if (got_h, got_w) != (tgt_h, tgt_w):
        raise ValueError(
            f"FaceRefine 裁剪画布和 H3 latent 空间不一致：编码 {got_h}x{got_w}，"
            f"期望 {tgt_h}x{tgt_w}。"
        )
    if got_t != tgt_t:
        if got_t > tgt_t:
            encoded = encoded[..., :tgt_t, :, :]
        else:
            pad = video_tmpl[..., : tgt_t - got_t, :, :].to(encoded.device, encoded.dtype)
            encoded = torch.cat([encoded, pad], dim=-3)
        log.info("FaceRefine inject temporal pad/trim: encoded t=%s vs latent t=%s", got_t, tgt_t)
    members[0] = encoded.to(video_tmpl.device, video_tmpl.dtype)
    out = dict(av_latent)
    out["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))
    return out
