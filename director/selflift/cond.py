"""Resize MiniMax H3 cond / AV latents to a SelfLift low-res grid."""

from __future__ import annotations

import copy
import inspect
import logging
from typing import Any

import torch
import torch.nn.functional as F

from ..h3_motion_context import _repack_av_streams, _streams_from_latent

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.selflift.cond")


def interpolate_video_5d(video: torch.Tensor, dst_h: int, dst_w: int, mode: str) -> torch.Tensor:
    """Spatially resize BCTHW or CTHW video latent. Time axis is unchanged."""
    squeezed = False
    work = video
    if work.ndim == 4:
        work = work.unsqueeze(0)
        squeezed = True
    if work.ndim != 5:
        return video
    _b, _c, t, h, w = work.shape
    if int(h) == int(dst_h) and int(w) == int(dst_w):
        return video
    align = False if mode == "bilinear" else None
    kwargs: dict[str, Any] = {"mode": mode, "align_corners": align} if mode == "bilinear" else {"mode": mode}
    # B C T H W → (B*T) C H W
    flat = work.permute(0, 2, 1, 3, 4).reshape(_b * t, _c, h, w)
    out = F.interpolate(flat.float(), size=(int(dst_h), int(dst_w)), **kwargs)
    out = out.to(dtype=work.dtype).reshape(_b, t, _c, int(dst_h), int(dst_w)).permute(0, 2, 1, 3, 4)
    if squeezed:
        out = out.squeeze(0)
    return out.contiguous()


def resize_av_video(latent: dict, dst_h: int, dst_w: int, mode: str = "bilinear") -> dict:
    """Resize only the video stream; audio is kept. Also resizes noise_mask video."""
    streams = list(_streams_from_latent(latent))
    video = streams[0]
    streams[0] = interpolate_video_5d(video, dst_h, dst_w, mode)
    out = dict(latent)
    out["samples"] = _repack_av_streams(streams, latent)
    mask = out.get("noise_mask")
    if mask is not None:
        out["noise_mask"] = _resize_noise_mask(mask, dst_h, dst_w, mode, template=mask)
    return out


def _resize_noise_mask(mask, dst_h: int, dst_w: int, mode: str, template=None):
    try:
        if torch.is_tensor(mask) and mask.ndim == 5:
            return interpolate_video_5d(mask, dst_h, dst_w, mode)
        if getattr(mask, "is_nested", False) and hasattr(mask, "unbind"):
            parts = list(mask.unbind())
        elif isinstance(mask, (tuple, list)):
            parts = list(mask)
        else:
            return mask
        if parts and torch.is_tensor(parts[0]) and parts[0].ndim >= 4:
            parts[0] = interpolate_video_5d(parts[0], dst_h, dst_w, mode)
        return _repack_av_streams(parts, {"samples": template} if template is not None else None)
    except Exception as exc:
        log.debug("SelfLift noise_mask resize skipped: %s", exc)
        return mask


def _spatial_match(tensor, src_h: int, src_w: int) -> bool:
    if not torch.is_tensor(tensor) or tensor.ndim < 4:
        return False
    return int(tensor.shape[-2]) == int(src_h) and int(tensor.shape[-1]) == int(src_w)


def _resize_maybe(tensor, src_h: int, src_w: int, dst_h: int, dst_w: int, mode: str):
    if not _spatial_match(tensor, src_h, src_w):
        return tensor
    return interpolate_video_5d(tensor, dst_h, dst_w, mode)


def _rebuild_layout(payload: dict, latent_t: int, tile_h: int, tile_w: int, audio_t: int):
    from comfy.ldm.minimax.model import PackedLayout

    old = payload.get("layout")
    text_len = 0
    signature = getattr(old, "signature", None) if old is not None else None
    if signature:
        text_len = int(signature[0])
    elif old is not None and getattr(old, "segments", None):
        start, stop, kind = old.segments[0]
        if kind == "text":
            text_len = int(stop - start)
    kwargs: dict[str, Any] = {
        "keyframes": payload.get("keyframes"),
        "refs": payload.get("refs"),
    }
    try:
        params = inspect.signature(PackedLayout.__init__).parameters
        if "frame_count" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            kwargs["frame_count"] = payload.get("frame_count")
    except (TypeError, ValueError):
        pass
    even_h = (int(tile_h) + 1) // 2 * 2
    even_w = (int(tile_w) + 1) // 2 * 2
    args = (text_len, int(latent_t), even_h, even_w, int(audio_t))
    try:
        return PackedLayout(*args, **kwargs)
    except TypeError:
        kwargs.pop("frame_count", None)
        return PackedLayout(*args, **kwargs)


def _resize_payload(payload: dict, src_h: int, src_w: int, dst_h: int, dst_w: int, mode: str) -> dict:
    new_payload = dict(payload)
    keyframes = []
    for item in list(payload.get("keyframes") or []):
        copied = dict(item)
        lat = item.get("latent")
        if torch.is_tensor(lat):
            copied["latent"] = _resize_maybe(lat, src_h, src_w, dst_h, dst_w, mode)
        keyframes.append(copied)
    if keyframes:
        new_payload["keyframes"] = keyframes
    cond_video = payload.get("cond_video_latents")
    if cond_video:
        new_payload["cond_video_latents"] = [
            _resize_maybe(item, src_h, src_w, dst_h, dst_w, mode) for item in cond_video
        ]
    layout = payload.get("layout")
    latent_t = int(getattr(layout, "latent_t", 0) or 0)
    audio_t = int(getattr(layout, "audio_t", 0) or 0)
    if latent_t <= 0:
        # PackedLayout stores sizes on the constructor args / signature.
        sig = getattr(layout, "signature", None)
        if sig and len(sig) >= 5:
            latent_t = int(sig[1])
            audio_t = int(sig[4]) if audio_t <= 0 else audio_t
    if latent_t <= 0:
        for kf in keyframes:
            lat = kf.get("latent")
            if torch.is_tensor(lat) and lat.ndim >= 3:
                latent_t = int(lat.shape[-3] if lat.ndim >= 5 else lat.shape[-3])
                break
    new_payload["layout"] = _rebuild_layout(new_payload, max(1, latent_t), dst_h, dst_w, audio_t)
    return new_payload


def _resize_shapes(shapes, src_h: int, src_w: int, dst_h: int, dst_w: int):
    if not shapes:
        return shapes
    out = []
    for shape in shapes:
        try:
            dims = list(shape)
        except TypeError:
            out.append(shape)
            continue
        if len(dims) >= 5 and int(dims[-2]) == int(src_h) and int(dims[-1]) == int(src_w):
            dims[-2] = int(dst_h)
            dims[-1] = int(dst_w)
            out.append(tuple(dims))
        else:
            out.append(shape)
    return type(shapes)(out) if not isinstance(shapes, list) else out


def _resize_model_cond(cond, src_h: int, src_w: int, dst_h: int, dst_w: int, mode: str):
    payload = getattr(cond, "cond", None)
    if isinstance(payload, dict) and (
        payload.get("keyframes") is not None
        or payload.get("layout") is not None
        or payload.get("cond_video_latents") is not None
    ):
        cloned = copy.copy(cond)
        cloned.cond = _resize_payload(payload, src_h, src_w, dst_h, dst_w, mode)
        return cloned
    if torch.is_tensor(payload) and _spatial_match(payload, src_h, src_w):
        cloned = copy.copy(cond)
        cloned.cond = interpolate_video_5d(payload, dst_h, dst_w, mode)
        return cloned
    if isinstance(payload, (list, tuple)) and payload:
        first = payload[0]
        if isinstance(first, (list, tuple)) and len(first) >= 5:
            cloned = copy.copy(cond)
            cloned.cond = _resize_shapes(payload, src_h, src_w, dst_h, dst_w)
            return cloned
    return cond


def resize_positive_spatial(positive, src_h: int, src_w: int, dst_h: int, dst_w: int, mode: str = "bilinear"):
    """Clone cond and rewrite MiniMax payload / keyframes to ``dst`` latent H×W."""
    if int(src_h) == int(dst_h) and int(src_w) == int(dst_w):
        return positive
    cloned = []
    for item in positive or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            cloned.append(item)
            continue
        tensor, meta = item[0], item[1]
        if not isinstance(meta, dict):
            cloned.append(list(item) if isinstance(item, tuple) else item)
            continue
        new_meta = dict(meta)
        mconds = meta.get("model_conds")
        if isinstance(mconds, dict):
            new_mconds = dict(mconds)
            for key, cond in mconds.items():
                new_mconds[key] = _resize_model_cond(cond, src_h, src_w, dst_h, dst_w, mode)
            new_meta["model_conds"] = new_mconds
        kfs = meta.get("minimax_keyframes")
        if isinstance(kfs, list) and kfs:
            resized = []
            for kf in kfs:
                if not isinstance(kf, dict):
                    resized.append(kf)
                    continue
                copied = dict(kf)
                lat = kf.get("latent")
                if torch.is_tensor(lat):
                    copied["latent"] = _resize_maybe(lat, src_h, src_w, dst_h, dst_w, mode)
                resized.append(copied)
            new_meta["minimax_keyframes"] = resized
        cloned.append([tensor, new_meta])
    return cloned
