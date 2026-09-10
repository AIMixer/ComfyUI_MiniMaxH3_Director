"""Optional spatial tiling for Director refine sampling.

Cuts the video latent along the long spatial axis, keeps audio whole, and
blends overlapping tile predictions inside each model call so the outer
sampler (Euler / res_multistep / …) still steps a single shared canvas.

Off by default. When disabled the caller never wraps the sampler, so the
path matches the previous full-frame sample.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.spatial_tiles")

VAE_SPATIAL = 16
# Long-side pixel size at or below this skips tiling (same 16× latent grid as H3).
MAX_PIXELS_NO_TILE = 384


def clamp_n_tiles(raw: Any) -> int:
    try:
        n = int(raw if raw is not None else 2)
    except (TypeError, ValueError):
        n = 2
    return max(1, min(8, n))


def clamp_tile_overlap(raw: Any) -> int:
    try:
        n = int(raw if raw is not None else 128)
    except (TypeError, ValueError):
        n = 128
    return max(0, min(2048, n))


def compute_tile_starts(total: int, n_tiles: int, overlap: int) -> tuple[list[int], int]:
    if n_tiles <= 1:
        return [0], int(total)
    stride = int(round(total / n_tiles))
    tile_size = stride + 2 * overlap
    starts: list[int] = []
    for i in range(n_tiles):
        starts.append(max(0, i * stride - overlap))
    dedup: list[int] = []
    for start in starts:
        if not dedup or start > dedup[-1]:
            dedup.append(start)
    return dedup, tile_size


def _raised_cosine_1d(length: int, ov_left: int, ov_right: int, *, dtype, device) -> torch.Tensor:
    w = torch.ones(length, dtype=dtype, device=device)
    if ov_left > 0:
        n = min(ov_left, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos(t * math.pi)
            w[:n] = torch.minimum(w[:n], fade)
    if ov_right > 0:
        n = min(ov_right, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos((1.0 - t) * math.pi)
            w[-n:] = torch.minimum(w[-n:], fade)
    return w


def _overlap_latent(overlap_pixels: int) -> int:
    overlap_pixels = max(0, int(overlap_pixels))
    units = int(round(overlap_pixels / float(VAE_SPATIAL)))
    if overlap_pixels > 0 and units == 0:
        return 1
    return units


def _crop_spatial_5d(tensor: torch.Tensor, axis: str, start: int, end: int) -> torch.Tensor:
    if axis == "H":
        return tensor[:, :, :, start:end, :].contiguous()
    return tensor[:, :, :, :, start:end].contiguous()


def _pad_even_hw(video: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    pad_h = (-int(video.shape[-2])) % 2
    pad_w = (-int(video.shape[-1])) % 2
    if pad_h or pad_w:
        video = F.pad(video, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
    return video, pad_h, pad_w


def _unpad_hw(video: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h:
        video = video[:, :, :, : video.shape[-2] - pad_h, :]
    if pad_w:
        video = video[:, :, :, :, : video.shape[-1] - pad_w]
    return video


def _crop_packed(packed, full_shapes, axis: str, start: int, end: int):
    import comfy.utils

    if packed is None:
        return None, full_shapes, (0, 0)
    streams = list(comfy.utils.unpack_latents(packed, full_shapes))
    video = streams[0]
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        return packed, full_shapes, (0, 0)
    cropped = _crop_spatial_5d(video, axis, start, end)
    cropped, pad_h, pad_w = _pad_even_hw(cropped)
    streams[0] = cropped
    packed_out, tile_shapes = comfy.utils.pack_latents(streams)
    return packed_out, tile_shapes, (pad_h, pad_w)


def _crop_matching_video(tensor, full_h: int, full_w: int, axis: str, start: int, end: int):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5:
        return tensor
    if int(tensor.shape[-2]) != int(full_h) or int(tensor.shape[-1]) != int(full_w):
        return tensor
    cropped = _crop_spatial_5d(tensor, axis, start, end)
    cropped, _, _ = _pad_even_hw(cropped)
    return cropped


def _iter_payloads(cfg_guider):
    conds = getattr(cfg_guider, "conds", None) or {}
    for group in conds.values():
        if not group:
            continue
        for cond in group:
            if not isinstance(cond, dict):
                continue
            model_conds = cond.get("model_conds") or {}
            yield model_conds


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
    args = (text_len, int(latent_t), int(tile_h), int(tile_w), int(audio_t))
    try:
        return PackedLayout(*args, **kwargs)
    except TypeError:
        kwargs.pop("frame_count", None)
        return PackedLayout(*args, **kwargs)


def _install_tile_payloads(cfg_guider, *, axis: str, start: int, end: int, tile_shapes, full_h: int, full_w: int):
    restorations: list[tuple[Any, str, Any]] = []
    vs = tile_shapes[0]
    tile_h = (int(vs[3]) + 1) // 2 * 2
    tile_w = (int(vs[4]) + 1) // 2 * 2
    latent_t = int(vs[2])
    audio_t = int(tile_shapes[1][-1]) if len(tile_shapes) > 1 else 0

    for model_conds in _iter_payloads(cfg_guider):
        shape_cond = model_conds.get("latent_shapes")
        if shape_cond is not None and hasattr(shape_cond, "cond"):
            restorations.append((shape_cond, "cond", shape_cond.cond))
            shape_cond.cond = tile_shapes

        payload_cond = model_conds.get("minimax_payload")
        payload = getattr(payload_cond, "cond", None) if payload_cond is not None else None
        if not isinstance(payload, dict):
            continue
        restorations.append((payload_cond, "cond", payload))
        new_payload = dict(payload)
        keyframes = []
        for item in list(payload.get("keyframes") or []):
            copied = dict(item)
            copied["latent"] = _crop_matching_video(
                item.get("latent"), full_h, full_w, axis, start, end
            )
            keyframes.append(copied)
        if keyframes:
            new_payload["keyframes"] = keyframes
        cond_video = payload.get("cond_video_latents")
        if cond_video:
            new_payload["cond_video_latents"] = [
                _crop_matching_video(item, full_h, full_w, axis, start, end)
                for item in cond_video
            ]
        new_payload["layout"] = _rebuild_layout(new_payload, latent_t, tile_h, tile_w, audio_t)
        payload_cond.cond = new_payload
    return restorations


def _restore_payloads(restorations: list[tuple[Any, str, Any]]) -> None:
    for obj, attr, value in restorations:
        setattr(obj, attr, value)


def _plan_regions(video_shape, n_tiles: int, overlap_pixels: int):
    _b, _c, _t, height, width = video_shape
    axis = "H" if int(height) >= int(width) else "W"
    axis_size = int(height if axis == "H" else width)
    max_latent = max(1, int(round(MAX_PIXELS_NO_TILE / float(VAE_SPATIAL))))
    if axis_size <= max_latent or n_tiles <= 1:
        return None
    overlap = _overlap_latent(overlap_pixels)
    starts, tile_size = compute_tile_starts(axis_size, n_tiles, overlap)
    if len(starts) <= 1:
        return None
    if tile_size >= axis_size * 0.9:
        log.warning(
            "spatial tiles: single tile %s is ~full axis %s (overlap %spx); little VRAM saved",
            tile_size,
            axis_size,
            overlap_pixels,
        )
    ranges = []
    for idx, ax_start in enumerate(starts):
        ax_end = min(ax_start + tile_size, axis_size)
        if idx == 0:
            ax_start = 0
        if idx == len(starts) - 1:
            ax_end = axis_size
        ranges.append((ax_start, ax_end))
    regions = []
    for idx, (ax_start, ax_end) in enumerate(ranges):
        actual = ax_end - ax_start
        ov_left = 0
        ov_right = 0
        if idx > 0:
            prev_end = ranges[idx - 1][1]
            ov_left = min(actual, max(0, min(prev_end, ax_end) - ax_start))
        if idx < len(ranges) - 1:
            next_start = ranges[idx + 1][0]
            ov_right = min(actual, max(0, ax_end - max(ax_start, next_start)))
        regions.append((ax_start, ax_end, ov_left, ov_right))
    return {
        "axis": axis,
        "axis_size": axis_size,
        "tile_size": tile_size,
        "overlap_pixels": overlap_pixels,
        "overlap_latent": overlap,
        "n_tiles": len(starts),
        "regions": regions,
    }


def _invoke_model(model_k, x, sigma, denoise_mask, call_kw: dict):
    kwargs = {key: value for key, value in call_kw.items() if value is not None}
    return type(model_k).__call__(model_k, x, sigma, denoise_mask, **kwargs)


def _tiled_model_call(
    model_k,
    x,
    sigma,
    *,
    denoise_mask=None,
    model_options=None,
    seed=None,
    n_tiles: int,
    overlap_pixels: int,
    **kwargs,
):
    import comfy.utils

    cfg = getattr(model_k, "inner_model", None)
    base = getattr(cfg, "inner_model", None) if cfg is not None else None
    full_shapes = getattr(base, "latent_shapes", None) if base is not None else None
    call_kw = dict(kwargs)
    if model_options is not None:
        call_kw["model_options"] = model_options
    if seed is not None:
        call_kw["seed"] = seed

    if not full_shapes or not isinstance(x, torch.Tensor):
        return _invoke_model(model_k, x, sigma, denoise_mask, call_kw)

    streams = list(comfy.utils.unpack_latents(x, full_shapes))
    video = streams[0]
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        return _invoke_model(model_k, x, sigma, denoise_mask, call_kw)

    plan = getattr(model_k, "_director_spatial_plan", None)
    if plan is None:
        plan = _plan_regions(tuple(video.shape), n_tiles, overlap_pixels)
        model_k._director_spatial_plan = plan
        if plan is None:
            log.info(
                "spatial tiling skipped (axis %dx%d, n_tiles=%s)",
                int(video.shape[-2]),
                int(video.shape[-1]),
                n_tiles,
            )
        else:
            log.info(
                "spatial tiles: axis=%s size=%s n=%s overlap=%spx/%s latent ranges=%s",
                plan["axis"],
                plan["axis_size"],
                plan["n_tiles"],
                plan["overlap_pixels"],
                plan["overlap_latent"],
                [(a, b) for a, b, *_ in plan["regions"]],
            )
    if plan is None:
        return _invoke_model(model_k, x, sigma, denoise_mask, call_kw)

    axis = plan["axis"]
    full_h = int(video.shape[-2])
    full_w = int(video.shape[-1])
    device = video.device
    dtype = torch.float32
    video_acc = torch.zeros_like(video, dtype=dtype)
    weight_shape = (1, 1, 1, full_h if axis == "H" else 1, full_w if axis == "W" else 1)
    weights = torch.zeros(weight_shape, dtype=dtype, device=device)
    audio_acc = None
    audio_n = 0
    saved_shapes = base.latent_shapes
    saved_latent_image = getattr(model_k, "latent_image", None)
    saved_noise = getattr(model_k, "noise", None)

    try:
        for ax_start, ax_end, ov_left, ov_right in plan["regions"]:
            tile_x, tile_shapes, (pad_h, pad_w) = _crop_packed(x, full_shapes, axis, ax_start, ax_end)
            tile_mask = None
            if denoise_mask is not None:
                tile_mask, _, _ = _crop_packed(denoise_mask, full_shapes, axis, ax_start, ax_end)
            if saved_latent_image is not None:
                model_k.latent_image, _, _ = _crop_packed(
                    saved_latent_image, full_shapes, axis, ax_start, ax_end
                )
            if saved_noise is not None:
                model_k.noise, _, _ = _crop_packed(saved_noise, full_shapes, axis, ax_start, ax_end)
            restorations = []
            try:
                restorations = _install_tile_payloads(
                    cfg,
                    axis=axis,
                    start=ax_start,
                    end=ax_end,
                    tile_shapes=tile_shapes,
                    full_h=full_h,
                    full_w=full_w,
                )
            except Exception as exc:
                _restore_payloads(restorations)
                base.latent_shapes = saved_shapes
                model_k.latent_image = saved_latent_image
                model_k.noise = saved_noise
                log.warning(
                    "spatial tile layout failed (%s); this step uses a full-frame forward",
                    exc,
                )
                return _invoke_model(model_k, x, sigma, denoise_mask, call_kw)
            base.latent_shapes = tile_shapes
            try:
                pred = _invoke_model(model_k, tile_x, sigma, tile_mask, call_kw)
            finally:
                _restore_payloads(restorations)
                base.latent_shapes = saved_shapes
                model_k.latent_image = saved_latent_image
                model_k.noise = saved_noise

            pred_streams = list(comfy.utils.unpack_latents(pred, tile_shapes))
            pred_video = _unpad_hw(pred_streams[0], pad_h, pad_w).float()
            actual = ax_end - ax_start
            window = _raised_cosine_1d(
                actual, ov_left, ov_right, dtype=dtype, device=device
            )
            if axis == "H":
                win = window.view(1, 1, 1, -1, 1)
                video_acc[:, :, :, ax_start:ax_end, :] += pred_video * win
                weights[:, :, :, ax_start:ax_end, :] += win
            else:
                win = window.view(1, 1, 1, 1, -1)
                video_acc[:, :, :, :, ax_start:ax_end] += pred_video * win
                weights[:, :, :, :, ax_start:ax_end] += win
            if len(pred_streams) > 1:
                if audio_acc is None:
                    audio_acc = pred_streams[1].float()
                else:
                    audio_acc = audio_acc + pred_streams[1].float()
                audio_n += 1
            del tile_x, pred, pred_streams, pred_video
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        base.latent_shapes = saved_shapes
        model_k.latent_image = saved_latent_image
        model_k.noise = saved_noise

    merged = [video_acc / weights.clamp(min=1e-8)]
    if audio_acc is not None and len(streams) > 1:
        merged.append((audio_acc / float(max(1, audio_n))).to(dtype=streams[1].dtype))
        merged[0] = merged[0].to(dtype=streams[0].dtype)
    else:
        merged[0] = merged[0].to(dtype=streams[0].dtype)
        if len(streams) > 1:
            merged.extend(streams[1:])
    packed, _ = comfy.utils.pack_latents(merged)
    return packed.to(dtype=x.dtype)


class _TiledModelProxy:
    """Instance ``__call__`` assignment is ignored by ``obj()``; proxy instead."""

    def __init__(self, inner, *, n_tiles: int, overlap_pixels: int):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_n_tiles", n_tiles)
        object.__setattr__(self, "_overlap_pixels", overlap_pixels)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_inner"), name, value)

    def __call__(self, x, sigma, denoise_mask=None, model_options=None, seed=None, **call_kw):
        inner = object.__getattribute__(self, "_inner")
        return _tiled_model_call(
            inner,
            x,
            sigma,
            denoise_mask=denoise_mask,
            model_options=model_options,
            seed=seed,
            n_tiles=object.__getattribute__(self, "_n_tiles"),
            overlap_pixels=object.__getattribute__(self, "_overlap_pixels"),
            **call_kw,
        )


def wrap_sampler_spatial_tiles(
    sampler,
    *,
    n_tiles: int,
    overlap_pixels: int,
) -> Callable[[], None]:
    """Patch ``sampler.sampler_function`` so each model eval is spatially tiled.

    Returns a restore callback. No-op when n_tiles <= 1.
    """
    n_tiles = clamp_n_tiles(n_tiles)
    overlap_pixels = clamp_tile_overlap(overlap_pixels)
    if n_tiles <= 1 or sampler is None:
        return lambda: None
    orig_fn = getattr(sampler, "sampler_function", None)
    if orig_fn is None:
        return lambda: None

    def wrapped(model_k, noise, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        proxy = _TiledModelProxy(model_k, n_tiles=n_tiles, overlap_pixels=overlap_pixels)
        return orig_fn(
            proxy,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            **kwargs,
        )

    sampler.sampler_function = wrapped

    def restore() -> None:
        sampler.sampler_function = orig_fn

    return restore
