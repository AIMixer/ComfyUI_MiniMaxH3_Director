"""X2-aware H3 video decode for Director's internal decode points.

Director ships a single ``VAEDecode`` call per output path, which assumes the
stock MiniMax H3 video VAE decoder emits 3 packed RGB channels. The 2X family
(``MiniMax-H3-X2-Detail-v1.safetensors`` and relatives) replaces the decoder
projection with a PixelShuffle-packed head, so the same call would hand ComfyUI
a 12-channel "image" and the run would break downstream.

ComfyUI's own ``VAEDecode`` cannot absorb this: ``decode_output_shape`` derives
the output buffer from ``decoder.out_channels``, and the projection weight shape
is the only reliable signal of which head a loaded VAE actually has. This module
is that signal.

Detection is deliberately shape-based rather than filename-based: the X2 family
reuses stock H3 tensor names, and the checkpoint name has changed across
community releases. ``proj_out`` rows are ``out_ch * patch_size_t * patch_size^2``
= ``out_ch * 1024`` for H3, so 3072 rows is 3-channel and 12288 rows is a 2x2
PixelShuffle of RGB (12 channels).

A stock VAE takes the untouched ComfyUI path, so wiring this in cannot change
behaviour for users who never load an X2 checkpoint. X2 output is byte-for-byte
equivalent to decoding with the reference ``MiniMaxH3VAEDecodeFast`` node at its
validated settings (tiling on, 256px tile, 64px overlap, causal temporal decode
left off).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

import comfy.model_management as model_management

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.h3_x2_decode")

# H3's ViT3DDecoder projection is Linear(dim, out_ch * patch_size_t * patch_size^2).
# vae_ratio_t = prod(time_down) = 4, vae_ratio = prod(space_down) = 16.
_H3_PATCH_VOLUME = 4 * 16 * 16
_NATIVE_CHANNELS = 3

# ComfyUI's H3 decoder hardcodes 256px spatial tiles with 64px overlap; see
# comfy/ldm/minimax/vae.py. VAEDecodeTiled cannot change these because
# decode_tiled() swallows kwargs, and the OOM fallback in sd.py drops into a
# generic 3D tiler that does not know H3's blend rules (seams read as a block
# grid). Keep the reference values so the X2 path matches the validated recipe.
_STOCK_TILE = 256
_STOCK_OVERLAP = 64
_TILE_MULTIPLE = 16  # native spatial VAE ratio

_MISSING_INNER = (
    "Director X2 decode expects the MiniMax H3 *video* VAE. "
    "Connect the video VAE, not the audio VAE."
)


def _video_vae_model(vae):
    """Return the H3 video VAE module, or None if this is not one."""
    inner = getattr(vae, "first_stage_model", None)
    if inner is None or getattr(inner, "decoder", None) is None:
        return None
    if not hasattr(inner, "tiling"):
        return None
    return inner


def detect_packed_output(vae) -> tuple[int, int]:
    """Return ``(packed_channels, pixel_shuffle_upscale)`` for a video VAE.

    ``(0, 1)`` means "not an H3-family decoder" and callers should use the
    stock ComfyUI decode path untouched.
    """
    inner = _video_vae_model(vae)
    if inner is None:
        return 0, 1
    proj_out = getattr(inner.decoder, "proj_out", None)
    weight = getattr(proj_out, "weight", None)
    if weight is None or weight.ndim != 2:
        return 0, 1
    rows = int(weight.shape[0])
    if rows <= 0 or rows % _H3_PATCH_VOLUME:
        log.warning(
            "H3 X2 decode: decoder projection has %d rows, not divisible by the "
            "H3 patch volume %d. Falling back to the stock decode path.",
            rows,
            _H3_PATCH_VOLUME,
        )
        return 0, 1

    packed = rows // _H3_PATCH_VOLUME
    if packed == _NATIVE_CHANNELS:
        return 0, 1
    if packed % _NATIVE_CHANNELS:
        log.warning(
            "H3 X2 decode: decoder projection implies %d packed channels, not "
            "divisible by %d RGB channels. Falling back to the stock decode path.",
            packed,
            _NATIVE_CHANNELS,
        )
        return 0, 1

    ratio_squared = packed // _NATIVE_CHANNELS
    ratio = math.isqrt(ratio_squared)
    if ratio * ratio != ratio_squared:
        log.warning(
            "H3 X2 decode: %d packed channels do not form a square PixelShuffle "
            "ratio. Falling back to the stock decode path.",
            packed,
        )
        return 0, 1
    return packed, ratio


def is_x2_video_vae(vae) -> bool:
    """True when this VAE needs the packed-output path."""
    return detect_packed_output(vae)[0] > 0


def _snap_multiple(value: int, multiple: int, minimum: int) -> int:
    value = max(int(value), int(minimum))
    return max(minimum, (value // multiple) * multiple)


def _effective_overlap(tile_size: int, tile_overlap: int) -> int:
    """Keep the stock 64/256 overlap ratio so larger tiles do not seam."""
    scaled = max(_STOCK_OVERLAP, (int(tile_size) * _STOCK_OVERLAP) // _STOCK_TILE)
    overlap = max(int(tile_overlap), scaled)
    overlap = _snap_multiple(overlap, _TILE_MULTIPLE, _TILE_MULTIPLE)
    return min(overlap, max(_TILE_MULTIPLE, tile_size - _TILE_MULTIPLE))


def _expand_rgb_stat(stat: torch.Tensor, packed_channels: int) -> torch.Tensor:
    """Expand an RGB normalization buffer to packed PixelShuffle channels.

    The converted projection lays out R phases, G phases, B phases, so each
    native statistic repeats consecutively rather than the whole RGB triplet
    repeating as a block. ``pixel_mean``/``pixel_std`` are registered with
    ``persistent=False``, so dynamic VRAM loading can restore the checkpoint's
    3-channel copies at any time — expand after the load, never before.
    """
    if stat is None or packed_channels == _NATIVE_CHANNELS:
        return stat
    if any(size == packed_channels for size in stat.shape):
        return stat
    if packed_channels % _NATIVE_CHANNELS:
        raise ValueError(f"Cannot expand RGB statistics to {packed_channels} channels.")

    channel_dims = [i for i, size in enumerate(stat.shape) if size == _NATIVE_CHANNELS]
    if not channel_dims:
        raise ValueError(
            "Expected an RGB normalization buffer with a size-3 dimension, got "
            f"{tuple(stat.shape)}."
        )
    channel_dim = 1 if stat.ndim > 1 and stat.shape[1] == _NATIVE_CHANNELS else channel_dims[0]
    return stat.repeat_interleave(packed_channels // _NATIVE_CHANNELS, dim=channel_dim)


def _as_video_latent(samples) -> torch.Tensor:
    """Accept a LATENT dict, a bare tensor, or ComfyUI's nested/latent wrappers."""
    if isinstance(samples, dict):
        if "samples" not in samples:
            raise KeyError('LATENT dict missing "samples"')
        samples = samples["samples"]
    while hasattr(samples, "is_nested") and samples.is_nested:
        samples = samples.unbind()[0]
    if not torch.is_tensor(samples):
        raise TypeError(f"Expected a latent tensor, got {type(samples).__name__}.")
    if samples.ndim == 4:
        samples = samples.unsqueeze(0)
    if samples.ndim != 5:
        raise ValueError(f"Expected H3 video latent BxCxTxHxW, got shape {tuple(samples.shape)}")
    return samples


def _pixel_shuffle_video(pixels: torch.Tensor, upscale: int) -> torch.Tensor:
    """Fold time into batch, PixelShuffle each frame, then restore B,C,T,H,W.

    ``F.pixel_shuffle`` only understands N,C,H,W, but H3's decoder output is
    B,C,T,H,W. The trailing channel dimension is still expected by the rest of
    Director, so the result stays B,C,T,H,W and callers unbind as usual.
    """
    batch, channels, frames, height, width = pixels.shape
    frame_batch = pixels.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    frame_batch = F.pixel_shuffle(frame_batch, upscale_factor=upscale)
    out_channels = channels // (upscale * upscale)
    return (
        frame_batch.reshape(batch, frames, out_channels, height * upscale, width * upscale)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def _decode_x2(
    vae,
    video: torch.Tensor,
    output_device: torch.device,
    upscale: int,
    *,
    tile_size: int,
    tile_overlap: int,
) -> torch.Tensor:
    """Decode through H3's own tiled path, then unpack the X2 head.

    Mirrors ``comfy.sd.VAE.decode``'s load + chunked-IO sequence, but calls the
    H3 decoder directly so the generic 3D-tiler OOM fallback cannot silently
    replace H3's blend rules.
    """
    inner = vae.first_stage_model
    vae.throw_exception_if_invalid()
    with model_management.cuda_device_context(vae.device):
        memory_used = vae.memory_used_decode(video.shape, vae.vae_dtype)
        model_management.load_models_gpu(
            [vae.patcher],
            memory_required=memory_used,
            force_full_load=vae.disable_offload,
        )
        packed_channels = int(inner.decoder.out_channels)
        # Must happen after load_models_gpu: dynamic loading can restore the
        # checkpoint's 3-channel buffers, and this is the final mutation before
        # the native decode writes into the preallocated buffer.
        inner.pixel_mean = _expand_rgb_stat(inner.pixel_mean, packed_channels)
        inner.pixel_std = _expand_rgb_stat(inner.pixel_std, packed_channels)
        pixels = torch.empty(
            inner.decode_output_shape(video.shape),
            device=output_device,
            dtype=vae.vae_output_dtype(),
        )
        inner.decode(video.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=pixels)
        vae.process_output(pixels)
    return _pixel_shuffle_video(pixels, upscale)


def _to_images(video: torch.Tensor) -> torch.Tensor:
    """B,C,T,H,W float -> B*T,H,W,C, matching VAEDecode's contract."""
    height, width = video.shape[-2], video.shape[-1]
    return video.movedim(1, -1).reshape(-1, height, width, video.shape[1])


def decode_video_latent(
    vae,
    samples,
    *,
    tile_size: int = _STOCK_TILE,
    tile_overlap: int = _STOCK_OVERLAP,
    output_device: str = "cpu",
    context: str = "director",
):
    """Decode a Director video latent, transparently handling the X2 head.

    Accepts whatever the surrounding Director path produces — a LATENT dict, a
    bare tensor, or a nested/AV latent — and returns IMAGE-shaped frames
    (B*T,H,W,C) exactly as the stock ``VAEDecode`` call it replaces would.
    """
    packed_channels, upscale = detect_packed_output(vae)
    if packed_channels == 0:
        from nodes import VAEDecode

        return VAEDecode().decode(vae, samples)[0]

    video = _as_video_latent(samples)
    inner = vae.first_stage_model
    if inner is None:
        raise ValueError(_MISSING_INNER)

    if output_device == "gpu":
        out_dev = model_management.get_torch_device()
    else:
        out_dev = model_management.intermediate_device()

    tile_size = _snap_multiple(tile_size, _TILE_MULTIPLE, _STOCK_TILE)
    overlap = _effective_overlap(tile_size, tile_overlap)

    # The X2 checkpoint's projection emits packed RGB subpixels at H3's native
    # spatial ratio; PixelShuffle turns them into the final 2x image afterwards.
    # Snapshot every field we touch so the rest of the graph is unaffected.
    saved = {
        name: getattr(inner, name)
        for name in ("tiling", "tile_size", "tile_overlap_min", "decoder")
    }
    saved_out_channels = inner.decoder.out_channels
    saved_pixel_mean = inner.pixel_mean
    saved_pixel_std = inner.pixel_std
    try:
        inner.decoder.out_channels = packed_channels
        inner.tiling = True
        inner.tile_size = tile_size
        inner.tile_overlap_min = overlap
        log.info(
            "Director X2 decode (%s): %d packed channels, PixelShuffle %dx, "
            "spatial tile %dpx overlap %dpx.",
            context,
            packed_channels,
            upscale,
            tile_size,
            overlap,
        )
        pixels = _decode_x2(
            vae,
            video,
            out_dev,
            upscale,
            tile_size=tile_size,
            tile_overlap=overlap,
        )
    finally:
        inner.decoder.out_channels = saved_out_channels
        inner.pixel_mean = saved_pixel_mean
        inner.pixel_std = saved_pixel_std
        for name, value in saved.items():
            setattr(inner, name, value)

    return _to_images(pixels.to(out_dev))
