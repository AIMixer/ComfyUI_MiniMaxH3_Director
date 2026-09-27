"""Low-res canvas / latent grid for SelfLift (H3 VAE×16 then 2×2 patch)."""

from __future__ import annotations

from ...lib.image_prep import MINIMAX_CANVAS_STRIDE, ensure_minimax_canvas

VAE_SPATIAL = 16


def clamp_lowres_scale(raw) -> float:
    try:
        scale = float(raw)
    except (TypeError, ValueError):
        scale = 0.5
    if scale != scale:  # NaN
        scale = 0.5
    return max(0.25, min(1.0, scale))


def lowres_canvas(width: int, height: int, scale: float) -> tuple[int, int]:
    """Target-canvas × scale, snapped to MiniMax ×32. Never larger than source."""
    src_w = max(int(width), MINIMAX_CANVAS_STRIDE)
    src_h = max(int(height), MINIMAX_CANVAS_STRIDE)
    scale = clamp_lowres_scale(scale)
    if scale >= 0.999:
        return src_w, src_h
    raw_w = max(MINIMAX_CANVAS_STRIDE, int(round(src_w * scale)))
    raw_h = max(MINIMAX_CANVAS_STRIDE, int(round(src_h * scale)))
    low_w, low_h = ensure_minimax_canvas(raw_w, raw_h)
    low_w = min(low_w, src_w)
    low_h = min(low_h, src_h)
    if low_w >= src_w and low_h >= src_h:
        return src_w, src_h
    return max(MINIMAX_CANVAS_STRIDE, low_w), max(MINIMAX_CANVAS_STRIDE, low_h)


def pixel_to_latent_hw(width: int, height: int) -> tuple[int, int]:
    """Pixel canvas W×H → video-latent ``(H, W)`` (VAE spatial ÷16).

    MiniMax latents are ``[B, C, T, H, W]``. Do not return ``(W, H)``.
    """
    return max(1, int(height) // VAE_SPATIAL), max(1, int(width) // VAE_SPATIAL)


def is_same_canvas(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return int(a[0]) == int(b[0]) and int(a[1]) == int(b[1])
