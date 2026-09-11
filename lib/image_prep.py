"""Image and video preprocessing for Bernini semantic context streams."""

from __future__ import annotations

import torch
from comfy.utils import common_upscale


# MiniMax H3: VAE spatial ÷16, then 2×2 patchify → canvas must be multiples of 32.
MINIMAX_CANVAS_STRIDE = 32


def snap_dimension(value: int, stride: int = MINIMAX_CANVAS_STRIDE) -> int:
    """Round *value* to the nearest multiple of *stride*, keeping at least *stride*."""
    return max(stride, round(value / stride) * stride)


def ensure_minimax_canvas(width: int, height: int) -> tuple[int, int]:
    """Snap width/height to MiniMax H3 canvas multiples (32)."""
    return snap_dimension(int(width)), snap_dimension(int(height))


def assert_minimax_canvas(width: int, height: int) -> None:
    """Raise a clear error when W/H cannot be patchified by MiniMax H3."""
    w, h = int(width), int(height)
    if w > 0 and h > 0 and w % MINIMAX_CANVAS_STRIDE == 0 and h % MINIMAX_CANVAS_STRIDE == 0:
        return
    nw, nh = ensure_minimax_canvas(max(w, 1), max(h, 1))
    raise ValueError(
        f"输出分辨率 {w}×{h} 不是 32 的倍数（MiniMax H3 要求：VAE÷16 后再 2×2 patch）。"
        f"请改为 {nw}×{nh}，或在输出面板重新选择最长边/固定尺寸后重跑。"
        "若同时安装了独立的 ComfyUI-H3-Motion-Context，请卸载或禁用后重启 ComfyUI（与 Director 内置连贯冲突）。"
    )


def fit_edge_limit(
    image: torch.Tensor,
    max_px: int,
    *,
    edge: str = "short",
    stride: int = 2,
) -> torch.Tensor:
    """Downscale only so short or long edge is at most ``max_px``. Never upscale."""
    rgb = image[..., :3]
    squeezed = False
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
        squeezed = True
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    current = min(height, width) if str(edge).strip().lower() == "short" else max(height, width)
    limit = max(int(stride), int(max_px))
    if current <= limit:
        return image if not squeezed else rgb
    scale = limit / float(current)
    new_h = max(stride, int(round(height * scale / stride) * stride))
    new_w = max(stride, int(round(width * scale / stride) * stride))
    if new_h == height and new_w == width:
        return image if not squeezed else rgb
    out = common_upscale(
        rgb.movedim(-1, 1), new_w, new_h, "area", "disabled"
    ).movedim(1, -1)
    return out[0] if squeezed and image.ndim == 3 else out


def limit_ref_image_dict(
    images: dict | None,
    max_px: int,
    edge: str = "short",
) -> tuple[dict | None, int]:
    """Apply :func:`fit_edge_limit` to each H3 ``ref_image_*`` tensor."""
    if not images:
        return images, 0
    changed = 0
    out: dict = {}
    for key, tensor in images.items():
        if tensor is None:
            continue
        resized = fit_edge_limit(tensor, max_px, edge=edge)
        if resized is not tensor and tuple(resized.shape[-3:-1]) != tuple(tensor.shape[-3:-1]):
            changed += 1
        out[key] = resized
    return out or None, changed


def fit_long_edge(image: torch.Tensor, max_edge: int, stride: int = 32) -> torch.Tensor:
    """Long-edge fit with MiniMax H3 canvas snap (default stride 32).

    H3 VAE is 16× spatially, then patchified 2×2 → effective 32×. Odd latent
    widths (e.g. 496px → 31) crash ``patchify_video`` during sampling.
    """
    rgb = image[:, :, :, :3]
    height, width = rgb.shape[1], rgb.shape[2]
    scale = min(max_edge / max(height, width), 1.0)
    new_h = max(stride, round(height * scale / stride) * stride)
    new_w = max(stride, round(width * scale / stride) * stride)
    return common_upscale(
        rgb.movedim(-1, 1), new_w, new_h, "area", "disabled"
    ).movedim(1, -1)


def fit_canvas(
    frames: torch.Tensor,
    width: int,
    height: int,
    upscale_mode: str = "area",
) -> torch.Tensor:
    """Resize video frames to a fixed canvas (width x height)."""
    rgb = frames[..., :3]
    return common_upscale(
        rgb.movedim(-1, 1), width, height, upscale_mode, "center"
    ).movedim(1, -1)


def fit_video_long_edge(frames: torch.Tensor, max_edge: int, stride: int = 32) -> torch.Tensor:
    """Scale each frame so its long side is at most *max_edge*, preserving aspect ratio."""
    if frames.shape[0] == 0:
        return frames
    chunks = [fit_long_edge(frames[i : i + 1], max_edge, stride=stride) for i in range(frames.shape[0])]
    return torch.cat(chunks, dim=0)


def pad_frames_to_canvas(
    frames: torch.Tensor,
    width: int,
    height: int,
    *,
    fill: float = 0.5,
) -> torch.Tensor:
    """Center-pad a frame sequence onto a fixed canvas (letterbox)."""
    if frames.ndim != 4:
        raise ValueError("frames must be [F, H, W, C]")
    f, h, w, c = frames.shape
    if h == height and w == width:
        return frames
    out = torch.full((f, height, width, c), fill, dtype=frames.dtype, device=frames.device)
    y0 = max(0, (height - h) // 2)
    x0 = max(0, (width - w) // 2)
    copy_h = min(h, height - y0)
    copy_w = min(w, width - x0)
    out[:, y0 : y0 + copy_h, x0 : x0 + copy_w, :] = frames[:, :copy_h, :copy_w, :]
    return out


def cat_frames_variable_size(clips: list[torch.Tensor], *, fill: float = 0.5) -> torch.Tensor:
    """Concatenate frame clips along time, padding spatial dims when aspect ratios differ."""
    if not clips:
        raise ValueError("No clips to concatenate.")
    if len(clips) == 1:
        return clips[0]
    shapes = {(int(c.shape[1]), int(c.shape[2])) for c in clips}
    if len(shapes) == 1:
        return torch.cat(clips, dim=0)
    max_h = max(c.shape[1] for c in clips)
    max_w = max(c.shape[2] for c in clips)
    padded = [pad_frames_to_canvas(c, max_w, max_h, fill=fill) for c in clips]
    return torch.cat(padded, dim=0)


def resolve_output_dimensions(
    source_w: int,
    source_h: int,
    *,
    mode: str = "long_edge",
    long_edge: int = 848,
    fixed_width: int = 832,
    fixed_height: int = 480,
    stride: int = 32,
) -> tuple[int, int, int, str]:
    """Return (width, height, ref_max_size, mode) for MiniMax H3 Director output.

    ``stride`` defaults to 32 (H3 patch grid). Using 16 can yield e.g. 496×864
    which encodes to an odd latent width and crashes ``patchify_video``.
    """
    mode = (mode or "long_edge").lower()
    if mode == "fixed":
        w = snap_dimension(int(fixed_width), stride)
        h = snap_dimension(int(fixed_height), stride)
        return w, h, max(w, h), "fixed"

    long_edge = max(stride, int(long_edge))
    if source_w <= 0 or source_h <= 0:
        w = snap_dimension(int(fixed_width), stride)
        h = snap_dimension(int(fixed_height), stride)
        return w, h, long_edge, "long_edge"

    if max(source_w, source_h) <= long_edge:
        return snap_dimension(source_w, stride), snap_dimension(source_h, stride), long_edge, "long_edge"

    scale = long_edge / max(source_w, source_h)
    w = snap_dimension(int(round(source_w * scale)), stride)
    h = snap_dimension(int(round(source_h * scale)), stride)
    return w, h, long_edge, "long_edge"


def normalize_to_vae_range(frames: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] images to [-1, 1] for Wan-style VAE encoding."""
    return frames * 2.0 - 1.0
