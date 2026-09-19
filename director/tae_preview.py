"""Lightweight MiniMax H3 TAE / Latent2RGB previews during sampling.

Uses ``models/vae_approx/taeh3.safetensors`` when present (Kijai / madebyollin
flat TinyVAE, 24-ch). Falls back to Latent2RGB so the UI still updates.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.tae_preview")

_DEFAULT_TAE_NAME = "taeh3.safetensors"
_lock = threading.Lock()
_decoder: Any | None = None
_decoder_failed = False


def _place(model, device, dtype):
    model = model.eval().to(device=device, dtype=dtype)
    if torch.device(device).type == "cuda":
        model.to(memory_format=torch.channels_last)
    return model


def _build_tae_decoder(sd: dict):
    # Mirror KJNodes TinyVAEDecoder layout recovery (flat indexed modules).
    from comfy.taesd.taesd import Block, Clamp, conv

    by_index: dict[int, dict] = {}
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        if not head.isdigit():
            raise ValueError(f"not a flat TAE decoder state dict (unexpected key '{k}')")
        by_index.setdefault(int(head), {})[rest] = v

    modules = []
    for i in range(max(by_index) + 1):
        entry = by_index.get(i)
        if entry is None:
            modules.append(Clamp() if i == 0 else nn.ReLU() if i == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            w = entry["conv.0.weight"]
            if "pool.0.weight" in entry:
                modules.append(Block(w.shape[1], w.shape[0], use_midblock_gn=True))
            else:
                modules.append(Block(w.shape[1], w.shape[0]))
        elif "weight" in entry:
            w = entry["weight"]
            modules.append(conv(w.shape[1], w.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"unrecognized TAE decoder module at index {i}: {sorted(entry)}")
    return nn.Sequential(*modules)


class _TinyVAEDecoder:
    def __init__(self, sd, device=None, dtype=None):
        import comfy.model_management as mm

        prefix = ""
        first = next(iter(sd))
        if not first.split(".")[0].isdigit():
            prefix = first.split(".")[0] + "."
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        self.device = device if device is not None else mm.vae_device()
        self.dtype = dtype if dtype is not None else mm.vae_dtype(
            self.device, [torch.float16, torch.bfloat16]
        )
        self.model = _build_tae_decoder(sd)
        self.model.load_state_dict(sd)
        self.model = _place(self.model, self.device, self.dtype)
        self.latent_channels = self.model[1].weight.shape[1]

    @torch.inference_mode()
    def decode_frame(self, latent_bchw: torch.Tensor) -> torch.Tensor:
        """[1,C,H,W] -> [H',W',3] float in 0..1"""
        out = self.model(latent_bchw.to(device=self.device, dtype=self.dtype))
        return out[0].movedim(0, -1).float().clamp(0, 1).cpu()


def get_tae_decoder(name: str = _DEFAULT_TAE_NAME):
    global _decoder, _decoder_failed
    if _decoder_failed:
        return None
    if _decoder is not None:
        return _decoder
    with _lock:
        if _decoder is not None or _decoder_failed:
            return _decoder
        try:
            import folder_paths
            import comfy.utils

            path = folder_paths.get_full_path("vae_approx", name)
            if path is None:
                log.info("TAE preview: %s not found in models/vae_approx — using Latent2RGB.", name)
                _decoder_failed = True
                return None
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            if "decoder.1.weight" in sd:
                log.warning("TAE preview: temporal TAEHV weights not supported yet (%s).", name)
                _decoder_failed = True
                return None
            _decoder = _TinyVAEDecoder(sd)
            log.info("TAE preview: loaded %s (%d-ch).", name, _decoder.latent_channels)
            return _decoder
        except Exception as exc:
            log.warning("TAE preview: failed to load %s (%s) — using Latent2RGB.", name, exc)
            _decoder_failed = True
            return None


def _video_latent_from_x0(x0: Any) -> torch.Tensor | None:
    """Return video stream as [B,C,T,H,W] from NestedTensor / plain tensor."""
    try:
        # NestedTensor has unbind(); plain torch.Tensor also has unbind — do not use that.
        if not isinstance(x0, torch.Tensor) and hasattr(x0, "unbind"):
            parts = x0.unbind()
            if parts:
                x0 = parts[0]
        elif isinstance(x0, (tuple, list)) and x0:
            x0 = x0[0]
        if not isinstance(x0, torch.Tensor):
            return None
        if x0.ndim == 5:
            return x0
        if x0.ndim == 4:
            return x0.unsqueeze(2)
    except Exception as exc:
        log.debug("TAE preview: could not unpack x0: %s", exc)
    return None


# Match KJNodes ModelPreviewOverride defaults for animated step previews.
LIVE_PREVIEW_MAX_FRAMES = 16
LIVE_PREVIEW_FPS = 12


def _pick_temporal_indices(t_total: int, max_frames: int) -> list[int]:
    if t_total <= 0:
        return []
    cap = int(max_frames or 0)
    if cap <= 0 or cap >= t_total:
        return list(range(t_total))
    return np.linspace(0, t_total - 1, cap).round().astype(int).tolist()


def _rgb_to_pil(rgb_hwc: torch.Tensor) -> Image.Image:
    arr = (rgb_hwc.detach().float().clamp(0, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _latent2rgb_frames(video: torch.Tensor, indices: list[int]) -> list[Image.Image]:
    try:
        from comfy.latent_formats import MiniMaxH3Video
        import latent_preview

        fmt = MiniMaxH3Video()
        previewer = latent_preview.Latent2RGBPreviewer(
            fmt.latent_rgb_factors,
            fmt.latent_rgb_factors_bias,
        )
        out: list[Image.Image] = []
        for t in indices:
            frame = previewer.decode_latent_to_preview(video[:1, :, int(t)])
            if isinstance(frame, Image.Image):
                out.append(frame.convert("RGB"))
        return out
    except Exception as exc:
        log.debug("Latent2RGB preview failed: %s", exc)
    return []


def _tae_frames(video: torch.Tensor, indices: list[int]) -> list[Image.Image]:
    dec = get_tae_decoder()
    if dec is None or int(video.shape[1]) != int(dec.latent_channels):
        return []
    try:
        return [_rgb_to_pil(dec.decode_frame(video[:1, :, int(t)])) for t in indices]
    except Exception as exc:
        log.warning("TAE decode failed, falling back to Latent2RGB: %s", exc)
        return []


def _fit_preview_pil(pil: Image.Image, max_side: int) -> Image.Image:
    # Latent2RGB frames are often tiny (latent spatial size); upscale so UI
    # preview slots are not a speck in a large card.
    min_side = 256
    longest = max(int(pil.width), int(pil.height))
    if longest > 0 and longest < min_side:
        scale = min_side / float(longest)
        nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        pil = pil.resize(
            (max(1, int(round(pil.width * scale))), max(1, int(round(pil.height * scale)))),
            nearest,
        )
    if max_side and max_side > 0 and (pil.width > max_side or pil.height > max_side):
        pil = ImageOps.contain(pil, (max_side, max_side), Image.LANCZOS)
    return pil if pil.mode == "RGB" else pil.convert("RGB")


def x0_to_preview_frames(
    x0: Any,
    *,
    max_frames: int = LIVE_PREVIEW_MAX_FRAMES,
    max_side: int = 512,
) -> list[Image.Image]:
    """Decode evenly spaced temporal frames from a video x0 (KJNodes-style)."""
    video = _video_latent_from_x0(x0)
    if video is None or video.numel() == 0:
        return []
    # Detach so TAE / Latent2RGB cannot alias the sampler's live x0.
    video = video.detach()
    indices = _pick_temporal_indices(int(video.shape[2]), max_frames)
    if not indices:
        return []
    frames = _tae_frames(video, indices) or _latent2rgb_frames(video, indices)
    return [_fit_preview_pil(frame, max_side) for frame in frames]


def x0_to_preview_pil(x0: Any, *, max_side: int = 512) -> Image.Image | None:
    frames = x0_to_preview_frames(x0, max_frames=LIVE_PREVIEW_MAX_FRAMES, max_side=max_side)
    if not frames:
        return None
    return frames[len(frames) // 2]


def pil_to_jpeg_b64(pil: Image.Image, *, quality: int = 80) -> str:
    import base64
    import io

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_animated_webp(frames: list[Image.Image], *, fps: int = LIVE_PREVIEW_FPS, quality: int = 80) -> str:
    import base64
    import io

    if not frames:
        return ""
    duration_ms = max(1, int(round(1000 / max(1, int(fps)))))
    buf = io.BytesIO()
    try:
        frames[0].save(
            buf,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            quality=int(quality),
            method=4,
        )
    except Exception as exc:
        log.warning("Animated WebP encode failed: %s", exc)
        return ""
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_preview_payload(
    frames: list[Image.Image],
    *,
    fps: int = LIVE_PREVIEW_FPS,
    quality: int = 80,
) -> tuple[str, str, int, int]:
    """Return (b64, mime, width, height). Multi-frame → looping WebP like KJNodes."""
    if not frames:
        return "", "image/jpeg", 0, 0
    first = frames[0]
    if len(frames) == 1:
        return pil_to_jpeg_b64(first, quality=quality), "image/jpeg", first.width, first.height
    b64 = encode_animated_webp(frames, fps=fps, quality=quality)
    if not b64:
        mid = frames[len(frames) // 2]
        return pil_to_jpeg_b64(mid, quality=quality), "image/jpeg", mid.width, mid.height
    return b64, "image/webp", first.width, first.height
