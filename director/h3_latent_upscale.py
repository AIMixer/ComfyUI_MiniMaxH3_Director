"""In-repo MiniMax H3 24-ch latent spatial upscaler (3D).

Inference-only port of the Apache-2.0 3D architecture from
LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler so Director Refine does not
depend on that custom node. Weights stay user-provided under
``ComfyUI/models/latent_upscale_models/``.

Only H×W are scaled; the time axis is preserved. Audio is not touched.
"""

from __future__ import annotations

import glob
import logging
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.h3_latent_upscale")

LATENT_UPSCALE_FOLDER = "latent_upscale_models"
MISSING_MODEL_LABEL = "(将 3D 权重放入 models/latent_upscale_models)"

# Per-channel mean/std from LBH-123-AI H3 latent-upscaler training.
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053832, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450743, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]

_MODEL_CACHE: dict[str, nn.Module] = {}


def ensure_latent_upscale_folder() -> str | None:
    try:
        import folder_paths

        if LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                LATENT_UPSCALE_FOLDER,
                os.path.join(folder_paths.models_dir, LATENT_UPSCALE_FOLDER),
            )
        paths = folder_paths.get_folder_paths(LATENT_UPSCALE_FOLDER)
        return paths[0] if paths else None
    except Exception:
        return None


def list_h3_latent_upscale_models() -> list[str]:
    root = ensure_latent_upscale_folder()
    names: list[str] = []
    if root:
        try:
            os.makedirs(root, exist_ok=True)
        except OSError:
            pass
        for ext in ("*.safetensors", "*.pth"):
            names.extend(os.path.basename(p) for p in glob.glob(os.path.join(root, ext)))
        try:
            import folder_paths

            names.extend(folder_paths.get_filename_list(LATENT_UPSCALE_FOLDER) or [])
        except Exception:
            pass
    out = sorted({n for n in names if n and not n.startswith("(")})
    preferred = [n for n in out if "3d" in n.lower()]
    return preferred or out or [MISSING_MODEL_LABEL]


def _norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels
            else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return x + h


class LatentResizer3D(nn.Module):
    def __init__(
        self,
        in_channels=24,
        in_blocks=12,
        out_blocks=12,
        channels=512,
        dropout=0.1,
        temporal_every=2,
        temporal_kernel=5,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale: float, target_size: tuple[int, int, int]):
        if tuple(target_size) == tuple(x.shape[-3:]):
            return x
        scale_emb = torch.tensor(
            [float(scale) - 1.0], dtype=x.dtype, device=x.device
        ).unsqueeze(0)
        emb = self.embed(scale_emb)
        x = self.conv_in(x)
        for block in self.in_blocks:
            if isinstance(block, ResBlockEmb3D):
                x = block(x, emb.expand(x.shape[0], -1))
            else:
                x = block(x)
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            if isinstance(block, ResBlockEmb3D):
                x = block(x, emb.expand(x.shape[0], -1))
            else:
                x = block(x)
        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


def _load_raw_sd(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
    out = {}
    for key, value in sd.items():
        if getattr(value, "dtype", None) == torch.float8_e4m3fn:
            value = value.to(torch.float16)
        out[key] = value
    return out


def _extract_upscaler_sd(sd: dict) -> dict:
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler.") :]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def _detect_arch(sd: dict) -> dict:
    cfg = {
        "in_channels": 24,
        "in_blocks": 12,
        "out_blocks": 12,
        "channels": 512,
        "dropout": 0.1,
        "temporal_every": 2,
        "temporal_kernel": 5,
    }
    if "conv_in.weight" in sd:
        cfg["in_channels"] = int(sd["conv_in.weight"].shape[1])
        cfg["channels"] = int(sd["conv_in.weight"].shape[0])
    in_ids, out_ids = set(), set()
    for key in sd:
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", key)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", key)
        if m:
            out_ids.add(int(m.group(1)))
    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    has_temporal = any("dwconv.weight" in k for k in sd)
    if has_temporal:
        for key, value in sd.items():
            if key.endswith("dwconv.weight"):
                cfg["temporal_kernel"] = int(value.shape[2])
                break
    else:
        cfg["temporal_every"] = 0
    return cfg


def _resolve_model_path(name: str) -> str:
    ensure_latent_upscale_folder()
    import folder_paths

    path = folder_paths.get_full_path(LATENT_UPSCALE_FOLDER, name)
    if path:
        return path
    root = ensure_latent_upscale_folder()
    if root:
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"H3 latent upscaler weights not found: {name}. "
        f"Put the 3D safetensors in ComfyUI/models/{LATENT_UPSCALE_FOLDER}/"
    )


def load_h3_latent_upscaler(name: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    cache_key = f"{name}::{dtype}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached.to(device=device, dtype=dtype)
    path = _resolve_model_path(name)
    raw = _load_raw_sd(path)
    sd = _extract_upscaler_sd(raw)
    if "resizer.conv_in.weight" in sd and "conv_in.weight" not in sd:
        raise ValueError(
            f"{name} looks like the 2D checkpoint. Director uses the 3D weights "
            "(minimax_h3_latent_upscaler_3d_*.safetensors)."
        )
    cfg = _detect_arch(sd)
    model = LatentResizer3D(**cfg)
    model.load_state_dict(sd, strict=True)
    model = model.to(device="cpu", dtype=dtype).eval()
    _MODEL_CACHE[cache_key] = model
    log.info(
        "H3 latent upscaler loaded: %s (params=%s, C=%d)",
        name,
        f"{sum(p.numel() for p in model.parameters()):,}",
        cfg["in_channels"],
    )
    return model.to(device=device, dtype=dtype)


def _as_bcthw(samples: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Director video latents are [B,C,T,H,W] or [C,T,H,W] (same as video_from_latent)."""
    if samples.ndim == 5:
        return samples, "none"
    if samples.ndim == 4:
        return samples.unsqueeze(0), "batch"
    raise ValueError(f"H3 latent upscale expected 4D/5D video latent, got {tuple(samples.shape)}")


def upscale_h3_video_latent(
    video_latent: dict,
    *,
    target_width: int,
    target_height: int,
    source_width: int,
    source_height: int,
    model_name: str = "",
    model=None,
) -> dict:
    """Spatially upscale MiniMax H3 video latent to a pixel canvas (×16 VAE)."""
    if model is None and (not model_name or str(model_name).startswith("(")):
        raise ValueError(
            "未选择 H3 latent 放大模型。请在 Refine 的 latent_upscale_model 下拉框里选 3D 权重，"
            f"并把 safetensors 放到 ComfyUI/models/{LATENT_UPSCALE_FOLDER}/"
        )
    samples = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
    if not torch.is_tensor(samples):
        raise ValueError("H3 latent upscale needs a video latent tensor.")
    work, squeezed = _as_bcthw(samples)
    lat_h, lat_w = int(work.shape[-2]), int(work.shape[-1])
    src_h = max(1, int(source_height or 0))
    src_w = max(1, int(source_width or 0))
    ratio_h = src_h / float(lat_h)
    ratio_w = src_w / float(lat_w)
    dst_h = max(1, int(round(int(target_height) / ratio_h)))
    dst_w = max(1, int(round(int(target_width) / ratio_w)))
    t_size = int(work.shape[2])
    if dst_h == lat_h and dst_w == lat_w:
        return video_latent if isinstance(video_latent, dict) else {"samples": samples}

    scale = max(
        float(int(target_width)) / float(src_w),
        float(int(target_height)) / float(src_h),
    )
    if scale < 0.999:
        raise ValueError(
            f"H3 latent upscale is enlarge-only (got {src_w}×{src_h} → "
            f"{int(target_width)}×{int(target_height)})."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Official 3D node defaults to fp32; bf16 GroupNorm on this net can collapse
    # to NaN/mud and decode as a brown static frame.
    dtype = torch.float32
    if model is not None:
        net = model.model if hasattr(model, "model") and not hasattr(model, "conv_in") else model
        model = net.to(device=device, dtype=dtype).eval()
    else:
        model = load_h3_latent_upscaler(model_name, device, dtype)
    orig_dtype = work.dtype
    mean, std = _norm_tensors(device, dtype)
    x = work.to(device=device, dtype=dtype)
    x = (x - mean) / std
    try:
        with torch.no_grad():
            out = model(x, scale=scale, target_size=(t_size, dst_h, dst_w))
        out = out * std + mean
        out = out.to(device="cpu", dtype=orig_dtype).contiguous()
    finally:
        # Keep the 3D net off the GPU so MiniMaxH3 does not re-stage ~20GB.
        model.to("cpu")
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if squeezed == "batch" and out.shape[0] == 1:
        out = out.squeeze(0)
    packed = dict(video_latent) if isinstance(video_latent, dict) else {}
    packed["samples"] = out
    packed.pop("noise_mask", None)
    log.info(
        "H3 latent upscale: %s %d×%d latent %dx%d → %dx%d (pixel %d×%d → %d×%d, scale=%.2f)",
        model_name,
        src_w,
        src_h,
        lat_w,
        lat_h,
        dst_w,
        dst_h,
        src_w,
        src_h,
        int(target_width),
        int(target_height),
        scale,
    )
    return packed
