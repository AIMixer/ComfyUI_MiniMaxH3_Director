"""Graph pack for MiniMax H3 Director.selflift (first-pass progressive sample)."""

from __future__ import annotations

from typing import Any

from ..h3_latent_upscale import list_h3_latent_upscale_models
from ..refine_pack import (
    DEFAULT_SPATIAL_TILES,
    DEFAULT_TILE_OVERLAP,
    MAX_SPATIAL_TILES,
    resolve_latent_upscale_ref,
)

MMX_DIR_SELFLIFT = "MMX_DIR_SELFLIFT"

SPLIT_MODES = ("highres_steps", "transition_step")
UPSAMPLE_MODES = ("bilinear", "nearest")
SAMPLER_MODES = ("euler", "follow_director")
EULER_NAMES = frozenset({"euler"})

DEFAULT_HIGHRES_STEPS = 2
DEFAULT_TRANSITION_STEP = 6
DEFAULT_LOWRES_SCALE = 0.5
DEFAULT_RHO = 0.0
DEFAULT_W_MIN = 0.5
DEFAULT_W_MAX = 1.0


def _clamp_int(raw, default: int, lo: int, hi: int) -> int:
    try:
        n = int(raw if raw is not None else default)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _clamp_float(raw, default: float, lo: float, hi: float) -> float:
    try:
        n = float(raw if raw is not None else default)
    except (TypeError, ValueError):
        n = default
    if n != n:
        n = default
    return max(lo, min(hi, n))


def pack_selflift(
    *,
    split_mode: str = "highres_steps",
    highres_steps: int = DEFAULT_HIGHRES_STEPS,
    transition_step: int = DEFAULT_TRANSITION_STEP,
    lowres_scale: float = DEFAULT_LOWRES_SCALE,
    latent_upscale_model=None,
    sampler_mode: str = "euler",
    native_low_carry: bool = True,
    rho: float = DEFAULT_RHO,
    w_min: float = DEFAULT_W_MIN,
    w_max: float = DEFAULT_W_MAX,
    latent_upsample: str = "bilinear",
    enable_latent_chunking: bool = False,
    enable_tiling: bool = False,
    tile_count: int = DEFAULT_SPATIAL_TILES,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    model_hires=None,
) -> dict[str, Any]:
    mode = str(split_mode or "highres_steps").strip().lower()
    if mode not in SPLIT_MODES:
        mode = "highres_steps"
    samp = str(sampler_mode or "euler").strip().lower()
    if samp not in SAMPLER_MODES:
        samp = "euler"
    up = str(latent_upsample or "bilinear").strip().lower()
    if up not in UPSAMPLE_MODES:
        up = "bilinear"
    w_lo = _clamp_float(w_min, DEFAULT_W_MIN, 0.0, 1.0)
    w_hi = _clamp_float(w_max, DEFAULT_W_MAX, 0.0, 1.0)
    if w_hi < w_lo:
        w_lo, w_hi = w_hi, w_lo
    latent_mod, latent_name = resolve_latent_upscale_ref(latent_upscale_model)
    return {
        "enabled": True,
        "split_mode": mode,
        "highres_steps": _clamp_int(highres_steps, DEFAULT_HIGHRES_STEPS, 1, 64),
        "transition_step": _clamp_int(transition_step, DEFAULT_TRANSITION_STEP, 1, 200),
        "lowres_scale": _clamp_float(lowres_scale, DEFAULT_LOWRES_SCALE, 0.25, 1.0),
        "sampler_mode": samp,
        "native_low_carry": bool(native_low_carry),
        "rho": _clamp_float(rho, DEFAULT_RHO, 0.0, 1.0),
        "w_min": w_lo,
        "w_max": w_hi,
        "latent_upsample": up,
        "enable_latent_chunking": bool(enable_latent_chunking),
        "enable_tiling": bool(enable_tiling),
        "tile_count": _clamp_int(tile_count, DEFAULT_SPATIAL_TILES, 1, MAX_SPATIAL_TILES),
        "tile_overlap": _clamp_int(tile_overlap, DEFAULT_TILE_OVERLAP, 0, 2048),
        "latent_upscale_ref": latent_mod,
        "h3_latent_model": latent_name,
        "sample_model": model_hires,
    }


def normalize_selflift_pack(raw) -> dict[str, Any] | None:
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        return None
    if not raw.get("enabled"):
        return None
    return pack_selflift(
        split_mode=raw.get("split_mode") or "highres_steps",
        highres_steps=raw.get("highres_steps", DEFAULT_HIGHRES_STEPS),
        transition_step=raw.get("transition_step", DEFAULT_TRANSITION_STEP),
        lowres_scale=raw.get("lowres_scale", DEFAULT_LOWRES_SCALE),
        latent_upscale_model=(
            raw.get("latent_upscale_ref")
            if raw.get("latent_upscale_ref") is not None
            else raw.get("h3_latent_model") or raw.get("latent_upscale_model")
        ),
        sampler_mode=raw.get("sampler_mode") or "euler",
        native_low_carry=raw.get("native_low_carry", True),
        rho=raw.get("rho", DEFAULT_RHO),
        w_min=raw.get("w_min", DEFAULT_W_MIN),
        w_max=raw.get("w_max", DEFAULT_W_MAX),
        latent_upsample=raw.get("latent_upsample") or "bilinear",
        enable_latent_chunking=raw.get("enable_latent_chunking", False),
        enable_tiling=raw.get("enable_tiling", False),
        tile_count=raw.get("tile_count", DEFAULT_SPATIAL_TILES),
        tile_overlap=raw.get("tile_overlap", DEFAULT_TILE_OVERLAP),
        model_hires=raw.get("sample_model"),
    )


def selflift_enabled(plan) -> bool:
    pack = getattr(plan, "selflift", None)
    return isinstance(pack, dict) and bool(pack.get("enabled"))


def selflift_will_run(plan, _seg) -> bool:
    return selflift_enabled(plan)


def resolve_transition_k(pack: dict[str, Any], n_steps: int) -> int:
    """Index k in sigmas where low-res ends (k model evals at low-res)."""
    n = max(2, int(n_steps))
    mode = str(pack.get("split_mode") or "highres_steps")
    if mode == "transition_step":
        k = int(pack.get("transition_step") or DEFAULT_TRANSITION_STEP)
    else:
        high = int(pack.get("highres_steps") or DEFAULT_HIGHRES_STEPS)
        k = n - high
    if k < 1:
        k = 1
    if k > n - 1:
        k = n - 1
    return int(k)


def require_euler(sampler_name: str, pack: dict[str, Any]) -> str:
    """SelfLift prefix uses Euler; do not silently fall back to another sampler."""
    name = str(sampler_name or "").strip() or "euler"
    mode = str(pack.get("sampler_mode") or "euler")
    if mode == "euler":
        return "euler"
    if name.lower() not in EULER_NAMES:
        raise ValueError(
            f"SelfLift 只支持 Euler（s_churn=0）。导演台当前采样器是 {name!r}。"
            "请把导演台采样器改成 euler，或在 SelfLift 节点把「采样器」设为 euler。"
        )
    return name


def selflift_fingerprint(plan) -> dict[str, Any]:
    """Empty when unconnected so existing first-pass fingerprints stay identical."""
    pack = getattr(plan, "selflift", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return {}
    return {
        "selflift": True,
        "sl_split": pack.get("split_mode") or "",
        "sl_high": int(pack.get("highres_steps") or 0),
        "sl_trans": int(pack.get("transition_step") or 0),
        "sl_scale": round(float(pack.get("lowres_scale") or 0), 4),
        "sl_model": pack.get("h3_latent_model") or "",
        "sl_samp": pack.get("sampler_mode") or "",
        "sl_carry": bool(pack.get("native_low_carry", True)),
        "sl_rho": round(float(pack.get("rho") or 0), 4),
        "sl_wmin": round(float(pack.get("w_min") or 0), 4),
        "sl_wmax": round(float(pack.get("w_max") or 0), 4),
        "sl_up": pack.get("latent_upsample") or "",
        "sl_chunk": bool(pack.get("enable_latent_chunking")),
        "sl_tile": bool(pack.get("enable_tiling")),
        "sl_tiles": int(pack.get("tile_count") or 0),
        "sl_overlap": int(pack.get("tile_overlap") or 0),
        "sl_hires_model": bool(pack.get("sample_model") is not None),
    }


def selflift_report_line(plan) -> str | None:
    if not selflift_enabled(plan):
        return None
    pack = plan.selflift
    scale = float(pack.get("lowres_scale") or DEFAULT_LOWRES_SCALE)
    mode = pack.get("split_mode") or "highres_steps"
    if mode == "transition_step":
        split = f"transition_step={int(pack.get('transition_step') or 0)}"
    else:
        split = f"highres_steps={int(pack.get('highres_steps') or 0)}"
    return (
        "SelfLift: "
        f"scale={scale:.2f} {split} "
        f"3D={pack.get('h3_latent_model') or '(none)'} "
        f"rho={float(pack.get('rho') or 0):.2f} "
        f"carry={'on' if pack.get('native_low_carry', True) else 'off'}"
    )


def default_latent_upscale_choice() -> str:
    names = list_h3_latent_upscale_models()
    return names[0] if names else ""
