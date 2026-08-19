"""External Refine pack for MiniMax H3 Director (graph-wired config).

Connect ``MiniMaxH3DirectorRefine.refine`` → ``MiniMaxH3Director.refine``.
Unconnected = current single-pass sampling.
"""

from __future__ import annotations

import math
from typing import Any

from ..lib.image_prep import MINIMAX_CANVAS_STRIDE, ensure_minimax_canvas

MMX_DIR_REFINE = "MMX_DIR_REFINE"

REFINE_MODES = ("refine", "upscale")
SEED_MODES = ("inherit", "offset")
UPSCALE_METHODS = ("lanczos", "nvidia_rtx_vsr", "latent")
MAX_REFINE_PASSES = 9999

FOLLOW_DIRECTOR_ASPECT = "跟随导演台"
CUSTOM_ASPECT_RATIO = "自定义"
DEFAULT_UPSCALE_MEGAPIXELS = 1.0

# Same labels/ratios as Director output bar / official ResolutionSelector.
RESOLUTION_ASPECTS = (
    ("1:1 (方形)", 1, 1),
    ("2:3 (竖版照片)", 2, 3),
    ("3:2 (横版照片)", 3, 2),
    ("3:4 (竖版标准)", 3, 4),
    ("4:3 (标准)", 4, 3),
    ("9:16 (竖屏)", 9, 16),
    ("16:9 (宽屏)", 16, 9),
    ("21:9 (超宽)", 21, 9),
)

ASPECT_RATIO_CHOICES = (
    FOLLOW_DIRECTOR_ASPECT,
    *[row[0] for row in RESOLUTION_ASPECTS],
    CUSTOM_ASPECT_RATIO,
)

_ASPECT_ALIASES = {
    "Follow Director": FOLLOW_DIRECTOR_ASPECT,
    "follow": FOLLOW_DIRECTOR_ASPECT,
    "Custom": CUSTOM_ASPECT_RATIO,
    "自定义 (Custom)": CUSTOM_ASPECT_RATIO,
    "1:1 (Square)": "1:1 (方形)",
    "2:3 (Portrait Photo)": "2:3 (竖版照片)",
    "3:2 (Photo)": "3:2 (横版照片)",
    "3:4 (Portrait Standard)": "3:4 (竖版标准)",
    "4:3 (Standard)": "4:3 (标准)",
    "9:16 (Portrait Widescreen)": "9:16 (竖屏)",
    "16:9 (Widescreen)": "16:9 (宽屏)",
    "21:9 (Ultrawide)": "21:9 (超宽)",
}


def infer_upscale_target(base_w: int, base_h: int) -> tuple[int, int]:
    """Default 720p-class canvas from a 480p-class source (snap to H3 ×32)."""
    w, h = int(base_w or 0), int(base_h or 0)
    if w <= 0 or h <= 0:
        return ensure_minimax_canvas(1280, 720)
    if w >= h:
        nh = 720
        nw = max(32, round(w * nh / h))
        return ensure_minimax_canvas(nw, nh)
    nw = 720
    nh = max(32, round(h * nw / w))
    return ensure_minimax_canvas(nw, nh)


def normalize_aspect_ratio(aspect_ratio: str | None) -> str:
    # Legacy workflows mapped old target_width=0 onto this combo.
    if aspect_ratio in (None, "", 0, 0.0, "0", "0.0", False):
        return FOLLOW_DIRECTOR_ASPECT
    v = str(aspect_ratio).strip()
    if not v or v in {"0", "0.0", "None", "null"}:
        return FOLLOW_DIRECTOR_ASPECT
    if v in _ASPECT_ALIASES:
        return _ASPECT_ALIASES[v]
    if v in ASPECT_RATIO_CHOICES:
        return v
    prefix = v.split(" ", 1)[0]
    for label, _, _ in RESOLUTION_ASPECTS:
        if label == prefix or label.startswith(f"{prefix} "):
            return label
    if v.startswith("自定义") or v.lower() == "custom":
        return CUSTOM_ASPECT_RATIO
    return FOLLOW_DIRECTOR_ASPECT


def is_follow_director_aspect(aspect_ratio: str | None) -> bool:
    return normalize_aspect_ratio(aspect_ratio) == FOLLOW_DIRECTOR_ASPECT


def is_custom_aspect_ratio(aspect_ratio: str | None) -> bool:
    return normalize_aspect_ratio(aspect_ratio) == CUSTOM_ASPECT_RATIO


def resolution_from_selector(
    aspect_ratio: str,
    megapixels: float,
    multiple: int = MINIMAX_CANVAS_STRIDE,
) -> tuple[int, int] | None:
    """Director ResolutionSelector math: aspect + MP → W×H snapped to ×32."""
    ar = normalize_aspect_ratio(aspect_ratio)
    if ar in {FOLLOW_DIRECTOR_ASPECT, CUSTOM_ASPECT_RATIO}:
        return None
    row = next((r for r in RESOLUTION_ASPECTS if r[0] == ar), None)
    if row is None:
        return None
    _, w_ratio, h_ratio = row
    try:
        mp = float(megapixels)
    except (TypeError, ValueError):
        mp = DEFAULT_UPSCALE_MEGAPIXELS
    mp = min(16.0, max(0.1, mp))
    mult = max(8, int(multiple or MINIMAX_CANVAS_STRIDE))
    scale = math.sqrt((mp * 1024 * 1024) / (w_ratio * h_ratio))
    width = int(round((w_ratio * scale) / mult) * mult)
    height = int(round((h_ratio * scale) / mult) * mult)
    return ensure_minimax_canvas(max(width, mult), max(height, mult))


def resolve_refine_target(
    *,
    aspect_ratio: str = FOLLOW_DIRECTOR_ASPECT,
    megapixels: float = DEFAULT_UPSCALE_MEGAPIXELS,
    width: int = 0,
    height: int = 0,
    target_width: int = 0,
    target_height: int = 0,
) -> tuple[int, int]:
    """Return (0, 0) to follow Director canvas; otherwise an explicit ×32 canvas."""
    ar = normalize_aspect_ratio(aspect_ratio)
    legacy_w, legacy_h = int(target_width or 0), int(target_height or 0)
    w, h = int(width or 0), int(height or 0)
    if is_follow_director_aspect(ar):
        if (legacy_w > 0 or legacy_h > 0) and w <= 0 and h <= 0:
            return ensure_minimax_canvas(max(legacy_w, 32), max(legacy_h, 32))
        return 0, 0
    if is_custom_aspect_ratio(ar):
        cw = w or legacy_w or 1280
        ch = h or legacy_h or 720
        return ensure_minimax_canvas(max(cw, 32), max(ch, 32))
    resolved = resolution_from_selector(ar, megapixels)
    if resolved is not None:
        return resolved
    return 0, 0


def pack_refine(
    *,
    mode: str = "refine",
    denoise: float = 0.25,
    steps: int = 10,
    passes: int = 1,
    seed_mode: str = "inherit",
    aspect_ratio: str = FOLLOW_DIRECTOR_ASPECT,
    megapixels: float = DEFAULT_UPSCALE_MEGAPIXELS,
    width: int = 0,
    height: int = 0,
    target_width: int = 0,
    target_height: int = 0,
    skip_fl2v: bool = True,
    upscale_method: str = "lanczos",
    upscale_model=None,
    sample_model=None,
) -> dict[str, Any]:
    mode = str(mode or "refine").strip().lower()
    if mode not in REFINE_MODES:
        mode = "refine"
    seed_mode = str(seed_mode or "inherit").strip().lower()
    if seed_mode not in SEED_MODES:
        seed_mode = "inherit"
    method = str(upscale_method or "lanczos").strip().lower()
    if method not in UPSCALE_METHODS:
        method = "lanczos"
    ar = normalize_aspect_ratio(aspect_ratio)
    tw, th = resolve_refine_target(
        aspect_ratio=ar,
        megapixels=megapixels,
        width=width,
        height=height,
        target_width=target_width,
        target_height=target_height,
    )
    return {
        "enabled": True,
        "mode": mode,
        "denoise": float(max(0.05, min(0.85, denoise))),
        "steps": int(max(0, min(200, steps))),
        "passes": refine_passes_for({"passes": passes}),
        "seed_mode": seed_mode,
        "aspect_ratio": ar,
        "megapixels": float(megapixels or DEFAULT_UPSCALE_MEGAPIXELS),
        "target_width": tw,
        "target_height": th,
        "skip_fl2v": bool(skip_fl2v),
        "upscale_method": method,
        "upscale_model": upscale_model,
        "has_upscale_model": upscale_model is not None,
        "sample_model": sample_model,
        "has_sample_model": sample_model is not None,
    }


def normalize_refine_pack(
    raw,
    *,
    base_width: int = 0,
    base_height: int = 0,
) -> dict[str, Any] | None:
    """Director execute: None if unconnected / invalid."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("enabled") is False:
        return None
    mode = str(raw.get("mode") or "refine").strip().lower()
    if mode not in REFINE_MODES:
        mode = "refine"
    ar = normalize_aspect_ratio(raw.get("aspect_ratio"))
    tw, th = int(raw.get("target_width") or 0), int(raw.get("target_height") or 0)
    if mode == "upscale" and (tw <= 0 or th <= 0):
        if not is_follow_director_aspect(ar) and not is_custom_aspect_ratio(ar):
            resolved = resolution_from_selector(ar, raw.get("megapixels") or DEFAULT_UPSCALE_MEGAPIXELS)
            if resolved:
                tw, th = resolved
        if tw <= 0 or th <= 0:
            tw, th = infer_upscale_target(base_width, base_height)
    elif tw > 0 and th > 0:
        tw, th = ensure_minimax_canvas(tw, th)
    seed_mode = str(raw.get("seed_mode") or "inherit").strip().lower()
    if seed_mode not in SEED_MODES:
        seed_mode = "inherit"
    upscale = raw.get("upscale_model")
    method = str(raw.get("upscale_method") or "lanczos").strip().lower()
    if method not in UPSCALE_METHODS:
        method = "lanczos"
    sample_model = raw.get("sample_model")
    if sample_model is None:
        sample_model = raw.get("model")
    return {
        "enabled": True,
        "mode": mode,
        "denoise": float(max(0.05, min(0.85, float(raw.get("denoise") or 0.25)))),
        "steps": int(max(0, min(200, int(raw.get("steps") or 0)))),
        "passes": refine_passes_for(raw),
        "seed_mode": seed_mode,
        "aspect_ratio": ar,
        "megapixels": float(raw.get("megapixels") or DEFAULT_UPSCALE_MEGAPIXELS),
        "target_width": tw,
        "target_height": th,
        "skip_fl2v": bool(raw.get("skip_fl2v", True)),
        "upscale_method": method,
        "upscale_model": upscale,
        "has_upscale_model": upscale is not None,
        "sample_model": sample_model,
        "has_sample_model": sample_model is not None,
    }


def refine_will_sample(plan, seg) -> bool:
    """True when this segment will run a second sample / upscale pass."""
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return False
    if pack.get("skip_fl2v", True) and getattr(seg, "task_key", "") == "fl2v":
        return False
    return True


def refine_steps_for(pack: dict[str, Any], first_steps: int) -> int:
    n = int(pack.get("steps") or 0)
    if n > 0:
        return n
    return max(8, int(round(int(first_steps) * 0.4)))


def refine_passes_for(pack: dict[str, Any] | None) -> int:
    try:
        n = int((pack or {}).get("passes") or 1)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(MAX_REFINE_PASSES, n))


def refine_model_for(pack: dict[str, Any] | None, fallback):
    """Second-pass UNET. Unconnected Refine.refine_model → Director main model."""
    custom = (pack or {}).get("sample_model")
    if custom is None:
        custom = (pack or {}).get("model")
    return fallback if custom is None else custom


def refine_seed_for(pack: dict[str, Any], seed: int, pass_index: int = 0) -> int:
    if pack.get("seed_mode") == "offset":
        return int(seed) + 1 + int(max(0, pass_index))
    return int(seed)


def refine_fingerprint(plan) -> dict[str, Any]:
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return {"refine": False}
    return {
        "refine": True,
        "refine_mode": pack.get("mode") or "refine",
        "refine_denoise": round(float(pack.get("denoise") or 0), 3),
        "refine_steps": int(pack.get("steps") or 0),
        "refine_passes": refine_passes_for(pack),
        "refine_seed_mode": pack.get("seed_mode") or "inherit",
        "refine_target": f"{int(pack.get('target_width') or 0)}x{int(pack.get('target_height') or 0)}",
        "refine_aspect": pack.get("aspect_ratio") or FOLLOW_DIRECTOR_ASPECT,
        "refine_megapixels": round(float(pack.get("megapixels") or 0), 3),
        "refine_upscale_method": pack.get("upscale_method") or "lanczos",
        "refine_upscale_model": bool(pack.get("has_upscale_model")),
        "refine_sample_model": bool(pack.get("has_sample_model") or pack.get("sample_model") is not None),
        "refine_skip_fl2v": bool(pack.get("skip_fl2v", True)),
    }


def refine_report_line(plan) -> str | None:
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return None
    mode = pack.get("mode") or "refine"
    extra = ""
    if mode == "upscale":
        ar = pack.get("aspect_ratio") or FOLLOW_DIRECTOR_ASPECT
        method = pack.get("upscale_method") or "lanczos"
        if method == "nvidia_rtx_vsr":
            how = "nvidia_rtx_vsr"
        elif pack.get("has_upscale_model"):
            how = "upscale_model"
        else:
            how = "interpolate"
        extra = (
            f", {ar} → {int(pack.get('target_width') or 0)}×{int(pack.get('target_height') or 0)}"
            f", {how}"
        )
    n_passes = refine_passes_for(pack)
    pass_note = f", passes={n_passes}" if n_passes > 1 else ""
    model_note = (
        ", 二采模型" if (pack.get("has_sample_model") or pack.get("sample_model") is not None) else ""
    )
    return (
        f"Refine: ON ({mode}, denoise={float(pack.get('denoise') or 0):.2f}, "
        f"steps={int(pack.get('steps') or 0) or 'auto'}{pass_note}{model_note}{extra})"
    )
