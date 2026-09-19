"""Graph pack for Director.face_refine. Unconnected → None, zero execute impact."""

from __future__ import annotations

from typing import Any

MMX_DIR_FACE_REFINE = "MMX_DIR_FACE_REFINE"

SELECT_MODES = ("largest_face", "centre_most")
CANVAS_MODES = ("manual", "auto_capped_768")
PASTE_REGIONS = ("face_only", "face_ellipse", "full_crop")
SEED_MODES = ("inherit", "offset")
DEFAULT_DETECTOR = "face_yolov8m.pt"
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"


def detector_choices() -> list[str]:
    """Ultralytics face weights; node must load even if the folder is empty."""
    import os

    import folder_paths

    names: list[str] = []
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            names.extend(folder_paths.get_filename_list(key))
        except Exception:
            pass
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    if out:
        return out
    base = getattr(folder_paths, "models_dir", "models")
    for sub in ("ultralytics/bbox", "ultralytics"):
        folder = os.path.join(base, *sub.split("/"))
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(".pt") and fname not in seen:
                seen.add(fname)
                out.append(fname)
    return out or [DEFAULT_DETECTOR]


def _clamp_int(raw: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _clamp_float(raw: Any, lo: float, hi: float, default: float) -> float:
    try:
        n = float(raw)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def pack_face_refine(
    *,
    detector: str = DEFAULT_DETECTOR,
    confidence: float = 0.35,
    crop_factor: float = 2.5,
    canvas_width: int = 768,
    canvas_height: int = 768,
    canvas_mode: str = "manual",
    select: str = "largest_face",
    denoise: float = 0.40,
    steps: int = 8,
    sampler: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    seed_mode: str = "inherit",
    paste_region: str = "face_only",
    mask_dilation: int = 16,
    feather: int = 24,
    colour_match: float = 1.0,
    blend: float = 1.0,
    sigmas=None,
) -> dict[str, Any]:
    mode = str(canvas_mode or "manual").strip().lower()
    if mode not in CANVAS_MODES:
        mode = "manual"
    sel = str(select or "largest_face").strip().lower()
    if sel not in SELECT_MODES:
        sel = "largest_face"
    paste = str(paste_region or "face_only").strip().lower()
    if paste not in PASTE_REGIONS:
        paste = "face_only"
    seed = str(seed_mode or "inherit").strip().lower()
    if seed not in SEED_MODES:
        seed = "inherit"
    from ...lib.image_prep import ensure_minimax_canvas
    from ..refine_pack import is_refine_sigmas_tensor, parse_refine_sigmas

    sigma_tensor = sigmas if is_refine_sigmas_tensor(sigmas) else None
    parsed = (
        parse_refine_sigmas(sigma_tensor, fallback=False) if sigma_tensor is not None else ()
    )
    canvas_width, canvas_height = ensure_minimax_canvas(
        _clamp_int(canvas_width, 128, 1344, 768),
        _clamp_int(canvas_height, 128, 1344, 768),
    )
    return {
        "enabled": True,
        "detector": str(detector or DEFAULT_DETECTOR).strip() or DEFAULT_DETECTOR,
        "confidence": _clamp_float(confidence, 0.05, 0.95, 0.35),
        "crop_factor": _clamp_float(crop_factor, 1.2, 8.0, 2.5),
        "canvas_width": int(canvas_width),
        "canvas_height": int(canvas_height),
        "canvas_mode": mode,
        "select": sel,
        "denoise": _clamp_float(denoise, 0.02, 1.0, 0.40),
        "steps": _clamp_int(steps, 1, 50, 8),
        "sampler": str(sampler or DEFAULT_SAMPLER).strip() or DEFAULT_SAMPLER,
        "scheduler": str(scheduler or DEFAULT_SCHEDULER).strip() or DEFAULT_SCHEDULER,
        "seed_mode": seed,
        "paste_region": paste,
        "mask_dilation": _clamp_int(mask_dilation, 0, 256, 16),
        "feather": _clamp_int(feather, 0, 256, 24),
        "colour_match": _clamp_float(colour_match, 0.0, 1.0, 1.0),
        "blend": _clamp_float(blend, 0.0, 1.0, 1.0),
        "sigmas": ",".join(f"{x:g}" for x in parsed),
        "sigmas_parsed": parsed,
        "sigmas_tensor": sigma_tensor,
        "has_sigmas_tensor": sigma_tensor is not None,
    }


def normalize_face_refine_pack(raw) -> dict[str, Any] | None:
    """Director execute: None if unconnected / invalid."""
    if raw is None or not isinstance(raw, dict):
        return None
    if raw.get("enabled") is False:
        return None
    return pack_face_refine(
        detector=raw.get("detector") or DEFAULT_DETECTOR,
        confidence=raw.get("confidence", 0.35),
        crop_factor=raw.get("crop_factor", 2.5),
        canvas_width=raw.get("canvas_width", 768),
        canvas_height=raw.get("canvas_height", 768),
        canvas_mode=raw.get("canvas_mode") or "manual",
        select=raw.get("select") or "largest_face",
        denoise=raw.get("denoise", 0.40),
        steps=raw.get("steps", 8),
        sampler=raw.get("sampler") or DEFAULT_SAMPLER,
        scheduler=raw.get("scheduler") or DEFAULT_SCHEDULER,
        seed_mode=raw.get("seed_mode") or "inherit",
        paste_region=raw.get("paste_region") or "face_only",
        mask_dilation=raw.get("mask_dilation", 16),
        feather=raw.get("feather", 24),
        colour_match=raw.get("colour_match", 1.0),
        blend=raw.get("blend", 1.0),
        sigmas=raw.get("sigmas_tensor") if raw.get("sigmas_tensor") is not None else raw.get("sigmas"),
    )


def face_refine_fingerprint(plan) -> dict[str, Any]:
    """Empty when unconnected so existing segment-cache fingerprints stay identical."""
    pack = getattr(plan, "face_refine", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return {}
    return {
        "face_refine": True,
        "fr_detector": pack.get("detector") or "",
        "fr_confidence": round(float(pack.get("confidence") or 0), 4),
        "fr_crop": round(float(pack.get("crop_factor") or 0), 4),
        "fr_canvas": (
            f"{int(pack.get('canvas_width') or 0)}x{int(pack.get('canvas_height') or 0)}"
            f":{pack.get('canvas_mode') or 'manual'}"
        ),
        "fr_select": pack.get("select") or "",
        "fr_denoise": round(float(pack.get("denoise") or 0), 4),
        "fr_steps": int(pack.get("steps") or 0),
        "fr_sampler": pack.get("sampler") or "",
        "fr_scheduler": pack.get("scheduler") or "",
        "fr_seed_mode": pack.get("seed_mode") or "",
        "fr_paste": pack.get("paste_region") or "",
        "fr_feather": int(pack.get("feather") or 0),
        "fr_dilation": int(pack.get("mask_dilation") or 0),
        "fr_colour": round(float(pack.get("colour_match") or 0), 4),
        "fr_blend": round(float(pack.get("blend") or 0), 4),
        "fr_sigmas": ",".join(f"{x:.4f}" for x in (pack.get("sigmas_parsed") or ())),
        # Bump when continuity seam fade lands so old unfaded stitch caches miss.
        "fr_seam_fade": 12,
    }


def face_refine_enabled(plan) -> bool:
    pack = getattr(plan, "face_refine", None)
    return isinstance(pack, dict) and bool(pack.get("enabled"))


def face_refine_report_line(plan) -> str | None:
    if not face_refine_enabled(plan):
        return None
    pack = plan.face_refine
    return (
        "FaceRefine: "
        f"{pack.get('detector') or DEFAULT_DETECTOR} "
        f"denoise={float(pack.get('denoise') or 0):.2f} "
        f"steps={int(pack.get('steps') or 0)} "
        f"canvas={int(pack.get('canvas_width') or 0)}x{int(pack.get('canvas_height') or 0)} "
        f"({pack.get('canvas_mode') or 'manual'})"
    )
