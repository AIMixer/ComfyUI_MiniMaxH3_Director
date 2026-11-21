"""Graph pack for Director.vsr. Unconnected → None, zero execute impact."""

from __future__ import annotations

from typing import Any

MMX_DIR_VSR = "MMX_DIR_VSR"
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"


def _clamp_int(raw: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _seed(raw: Any) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 0
    if n < 0:
        n = 0
    return n & 0xFFFFFFFFFFFFFFFF


def _clamp_float(raw: Any, lo: float, hi: float, default: float) -> float:
    try:
        n = float(raw)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _factor_tag(model, vae) -> str:
    """x2/x4 identity so a swapped Flash decoder invalidates the final cache."""
    factor = ""
    stride = ""
    try:
        fsm = getattr(vae, "first_stage_model", None)
        raw = getattr(fsm, "upscale_factor", None)
        if raw is not None:
            factor = str(int(raw))
    except (TypeError, ValueError):
        factor = ""
    try:
        stride = str(getattr(model.model.diffusion_model, "latent_stride", "") or "")
    except Exception:
        stride = ""
    return f"{factor}:{stride}"


def pack_relay_vsr(
    *,
    model,
    vae,
    seed: int = 0,
    steps: int = 1,
    cfg: float = 1.0,
    sampler: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    denoise: float = 1.0,
    detail: float = 1.0,
) -> dict[str, Any]:
    steps_n = _clamp_int(steps, 1, 100, 1)
    return {
        "enabled": True,
        "model": model,
        "vae": vae,
        "seed": _seed(seed),
        "steps": steps_n,
        "cfg": _clamp_float(cfg, 0.0, 100.0, 1.0),
        "sampler": str(sampler or DEFAULT_SAMPLER),
        "scheduler": str(scheduler or DEFAULT_SCHEDULER),
        "denoise": _clamp_float(denoise, 0.0, 1.0, 1.0),
        "detail": _clamp_float(detail, 0.0, 1.0, 1.0),
        "factor_tag": _factor_tag(model, vae),
    }


def normalize_relay_vsr_pack(raw) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or raw.get("enabled") is False:
        return None
    if raw.get("model") is None or raw.get("vae") is None:
        return None
    if raw.get("factor_tag") is None:
        raw = dict(raw)
        raw["factor_tag"] = _factor_tag(raw["model"], raw["vae"])
        raw["enabled"] = True
    return raw


def relay_vsr_enabled(plan) -> bool:
    pack = getattr(plan, "vsr", None)
    return isinstance(pack, dict) and bool(pack.get("enabled")) and pack.get("model") is not None and pack.get("vae") is not None


def relay_vsr_fingerprint(plan) -> dict[str, Any]:
    """Empty when unconnected so existing segment-cache fingerprints stay identical."""
    if not relay_vsr_enabled(plan):
        return {}
    pack = plan.vsr
    return {
        "vsr": True,
        "vsr_steps": int(pack.get("steps") or 1),
        "vsr_cfg": round(float(pack.get("cfg") or 1.0), 4),
        "vsr_sampler": pack.get("sampler") or DEFAULT_SAMPLER,
        "vsr_scheduler": pack.get("scheduler") or DEFAULT_SCHEDULER,
        "vsr_denoise": round(float(pack.get("denoise") or 1.0), 4),
        "vsr_detail": round(float(pack.get("detail") or 1.0), 4),
        "vsr_seed": int(pack.get("seed") or 0),
        "vsr_factor": str(pack.get("factor_tag") or ""),
    }


def relay_vsr_report_line(plan) -> str | None:
    if not relay_vsr_enabled(plan):
        return None
    pack = plan.vsr
    factor = str(pack.get("factor_tag") or "").split(":", 1)[0]
    factor_note = f" x{factor}" if factor else ""
    return (
        "RelayVSR:"
        f"{factor_note} steps={int(pack.get('steps') or 1)} "
        f"cfg={float(pack.get('cfg') or 1):.2f} "
        f"{pack.get('sampler') or DEFAULT_SAMPLER}/{pack.get('scheduler') or DEFAULT_SCHEDULER} "
        f"denoise={float(pack.get('denoise') or 1):.2f} "
        f"detail={float(pack.get('detail') or 1):.2f} "
        f"seed={int(pack.get('seed') or 0)} "
        "(after refine and face refine)"
    )
