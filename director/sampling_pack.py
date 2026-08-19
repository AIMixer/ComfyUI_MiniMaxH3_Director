"""External Sampling pack for MiniMax H3 Director (graph-wired config).

Connect ``MiniMaxH3DirectorSampling.sampling`` → ``MiniMaxH3Director.sampling``.
Unconnected = the Director's own widgets (backward compatible).

The pack node owns every sampling knob: steps / sampler / scheduler / cfg /
seed / sigma shift / sigma refine. It also accepts ComfyUI-standard ``SAMPLER``
(KSamplerSelect or any third-party sampler node) and ``SIGMAS`` (BasicScheduler
or any third-party scheduler node) so the whole Director can reuse the
ecosystem's custom samplers and schedules.
"""

from __future__ import annotations

from typing import Any

MMX_DIR_SAMPLING = "MMX_DIR_SAMPLING"

DEFAULT_STEPS = 25
DEFAULT_SAMPLER = "res_multistep"
DEFAULT_SCHEDULER = "simple"
DEFAULT_CFG = 1.0
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0
DEFAULT_SIGMA_TAIL_STEPS = 1


def pack_sampling(
    *,
    steps: int = DEFAULT_STEPS,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler_name: str = DEFAULT_SCHEDULER,
    cfg: float = DEFAULT_CFG,
    seed: int = 0,
    shift_video: float = DEFAULT_SHIFT_VIDEO,
    shift_audio: float = DEFAULT_SHIFT_AUDIO,
    sigma_refine: bool = False,
    sigma_tail_steps: int = DEFAULT_SIGMA_TAIL_STEPS,
    sampler=None,
    sigmas=None,
) -> dict[str, Any]:
    """Pack sampling settings into one value to hand to the Director."""
    try:
        steps_i = int(steps)
    except (TypeError, ValueError):
        steps_i = DEFAULT_STEPS
    steps_i = max(1, min(200, steps_i))
    try:
        cfg_f = float(cfg)
    except (TypeError, ValueError):
        cfg_f = DEFAULT_CFG
    try:
        seed_i = int(seed)
    except (TypeError, ValueError):
        seed_i = 0
    try:
        sv = float(shift_video)
    except (TypeError, ValueError):
        sv = DEFAULT_SHIFT_VIDEO
    try:
        sa = float(shift_audio)
    except (TypeError, ValueError):
        sa = DEFAULT_SHIFT_AUDIO
    try:
        tail = int(sigma_tail_steps)
    except (TypeError, ValueError):
        tail = DEFAULT_SIGMA_TAIL_STEPS
    tail = max(0, min(50, tail))
    sampler_name = str(sampler_name or DEFAULT_SAMPLER)
    scheduler_name = str(scheduler_name or DEFAULT_SCHEDULER)
    return {
        "enabled": True,
        "steps": steps_i,
        "sampler": sampler_name,
        "scheduler": scheduler_name,
        "cfg": cfg_f,
        "seed": seed_i,
        "shift_video": sv,
        "shift_audio": sa,
        "sigma_refine": bool(sigma_refine),
        "sigma_tail_steps": tail,
        "sampler_obj": sampler,
        "sigmas": sigmas,
    }


def normalize_sampling_pack(raw) -> dict[str, Any] | None:
    """Director execute: None if unconnected / invalid."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("enabled") is False:
        return None
    steps = raw.get("steps") or DEFAULT_STEPS
    return {
        "enabled": True,
        "steps": max(1, min(200, int(steps))),
        "sampler": str(raw.get("sampler") or DEFAULT_SAMPLER),
        "scheduler": str(raw.get("scheduler") or DEFAULT_SCHEDULER),
        "cfg": float(raw.get("cfg") or DEFAULT_CFG),
        "seed": int(raw.get("seed") or 0),
        "shift_video": float(raw.get("shift_video") or DEFAULT_SHIFT_VIDEO),
        "shift_audio": float(raw.get("shift_audio") or DEFAULT_SHIFT_AUDIO),
        "sigma_refine": bool(raw.get("sigma_refine", False)),
        "sigma_tail_steps": max(0, min(50, int(raw.get("sigma_tail_steps") or 0))),
        "sampler_obj": raw.get("sampler_obj"),
        "sigmas": raw.get("sigmas"),
    }


def sampling_steps_for(pack: dict[str, Any] | None, fallback: int) -> int:
    if isinstance(pack, dict) and pack.get("enabled"):
        return int(pack.get("steps") or fallback)
    return int(fallback)


def sampling_sampler_name_for(pack: dict[str, Any] | None, fallback: str) -> str:
    if isinstance(pack, dict) and pack.get("enabled"):
        return str(pack.get("sampler") or fallback)
    return str(fallback)


def sampling_scheduler_for(pack: dict[str, Any] | None, fallback: str) -> str:
    if isinstance(pack, dict) and pack.get("enabled"):
        return str(pack.get("scheduler") or fallback)
    return str(fallback)


def sampling_cfg_for(pack: dict[str, Any] | None, fallback: float) -> float:
    if isinstance(pack, dict) and pack.get("enabled"):
        return float(pack.get("cfg") or fallback)
    return float(fallback)


def sampling_seed_for(pack: dict[str, Any] | None, fallback: int) -> int:
    if isinstance(pack, dict) and pack.get("enabled"):
        return int(pack.get("seed") or fallback)
    return int(fallback)


def sampling_shift_for(
    pack: dict[str, Any] | None, fallback_video: float, fallback_audio: float
) -> tuple[float, float]:
    if isinstance(pack, dict) and pack.get("enabled"):
        return float(pack.get("shift_video") or fallback_video), float(
            pack.get("shift_audio") or fallback_audio
        )
    return float(fallback_video), float(fallback_audio)


def sampling_sampler_obj_for(pack: dict[str, Any] | None):
    if isinstance(pack, dict) and pack.get("enabled"):
        return pack.get("sampler_obj")
    return None


def sampling_sigmas_for(pack: dict[str, Any] | None):
    if isinstance(pack, dict) and pack.get("enabled"):
        return pack.get("sigmas")
    return None


def sampling_sigma_refine_for(pack: dict[str, Any] | None, fallback: bool = False) -> bool:
    if isinstance(pack, dict) and pack.get("enabled"):
        return bool(pack.get("sigma_refine", fallback))
    return bool(fallback)


def sampling_sigma_tail_for(pack: dict[str, Any] | None, fallback: int = 1) -> int:
    if isinstance(pack, dict) and pack.get("enabled"):
        return int(pack.get("sigma_tail_steps") or fallback)
    return int(fallback)


def sampling_fingerprint(plan) -> dict[str, Any]:
    pack = getattr(plan, "sampling", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return {"sampling": False}
    return {
        "sampling": True,
        "sampling_steps": int(pack.get("steps") or 0),
        "sampling_sampler": str(pack.get("sampler") or ""),
        "sampling_scheduler": str(pack.get("scheduler") or ""),
        "sampling_cfg": round(float(pack.get("cfg") or 0), 4),
        "sampling_seed": int(pack.get("seed") or 0),
        "sampling_shift_video": round(float(pack.get("shift_video") or 0), 4),
        "sampling_shift_audio": round(float(pack.get("shift_audio") or 0), 4),
        "sampling_sigma_refine": bool(pack.get("sigma_refine", False)),
        "sampling_sigma_tail_steps": int(pack.get("sigma_tail_steps") or 0),
        "sampling_external_sampler": bool(pack.get("sampler_obj") is not None),
        "sampling_external_sigmas": bool(pack.get("sigmas") is not None),
    }


def sampling_report_line(plan) -> str | None:
    pack = getattr(plan, "sampling", None)
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return None
    parts = [
        f"steps={int(pack.get('steps') or 0)}",
        str(pack.get("sampler") or ""),
        str(pack.get("scheduler") or ""),
        f"cfg={float(pack.get('cfg') or 0):.2f}",
        f"shift={float(pack.get('shift_video') or 0):.1f}/{float(pack.get('shift_audio') or 0):.1f}",
    ]
    if pack.get("sampler_obj") is not None:
        parts.append("external sampler")
    if pack.get("sigmas") is not None:
        parts.append("external sigmas")
    if pack.get("sigma_refine"):
        parts.append(f"sigma refine +{int(pack.get('sigma_tail_steps') or 0)}")
    return "Sampling: " + ", ".join(parts)