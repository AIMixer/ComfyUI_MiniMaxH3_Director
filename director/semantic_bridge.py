"""MiniMax H3 Semantic Bridge (first-party, Apache-2.0).

Original student math from speach1sdef178/MiniMax-H3-Semantic-Bridge:

    x  = RMS_norm(H)
    S  = student(x)          # 5120 → 512 SiLU → 512 SiLU → 5120
    S' = magnitude_match(S, H)
    C  = H + alpha * (S' - H)

Only ``CONDITIONING[0][0]`` (the [B, T, 5120] token tensor) is rewritten.
Keyframe / ref metadata is left untouched.

Weights stay user-provided under ``ComfyUI/models/semantic_bridge/``.
The published adapter was distilled on FL2VA. Ref2VA tasks (r2v / v2v / rv2v)
are forced-compat: same full-tensor math, with a report warning. This is not
T8 ``text_only``. Unconnected Director is a no-op.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.semantic_bridge")

MMX_DIR_SEMANTIC_BRIDGE = "MMX_DIR_SEMANTIC_BRIDGE"

SEMANTIC_BRIDGE_FOLDER = "semantic_bridge"
MISSING_ADAPTER_LABEL = "(将 Semantic Bridge 权重放入 models/semantic_bridge)"

HIDDEN_DIM = 5120
STUDENT_DIM = 512
DEFAULT_ALPHA = 0.15
META_APPLIED_KEY = "mmx_semantic_bridge"

REF2VA_TASKS = frozenset({"r2v", "v2v", "rv2v"})

_MODEL_CACHE: dict[str, nn.Module] = {}


class SemanticStudent(nn.Module):
    """5120→512 SiLU → 512 SiLU → 5120. Six tensors: fc1/2/3 weight+bias."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN_DIM, STUDENT_DIM)
        self.fc2 = nn.Linear(STUDENT_DIM, STUDENT_DIM)
        self.fc3 = nn.Linear(STUDENT_DIM, HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.silu(self.fc2(F.silu(self.fc1(x)))))


def _clamp_float(raw, default: float, lo: float, hi: float) -> float:
    try:
        n = float(raw if raw is not None else default)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def is_ref2va_task(task_key: str) -> bool:
    return str(task_key or "").strip().lower() in REF2VA_TASKS


def ensure_semantic_bridge_folder() -> str | None:
    try:
        import folder_paths

        if SEMANTIC_BRIDGE_FOLDER not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                SEMANTIC_BRIDGE_FOLDER,
                os.path.join(folder_paths.models_dir, SEMANTIC_BRIDGE_FOLDER),
            )
        paths = folder_paths.get_folder_paths(SEMANTIC_BRIDGE_FOLDER)
        return paths[0] if paths else None
    except Exception:
        return None


def list_semantic_bridge_adapters() -> list[str]:
    root = ensure_semantic_bridge_folder()
    names: list[str] = []
    if root:
        try:
            os.makedirs(root, exist_ok=True)
        except OSError:
            pass
        for ext in ("*.safetensors", "*.pt", "*.pth"):
            names.extend(os.path.basename(p) for p in glob.glob(os.path.join(root, ext)))
        try:
            import folder_paths

            names.extend(folder_paths.get_filename_list(SEMANTIC_BRIDGE_FOLDER) or [])
        except Exception:
            pass
    out = sorted({n for n in names if n and not n.startswith("(")})
    return out or [MISSING_ADAPTER_LABEL]


def resolve_semantic_bridge_path(filename: str) -> str | None:
    name = str(filename or "").strip()
    if not name or name.startswith("("):
        return None
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    try:
        import folder_paths

        ensure_semantic_bridge_folder()
        path = folder_paths.get_full_path(SEMANTIC_BRIDGE_FOLDER, name)
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    root = ensure_semantic_bridge_folder()
    if root:
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        return blob["state_dict"]
    if isinstance(blob, dict):
        return blob
    raise ValueError(f"Semantic Bridge adapter is not a state dict: {path}")


def load_semantic_student(path: str) -> SemanticStudent:
    cached = _MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    student = SemanticStudent()
    student.load_state_dict(_load_state_dict(path), strict=True)
    student.eval()
    _MODEL_CACHE[path] = student
    return student


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _magnitude_match(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s_norm = student.norm(dim=-1, keepdim=True)
    t_norm = teacher.norm(dim=-1, keepdim=True)
    return student * (t_norm / (s_norm + eps))


def pack_semantic_bridge(
    *,
    adapter: str,
    alpha: float = DEFAULT_ALPHA,
    magnitude_match: bool = True,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "adapter": str(adapter or "").strip(),
        "alpha": _clamp_float(alpha, DEFAULT_ALPHA, 0.0, 1.0),
        "magnitude_match": bool(magnitude_match),
    }


def normalize_semantic_bridge_pack(raw) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("enabled") is False:
        return None
    adapter = str(raw.get("adapter") or "").strip()
    if not adapter or adapter.startswith("("):
        return {
            "enabled": True,
            "adapter": "",
            "alpha": _clamp_float(raw.get("alpha"), DEFAULT_ALPHA, 0.0, 1.0),
            "magnitude_match": bool(raw.get("magnitude_match", True)),
        }
    return pack_semantic_bridge(
        adapter=adapter,
        alpha=raw.get("alpha", DEFAULT_ALPHA),
        magnitude_match=raw.get("magnitude_match", True),
    )


def semantic_bridge_enabled(plan) -> bool:
    pack = getattr(plan, "semantic_bridge", None)
    return isinstance(pack, dict) and bool(pack.get("enabled", True))


def semantic_bridge_fingerprint(plan) -> dict[str, Any]:
    """Empty when unconnected so existing first-pass caches still match."""
    if not semantic_bridge_enabled(plan):
        return {}
    pack = plan.semantic_bridge
    return {
        "semantic_bridge": True,
        "sb_adapter": str(pack.get("adapter") or ""),
        "sb_alpha": round(float(pack.get("alpha") or 0.0), 4),
        "sb_mag": bool(pack.get("magnitude_match", True)),
    }


def semantic_bridge_report_line(plan) -> str | None:
    if not semantic_bridge_enabled(plan):
        return None
    pack = plan.semantic_bridge
    adapter = pack.get("adapter") or "(missing)"
    alpha = float(pack.get("alpha") or 0.0)
    mag = "on" if pack.get("magnitude_match", True) else "off"
    line = f"Semantic Bridge: ON (adapter={adapter}, alpha={alpha:.3f}, magnitude_match={mag})"
    if is_ref2va_task(getattr(plan, "global_task_key", "")):
        line += " — Ref2VA 强制兼容，作者未蒸馏该任务，请谨慎使用"
    return line


def _already_applied(meta: dict | None) -> bool:
    if not isinstance(meta, dict):
        return False
    return bool(meta.get(META_APPLIED_KEY))


def _rewrite_hidden(hidden: torch.Tensor, student: SemanticStudent, alpha: float, match_mag: bool) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[-1] != HIDDEN_DIM:
        raise ValueError(
            f"Semantic Bridge expects [B, T, {HIDDEN_DIM}], got {tuple(hidden.shape)}"
        )
    device = hidden.device
    dtype = hidden.dtype
    student = student.to(device=device, dtype=torch.float32)
    x = hidden.float()
    with torch.no_grad():
        s = student(_rms_norm(x))
        if match_mag:
            s = _magnitude_match(s, x)
        out = x + float(alpha) * (s - x)
    return out.to(device=device, dtype=dtype)


def apply_semantic_bridge(positive, plan, *, task_key: str = ""):
    """Rewrite cond item[0] in place-copy. No-op when unconnected.

    Returns ``(positive, note)``. ``note`` is None when nothing changed.
    """
    if not semantic_bridge_enabled(plan):
        return positive, None
    pack = plan.semantic_bridge
    adapter = str(pack.get("adapter") or "").strip()
    path = resolve_semantic_bridge_path(adapter)
    if not path:
        note = f"Semantic Bridge skipped: adapter not found ({adapter or 'empty'})"
        log.warning("MiniMax H3 Director: %s", note)
        return positive, note

    if not isinstance(positive, list) or not positive:
        note = "Semantic Bridge skipped: empty conditioning"
        log.warning("MiniMax H3 Director: %s", note)
        return positive, note

    item0 = positive[0]
    if not isinstance(item0, (list, tuple)) or not item0:
        note = "Semantic Bridge skipped: unexpected conditioning layout"
        log.warning("MiniMax H3 Director: %s", note)
        return positive, note

    hidden, meta = item0[0], (item0[1] if len(item0) > 1 else None)
    if _already_applied(meta if isinstance(meta, dict) else None):
        return positive, None
    if not isinstance(hidden, torch.Tensor):
        note = "Semantic Bridge skipped: cond[0][0] is not a tensor"
        log.warning("MiniMax H3 Director: %s", note)
        return positive, note

    try:
        student = load_semantic_student(path)
        rewritten = _rewrite_hidden(
            hidden,
            student,
            float(pack.get("alpha") or DEFAULT_ALPHA),
            bool(pack.get("magnitude_match", True)),
        )
    except Exception as exc:
        note = f"Semantic Bridge skipped: {exc}"
        log.warning("MiniMax H3 Director: %s", note)
        return positive, note

    new_meta = dict(meta) if isinstance(meta, dict) else {}
    new_meta[META_APPLIED_KEY] = True
    new_item = [rewritten, new_meta]
    if len(item0) > 2:
        new_item.extend(item0[2:])
    out = list(positive)
    out[0] = new_item

    task = str(task_key or getattr(plan, "global_task_key", "") or "")
    note = (
        f"Semantic Bridge applied (alpha={float(pack.get('alpha') or 0):.3f}, "
        f"tokens={int(rewritten.shape[1])})"
    )
    if is_ref2va_task(task):
        note += " — Ref2VA 强制兼容，请谨慎使用"
    return out, note
