"""Per-segment LoRA stacks for MiniMax H3 Director.

Every timeline segment can carry its own list of LoRAs. They are applied to a
clone of the segment's base MODEL right before sampling, so one Director run
can change character / motion LoRAs shot by shot while the inter-segment guide
(continuity_from_prev) keeps working across the whole timeline.

Nothing here loads a model: patches are ``ModelPatcher.clone() + add_patches``,
exactly what the stock LoraLoader node does.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.segment_loras")

MAX_SEGMENT_LORAS = 8
MIN_LORA_STRENGTH = -10.0
MAX_LORA_STRENGTH = 10.0
# Below this a LoRA is a no-op; skip the clone entirely.
LORA_EPSILON = 1e-6


def _coerce_strength(value: Any, default: float = 1.0) -> float:
    try:
        s = float(value)
    except (TypeError, ValueError):
        return default
    if s != s:  # NaN
        return default
    return max(MIN_LORA_STRENGTH, min(MAX_LORA_STRENGTH, s))


def normalize_lora_rows(raw: Any) -> list[dict[str, Any]]:
    """Accept the UI's segment.loras payload and return clean rows.

    Rows look like ``{"name": "sub/dir/file.safetensors", "strength": 1.0,
    "active": True}``. Unknown keys are dropped, blank names are skipped, and
    the list is capped at MAX_SEGMENT_LORAS so a malformed timeline cannot
    stack an unbounded number of patches.
    """
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("lora") or item.get("lora_name") or "").strip()
        if not name:
            continue
        active = item.get("active")
        if active is None:
            active = item.get("enabled")
        out.append(
            {
                "name": name,
                "strength": _coerce_strength(
                    item.get("strength", item.get("strength_model", 1.0))
                ),
                "active": True if active is None else bool(active),
            }
        )
        if len(out) >= MAX_SEGMENT_LORAS:
            break
    return out


def effective_lora_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Rows that will actually change the model (active, non-zero strength)."""
    return [
        r
        for r in (rows or [])
        if r.get("active", True) and abs(float(r.get("strength", 0.0))) > LORA_EPSILON
    ]


def lora_signature(rows: list[dict[str, Any]] | None) -> tuple:
    """Cache key: identical stacks reuse one patched MODEL inside a run."""
    return tuple(
        (r["name"], round(float(r["strength"]), 6)) for r in effective_lora_rows(rows)
    )


def describe_lora_rows(rows: list[dict[str, Any]] | None) -> str:
    """One-line summary for the Director report."""
    eff = effective_lora_rows(rows)
    if not eff:
        return ""
    return ", ".join("%s(%.2f)" % (r["name"], float(r["strength"])) for r in eff)


def apply_segment_loras(
    model,
    rows: list[dict[str, Any]] | None,
    *,
    state_dict_cache: dict | None = None,
):
    """Return ``model`` with this segment's LoRAs patched in.

    ``state_dict_cache`` keeps the loaded tensors for the duration of one
    Director run so a LoRA reused by several segments is read from disk once.
    The original model is never mutated — each patch clones it.
    """
    eff = effective_lora_rows(rows)
    if not eff or model is None:
        return model

    import comfy.sd
    import comfy.utils
    import folder_paths

    patched = model
    for row in eff:
        name = row["name"]
        try:
            path = folder_paths.get_full_path_or_raise("loras", name)
        except Exception as exc:
            raise ValueError(
                "MiniMax H3 Director: segment LoRA '%s' not found under models/loras (%s)."
                % (name, exc)
            ) from exc

        state_dict = state_dict_cache.get(path) if state_dict_cache is not None else None
        if state_dict is None:
            state_dict = comfy.utils.load_torch_file(path, safe_load=True)
            if state_dict_cache is not None:
                state_dict_cache[path] = state_dict

        patched, _ = comfy.sd.load_lora_for_models(
            patched, None, state_dict, float(row["strength"]), 0.0
        )
        log.debug("Segment LoRA applied: %s @ %.2f", name, float(row["strength"]))
    return patched
