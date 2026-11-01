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
from pathlib import Path
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


def resolve_pika_task_loras(rows, task_key: str) -> list[dict[str, Any]]:
    """Keep Pika's task adapter paired with the conditioning path."""
    rows = normalize_lora_rows(rows)
    variant = "ref2va" if task_key in {"r2v", "v2v", "rv2v"} else "fl2va"
    for row in rows:
        for source in ("fl2va", "ref2va"):
            row["name"] = row["name"].replace(
                f"pika_h3_flex_{source}_", f"pika_h3_flex_{variant}_"
            )
    pika_rows = [row for row in rows if "pika_h3_flex_" in row["name"]]
    if pika_rows:
        adapter = pika_rows[0]
        rows = [row for row in rows if row not in pika_rows]
    else:
        adapter = {"name": f"pika_h3_flex_{variant}_v1_bf16.safetensors"}
    rows.insert(0, {**adapter, "strength": 1.0, "active": True})
    if len(rows) > MAX_SEGMENT_LORAS:
        raise ValueError(f"Pika's task adapter occupies one LoRA slot; keep at most {MAX_SEGMENT_LORAS - 1} additional LoRAs.")
    return rows


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
    loader_cache: dict | None = None,
    model_cache: dict | None = None,
):
    """Return ``model`` with this segment's LoRAs patched in.

    ``loader_cache`` keeps native LoRA loaders for the duration of one
    Director run so a LoRA reused by several segments is read from disk once.
    The original model is never mutated — each patch clones it.
    """
    eff = effective_lora_rows(rows)
    if not eff or model is None:
        return model

    import folder_paths
    from nodes import LoraLoaderModelOnly

    paths = [folder_paths.get_full_path_or_raise("loras", row["name"]) for row in eff]
    stamps = [Path(path).stat() for path in paths]
    key = (id(model), tuple((path, stamp.st_mtime_ns, stamp.st_size, float(row["strength"])) for path, row, stamp in zip(paths, eff, stamps)))
    if model_cache is not None and key in model_cache:
        return model_cache[key]
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

        stamp = Path(path).stat()
        loader_key = (path, stamp.st_mtime_ns, stamp.st_size)
        loader = loader_cache.get(loader_key) if loader_cache is not None else None
        if loader is None:
            loader = LoraLoaderModelOnly()
            if loader_cache is not None:
                for stale in tuple(loader_cache):
                    if stale[0] == path:
                        loader_cache.pop(stale)
                loader_cache[loader_key] = loader
        patched, = loader.load_lora_model_only(
            patched, name, float(row["strength"])
        )
        log.debug("Segment LoRA applied: %s @ %.2f", name, float(row["strength"]))
    if model_cache is not None:
        if len(model_cache) >= 4:
            model_cache.pop(next(iter(model_cache)))
        model_cache[key] = patched
    return patched
