"""Release GPU & system memory between MiniMax H3 Director segment runs.

This module mirrors the "pressure-aware cleanup" from the PRO/Plus studio node so
that the Director (non-Plus) also survives long runs (≥20 segments) on machines
with limited RAM / pagefile. The single most important change is a RAM pressure
probe (via psutil) before each cleanup call: when free system memory drops below
a safety threshold or used-RAM % crosses 88%, the cleanup is upgraded to a
*deep* pass (unload ALL staged models) so 49GB TE staged memory can actually be
returned to the OS.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")

# ---------------------------------------------------------------------------
# Memory snapshot (mirrors Plus studio_node._memory_snapshot)
# ---------------------------------------------------------------------------

def memory_snapshot() -> dict[str, Any]:
    """Read current RAM / VRAM status; returns dict with pressure flag.

    pressure=True triggers deep release to prevent staged model memory from
    being starved out of system pagefile. Mirrors Plus thresholds:
        reserve  = min(8GB, max(3GB, 18% of total RAM))
        pressure = available < reserve  OR  percent >= 88%
    """
    snap: dict[str, Any] = {
        "ram_total": None,
        "ram_available": None,
        "ram_percent": None,
        "ram_reserve": None,
        "vram_free": None,
        "pressure": False,
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total = int(vm.total)
        available = int(vm.available)
        reserve = int(min(8 * 1024 ** 3, max(3 * 1024 ** 3, total * 0.18)))
        snap.update({
            "ram_total": total,
            "ram_available": available,
            "ram_percent": float(vm.percent),
            "ram_reserve": reserve,
            "pressure": available < reserve or float(vm.percent) >= 88.0,
        })
    except Exception:
        pass
    try:
        import comfy.model_management as mm  # type: ignore

        snap["vram_free"] = int(mm.get_free_memory())
    except Exception:
        pass
    return snap


def _format_snap(snap: dict[str, Any]) -> str:
    parts: list[str] = []
    if snap.get("ram_total") is not None:
        gb = lambda v: v / 1024 ** 3  # noqa: E731
        parts.append("RAM %.1f/%.1fGB (%.0f%%, avail %.1fGB)" % (
            gb(snap["ram_total"] - snap["ram_available"]),
            gb(snap["ram_total"]),
            snap["ram_percent"],
            gb(snap["ram_available"]),
        ))
    if snap.get("vram_free") is not None:
        parts.append("VRAM free %.1fGB" % (snap["vram_free"] / 1024 ** 3))
    if snap.get("pressure"):
        parts.append("PRESSURE=ON")
    return " | ".join(parts) if parts else "memory state unavailable"


# ---------------------------------------------------------------------------
# Loaded-model counter (for logs only)
# ---------------------------------------------------------------------------

def _loaded_model_count() -> int | None:
    try:
        import comfy.model_management as mm  # type: ignore

        loaded = mm.loaded_models()
        return len(loaded) if loaded is not None else None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Main cleanup — pressure-aware.  Keep the legacy signature so callers inside
# executor_core (which pass `unload_models=` style semantics) still work.
# ---------------------------------------------------------------------------

def cleanup_segment_vram(
    *,
    enabled: bool = True,
    unload_models: bool = True,
    force_deep: bool = False,
    segment_index: int | None = None,
) -> None:
    """Release segment GPU + system memory.

    Parameters
    ----------
    enabled:
        Master kill switch (False = no-op).
    unload_models:
        Legacy flag — when True (the default), at least a model-keep-loaded
        (cache-only) pass is performed; actual unload is decided by pressure.
    force_deep:
        Override pressure probe and always do the deepest cleanup.  Used by
        deferred_merge just before the ffmpeg-based final pass.
    segment_index:
        Optional, used only for log lines so operators can tell which
        segment boundary caused a pressure-elevated cleanup.
    """
    if not enabled:
        return

    snap = memory_snapshot()
    deep = force_deep or bool(snap.get("pressure"))
    before = _loaded_model_count()

    # Pass 1: light GC (reference cycle collection) before touching mm APIs.
    # Two generations + return-to-system to shake out large mmap'd staged areas.
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    # Aggressively trim python process working set (Windows / Linux) so staged
    # weight mmaps can be physically evicted.  Without this, 49GB TE staged
    # memory accumulates across segments and crashes the next fopen/fread on
    # pagefile-backed weight files (the exact crash the user saw on seg #14).
    try:
        if os.name == "nt":
            import ctypes  # type: ignore

            # SetProcessWorkingSetSize with (-1,-1) tells the kernel to trim
            # all possible pages back to the working-set minimum.
            kernel32 = ctypes.windll.kernel32
            HANDLE = ctypes.c_void_p
            SIZE_T = ctypes.c_size_t
            kernel32.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
            kernel32.SetProcessWorkingSetSizeEx.argtypes = [HANDLE, SIZE_T, SIZE_T, ctypes.c_ulong]
            hProc = ctypes.c_void_p(-1)  # GetCurrentProcess() pseudo-handle
            kernel32.SetProcessWorkingSetSizeEx(hProc, SIZE_T(-1), SIZE_T(-1), 0)
    except Exception:
        # Never crash the whole run just because trim failed.
        pass

    try:
        import comfy.model_management as mm  # type: ignore

        mm.cleanup_models_gc()
        if deep and unload_models:
            try:
                mm.unload_all_models()
            except Exception as exc:
                log.warning("Segment VRAM deep unload failed (%s); continuing with cache clear.", exc)
        mm.cleanup_models()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return

    # Pass 2: final GC round after mm has released all references.
    gc.collect(2)
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSizeEx(
                ctypes.c_void_p(-1), ctypes.c_size_t(-1), ctypes.c_size_t(-1), 0)
    except Exception:
        pass

    after = _loaded_model_count()
    seg_tag = f" (seg {segment_index})" if segment_index is not None else ""
    if deep:
        count_note = ""
        if before is not None and after is not None:
            count_note = f", models {before}->{after}"
        log.info(
            "Segment VRAM cleanup: DEEP release%s%s; %s",
            seg_tag, count_note, _format_snap(snap),
        )
    else:
        log.debug(
            "Segment VRAM cleanup: light pass%s; %s",
            seg_tag, _format_snap(snap),
        )
