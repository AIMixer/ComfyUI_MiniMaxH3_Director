"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")


def release_director_plan_memory(plan) -> None:
    """Drop execution-only source/reference tensors after Director returns."""
    if plan is None:
        return
    cache = getattr(plan, "audio_decode_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    for item in getattr(plan, "global_ref_audios", None) or []:
        if hasattr(item, "audio"):
            item.audio = None
    for item in getattr(plan, "global_refs", None) or []:
        if hasattr(item, "tensor"):
            item.tensor = None
    for seg in getattr(plan, "segments", None) or []:
        seg.source_clip = None
        for item in getattr(seg, "refs", None) or []:
            if hasattr(item, "tensor"):
                item.tensor = None
        for item in list(getattr(seg, "ref_videos", None) or []) + list(getattr(seg, "ref_video_audios", None) or []):
            if hasattr(item, "tensor"):
                item.tensor = None
            if hasattr(item, "audio"):
                item.audio = None
        for item in getattr(seg, "ref_audios", None) or []:
            if hasattr(item, "audio"):
                item.audio = None
    if hasattr(plan, "source_video"):
        plan.source_video = None
    raw = getattr(plan, "raw", None)
    if isinstance(raw, dict):
        raw.clear()
    for name in ("refine", "face_refine", "selflift"):
        value = getattr(plan, name, None)
        if isinstance(value, dict):
            value.clear()
    if isinstance(getattr(plan, "segments", None), list):
        plan.segments.clear()
    if isinstance(getattr(plan, "global_refs", None), list):
        plan.global_refs.clear()
    if isinstance(getattr(plan, "global_ref_audios", None), list):
        plan.global_ref_audios.clear()


def _evict_dead_loaded_models() -> int:
    """Pop Comfy LoadedModel slots that ``free_memory`` will skip forever.

    ``is_dead()`` means the ModelPatcher weakref is gone while the shared
    MiniMaxH3 module is still alive (graph MODEL / Sage cycle). Those slots
    log ``Potential memory leak detected with model MiniMaxH3`` and then sit
    in ``current_loaded_models``, so later unloads cannot touch them.
    Evicting the slot does not copy weights; it restores unload bookkeeping.
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    models = getattr(mm, "current_loaded_models", None)
    if not models:
        return 0
    evicted = 0
    for i in range(len(models) - 1, -1, -1):
        cur = models[i]
        try:
            if not cur.is_dead():
                continue
            name = "?"
            try:
                real = cur.real_model()
                name = type(real).__name__ if real is not None else "?"
            except Exception:
                pass
            models.pop(i)
            evicted += 1
            log.info("MiniMax H3 Director: evicted dead LoadedModel slot (%s)", name)
        except Exception:
            continue
    return evicted


def cleanup_segment_vram(*, enabled: bool = True, unload_models: bool = True) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache."""
    if not enabled:
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        reset_cast_buffers = getattr(mm, "reset_cast_buffers", None)
        if callable(reset_cast_buffers):
            reset_cast_buffers()
        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
            if callable(reset_cast_buffers):
                reset_cast_buffers()
        _evict_dead_loaded_models()
        gc.collect()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")
