"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")


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


def _patcher_is_diffusion(patcher) -> bool:
    """True when a LoadedModel's patcher wraps a diffusion model (DiT / UNET).

    ``comfy.model_base.BaseModel`` is only used for diffusion backbones; text
    encoders (CLIP/Qwen) and VAEs are plain modules, so an isinstance test is
    enough to tell "the big sampler model" apart from everything else.
    """
    if patcher is None:
        return False
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    try:
        from comfy.model_base import BaseModel

        return isinstance(model, BaseModel)
    except Exception:
        return False


def _loaded_gb(models) -> float:
    total = 0
    for cur in models or []:
        try:
            total += int(cur.model.loaded_size())
        except Exception:
            continue
    return total / (1024 ** 3)


def _slot_label(cur) -> tuple[str, float, bool]:
    """(class name, loaded GiB, is_diffusion) for a LoadedModel slot."""
    patcher = getattr(cur, "model", None)
    inner = getattr(patcher, "model", None) if patcher is not None else None
    name = type(inner).__name__ if inner is not None else "?"
    gb = -1.0
    if patcher is not None:
        try:
            gb = int(patcher.loaded_size()) / (1024 ** 3)
        except Exception:
            gb = -1.0
    return name, gb, _patcher_is_diffusion(patcher)


def _log_inventory(tag: str, models) -> None:
    """Log every resident LoadedModel slot — the evidence needed to debug unloads."""
    if not models:
        log.info("MiniMax H3 Director: %s — no model slots resident", tag)
        return
    parts = []
    for cur in models:
        name, gb, is_dit = _slot_label(cur)
        parts.append(f"{name}{'[DiT]' if is_dit else ''}={gb:.2f}GiB")
    log.info("MiniMax H3 Director: %s — resident: %s", tag, ", ".join(parts))


def cleanup_before_decode(*, enabled: bool = True) -> float:
    """Unload only the diffusion model (DiT) before the video VAE decode.

    Why: MiniMax H3 DiT is large enough to fill a 12 GB card on its own. When
    sampling finishes and the DiT stays resident, the video VAE decode has to
    fight it for VRAM, so ComfyUI's dynamic weight loading keeps swapping
    pages in and out and the GPU idles at 15%-35% for minutes. Everything the
    decode needs (the AV latent) is already computed at this point, so the DiT
    can be released before decoding and re-loaded when the next segment samples.

    Unlike :func:`cleanup_segment_vram` this never unloads the VAEs or the text
    encoder: the video/audio VAE stay resident across segments (no reload for
    segment 2+), and the text encoder keeps its patches for a second encode.

    Returns the approximate GiB that were resident before/after the unload,
    as a single "freed" number for logging.
    """
    if not enabled:
        return 0.0
    gc.collect()
    freed = 0.0
    try:
        import comfy.model_management as mm

        models = list(getattr(mm, "current_loaded_models", None) or [])
        device = mm.get_torch_device()
        keep_loaded = []
        dit_slots = 0
        for cur in models:
            if _patcher_is_diffusion(getattr(cur, "model", None)):
                dit_slots += 1
                continue
            keep_loaded.append(cur)
        _log_inventory("pre-decode", models)
        if dit_slots == 0:
            log.info(
                "MiniMax H3 Director: pre-decode cleanup — no DiT resident, nothing to unload"
            )
            return 0.0
        before = _loaded_gb(models)
        try:
            before_free = mm.get_free_memory(device) / (1024 ** 3)
        except Exception:
            before_free = -1.0
        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        # Same machinery as unload_all_models(), but every non-DiT slot is
        # pinned through keep_loaded so the VAEs / text encoder survive.
        mm.free_memory(1e30, device, keep_loaded)
        _evict_dead_loaded_models()
        remaining = list(getattr(mm, "current_loaded_models", None) or [])
        _log_inventory("post-decode-cleanup", remaining)
        after = _loaded_gb(remaining)
        try:
            after_free = mm.get_free_memory(device) / (1024 ** 3)
        except Exception:
            after_free = -1.0
        # Dynamic VRAM reports loaded_size≈0 (weights are paged on demand), so
        # the free-VRAM delta is the number that actually explains the win.
        if after_free >= 0 and before_free >= 0:
            freed = max(0.0, after_free - before_free)
        else:
            freed = max(0.0, before - after)
        gc.collect()
        try:
            import comfy.model_prefetch

            comfy.model_prefetch.cleanup_prefetch_queues()
        except Exception as exc:
            log.debug("cleanup_prefetch_queues skipped: %s", exc)
        mm.soft_empty_cache()
        log.info(
            "MiniMax H3 Director: pre-decode cleanup unloaded %d DiT slot(s), kept %d model(s) resident, VRAM free %.2f → %.2f GiB (+%.2f)",
            dit_slots,
            len(keep_loaded),
            before_free,
            after_free,
            freed,
        )
    except Exception as exc:
        log.warning("Pre-decode VRAM cleanup failed: %s", exc)
        return 0.0
    return freed


def cleanup_segment_vram(*, enabled: bool = True, unload_models: bool = True) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache.

    Also drops ComfyUI prefetch queues. Those are normally cleared once, in the
    per-node finally, but this node loops every segment inside one execution.
    Leaving the queues pinned makes later segments fault back to host-to-device
    copies. Model unload policy is unchanged.
    """
    if not enabled:
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
        _evict_dead_loaded_models()
        gc.collect()
        try:
            import comfy.model_prefetch

            comfy.model_prefetch.cleanup_prefetch_queues()
        except Exception as exc:
            log.debug("cleanup_prefetch_queues skipped: %s", exc)
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")
