"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging
import threading
import time

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")

# Auto memory guard thresholds (fraction of physical RAM still available).
# Above LOW: do nothing. LOW..CRITICAL: drop caches only (models stay resident).
# Below CRITICAL: unload models. Keeping weights resident avoids pagefile
# churn, because model weights are clean file-backed pages while frame
# tensors are anonymous dirty pages that must be written out.
MEMORY_GUARD_LOW_FRACTION = 0.25
MEMORY_GUARD_CRITICAL_FRACTION = 0.10

# Safety margin on the measured per-phase RAM consumption when predicting the
# next phase's need. 1.15 = last measured Δ + 15%.
MEMORY_GUARD_PREDICT_MARGIN = 1.15

# In-phase hard floor (fraction of physical RAM still available). Below this the
# background sampler does an emergency ``gc.collect()`` only — it never unloads
# models, because unloading mid-phase breaks the running sample. All heavy
# cleanup stays at phase boundaries via ``auto_memory_guard``.
MEMORY_GUARD_HARD_FLOOR_FRACTION = 0.05

# Background sampler cadence / guard-rails.
_MONITOR_INTERVAL_S = 1.0
_MONITOR_ACTION_COOLDOWN_S = 5.0
_MONITOR_IDLE_TIMEOUT_S = 1800.0


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

        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
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


def ram_status() -> dict | None:
    """Physical RAM status as ``{"available": bytes, "total": bytes}``, or None.

    ``psutil`` first; on Windows fall back to ``GlobalMemoryStatusEx`` so the
    guard still works when psutil is not installed. Never raises.
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        total = int(vm.total)
        if total > 0:
            return {"available": int(vm.available), "total": total}
    except Exception:
        pass
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        total = int(stat.ullTotalPhys)
        if total <= 0:
            return None
        return {"available": int(stat.ullAvailPhys), "total": total}
    except Exception:
        return None


def pagefile_status() -> dict | None:
    """Pagefile usage as ``{"used": bytes, "total": bytes}``, or None.

    On Windows ``psutil.swap_memory()`` reports the pagefile, and rising usage
    is the first real sign that dirty pages are being written out. That is the
    disk-wear signal worth watching: model weights are clean file-backed pages
    (Windows can drop them without writing), while frame tensors are anonymous
    dirty pages that must go to the pagefile and be read back later.
    """
    try:
        import psutil

        sw = psutil.swap_memory()
        used = int(getattr(sw, "used", 0) or 0)
        total = int(getattr(sw, "total", 0) or 0)
        if used > 0 or total > 0:
            return {"used": used, "total": total}
    except Exception:
        pass
    return None


def auto_memory_guard(
    *,
    enabled: bool = True,
    label: str = "",
    low_fraction: float = MEMORY_GUARD_LOW_FRACTION,
    critical_fraction: float = MEMORY_GUARD_CRITICAL_FRACTION,
    predicted_peak_bytes: int = 0,
) -> str | None:
    """Water-level guard run once before each segment starts.

    ``predicted_peak_bytes`` is the measured footprint of the previous phase
    (see :class:`MemoryPeakMonitor`). When given, the tier is chosen on the
    *projected* level ``available - predicted_peak_bytes`` instead of today's
    level, so cleanup happens while there is still room to do it cheaply rather
    than after the phase has already started swapping. With the default 0 the
    behaviour is exactly the original percentage-only guard.

    Always logs the real numbers so the true water level can be observed.
    Returns a Chinese one-line report only when an action was taken, so callers
    can append it to ``reports``; returns None when nothing was done.
    """
    if not enabled:
        return None
    tag = f"{label} " if label else ""
    # A critical line above the low line would silently disable the middle tier;
    # clamp so the three levels always stay ordered.
    critical_fraction = min(float(critical_fraction), float(low_fraction))
    try:
        status = ram_status()
    except Exception:
        status = None
    if not status or not status.get("total"):
        log.info(
            "MiniMax H3 Director: %smemory guard — RAM status unavailable, no action",
            tag,
        )
        return None

    total = float(status["total"])
    available = float(status["available"])
    predicted = max(0.0, float(predicted_peak_bytes or 0))
    # Tier decisions use the projected level: what will be left once the next
    # phase has taken its measured share.
    projected = available - predicted
    fraction = projected / total if total > 0 else 0.0
    avail_gb = available / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    pred_gb = predicted / (1024 ** 3)
    proj_gb = projected / (1024 ** 3)
    pct = fraction * 100.0
    pred_note = (
        f", next phase predicted {pred_gb:.1f} GB → {proj_gb:.1f} GB left"
        if predicted > 0.0
        else ""
    )
    pred_note_zh = (
        f"，预测下一阶段需 {pred_gb:.1f} GB、余 {proj_gb:.1f} GB"
        if predicted > 0.0
        else ""
    )

    if fraction >= low_fraction:
        log.info(
            "MiniMax H3 Director: %smemory guard — RAM ok (%.1f/%.1f GB, %.0f%% projected free%s), no action",
            tag, avail_gb, total_gb, pct, pred_note,
        )
        return None

    if fraction >= critical_fraction:
        log.info(
            "MiniMax H3 Director: %smemory guard — RAM low (%.1f/%.1f GB, %.0f%% projected free%s), "
            "releasing caches (models kept loaded)",
            tag, avail_gb, total_gb, pct, pred_note,
        )
        try:
            gc.collect()
            import comfy.model_management as mm

            mm.cleanup_models_gc()
            _evict_dead_loaded_models()
            mm.soft_empty_cache()
        except Exception as exc:
            log.warning("MiniMax H3 Director: memory guard cache release failed: %s", exc)
        return (
            f"{tag}内存偏低（可用 {avail_gb:.1f}/{total_gb:.1f} GB{pred_note_zh}），"
            f"已释放缓存、保留模型"
        )

    log.warning(
        "MiniMax H3 Director: %smemory guard — RAM critical (%.1f/%.1f GB, %.0f%% projected free%s), "
        "unloading models",
        tag, avail_gb, total_gb, pct, pred_note,
    )
    try:
        cleanup_segment_vram(enabled=True, unload_models=True)
    except Exception as exc:
        log.warning("MiniMax H3 Director: memory guard model unload failed: %s", exc)
    return (
        f"{tag}内存告急（可用 {avail_gb:.1f}/{total_gb:.1f} GB{pred_note_zh}），"
        f"已卸载模型并清空缓存"
    )


class MemoryPeakMonitor:
    """Background RAM sampler that measures each phase's real footprint.

    A static table cannot predict the footprint (it varies 2–5× with resolution,
    frame count, refine/FaceRefine/SelfLift and continuity); the *previous*
    measured window can. Same-shaped segments peak within ~10% of each other, so
    the guard runs on ``last window Δ × margin`` instead of a hand-tuned percent.

    The sampler never unloads models. Unloading mid-phase would abort the
    running sample, and it would also be counter-productive for disk wear:
    weights are clean file-backed pages Windows can discard for free, whereas
    evicting them forces a re-read of the 113 GB checkpoint. Only a last-resort
    ``gc.collect()`` fires here when RAM drops under the hard floor.
    """

    def __init__(self, *, interval_s: float = _MONITOR_INTERVAL_S) -> None:
        self._interval_s = max(0.2, float(interval_s))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_seen = time.monotonic()
        self._last_action = 0.0
        # Current window = one phase/segment.
        self._win_baseline: int | None = None
        self._win_min: int | None = None
        self._win_peak_pagefile = 0
        # Whole run.
        self._run_baseline: int | None = None
        self._run_min: int | None = None
        self._run_peak_pagefile = 0
        self._hard_floor_hits = 0
        self._samples = 0

    # ---- lifecycle ------------------------------------------------------
    def _reset_counters(self) -> None:
        """Fresh trackers for a new run (the singleton is reused across runs)."""
        self._win_baseline = None
        self._win_min = None
        self._win_peak_pagefile = 0
        self._run_baseline = None
        self._run_min = None
        self._run_peak_pagefile = 0
        self._hard_floor_hits = 0
        self._samples = 0
        self._last_action = 0.0

    def start(self) -> None:
        with self._lock:
            self._last_seen = time.monotonic()
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._reset_counters()
            thread = threading.Thread(
                target=self._run, name="h3-mem-monitor", daemon=True
            )
            self._thread = thread
        thread.start()
        log.info(
            "MiniMax H3 Director: memory monitor started (sampling every %.1fs, "
            "hard floor %.0f%% free → gc only, never unload mid-phase)",
            self._interval_s, MEMORY_GUARD_HARD_FLOOR_FRACTION * 100.0,
        )

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass
        with self._lock:
            self._thread = None

    # ---- sampling -------------------------------------------------------
    def _sample_once(self) -> None:
        status = ram_status()
        page = pagefile_status()
        now = time.monotonic()
        under_floor = False
        with self._lock:
            self._samples += 1
            if status and status.get("total"):
                available = int(status["available"])
                total = float(status["total"])
                if self._win_baseline is None:
                    self._win_baseline = available
                if self._run_baseline is None:
                    self._run_baseline = available
                if self._win_min is None or available < self._win_min:
                    self._win_min = available
                if self._run_min is None or available < self._run_min:
                    self._run_min = available
                if (
                    total > 0
                    and available / total < MEMORY_GUARD_HARD_FLOOR_FRACTION
                    and now - self._last_action >= _MONITOR_ACTION_COOLDOWN_S
                ):
                    self._last_action = now
                    self._hard_floor_hits += 1
                    under_floor = True
            if page:
                used = int(page.get("used") or 0)
                if used > self._win_peak_pagefile:
                    self._win_peak_pagefile = used
                if used > self._run_peak_pagefile:
                    self._run_peak_pagefile = used
        if under_floor:
            log.warning(
                "MiniMax H3 Director: memory monitor — RAM under hard floor "
                "(<%.0f%% free), emergency gc.collect() (models kept loaded; "
                "unloading mid-phase would abort the sample)",
                MEMORY_GUARD_HARD_FLOOR_FRACTION * 100.0,
            )
            try:
                gc.collect()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._sample_once()
            except Exception:
                continue
            with self._lock:
                idle = time.monotonic() - self._last_seen
            if idle > _MONITOR_IDLE_TIMEOUT_S:
                # The run died without calling stop(); do not leak the thread.
                log.info(
                    "MiniMax H3 Director: memory monitor idle > %.0fs, exiting",
                    _MONITOR_IDLE_TIMEOUT_S,
                )
                return

    # ---- read-outs ------------------------------------------------------
    def snapshot(self) -> dict:
        """Measured stats of the current window (i.e. the previous phase)."""
        with self._lock:
            self._last_seen = time.monotonic()
            base, mn = self._win_baseline, self._win_min
            delta = int(base - mn) if (base is not None and mn is not None and base > mn) else 0
            return {
                "delta": delta,
                "baseline": base,
                "min_available": mn,
                "peak_pagefile": int(self._win_peak_pagefile),
                "samples": int(self._samples),
            }

    def reset_window(self) -> None:
        """Start a fresh window; call *after* the boundary cleanup ran.

        Resetting post-cleanup is what keeps the baseline honest: if the guard
        just unloaded models, the freed memory must not be counted as headroom
        the next phase will keep.
        """
        status = ram_status()
        page = pagefile_status()
        available = (
            int(status["available"]) if status and status.get("total") else None
        )
        with self._lock:
            self._last_seen = time.monotonic()
            self._win_baseline = available
            self._win_min = available
            self._win_peak_pagefile = int(page.get("used") or 0) if page else 0

    def predicted_peak(self, *, margin: float = MEMORY_GUARD_PREDICT_MARGIN) -> int:
        """Next phase's expected footprint = last window's measured Δ × margin."""
        return int(self.snapshot()["delta"] * float(margin))

    def summary(self) -> dict:
        """Whole-run stats for the closing report."""
        with self._lock:
            self._last_seen = time.monotonic()
            base, mn = self._run_baseline, self._run_min
            delta = int(base - mn) if (base is not None and mn is not None and base > mn) else 0
            return {
                "delta": delta,
                "baseline": base,
                "min_available": mn,
                "peak_pagefile": int(self._run_peak_pagefile),
                "hard_floor_hits": int(self._hard_floor_hits),
                "samples": int(self._samples),
            }


_MONITOR: MemoryPeakMonitor | None = None
_MONITOR_LOCK = threading.Lock()


def begin_run_monitor(*, enabled: bool = True) -> None:
    """Start the background RAM sampler for one Director execute run."""
    if not enabled:
        return
    global _MONITOR
    with _MONITOR_LOCK:
        if _MONITOR is None:
            _MONITOR = MemoryPeakMonitor()
        monitor = _MONITOR
    try:
        monitor.start()
    except Exception as exc:
        log.warning("MiniMax H3 Director: memory monitor failed to start: %s", exc)


def monitor_snapshot() -> dict | None:
    monitor = _MONITOR
    if monitor is None:
        return None
    try:
        return monitor.snapshot()
    except Exception:
        return None


def monitor_reset_window() -> None:
    monitor = _MONITOR
    if monitor is None:
        return
    try:
        monitor.reset_window()
    except Exception:
        pass


def predicted_peak_from_monitor() -> int:
    """Measured footprint of the previous window, or 0 when not monitoring."""
    monitor = _MONITOR
    if monitor is None:
        return 0
    try:
        return monitor.predicted_peak()
    except Exception:
        return 0


def end_run_monitor() -> str | None:
    """Stop the sampler and return the Chinese one-line run summary, or None."""
    monitor = _MONITOR
    if monitor is None:
        return None
    try:
        stats = monitor.summary()
    except Exception:
        stats = {}
    try:
        monitor.stop()
    except Exception:
        pass
    if not stats:
        return None
    delta_gb = stats.get("delta", 0) / (1024 ** 3)
    min_gb = (stats.get("min_available") or 0) / (1024 ** 3)
    pagefile_gb = stats.get("peak_pagefile", 0) / (1024 ** 3)
    return (
        f"内存监视：整轮最高消耗 {delta_gb:.1f} GB，最低可用 {min_gb:.1f} GB，"
        f"页面文件峰值 {pagefile_gb:.1f} GB，硬地板触发 {stats.get('hard_floor_hits', 0)} 次"
    )


def release_after_segment_failure(*, label: str = "") -> None:
    """Deep release after a segment raised or was cancelled.

    A failed segment can leave partial latents, decode buffers and a half-staged
    model behind. Unloading keeps a retry from inheriting that wreckage and
    failing the same way again. Always runs (not gated by the water-level
    switch): this serves "do not crash", and it only fires on failure.
    """
    tag = f"{label} " if label else ""
    log.warning(
        "MiniMax H3 Director: %ssegment failed/cancelled — deep release "
        "(unload models + clear cache) so the retry starts clean",
        tag,
    )
    try:
        cleanup_segment_vram(enabled=True, unload_models=True)
    except Exception as exc:
        log.warning("MiniMax H3 Director: %sdeep release after failure failed: %s", tag, exc)
    # A segment failure aborts the whole run, so the sampler must not linger
    # until its idle timeout.
    try:
        monitor = _MONITOR
        if monitor is not None:
            monitor.stop()
    except Exception:
        pass
