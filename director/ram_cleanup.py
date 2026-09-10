"""「段间清理内存」system RAM cleanup between Director segments.

The same routine as the comfyui_memory_cleanup RAM-Cleanup node, run by the
Director itself between segments instead of only at graph start / end:

- flush the system file cache (Windows: SetSystemFileCacheSize; needs the
  increase-quota privilege, otherwise a silent no-op. Linux: malloc_trim),
- trim the working set of every process (EmptyWorkingSet),
- trim this process's working set (SetProcessWorkingSetSize(self, -1, -1)).

Win32 prototypes are declared on private WinDLL handles, so 64-bit handles and
SIZE_T values are passed correctly without touching ``ctypes.windll`` for
other packs.
"""

from __future__ import annotations

import logging
import platform
import time

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.ram")

_PROCESS_SET_QUOTA = 0x0100
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _ram_usage() -> tuple[float, float]:
    """(percent of RAM used, available MB)."""
    import psutil

    vm = psutil.virtual_memory()
    return float(vm.percent), vm.available / (1024 * 1024)


def _win_api():
    """kernel32 / psapi with explicit 64-bit-safe prototypes."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ps = ctypes.WinDLL("psapi", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessWorkingSetSize.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t]
    k32.SetProcessWorkingSetSize.restype = wintypes.BOOL
    k32.SetSystemFileCacheSize.argtypes = [ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
    k32.SetSystemFileCacheSize.restype = wintypes.BOOL
    ps.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    ps.EmptyWorkingSet.restype = wintypes.BOOL
    return k32, ps


def _size_max() -> int:
    import ctypes

    return ctypes.c_size_t(-1).value


def _trim_all_processes(k32, ps) -> int:
    """EmptyWorkingSet on every process we may open; returns how many succeeded."""
    import psutil

    access = _PROCESS_SET_QUOTA | _PROCESS_QUERY_LIMITED_INFORMATION
    trimmed = 0
    for proc in psutil.process_iter(["pid"]):
        pid = proc.info.get("pid")
        if not pid:
            continue
        handle = k32.OpenProcess(access, False, int(pid))
        if not handle:
            continue
        try:
            if ps.EmptyWorkingSet(handle):
                trimmed += 1
        finally:
            k32.CloseHandle(handle)
    return trimmed


def system_ram_cleanup(*, retries: int = 3, pause_s: float = 1.0) -> dict:
    """Run the RAM-Cleanup routine ``retries`` times. Never raises."""
    report = {"before_pct": 0.0, "after_pct": 0.0, "freed_mb": 0.0, "processes": 0}
    try:
        before_pct, before_avail = _ram_usage()
    except Exception as exc:
        log.warning("段间清理内存 / RAM cleanup skipped: %s", exc)
        return report
    system = platform.system()
    attempts = max(1, int(retries))
    for attempt in range(attempts):
        try:
            if system == "Windows":
                k32, ps = _win_api()
                size_max = _size_max()
                try:
                    k32.SetSystemFileCacheSize(size_max, size_max, 0)
                except Exception:
                    pass
                report["processes"] = _trim_all_processes(k32, ps)
                try:
                    k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(), size_max, size_max)
                except Exception:
                    pass
            elif system == "Linux":
                import ctypes

                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
        except Exception as exc:
            log.warning("段间清理内存 / RAM cleanup attempt %d failed: %s", attempt + 1, exc)
        if attempt + 1 < attempts and pause_s > 0:
            time.sleep(pause_s)
    try:
        after_pct, after_avail = _ram_usage()
    except Exception:
        after_pct, after_avail = before_pct, before_avail
    report.update(
        before_pct=before_pct,
        after_pct=after_pct,
        freed_mb=after_avail - before_avail,
    )
    log.info(
        "段间清理内存 / RAM cleanup: %.1f%% -> %.1f%%, freed %.0f MB, trimmed %d processes",
        before_pct, after_pct, report["freed_mb"], report["processes"],
    )
    return report
