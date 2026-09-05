"""Incremental per-segment MP4 export for「分段导出」runs.

Best-effort: encode failures must never abort generation. Each run uses a
timestamp folder: ``output/minimax_seg_export/<YYYYMMDD_HHMMSS>/``.

Files:
  ``seg_XXXX.mp4`` — final clip (last refine pass / no Refine)
  ``seg_XXXX_pre.mp4`` — first pass (一采), only when Refine ran
  ``seg_XXXX_pN.mp4`` — refine pass N (分段导出且次数>1)
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import folder_paths
import torch

from .audio_export import prepare_segment_audio_for_file_export
from .plan import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.mp4_export")

VIDEO_EXPORT_TASKS = frozenset({"t2v", "i2v", "r2v", "fl2v", "v2v", "rv2v"})


def new_segment_mp4_run_dir(plan: DirectorPlan) -> Path | None:
    """Create ``minimax_seg_export/<YYYYMMDD_HHMMSS>/`` for one Director execute.

    Returns None when not in segments mode or the output dir is unavailable.
    """
    if getattr(plan, "export_mode", "all") != "segments":
        return None
    try:
        base = Path(folder_paths.get_output_directory()) / "minimax_seg_export"
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = base / stamp
        if root.exists():
            # Same-second collision (rare): append a short suffix.
            for i in range(1, 1000):
                candidate = base / f"{stamp}_{i:03d}"
                if not candidate.exists():
                    root = candidate
                    break
        root.mkdir(parents=True, exist_ok=False)
        log.info("MiniMax H3 Director segment mp4 run dir: %s", root)
        return root
    except OSError as exc:
        log.warning("Segment mp4 export dir unavailable (%s); skipped.", exc)
        return None


def _safe_mp4_suffix(suffix: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(suffix or ""))


def segment_mp4_path(run_dir: Path, seg: SegmentPlan, *, suffix: str = "") -> Path:
    tag = f"_{_safe_mp4_suffix(suffix)}" if _safe_mp4_suffix(suffix) else ""
    return Path(run_dir) / f"seg_{int(seg.index):04d}{tag}.mp4"


def mp4_export_kind(path: str | None) -> str:
    name = Path(str(path or "")).name
    if name.endswith("_pre.mp4"):
        return "一采 mp4"
    m = re.search(r"_p(\d+)\.mp4$", name)
    if m:
        return f"第{m.group(1)}轮精修 mp4"
    return "mp4"


def _pre_frames_distinct(pre_frames, frames) -> bool:
    if pre_frames is None or frames is None:
        return False
    if pre_frames is frames:
        return False
    if not isinstance(pre_frames, torch.Tensor) or pre_frames.ndim != 4:
        return False
    return int(pre_frames.shape[0]) > 0


def maybe_export_segment_mp4(
    run_dir: Path | None,
    plan: DirectorPlan,
    seg: SegmentPlan,
    frames: torch.Tensor,
    audio_dict: dict[str, Any] | None = None,
    *,
    suffix: str = "",
) -> str | None:
    """Write one segment mp4 into ``run_dir``. Never raises.

    ``suffix="pre"`` writes the first-pass clip (``seg_XXXX_pre.mp4``).
    ``suffix="p2"`` writes refine pass 2 (``seg_XXXX_p2.mp4``).

    Returns the absolute path string on success, otherwise None.
    """
    if run_dir is None or getattr(plan, "export_mode", "all") != "segments":
        return None
    task = str(getattr(seg, "task_key", "") or getattr(plan, "global_task_key", "") or "")
    if task not in VIDEO_EXPORT_TASKS:
        return None
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or int(frames.shape[0]) <= 0:
        return None

    dest = segment_mp4_path(run_dir, seg, suffix=suffix)

    try:
        from ..lib.video_export import write_frames_to_mp4

        audio = prepare_segment_audio_for_file_export(
            plan,
            seg,
            audio_dict=audio_dict,
            frame_count=int(frames.shape[0]),
        )
        path = write_frames_to_mp4(
            dest,
            frames.detach().cpu().float(),
            fps=float(getattr(plan, "frame_rate", 24) or 24),
            audio=audio,
        )
        log.info(
            "MiniMax H3 Director segment #%d %smp4 saved: %s",
            int(seg.index) + 1,
            "first-pass " if suffix == "pre" else "",
            path,
        )
        return str(path)
    except Exception as exc:
        log.warning(
            "Segment #%d %smp4 export failed (generation continues): %s",
            int(seg.index) + 1,
            "first-pass " if suffix == "pre" else "",
            exc,
        )
        return None


def maybe_export_segment_mp4s(
    run_dir: Path | None,
    plan: DirectorPlan,
    seg: SegmentPlan,
    frames: torch.Tensor,
    audio_dict: dict[str, Any] | None = None,
    *,
    pre_frames: torch.Tensor | None = None,
) -> list[str]:
    """Write final clip, plus first-pass when Refine produced a distinct tensor."""
    paths: list[str] = []
    final_path = maybe_export_segment_mp4(
        run_dir, plan, seg, frames, audio_dict,
    )
    if final_path:
        paths.append(final_path)
    if _pre_frames_distinct(pre_frames, frames):
        pre_path = maybe_export_segment_mp4(
            run_dir, plan, seg, pre_frames, audio_dict, suffix="pre",
        )
        if pre_path:
            paths.append(pre_path)
    return paths


def copy_segment_mp4_suffix(
    run_dir: Path | None,
    plan: DirectorPlan,
    seg: SegmentPlan,
    *,
    dest_suffix: str,
) -> str | None:
    """Copy ``seg_XXXX.mp4`` to ``seg_XXXX_<suffix>.mp4``. Never raises."""
    if run_dir is None or getattr(plan, "export_mode", "all") != "segments":
        return None
    tag = _safe_mp4_suffix(dest_suffix)
    if not tag:
        return None
    src = segment_mp4_path(run_dir, seg)
    dest = segment_mp4_path(run_dir, seg, suffix=tag)
    try:
        if not src.is_file():
            return None
        shutil.copy2(src, dest)
        log.info(
            "MiniMax H3 Director segment #%d copied %s → %s",
            int(seg.index) + 1,
            src.name,
            dest.name,
        )
        return str(dest)
    except Exception as exc:
        log.warning(
            "Segment #%d copy to %s failed: %s",
            int(seg.index) + 1,
            dest.name,
            exc,
        )
        return None


def is_released_poster(tensor, expected_frames: int) -> bool:
    """True when IMAGE slot was replaced by a 1-frame stand-in after mp4 flush."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        return False
    expected = int(expected_frames or 0)
    got = int(tensor.shape[0])
    return expected > 1 and got < expected


def released_output_slots(segment_outputs: list, frame_counts: list[int] | None) -> list[int]:
    """Indexes whose IMAGE slot is a poster; full clip is already on disk."""
    counts = frame_counts or []
    out: list[int] = []
    for pos, tensor in enumerate(segment_outputs):
        expected = int(counts[pos]) if pos < len(counts) else 0
        if is_released_poster(tensor, expected):
            out.append(pos)
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            out.append(pos)
            continue
        # 1x1 / odd-tiny leftovers encode to H.264 that Movies & TV rejects (0x80004005).
        if int(tensor.shape[1]) < 2 or int(tensor.shape[2]) < 2:
            out.append(pos)
    return out
