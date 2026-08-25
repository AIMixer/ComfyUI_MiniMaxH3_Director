"""Deferred merge with per-seam local re-encode (Scheme A) — zero memory spike.

Overview
========
Scheme A keeps 99% of frames on disk via ``ffmpeg concat -c copy`` (stream copy,
no decode / re-encode, 100% lossless), and only decodes + applies ``seam_blending``
on a tiny **24-frame window** (last 12 frames of segment i + first 12 frames of
segment i+1) per seam.  The blended 24 frames are re-encoded once at CRF 18
(visually transparent) before being re-assembled back with ffmpeg stream copy.

Frame accounting (no overlaps / no drops) for 3 segments × N frames each::

    Input:   seg0 [0..N-1]    seg1 [0..N-1]    seg2 [0..N-1]

    Slice:   seg0_head.mp4     seam01_piece.mp4     seg1_mid.mp4    seam12_piece.mp4   seg2_tail.mp4
           frames: 0..N-13       24 blended         12..N-13         24 blended        12..N-1
                    (N-12f)       (24f)              (N-24f)          (24f)             (N-12f)

    Sum: (N-12) + 24 + (N-24) + 24 + (N-12) = 3N   ✓ original total
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director")

# Number of frames from each side of a seam that participate in seam_blending.
# Exact values taken from segment_continuity.py constants:
#   CONTINUITY_SEAM_ADD_LUMA_FRAMES   = 12 (frames of additive luma opening on curr)
#   CONTINUITY_HOLD_POP_SCAN + looka  = 8  (but 12 covers window safely)
#   CONTINUITY_MICRO_SEAM_MAD_SPAN    = 3  (weights applied on last/first 3)
# Using 12 on each side covers all steps with a small safety margin.
SEAM_HALF_WINDOW_FRAMES = 12
SEAM_TOTAL_FRAMES = 2 * SEAM_HALF_WINDOW_FRAMES


# ---------------------------------------------------------------------------
# FFmpeg discovery (mirrors lib.video_export logic — no extra deps)
# ---------------------------------------------------------------------------
def _ffmpeg_bin() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ffprobe_bin() -> str | None:
    p = shutil.which("ffprobe")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore
        base = Path(imageio_ffmpeg.get_ffmpeg_exe())
        cand = base.parent / ("ffprobe" + (".exe" if os.name == "nt" else ""))
        return str(cand) if cand.exists() else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MP4 helpers — metadata / slice / decode-range / concat-copy
# ---------------------------------------------------------------------------
def _probe_mp4(path: Path) -> dict[str, Any]:
    """Return dict with keys: frames, fps, sample_rate, audio_samples, duration_s, width, height."""
    ffprobe = _ffprobe_bin()
    frames = 0
    fps = 24.0
    sr = 0
    audio_samples = 0
    duration_s = 0.0
    w, h = 0, 0
    if not ffprobe:
        # Fallback: trust caller defaults.  probe returns 0-frames guard fires later.
        return {"frames": 0, "fps": fps, "sample_rate": sr, "audio_samples": audio_samples,
                "duration_s": duration_s, "width": 0, "height": 0}
    try:
        # Video stream
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,duration",
               "-count_frames", "-of", "default=nokey=0:noprint_wrappers=1", str(path)]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "ignore")
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("nb_read_frames="):
                try:
                    frames = int(line.split("=", 1)[1])
                except Exception:
                    pass
            elif line.startswith("r_frame_rate="):
                val = line.split("=", 1)[1]
                if "/" in val:
                    a, b = val.split("/", 1)
                    try:
                        fps = float(a) / max(1, float(b))
                    except Exception:
                        pass
                else:
                    try:
                        fps = float(val)
                    except Exception:
                        pass
            elif line.startswith("width="):
                try:
                    w = int(line.split("=", 1)[1])
                except Exception:
                    pass
            elif line.startswith("height="):
                try:
                    h = int(line.split("=", 1)[1])
                except Exception:
                    pass
            elif line.startswith("duration="):
                try:
                    duration_s = float(line.split("=", 1)[1])
                except Exception:
                    pass
        # Audio stream
        cmd_a = [ffprobe, "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=sample_rate,nb_samples,duration",
                 "-of", "default=nokey=0:noprint_wrappers=1", str(path)]
        out_a = subprocess.check_output(cmd_a, stderr=subprocess.STDOUT).decode("utf-8", "ignore")
        for line in out_a.splitlines():
            line = line.strip()
            if line.startswith("sample_rate="):
                try:
                    sr = int(line.split("=", 1)[1])
                except Exception:
                    pass
            elif line.startswith("nb_samples="):
                try:
                    audio_samples = int(line.split("=", 1)[1])
                except Exception:
                    pass
    except Exception as exc:
        log.warning("ffprobe failed on %s: %s", path, exc)
    if frames <= 0 and fps > 0 and duration_s > 0:
        frames = int(round(duration_s * fps))
    if audio_samples <= 0 and sr > 0 and duration_s > 0:
        audio_samples = int(round(duration_s * sr))
    return {"frames": frames, "fps": fps, "sample_rate": sr, "audio_samples": audio_samples,
            "duration_s": duration_s, "width": w, "height": h}


def _slice_mp4(src: Path, dst: Path, start_frame: int, end_frame: int, fps: float) -> None:
    """Slice ``[start_frame:end_frame]`` (0-based, end exclusive) out of ``src``.

    **Critical**: we MUST fully re-encode (not ``-c copy``) when slicing tiny
    12-frame head/tail pieces so the piece starts on a clean keyframe with PTS
    reset to 0; otherwise concat demuxer shows glitches at piece boundaries.
    For long middle pieces (hundreds of frames) we still re-encode CRF 18 to
    keep concat PTS stable; the cost is negligible compared to GPU generation.
    """
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; install FFmpeg or pip install imageio-ffmpeg")
    start_s = max(0.0, float(start_frame) / float(fps or 24.0))
    end_s = max(start_s, float(end_frame) / float(fps or 24.0))
    dur_s = max(1e-3, end_s - start_s)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.6f}", "-t", f"{dur_s:.6f}",
        "-i", str(src),
        "-vf", "setpts=PTS-STARTPTS",
        "-r", f"{fps:.6f}",
        "-af", "asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)


def _decode_frame_range(path: Path, start_frame: int, n_frames: int, fps: float) -> torch.Tensor:
    """Decode exactly ``n_frames`` starting from ``start_frame`` into a [F,H,W,3] float32 tensor."""
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    start_s = max(0.0, float(start_frame) / float(fps or 24.0))
    probe = _probe_mp4(path)
    w = int(probe["width"] or 0)
    h = int(probe["height"] or 0)
    if w <= 0 or h <= 0:
        # Fallback: decode 1 frame to learn size
        # NOTE: 参数顺序 — -frames:v 必须放 -i 后面 (OUTPUT option)，否则
        # ffmpeg 7.1 严格校验会拒绝并返回非零退出码（如 -22 / 4294967274）。
        probe_frames_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.6f}",
            "-i", str(path),
            "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        try:
            out = subprocess.check_output(probe_frames_cmd, stderr=subprocess.PIPE, timeout=30)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", "ignore")[-1500:]
            raise RuntimeError(
                f"_decode_frame_range probe fallback failed (exit={exc.returncode}) "
                f"on {path}:\n{stderr}"
            ) from exc
        # guess: no way; caller should have probe with w/h
        if len(out) % 3 != 0:
            raise RuntimeError(f"Cannot determine resolution of {path}")
        pix = len(out) // 3
        raise RuntimeError(f"Cannot determine resolution of {path}; decoded {pix} px")
    buf_size = n_frames * h * w * 3
    # NOTE: -ss 可作 input option (在 -i 前快速 seek)，但 -frames:v 是 OUTPUT
    # option 必须在 -i 后；否则 ffmpeg 7.1 拒绝并返回非零退出码。
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.6f}",
        "-i", str(path),
        "-frames:v", str(int(n_frames)),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    try:
        data = subprocess.check_output(cmd, stderr=subprocess.PIPE, timeout=60)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "ignore")[-1500:]
        raise RuntimeError(
            f"_decode_frame_range ffmpeg failed (exit={exc.returncode}) "
            f"on {path} start_frame={start_frame} n={n_frames} fps={fps}:\n{stderr}"
        ) from exc
    if len(data) < buf_size:
        # ffmpeg returns fewer frames than requested (segment shorter than expected)
        pad = b"\x80" * max(0, buf_size - len(data))
        data = data + pad
    data = data[:buf_size]
    arr = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(n_frames, h, w, 3)
    return arr.to(dtype=torch.float32) / 255.0


def _decode_audio_range(path: Path, start_sample: int, n_samples: int, sample_rate: int) -> torch.Tensor:
    """Decode exactly ``n_samples`` of mono float32 audio starting at ``start_sample``."""
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        # Return silent; caller will log warning once
        return torch.zeros(1, 1, max(1, n_samples), dtype=torch.float32)
    probe = _probe_mp4(path)
    sr = int(probe["sample_rate"] or sample_rate or 24000)
    if sr <= 0:
        sr = 24000
    start_s = max(0.0, float(start_sample) / float(sr))
    dur_s = max(1e-3, float(n_samples) / float(sr))
    # 同上：-ss 可作 input option；-t / -vn / -f / -ac / -ar 是 OUTPUT option
    # 必须放 -i 后。原代码 -t 也在 -i 前是错的（虽然 ffmpeg 对 -t 宽容）。
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.6f}",
        "-i", str(path),
        "-t", f"{dur_s:.6f}", "-vn",
        "-f", "f32le", "-ac", "1", "-ar", str(sr),
        "-",
    ]
    try:
        data = subprocess.check_output(cmd, stderr=subprocess.PIPE, timeout=30)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "ignore")[-1500:]
        log.warning(
            "[DeferredMerge-A] audio decode failed (exit=%s) on %s: %s",
            exc.returncode, path, stderr,
        )
        return torch.zeros(1, 1, max(1, n_samples), dtype=torch.float32)
    arr = torch.frombuffer(bytearray(data), dtype=torch.float32)
    if arr.numel() < n_samples:
        pad = torch.zeros(n_samples - arr.numel(), dtype=torch.float32)
        arr = torch.cat([arr, pad], dim=0)
    arr = arr[:n_samples].view(1, 1, n_samples)
    return arr.clamp(-1.0, 1.0)


def _concat_copy(piece_paths: list[Path], out_path: Path) -> None:
    """Losslessly concatenate a list of MP4 pieces via ``ffmpeg -f concat -c copy``.

    All inputs MUST have identical codec parameters (libx264/yuv420p + AAC 192k)
    and aligned PTS (every piece is produced with setpts/asetpts PTS-STARTPTS).
    """
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="mmx_concat_"))
    list_file = tmp_dir / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in piece_paths:
            # ffmpeg concat demuxer file format — single quotes with \' escaping.
            abs_path = str(Path(p).resolve()).replace("'", r"'\''")
            f.write(f"file '{abs_path}'\n")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _audio_crossfade(left: torch.Tensor, right: torch.Tensor, fade_samples: int = 240) -> torch.Tensor:
    """Linear crossfade ``left`` tail into ``right`` head over ``fade_samples`` (default 10ms @ 24kHz).

    Output length = left_len + right_len - fade_samples, which is exactly the
    expected length for (left_n + right_n) frames when both sides have the
    correct per-frame sample count.
    """
    left = left.float().view(-1)
    right = right.float().view(-1)
    f = max(1, min(int(fade_samples), min(left.shape[-1], right.shape[-1]) // 2))
    fade_out = torch.linspace(1.0, 0.0, f, dtype=left.dtype, device=left.device)
    fade_in = torch.linspace(0.0, 1.0, f, dtype=right.dtype, device=right.device)
    merged = left.clone()
    merged[-f:] = left[-f:] * fade_out + right[:f] * fade_in
    merged = torch.cat([merged, right[f:]], dim=0)
    return merged.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Seam blending — re-use segment_continuity pipeline on exactly 24 frames
# ---------------------------------------------------------------------------
def _run_seam_blending(left_tail: torch.Tensor, right_head: torch.Tensor,
                       continuity_enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply seam_blending steps to two 12-frame halves.

    Returns (left_blended, right_blended) each of shape [12, H, W, 3].
    If seam_blending is disabled or continuity is off, passthrough unchanged.
    """
    if not continuity_enabled or int(left_tail.shape[0]) == 0 or int(right_head.shape[0]) == 0:
        return left_tail.clone(), right_head.clone()
    try:
        from .segment_continuity import (
            _additive_opening_luma,
            _break_hold_pop_window,
            _micro_seam_bridge,
            _soften_body0_toward_prev,
            _unfreeze_held_tail,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("deferred_merge: seam_blending import failed (%s); skipping.", exc)
        return left_tail.clone(), right_head.clone()
    left = left_tail.float().cpu()
    body = right_head.float().cpu()
    try:
        left = _unfreeze_held_tail(left)
        left = _break_hold_pop_window(left, from_end=True)
        body = _break_hold_pop_window(body, from_end=False)
        body = _soften_body0_toward_prev(body, left)
        body = _additive_opening_luma(body, left)
        left, body = _micro_seam_bridge(left, body)
    except Exception as exc:
        log.warning("deferred_merge: seam_blending step failed (%s); passthrough seam halves.", exc)
        return left_tail.clone(), right_head.clone()
    return left.to(dtype=left_tail.dtype), body.to(dtype=right_head.dtype)


# ---------------------------------------------------------------------------
# Top-level public API
# ---------------------------------------------------------------------------
class DeferredMergeResult:
    preview_frames: torch.Tensor           # [preview_N, H, W, 3] float32  → returned to ComfyUI as IMAGE
    merged_audio: torch.Tensor | None      # [1, ch, samples] or None
    merged_video_path: Path | None         # Final merged MP4 on disk (or None if "no merge" mode)
    per_segment_paths: list[Path]          # Per-segment MP4 paths (same order as input)
    merge_script_path: Path | None         # Windows .bat for user manual merge (if >3 rounds)
    total_frames: int
    fps: float
    sample_rate: int

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _seam_window(frame_count: int, side: str) -> tuple[int, int]:
    """Return (start_frame, end_frame_exclusive) of the 12-frame half-window.

    ``side`` = "tail" for previous segment (last 12 frames),
    ``side`` = "head" for next segment (first 12 frames).
    """
    if frame_count <= SEAM_HALF_WINDOW_FRAMES:
        if side == "tail":
            return 0, max(1, frame_count)
        return 0, max(1, frame_count)
    if side == "tail":
        return frame_count - SEAM_HALF_WINDOW_FRAMES, frame_count
    return 0, SEAM_HALF_WINDOW_FRAMES


def deferred_merge_with_seam_reencode(
    *,
    segment_mp4_paths: list[Path],
    fps: float,
    seam_blending_enabled: bool,
    continuity_enabled: bool,
    release_vram_fn: Any | None = None,
    preview_only_frames: int = 50,
) -> DeferredMergeResult:
    """Scheme A deferred merge entry point.

    Parameters
    ----------
    segment_mp4_paths:
        Per-segment MP4 files produced by ``maybe_export_segment_mp4``.  At least 1.
    fps:
        Timeline FPS.
    seam_blending_enabled:
        Plan-level user toggle.
    continuity_enabled:
        Plan-level continuity toggle (seam blending is a no-op without it).
    release_vram_fn:
        Optional callable invoked right before merge starts (``lambda: cleanup_segment_vram()``
        or similar).  GPU VRAM held by sampling models is flushed here so that decode /
        blend CPU-side work never competes for RAM.
    preview_only_frames:
        Number of frames returned on ``preview_frames``. The full merged MP4 lives on
        disk at ``merged_video_path``; we never return the whole movie as a tensor
        (that would blow up RAM again).
    """
    if release_vram_fn is not None:
        try:
            release_vram_fn()
        except Exception as exc:
            log.warning("deferred_merge: release_vram_fn failed (%s); continuing.", exc)

    # Normalize
    mp4s = [Path(p) for p in segment_mp4_paths if Path(p).exists()]
    if not mp4s:
        raise ValueError("deferred_merge: no segment MP4 paths provided (all missing on disk)")
    if len(mp4s) == 1:
        # Nothing to merge; just probe and return the single segment.
        meta = _probe_mp4(mp4s[0])
        frames_read = min(int(meta["frames"] or 0), max(1, int(preview_only_frames)))
        if frames_read <= 0:
            frames_read = 1
        preview = _decode_frame_range(mp4s[0], 0, frames_read, fps or float(meta["fps"] or 24.0))
        total_fr = int(meta["frames"] or frames_read)
        sr = int(meta["sample_rate"] or 24000)
        merged_audio = None
        if total_fr > 0 and sr > 0:
            nsamples = int(round(total_fr * sr / float(fps or meta["fps"] or 24.0)))
            if nsamples > 0:
                merged_audio = _decode_audio_range(mp4s[0], 0, nsamples, sr)
        return DeferredMergeResult(
            preview_frames=preview.cpu().float(),
            merged_audio=merged_audio,
            merged_video_path=mp4s[0],
            per_segment_paths=mp4s,
            merge_script_path=None,
            total_frames=total_fr,
            fps=float(fps or meta["fps"] or 24.0),
            sample_rate=sr,
        )

    # Probe every segment once.
    probes: list[dict[str, Any]] = []
    for p in mp4s:
        meta = _probe_mp4(p)
        probes.append(meta)

    fps = float(probes[0]["fps"] or 24.0) if probes else 24.0
    for _i, (_p, _m) in enumerate(zip(mp4s, probes)):
        log.info("deferred_merge: seg %d: %s | frames=%d fps=%.3f sr=%d audio_samples=%d %dx%d",
                 _i, _p.name, int(_m["frames"] or 0), float(_m["fps"] or 0),
                 int(_m["sample_rate"] or 0), int(_m["audio_samples"] or 0),
                 int(_m["width"] or 0), int(_m["height"] or 0))

    total_frames_sum = sum(int(m["frames"] or 0) for m in probes)
    if total_frames_sum <= 4000:
        log.info("deferred_merge: 总帧数 %d ≤ 4000，走 精确uint8拼接单文件编码 模式（零帧误差+零PTS跳变）", total_frames_sum)
        return _scheme_a_precise_merge(
            mp4s, probes, fps,
            seam_blending_enabled=seam_blending_enabled,
            continuity_enabled=continuity_enabled,
            preview_only_frames=preview_only_frames,
        )
    log.info("deferred_merge: 总帧数 %d > 4000，走 流式精确拼接 模式（逐段解码→pipe，内存峰值≈1段+24帧）", total_frames_sum)
    return _scheme_a_stream_merge(
        mp4s, probes, fps,
        seam_blending_enabled=seam_blending_enabled,
        continuity_enabled=continuity_enabled,
        preview_only_frames=preview_only_frames,
    )

    # --- Legacy slice + concat path below (kept as dead code for reference) ---
    tmp_root = mp4s[0].parent / f"_seam_merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    piece_paths: list[Path] = []   # Will be len(segs) + 2*len(seams) slices + seam pieces; N:1 concat at end.

    try:
        for i, (mp4, meta) in enumerate(zip(mp4s, probes)):
            fc = int(meta["frames"] or 0)
            if fc <= 0:
                # Probe failed badly; include whole file with a log.  concat demuxer will try.
                log.warning("deferred_merge: probe gave 0 frames for %s; adding whole file.", mp4)
                piece_paths.append(mp4)
                continue

            is_first = (i == 0)
            is_last = (i == len(mp4s) - 1)

            # --- Slices within segment i ---
            # seg i layout: [ head | (tail_12 → consumed by seam_{i-1,i})? ]
            #                    [ (head_12 → consumed by seam_{i,i+1})? | tail ]
            # Strategy: slice into (at most) 3 parts.  Parts that get "replaced" by a seam
            # piece are NOT added to piece_paths — instead the seam piece slots between them.

            # Determine whether this segment's seam windows will be blended or passthrough.
            blend_left_seam = (not is_first) and seam_blending_enabled and continuity_enabled
            blend_right_seam = (not is_last) and seam_blending_enabled and continuity_enabled

            # Left head slice (everything before the first-12-frames window that was
            # consumed by the seam_{i-1,i} seam piece):
            # The last non-first segment's first 12 frames are already consumed
            # by the seam_{i-1,i} seam piece.  Skip its head slice entirely;
            # only the tail (frames[12:fc]) is needed for the last segment.
            if not is_last:
                if is_first:
                    left_bound = fc - SEAM_HALF_WINDOW_FRAMES if (not is_last and blend_right_seam) else fc
                else:
                    # 12 first frames of this segment are "owned" by seam_{i-1,i}.
                    # So head slice starts at frame 12.
                    if not is_last and blend_right_seam:
                        left_bound = fc - SEAM_HALF_WINDOW_FRAMES  # head runs [12, fc-12)
                    else:
                        left_bound = fc  # head runs [12, fc)
                head_start = 0 if is_first else SEAM_HALF_WINDOW_FRAMES
                head_end = left_bound
                if head_end - head_start > 0:
                    p_head = tmp_root / f"s{i:03d}_head.mp4"
                    _slice_mp4(mp4, p_head, head_start, head_end, fps)
                    piece_paths.append(p_head)

            # Build seam piece for seam (i, i+1) if not last.
            if not is_last:
                j = i + 1
                mp4_next = mp4s[j]
                meta_next = probes[j]
                fc_next = int(meta_next["frames"] or 0)
                if fc_next <= 0:
                    # Can't blend a failed-probe neighbour; just insert nothing (full pieces
                    # will still be concatenated correctly).
                    log.warning("deferred_merge: skip seam %d→%d because next probe=0frames.", i, j)
                else:
                    # Decode seam halves.
                    lt_s, lt_e = _seam_window(fc, "tail")
                    rh_s, rh_e = _seam_window(fc_next, "head")
                    half_n = min(lt_e - lt_s, rh_e - rh_s)
                    if half_n <= 0:
                        log.warning("deferred_merge: seam %d→%d 0-frame window; skip blend.", i, j)
                    else:
                        left_tail = _decode_frame_range(mp4, lt_s, half_n, fps)
                        right_head = _decode_frame_range(mp4_next, rh_s, half_n, fps)
                        sr = max(int(meta["sample_rate"] or 0), int(meta_next["sample_rate"] or 0), 24000)
                        # 音频：对应 seam 24 帧 × samples_per_frame
                        spf = int(round(float(sr) / float(fps or 24.0)))
                        audio_n = half_n * spf
                        # Left tail audio samples
                        lt_audio_start = max(0, int(meta["audio_samples"] or 0) - audio_n)
                        if lt_audio_start < 0 or int(meta["audio_samples"] or 0) < audio_n:
                            lt_audio_start = 0
                        left_audio = _decode_audio_range(mp4, lt_audio_start, audio_n, sr)
                        # Right head audio samples
                        right_audio = _decode_audio_range(mp4_next, 0, audio_n, sr)

                        if blend_right_seam:
                            left_frames, right_frames = _run_seam_blending(left_tail, right_head, continuity_enabled)
                        else:
                            left_frames, right_frames = left_tail.clone(), right_head.clone()

                        # Seam piece audio: left_audio 10ms xfaded → right_audio.
                        xf = max(1, int(round(0.010 * sr)))  # 10ms
                        seam_audio = _audio_crossfade(left_audio, right_audio, fade_samples=xf)
                        # Ensure audio sample count exactly matches frame count:
                        # half_n*2 frames at sr Hz => half_n*2*spf samples.
                        expected_samples = half_n * 2 * spf
                        cur = seam_audio.numel()
                        if cur < expected_samples:
                            seam_audio = torch.cat([seam_audio, torch.zeros(expected_samples - cur, dtype=seam_audio.dtype, device=seam_audio.device)])
                        elif cur > expected_samples:
                            seam_audio = seam_audio[:expected_samples]
                        seam_frames = torch.cat([left_frames, right_frames], dim=0)

                        p_seam = tmp_root / f"seam_{i:03d}_{j:03d}.mp4"
                        from ..lib.video_export import write_frames_to_mp4
                        write_frames_to_mp4(
                            p_seam,
                            seam_frames.cpu().float(),
                            fps=float(fps or 24.0),
                            audio={
                                "waveform": seam_audio.cpu().float(),
                                "sample_rate": int(sr),
                            },
                        )
                        piece_paths.append(p_seam)

            # Right tail slice — only relevant for LAST segment (all non-last segments
            # already handled everything after head via seam piece's right half).
            if is_last and is_first:
                # Single segment (impossible here; len >= 2 guard above).
                pass
            elif is_last:
                tail_start = SEAM_HALF_WINDOW_FRAMES
                tail_end = fc
                if tail_end - tail_start > 0:
                    p_tail = tmp_root / f"s{i:03d}_tail.mp4"
                    _slice_mp4(mp4, p_tail, tail_start, tail_end, fps)
                    piece_paths.append(p_tail)

        # Safety dedupe + drop zero-sized pieces.
        cleaned: list[Path] = []
        for p in piece_paths:
            try:
                if not p.exists() or p.stat().st_size == 0:
                    log.warning("deferred_merge: dropping empty piece %s", p)
                    continue
            except OSError:
                continue
            cleaned.append(p)
        piece_paths = cleaned

        # Final concat copy.
        out_dir = mp4s[0].parent
        out_path = out_dir / f"deferred_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        # Debug: log piece list
        log.info("deferred_merge: piece list (%d items):", len(piece_paths))
        for _pi, _pp in enumerate(piece_paths):
            try:
                _sz = _pp.stat().st_size
                log.info("  piece %d: %s (%d bytes)", _pi, _pp.name, _sz)
            except Exception:
                log.info("  piece %d: %s", _pi, _pp)

        _concat_copy(piece_paths, out_path)

        # Calculate expected total and verify actual output
        total_frames_sum = sum(int(m["frames"] or 0) for m in probes)
        _out_meta = _probe_mp4(out_path)
        _actual_frames = int(_out_meta["frames"] or 0)
        log.info("deferred_merge: output probe: frames=%d fps=%.3f (expected sum=%d)",
                 _actual_frames, float(_out_meta["fps"] or 0), total_frames_sum)

        # Preview: only first preview_only_frames frames from merged output.
        preview = _decode_frame_range(out_path, 0, max(1, int(preview_only_frames)), fps)

        # Audio for the whole merged file.
        merged_audio = None
        sr = int(max(int(m.get("sample_rate") or 0) for m in probes) if probes else 24000)
        if sr <= 0:
            sr = 24000
        total_audio_samples = int(round(total_frames_sum * float(sr) / float(fps or 24.0)))
        if total_audio_samples > 0:
            try:
                merged_audio = _decode_audio_range(out_path, 0, total_audio_samples, sr)
            except Exception as exc:
                log.warning("deferred_merge: full merged audio decode failed (%s); passing None.", exc)
                merged_audio = None

        # Generate a Windows .bat convenience script even though we already merged —
        # keeps consistent UX with Plus and lets users retry if disk piece paths need to move.
        bat_path = out_dir / f"merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bat"
        try:
            with open(bat_path, "w", encoding="utf-8") as bf:
                bf.write("@echo off\r\n")
                bf.write(f"REM MiniMax H3 Director deferred merge — Scheme A final file:\r\n")
                bf.write(f"REM   {out_path}\r\n")
                bf.write(f"REM (Already generated automatically.)\r\n")
                if _ffmpeg_bin():
                    bf.write(f"echo Already merged to: \"{out_path}\"\r\n")
                bf.write("pause\r\n")
        except OSError:
            bat_path = None

        # 成功后清理临时切片目录 _seam_merge_xxx — 最终合并视频 out_path 在
        # tmp_root 的同级目录（mp4s[0].parent），删除 tmp_root 不影响 out_path。
        # 失败时（except 分支）不清理，保留中间切片供调试。
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
            log.info("deferred_merge: 临时切片目录已清理: %s", tmp_root)
        except Exception as _e:
            log.warning("deferred_merge: 清理临时目录失败 (%s): %s", _e, tmp_root)

        return DeferredMergeResult(
            preview_frames=preview.cpu().float(),
            merged_audio=merged_audio,
            merged_video_path=out_path,
            per_segment_paths=mp4s,
            merge_script_path=bat_path,
            total_frames=max(total_frames_sum, int(preview.shape[0])),
            fps=float(fps or 24.0),
            sample_rate=int(sr),
        )
    except Exception:
        # On failure try to clean up temporary pieces, but NEVER raise without the original
        # per-seg paths still reported — the user still has all clips on disk.
        # Do not rmtree tmp_root blindly because out_path might be inside; keep for forensics.
        log.warning("deferred_merge: 失败，保留临时切片目录供调试: %s", tmp_root)
        raise


# ---------------------------------------------------------------------------
# Scheme A — Precise mode (≤ 4000 frames): CPU uint8 tensor concat, single-file encode.
# Memory: uint8 = 1/4 of float32.  4000 frames @ 854×480 ≈ 4.7 GB.
# ---------------------------------------------------------------------------


def _write_merged_mp4_uint8(path: Path, frames_u8: torch.Tensor, audio_wav: torch.Tensor, fps: float, sr: int) -> None:
    """Write final merged .mp4 from uint8 frames, streaming in chunks to avoid peak RAM."""
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg unavailable for _write_merged_mp4_uint8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames_u8.shape
    eh = h + (h % 2)
    ew = w + (w % 2)
    need_pad = (eh != h or ew != w)

    import tempfile as _tmp, wave as _wave
    tmp_dir = _tmp.mkdtemp(prefix="mmx_merge_")
    wav_path = os.path.join(tmp_dir, "merge_audio.wav")
    try:
        with _wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            audio_i16 = (audio_wav.float().clamp(-1, 1).view(-1).cpu().numpy() * 32767).astype("<i2")
            wf.writeframes(audio_i16.tobytes())
    except Exception:
        wav_path = None

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
    ]
    if wav_path:
        cmd += ["-i", wav_path]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18",
            "-movflags", "+faststart"]
    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(path))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunk = 200
    try:
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            chunk_u8 = frames_u8[start:end]
            if need_pad:
                if eh != h:
                    chunk_u8 = torch.cat([chunk_u8, chunk_u8[:, -1:, :1].expand(-1, eh - h, ew, -1)], dim=1) if eh != h else chunk_u8
                if ew != w:
                    chunk_u8 = torch.cat([chunk_u8, chunk_u8[:, :, -1:].expand(-1, eh, ew - w, -1)], dim=2) if ew != w else chunk_u8
            proc.stdin.write(chunk_u8.numpy().tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "ignore")[-800:]
        raise RuntimeError(f"_write_merged_mp4_uint8 ffmpeg failed (code={proc.returncode}): {err}")
    try:
        import shutil as _sh
        _sh.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass


def _scheme_a_precise_merge(
    mp4s: list[Path],
    probes: list[dict],
    fps: float,
    *,
    seam_blending_enabled: bool = True,
    continuity_enabled: bool = False,
    preview_only_frames: int = 50,
) -> DeferredMergeResult:
    """Exact per-frame concat + seam_blending on CPU tensors → one .mp4 encode.

    Replaces the slice + ffmpeg-concat-copy path for small projects to avoid:
      - ±1 frame rounding from -ss / -t timestamps.
      - PTS / keyframe discontinuities across independently encoded pieces.
    """
    n_seg = len(mp4s)
    half = SEAM_HALF_WINDOW_FRAMES
    sr = max((int(m["sample_rate"] or 0) for m in probes), default=0)
    if sr <= 0:
        sr = 24000
    spf = max(1, int(round(float(sr) / float(fps or 24.0))))

    # 1. Full decode of every segment's video & audio onto CPU (uint8, 4x less RAM than float32).
    seg_frames: list[torch.Tensor] = []   # uint8 [F, H, W, 3]
    seg_audios: list[torch.Tensor] = []
    fcs: list[int] = []
    for i, (path, meta) in enumerate(zip(mp4s, probes)):
        fc = int(meta["frames"] or 0)
        if fc <= 0:
            log.warning("deferred_merge-precise: seg %d probe frames=0; fallback full decode", i)
            # Fallback: imageio decode entire clip.
            import imageio_ffmpeg  # type: ignore
            try:
                reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
                size = None
                frames = []
                for frame in reader:
                    if isinstance(frame, dict):
                        size = frame.get("size")
                    else:
                        if size is not None and (
                            getattr(frame, "shape", None) is None
                            or frame.shape[0] != size[1]
                            or frame.shape[1] != size[0]
                        ):
                            try:
                                import numpy as _np
                                frame = _np.asarray(frame, dtype=_np.uint8).reshape(size[1], size[0], 3)
                            except Exception:
                                continue
                        frames.append(frame)
            except Exception:
                frames = []
            if not frames:
                log.error("deferred_merge-precise: seg %d fallback decode failed; skipping", i)
                continue
            import numpy as _np
            buf = _np.stack(frames, axis=0)  # [F, H, W, 3] uint8
            t = torch.from_numpy(buf).to(torch.uint8)
            fc = t.shape[0]
            seg_frames.append(t)
            fcs.append(fc)
            audio_n = fc * spf
            try:
                a = _decode_audio_range(path, 0, audio_n, sr)
            except Exception:
                a = torch.zeros(audio_n, dtype=torch.float32)
            a = a.view(-1)  # flatten (1,1,N)→(N,) for consistent 1D slicing
            seg_audios.append(a)
            continue
        t = _decode_frame_range(path, 0, fc, fps)   # returns float32 [F, H, W, 3] [0,1]
        seg = (t * 255).clamp(0, 255).to(torch.uint8)    # → uint8, 4x less RAM
        del t
        seg_frames.append(seg)
        fcs.append(fc)
        audio_n = fc * spf
        try:
            a = _decode_audio_range(path, 0, audio_n, sr)
        except Exception:
            a = torch.zeros(audio_n, dtype=torch.float32)
        a = a.view(-1)  # flatten (1,1,N)→(N,) for consistent 1D slicing
        if a.numel() < audio_n:
            a = torch.cat([a, torch.zeros(audio_n - a.numel(), dtype=a.dtype)])
        elif a.numel() > audio_n:
            a = a[:audio_n]
        seg_audios.append(a)
        log.info("deferred_merge-precise: seg %d decoded: %d frames, audio %d samples @ %dHz",
                 i, fc, a.numel(), sr)

    # 2. Assemble pieces on CPU, blending seam (i, i+1) if enabled.
    video_parts: list[torch.Tensor] = []
    audio_parts: list[torch.Tensor] = []
    total_expected_frames = 0
    for i in range(n_seg):
        fc = fcs[i]
        is_first = (i == 0)
        is_last = (i == n_seg - 1)
        blend_right = (not is_last) and seam_blending_enabled and continuity_enabled

        # Head / body
        if is_last:
            head_end = fc
        else:
            head_end = fc - half if blend_right else fc
        head_start = 0 if is_first else half
        if head_end - head_start > 0:
            video_parts.append(seg_frames[i][head_start:head_end].contiguous())
            a_s = head_start * spf
            a_e = head_end * spf
            audio_parts.append(seg_audios[i][a_s:a_e].contiguous())
            total_expected_frames += (head_end - head_start)
            log.info("deferred_merge-precise: seg %d body: frames [%d:%d) = %d",
                     i, head_start, head_end, head_end - head_start)

        # Seam piece between seg i and seg i+1
        if not is_last:
            j = i + 1
            fc_j = fcs[j]
            if fc > half and fc_j >= half and blend_right:
                # Convert uint8→float32[0,1] for seam_blending (expects [0,1] range)
                lt_start = fc - half
                left_tail_u8 = seg_frames[i][lt_start:lt_start + half]
                right_head_u8 = seg_frames[j][0:half]
                left_tail = left_tail_u8.float() / 255.0
                right_head = right_head_u8.float() / 255.0
                del left_tail_u8, right_head_u8
                left_frames, right_frames = _run_seam_blending(left_tail, right_head, continuity_enabled)
                # Convert back to uint8
                left_frames = (left_frames * 255).clamp(0, 255).to(torch.uint8)
                right_frames = (right_frames * 255).clamp(0, 255).to(torch.uint8)

                la_start = lt_start * spf
                left_audio = seg_audios[i][la_start:la_start + half * spf].contiguous()
                right_audio = seg_audios[j][0:half * spf].contiguous()
                xf = max(1, int(round(0.010 * sr)))
                seam_audio = _audio_crossfade(left_audio, right_audio, fade_samples=xf)
                expected = half * 2 * spf
                cur = seam_audio.numel()
                if cur < expected:
                    seam_audio = torch.cat([seam_audio, torch.zeros(expected - cur, dtype=seam_audio.dtype)])
                elif cur > expected:
                    seam_audio = seam_audio[:expected]
                video_parts.append(torch.cat([left_frames, right_frames], dim=0).contiguous())
                audio_parts.append(seam_audio.contiguous())
                total_expected_frames += half * 2
                log.info("deferred_merge-precise: seam %d→%d blended %d frames (xf=%d)",
                         i, j, half * 2, xf)
            elif fc > half and fc_j >= half:
                # seam_blending disabled: still own 12+12 overlap area (head/tail alignment)
                lt_start = fc - half
                left_tail = seg_frames[i][lt_start:lt_start + half]
                right_head = seg_frames[j][0:half]
                la_start = lt_start * spf
                left_audio = seg_audios[i][la_start:la_start + half * spf]
                right_audio = seg_audios[j][0:half * spf]
                video_parts.append(torch.cat([left_tail, right_head], dim=0).contiguous())
                audio_parts.append(torch.cat([left_audio, right_audio], dim=0).contiguous())
                total_expected_frames += half * 2
                log.info("deferred_merge-precise: seam %d→%d passthrough %d frames (seam_blending disabled)",
                         i, j, half * 2)

    del seg_frames, seg_audios
    import gc as _gc; _gc.collect()

    merged_video = torch.cat(video_parts, dim=0) if video_parts else torch.zeros((1, 480, 854, 3))
    merged_audio = torch.cat(audio_parts, dim=0) if audio_parts else torch.zeros(0, dtype=torch.float32)
    del video_parts, audio_parts; _gc.collect()

    actual_frames = int(merged_video.shape[0])
    expected_audio_n = actual_frames * spf
    if merged_audio.numel() < expected_audio_n:
        merged_audio = torch.cat([merged_audio, torch.zeros(expected_audio_n - merged_audio.numel(), dtype=merged_audio.dtype)])
    elif merged_audio.numel() > expected_audio_n:
        merged_audio = merged_audio[:expected_audio_n]
    log.info("deferred_merge-precise: assembled final tensor: %d frames, audio %d samples (expected %d frames, diff=%+d)",
             actual_frames, merged_audio.numel(), total_expected_frames, actual_frames - total_expected_frames)

    # 3. Single encode of the final .mp4 via uint8 streaming (zero float32 spike).
    out_dir = mp4s[0].parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"deferred_merged_a_precise_{ts}.mp4"
    _write_merged_mp4_uint8(
        out_path,
        merged_video,
        merged_audio,
        fps=float(fps or 24.0),
        sr=int(sr),
    )
    log.info("deferred_merge-precise: ✅ Wrote final file: %s (%d frames, seam_blending=%s, single-encode zero PTS jump)",
             out_path, actual_frames, seam_blending_enabled)

    # 4. Preview + return (convert uint8→float32 only for preview frames).
    prev_n = max(1, min(actual_frames, int(preview_only_frames)))
    preview = merged_video[:prev_n].float() / 255.0
    result_audio = merged_audio.clone() if merged_audio.numel() > 0 else None
    del merged_video, merged_audio; _gc.collect()
    return DeferredMergeResult(
        preview_frames=preview,
        merged_audio=result_audio,
        merged_video_path=out_path,
        per_segment_paths=mp4s,
        merge_script_path=None,
        total_frames=max(actual_frames, 0),
        fps=float(fps or 24.0),
        sample_rate=int(sr),
    )


# ---------------------------------------------------------------------------
# Scheme A — Stream mode (> 4000 frames): segment-by-segment decode → pipe.
# Memory peak = 1 segment + 24 seam frames, independent of total segment count.
# ---------------------------------------------------------------------------
def _scheme_a_stream_merge(
    mp4s: list[Path],
    probes: list[dict],
    fps: float,
    *,
    seam_blending_enabled: bool = True,
    continuity_enabled: bool = False,
    preview_only_frames: int = 50,
) -> DeferredMergeResult:
    """Stream-exact merge: decode segment-by-segment, write to ffmpeg pipe."""
    import gc as _gc
    n_seg = len(mp4s)
    half = SEAM_HALF_WINDOW_FRAMES
    sr = max((int(m["sample_rate"] or 0) for m in probes), default=0)
    if sr <= 0:
        sr = 24000
    spf = max(1, int(round(float(sr) / float(fps or 24.0))))
    fcs = [int(m["frames"] or 0) for m in probes]

    # --- Phase 1: Pre-assemble audio ---
    audio_parts: list[torch.Tensor] = []
    total_expected_frames = 0
    seg_audios: list[torch.Tensor] = []
    for i, (path, meta) in enumerate(zip(mp4s, probes)):
        fc = fcs[i]
        audio_n = max(1, fc) * spf
        try:
            a = _decode_audio_range(path, 0, audio_n, sr)
        except Exception:
            a = torch.zeros(audio_n, dtype=torch.float32)
        a = a.view(-1)  # flatten (1,1,N)→(N,) for consistent 1D slicing
        if a.numel() < audio_n:
            a = torch.cat([a, torch.zeros(audio_n - a.numel(), dtype=a.dtype)])
        elif a.numel() > audio_n:
            a = a[:audio_n]
        seg_audios.append(a)

    for i in range(n_seg):
        fc = fcs[i]
        is_first = (i == 0)
        is_last = (i == n_seg - 1)
        blend_right = (not is_last) and seam_blending_enabled and continuity_enabled
        if is_last:
            head_end = fc
        else:
            head_end = fc - half if blend_right else fc
        head_start = 0 if is_first else half
        if head_end - head_start > 0:
            a_s = head_start * spf
            a_e = head_end * spf
            audio_parts.append(seg_audios[i][a_s:a_e].contiguous())
            total_expected_frames += (head_end - head_start)
        if not is_last:
            j = i + 1
            fc_j = fcs[j]
            if fc > half and fc_j >= half and blend_right:
                lt_start = fc - half
                la_start = lt_start * spf
                left_audio = seg_audios[i][la_start:la_start + half * spf].contiguous()
                right_audio = seg_audios[j][0:half * spf].contiguous()
                xf = max(1, int(round(0.010 * sr)))
                seam_audio = _audio_crossfade(left_audio, right_audio, fade_samples=xf)
                expected = half * 2 * spf
                cur = seam_audio.numel()
                if cur < expected:
                    seam_audio = torch.cat([seam_audio, torch.zeros(expected - cur, dtype=seam_audio.dtype)])
                elif cur > expected:
                    seam_audio = seam_audio[:expected]
                audio_parts.append(seam_audio.contiguous())
                total_expected_frames += half * 2

    del seg_audios; _gc.collect()
    merged_audio = torch.cat(audio_parts, dim=0) if audio_parts else torch.zeros(0, dtype=torch.float32)
    del audio_parts; _gc.collect()
    expected_audio_n = total_expected_frames * spf
    if merged_audio.numel() < expected_audio_n:
        merged_audio = torch.cat([merged_audio, torch.zeros(expected_audio_n - merged_audio.numel(), dtype=merged_audio.dtype)])
    elif merged_audio.numel() > expected_audio_n:
        merged_audio = merged_audio[:expected_audio_n]
    log.info("deferred_merge-stream: audio pre-assembled: %d samples", merged_audio.numel())

    # --- Phase 2: Write audio wav + open ffmpeg pipe ---
    w = int(probes[0]["width"] or 0)
    h = int(probes[0]["height"] or 0)
    if w <= 0 or h <= 0:
        w, h = 854, 480
    eh = h + (h % 2)
    ew = w + (w % 2)
    need_pad = (eh != h or ew != w)

    out_dir = mp4s[0].parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"deferred_merged_a_stream_{ts}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import tempfile as _tmp, wave as _wave
    tmp_dir = _tmp.mkdtemp(prefix="mmx_stream_")
    wav_path = os.path.join(tmp_dir, "stream_audio.wav")
    try:
        with _wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            audio_i16 = (merged_audio.float().clamp(-1, 1).view(-1).cpu().numpy() * 32767).astype("<i2")
            wf.writeframes(audio_i16.tobytes())
    except Exception:
        wav_path = None

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg unavailable for _scheme_a_stream_merge")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
    ]
    if wav_path:
        cmd += ["-i", wav_path]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18",
            "-movflags", "+faststart"]
    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(out_path))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunk_size = 200
    actual_frames = 0

    def _write_pipe(frames_u8: torch.Tensor):
        nonlocal actual_frames
        n = frames_u8.shape[0]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_u8 = frames_u8[start:end]
            if need_pad:
                if eh != h:
                    chunk_u8 = torch.cat([chunk_u8, chunk_u8[:, -1:, :1].expand(-1, eh - h, ew, -1)], dim=1)
                if ew != w:
                    chunk_u8 = torch.cat([chunk_u8, chunk_u8[:, :, -1:].expand(-1, eh, ew - w, -1)], dim=2)
            proc.stdin.write(chunk_u8.numpy().tobytes())
        actual_frames += n

    # --- Phase 3: Stream video segment-by-segment ---
    for i in range(n_seg):
        fc = fcs[i]
        is_first = (i == 0)
        is_last = (i == n_seg - 1)
        blend_right = (not is_last) and seam_blending_enabled and continuity_enabled
        decode_start = 0 if is_first else half
        decode_n = fc - decode_start
        if decode_n <= 0:
            continue

        t = _decode_frame_range(mp4s[i], decode_start, decode_n, fps)
        seg = (t * 255).clamp(0, 255).to(torch.uint8)
        del t

        if not is_last and blend_right and fc > half and fcs[i + 1] >= half:
            body = seg[:-half].contiguous()
            left_tail_u8 = seg[-half:].contiguous()
            del seg
            j = i + 1
            nt = _decode_frame_range(mp4s[j], 0, half, fps)
            right_head_u8 = (nt * 255).clamp(0, 255).to(torch.uint8)
            del nt
            left_tail = left_tail_u8.float() / 255.0
            right_head = right_head_u8.float() / 255.0
            del left_tail_u8, right_head_u8
            left_frames, right_frames = _run_seam_blending(left_tail, right_head, continuity_enabled)
            del left_tail, right_head
            left_frames = (left_frames * 255).clamp(0, 255).to(torch.uint8)
            right_frames = (right_frames * 255).clamp(0, 255).to(torch.uint8)
            _write_pipe(body)
            _write_pipe(left_frames)
            _write_pipe(right_frames)
            log.info("deferred_merge-stream: seg %d: body=%d + seam=%d", i, body.shape[0], half * 2)
            del body, left_frames, right_frames
        elif not is_last and not blend_right and fc > half and fcs[i + 1] >= half:
            _write_pipe(seg)
            log.info("deferred_merge-stream: seg %d: %d frames (no blend)", i, seg.shape[0])
            del seg
        else:
            _write_pipe(seg)
            log.info("deferred_merge-stream: seg %d (last): %d frames", i, seg.shape[0])
            del seg
        _gc.collect()

    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "ignore")[-800:]
        raise RuntimeError(f"[stream] ffmpeg failed (code={proc.returncode}): {err}")
    try:
        import shutil as _sh
        _sh.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    log.info("deferred_merge-stream: ✅ Wrote: %s (%d frames)", out_path, actual_frames)

    prev_n = max(1, min(actual_frames, int(preview_only_frames)))
    preview = _decode_frame_range(out_path, 0, prev_n, fps)
    result_audio = merged_audio.clone() if merged_audio.numel() > 0 else None
    del merged_audio; _gc.collect()
    return DeferredMergeResult(
        preview_frames=preview,
        merged_audio=result_audio,
        merged_video_path=out_path,
        per_segment_paths=mp4s,
        merge_script_path=None,
        total_frames=max(actual_frames, 0),
        fps=float(fps or 24.0),
        sample_rate=int(sr),
    )
