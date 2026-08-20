"""Streaming render of Director segment caches into a seam-fixed final video.

Why this exists
---------------
The regular「全部导出」path concatenates every segment's decoded frames in
memory and runs the seam fix on the full timeline, then hands the whole tensor
to CreateVideo. For long multi-segment projects the peak RAM is ~3x the total
frame volume plus the resident model weights, so a 90s+ project can be SIGKILLed
by the kernel OOM killer (cgroup memory.max) with no traceback.

This module renders the same output in a streaming fashion: it walks the
segment boundary loop one pair at a time (exactly mirroring
``concat_continuous_chunks`` in segment_continuity.py, so the seam fix is
bit-identical), writes each finalized segment's frames straight into ffmpeg
stdin and frees them immediately. Peak memory is ~2 segments (~6 GB), constant
regardless of total duration. No model weights are loaded.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess

import torch

_log = logging.getLogger("ComfyUI-MiniMaxH3-Director.cache_stream_render")

_SEG_RE = re.compile(r"seg_\d{4}\.pt$")


def _ffmpeg_exe() -> str:
    """Locate an ffmpeg binary.

    Prefer the binary bundled with imageio-ffmpeg (the same one ComfyUI's
    CreateVideo/SaveVideo rely on, so any video-capable install has it), then
    fall back to a system ffmpeg, then raise with a clear message.
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError(
        "ffmpeg not found: install ffmpeg (apt install ffmpeg) or "
        "pip install imageio-ffmpeg")



def find_cache_dir(output_dir: str | None = None) -> str | None:
    """Pick the cache dir with the most segments under output/minimax_seg_cache."""
    base = os.path.join(output_dir or _default_output_dir(), "minimax_seg_cache")
    if not os.path.isdir(base):
        return None
    best, best_n = None, 0
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        n = len([f for f in os.listdir(d) if _SEG_RE.search(f)])
        if n > best_n:
            best, best_n = d, n
    return best


def _default_output_dir() -> str:
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return "output"


def _seg_files(cache_dir: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(cache_dir, "seg_*.pt"))
                  if _SEG_RE.search(os.path.basename(f)))


class _FrameWriter:
    """Stream frames into a single ffmpeg rawvideo pipe (memory friendly)."""

    def __init__(self, width: int, height: int, fps: int, raw_mp4: str):
        cmd = [_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", raw_mp4]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.total = 0

    def write(self, frames: torch.Tensor) -> None:
        for f in range(int(frames.shape[0])):
            self.proc.stdin.write(
                frames[f].mul_(255.0).clamp_(0.0, 255.0).to(torch.uint8).numpy().tobytes())
        self.total += int(frames.shape[0])

    def close(self) -> int:
        self.proc.stdin.close()
        return self.proc.wait()


def stream_render_from_cache(
    cache_dir: str,
    out_path: str,
    fps: int = 24,
    seam_fix: bool = True,
) -> tuple[str, int]:
    """Render cached segments to ``out_path`` with the Director seam fix.

    Returns ``(out_path, total_frames)``. Raises on failure.

    The boundary loop below is a line-for-line streaming mirror of
    ``concat_continuous_chunks`` (segment_continuity.py) so the seam fix is
    identical to the full export, but only two segments are ever resident.
    """
    from .segment_continuity import (
        CONTINUITY_HOLD_POP_ON_TAIL,
        CONTINUITY_SPIKE_WEIGHT,
        _unfreeze_held_tail,
        _break_hold_pop_window,
        _ease_opening_spikes,
        _soften_body0_toward_prev,
        _additive_opening_luma,
        _micro_seam_bridge,
    )

    files = _seg_files(cache_dir)
    if not files:
        raise RuntimeError(f"MiniMax H3 cache render: no seg_XXXX.pt in {cache_dir}")

    first = torch.load(files[0], map_location="cpu")
    H, W = int(first.shape[1]), int(first.shape[2])
    total = int(first.shape[0])

    raw = out_path + ".raw.mp4"
    writer = _FrameWriter(W, H, int(fps), raw)
    try:
        if len(files) == 1 or not seam_fix:
            writer.write(first)
        else:
            cur = _unfreeze_held_tail(first)
            del first
            for i in range(1, len(files)):
                body = torch.load(files[i], map_location="cpu")
                total += int(body.shape[0])
                left = cur
                if CONTINUITY_HOLD_POP_ON_TAIL:
                    left = _break_hold_pop_window(left, from_end=True)
                body = _break_hold_pop_window(body, from_end=False)
                if float(CONTINUITY_SPIKE_WEIGHT) > 0:
                    body = _ease_opening_spikes(body)
                body = _soften_body0_toward_prev(body, left)
                body = _additive_opening_luma(body, left)
                left, body = _micro_seam_bridge(left, body)
                writer.write(left)
                del left
                cur = body
                _log.info("cache stream: segment %d/%d flushed (%d frames)",
                          i + 1, len(files), writer.total)
            writer.write(cur)
            del cur
        rc = writer.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg video encode failed rc={rc}")
    except Exception:
        try:
            writer.proc.kill()
        except Exception:
            pass
        raise

    # ---- audio: per-segment export waveforms are small; concat + mux ----
    audio_files = sorted(glob.glob(os.path.join(cache_dir, "seg_*.audio.pt")))
    if audio_files:
        waves, sr = [], 32000
        for f in audio_files:
            a = torch.load(f, map_location="cpu")
            if isinstance(a, dict) and "waveform" in a:
                waves.append(a["waveform"])
                sr = int(a.get("sample_rate", 32000))
        if waves:
            audio = torch.cat(waves, dim=-1)[0].contiguous()
            if audio.is_floating_point():
                audio = audio.clamp(-1.0, 1.0)
            wav = out_path + ".tmp.wav"
            try:
                from scipy.io import wavfile
                wavfile.write(wav, sr, (audio.numpy() * 32767.0).astype("int16").T)
            except ImportError:
                raise RuntimeError("scipy is required to mux cache audio")
            r = subprocess.run([_ffmpeg_exe(), "-y", "-loglevel", "error",
                                "-i", raw, "-i", wav,
                                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                                "-shortest", out_path])
            os.remove(wav)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg mux failed rc={r.returncode}")
            os.remove(raw)
        else:
            os.replace(raw, out_path)
    else:
        os.replace(raw, out_path)

    _log.info("cache stream render done: %s (%d frames)", out_path, total)
    return out_path, total
