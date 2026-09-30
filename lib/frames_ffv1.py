"""Lossless FFV1 codec for the director segment frame cache.

A segment's frames are pixel-exact data: the continuity handoff reuses the tail
frames of the previous segment as the next segment's locked prefix, so the disk
copy must survive a round-trip byte-for-byte. ``torch.save`` of the raw uint8
tensor is lossless but uncompressed and dominates disk usage (a 1920x1088x243
segment is ~1.4 GB). FFV1 stores the same pixels bit-exactly at roughly a third
of the size, is intra-only (so reading only the last N frames stays cheap) and
needs no extra Python dependency — it ships with the ffmpeg the plugin already
uses for audio extraction and mp4 export.

Notes:
  * ``-f matroska`` is passed explicitly: the cache's temp-publish path writes
    through a ``.<name>.<uuid>.tmp`` file, and ffmpeg cannot infer a container
    from a ``.tmp`` suffix.
  * ``-pix_fmt rgb24`` in/out (``bgr0(pc)`` in the container) keeps the full
    RGB range. Plain libx264/ffv1 with yuv420p would be lossy, which would
    silently corrupt every seam.
  * ``-slices`` is clamped to a divisor of the frame height; ffmpeg rejects
    slice counts that do not evenly divide it.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.frames_ffv1")

#: Suffix used for frame payloads on disk (``seg_0000.frames.mkv``).
FRAMES_SUFFIX = ".frames.mkv"

#: FFV1 level (format version) and slice target.
_FFV1_LEVEL = "3"
_FFV1_SLICE_TARGET = 4

_VIDEO_SIZE_RE = re.compile(rb"Video:.*?(\d{2,5})x(\d{2,5})")


def ffmpeg_bin() -> str | None:
    """Locate ffmpeg the same way the rest of the plugin does."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def frames_codec_available() -> bool:
    return bool(ffmpeg_bin())


def _slice_count(height: int) -> int:
    slices = _FFV1_SLICE_TARGET
    while slices > 1 and height % slices != 0:
        slices //= 2
    return slices


def _as_rgb_u8(frames: torch.Tensor) -> np.ndarray:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError(
            f"expected NHWC uint8 frames, got {type(frames)} "
            f"shape={getattr(frames, 'shape', None)}"
        )
    if frames.shape[-1] < 3:
        raise ValueError(f"expected >=3 channels, got shape {tuple(frames.shape)}")
    x = frames.detach().to("cpu")
    if x.dtype != torch.uint8:
        x = x.float().clamp(0.0, 1.0).mul(255.0).round().clamp(0, 255).to(torch.uint8)
    x = x[..., :3].contiguous()
    if int(x.shape[0]) <= 0:
        raise ValueError("no frames to encode")
    return x.numpy()


def encode_frames_ffv1(dest: str | Path, frames: torch.Tensor, *, fps: float = 24.0) -> Path:
    """Write uint8 NHWC ``frames`` to ``dest`` as lossless FFV1. Raises on failure."""
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg unavailable (install FFmpeg on PATH or `pip install imageio-ffmpeg`)"
        )
    rgb = _as_rgb_u8(frames)
    n, h, w, _ = rgb.shape
    try:
        rate = float(fps or 24.0)
    except (TypeError, ValueError):
        rate = 24.0
    if rate <= 0:
        rate = 24.0

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{rate:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        _FFV1_LEVEL,
        "-slices",
        str(_slice_count(h)),
        "-slicecrc",
        "0",
        "-f",
        "matroska",
        str(dest),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # Stream the frame buffer straight into the pipe: ``.tobytes()`` on a 1080p
    # segment would allocate another ~1.4 GB of RSS for no reason.
    payload = memoryview(np.ascontiguousarray(rgb)).cast("B")
    try:
        proc.stdin.write(payload)
        proc.stdin.close()
    except (BrokenPipeError, ValueError, OSError):
        # ffmpeg closed stdin early; the return-code check below decides.
        pass
    try:
        stderr = proc.stderr.read()
    except Exception:
        stderr = b""
    try:
        proc.wait(timeout=180)
    except Exception:
        proc.kill()
        proc.wait()
    if proc.returncode not in (0, None) or not dest.is_file() or dest.stat().st_size <= 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg ffv1 encode failed (code={proc.returncode}): {err or 'unknown'}"
        )
    log.debug(
        "Encoded %d frame(s) %dx%d to FFV1 (%s, %.1f MB)",
        n,
        w,
        h,
        dest.name,
        dest.stat().st_size / 1048576.0,
    )
    return dest


def decode_frames_ffv1(src: str | Path) -> torch.Tensor:
    """Read an FFV1 frame payload back as uint8 NHWC. Raises on failure."""
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg unavailable (install FFmpeg on PATH or `pip install imageio-ffmpeg`)"
        )
    src = Path(src)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-i",
        str(src),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = proc.stderr or b""
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg ffv1 decode failed (code={proc.returncode}): {err or 'unknown'}"
        )
    match = _VIDEO_SIZE_RE.search(stderr)
    if not match:
        raise RuntimeError(f"ffmpeg ffv1 decode: cannot determine frame size for {src.name}")
    w, h = int(match.group(1)), int(match.group(2))
    frame_bytes = w * h * 3
    data = proc.stdout or b""
    n = len(data) // frame_bytes
    if n <= 0:
        raise RuntimeError(
            f"ffmpeg ffv1 decode: no frames in {src.name} "
            f"({len(data)} bytes for {w}x{h})"
        )
    if len(data) != n * frame_bytes:
        log.warning(
            "FFV1 payload %s has %d trailing byte(s); ignoring.",
            src.name,
            len(data) - n * frame_bytes,
        )
    arr = np.frombuffer(data, dtype=np.uint8, count=n * frame_bytes).reshape(n, h, w, 3)
    # ``frombuffer`` is read-only; copy so downstream in-place ops stay legal.
    return torch.from_numpy(np.array(arr, copy=True))
