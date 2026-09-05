"""Encode IMAGE tensors (+ optional AUDIO) to MP4 via ffmpeg."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.video_export")


def _ffmpeg_bin() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def _even(n: int) -> int:
    n = max(2, int(n))
    return n if n % 2 == 0 else n + 1


def _frames_to_rgb_u8(frames: torch.Tensor) -> np.ndarray:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError(f"Expected NHWC frames tensor, got {type(frames)} shape={getattr(frames, 'shape', None)}")
    arr = frames.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Expected at least 3 channels, got shape {arr.shape}")
    return (arr * 255.0).astype(np.uint8)


def _pad_even_hw(rgb: np.ndarray) -> np.ndarray:
    """Pad H/W to even sizes required by yuv420p / libx264."""
    n, h, w, c = rgb.shape
    eh, ew = _even(h), _even(w)
    if eh == h and ew == w:
        return rgb
    out = np.zeros((n, eh, ew, c), dtype=np.uint8)
    out[:, :h, :w, :] = rgb
    return out


def _write_wav(path: Path, audio: dict[str, Any]) -> bool:
    wave_t = audio.get("waveform")
    if not isinstance(wave_t, torch.Tensor) or wave_t.numel() <= 0:
        return False
    sr = int(audio.get("sample_rate") or 0)
    if sr <= 0:
        return False
    # [B, C, T] → [T, C] int16
    w = wave_t.detach().cpu().float()
    if w.ndim == 2:
        w = w.unsqueeze(0)
    if w.ndim != 3:
        return False
    w = w[0].transpose(0, 1).contiguous()  # T, C
    channels = int(w.shape[1])
    if channels <= 0:
        return False
    pcm = (w.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return True


def write_frames_to_mp4(
    path: str | Path,
    frames: torch.Tensor,
    *,
    fps: float,
    audio: dict[str, Any] | None = None,
) -> Path:
    """Write NHWC float frames to ``path`` as H.264 MP4. Raises on failure."""
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg unavailable (install FFmpeg on PATH or `pip install imageio-ffmpeg`)"
        )

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fps = float(fps or 24.0)
    if fps <= 0:
        fps = 24.0

    rgb = _pad_even_hw(_frames_to_rgb_u8(frames))
    n, h, w, _ = rgb.shape
    if n <= 0:
        raise ValueError("No frames to encode")

    tmp_dir = tempfile.mkdtemp(prefix="minimax_mp4_")
    tmp_mp4 = Path(tmp_dir) / "out.mp4"
    wav_path = Path(tmp_dir) / "audio.wav"
    try:
        has_audio = bool(audio) and _write_wav(wav_path, audio)
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
            f"{fps:.6f}",
            "-i",
            "-",
        ]
        if has_audio:
            cmd += ["-i", str(wav_path)]
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
        ]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd.append(str(tmp_mp4))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Pass the payload via communicate(input=...) instead of a manual
        # stdin.write()+stdin.close(): communicate() flushes stdin again, and
        # flushing an already-closed pipe raises "ValueError: flush of closed
        # file". communicate() also ignores BrokenPipeError internally when
        # ffmpeg exits early, so no extra guard is needed here.
        _, stderr = proc.communicate(input=rgb.tobytes())
        if proc.returncode != 0 or not tmp_mp4.is_file() or tmp_mp4.stat().st_size <= 0:
            err = (stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg encode failed (code={proc.returncode}): {err or 'unknown'}")

        # Atomic-ish publish: write to sibling temp then replace.
        publish_tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        try:
            if publish_tmp.exists():
                publish_tmp.unlink()
            shutil.copy2(tmp_mp4, publish_tmp)
            os.replace(publish_tmp, dest)
        finally:
            if publish_tmp.exists():
                try:
                    publish_tmp.unlink()
                except OSError:
                    pass
        return dest
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
