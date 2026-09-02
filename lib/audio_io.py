"""Extract source audio aligned with MiniMax H3 Director timeline (v2v / rv2v).

Independent of video tensors (does not touch decode / 娈甸棿寮曞). Same frame
selection as ``load_video_resampled``, with PCM-safe clocks:

  1. logical 鈫?frameMap 鈫?source index
  2. native = round((src / timeline_fps) * opencv_fps)  # same as video decode
  3. seek PCM at container time of that native, converted to the audio stream:
       pcm_start = video_pts0 + native0 * frame_dur 鈭?audio_start
     (full-file decode starts at sample 0 鈮?audio_start, not video_pts0)
  4. take exactly ``count / timeline_fps`` of audio (IMAGE / Combine clock).
     Pad/trim only 鈥?never stretch PTS media into a shorter timeline window
     (that reads as 鈥滃姞閫熸斁瀹屸€?.
  5. Join spans on consecutive *source* frames (not native decode indices) so
     timeline fps remapping does not shatter PCM into per-frame clicks; apply a
     short edge fade only at real multi-clip / gap boundaries.

If a PTS-based seek lands past the decoded PCM (common after deleting leading
timeline segments when fps/PTS disagree), fall back to ``native / file_fps``
so trim-leading does not become all silence.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Any

import torch

from .video_io import (
    ffprobe_bin,
    resolve_logical_frame_entry,
    resolve_video_path,
    video_clips_from_timeline,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.audio")

_ENCODE_ARGS = ("utf-8", "backslashreplace")

_OPENCV_FPS_CACHE: dict[str, float] = {}
# path -> (video_pts0, audio_start, frame_dur)
_AV_TIMING_CACHE: dict[str, tuple[float, float, float]] = {}


def _ffmpeg_bin() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def video_has_audio(path: str) -> bool | None:
    probe = ffprobe_bin()
    if not probe:
        return None
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            check=True,
        )
        return b"audio" in res.stdout.lower()
    except subprocess.CalledProcessError:
        return False


def _parse_ffmpeg_audio_info(stderr: str) -> tuple[int, int]:
    match = re.search(r", (\d+) Hz, (\w+), ", stderr)
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}.get(match.group(2), 2)
        return ar, ac
    return 44100, 2


def _probe_audio_stream(path: str) -> tuple[int, int]:
    probe = ffprobe_bin()
    if not probe or not path:
        return 44100, 2
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            check=True,
        )
        text = res.stdout.decode(*_ENCODE_ARGS).strip()
        parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
        ar = int(float(parts[0])) if parts else 44100
        ac = int(parts[1]) if len(parts) > 1 else 2
        if ar <= 0:
            ar = 44100
        if ac <= 0:
            ac = 2
        return ar, ac
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 44100, 2


def frames_to_audio_samples(frame_count: int, fps: float, sample_rate: int) -> int:
    frame_count = max(0, int(frame_count))
    fps = float(fps or 24.0)
    sample_rate = max(1, int(sample_rate))
    if frame_count <= 0 or fps <= 0:
        return 0
    return int(round(frame_count * sample_rate / fps))


def _median_positive_pts_delta(times: list[float]) -> float:
    ordered = sorted(set(float(value) for value in times))
    deltas = [
        right - left
        for left, right in zip(ordered, ordered[1:])
        if right - left > 1e-9
    ]
    if not deltas:
        return 0.0
    deltas.sort()
    middle = len(deltas) // 2
    if len(deltas) % 2:
        return float(deltas[middle])
    return float((deltas[middle - 1] + deltas[middle]) / 2.0)


def _clip_probe_fps(clip: dict, timeline_fps: float) -> float:
    file_fps = float(clip.get("nativeFps") or clip.get("native_fps") or 0.0)
    if file_fps <= 0:
        file_fps = float(timeline_fps or 24.0)
    return file_fps


def _opencv_file_fps(path: str, fallback: float) -> float:
    """Same FPS ``load_video_resampled`` reads via ``cv2.CAP_PROP_FPS``."""
    cached = _OPENCV_FPS_CACHE.get(path)
    if cached is not None:
        return cached
    fps = 0.0
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
    except Exception as exc:
        log.debug("OpenCV FPS probe failed for %s: %s", path, exc)
    if fps <= 0:
        fps = float(fallback or 24.0)
    _OPENCV_FPS_CACHE[path] = fps
    return fps


def _src_frame_to_native(src_frame: int, *, timeline_fps: float, file_fps: float) -> int:
    """Identical to ``load_video_resampled`` index mapping."""
    t_sec = max(0.0, float(int(src_frame)) / float(timeline_fps or 24.0))
    return int(round(t_sec * float(file_fps or timeline_fps or 24.0)))


def _probe_av_timing(path: str, *, fallback_fps: float) -> tuple[float, float, float]:
    """Return (video_pts0, audio_start, frame_dur) for PCM 鈫?picture sync.

    Full-file PCM from ffmpeg starts at sample 0 鈮?audio stream start_time.
    Video frame ``n`` presents at video_pts0 + n * frame_dur. Convert with
    ``pcm_t = video_t - audio_start`` before indexing samples.
    """
    cached = _AV_TIMING_CACHE.get(path)
    if cached is not None:
        return cached

    video_pts0 = 0.0
    audio_start = 0.0
    frame_dur = 1.0 / float(fallback_fps or 24.0) if fallback_fps > 0 else 1.0 / 24.0
    probe = ffprobe_bin()
    if probe and path and os.path.isfile(path):
        try:
            import json

            res = subprocess.run(
                [
                    probe,
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=index,codec_type,start_time,r_frame_rate",
                    "-of",
                    "json",
                    path,
                ],
                capture_output=True,
                check=True,
            )
            saw_video = False
            saw_audio = False
            for stream in json.loads(res.stdout.decode(*_ENCODE_ARGS)).get("streams") or []:
                ctype = str(stream.get("codec_type") or "")
                try:
                    st = float(stream.get("start_time") or 0.0)
                except (TypeError, ValueError):
                    st = 0.0
                if ctype == "video" and not saw_video:
                    video_pts0 = st
                    saw_video = True
                    rate = str(stream.get("r_frame_rate") or "")
                    if "/" in rate:
                        num, den = rate.split("/", 1)
                        try:
                            r = float(num) / float(den)
                            if r > 0:
                                frame_dur = 1.0 / r
                        except ValueError:
                            pass
                elif ctype == "audio" and not saw_audio:
                    audio_start = st
                    saw_audio = True
        except (subprocess.CalledProcessError, OSError, ValueError, KeyError):
            pass

        try:
            res = subprocess.run(
                [
                    probe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-read_intervals",
                    "%+#64",
                    "-show_entries",
                    "frame=pts_time",
                    "-of",
                    "csv=p=0",
                    path,
                ],
                capture_output=True,
                check=True,
            )
            times: list[float] = []
            for line in res.stdout.decode(*_ENCODE_ARGS).splitlines():
                val = line.split(",")[0].strip()
                if not val or val.upper() == "N/A":
                    continue
                try:
                    times.append(float(val))
                except ValueError:
                    continue
                if len(times) >= 64:
                    break
            if times:
                video_pts0 = float(min(times))
            measured_frame_dur = _median_positive_pts_delta(times)
            if measured_frame_dur > 0:
                nominal_frame_dur = float(frame_dur)
                ratio = (
                    measured_frame_dur / nominal_frame_dur
                    if nominal_frame_dur > 0
                    else 1.0
                )
                if nominal_frame_dur <= 0 or 0.5 <= ratio <= 2.0:
                    frame_dur = measured_frame_dur
                else:
                    log.warning(
                        "Ignoring implausible source PTS frame tick for %s: "
                        "measured=%.6fs, stream/fallback=%.6fs (ratio=%.3f)",
                        os.path.basename(path),
                        measured_frame_dur,
                        nominal_frame_dur,
                        ratio,
                    )
        except (subprocess.CalledProcessError, OSError, ValueError):
            pass

    log.info(
        "Source audio A/V timing for %s: video_pts0=%.6fs audio_start=%.6fs "
        "frame_dur=%.6fs (%.4f fps tick)",
        os.path.basename(path),
        video_pts0,
        audio_start,
        frame_dur,
        (1.0 / frame_dur) if frame_dur > 0 else 0.0,
    )
    _AV_TIMING_CACHE[path] = (video_pts0, audio_start, frame_dur)
    return video_pts0, audio_start, frame_dur


def _pcm_time_for_native(
    native: int,
    *,
    video_pts0: float,
    audio_start: float,
    frame_dur: float,
) -> float:
    """Seconds into decoded PCM for video frame ``native``."""
    dur = float(frame_dur) if frame_dur > 0 else 1.0 / 24.0
    video_t = float(video_pts0) + float(max(0, int(native))) * dur
    return max(0.0, video_t - float(audio_start))


def load_reference_audio(
    path: str,
    *,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Load a standalone audio file into ComfyUI AUDIO dict ``{waveform, sample_rate}``."""
    return _load_full_audio(path, cache=cache)


def _load_full_audio(
    path: str,
    *,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Decode a complete audio stream, optionally reusing a caller-owned cache.

    No process-wide PCM cache: callers that slice the same source during one
    Director execution may pass an execution-scoped ``cache``.
    """
    if cache is not None:
        cached = cache.get(path)
        if cached is not None:
            return cached
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg or not path or not os.path.isfile(path):
        return None
    ar, _ac = _probe_audio_stream(path)
    out_ac = 2
    args = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        path,
        "-vn",
        "-ac",
        str(out_ac),
        "-ar",
        str(ar),
        "-f",
        "f32le",
        "-",
    ]
    try:
        res = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode(*_ENCODE_ARGS)
        log.debug("No audio loaded from %s: %s", path, err.strip())
        return None
    if not res.stdout:
        return None
    parsed_ar, _ = _parse_ffmpeg_audio_info(res.stderr.decode(*_ENCODE_ARGS))
    if parsed_ar > 0:
        ar = parsed_ar
    audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
    usable = (int(audio.numel()) // out_ac) * out_ac
    if usable < out_ac:
        return None
    if usable < int(audio.numel()):
        audio = audio[:usable]
    wave = audio.reshape((-1, out_ac)).transpose(0, 1).unsqueeze(0).contiguous()
    out = {"waveform": wave, "sample_rate": int(ar)}
    if cache is not None:
        cache[path] = out
    return out


def _slice_samples(wave: torch.Tensor, *, src_start: int, n_samples: int) -> torch.Tensor:
    channels = int(wave.shape[1])
    have = int(wave.shape[-1])
    n_samples = max(0, int(n_samples))
    if n_samples <= 0:
        return wave[..., :0]
    src_start = max(0, int(src_start))
    if src_start >= have:
        return torch.zeros(1, channels, n_samples, dtype=wave.dtype, device=wave.device)
    take = wave[..., src_start : src_start + n_samples]
    got = int(take.shape[-1])
    if got == n_samples:
        return take.contiguous()
    pad = torch.zeros(1, channels, n_samples - got, dtype=wave.dtype, device=wave.device)
    return torch.cat([take, pad], dim=-1)


def _fit_wave_to_samples(wave: torch.Tensor, want: int) -> torch.Tensor:
    have = int(wave.shape[-1])
    if want <= 0:
        return wave[..., :0]
    if have == want:
        return wave
    if have > want:
        return wave[..., :want].contiguous()
    pad = torch.zeros(
        1, int(wave.shape[1]), want - have, dtype=wave.dtype, device=wave.device
    )
    return torch.cat([wave, pad], dim=-1)


def _resolve_pcm_start_sample(
    wave: torch.Tensor,
    *,
    pcm_start_sec: float,
    native0: int,
    file_fps: float,
    sr: int,
) -> int:
    """Pick a PCM sample index; fall back to OpenCV-index clock if PTS seek EOFs."""
    have = int(wave.shape[-1])
    i0 = max(0, int(round(float(pcm_start_sec) * sr)))
    if i0 < have:
        return i0
    if file_fps > 0 and have > 0:
        alt = max(0, int(round(float(max(0, int(native0))) / float(file_fps) * sr)))
        if alt < have:
            log.debug(
                "Source audio: PTS seek past PCM (i0=%d have=%d); "
                "fallback native/file_fps 鈫?sample %d",
                i0,
                have,
                alt,
            )
            return alt
    return min(i0, max(0, have - 1)) if have > 0 else 0


def _timeline_audio_spans(
    timeline: dict,
    logical_start: int,
    logical_end: int,
    frame_rate: float,
) -> list[tuple[str, float, float, int, float]]:
    """Spans: (path, pcm_start_sec, out_dur, native0, file_fps).

    Contiguity follows consecutive *source* frames on the timeline (not native
    decode indices). After fps remapping, native often skips/duplicates; requiring
    ``native == run_n0 + run_count`` splinters audio into per-frame chunks and
    causes seam clicks when concatenating.
    """
    if logical_end <= logical_start or frame_rate <= 0:
        return []
    clips = video_clips_from_timeline(timeline)
    if not clips:
        return []

    fps = float(frame_rate)
    spans: list[tuple[str, float, float, int, float]] = []
    run_path: str | None = None
    run_clip_idx = -1
    run_file_fps = fps
    run_n0 = 0
    run_last_src_frame = -1
    run_count = 0
    run_video_pts0 = 0.0
    run_audio_start = 0.0
    run_frame_dur = 1.0 / fps

    def flush_run() -> None:
        nonlocal run_path, run_clip_idx, run_file_fps, run_n0
        nonlocal run_last_src_frame, run_count
        nonlocal run_video_pts0, run_audio_start, run_frame_dur
        if run_path and run_count > 0:
            pcm_start = _pcm_time_for_native(
                run_n0,
                video_pts0=run_video_pts0,
                audio_start=run_audio_start,
                frame_dur=run_frame_dur,
            )
            out_dur = float(run_count) / fps
            if out_dur > 0:
                spans.append(
                    (run_path, pcm_start, out_dur, int(run_n0), float(run_file_fps))
                )
        run_path = None
        run_clip_idx = -1
        run_last_src_frame = -1
        run_count = 0

    for logical in range(logical_start, logical_end):
        clip_idx, src_frame = resolve_logical_frame_entry(timeline, logical)
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        clip = clips[clip_idx]
        try:
            path = resolve_video_path(clip)
        except ValueError:
            flush_run()
            continue
        fallback_fps = _clip_probe_fps(clip, fps)
        file_fps = _opencv_file_fps(path, fallback_fps)
        native = _src_frame_to_native(src_frame, timeline_fps=fps, file_fps=file_fps)
        video_pts0, audio_start, frame_dur = _probe_av_timing(
            path, fallback_fps=file_fps
        )
        if (
            run_path == path
            and run_clip_idx == clip_idx
            and run_count > 0
            and abs(run_file_fps - file_fps) < 1e-6
            and abs(run_frame_dur - frame_dur) < 1e-9
            and abs(run_video_pts0 - video_pts0) < 1e-9
            and abs(run_audio_start - audio_start) < 1e-9
            and src_frame == run_last_src_frame + 1
        ):
            run_count += 1
            run_last_src_frame = src_frame
            continue
        flush_run()
        run_path = path
        run_clip_idx = clip_idx
        run_file_fps = file_fps
        run_n0 = native
        run_last_src_frame = src_frame
        run_count = 1
        run_video_pts0 = video_pts0
        run_audio_start = audio_start
        run_frame_dur = frame_dur
    flush_run()
    return spans


def _fade_audio_chunk_boundaries(
    chunks: list[torch.Tensor], sample_rate: int, fade_ms: float = 3.0
) -> list[torch.Tensor]:
    """Light edge fades between real timeline joins (multi-clip / deleted ranges)."""
    if len(chunks) < 2 or sample_rate <= 0 or fade_ms <= 0:
        return chunks
    faded = [chunk.clone() for chunk in chunks]
    requested = max(1, int(round(float(sample_rate) * float(fade_ms) / 1000.0)))
    for previous, following in zip(faded, faded[1:]):
        count = min(requested, int(previous.shape[-1]), int(following.shape[-1]))
        if count <= 1:
            continue
        fade_out = torch.linspace(
            1.0,
            0.0,
            count,
            dtype=previous.dtype,
            device=previous.device,
        )
        fade_in = torch.linspace(
            0.0,
            1.0,
            count,
            dtype=following.dtype,
            device=following.device,
        )
        previous[..., -count:] *= fade_out
        following[..., :count] *= fade_in
    return faded


def extract_timeline_audio(
    timeline: dict,
    logical_start: int,
    logical_end: int,
    frame_rate: float,
    *,
    audio_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Extract source audio for logical timeline range [start, end)."""
    spans = _timeline_audio_spans(timeline, logical_start, logical_end, frame_rate)
    if not spans:
        return None
    if not _ffmpeg_bin():
        log.warning(
            "Source audio skipped: ffmpeg unavailable "
            "(install FFmpeg on PATH or `pip install imageio-ffmpeg`)."
        )
        return None
    paths = {path for path, _, _, _, _ in spans}
    probes = [video_has_audio(p) for p in paths]
    if probes and all(p is False for p in probes):
        return None

    chunks: list[torch.Tensor] = []
    sr = 44100
    fallback_seeks = 0
    for path, pcm_start, out_dur, native0, file_fps in spans:
        full = _load_full_audio(path, cache=audio_cache)
        if full is None:
            return None
        sr = int(full["sample_rate"] or sr)
        wave = full["waveform"]
        n_out = max(1, int(round(float(out_dur) * sr)))
        i0_pts = max(0, int(round(float(pcm_start) * sr)))
        i0 = _resolve_pcm_start_sample(
            wave,
            pcm_start_sec=pcm_start,
            native0=native0,
            file_fps=file_fps,
            sr=sr,
        )
        if i0 != i0_pts and i0_pts >= int(wave.shape[-1]):
            fallback_seeks += 1
        # Take timeline-length audio only (pad if short). Never stretch longer
        # PTS media into a shorter window 鈥?that causes 鈥滃姞閫熸斁瀹屸€?
        chunks.append(_slice_samples(wave, src_start=i0, n_samples=n_out))

    if fallback_seeks > 0:
        log.info(
            "Source audio: %d span(s) used native/file_fps seek "
            "(PTS start was past decoded PCM 鈥?typical after trimming the head).",
            fallback_seeks,
        )
    if len(spans) > 50:
        log.warning(
            "Source audio: %d fragmented spans (timeline fps likely 鈮?source). "
            "Prefer matching Director fps to source for cleaner cuts.",
            len(spans),
        )

    if not chunks:
        return None
    if len(chunks) > 1:
        chunks = _fade_audio_chunk_boundaries(chunks, sr)
        log.info(
            "Source audio: applied 3 ms edge fades at %d real timeline boundary(s).",
            len(chunks) - 1,
        )
    merged = torch.cat(chunks, dim=-1)
    fps = float(frame_rate or 24.0)
    n_frames = max(0, int(logical_end) - int(logical_start))
    want = frames_to_audio_samples(n_frames, fps, sr)
    if want > 0:
        merged = _fit_wave_to_samples(merged, want)
    return {"waveform": merged, "sample_rate": sr}


def extract_audio_segment(path: str, start_sec: float, duration_sec: float) -> dict[str, Any] | None:
    """Extract a time range from a file (legacy helper)."""
    if duration_sec <= 0:
        return None
    full = _load_full_audio(path)
    if full is None:
        return None
    sr = int(full["sample_rate"])
    i0 = max(0, int(round(float(start_sec) * sr)))
    n = max(1, int(round(float(duration_sec) * sr)))
    return {
        "waveform": _slice_samples(full["waveform"], src_start=i0, n_samples=n),
        "sample_rate": sr,
    }


def diagnose_source_audio_failure(
    timeline: dict,
    logical_start: int,
    logical_end: int,
    frame_rate: float,
) -> str:
    if not _ffmpeg_bin():
        return (
            "ffmpeg unavailable (install FFmpeg on PATH or `pip install imageio-ffmpeg`)"
        )
    spans = _timeline_audio_spans(timeline, logical_start, logical_end, frame_rate)
    if not spans:
        return "could not map timeline frames to a source video path"
    paths = sorted({path for path, _, _, _, _ in spans})
    probes = [video_has_audio(p) for p in paths]
    if probes and all(p is False for p in probes):
        return "input video has no audio track"
    if ffprobe_bin() is None:
        return (
            "audio extraction failed 鈥?ffprobe not found "
            "(install a full FFmpeg build with ffprobe on PATH) "
            "and/or ffmpeg could not decode audio from the source"
        )
    if any(p is True for p in probes):
        return "ffmpeg failed to extract audio despite an audio stream being present"
    return "audio extraction failed"
