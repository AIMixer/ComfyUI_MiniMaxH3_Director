"""Load source videos from ComfyUI input folder (VHS-style file references)."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

import folder_paths

from .image_prep import resolve_output_dimensions

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.video_io")


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for MiniMax H3 Director video loading. "
            "Install: pip install opencv-python-headless"
        ) from exc


def resolve_video_path(video: dict) -> str:
    """Resolve timeline video metadata to an absolute path under ComfyUI input."""
    video_file = (video.get("videoFile") or video.get("fileName") or "").strip()
    if not video_file:
        raise ValueError("No video file in MiniMax H3 Director timeline.")

    base = folder_paths.get_input_directory()
    subfolder = (video.get("subfolder") or "").strip().replace("\\", "/")

    candidates = []
    if subfolder and not video_file.startswith(subfolder):
        candidates.append(os.path.join(base, subfolder, os.path.basename(video_file)))
    candidates.append(os.path.join(base, video_file.replace("/", os.sep)))
    candidates.append(os.path.join(base, os.path.basename(video_file)))

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise ValueError(f"Video file not found in ComfyUI input: {video_file}")


def ffprobe_bin() -> str | None:
    """Resolve ffprobe: PATH first, then imageio-ffmpeg sibling binary if present."""
    import shutil

    probe = shutil.which("ffprobe")
    if probe:
        return probe
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        import os

        ff = get_ffmpeg_exe()
        stem = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        candidate = os.path.join(os.path.dirname(ff), stem)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    return None


# Back-compat alias used by older call sites / hot-reload.
_ffprobe_bin = ffprobe_bin


def _parse_rate(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _ffprobe_stream_info(path: str) -> dict | None:
    import json
    import subprocess

    probe = _ffprobe_bin()
    if not probe:
        return None
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration,nb_frames,r_frame_rate,avg_frame_rate",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            check=True,
        )
        payload = json.loads(res.stdout.decode("utf-8", "replace"))
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
        log.debug("ffprobe stream info failed for %s: %s", path, exc)
        return None
    streams = payload.get("streams") or []
    return streams[0] if streams else None


def peek_video_size(path: str) -> tuple[int, int]:
    """Width/height from container metadata. Does not decode frames or count them."""
    if not path or not os.path.isfile(path):
        return 0, 0
    try:
        stream = _ffprobe_stream_info(path)
        if stream:
            w = int(stream.get("width") or 0)
            h = int(stream.get("height") or 0)
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    try:
        meta = _opencv_probe(path)
        return int(meta.get("width") or 0), int(meta.get("height") or 0)
    except Exception:
        return 0, 0


def _ffprobe_count_frames(path: str) -> int | None:
    import subprocess

    probe = _ffprobe_bin()
    if not probe:
        return None
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            check=True,
        )
        text = res.stdout.decode("utf-8", "replace").strip()
        if text.isdigit():
            return int(text)
    except (subprocess.CalledProcessError, OSError) as exc:
        log.debug("ffprobe count_frames failed for %s: %s", path, exc)
    return None


def _opencv_probe(path: str) -> dict:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = float(frame_count / native_fps) if frame_count > 0 and native_fps > 0 else 0.0
        return {
            "width": width,
            "height": height,
            "duration": duration,
            "native_fps": native_fps,
            "frame_count": max(0, frame_count),
        }
    finally:
        cap.release()


def probe_video_file(path: str) -> dict:
    """Probe container metadata and an accurate frame count for Director UI."""
    if not path or not os.path.isfile(path):
        raise ValueError(f"Video file not found: {path}")

    method = "estimated"
    stream = _ffprobe_stream_info(path)
    opencv_meta = None

    width = height = 0
    duration = 0.0
    native_fps = 0.0
    frame_count = 0

    if stream:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration = float(stream.get("duration") or 0.0)
        native_fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
        nb_frames = str(stream.get("nb_frames") or "").strip()
        if nb_frames.isdigit() and int(nb_frames) > 0:
            frame_count = int(nb_frames)
            method = "ffprobe_nb_frames"

    if frame_count <= 0:
        counted = _ffprobe_count_frames(path)
        if counted is not None and counted > 0:
            frame_count = counted
            method = "ffprobe_count_frames"

    if frame_count <= 0 or width <= 0 or height <= 0 or native_fps <= 0:
        try:
            opencv_meta = _opencv_probe(path)
        except ImportError:
            opencv_meta = None
        if opencv_meta:
            width = width or int(opencv_meta["width"])
            height = height or int(opencv_meta["height"])
            native_fps = native_fps or float(opencv_meta["native_fps"])
            if frame_count <= 0 and int(opencv_meta["frame_count"]) > 0:
                frame_count = int(opencv_meta["frame_count"])
                method = "opencv"

    if duration <= 0 and frame_count > 0 and native_fps > 0:
        duration = frame_count / native_fps
    elif duration <= 0 and opencv_meta:
        duration = float(opencv_meta.get("duration") or 0.0)

    if frame_count <= 0 and duration > 0 and native_fps > 0:
        frame_count = max(1, int(round(duration * native_fps)))
        if method == "estimated":
            method = "duration_estimate"

    if frame_count <= 0:
        raise ValueError(f"Could not determine frame count for video: {path}")

    if native_fps <= 0:
        native_fps = 24.0

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "native_fps": native_fps,
        "frame_count": frame_count,
        "probe_method": method,
    }


def probe_video_clip(video: dict) -> dict:
    """Probe a timeline clip dict (videoFile / subfolder / type)."""
    return probe_video_file(resolve_video_path(video))


def load_video_resampled(
    path: str,
    frame_rate: float,
    frame_indices: Sequence[int],
    *,
    storage_width: int | None = None,
    storage_height: int | None = None,
    long_edge: int = 848,
) -> torch.Tensor:
    """Decode selected resampled frame indices from a video file."""
    if not frame_indices:
        raise ValueError("No frames requested from video.")

    cv2 = _require_cv2()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = float(frame_rate or 24.0)

    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out_w, out_h, rotate_90_cw = _resolve_load_dimensions(
        source_w,
        source_h,
        storage_width=storage_width,
        storage_height=storage_height,
        long_edge=long_edge,
    )

    unique = sorted({int(i) for i in frame_indices})
    decoded: dict[int, np.ndarray] = {}
    fallback: np.ndarray | None = None

    for src_idx in unique:
        t_sec = max(0.0, src_idx / float(frame_rate or 24.0))
        native_frame = int(round(t_sec * native_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, native_frame)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            log.warning("Failed to read frame %d (t=%.3fs) from %s", native_frame, t_sec, path)
            if fallback is not None:
                decoded[src_idx] = fallback
            continue

        if rotate_90_cw:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if (bgr.shape[1], bgr.shape[0]) != (out_w, out_h):
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        decoded[src_idx] = rgb
        fallback = rgb

    cap.release()

    if not decoded:
        raise ValueError(f"No frames decoded from video: {path}")

    rows = []
    last = next(iter(decoded.values()))
    for idx in frame_indices:
        rows.append(decoded.get(int(idx), last))
        last = rows[-1]

    return torch.from_numpy(np.stack(rows, axis=0))


def _aspect_ratio(w: int, h: int) -> float:
    return w / h if h > 0 else 0.0


def _aspect_close(a: float, b: float, *, tol: float = 0.04) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= tol


def _resolve_load_dimensions(
    source_w: int,
    source_h: int,
    *,
    storage_width: int | None,
    storage_height: int | None,
    long_edge: int,
) -> tuple[int, int, bool]:
    """Return (out_w, out_h, rotate_90_cw) for proportional long-edge loading."""
    if source_w <= 0 or source_h <= 0:
        out_w, out_h, _, _ = resolve_output_dimensions(
            source_w,
            source_h,
            mode="long_edge",
            long_edge=long_edge,
        )
        return out_w, out_h, False

    native_portrait = source_h > source_w
    native_landscape = source_w > source_h

    if storage_width and storage_height:
        sw, sh = int(storage_width), int(storage_height)
        storage_portrait = sh > sw
        storage_landscape = sw > sh
        native_ar = _aspect_ratio(source_w, source_h)
        storage_ar = _aspect_ratio(sw, sh)
        transposed_ar = _aspect_ratio(source_h, source_w)

        if _aspect_close(native_ar, storage_ar):
            return sw, sh, False

        # Portrait-native video must not be rotated into a landscape target (common when
        # browser metadata or node defaults supply 832脳480 for a vertical phone clip).
        if native_portrait and storage_landscape:
            log.info(
                "Video %dx%d portrait native vs storage %dx%d landscape; keeping orientation",
                source_w,
                source_h,
                sw,
                sh,
            )
            out_w, out_h, _, _ = resolve_output_dimensions(
                source_w,
                source_h,
                mode="long_edge",
                long_edge=long_edge,
            )
            return out_w, out_h, False

        # Landscape-native with portrait storage (rotation metadata in container).
        if native_landscape and storage_portrait and _aspect_close(transposed_ar, storage_ar):
            log.info(
                "Video %dx%d landscape native vs storage %dx%d portrait; applying 90掳 rotation",
                source_w,
                source_h,
                sw,
                sh,
            )
            return sw, sh, True

        if _aspect_close(transposed_ar, storage_ar):
            log.info(
                "Video %dx%d decoded transposed vs storage %dx%d; applying 90掳 rotation before scale",
                source_w,
                source_h,
                sw,
                sh,
            )
            return sw, sh, True

        log.warning(
            "storage %dx%d aspect mismatch vs native %dx%d; using proportional long_edge=%d",
            sw,
            sh,
            source_w,
            source_h,
            long_edge,
        )

    out_w, out_h, _, _ = resolve_output_dimensions(
        source_w,
        source_h,
        mode="long_edge",
        long_edge=long_edge,
    )
    return out_w, out_h, False


def parse_frame_map_entry(entry: Any, default_clip: int = 0) -> tuple[int, int]:
    """Parse a frameMap entry to (clip_index, source_frame_index)."""
    if isinstance(entry, dict):
        clip = int(entry.get("clip", entry.get("videoClip", default_clip)))
        frame = int(entry.get("frame", 0))
        return clip, frame
    return default_clip, int(entry)


def video_clips_from_timeline(timeline: dict) -> list[dict]:
    """Return ordered video clip metadata; falls back to legacy single ``video`` block."""
    clips = timeline.get("videoClips") or timeline.get("video_clips")
    if clips:
        return list(clips)
    video = timeline.get("video") or {}
    if (video.get("videoFile") or video.get("fileName") or "").strip():
        return [video]
    return []


def deleted_source_ranges(timeline: dict) -> list[tuple[int, int]]:
    """Source-frame spans removed from the logical timeline (sparse single-clip edits)."""
    video = timeline.get("video") or {}
    raw = video.get("deletedSourceRanges") or video.get("deleted_source_ranges") or []
    ranges: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = int(item[0]), int(item[1])
            if end > start:
                ranges.append((start, end))
    return sorted(ranges)


def logical_frame_map(timeline: dict) -> list[Any]:
    """Explicit per-logical-frame map only; empty means sparse identity mapping."""
    video = timeline.get("video") or {}
    frame_map = video.get("frameMap")
    if frame_map:
        return list(frame_map)
    return []


def logical_frame_count(timeline: dict) -> int:
    frame_map = logical_frame_map(timeline)
    if frame_map:
        return len(frame_map)

    total = int(timeline.get("totalFrames") or 0)
    if total > 0:
        return total

    video = timeline.get("video") or {}
    source_count = int(video.get("sourceFrameCount") or 0)
    if source_count > 0:
        removed = sum(end - start for start, end in deleted_source_ranges(timeline))
        return max(0, source_count - removed)

    clips = video_clips_from_timeline(timeline)
    if len(clips) > 1:
        return sum(int(c.get("sourceFrameCount") or 0) for c in clips)

    return 0


def resolve_logical_frame_entry(timeline: dict, logical_index: int) -> tuple[int, int]:
    """Map a logical timeline index to (clip_index, source_frame_index)."""
    video = timeline.get("video") or {}
    frame_map = video.get("frameMap") or []
    if logical_index < len(frame_map):
        return parse_frame_map_entry(frame_map[logical_index])

    src = logical_index
    for start, end in deleted_source_ranges(timeline):
        if src >= start:
            src += end - start
        else:
            break

    clips = video_clips_from_timeline(timeline)
    if len(clips) <= 1:
        return 0, src

    offset = 0
    for clip_idx, clip in enumerate(clips):
        count = int(clip.get("sourceFrameCount") or 0)
        if logical_index < offset + count:
            return clip_idx, logical_index - offset
        offset += count

    last = clips[-1]
    last_count = max(1, int(last.get("sourceFrameCount") or 1))
    return len(clips) - 1, last_count - 1


def frame_indices_from_timeline(timeline: dict) -> list[int]:
    """Legacy helper: source-frame indices for single-clip timelines."""
    total = logical_frame_count(timeline)
    entries = [resolve_logical_frame_entry(timeline, i) for i in range(total)]
    if entries and all(c == 0 for c, _ in entries):
        return [f for _, f in entries]
    return list(range(total))


def _decode_timeline_entries(
    timeline: dict,
    entries: list[tuple[int, int]],
    *,
    frame_rate: float,
    default_long_edge: int,
) -> torch.Tensor:
    clips = video_clips_from_timeline(timeline)
    if not clips:
        raise ValueError("No video clips in MiniMax H3 Director timeline.")

    by_clip: dict[int, set[int]] = defaultdict(set)
    for clip_idx, frame_idx in entries:
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        by_clip[clip_idx].add(frame_idx)

    frame_tensors: dict[tuple[int, int], torch.Tensor] = {}
    for clip_idx, frame_set in sorted(by_clip.items()):
        clip = clips[clip_idx]
        path = resolve_video_path(clip)
        sorted_idx = sorted(frame_set)
        tensor = load_video_resampled(
            path,
            frame_rate,
            sorted_idx,
            storage_width=clip.get("storageWidth"),
            storage_height=clip.get("storageHeight"),
            long_edge=int(clip.get("longEdge") or default_long_edge),
        )
        for row, fi in enumerate(sorted_idx):
            frame_tensors[(clip_idx, fi)] = tensor[row]

    rows: list[torch.Tensor] = []
    fallback: torch.Tensor | None = None
    for clip_idx, frame_idx in entries:
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        key = (clip_idx, frame_idx)
        tensor = frame_tensors.get(key, fallback)
        if tensor is None:
            raise ValueError(f"Missing decoded frame for clip {clip_idx} frame {frame_idx}")
        fallback = tensor
        rows.append(tensor)

    return torch.stack(rows, dim=0)


def load_reference_video_clip(
    ref_block: dict,
    timeline: dict,
    num_frames: int,
    *,
    start_frame: int = 0,
) -> torch.Tensor | None:
    """Load an ads2v reference video clip, resampled to ``num_frames`` at timeline FPS.

    ``start_frame`` is the logical timeline offset (0 = from beginning). Used when
    global *continuous reference* is enabled so segment N uses ref frame N onward.
    """
    if not (ref_block.get("videoFile") or ref_block.get("fileName") or "").strip():
        return None

    path = resolve_video_path(ref_block)
    frame_rate = float(timeline.get("frameRate") or 24)
    output_block = timeline.get("output") or {}
    long_edge = int(
        output_block.get("longEdge")
        or output_block.get("long_edge")
        or timeline.get("refMaxSize")
        or 848
    )
    count = max(1, int(num_frames))
    offset = max(0, int(start_frame))
    frame_indices = list(range(offset, offset + count))
    return load_video_resampled(
        path,
        frame_rate,
        frame_indices,
        storage_width=ref_block.get("storageWidth"),
        storage_height=ref_block.get("storageHeight"),
        long_edge=long_edge,
    )


def load_timeline_segment(timeline: dict, start: int, end: int) -> torch.Tensor:
    """Decode only logical frames in [start, end) 鈥?supports arbitrarily long timelines."""
    total = logical_frame_count(timeline)
    start = max(0, min(int(start), total))
    end = max(start, min(int(end), total))
    if start >= end:
        raise ValueError(f"No frames in timeline range [{start}, {end})")

    video = timeline.get("video") or {}
    frames_b64 = video.get("frames") or []
    if frames_b64:
        chunks: list[torch.Tensor] = []
        for frame_b64 in frames_b64[start:end]:
            chunks.append(_decode_image_b64_inline(frame_b64))
        if not chunks:
            raise ValueError("Uploaded video has no decodable frames in range.")
        return torch.cat(chunks, dim=0)

    frame_rate = float(timeline.get("frameRate") or 24)
    output_block = timeline.get("output") or {}
    default_long_edge = int(
        output_block.get("longEdge")
        or output_block.get("long_edge")
        or timeline.get("refMaxSize")
        or 848
    )

    entries = [resolve_logical_frame_entry(timeline, i) for i in range(start, end)]
    return _decode_timeline_entries(
        timeline,
        entries,
        frame_rate=frame_rate,
        default_long_edge=default_long_edge,
    )


def _decode_image_b64_inline(b64_str: str) -> torch.Tensor:
    import base64
    import io

    from PIL import Image

    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_multi_clip_timeline(
    timeline: dict,
    frame_map: list[Any],
    *,
    frame_rate: float,
    default_long_edge: int,
) -> torch.Tensor:
    """Decode a logical timeline that may reference multiple source videos."""
    entries = [parse_frame_map_entry(e) for e in frame_map]
    return _decode_timeline_entries(
        timeline,
        entries,
        frame_rate=frame_rate,
        default_long_edge=default_long_edge,
    )
