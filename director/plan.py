"""Parse MiniMax H3 Director timeline JSON and prepare per-segment edit plans."""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

import folder_paths

from ..lib.audio_io import load_reference_audio
from ..lib.ref_audios import MAX_REFERENCE_AUDIOS, ref_audios_dict
from ..lib.ref_images import MAX_REFERENCE_IMAGES, REF_IMAGE_KEY_PREFIX
from ..lib.ref_videos import MAX_REFERENCE_VIDEOS, ref_videos_dict
from ..lib.image_prep import assert_minimax_canvas, resolve_output_dimensions
from ..lib.task_prompts import get_task_prompt_spec, resolve_task_key
from ..lib.video_io import (
    load_reference_video_clip,
    logical_frame_count,
    logical_frame_map,
    load_timeline_segment,
    video_clips_from_timeline,
)
from .gen_timeline import (
    build_gen_director_plan,
    is_gen_timeline,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

MIN_SEGMENT_FRAMES = 4
DEFAULT_CONTINUITY_OVERLAP = 22

# Same idea as Comfy LoadImage execution cache: skip re-decode when the file is unchanged.
_REF_IMAGE_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_REF_IMAGE_CACHE_MAX = 16


def _ref_image_cache_key(file_path: str) -> tuple | None:
    try:
        st = os.stat(file_path)
    except OSError:
        return None
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    return (file_path, int(st.st_size), mtime_ns)


def _ref_image_cache_get(key: tuple) -> torch.Tensor | None:
    hit = _REF_IMAGE_CACHE.get(key)
    if hit is None:
        return None
    _REF_IMAGE_CACHE.move_to_end(key)
    return hit


def _ref_image_cache_put(key: tuple, tensor: torch.Tensor) -> None:
    _REF_IMAGE_CACHE[key] = tensor
    _REF_IMAGE_CACHE.move_to_end(key)
    while len(_REF_IMAGE_CACHE) > _REF_IMAGE_CACHE_MAX:
        _REF_IMAGE_CACHE.popitem(last=False)


MIN_CONTINUITY_OVERLAP = 5
MAX_CONTINUITY_OVERLAP = 56
REF_IMAGE_SIZE_MATCH = "match"
REF_IMAGE_SIZE_MAX = "max"


def normalize_ref_image_size(value) -> str:
    raw = str(value or "").strip().lower()
    return REF_IMAGE_SIZE_MAX if raw == REF_IMAGE_SIZE_MAX else REF_IMAGE_SIZE_MATCH


def _timeline_dict(plan_or_timeline) -> dict:
    if isinstance(plan_or_timeline, dict):
        return plan_or_timeline
    raw = getattr(plan_or_timeline, "raw", None)
    return raw if isinstance(raw, dict) else {}


def _legacy_output_ref_image_size(timeline: dict | None) -> str | None:
    out = (timeline or {}).get("output") or {}
    raw = out.get("refImageSize")
    if raw is None:
        raw = out.get("ref_image_size")
    if raw is None or str(raw).strip() == "":
        return None
    return normalize_ref_image_size(raw)


def resolve_ref_image_size(seg_or_data=None, plan_or_timeline=None) -> str:
    """Per-segment MiniMax ``ref_image_size``; legacy ``output.refImageSize`` as fallback."""
    raw = None
    if isinstance(seg_or_data, dict):
        if "refImageSize" in seg_or_data or "ref_image_size" in seg_or_data:
            raw = seg_or_data.get("refImageSize")
            if raw is None:
                raw = seg_or_data.get("ref_image_size")
    elif seg_or_data is not None:
        raw = getattr(seg_or_data, "ref_image_size", None)
    if raw is not None and str(raw).strip() != "":
        return normalize_ref_image_size(raw)
    legacy = _legacy_output_ref_image_size(_timeline_dict(plan_or_timeline))
    return legacy if legacy is not None else REF_IMAGE_SIZE_MATCH


@dataclass
class SegmentRef:
    index: int
    tensor: torch.Tensor
    image_file: str = ""


@dataclass
class SegmentRefAudio:
    """Standalone reference audio for MiniMax ``<Audio N>`` (index 0-based)."""

    index: int
    audio: dict  # ComfyUI AUDIO: {waveform, sample_rate}
    audio_file: str = ""


@dataclass
class SegmentRefVideo:
    """Standalone reference video for MiniMax ``<Video N>`` (index 0-based)."""

    index: int
    tensor: torch.Tensor
    video_file: str = ""
    meta: dict = field(default_factory=dict)


def concat_common_segment_prompt(common: str | None, segment: str | None) -> str:
    """Join shared (common) prompt with per-group prompt.

    Both non-empty → ``common + blank line + segment``.
    Only one side → that side alone (replaces legacy ``segment or common`` fallback).
    """
    common_s = (common or "").strip()
    segment_s = (segment or "").strip()
    if common_s and segment_s:
        return f"{common_s}\n\n{segment_s}"
    return common_s or segment_s


def merge_indexed_refs(common: list, segment: list) -> list:
    """Merge common + per-group refs by slot index; segment wins on conflict."""
    by_idx: dict[int, object] = {}
    for item in common or []:
        by_idx[int(getattr(item, "index", 0))] = item
    for item in segment or []:
        by_idx[int(getattr(item, "index", 0))] = item
    return sorted(by_idx.values(), key=lambda r: int(getattr(r, "index", 0)))


@dataclass
class SegmentPlan:
    index: int
    start_frame: int
    end_frame: int
    prompt: str
    task_type: str
    task_key: str
    use_global: bool
    refs: list[SegmentRef] = field(default_factory=list)
    ref_audios: list[SegmentRefAudio] = field(default_factory=list)
    ref_videos: list[SegmentRefVideo] = field(default_factory=list)
    ref_video_audios: list[SegmentRefAudio] = field(default_factory=list)
    reference_video_meta: dict = field(default_factory=dict)
    reference_video_start_frame: int = 0
    negative_prompt: str = ""
    source_clip: torch.Tensor | None = None
    # When external groups filter by「选择运行」, plan.index is the compact run
    # order (0..N-1) while ui_index keeps the Director timeline card index.
    ui_index: int | None = None
    # Per-segment「引用上段」; master「段间引导」must also be on. Default True.
    continuity_from_prev: bool = True
    # Official MiniMaxH3ReferenceToVideo combo: match | max. Per r2v/rv2v group.
    ref_image_size: str = "match"

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    @property
    def timeline_index(self) -> int:
        """Index used for UI preview / highlight (timeline card)."""
        return int(self.index if self.ui_index is None else self.ui_index)


@dataclass
class DirectorPlan:
    frame_rate: float
    total_frames: int
    width: int
    height: int
    ref_max_size: int
    output_mode: str
    source_width: int
    source_height: int
    global_task_type: str
    global_task_key: str
    global_prompt: str
    global_refs: list[SegmentRef]
    segments: list[SegmentPlan]
    source_video: torch.Tensor
    edit_mode: str
    raw: dict
    source_total_frames: int = 0
    export_max_frames: int = 0
    export_mode: str = "all"  # "all" | "segments"
    run_indices: frozenset[int] | None = None  # None = run all segments
    continuity_enabled: bool = False
    continuity_overlap_frames: int = 0
    global_ref_audios: list[SegmentRefAudio] = field(default_factory=list)
    refine: dict | None = None
    # Sampling knobs stamped at execute time (first-pass cache fingerprint).
    sample_seed: int = 0
    sample_cfg: float = 1.0
    sample_steps: int = 25
    sample_sampler: str = ""
    sample_scheduler: str = ""
    sample_sigmas: tuple[float, ...] | None = None
    sample_sigmas_linked: bool = False
    sample_shift_video: float = 12.0
    sample_shift_audio: float = 3.0

    @property
    def segment_count(self) -> int:
        return len(self.segments)


def _ref_video_has_file(ref_block: dict | None) -> bool:
    if not ref_block:
        return False
    return bool((ref_block.get("videoFile") or ref_block.get("fileName") or "").strip())


def _continuous_reference_enabled(timeline: dict, edit_mode: str, task_key: str) -> bool:
    """Global ads2v only: align reference video timeline offset with each segment start."""
    if edit_mode != "global" or task_key != "ads2v":
        return False
    global_block = timeline.get("global") or {}
    return bool(
        global_block.get("continuousReference")
        or global_block.get("continuous_reference")
        or timeline.get("continuousReference")
        or timeline.get("continuous_reference")
    )


def _resolve_global_reference_video(timeline: dict) -> dict:
    global_block = timeline.get("global") or {}
    ref = global_block.get("referenceVideo") or global_block.get("reference_video") or {}
    if _ref_video_has_file(ref):
        return dict(ref)
    legacy = timeline.get("referenceVideo") or timeline.get("reference_video") or {}
    return dict(legacy) if isinstance(legacy, dict) else {}


from .frame_align import minimax_align_frame_count as wan_align_frame_count


def _decode_image_b64(b64_str: str) -> torch.Tensor:
    if not b64_str:
        raise ValueError("Empty image data.")
    if b64_str.startswith("/view?"):
        raise ValueError("Remote view URLs are not supported; upload images in the Director node.")
    payload = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
    img_bytes = base64.b64decode(payload)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_reference_tensor(ref: dict) -> torch.Tensor | None:
    if ref.get("imageFile"):
        rel = str(ref["imageFile"]).replace("\\", "/")
        file_path = os.path.join(folder_paths.get_input_directory(), rel.replace("/", os.sep))
        if os.path.exists(file_path):
            key = _ref_image_cache_key(file_path)
            cached = _ref_image_cache_get(key) if key else None
            if cached is not None:
                return cached
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)
            try:
                import comfy.model_management as mm

                tensor = tensor.to(device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
            except Exception:
                pass
            if key:
                _ref_image_cache_put(key, tensor)
            return tensor

    b64_str = ref.get("imageB64", "")
    if not b64_str:
        return None
    try:
        return _decode_image_b64(b64_str)
    except Exception as exc:
        log.warning("Failed to decode reference image: %s", exc)
        return None


def load_source_video_from_timeline(timeline: dict) -> torch.Tensor:
    """Load all logical frames (legacy). Prefer load_timeline_segment for long videos."""
    total = logical_frame_count(timeline)
    if total <= 0:
        video = timeline.get("video") or {}
        if not (video.get("frames") or []):
            raise ValueError("No frames in MiniMax H3 Director timeline.")
    return load_timeline_segment(timeline, 0, max(1, total))


def _load_refs(ref_list: list[dict]) -> list[SegmentRef]:
    refs: list[SegmentRef] = []
    for item in ref_list or []:
        index = int(item.get("index", item.get("slot", len(refs))))
        if index < 0 or index >= MAX_REFERENCE_IMAGES:
            continue
        tensor = load_reference_tensor(item)
        if tensor is not None:
            image_file = str(
                item.get("imageFile") or item.get("image_file") or item.get("fileName") or ""
            ).replace("\\", "/").strip()
            refs.append(SegmentRef(index=index, tensor=tensor, image_file=image_file))
    return sorted(refs, key=lambda r: r.index)


def load_reference_audio_item(item: dict) -> dict | None:
    """Load one timeline refAudios entry into a ComfyUI AUDIO dict."""
    rel = str(
        item.get("audioFile")
        or item.get("audio_file")
        or item.get("fileName")
        or item.get("file_name")
        or ""
    ).replace("\\", "/").strip()
    if not rel:
        return None
    sub = str(item.get("subfolder") or "").replace("\\", "/").strip().strip("/")
    if sub and not rel.startswith(sub + "/"):
        rel = f"{sub}/{rel}"
    file_path = os.path.join(folder_paths.get_input_directory(), rel.replace("/", os.sep))
    if not os.path.isfile(file_path):
        log.warning("Reference audio missing: %s", file_path)
        return None
    audio = load_reference_audio(file_path)
    if audio is None:
        log.warning("Failed to decode reference audio: %s", file_path)
    return audio


def _load_ref_audios(audio_list: list[dict]) -> list[SegmentRefAudio]:
    out: list[SegmentRefAudio] = []
    for item in audio_list or []:
        if not isinstance(item, dict):
            continue
        index = int(item.get("index", item.get("slot", len(out))))
        if index < 0 or index >= MAX_REFERENCE_AUDIOS:
            continue
        audio = load_reference_audio_item(item)
        if audio is None:
            continue
        rel = str(item.get("audioFile") or item.get("audio_file") or item.get("fileName") or "").strip()
        out.append(SegmentRefAudio(index=index, audio=audio, audio_file=rel))
    return sorted(out, key=lambda a: a.index)


def segment_ref_audios_for_context(task_key: str, audios: list[SegmentRefAudio]) -> list[SegmentRefAudio]:
    """Standalone ref audios apply to r2v / rv2v (official ReferenceToVideo)."""
    if task_key not in {"r2v", "rv2v"}:
        return []
    return audios


def ref_audios_to_dict(audios: list[SegmentRefAudio]) -> dict | None:
    return ref_audios_dict([(a.index, a.audio) for a in audios])


def _ref_video_entry_has_file(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    return bool((item.get("videoFile") or item.get("fileName") or "").strip())


def _load_ref_videos(
    video_list: list[dict],
    timeline: dict,
    num_frames: int,
) -> list[SegmentRefVideo]:
    """Load up to 3 standalone reference videos for r2v / ReferenceToVideo."""
    out: list[SegmentRefVideo] = []
    for item in video_list or []:
        if not isinstance(item, dict) or not _ref_video_entry_has_file(item):
            continue
        index = int(item.get("index", item.get("slot", len(out))))
        if index < 0 or index >= MAX_REFERENCE_VIDEOS:
            continue
        try:
            tensor = load_reference_video_clip(item, timeline, num_frames, start_frame=0)
        except Exception as exc:
            log.warning("Failed to load reference video slot %s: %s", index, exc)
            continue
        if tensor is None or tensor.numel() <= 0:
            continue
        rel = str(item.get("videoFile") or item.get("fileName") or "").strip()
        out.append(SegmentRefVideo(index=index, tensor=tensor, video_file=rel, meta=dict(item)))
    return sorted(out, key=lambda v: v.index)


def ref_videos_to_dict(videos: list[SegmentRefVideo]) -> dict | None:
    return ref_videos_dict([(v.index, v.tensor) for v in videos])


def reinforce_r2v_prompt(
    prompt: str,
    *,
    ref_indices: list[int] | None = None,
    video_indices: list[int] | None = None,
    audio_indices: list[int] | None = None,
) -> str:
    """Remind <Picture N> / <Video K> / <Audio J> when tags are missing (r2v batch)."""
    text = (prompt or "").strip() or "Generate a cinematic scene."
    pic_indices = sorted({int(i) for i in (ref_indices or []) if int(i) >= 0})
    vid_indices = sorted({int(i) for i in (video_indices or []) if int(i) >= 0})
    aud_indices = sorted({int(i) for i in (audio_indices or []) if int(i) >= 0})
    prefix_parts: list[str] = []
    if pic_indices and "<Picture" not in text and "<picture" not in text:
        prefix_parts.append(" ".join(f"<Picture {i + 1}>" for i in pic_indices))
    if vid_indices and "<Video" not in text and "<video" not in text:
        prefix_parts.append(" ".join(f"<Video {i + 1}>" for i in vid_indices))
    if aud_indices and "<Audio" not in text and "<audio" not in text:
        prefix_parts.append(" ".join(f"<Audio {i + 1}>" for i in aud_indices))
    if not prefix_parts:
        return text
    return f"{' '.join(prefix_parts)} {text}"


def _segment_ranges_from_timeline(timeline: dict, total: int) -> list[tuple[int, int, dict]]:
    segments = timeline.get("segments") or []
    if segments and ("length" in segments[0] or "end" in segments[0]):
        ranges: list[tuple[int, int, dict]] = []
        for raw in sorted(segments, key=lambda s: int(s.get("start", 0))):
            start = int(raw.get("start", 0))
            if "end" in raw:
                end = int(raw["end"])
            else:
                end = start + int(raw.get("length", 0))
            start = max(0, min(start, total))
            end = max(start, min(end, total))
            if end - start >= MIN_SEGMENT_FRAMES or not ranges:
                ranges.append((start, end, raw))
        if ranges:
            return ranges

    split_points = timeline.get("splitPoints") or timeline.get("split_points") or []
    auto_count = int(timeline.get("autoSegmentCount") or timeline.get("auto_segment_count") or 0)
    if auto_count > 1:
        points = [int(round(total * i / auto_count)) for i in range(1, auto_count)]
    else:
        points = sorted({int(p) for p in split_points if 0 < int(p) < total})

    edges = [0] + points + [total]
    ranges = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if end <= start:
            continue
        raw = segments[i] if i < len(segments) else {}
        ranges.append((start, end, raw))
    return ranges or [(0, total, {})]


def _resolve_export_total(timeline: dict, source_total: int) -> int:
    output_block = timeline.get("output") or {}
    max_export = int(output_block.get("maxExportFrames") or output_block.get("max_export_frames") or 0)
    if max_export <= 0 or source_total <= 0:
        return source_total
    return min(source_total, max_export)


def _resolve_export_mode(output_block: dict) -> str:
    mode = str(output_block.get("exportMode") or output_block.get("export_mode") or "all").lower()
    if mode in ("segments", "segment", "per_segment", "by_segment"):
        return "segments"
    return "all"


def _clip_segment_ranges(
    ranges: list[tuple[int, int, dict]], export_total: int
) -> list[tuple[int, int, dict]]:
    if export_total <= 0:
        return ranges
    clipped: list[tuple[int, int, dict]] = []
    for start, end, data in ranges:
        if start >= export_total:
            break
        end = min(end, export_total)
        if end <= start:
            continue
        if end - start < MIN_SEGMENT_FRAMES and clipped:
            ps, _, pd = clipped[-1]
            clipped[-1] = (ps, end, pd)
        else:
            clipped.append((start, end, data))
    if not clipped and export_total > 0:
        data = ranges[0][2] if ranges else {}
        clipped.append((0, export_total, data))
    return clipped


def _trim_timeline_for_export(timeline: dict, export_total: int) -> dict:
    t = copy.deepcopy(timeline)
    video = dict(t.get("video") or {})
    frames_b64 = video.get("frames") or []
    if frames_b64 and export_total < len(frames_b64):
        video["frames"] = frames_b64[:export_total]
    frame_map = video.get("frameMap") or []
    if frame_map and export_total < len(frame_map):
        video["frameMap"] = frame_map[:export_total]
    t["video"] = video
    t["totalFrames"] = export_total
    return t


def _parse_run_selection(timeline: dict, segment_count: int) -> frozenset[int] | None:
    """Return selected segment indices, or None when all segments should run."""
    enabled = bool(timeline.get("runSelectEnabled") or timeline.get("run_select_enabled"))
    if not enabled:
        return None
    raw = timeline.get("runSelection")
    if raw is None:
        raw = timeline.get("run_selection")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    indices = {int(i) for i in raw if 0 <= int(i) < segment_count}
    if not indices:
        raise ValueError(
            "MiniMax H3 Director: 「选择运行」已开启但未勾选任何片段/提示词组。请至少勾选一组再执行。"
        )
    if len(indices) >= segment_count:
        return None
    return frozenset(indices)


def count_all_timeline_segments(timeline_data: str) -> int:
    """Total segment count on the timeline (ignores run selection)."""
    if not timeline_data or not str(timeline_data).strip():
        return 1
    try:
        timeline = json.loads(timeline_data)
    except json.JSONDecodeError:
        return 1

    segments = timeline.get("segments") or []
    global_task = (timeline.get("global") or {}).get("taskType") or ""
    task_key = resolve_task_key(global_task) if global_task else ""
    if task_key == "fl2v" or str(timeline.get("timelineMode") or "").lower() == "fl2v":
        from .fl2v_timeline import count_fl2v_runnable_shots

        return count_fl2v_runnable_shots(timeline)
    if is_gen_timeline(timeline, task_key):
        return max(1, len(segments) or 1)

    source_total = logical_frame_count(timeline) or int(timeline.get("totalFrames") or 0)
    export_total = _resolve_export_total(timeline, source_total)
    plan_total = export_total or source_total or 1
    ranges = _segment_ranges_from_timeline(timeline, source_total or plan_total)
    return max(1, len(_clip_segment_ranges(ranges, plan_total)))


def count_timeline_segments(timeline_data: str) -> int:
    """Segments that will run (respects run selection when enabled)."""
    if not timeline_data or not str(timeline_data).strip():
        return 1
    try:
        timeline = json.loads(timeline_data)
    except json.JSONDecodeError:
        return 1

    seg_count = count_all_timeline_segments(timeline_data)
    run_sel = _parse_run_selection(timeline, seg_count)
    return len(run_sel) if run_sel is not None else seg_count


def build_director_plan(
    timeline_data: str,
    *,
    global_task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> DirectorPlan:
    timeline: dict = {}
    if timeline_data and timeline_data.strip():
        try:
            timeline = json.loads(timeline_data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid timeline_data JSON: {exc}") from exc

    global_block = timeline.get("global") or {}
    edit_mode = timeline.get("editMode") or timeline.get("edit_mode") or "global"
    if edit_mode not in ("global", "segment"):
        edit_mode = "global"

    task_type = global_block.get("taskType") or global_task_type or "v2v — 视频转视频(Video to Video)"
    prompt = global_block.get("prompt") or global_prompt or ""
    global_refs = _load_refs(global_block.get("refs") or [])
    global_ref_audios = _load_ref_audios(
        global_block.get("refAudios") or global_block.get("ref_audios") or []
    )
    global_ref_video = _resolve_global_reference_video(timeline)

    task_key_early = resolve_task_key(task_type)
    if task_key_early == "fl2v" or str(timeline.get("timelineMode") or "").lower() == "fl2v":
        from .fl2v_timeline import build_fl2v_director_plan

        return build_fl2v_director_plan(
            timeline,
            global_task_type=task_type,
            global_prompt=prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )
    if is_gen_timeline(timeline, task_key_early):
        return build_gen_director_plan(
            timeline,
            global_task_type=task_type,
            global_prompt=prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )

    frame_map = logical_frame_map(timeline)
    source_total = logical_frame_count(timeline) or int(timeline.get("totalFrames") or total_frames or 0)
    export_max = int(
        (timeline.get("output") or {}).get("maxExportFrames")
        or (timeline.get("output") or {}).get("max_export_frames")
        or 0
    )
    export_total = _resolve_export_total(timeline, source_total)

    load_timeline = _trim_timeline_for_export(timeline, export_total) if export_total < source_total else timeline

    clips = video_clips_from_timeline(load_timeline)
    if not clips and not (load_timeline.get("video") or {}).get("frames"):
        raise ValueError(
            "No source video in MiniMax H3 Director. Upload a video inside the node timeline UI before running."
        )

    try:
        probe = load_timeline_segment(load_timeline, 0, 1)
        loaded_h = int(probe.shape[1])
        loaded_w = int(probe.shape[2])
    except Exception as exc:
        log.warning("Could not probe source video frame: %s", exc)
        video_meta = load_timeline.get("video") or {}
        loaded_w = int(video_meta.get("width") or width)
        loaded_h = int(video_meta.get("height") or height)

    source_video = torch.zeros(0, max(1, loaded_h), max(1, loaded_w), 3)
    video_meta = timeline.get("video") or {}
    meta_w = int(video_meta.get("width") or 0)
    meta_h = int(video_meta.get("height") or 0)

    output_block = timeline.get("output") or {}
    export_mode = _resolve_export_mode(output_block)
    out_w, out_h, ref_max, output_mode = resolve_output_dimensions(
        loaded_w or meta_w or int(width),
        loaded_h or meta_h or int(height),
        mode=str(output_block.get("mode") or "long_edge"),
        long_edge=int(output_block.get("longEdge") or output_block.get("long_edge") or ref_max_size or 848),
        fixed_width=int(output_block.get("width") or timeline.get("width") or width),
        fixed_height=int(output_block.get("height") or timeline.get("height") or height),
    )
    assert_minimax_canvas(out_w, out_h)

    total = int(load_timeline.get("totalFrames") or export_total or total_frames or 0)
    if total <= 0:
        total = source_total

    segment_ranges = _segment_ranges_from_timeline(timeline, source_total or total)
    segment_ranges = _clip_segment_ranges(segment_ranges, total)
    segments: list[SegmentPlan] = []
    continuous_ref = _continuous_reference_enabled(timeline, edit_mode, resolve_task_key(task_type))

    for idx, (start, end, seg_data) in enumerate(segment_ranges):
        if edit_mode == "global":
            seg_prompt = prompt
            seg_task = task_type
            seg_refs = list(global_refs)
            seg_ref_audios = list(global_ref_audios)
            seg_ref_video = dict(global_ref_video)
            use_global = True
        else:
            use_global = False
            seg_prompt = (seg_data.get("prompt") or "").strip() or prompt
            seg_task = seg_data.get("taskType") or seg_data.get("task_type") or task_type
            # Segment mode: only this segment's refs — never inherit global.refs / refAudios.
            seg_refs = _load_refs(seg_data.get("refs") or [])
            seg_ref_audios = _load_ref_audios(
                seg_data.get("refAudios") or seg_data.get("ref_audios") or []
            )
            seg_ref_video = dict(seg_data.get("referenceVideo") or seg_data.get("reference_video") or {})

        seg_task_key = resolve_task_key(seg_task)
        seg_refs = segment_refs_for_context(seg_task_key, seg_refs)
        seg_ref_audios = segment_ref_audios_for_context(seg_task_key, seg_ref_audios)
        ref_start = start if continuous_ref and seg_task_key == "ads2v" else 0

        segments.append(
            SegmentPlan(
                index=idx,
                start_frame=start,
                end_frame=end,
                prompt=seg_prompt,
                task_type=seg_task,
                task_key=seg_task_key,
                use_global=use_global,
                refs=seg_refs,
                ref_audios=seg_ref_audios,
                reference_video_meta=seg_ref_video,
                reference_video_start_frame=ref_start,
            )
        )

    for seg in segments:
        if seg.task_key != "ads2v":
            continue
        if _ref_video_has_file(seg.reference_video_meta):
            continue
        raise ValueError(
            f"ads2v (广告植入) segment #{seg.index + 1} requires a reference video. "
            "Upload the content-to-insert clip for this segment in the Director node UI."
        )

    from .segment_continuity import (
        resolve_continuity_settings,
        resolve_segment_continuity_from_prev,
    )

    continuity_enabled, continuity_overlap = resolve_continuity_settings(
        timeline, segment_count=len(segments)
    )
    for seg, (_start, _end, seg_data) in zip(segments, segment_ranges):
        seg.continuity_from_prev = resolve_segment_continuity_from_prev(
            seg_data if isinstance(seg_data, dict) else {},
            segment_index=seg.index,
        )
        seg.ref_image_size = resolve_ref_image_size(
            seg_data if isinstance(seg_data, dict) else {},
            load_timeline,
        )

    return DirectorPlan(
        frame_rate=float(timeline.get("frameRate") or frame_rate or 24),
        total_frames=total,
        width=out_w,
        height=out_h,
        ref_max_size=ref_max,
        output_mode=output_mode,
        source_width=int(meta_w or loaded_w),
        source_height=int(meta_h or loaded_h),
        global_task_type=task_type,
        global_task_key=resolve_task_key(task_type),
        global_prompt=prompt,
        global_refs=global_refs,
        segments=segments,
        source_video=source_video,
        edit_mode=edit_mode,
        raw=load_timeline,
        source_total_frames=source_total or total,
        export_max_frames=export_max,
        export_mode=export_mode,
        run_indices=_parse_run_selection(timeline, len(segments)),
        continuity_enabled=continuity_enabled,
        continuity_overlap_frames=continuity_overlap,
        global_ref_audios=global_ref_audios,
    )


def slice_video_frames(source: torch.Tensor, start: int, end: int) -> torch.Tensor:
    end = min(end, source.shape[0])
    start = max(0, min(start, end))
    return source[start:end].clone()


def prepare_segment_clip(clip: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, int]:
    """Trim source toward MiniMax 17k+5 length. Do **not** pad with last-frame copies.

    Fabricating freeze frames in the source makes Bernini/Wan reproduce visible
    stutter / duplicate frames. Official BerniniConditioning simply encodes
    ``source[:length]`` even when the clip is shorter than ``length``.
    """
    actual = clip.shape[0]
    if actual <= 0:
        raise ValueError("Segment has no frames.")
    num_frames = wan_align_frame_count(max(actual, target_frames))
    if actual > num_frames:
        clip = clip[:num_frames]
    return clip, num_frames


# i2v/fl2v use keyframes; v2v uses source clip as <Video 1>; r2v uses ref_images.
CONTEXT_REFERENCE_EXCLUDED_KEYS = frozenset({"i2v", "fl2v", "t2v", "v2v"})


def segment_refs_for_context(task_key: str, refs: list[SegmentRef]) -> list[SegmentRef]:
    if task_key in CONTEXT_REFERENCE_EXCLUDED_KEYS:
        return []
    return refs


def refs_to_kwargs(refs: list[SegmentRef]) -> dict[str, torch.Tensor]:
    return {f"{REF_IMAGE_KEY_PREFIX}{ref.index}": ref.tensor for ref in refs}


def reinforce_v2v_prompt(prompt: str) -> str:
    """Ensure MiniMax ReferenceToVideo sees an explicit <Video 1> tag for source edit."""
    text = (prompt or "").strip()
    if not text:
        return "Edit <Video 1>."
    if "<Video" in text or "<video" in text:
        return text
    return f"<Video 1> {text}"


def reinforce_rv2v_prompt(
    prompt: str,
    *,
    ref_indices: list[int] | None = None,
    audio_indices: list[int] | None = None,
) -> str:
    """Source <Video 1> + remind <Picture N> / <Audio J> when tags are missing."""
    text = reinforce_v2v_prompt(prompt)
    pic_indices = sorted({int(i) for i in (ref_indices or []) if int(i) >= 0})
    aud_indices = sorted({int(i) for i in (audio_indices or []) if int(i) >= 0})
    prefix_parts: list[str] = []
    if pic_indices and "<Picture" not in text and "<picture" not in text:
        prefix_parts.append(" ".join(f"<Picture {i + 1}>" for i in pic_indices))
    if aud_indices and "<Audio" not in text and "<audio" not in text:
        prefix_parts.append(" ".join(f"<Audio {i + 1}>" for i in aud_indices))
    if not prefix_parts:
        return text
    return f"{' '.join(prefix_parts)} {text}"


def reference_video_for_segment(plan: DirectorPlan, seg: SegmentPlan, num_frames: int) -> torch.Tensor | None:
    """Optional separate reference video for r2v (not used by v2v — source clip is the ref)."""
    if seg.task_key != "r2v":
        return None
    if not _ref_video_has_file(seg.reference_video_meta):
        return None
    return load_reference_video_clip(
        seg.reference_video_meta,
        plan.raw,
        num_frames,
        start_frame=seg.reference_video_start_frame,
    )


def refs_to_kwargs_for_context(task_key: str, refs: list[SegmentRef]) -> dict[str, torch.Tensor]:
    return refs_to_kwargs(segment_refs_for_context(task_key, refs))


def plan_summary(plan: DirectorPlan) -> str:
    mode = str(plan.raw.get("timelineMode") or "")
    if mode in ("gen_blank", "gen_image", "prompt_batch", "image_batch", "fl2v"):
        if mode == "fl2v":
            mode_label = "首尾帧 (fl2v)"
        elif mode in ("prompt_batch", "image_batch"):
            mode_label = f"批量生成 ({plan.global_task_key})"
        else:
            mode_label = "空白画布" if mode == "gen_blank" else "图片生成"
        lines = [
            f"MiniMax H3 Director [{mode_label}] ({plan.edit_mode}): "
            f"{plan.segment_count} segment(s), {plan.total_frames} frames @ {plan.frame_rate:.2f} fps",
            f"Output: {plan.width}×{plan.height} ({plan.output_mode})",
            f"Global task: {get_task_prompt_spec(plan.global_task_type).label}",
        ]
        refine_line = None
        try:
            from .refine_pack import refine_report_line

            refine_line = refine_report_line(plan)
        except Exception:
            refine_line = None
        if refine_line:
            lines.append(refine_line)
        if plan.continuity_enabled:
            pinned = [
                seg.index + 1
                for seg in plan.segments
                if seg.index > 0 and getattr(seg, "continuity_from_prev", True)
            ]
            skipped_pin = [
                seg.index + 1
                for seg in plan.segments
                if seg.index > 0 and not getattr(seg, "continuity_from_prev", True)
            ]
            lines.append(
                f"Segment continuity: ON (motion context {plan.continuity_overlap_frames}f)"
            )
            if pinned:
                lines.append("  Pin from prev: #" + ", #".join(str(i) for i in pinned))
            if skipped_pin:
                lines.append(
                    "  Hard cut (per-segment off): #"
                    + ", #".join(str(i) for i in skipped_pin)
                )
        for seg in plan.segments:
            pin_note = ""
            if plan.continuity_enabled and seg.index > 0:
                pin_note = (
                    " — pin←prev"
                    if getattr(seg, "continuity_from_prev", True)
                    else " — hard cut"
                )
            lines.append(
                f"  #{seg.index + 1} [{seg.start_frame}:{seg.end_frame}] "
                f"{seg.frame_count}f — {seg.task_key}{pin_note} — "
                f"{seg.prompt[:60]}{'…' if len(seg.prompt) > 60 else ''}"
            )
        return "\n".join(lines)

    mode_label = (
        f"视频编辑 ({plan.global_task_key})"
        if plan.global_task_key in {"v2v", "rv2v"}
        else "源视频时间轴"
    )
    lines = [
        f"MiniMax H3 Director [{mode_label}] ({plan.edit_mode}): {plan.segment_count} segment(s), "
        f"{plan.total_frames} frames @ {plan.frame_rate:.2f} fps",
    ]
    if plan.export_max_frames > 0 and plan.source_total_frames > plan.total_frames:
        lines.append(
            f"Export cap: {plan.total_frames}/{plan.source_total_frames} frames "
            f"(max {plan.export_max_frames})"
        )
    export_label = "分段导出" if plan.export_mode == "segments" else "全部导出"
    lines.append(f"Export mode: {export_label}")
    if plan.continuity_enabled:
        pinned = [
            seg.index + 1
            for seg in plan.segments
            if seg.index > 0 and getattr(seg, "continuity_from_prev", True)
        ]
        skipped_pin = [
            seg.index + 1
            for seg in plan.segments
            if seg.index > 0 and not getattr(seg, "continuity_from_prev", True)
        ]
        lines.append(
            f"Segment continuity: ON (motion context {plan.continuity_overlap_frames}f "
            "→ pin previous tail + trim prefix; t2v/i2v/fl2v/r2v/v2v/rv2v)"
        )
        if pinned:
            lines.append(
                "  Pin from prev: #"
                + ", #".join(str(i) for i in pinned)
            )
        if skipped_pin:
            lines.append(
                "  Hard cut (per-segment off): #"
                + ", #".join(str(i) for i in skipped_pin)
            )
    elif plan.segment_count >= 2 and plan.global_task_key in {
        "t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v",
    }:
        lines.append(
            "Segment continuity: OFF — hard cuts between segments "
            "(enable「段间引导」in Director UI; recommend 22 frames)"
        )
    else:
        lines.append("Segment continuity: OFF (per-segment generation)")
    refine_line = None
    try:
        from .refine_pack import refine_report_line

        refine_line = refine_report_line(plan)
    except Exception:
        refine_line = None
    if refine_line:
        lines.append(refine_line)
    if plan.run_indices is not None:
        selected = sorted(plan.run_indices)
        skipped = [i + 1 for i in range(plan.segment_count) if i not in plan.run_indices]
        lines.append(
            f"Run selection: {len(selected)}/{plan.segment_count} segment(s) "
            f"(#{', #'.join(str(i + 1) for i in selected)}; skipped #{', #'.join(map(str, skipped)) or 'none'})"
        )
    lines.append(f"Global task: {get_task_prompt_spec(plan.global_task_type).label}")
    for seg in plan.segments:
        pin_note = ""
        if plan.continuity_enabled and seg.index > 0:
            pin_note = (
                " — pin←prev"
                if getattr(seg, "continuity_from_prev", True)
                else " — hard cut"
            )
        lines.append(
            f"  #{seg.index + 1} [{seg.start_frame}:{seg.end_frame}] "
            f"{seg.frame_count}f — {seg.task_key}{pin_note} — "
            f"{seg.prompt[:60]}{'…' if len(seg.prompt) > 60 else ''}"
        )
    return "\n".join(lines)
