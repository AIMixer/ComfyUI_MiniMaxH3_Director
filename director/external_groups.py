"""External multi-group packs for MiniMax H3 Director (graph-wired inputs).

Two packer nodes produce ``MMX_DIR_GROUPS`` lists that override the UI timeline
media/prompts at execute time (external-priority).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..lib.image_prep import assert_minimax_canvas, fit_canvas, fit_video_long_edge, resolve_output_dimensions
from ..lib.ref_audios import MAX_REFERENCE_AUDIOS
from ..lib.ref_images import MAX_REFERENCE_IMAGES
from ..lib.ref_videos import MAX_REFERENCE_VIDEOS
from ..lib.task_prompts import resolve_task_key, task_type_option_label, TASK_PROMPT_BY_KEY
from .fl2v_timeline import (
    DEFAULT_FL2V_DURATION_SEC,
    DEFAULT_FL2V_NEGATIVE,
    MIN_FL2V_FRAMES,
    _build_fl2v_endpoint_source,
    _duration_to_minimax_frames,
    _unify_fl2v_pair_canvas,
    reinforce_fl2v_prompt,
)
from .frame_align import minimax_align_frame_count

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.external_groups")

MMX_DIR_GROUP = "MMX_DIR_GROUP"
# Same Comfy type string so a single group dict or a combined list can both
# connect to Director.i2v_groups / r2v_groups (normalized at execute time).
MMX_DIR_GROUPS = MMX_DIR_GROUP
I2V_FAMILY = frozenset({"t2v", "i2v", "fl2v"})
R2V_FAMILY = frozenset({"r2v"})


def _as_image_batch(image: Any) -> torch.Tensor | None:
    if image is None:
        return None
    # ComfyUI may wrap a single IMAGE as a one-element list/tuple.
    if isinstance(image, (list, tuple)):
        if not image:
            return None
        image = image[0]
        if image is None:
            return None
    if not isinstance(image, torch.Tensor) or image.numel() <= 0:
        return None
    t = image
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(t.shape)}")
    return t.contiguous().float()


def _as_video_frames(video: Any) -> torch.Tensor | None:
    """Accept IMAGE frame-batches (N,H,W,C) as reference video."""
    return _as_image_batch(video)


def _as_audio(audio: Any) -> dict | None:
    if audio is None or not isinstance(audio, dict):
        return None
    if audio.get("waveform") is None:
        return None
    return audio


def _fit_image(
    img: torch.Tensor,
    *,
    width: int,
    height: int,
    output_mode: str,
    ref_max_size: int,
) -> torch.Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if output_mode == "fixed":
        return fit_canvas(img, width, height)
    return fit_video_long_edge(img, ref_max_size)


def infer_i2v_kind(first_frame, last_frame) -> str:
    """Classify packer frames. Official MiniMaxH3ImageToVideo allows first and/or last."""
    has_first = first_frame is not None
    has_last = last_frame is not None
    if has_first and has_last:
        return "fl2v"
    if has_last:
        # Last-frame only (official FL2VA supports end keyframe without start).
        return "fl2v"
    if has_first:
        return "i2v"
    return "t2v"


def pack_i2v_group(
    *,
    prompt: str = "",
    duration_sec: float = DEFAULT_FL2V_DURATION_SEC,
    first_frame=None,
    last_frame=None,
) -> dict[str, Any]:
    first = _as_image_batch(first_frame)
    last = _as_image_batch(last_frame)
    kind = infer_i2v_kind(first, last)
    return {
        "version": 1,
        "family": "i2v",
        "kind": kind,
        "prompt": (prompt or "").strip(),
        "duration_sec": float(duration_sec) if duration_sec is not None else DEFAULT_FL2V_DURATION_SEC,
        "first_frame": first[:1].clone() if first is not None else None,
        "last_frame": last[:1].clone() if last is not None else None,
        "ref_images": {},
        "ref_videos": {},
        "ref_video_audios": {},
        "ref_audios": {},
    }


def pack_r2v_group(
    *,
    prompt: str = "",
    duration_sec: float = DEFAULT_FL2V_DURATION_SEC,
    ref_images: dict[int, Any] | None = None,
    ref_videos: dict[int, Any] | None = None,
    ref_video_audios: dict[int, Any] | None = None,
    ref_audios: dict[int, Any] | None = None,
) -> dict[str, Any]:
    images: dict[int, torch.Tensor] = {}
    for idx, img in (ref_images or {}).items():
        t = _as_image_batch(img)
        if t is None:
            continue
        i = int(idx)
        if 0 <= i < MAX_REFERENCE_IMAGES:
            images[i] = t[:1].clone()

    videos: dict[int, torch.Tensor] = {}
    for idx, vid in (ref_videos or {}).items():
        t = _as_video_frames(vid)
        if t is None:
            continue
        i = int(idx)
        if 0 <= i < MAX_REFERENCE_VIDEOS:
            videos[i] = t.clone()

    v_audios: dict[int, dict] = {}
    for idx, aud in (ref_video_audios or {}).items():
        a = _as_audio(aud)
        if a is None:
            continue
        i = int(idx)
        if 0 <= i < MAX_REFERENCE_VIDEOS:
            v_audios[i] = a

    audios: dict[int, dict] = {}
    for idx, aud in (ref_audios or {}).items():
        a = _as_audio(aud)
        if a is None:
            continue
        i = int(idx)
        if 0 <= i < MAX_REFERENCE_AUDIOS:
            audios[i] = a

    if not (prompt or "").strip() and not images and not videos and not audios:
        raise ValueError(
            "Director Group (Reference to Video): provide a prompt and/or at least one "
            "reference image, video, or audio."
        )

    return {
        "version": 1,
        "family": "r2v",
        "kind": "r2v",
        "prompt": (prompt or "").strip(),
        "duration_sec": float(duration_sec) if duration_sec is not None else DEFAULT_FL2V_DURATION_SEC,
        "first_frame": None,
        "last_frame": None,
        "ref_images": images,
        "ref_videos": videos,
        "ref_video_audios": v_audios,
        "ref_audios": audios,
    }


def append_group(groups: list | None, group: dict[str, Any]) -> list[dict[str, Any]]:
    out = list(groups or [])
    if not isinstance(group, dict) or group.get("kind") is None:
        raise ValueError("Invalid MMX_DIR_GROUP payload.")
    out.append(group)
    return out


def normalize_groups_list(raw) -> list[dict[str, Any]]:
    """Accept a single group, a list of groups, or None."""
    if raw is None:
        return []
    if isinstance(raw, dict) and raw.get("kind") is not None:
        return [raw]
    if isinstance(raw, (list, tuple)):
        out: list[dict[str, Any]] = []
        for item in raw:
            if item is None:
                continue
            if isinstance(item, dict) and item.get("kind") is not None:
                out.append(item)
            else:
                raise ValueError(f"Invalid item in MMX_DIR_GROUPS: {type(item)!r}")
        return out
    raise ValueError(f"Expected MMX_DIR_GROUPS list, got {type(raw)!r}")


def _parse_timeline_meta(timeline_data: str | None) -> dict:
    import json

    if not timeline_data or not str(timeline_data).strip():
        return {}
    try:
        data = json.loads(timeline_data)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _run_selection_filter(timeline: dict, count: int) -> list[int]:
    from .plan import _parse_run_selection

    sel = _parse_run_selection(timeline, count)
    if sel is None:
        return list(range(count))
    return sorted(int(i) for i in sel)


def _resolve_dims(timeline: dict, width: int, height: int, ref_max_size: int, sample_img: torch.Tensor | None):
    from .plan import _resolve_export_mode

    output_block = timeline.get("output") or {}
    out_mode = str(output_block.get("mode") or "long_edge").lower()
    if out_mode not in ("fixed", "long_edge"):
        out_mode = "long_edge"
    src_w = int(sample_img.shape[2]) if sample_img is not None else int(width or 864)
    src_h = int(sample_img.shape[1]) if sample_img is not None else int(height or 480)
    out_w, out_h, ref_max, out_mode = resolve_output_dimensions(
        src_w,
        src_h,
        mode=out_mode,
        long_edge=int(
            output_block.get("longEdge")
            or output_block.get("long_edge")
            or ref_max_size
            or 848
        ),
        fixed_width=int(output_block.get("width") or timeline.get("width") or width or 864),
        fixed_height=int(output_block.get("height") or timeline.get("height") or height or 480),
    )
    export_mode = _resolve_export_mode(output_block)
    assert_minimax_canvas(out_w, out_h)
    return out_w, out_h, ref_max, out_mode, export_mode


def validate_external_group_inputs(
    *,
    task_type: str,
    i2v_groups,
    r2v_groups,
) -> tuple[str, list[dict[str, Any]] | None, str | None]:
    """Return (task_key, groups_or_None, family_or_None). None groups → use UI timeline."""
    i2v_linked = i2v_groups is not None
    r2v_linked = r2v_groups is not None
    i2v = normalize_groups_list(i2v_groups)
    r2v = normalize_groups_list(r2v_groups)
    if i2v_linked and not i2v:
        raise ValueError("MiniMax H3 Director: i2v_groups is connected but contains no groups.")
    if r2v_linked and not r2v:
        raise ValueError("MiniMax H3 Director: r2v_groups is connected but contains no groups.")
    if i2v and r2v:
        raise ValueError(
            "MiniMax H3 Director: connect either Image to Video groups (i2v_groups) "
            "or Reference to Video groups (r2v_groups), not both."
        )
    task_key = resolve_task_key(task_type)
    if not i2v and not r2v:
        return task_key, None, None

    if i2v:
        if task_key not in I2V_FAMILY:
            raise ValueError(
                f"MiniMax H3 Director: i2v_groups is connected but task_type is '{task_key}'. "
                "Set task to t2v / i2v / fl2v."
            )
        for idx, g in enumerate(i2v):
            if g.get("family") not in (None, "i2v") or g.get("kind") not in I2V_FAMILY:
                raise ValueError(
                    f"i2v_groups[{idx}] contains a non Image-to-Video group "
                    f"(kind={g.get('kind')!r})."
                )
            kind = g["kind"]
            if task_key == "t2v" and kind != "t2v":
                raise ValueError(
                    f"task_type is t2v but i2v_groups[{idx}] has first/last frames. "
                    "Remove frames or switch task to i2v/fl2v."
                )
            if task_key == "i2v" and kind == "fl2v":
                raise ValueError(
                    f"task_type is i2v but i2v_groups[{idx}] has last_frame. "
                    "Remove last_frame or switch task to fl2v."
                )
            if task_key == "i2v" and kind == "t2v":
                raise ValueError(
                    f"task_type is i2v but i2v_groups[{idx}] has no first_frame. "
                    "Connect first_frame on that Group node."
                )
            # fl2v task accepts mixed groups: t2v (prompt-only), i2v (first), 
            # fl2v (first+last or last-only). Official ImageToVideo allows any combo.
        return task_key, i2v, "i2v"

    # r2v
    if task_key != "r2v":
        raise ValueError(
            f"MiniMax H3 Director: r2v_groups is connected but task_type is '{task_key}'. "
            "Set task to r2v."
        )
    for g in r2v:
        if g.get("kind") != "r2v":
            raise ValueError(f"r2v_groups contains a non-r2v group (kind={g.get('kind')!r}).")
    return task_key, r2v, "r2v"


def build_plan_from_external_groups(
    groups: list[dict[str, Any]],
    *,
    family: str,
    timeline_data: str,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
):
    from .plan import (
        DirectorPlan,
        SegmentPlan,
        SegmentRef,
        SegmentRefAudio,
        SegmentRefVideo,
        _load_ref_audios,
        _load_refs,
        concat_common_segment_prompt,
        merge_indexed_refs,
        reinforce_r2v_prompt,
        resolve_ref_image_size,
    )

    timeline = _parse_timeline_meta(timeline_data)
    fps = float(timeline.get("frameRate") or frame_rate or 24.0)
    task_key = resolve_task_key(task_type)
    label = task_type_option_label(TASK_PROMPT_BY_KEY.get(task_key) or TASK_PROMPT_BY_KEY["t2v"])
    task_label = task_type or label
    global_block = timeline.get("global") or {}
    fallback_prompt = (
        (global_block.get("prompt") or global_prompt or "")
    ).strip()
    common_enabled = bool(
        global_block.get("commonEnabled")
        if global_block.get("commonEnabled") is not None
        else global_block.get("common_enabled")
    )

    indices = _run_selection_filter(timeline, len(groups))
    if not indices:
        raise ValueError("External groups: no groups selected to run.")

    # Keep the full group list so「段间引导」can pin the true previous neighbor
    # (via cache) even when「选择运行」only samples a subset — same model as
    # native prompt_batch / source timelines (run_indices, not compacted plan).
    all_indexed = list(enumerate(groups))
    run_indices = (
        frozenset(indices) if len(indices) < len(groups) else None
    )
    sample_img = None
    for _, g in all_indexed:
        sample_img = g.get("first_frame")
        if sample_img is None:
            sample_img = g.get("last_frame")
        if sample_img is None:
            refs = g.get("ref_images") or {}
            sample_img = next(iter(refs.values()), None) if refs else None
        if sample_img is not None:
            break

    # Canvas always from Director UI / timeline (unified for multi-group batch).
    out_w, out_h, ref_max, out_mode, export_mode = _resolve_dims(
        timeline, width, height, ref_max_size, sample_img
    )

    common_refs_raw = (
        _load_refs(global_block.get("refs") or []) if (family == "r2v" and common_enabled) else []
    )
    common_audios_raw = (
        _load_ref_audios(
            global_block.get("refAudios") or global_block.get("ref_audios") or []
        )
        if (family == "r2v" and common_enabled)
        else []
    )

    from .segment_continuity import (
        resolve_segment_continuity_from_prev,
        timeline_row_for_index,
    )

    segments: list[SegmentPlan] = []
    cursor = 0
    for plan_idx, (src_index, g) in enumerate(all_indexed):
        group_prompt = (g.get("prompt") or "").strip()
        if family == "r2v" and common_enabled:
            prompt = concat_common_segment_prompt(fallback_prompt, group_prompt)
        else:
            prompt = group_prompt or fallback_prompt
        try:
            dur = float(g.get("duration_sec") or DEFAULT_FL2V_DURATION_SEC)
        except (TypeError, ValueError):
            dur = DEFAULT_FL2V_DURATION_SEC
        fc = max(MIN_FL2V_FRAMES, _duration_to_minimax_frames(dur, fps))
        fc = minimax_align_frame_count(fc)
        start_f = cursor
        end_f = cursor + fc
        cursor = end_f
        seg_w, seg_h, seg_mode = out_w, out_h, out_mode

        if family == "i2v":
            kind = g["kind"]
            first = g.get("first_frame")
            last = g.get("last_frame")
            refs: list[SegmentRef] = []
            source_clip = None
            # Prefer per-group kind (t2v / i2v / fl2v) so prompt-only groups stay t2v
            # even when the Director task dropdown is fl2v.
            if kind in I2V_FAMILY:
                seg_task_key = kind
            elif task_key in I2V_FAMILY:
                seg_task_key = task_key
            else:
                seg_task_key = "t2v"

            if first is not None or last is not None:
                start_img = (
                    _fit_image(
                        first, width=seg_w, height=seg_h, output_mode=seg_mode, ref_max_size=ref_max
                    )
                    if first is not None
                    else None
                )
                end_img = (
                    _fit_image(
                        last, width=seg_w, height=seg_h, output_mode=seg_mode, ref_max_size=ref_max
                    )
                    if last is not None
                    else None
                )
                start_img, end_img = _unify_fl2v_pair_canvas(start_img, end_img)
                refs = []
                if start_img is not None:
                    refs.append(SegmentRef(index=0, tensor=start_img[:1].clone()))
                if end_img is not None:
                    refs.append(SegmentRef(index=1, tensor=end_img[:1].clone()))
                # Last-only: skip source_clip so executor won't treat held end as first_frame.
                source_clip = (
                    _build_fl2v_endpoint_source(start_img, end_img, fc)
                    if start_img is not None
                    else None
                )
                if seg_task_key in {"fl2v", "i2v"}:
                    prompt = reinforce_fl2v_prompt(
                        prompt,
                        has_end_frame=end_img is not None,
                        has_start_frame=start_img is not None,
                    )

            row = timeline_row_for_index(timeline, int(src_index))
            if not row and isinstance(g, dict):
                row = g
            segments.append(
                SegmentPlan(
                    index=plan_idx,
                    start_frame=start_f,
                    end_frame=end_f,
                    prompt=prompt,
                    task_type=task_label,
                    task_key=seg_task_key,
                    use_global=False,
                    refs=refs,
                    negative_prompt=DEFAULT_FL2V_NEGATIVE if seg_task_key == "fl2v" else "",
                    source_clip=source_clip,
                    ui_index=int(src_index),
                    continuity_from_prev=resolve_segment_continuity_from_prev(
                        row, segment_index=plan_idx
                    ),
                    ref_image_size=resolve_ref_image_size(row, timeline),
                )
            )
        else:
            # r2v — per-group media + Director timeline.global common media/prompt
            refs = []
            for idx, tensor in sorted((g.get("ref_images") or {}).items()):
                fitted = _fit_image(
                    tensor, width=seg_w, height=seg_h, output_mode=seg_mode, ref_max_size=ref_max
                )
                refs.append(SegmentRef(index=int(idx), tensor=fitted[:1].clone()))
            if common_refs_raw:
                common_fitted = []
                for cref in common_refs_raw:
                    fitted = _fit_image(
                        cref.tensor,
                        width=seg_w,
                        height=seg_h,
                        output_mode=seg_mode,
                        ref_max_size=ref_max,
                    )
                    common_fitted.append(
                        SegmentRef(
                            index=int(cref.index),
                            tensor=fitted[:1].clone(),
                            image_file=getattr(cref, "image_file", "") or "",
                        )
                    )
                refs = merge_indexed_refs(common_fitted, refs)
            ref_videos = []
            for idx, frames in sorted((g.get("ref_videos") or {}).items()):
                fitted = _fit_image(
                    frames, width=seg_w, height=seg_h, output_mode=seg_mode, ref_max_size=ref_max
                )
                ref_videos.append(
                    SegmentRefVideo(index=int(idx), tensor=fitted.clone(), video_file="", meta={"external": True})
                )
            ref_audios = [
                SegmentRefAudio(index=int(idx), audio=aud, audio_file="")
                for idx, aud in sorted((g.get("ref_audios") or {}).items())
            ]
            if common_audios_raw:
                ref_audios = merge_indexed_refs(common_audios_raw, ref_audios)
            ref_video_audios = [
                SegmentRefAudio(index=int(idx), audio=aud, audio_file="")
                for idx, aud in sorted((g.get("ref_video_audios") or {}).items())
            ]
            prompt = reinforce_r2v_prompt(
                prompt,
                ref_indices=[r.index for r in refs],
                video_indices=[v.index for v in ref_videos],
                audio_indices=[a.index for a in ref_audios],
            )
            row = timeline_row_for_index(timeline, int(src_index))
            if not row and isinstance(g, dict):
                row = g
            segments.append(
                SegmentPlan(
                    index=plan_idx,
                    start_frame=start_f,
                    end_frame=end_f,
                    prompt=prompt,
                    task_type=task_label,
                    task_key="r2v",
                    use_global=False,
                    refs=refs,
                    ref_audios=ref_audios,
                    ref_videos=ref_videos,
                    ref_video_audios=ref_video_audios,
                    source_clip=None,
                    ui_index=int(src_index),
                    continuity_from_prev=resolve_segment_continuity_from_prev(
                        row, segment_index=plan_idx
                    ),
                    ref_image_size=resolve_ref_image_size(row, timeline),
                )
            )

    if not segments:
        raise ValueError("External groups produced no runnable segments.")

    total = int(segments[-1].end_frame)
    source_video = torch.full((len(segments), 16, 16, 3), 0.5, dtype=torch.float32)
    raw = dict(timeline)
    raw["externalGroups"] = {
        "source": family,
        "count": len(groups),
        "selected": list(indices),
        "active": True,
    }
    if family == "i2v":
        raw["timelineMode"] = "fl2v" if task_key == "fl2v" else "prompt_batch"
    else:
        raw["timelineMode"] = "prompt_batch"
    raw["totalFrames"] = total
    raw["editMode"] = "segment"

    from .segment_continuity import resolve_continuity_settings

    continuity_enabled, continuity_overlap = resolve_continuity_settings(
        timeline, segment_count=len(segments)
    )

    return DirectorPlan(
        frame_rate=fps,
        total_frames=total or int(total_frames or 0),
        width=out_w,
        height=out_h,
        ref_max_size=ref_max,
        output_mode=out_mode,
        source_width=out_w,
        source_height=out_h,
        global_task_type=task_label,
        global_task_key=task_key,
        global_prompt=fallback_prompt,
        global_refs=list(common_refs_raw) if family == "r2v" else [],
        source_video=source_video,
        segments=segments,
        edit_mode="segment",
        raw=raw,
        export_mode=export_mode,
        run_indices=run_indices,
        continuity_enabled=continuity_enabled,
        continuity_overlap_frames=continuity_overlap,
        global_ref_audios=list(common_audios_raw) if family == "r2v" else [],
    )
