"""Disk cache for MiniMax H3 Director segment decode outputs (partial re-run + merge).

Cache is best-effort: write failures (cloud RO mounts, same-name overwrite
blocks, full disks) must never abort the main generation run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import torch

import folder_paths

from .h3_latent_continue import CONTINUE_PIPELINE_ID, clamp_seam_min_mask
from .h3_motion_context import CONTINUITY_PIPELINE_ID, trim_context_prefix, trim_export_tail
from .plan import DirectorPlan, SegmentPlan, resolve_ref_image_size

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache")

SOURCE_VIDEO_FP_KEY = "source_video"

# External-group (graph-wired i2v_groups / r2v_groups) identity. Stamped by
# build_plan_from_external_groups from the frontend wiring witness's per-group
# record — one group = one segment = one cache slot.
#
# Only the segment's **own** record is stored, never the whole chain: a segment is
# reused on its own identity, so editing group_1 leaves group_0's cache alone.
# The chain itself lives in ``timeline_data`` (status request and execute both
# receive it) and is never persisted.
EXTERNAL_SEGMENT_FP_KEY = "external_group"

# Fingerprint keys that come out of the executed group payloads (prompts,
# durations, reference counts/slots, per-row continuity). The cache-status panel
# cannot rebuild them for graph-wired groups, so in external-group mode it
# compares only the remaining keys plus the wiring witness.
GROUP_DERIVED_FP_KEYS = frozenset(
    {
        "index",
        "start",
        "end",
        "prompt",
        "negative",
        "task_key",
        "refs",
        "ref_audios",
        "ref_videos",
        "ref_video",
        "ref_video_start",
        "continuity_from_prev",
        "ref_image_size",
        SOURCE_VIDEO_FP_KEY,
    }
)


def has_external_marker(stored: Any) -> bool:
    """Whether this cache was written from a graph-wired external-group plan."""
    if not isinstance(stored, dict):
        return False
    return EXTERNAL_SEGMENT_FP_KEY in stored


def source_video_identity(plan: DirectorPlan) -> list[str]:
    """Stable source-clip identity: relative path + size + mtime (overwrite-safe)."""
    from ..lib.video_io import resolve_video_path, video_clips_from_timeline

    clips = video_clips_from_timeline((plan.raw or {}) if plan is not None else {})
    tokens: list[str] = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        rel = str(clip.get("videoFile") or clip.get("fileName") or "").strip().replace("\\", "/")
        if not rel:
            continue
        try:
            path = resolve_video_path(clip)
            st = os.stat(path)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
            tokens.append(f"{rel}:{st.st_size}:{mtime_ns}")
        except Exception:
            tokens.append(f"{rel}:missing")
    return tokens


def source_identity_changed(stored: Any, expected: dict[str, Any]) -> bool:
    """True when the current plan has a source video that does not match cache meta.

    Gen timelines (no source clips) never count as a source change, so stale
    fill/continuity still work after pipeline-only fingerprint churn.
    """
    exp = expected.get(SOURCE_VIDEO_FP_KEY) or []
    if not exp:
        return False
    if not isinstance(stored, dict) or SOURCE_VIDEO_FP_KEY not in stored:
        return True
    return stored.get(SOURCE_VIDEO_FP_KEY) != exp


def _reject_source_stale(
    stored: Any,
    expected: dict[str, Any],
    *,
    seg_index: int,
    quiet: bool = False,
) -> bool:
    if not source_identity_changed(stored, expected):
        return False
    if not quiet:
        log.info(
            "Segment %d cache is from a different source video; ignoring stale render.",
            seg_index + 1,
        )
    return True


def _cache_root(node_id: str) -> Path | None:
    try:
        root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError as exc:
        log.warning("Segment cache dir unavailable (%s); cache disabled for this run.", exc)
        return None


def _ref_audio_file_stamp(audio: Any, fallback_index: int) -> str:
    """Fingerprint fragment for one reference audio.

    Keying on the uploaded filename alone misses the case where the user keeps
    the same slot/filename but replaces the audio content. Stamp the source
    file's mtime+size so a content swap invalidates the first-pass cache.
    """
    index = getattr(audio, "index", fallback_index)
    name = getattr(audio, "audio_file", "") or ""
    path = getattr(audio, "audio_path", "") or ""
    stamp = ""
    if path:
        try:
            st = os.stat(path)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
            stamp = f"{mtime_ns}:{int(st.st_size)}"
        except OSError:
            stamp = ""
    return f"aud{index}:{name}:{stamp}"


def _segment_identity_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Identity that affects first-pass sampling (no Refine settings)."""
    ref_files = sorted(
        f"img{ref.index}:{(getattr(ref, 'image_file', '') or '')}"
        for ref in seg.refs
    )
    ref_audio_files = sorted(
        _ref_audio_file_stamp(a, i)
        for i, a in enumerate(getattr(seg, "ref_audios", None) or [])
    )
    ref_video_files = sorted(
        f"vid{getattr(v, 'index', i)}:{(getattr(v, 'video_file', '') or '')}"
        for i, v in enumerate(getattr(seg, "ref_videos", None) or [])
    )
    ref_video_file = (
        seg.reference_video_meta.get("videoFile")
        or seg.reference_video_meta.get("fileName")
        or ""
    ).strip()
    payload = {
        "index": seg.index,
        "start": seg.start_frame,
        "end": seg.end_frame,
        "prompt": seg.prompt,
        "negative": seg.negative_prompt,
        "task_key": seg.task_key,
        "width": plan.width,
        "height": plan.height,
        "frame_rate": float(getattr(plan, "frame_rate", 24) or 24),
        "output_mode": plan.output_mode,
        "ref_max": plan.ref_max_size,
        "ref_image_size": resolve_ref_image_size(seg, plan),
        "refs": ref_files,
        "ref_audios": ref_audio_files,
        "ref_videos": ref_video_files,
        "ref_video": ref_video_file,
        "ref_video_start": seg.reference_video_start_frame,
        SOURCE_VIDEO_FP_KEY: source_video_identity(plan),
        "continuity": plan.continuity_enabled,
        "continuity_overlap": plan.continuity_overlap_frames if plan.continuity_enabled else 0,
        "continuity_from_prev": bool(getattr(seg, "continuity_from_prev", True)),
        "continuity_mode": (
            str(getattr(plan, "continuity_mode", "guide") or "guide")
            if plan.continuity_enabled
            else "off"
        ),
        "continuity_redraw": (
            round(clamp_seam_min_mask(getattr(plan, "continuity_redraw", 0.10)), 2)
            if plan.continuity_enabled
            and str(getattr(plan, "continuity_mode", "guide") or "guide") == "continue"
            else 0
        ),
        "continuity_pipeline": (
            CONTINUE_PIPELINE_ID
            if plan.continuity_enabled
            and str(getattr(plan, "continuity_mode", "guide") or "guide") == "continue"
            else CONTINUITY_PIPELINE_ID
        ),
    }
    if plan.continuity_enabled and bool(getattr(plan, "continuity_keep_tail", True)):
        payload["continuity_keep_tail"] = True
    witness = getattr(plan, "external_groups_witness", None)
    if isinstance(witness, dict):
        # This segment's own group only. The whole chain is deliberately *not*
        # folded in: a segment is reused on its own identity, so editing one
        # group must not invalidate the others (it would also make the panel
        # report every segment as mismatched).
        groups = witness.get("groups")
        index = int(getattr(seg, "index", 0) or 0)
        if isinstance(groups, list) and 0 <= index < len(groups):
            record = groups[index]
            if isinstance(record, dict):
                payload[EXTERNAL_SEGMENT_FP_KEY] = record
    return payload



def first_pass_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Exact-match key for first-pass AV latent. Refine knobs are excluded."""
    fp = _segment_identity_fingerprint(seg, plan)
    sigmas = getattr(plan, "sample_sigmas", None)
    linked = bool(sigmas) or bool(getattr(plan, "sample_sigmas_linked", False))
    fp.update({
        "kind": "first_pass",
        "seed": int(getattr(plan, "sample_seed", 0) or 0),
        "cfg": round(float(getattr(plan, "sample_cfg", 1.0) or 1.0), 6),
        "sampler": str(getattr(plan, "sample_sampler", "") or ""),
        "shift_video": round(float(getattr(plan, "sample_shift_video", 12.0) or 12.0), 6),
        "shift_audio": round(float(getattr(plan, "sample_shift_audio", 3.0) or 3.0), 6),
    })
    if linked:
        fp["steps"] = 0
        fp["scheduler"] = "external_sigmas"
        fp["sigmas_source"] = "linked"
        if sigmas:
            fp["sigmas"] = [round(float(x), 6) for x in sigmas]
    else:
        fp["steps"] = int(getattr(plan, "sample_steps", 25) or 25)
        fp["scheduler"] = str(getattr(plan, "sample_scheduler", "") or "")
    return fp


def segment_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Stable identity for a segment — cache invalidates when edit params change."""
    fp = _segment_identity_fingerprint(seg, plan)
    from .refine_pack import refine_fingerprint

    fp.update(refine_fingerprint(plan))
    return fp


def _safe_unlink(path: Path) -> bool:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        return True
    except OSError:
        return False


def _atomic_publish(tmp: Path, dest: Path) -> None:
    """Move ``tmp`` 鈫?``dest``, tolerating clouds that block same-name overwrite."""
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    # Some cloud mounts reject overwrite of an existing name 鈥?remove then rename.
    _safe_unlink(dest)
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    try:
        tmp.rename(dest)
        return
    except OSError:
        # Last resort: keep the unique temp as the published file name is blocked.
        # Caller may still fail if even create-new is denied.
        raise


def _write_via_temp(dest: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a unique temp name in the same folder, then publish to ``dest``."""
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_fn(tmp)
        _atomic_publish(tmp, dest)
    finally:
        _safe_unlink(tmp)


def _audio_payload_to_cpu(audio: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize export AUDIO dict for disk cache (waveform on CPU)."""
    if not isinstance(audio, dict):
        return None
    wave = audio.get("waveform")
    if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
        return None
    sr = int(audio.get("sample_rate") or 0) or 32000
    return {
        "waveform": wave.detach().cpu().contiguous(),
        "sample_rate": sr,
    }


def _frames_to_disk(tensor: torch.Tensor) -> torch.Tensor:
    """Store pixel frames as uint8 [0,255]. Export is 8-bit anyway; float32 is 4× larger."""
    x = tensor.detach().cpu()
    if x.dtype == torch.uint8:
        return x.contiguous()
    return x.float().clamp(0, 1).mul(255).round().clamp(0, 255).to(torch.uint8).contiguous()


def _frames_from_disk(loaded: Any) -> torch.Tensor | None:
    """Restore uint8 cache to float32 [0,1]; pass through legacy float caches."""
    if not isinstance(loaded, torch.Tensor):
        return None
    if loaded.dtype == torch.uint8:
        return loaded.float().div(255.0)
    return loaded.float()


def save_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    tensor: torch.Tensor,
    *,
    av_latent: dict | None = None,
    handoff: dict[str, Any] | None = None,
    audio: dict[str, Any] | None = None,
    replace_audio: bool = True,
) -> None:
    """Persist a segment tensor (+ optional AV latent / export audio). Never raises.

    ``replace_audio``:
      - True (default): write ``audio`` when present, otherwise delete stale audio.pt
        (fresh sample with mute/empty decode).
      - False: write ``audio`` when present, otherwise **keep** existing audio.pt
        (phase-align trim re-save must not wipe a prior audio cache).
    """
    if not node_id:
        return
    root = _cache_root(node_id)
    if root is None:
        return
    fp = segment_cache_fingerprint(seg, plan)
    idx = seg.index
    pt_path = root / f"seg_{idx:04d}.pt"
    meta_path = root / f"seg_{idx:04d}.meta.json"
    latent_path = root / f"seg_{idx:04d}.av.pt"
    handoff_path = root / f"seg_{idx:04d}.handoff.json"
    audio_path = root / f"seg_{idx:04d}.audio.pt"
    try:
        payload = _frames_to_disk(tensor)
        _write_via_temp(pt_path, lambda p: torch.save(payload, p))
        text = json.dumps(fp, ensure_ascii=False, sort_keys=True)
        _write_via_temp(
            meta_path,
            lambda p: p.write_text(text, encoding="utf-8"),
        )
        if av_latent is not None and isinstance(av_latent, dict) and "samples" in av_latent:
            cpu_latent = _av_latent_to_cpu(av_latent)
            _write_via_temp(latent_path, lambda p: torch.save(cpu_latent, p))
        if handoff:
            _write_via_temp(
                handoff_path,
                lambda p: p.write_text(
                    json.dumps(handoff, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                ),
            )
        audio_cpu = _audio_payload_to_cpu(audio)
        if audio_cpu is not None:
            _write_via_temp(audio_path, lambda p: torch.save(audio_cpu, p))
        elif replace_audio:
            # Fresh sample with no waveform — drop stale audio from an older run.
            _safe_unlink(audio_path)
        log.debug(
            "Cached segment %d for node %s (%d frames%s%s)",
            idx + 1,
            node_id,
            int(tensor.shape[0]),
            ", +av_latent" if av_latent is not None else "",
            ", +audio" if audio_cpu is not None else (
                ", keep-audio" if not replace_audio else ""
            ),
        )
    except Exception as exc:
        # Xiangong / similar: RO mount or same-name write → skip cache, keep run alive.
        log.warning(
            "Segment %d cache write skipped (%s). Generation continues without disk cache.",
            idx + 1,
            exc,
        )
        for stray in root.glob(f".seg_{idx:04d}.*"):
            _safe_unlink(stray)


def _fingerprint_diff_keys(stored: Any, expected: dict[str, Any]) -> list[str]:
    if not isinstance(stored, dict):
        return ["<invalid-meta>"]
    keys = sorted(set(stored) | set(expected))
    return [k for k in keys if stored.get(k) != expected.get(k)]


def load_segment_handoff_meta(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict[str, Any] | None:
    """Load trim/export handoff metadata (fingerprint must match unless ``allow_stale``)."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.meta.json"
    handoff_path = root / f"seg_{idx:04d}.handoff.json"
    if not meta_path.is_file() or not handoff_path.is_file():
        return None
    try:
        expected = segment_cache_fingerprint(seg, plan)
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored != expected:
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True) or not allow_stale:
                return None
        data = json.loads(handoff_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _av_latent_to_cpu(av_latent: dict) -> dict:
    samples = av_latent["samples"]
    if hasattr(samples, "unbind"):
        parts = [p.detach().cpu().contiguous() for p in samples.unbind()]
        try:
            import comfy.nested_tensor

            samples_cpu = comfy.nested_tensor.NestedTensor(tuple(parts))
        except Exception:
            samples_cpu = tuple(parts)
    elif isinstance(samples, (tuple, list)):
        samples_cpu = tuple(p.detach().cpu().contiguous() for p in samples)
    elif torch.is_tensor(samples):
        samples_cpu = samples.detach().cpu().contiguous()
    else:
        samples_cpu = samples
    out = {"samples": samples_cpu}
    for key, value in av_latent.items():
        if key == "samples":
            continue
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().contiguous()
        else:
            out[key] = value
    return out


def load_first_pass_av_latent(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict | None:
    """Load ``.pre.av.pt`` for continuity pin. Source-video mismatch still rejects."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.pre.meta.json"
    latent_path = root / f"seg_{idx:04d}.pre.av.pt"
    if not latent_path.is_file():
        return None
    try:
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = first_pass_cache_fingerprint(seg, plan)
            if stored != expected:
                if _reject_source_stale(stored, expected, seg_index=idx, quiet=True):
                    return None
                if not allow_stale:
                    return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        return payload
    except Exception as exc:
        log.debug("Segment %d first-pass AV latent skipped: %s", idx + 1, exc)
        return None


def load_segment_av_latent(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict | None:
    """Load cached AV latent for continuity handoff (fingerprint must match unless stale-ok)."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.meta.json"
    latent_path = root / f"seg_{idx:04d}.av.pt"
    if not meta_path.is_file() or not latent_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = segment_cache_fingerprint(seg, plan)
        if stored != expected:
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True) or not allow_stale:
                return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        return payload
    except Exception as exc:
        log.warning("Failed to load segment %d AV latent cache: %s", idx + 1, exc)
        return None


def _fingerprint_matches(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> bool:
    if not node_id:
        return False
    root = _cache_root(node_id)
    if root is None:
        return False
    meta_path = root / f"seg_{seg.index:04d}.meta.json"
    tensor_path = root / f"seg_{seg.index:04d}.pt"
    if not meta_path.is_file():
        return False
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = segment_cache_fingerprint(seg, plan)
        if stored == expected:
            return True
        if _reject_source_stale(stored, expected, seg_index=seg.index, quiet=True):
            return False
        return bool(allow_stale and tensor_path.is_file())
    except Exception:
        return False


def load_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> torch.Tensor | None:
    """Load cached segment frames.

    ``allow_stale=True``: used for「选择运行」+「全部导出」fill of unselected
    segments. Prefer the last render on disk over blank/gray source placeholders
    when the fingerprint drifted (pipeline bump, minor plan churn). A different
    source video is never treated as usable stale — callers then passthrough
    the current clip (v2v/rv2v) or skip (gen timelines).
    """
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.meta.json"
    tensor_path = root / f"seg_{idx:04d}.pt"
    if not tensor_path.is_file():
        return None
    try:
        expected = segment_cache_fingerprint(seg, plan)
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if stored != expected:
                if _reject_source_stale(stored, expected, seg_index=idx):
                    return None
                diff = _fingerprint_diff_keys(stored, expected)
                if not allow_stale:
                    log.info(
                        "Segment %d cache stale (diff=%s); re-run this segment to refresh.",
                        idx + 1,
                        diff[:8],
                    )
                    return None
                log.warning(
                    "Segment %d: using stale cache for export fill (diff=%s).",
                    idx + 1,
                    diff[:8],
                )
        elif not allow_stale:
            log.info(
                "Segment %d cache missing meta; re-run this segment to refresh.",
                idx + 1,
            )
            return None
        else:
            log.warning(
                "Segment %d: using cache without meta for export fill.",
                idx + 1,
            )
        return _frames_from_disk(
            torch.load(tensor_path, map_location="cpu", weights_only=True)
        )
    except Exception as exc:
        log.warning("Failed to load segment %d cache: %s", idx + 1, exc)
        return None


def load_segment_audio(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict[str, Any] | None:
    """Load cached export audio for a segment (same fingerprint policy as video)."""
    if not node_id or not _fingerprint_matches(
        node_id, seg, plan, allow_stale=allow_stale
    ):
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    audio_path = root / f"seg_{seg.index:04d}.audio.pt"
    if not audio_path.is_file():
        return None
    try:
        payload = torch.load(audio_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return None
        wave = payload.get("waveform")
        if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
            return None
        sr = int(payload.get("sample_rate") or 0) or 32000
        return {"waveform": wave.contiguous(), "sample_rate": sr}
    except Exception as exc:
        log.warning("Failed to load segment %d audio cache: %s", seg.index + 1, exc)
        return None


def save_first_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    av_latent: dict | None = None,
    frames: torch.Tensor | None = None,
    handoff: dict[str, Any] | None = None,
) -> None:
    """Persist first-pass AV latent for confirm-then-refine. Never raises."""
    if not node_id:
        return
    if av_latent is None or not isinstance(av_latent, dict) or "samples" not in av_latent:
        return
    root = _cache_root(node_id)
    if root is None:
        return
    fp = first_pass_cache_fingerprint(seg, plan)
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.pre.meta.json"
    latent_path = root / f"seg_{idx:04d}.pre.av.pt"
    frames_path = root / f"seg_{idx:04d}.pre.pt"
    handoff_path = root / f"seg_{idx:04d}.pre.handoff.json"
    try:
        cpu_latent = _av_latent_to_cpu(av_latent)
        _write_via_temp(latent_path, lambda p: torch.save(cpu_latent, p))
        text = json.dumps(fp, ensure_ascii=False, sort_keys=True)
        _write_via_temp(meta_path, lambda p: p.write_text(text, encoding="utf-8"))
        if handoff:
            _write_via_temp(
                handoff_path,
                lambda p: p.write_text(
                    json.dumps(handoff, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                ),
            )
        if isinstance(frames, torch.Tensor) and frames.numel() > 0:
            payload = _frames_to_disk(frames)
            _write_via_temp(frames_path, lambda p: torch.save(payload, p))
        log.debug(
            "Cached first-pass segment %d for node %s (seed=%s)",
            idx + 1,
            node_id,
            fp.get("seed"),
        )
    except Exception as exc:
        log.warning(
            "Segment %d first-pass cache write skipped (%s).",
            idx + 1,
            exc,
        )
        for stray in root.glob(f".seg_{idx:04d}.pre.*"):
            _safe_unlink(stray)


def _trim_stale_first_pass_frames(
    frames: torch.Tensor,
    *,
    plan: DirectorPlan,
    handoff: dict[str, Any] | None,
    match_len: int | None,
) -> torch.Tensor | None:
    """Match in-memory first-pass export: drop context prefix, then crop length."""
    fps = float(getattr(plan, "frame_rate", 24) or 24)
    trim_frames = int((handoff or {}).get("trim_frames") or 0)
    export_len = int((handoff or {}).get("export_frames") or 0)
    if trim_frames > 0:
        if int(frames.shape[0]) <= trim_frames:
            return None
        frames, _ = trim_context_prefix(
            frames, None, trim_frames, fps=fps, match_tail=True
        )
    if export_len > 0 and int(frames.shape[0]) > export_len:
        frames = frames[:export_len]
    want = int(match_len or 0)
    extra = int(frames.shape[0]) - want if want > 0 else 0
    if extra > 0:
        frames, _ = trim_export_tail(frames, None, extra, fps=fps)
    return frames


def load_first_pass_frames_stale(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    match_len: int | None = None,
) -> torch.Tensor | None:
    """Load ``.pre.pt`` frames for unselected-segment pre-refine fill.

    Stale-tolerant counterpart of :func:`load_first_pass_cache`: fingerprint
    drift (different seed, sampling-knob churn) does NOT invalidate the fill,
    so「选择运行」re-roll previews merge all-first-pass frames instead of
    mixing a fresh first pass with cached refined renders. A different source
    video still rejects (same rule as the final-cache fill). Never raises.

    Disk ``.pre.pt`` is written before export trim; this reapplies
    ``.pre.handoff.json`` (context prefix + export length) and optionally
    matches the final-cache frame count after later phase-align tail trims.
    """
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    frames_path = root / f"seg_{idx:04d}.pre.pt"
    meta_path = root / f"seg_{idx:04d}.pre.meta.json"
    handoff_path = root / f"seg_{idx:04d}.pre.handoff.json"
    if not frames_path.is_file():
        return None
    try:
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = first_pass_cache_fingerprint(seg, plan)
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True):
                return None
        loaded = torch.load(frames_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, torch.Tensor) or loaded.numel() <= 0:
            return None
        frames = _frames_from_disk(loaded)
        if frames is None:
            return None
        handoff = None
        if handoff_path.is_file():
            try:
                data = json.loads(handoff_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    handoff = data
            except Exception:
                handoff = None
        return _trim_stale_first_pass_frames(
            frames, plan=plan, handoff=handoff, match_len=match_len
        )
    except Exception as exc:
        log.debug("Segment %d first-pass stale frames skipped: %s", idx + 1, exc)
    return None


def load_first_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
) -> dict[str, Any] | None:
    """Load first-pass cache only on exact fingerprint match. Never stale."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    meta_path = root / f"seg_{idx:04d}.pre.meta.json"
    latent_path = root / f"seg_{idx:04d}.pre.av.pt"
    frames_path = root / f"seg_{idx:04d}.pre.pt"
    handoff_path = root / f"seg_{idx:04d}.pre.handoff.json"
    if not meta_path.is_file() or not latent_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = first_pass_cache_fingerprint(seg, plan)
        if not isinstance(stored, dict) or stored != expected:
            if isinstance(stored, dict) and _reject_source_stale(
                stored, expected, seg_index=idx, quiet=True,
            ):
                return None
            diff = _fingerprint_diff_keys(stored, expected) if isinstance(stored, dict) else ["<invalid-meta>"]
            log.info(
                "Segment %d first-pass cache miss (diff=%s); will sample first pass.",
                idx + 1,
                diff[:8],
            )
            return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        frames = None
        if frames_path.is_file():
            try:
                loaded = torch.load(frames_path, map_location="cpu", weights_only=True)
                if isinstance(loaded, torch.Tensor) and loaded.numel() > 0:
                    frames = _frames_from_disk(loaded)
            except Exception as exc:
                log.debug("Segment %d first-pass frames skipped: %s", idx + 1, exc)
        handoff: dict[str, Any] = {}
        if handoff_path.is_file():
            try:
                data = json.loads(handoff_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    handoff = data
            except Exception:
                handoff = {}
        return {"av_latent": payload, "frames": frames, "handoff": handoff}
    except Exception as exc:
        log.warning("Failed to load segment %d first-pass cache: %s", idx + 1, exc)
        return None


_SEG_CACHE_FILE_RE = re.compile(r"^seg_(\d+)\.")


def prune_segment_cache(node_id: str | None, valid_indices) -> None:
    """Remove ``seg_XXXX.*`` files whose index is no longer on the timeline.

    Does not create the cache dir. Uses all current segment indices (not
    「选择运行」), so unselected slots keep merge/export fill. Never raises.
    """
    if not node_id:
        return
    try:
        root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
        if not root.is_dir():
            return
        valid = {int(i) for i in valid_indices}
        removed = 0
        for path in root.iterdir():
            if not path.is_file():
                continue
            m = _SEG_CACHE_FILE_RE.match(path.name)
            if not m or int(m.group(1)) in valid:
                continue
            if _safe_unlink(path):
                removed += 1
        if removed:
            log.info(
                "Segment cache pruned %d stale file(s) for node %s.", removed, node_id
            )
    except Exception as exc:
        log.debug("Segment cache prune skipped (%s).", exc)


def first_pass_cache_disk_signature(node_id: str | None) -> str:
    """Fingerprint confirm-first-pass ``*.pre.*`` files without creating the cache dir.

    Director ``IS_CHANGED`` cannot see the linked Refine pack (ComfyUI only
    forwards widgets). These ``.pre`` files are written only by the confirmation
    hold, so a second Queue observes a new signature and continues into refine.
    """
    if not node_id:
        return ""
    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    if not root.is_dir():
        return ""
    parts: list[str] = []
    try:
        for path in sorted(root.glob("seg_*.pre.*")):
            try:
                st = path.stat()
            except OSError:
                continue
            parts.append(f"{path.name}:{int(st.st_mtime_ns)}:{int(st.st_size)}")
    except OSError:
        return ""
    return "|".join(parts)


def _comparable_first_pass_fingerprint(plan: DirectorPlan) -> dict[str, Any]:
    """First-pass keys that do not depend on the executed group payloads.

    Built from the real fingerprint function so the panel compares exactly what
    the run writes; group-derived keys are dropped because the panel only has
    widget values (see ``GROUP_DERIVED_FP_KEYS``).
    """
    probe = SegmentPlan(
        index=0,
        start_frame=0,
        end_frame=0,
        prompt="",
        task_type="",
        task_key="",
        use_global=False,
    )
    try:
        fp = first_pass_cache_fingerprint(probe, plan)
    except Exception:
        return {}
    return {key: value for key, value in fp.items() if key not in GROUP_DERIVED_FP_KEYS}


_PRE_META_NAME_RE = re.compile(r"^seg_(\d+)\.pre\.meta\.json$")
_FINAL_FRAMES_NAME_RE = re.compile(r"^seg_\d+\.pt$")

# Buckets of a group record's digest. Only used to *name* the change: the
# per-group record is compared as a whole (that is what makes one group's edit
# invalidate that group's segment alone), and these labels say which part of it
# moved. ``timeline`` is skipped because timeline-level knobs are plan-wide and
# are compared through the fingerprint itself.
_WITNESS_FACET_DIFF_KEYS = {
    "wiring": "external_wiring",
    "prompt": "external_prompt",
    "length": "external_length",
    "media": "external_media",
    "other": "external_other",
    "timeline": "external_timeline",
}


def _external_witness_fps(witness: Any, plan: DirectorPlan) -> float:
    """Frame rate the run uses: the timeline's, then the Director widget."""
    if not isinstance(witness, dict):
        witness = {}
    try:
        fps = float(witness.get("fps"))
    except (TypeError, ValueError):
        fps = 0.0
    if not fps or fps <= 0:
        try:
            fps = float(getattr(plan, "frame_rate", 0) or 0)
        except (TypeError, ValueError):
            fps = 0.0
    return max(1.0, fps or 24.0)


def _external_group_seconds(duration_sec: Any) -> float:
    """Effective clip duration: the group's own value, else the packer default."""
    from .fl2v_timeline import DEFAULT_FL2V_DURATION_SEC

    try:
        seconds = float(duration_sec)
    except (TypeError, ValueError):
        seconds = 0.0
    return seconds if seconds else float(DEFAULT_FL2V_DURATION_SEC)


def _external_group_frames(duration_sec: Any, fps: float) -> int:
    """Frame count of one group — the maths ``build_plan_from_external_groups``
    applies to that group's ``duration_sec`` when it turns groups into segments."""
    from .fl2v_timeline import MIN_FL2V_FRAMES, _duration_to_minimax_frames
    from .frame_align import minimax_align_frame_count

    return int(
        minimax_align_frame_count(
            max(
                MIN_FL2V_FRAMES,
                _duration_to_minimax_frames(_external_group_seconds(duration_sec), fps),
            )
        )
    )


def _external_group_timings(groups: list[Any], fps: float) -> list[dict[str, Any]]:
    """Per-group `{duration, frames, start, end}` in run order (cumulative)."""
    timings: list[dict[str, Any]] = []
    cursor = 0
    for record in groups:
        raw_duration = record.get("dur") if isinstance(record, dict) else None
        duration = _external_group_seconds(raw_duration)
        frames = _external_group_frames(raw_duration, fps)
        start, end = cursor, cursor + frames
        cursor = end
        timings.append(
            {
                "duration": round(duration, 3),
                "frames": frames,
                "start": start,
                "end": end,
            }
        )
    return timings


def _external_selected_indices(
    witness: dict[str, Any],
    count: int,
) -> frozenset[int] | None:
    """「选择运行」selection out of the witness, using the run's own parser."""
    sel = witness.get("sel")
    if not isinstance(sel, dict) or not sel.get("on"):
        return None
    idx = sel.get("idx")
    if not isinstance(idx, list) or not idx or count <= 0:
        return None
    from .plan import _parse_run_selection

    try:
        return _parse_run_selection({"runSelectEnabled": True, "runSelection": idx}, count)
    except Exception:
        return None


def _external_segment_diff(stored_record: Any, expected_record: Any) -> list[str]:
    """Name what changed inside **this** segment's own group record.

    The record itself is the identity (the run compares it as a whole), so the
    caller already knows it differs; this only turns that into a label. The
    explicit fields come first — the prompt digest and the duration — because they
    are what users actually edit; the facet digests then say whether it was
    reference media, wiring or something else. Anything that differs only in a
    part with no bucket (the node was swapped for another with the same widgets)
    reports the generic wiring key instead of guessing.
    """
    if not isinstance(stored_record, dict) or not isinstance(expected_record, dict):
        return []
    keys: list[str] = []
    if stored_record.get("prompt") != expected_record.get("prompt"):
        keys.append("external_prompt")
    if stored_record.get("dur") != expected_record.get("dur"):
        keys.append("external_length")
    stored_facets = stored_record.get("facets")
    expected_facets = expected_record.get("facets")
    for facet, diff_key in _WITNESS_FACET_DIFF_KEYS.items():
        if facet == "timeline" or diff_key in keys:
            continue
        if not isinstance(stored_facets, dict) or not isinstance(expected_facets, dict):
            continue
        if stored_facets.get(facet) != expected_facets.get(facet):
            keys.append(diff_key)
    return keys or ["external_wiring"]


def _count_final_segment_files(root: Path) -> int:
    """Number of ``seg_XXXX.pt`` (final render) files; never raises."""
    if not root.is_dir():
        return 0
    try:
        return sum(
            1
            for path in root.glob("seg_*.pt")
            if _FINAL_FRAMES_NAME_RE.match(path.name)
        )
    except OSError:
        return 0


def _inspect_external_group_cache(
    node_id: str | None,
    plan: DirectorPlan,
    witness: dict[str, Any],
) -> dict[str, Any]:
    """Cache status for graph-wired external groups.

    ``i2v_groups`` / ``r2v_groups`` arrive as tensors at execute time, so the
    panel cannot rebuild the executed segments from widget values. Per segment it
    therefore compares the plan-level sampling knobs it *can* see, that segment's
    own group record and its frame range — the same set the run uses, which keeps
    one group's edit from reporting every other group as changed.

    One group = one segment = one cache slot, so the segments are enumerated from
    the witness's group list — the same order the run uses — rather than from the
    UI timeline, whose cards external groups override.
    """
    result: dict[str, Any] = {
        "exists": False,
        "matches": False,
        "current_seed": int(getattr(plan, "sample_seed", 0) or 0),
        "cached_seeds": [],
        "segment_total": 0,
        "cached_count": 0,
        "matched_count": 0,
        "selected_total": 0,
        "selected_cached": 0,
        "selected_matched": 0,
        "final_cached_count": 0,
        "diff_keys": [],
        "segments": [],
        "mode": "external_groups",
        "external": dict(witness),
        "unverified_count": 0,
    }
    if not node_id:
        return result

    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    result["final_cached_count"] = _count_final_segment_files(root)

    # Plan-level knobs only. The group-derived keys cannot be rebuilt without the
    # executed payloads, and this segment's own record is compared below as a
    # whole — that is the per-group identity.
    expected_knobs = {
        key: value
        for key, value in _comparable_first_pass_fingerprint(plan).items()
        if key != EXTERNAL_SEGMENT_FP_KEY
    }
    raw_groups = witness.get("groups")
    groups = [g for g in raw_groups if isinstance(g, dict)] if isinstance(raw_groups, list) else []
    timings = _external_group_timings(groups, _external_witness_fps(witness, plan))
    local_slots = not timings
    if timings:
        result["segment_total"] = len(timings)
        selected = _external_selected_indices(witness, len(timings))
        indices = list(range(len(timings)))
    else:
        # Caches written by an older frontend carry no group list: fall back to
        # whatever is on disk (the segment count is then unknown).
        selected = None
        try:
            indices = sorted(
                int(match.group(1))
                for path in root.glob("seg_*.pre.meta.json")
                if (match := _PRE_META_NAME_RE.match(path.name))
            )
        except OSError:
            return result
        result["segment_total"] = len(indices)

    rows: list[dict[str, Any]] = []
    cached_seeds: set[int] = set()
    all_diffs: set[str] = set()
    unverified = 0

    for idx in indices:
        timing = timings[idx] if idx < len(timings) else {}
        record = groups[idx] if idx < len(groups) else None
        meta_path = root / f"seg_{idx:04d}.pre.meta.json"
        latent_path = root / f"seg_{idx:04d}.pre.av.pt"
        meta_exists = meta_path.is_file()
        cache_exists = meta_exists and latent_path.is_file()
        stored: Any = None
        read_error = ""
        if meta_exists:
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                read_error = str(exc)

        cached_seed = stored.get("seed") if isinstance(stored, dict) else None
        try:
            if cached_seed is not None:
                cached_seed = int(cached_seed)
                cached_seeds.add(cached_seed)
        except (TypeError, ValueError):
            cached_seed = None

        diff: list[str] = []
        matches = False
        stored_record = (
            stored.get(EXTERNAL_SEGMENT_FP_KEY) if isinstance(stored, dict) else None
        )
        if meta_exists and not isinstance(stored, dict):
            diff = ["<invalid-meta>"]
            status = "mismatch"
        elif not isinstance(stored, dict):
            status = "missing"
        elif not isinstance(stored_record, dict):
            # Written before per-group records existed (older revision, or queued
            # straight through the API): the group's own identity cannot be
            # checked, and the run re-samples this segment once.
            status = "unverified"
            unverified += 1
        else:
            diff = [
                key for key, value in expected_knobs.items() if stored.get(key) != value
            ]
            # This segment's own group is the only per-group input to reuse.
            # A peer group's edit is deliberately invisible here — that is what
            # keeps one group's cache independent of the others.
            if isinstance(record, dict) and stored_record != record:
                diff.extend(_external_segment_diff(stored_record, record))
            expected_end = timing.get("end")
            stored_end = stored.get("end")
            if expected_end is not None and stored_end is not None:
                try:
                    if int(stored_end) != int(expected_end):
                        # This segment's own duration is unchanged but its frame
                        # range moved: an earlier group got longer/shorter. Segments
                        # are laid out back to back and each one is conditioned on
                        # the previous clip's tail, so the range has to shift and
                        # the segment must be re-sampled — but it is NOT this
                        # group's duration that changed, so it gets its own label
                        # instead of blaming「外接组时长」on an untouched group.
                        own_dur_moved = (
                            isinstance(stored_record, dict)
                            and isinstance(record, dict)
                            and stored_record.get("dur") != record.get("dur")
                        )
                        key = "external_length" if own_dur_moved else "external_shift"
                        if key not in diff:
                            diff.insert(0, key)
                except (TypeError, ValueError):
                    pass
            matches = bool(cache_exists and not diff)
            status = "valid" if matches else ("missing" if not cache_exists else "mismatch")

        all_diffs.update(diff)
        rows.append(
            {
                "segment": idx + 1,
                "index": idx,
                "slot": str(record.get("slot") or "") if isinstance(record, dict) else "",
                "node": str(record.get("node") or "") if isinstance(record, dict) else "",
                "duration": timing.get("duration"),
                "frames": timing.get("frames"),
                "start": timing.get("start"),
                "end": timing.get("end"),
                "stored_end": stored.get("end") if isinstance(stored, dict) else None,
                "exists": cache_exists,
                "matches": matches,
                "status": status,
                "selected": selected is None or idx in selected,
                "cached_seed": cached_seed,
                "diff_keys": diff,
                "error": read_error,
            }
        )

    cached_count = sum(1 for row in rows if row["exists"])
    matched_count = sum(1 for row in rows if row["matches"])
    selected_rows = [row for row in rows if row["selected"]]
    diff_keys = sorted(all_diffs)
    result.update(
        {
            "exists": cached_count > 0,
            "matches": bool(rows) and matched_count == len(rows),
            "cached_seeds": sorted(cached_seeds),
            "cached_count": cached_count,
            "matched_count": matched_count,
            "selected_total": len(selected_rows),
            "selected_cached": sum(1 for row in selected_rows if row["exists"]),
            "selected_matched": sum(1 for row in selected_rows if row["matches"]),
            "diff_keys": diff_keys,
            "segments": rows,
            "unverified_count": unverified,
        }
    )
    # Only the group list knows the true segment count; the disk scan does not.
    if local_slots:
        result["count_source"] = "cache_files"
    return result


def inspect_first_pass_cache(
    node_id: str | None,
    plan: DirectorPlan,
    *,
    external_groups: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect first-pass cache files without loading their tensor payloads.

    Always walks the whole timeline so「选择运行」unselected slots stay visible.
    ``final_cached_count`` is file presence only (``seg_XXXX.pt``), not a
    fingerprint match — Refine knobs are not on this status request.

    ``external_groups``: wiring witness of graph-wired ``i2v_groups`` /
    ``r2v_groups``. When present the group-derived keys cannot be rebuilt from
    widget values, so per segment only the plan-level knobs, that segment's own
    group record and its frame range are compared.
    """
    if external_groups:
        return _inspect_external_group_cache(node_id, plan, external_groups)

    current_seed = int(getattr(plan, "sample_seed", 0) or 0)
    result: dict[str, Any] = {
        "exists": False,
        "matches": False,
        "current_seed": current_seed,
        "cached_seeds": [],
        "segment_total": 0,
        "cached_count": 0,
        "matched_count": 0,
        "selected_total": 0,
        "selected_cached": 0,
        "selected_matched": 0,
        "final_cached_count": 0,
        "diff_keys": [],
        "segments": [],
        "mode": "timeline",
        "stale_external_count": 0,
    }
    if not node_id:
        return result

    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    all_segments = list(getattr(plan, "segments", None) or [])
    run_indices = getattr(plan, "run_indices", None)
    selected_set = frozenset(run_indices) if run_indices is not None else None
    result["segment_total"] = len(all_segments)

    cached_seeds: set[int] = set()
    all_diffs: set[str] = set()
    rows: list[dict[str, Any]] = []
    final_cached = 0
    stale_external = 0
    for seg in all_segments:
        is_selected = selected_set is None or int(seg.index) in selected_set
        idx = int(seg.index)
        meta_path = root / f"seg_{idx:04d}.pre.meta.json"
        latent_path = root / f"seg_{idx:04d}.pre.av.pt"
        meta_exists = meta_path.is_file()
        latent_exists = latent_path.is_file()
        cache_exists = meta_exists and latent_exists
        if (root / f"seg_{idx:04d}.pt").is_file():
            final_cached += 1
        stored: Any = None
        read_error = ""
        if meta_exists:
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                read_error = str(exc)

        expected = first_pass_cache_fingerprint(seg, plan)
        stored_cmp = stored
        expected_cmp = expected
        if getattr(plan, "sample_sigmas_linked", False) and isinstance(stored, dict):
            stored_cmp = {k: v for k, v in stored.items() if k != "sigmas"}
            expected_cmp = {k: v for k, v in expected.items() if k != "sigmas"}
        matches = bool(
            cache_exists
            and isinstance(stored, dict)
            and stored_cmp == expected_cmp
        )
        # A row stamped with an external-group witness was written by a group
        # run, while this request has no group input at all (link removed, or a
        # queued/API submit). Such a row can never be reused, and the UI-card
        # diffs below would be noise — the group prompt / duration / reference
        # media never lived on the card. Report the wiring change only.
        stale_external_row = has_external_marker(stored)
        if stale_external_row:
            diff = ["external_groups_off"]
            stale_external += 1
        elif isinstance(stored, dict):
            diff = _fingerprint_diff_keys(stored_cmp, expected_cmp)
        else:
            diff = ["<invalid-meta>"] if meta_exists else ["<missing-cache>"]
        if not cache_exists:
            status = "missing"
        elif matches:
            status = "valid"
        else:
            status = "mismatch"
        cached_seed = stored.get("seed") if isinstance(stored, dict) else None
        try:
            if cached_seed is not None:
                cached_seed = int(cached_seed)
                cached_seeds.add(cached_seed)
        except (TypeError, ValueError):
            cached_seed = None
        all_diffs.update(diff)
        rows.append(
            {
                "segment": idx + 1,
                "exists": cache_exists,
                "matches": matches,
                "status": status,
                "selected": is_selected,
                "cached_seed": cached_seed,
                "diff_keys": diff,
                "error": read_error,
            }
        )

    cached_count = sum(1 for row in rows if row["exists"])
    matched_count = sum(1 for row in rows if row["matches"])
    selected_rows = [row for row in rows if row["selected"]]
    selected_total = len(selected_rows) if selected_set is not None else len(rows)
    total = len(rows)
    result.update(
        {
            "exists": cached_count > 0,
            "matches": total > 0 and matched_count == total,
            "cached_seeds": sorted(cached_seeds),
            "cached_count": cached_count,
            "matched_count": matched_count,
            "selected_total": selected_total,
            "selected_cached": sum(1 for row in selected_rows if row["exists"]),
            "selected_matched": sum(1 for row in selected_rows if row["matches"]),
            "final_cached_count": final_cached,
            "diff_keys": sorted(all_diffs),
            "segments": rows,
            "stale_external_count": stale_external,
        }
    )
    return result


def clear_segment_cache(node_id: str | None, kind: str = "final") -> int:
    """Delete cached segment files for this Director node.

    ``kind``:
      - ``first_pass``: only ``seg_XXXX.pre.*`` (一采)
      - ``final``: everything except ``.pre.*`` (成片 / 二采，含 ``.audio.pt``)
      - ``all``: both

    Never creates the cache dir. Returns the number of files removed.
    """
    if not node_id:
        return 0
    if kind not in {"first_pass", "final", "all"}:
        raise ValueError("kind must be first_pass, final or all")
    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    if not root.is_dir():
        return 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    removed = 0
    for path in entries:
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        is_pre = ".pre." in path.name
        if kind == "first_pass" and not is_pre:
            continue
        if kind == "final" and is_pre:
            continue
        if _safe_unlink(path):
            removed += 1
    if removed:
        log.info("Cleared %s cache for node %s (%d file(s)).", kind, node_id, removed)
    return removed
