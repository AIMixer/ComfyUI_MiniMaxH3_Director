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

from .h3_motion_context import CONTINUITY_PIPELINE_ID, trim_context_prefix, trim_export_tail
from .plan import DirectorPlan, SegmentPlan, resolve_ref_image_size

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache")

SOURCE_VIDEO_FP_KEY = "source_video"


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


def _segment_identity_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Identity that affects first-pass sampling (no Refine settings)."""
    ref_files = sorted(
        f"img{ref.index}:{(getattr(ref, 'image_file', '') or '')}"
        for ref in seg.refs
    )
    ref_audio_files = sorted(
        f"aud{getattr(a, 'index', i)}:{(getattr(a, 'audio_file', '') or '')}"
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
    return {
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
        "continuity_pipeline": CONTINUITY_PIPELINE_ID,
    }


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


def inspect_first_pass_cache(
    node_id: str | None,
    plan: DirectorPlan,
) -> dict[str, Any]:
    """Inspect first-pass cache files without loading their tensor payloads."""
    current_seed = int(getattr(plan, "sample_seed", 0) or 0)
    result: dict[str, Any] = {
        "exists": False,
        "matches": False,
        "current_seed": current_seed,
        "cached_seeds": [],
        "segment_total": 0,
        "cached_count": 0,
        "matched_count": 0,
        "diff_keys": [],
        "segments": [],
    }
    if not node_id:
        return result

    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    all_segments = list(getattr(plan, "segments", None) or [])
    run_indices = getattr(plan, "run_indices", None)
    if run_indices is None:
        selected = all_segments
    else:
        selected = [
            all_segments[i]
            for i in sorted(run_indices)
            if 0 <= i < len(all_segments)
        ]
    result["segment_total"] = len(selected)

    cached_seeds: set[int] = set()
    all_diffs: set[str] = set()
    rows: list[dict[str, Any]] = []
    for seg in selected:
        idx = int(seg.index)
        meta_path = root / f"seg_{idx:04d}.pre.meta.json"
        latent_path = root / f"seg_{idx:04d}.pre.av.pt"
        meta_exists = meta_path.is_file()
        latent_exists = latent_path.is_file()
        cache_exists = meta_exists and latent_exists
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
        matches = bool(cache_exists and isinstance(stored, dict) and stored_cmp == expected_cmp)
        diff = (
            _fingerprint_diff_keys(stored_cmp, expected_cmp)
            if isinstance(stored, dict)
            else (["<invalid-meta>"] if meta_exists else ["<missing-cache>"])
        )
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
                "cached_seed": cached_seed,
                "diff_keys": diff,
                "error": read_error,
            }
        )

    cached_count = sum(1 for row in rows if row["exists"])
    matched_count = sum(1 for row in rows if row["matches"])
    total = len(rows)
    result.update(
        {
            "exists": cached_count > 0,
            "matches": total > 0 and matched_count == total,
            "cached_seeds": sorted(cached_seeds),
            "cached_count": cached_count,
            "matched_count": matched_count,
            "diff_keys": sorted(all_diffs),
            "segments": rows,
        }
    )
    return result
