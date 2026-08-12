"""Disk cache for MiniMax H3 Director segment decode outputs (partial re-run + merge).

Cache is best-effort: write failures (cloud RO mounts, same-name overwrite
blocks, full disks) must never abort the main generation run.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable

import torch

import folder_paths

from .h3_motion_context import CONTINUITY_PIPELINE_ID
from .plan import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache")


def _cache_root(node_id: str) -> Path | None:
    try:
        root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError as exc:
        log.warning("Segment cache dir unavailable (%s); cache disabled for this run.", exc)
        return None


def segment_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Stable identity for a segment 鈥?cache invalidates when edit params change."""
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
        "refs": ref_files,
        "ref_audios": ref_audio_files,
        "ref_videos": ref_video_files,
        "ref_video": ref_video_file,
        "ref_video_start": seg.reference_video_start_frame,
        "continuity": plan.continuity_enabled,
        "continuity_overlap": plan.continuity_overlap_frames if plan.continuity_enabled else 0,
        "continuity_from_prev": bool(getattr(seg, "continuity_from_prev", True)),
        # Bump CONTINUITY_PIPELINE_ID when sampling/handoff semantics change.
        "continuity_pipeline": CONTINUITY_PIPELINE_ID,
    }


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
        payload = tensor.cpu().float().contiguous()
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
        if stored != expected and not allow_stale:
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
        if stored != expected and not allow_stale:
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
    if allow_stale:
        # Companion loads (audio/av) may follow a video that was accepted stale.
        return tensor_path.is_file()
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        return stored == segment_cache_fingerprint(seg, plan)
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
    when the fingerprint drifted (pipeline bump, minor plan churn).
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
        return torch.load(tensor_path, map_location="cpu", weights_only=True)
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
