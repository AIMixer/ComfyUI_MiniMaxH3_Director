"""Build ComfyUI AUDIO outputs for MiniMax H3 Director runs.

Prefers model-generated audio from AV latent decode; falls back to source-video extract.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..lib.audio_io import (
    diagnose_source_audio_failure,
    extract_timeline_audio,
    frames_to_audio_samples,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.audio_export")

SILENT_SAMPLE_RATE = 44100

AUDIO_MODE_GENERATE = "generate"
AUDIO_MODE_SOURCE = "source"
AUDIO_MODE_MUTE = "mute"
VIDEO_EDIT_AUDIO_TASKS = frozenset({"v2v", "rv2v", "r2v"})


def task_passes_source_audio(task_key: str) -> bool:
    # v2v/rv2v: source-timeline extract; others may fall back after empty model audio.
    return task_key in {"i2v", "fl2v", "r2v", "v2v", "rv2v"}


def resolve_audio_mode(plan) -> str:
    """Return generate | source | mute from timeline.output.audioMode.

    ``source`` is only honored for v2v/rv2v; other tasks treat it as generate.
    """
    out = (getattr(plan, "raw", None) or {}).get("output") or {}
    raw = str(out.get("audioMode") or out.get("audio_mode") or AUDIO_MODE_GENERATE).strip().lower()
    aliases = {
        "generate": AUDIO_MODE_GENERATE,
        "generated": AUDIO_MODE_GENERATE,
        "model": AUDIO_MODE_GENERATE,
        "source": AUDIO_MODE_SOURCE,
        "original": AUDIO_MODE_SOURCE,
        "passthrough": AUDIO_MODE_SOURCE,
        "mute": AUDIO_MODE_MUTE,
        "silent": AUDIO_MODE_MUTE,
        "silence": AUDIO_MODE_MUTE,
    }
    mode = aliases.get(raw, AUDIO_MODE_GENERATE)
    task_key = str(getattr(plan, "global_task_key", "") or "")
    if mode == AUDIO_MODE_SOURCE and task_key not in VIDEO_EDIT_AUDIO_TASKS:
        return AUDIO_MODE_GENERATE
    return mode


def _normalize_audio(audio: dict[str, Any] | None) -> dict[str, Any] | None:
    """Force reference audio to 44.1 kHz stereo before mux (any input format)."""
    if audio is None or not _audio_has_samples(audio):
        return audio
    wave = audio["waveform"]
    sr = int(audio.get("sample_rate") or SILENT_SAMPLE_RATE)
    try:
        import torchaudio
        if sr != SILENT_SAMPLE_RATE:
            wave = torchaudio.functional.resample(wave, sr, SILENT_SAMPLE_RATE)
            sr = SILENT_SAMPLE_RATE
        if wave.shape[1] < 2:
            wave = wave.repeat(1, 2, 1)
        return {"waveform": wave.contiguous(), "sample_rate": sr}
    except Exception:
        return audio


def _ref_audio_for_segment(seg) -> dict[str, Any] | None:
    """First usable reference audio on a segment (the 'original sound' source)."""
    for ref in (getattr(seg, "ref_audios", None) or []):
        audio = getattr(ref, "audio", None)
        if _audio_has_samples(audio):
            return _normalize_audio(audio)
    return None


def _first_ref_audio(plan) -> dict[str, Any] | None:
    """First usable reference audio across the plan (global source fallback)."""
    for seg in (getattr(plan, "segments", None) or []):
        audio = _ref_audio_for_segment(seg)
        if audio is not None:
            return audio
    return None


def empty_audio_dict(sample_rate: int = SILENT_SAMPLE_RATE) -> dict[str, Any]:
    return {"waveform": torch.zeros(1, 1, 0), "sample_rate": int(sample_rate)}


def prepare_segment_audio_for_file_export(
    plan,
    seg,
    *,
    audio_dict: dict[str, Any] | None,
    frame_count: int,
) -> dict[str, Any] | None:
    """Return AUDIO to mux into a per-segment mp4, or None for video-only.

    Matches timeline ``audioMode``: mute → None; generate prefers model audio
    (source extract fallback when the task allows); source extracts timeline audio.
    """
    mode = resolve_audio_mode(plan)
    if mode == AUDIO_MODE_MUTE:
        return None

    fps = float(getattr(plan, "frame_rate", 24) or 24)
    n_frames = max(0, int(frame_count))
    if n_frames <= 0:
        return None

    if mode == AUDIO_MODE_GENERATE and _audio_has_samples(audio_dict):
        sr = int(audio_dict.get("sample_rate") or SILENT_SAMPLE_RATE)
        return _pad_or_trim_audio_to_frames(
            audio_dict, frame_count=n_frames, fps=fps, sample_rate=sr
        )

    # source mode, or generate with empty model audio on tasks that can pass source.
    if mode == AUDIO_MODE_SOURCE or (
        mode == AUDIO_MODE_GENERATE and task_passes_source_audio(str(getattr(plan, "global_task_key", "") or ""))
    ):
        timeline = getattr(plan, "raw", None) or {}
        start = int(getattr(seg, "start_frame", 0) or 0)
        end = int(getattr(seg, "end_frame", start + n_frames) or (start + n_frames))
        extracted = extract_timeline_audio(timeline, start, end, fps)
        if _audio_has_samples(extracted):
            sr = int(extracted.get("sample_rate") or SILENT_SAMPLE_RATE)
            return _pad_or_trim_audio_to_frames(
                extracted, frame_count=n_frames, fps=fps, sample_rate=sr
            )
        # r2v has no source-timeline video; fall back to the segment's
        # first reference audio (the "original sound" option) when present.
        ref_audios = getattr(seg, "ref_audios", None) or []
        if ref_audios and _audio_has_samples(ref_audios[0].audio):
            ref = _normalize_audio(ref_audios[0].audio)
            sr = int(ref.get("sample_rate") or SILENT_SAMPLE_RATE)
            return _pad_or_trim_audio_to_frames(
                ref, frame_count=n_frames, fps=fps, sample_rate=sr
            )
    return None


def _coerce_audio_output(audio: dict[str, Any] | None, *, sample_rate: int) -> dict[str, Any]:
    if audio is None:
        return empty_audio_dict(sample_rate)
    wave = audio.get("waveform")
    if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
        return empty_audio_dict(int(audio.get("sample_rate") or sample_rate))
    return audio


def _audio_has_samples(audio: dict[str, Any] | None) -> bool:
    return (
        isinstance(audio, dict)
        and isinstance(audio.get("waveform"), torch.Tensor)
        and int(audio["waveform"].numel()) > 0
    )


def _pad_or_trim_audio_to_frames(
    audio: dict[str, Any] | None,
    *,
    frame_count: int,
    fps: float,
    sample_rate: int = SILENT_SAMPLE_RATE,
) -> dict[str, Any]:
    fps = float(fps or 24.0)
    frame_count = max(0, int(frame_count))

    if audio is None or not isinstance(audio.get("waveform"), torch.Tensor):
        sr = int(sample_rate)
        target_samples = frames_to_audio_samples(frame_count, fps, sr)
        if target_samples <= 0:
            return empty_audio_dict(sr)
        return {
            "waveform": torch.zeros(1, 2, target_samples),
            "sample_rate": sr,
        }

    wave = audio["waveform"]
    sr = int(audio.get("sample_rate") or sample_rate)
    if sr <= 0:
        sr = sample_rate
    target_samples = frames_to_audio_samples(frame_count, fps, sr)
    if wave.ndim != 3:
        return _coerce_audio_output(audio, sample_rate=sr)
    channels = int(wave.shape[1])
    have = int(wave.shape[-1])
    if target_samples <= 0:
        return empty_audio_dict(sr)
    if have == target_samples:
        return {"waveform": wave, "sample_rate": sr}
    if have > target_samples:
        return {"waveform": wave[..., :target_samples].contiguous(), "sample_rate": sr}
    pad = torch.zeros(1, channels, target_samples - have, dtype=wave.dtype, device=wave.device)
    return {"waveform": torch.cat([wave, pad], dim=-1), "sample_rate": sr}


def _align_audio_channels(wave: torch.Tensor, channels: int) -> torch.Tensor:
    """Ensure waveform is [1, C, T] with C == channels (duplicate mono / trim extra)."""
    if wave.ndim != 3:
        return wave
    have = int(wave.shape[1])
    if have == channels:
        return wave
    if have == 1 and channels > 1:
        return wave.expand(1, channels, -1).contiguous()
    if have > channels:
        return wave[:, :channels, :].contiguous()
    # Pad missing channels with zeros.
    pad = torch.zeros(1, channels - have, wave.shape[-1], dtype=wave.dtype, device=wave.device)
    return torch.cat([wave, pad], dim=1)


def _ui_segment_frame_counts(plan) -> list[int]:
    """UI / plan segment lengths (fallback when export chunk lengths unavailable)."""
    counts = []
    for seg in list(getattr(plan, "segments", None) or []):
        fc = int(getattr(seg, "frame_count", 0) or 0)
        if fc <= 0:
            fc = max(
                0,
                int(getattr(seg, "end_frame", 0) or 0)
                - int(getattr(seg, "start_frame", 0) or 0),
            )
        counts.append(max(0, fc))
    return counts


def _segment_frame_counts_for_audio(
    plan,
    n_audios: int,
    total_frames: int,
    *,
    frame_counts: list[int] | None = None,
) -> list[int]:
    """Per-segment frame lengths used to stitch generated audio under「全部导出」.

    Prefer explicit ``frame_counts`` (actual export chunk lengths). Never silently
    even-split the whole timeline across a partial audio list — that desyncs A/V
    after「选择运行」partial re-runs.
    """
    if frame_counts is not None and len(frame_counts) == n_audios:
        return [max(0, int(c)) for c in frame_counts]

    ui_counts = _ui_segment_frame_counts(plan)
    if len(ui_counts) == n_audios and sum(ui_counts) > 0:
        return ui_counts

    if n_audios <= 0:
        return []

    log.warning(
        "Audio clip count (%d) does not match export/plan segment counts "
        "(export_counts=%s, plan_segments=%d). Filling missing slots with silence "
        "by UI segment lengths where possible — re-run all segments once to refresh "
        "audio cache if sync looks wrong.",
        n_audios,
        None if frame_counts is None else len(frame_counts),
        len(ui_counts),
    )
    if ui_counts:
        # Align by index into the plan; pad/truncate to n_audios.
        out = list(ui_counts[:n_audios])
        while len(out) < n_audios:
            out.append(ui_counts[-1] if ui_counts else 0)
        return out

    # Last resort only when plan has no segments.
    n = max(1, n_audios)
    base = max(0, int(total_frames)) // n
    rem = max(0, int(total_frames)) - base * n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _merge_generated_segment_audios(
    plan,
    segment_audios: list,
    *,
    total_frames: int,
    fps: float,
    frame_counts: list[int] | None = None,
) -> dict[str, Any]:
    """Concatenate per-segment AV audio into one timeline for merged video export."""
    sr = SILENT_SAMPLE_RATE
    for gen in segment_audios:
        if _audio_has_samples(gen):
            sr = int(gen.get("sample_rate") or sr) or SILENT_SAMPLE_RATE
            break
    counts = _segment_frame_counts_for_audio(
        plan, len(segment_audios), total_frames, frame_counts=frame_counts,
    )
    parts: list[torch.Tensor] = []
    max_ch = 2
    for i, gen in enumerate(segment_audios):
        fc = counts[i] if i < len(counts) else 0
        part = _pad_or_trim_audio_to_frames(
            gen if _audio_has_samples(gen) else None,
            frame_count=fc,
            fps=fps,
            sample_rate=sr,
        )
        wave = part.get("waveform")
        if isinstance(wave, torch.Tensor) and wave.ndim == 3 and wave.numel() > 0:
            max_ch = max(max_ch, int(wave.shape[1]))
            parts.append(wave)
        else:
            n = frames_to_audio_samples(fc, fps, sr)
            parts.append(torch.zeros(1, 2, max(0, n)))
    if not parts:
        return empty_audio_dict(sr)
    parts = [_align_audio_channels(p, max_ch) for p in parts]
    merged = torch.cat(parts, dim=-1)
    return _pad_or_trim_audio_to_frames(
        {"waveform": merged, "sample_rate": sr},
        frame_count=total_frames,
        fps=fps,
        sample_rate=sr,
    )


def build_director_audio_outputs(
    plan,
    images_out: list,
    *,
    export_segments: bool,
    output_frame_end: int | None = None,
    segment_audios: list | None = None,
    segment_frame_counts: list[int] | None = None,
    audio_mode: str | None = None,
    mute_audio: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return (AUDIO list, source_fallback).

    source_fallback is None or "silent" when audioMode=source could not
    extract a usable track (falls back to mute, never model audio).

    ``segment_frame_counts`` should match ``segment_audios`` and the export
    chunk lengths (post continuity trim) so partial re-runs stay in sync.
    """
    fps = float(plan.frame_rate or 24.0)
    mode = AUDIO_MODE_MUTE if mute_audio else (audio_mode or resolve_audio_mode(plan))
    log.info("Director audio mode: %s (task=%s)", mode, getattr(plan, "global_task_key", ""))

    if mode == AUDIO_MODE_MUTE:
        return [empty_audio_dict(SILENT_SAMPLE_RATE) for _ in images_out], None

    # Prefer model audio only in generate mode.
    if mode == AUDIO_MODE_GENERATE and segment_audios:
        # 「全部导出」合并画面时 images_out 只有 1 条，必须把各组音频按时间轴拼接，
        # 否则只会用第 1 组音频并静音填充后半段（第二组及之后无声）。
        # Also merge when a single clip is present but we still have a full
        # timeline audio table (partial re-run + cached audio restore).
        merge_all = (
            len(images_out) == 1
            and (
                len(segment_audios) > 1
                or (
                    segment_frame_counts is not None
                    and len(segment_frame_counts) == len(segment_audios)
                    and len(segment_audios) >= 1
                    and len(list(getattr(plan, "segments", None) or [])) > 1
                )
            )
        )
        if merge_all:
            n_frames = int(getattr(images_out[0], "shape", [0])[0] or 0)
            if n_frames <= 0:
                n_frames = int(output_frame_end or getattr(plan, "total_frames", 0) or 0)
            merged = _merge_generated_segment_audios(
                plan,
                segment_audios,
                total_frames=n_frames,
                fps=fps,
                frame_counts=segment_frame_counts,
            )
            if _audio_has_samples(merged):
                log.info(
                    "Merged %d segment audio clip(s) for export=all (%d frames).",
                    len(segment_audios), n_frames,
                )
                return [merged], None

        outputs: list[dict[str, Any]] = []
        for i, tensor in enumerate(images_out):
            gen = segment_audios[i] if i < len(segment_audios) else None
            if _audio_has_samples(gen):
                n_frames = int(getattr(tensor, "shape", [0])[0] or 0)
                sr = int(gen.get("sample_rate") or SILENT_SAMPLE_RATE)
                outputs.append(
                    _pad_or_trim_audio_to_frames(gen, frame_count=n_frames, fps=fps, sample_rate=sr)
                )
            else:
                outputs.append(empty_audio_dict(SILENT_SAMPLE_RATE))
        if any(_audio_has_samples(a) for a in outputs):
            return outputs, None

    silent_sample_rate = SILENT_SAMPLE_RATE
    # source mode always extracts; generate falls back to source when task allows.
    if mode != AUDIO_MODE_SOURCE and not task_passes_source_audio(plan.global_task_key):
        return [empty_audio_dict(silent_sample_rate) for _ in images_out], None

    timeline = plan.raw or {}
    source_fallback: str | None = None

    if export_segments:
        if plan.run_indices is not None:
            seg_indices = sorted(plan.run_indices)
        else:
            seg_indices = list(range(len(plan.segments)))
        outputs = []
        for i, tensor in enumerate(images_out):
            if i >= len(seg_indices):
                outputs.append(empty_audio_dict(silent_sample_rate))
                continue
            seg = plan.segments[seg_indices[i]]
            extracted = extract_timeline_audio(timeline, seg.start_frame, seg.end_frame, fps)
            if mode == AUDIO_MODE_SOURCE and not _audio_has_samples(extracted):
                extracted = _ref_audio_for_segment(seg) or extracted
            if mode == AUDIO_MODE_SOURCE and not _audio_has_samples(extracted):
                hint = diagnose_source_audio_failure(
                    timeline, seg.start_frame, seg.end_frame, fps
                )
                # Soft fallback: silent only (do not substitute model audio).
                log.warning(
                    "声音=使用原声，但片段 #%s 无源音轨（%s）；已回退为静音。",
                    seg.index + 1, hint,
                )
                extracted = None
                source_fallback = "silent"
            audio = _coerce_audio_output(extracted, sample_rate=silent_sample_rate)
            n_frames = int(getattr(tensor, "shape", [0])[0] or seg.frame_count or 0)
            sr = int(audio.get("sample_rate") or silent_sample_rate)
            outputs.append(
                _pad_or_trim_audio_to_frames(
                    audio, frame_count=n_frames, fps=fps, sample_rate=sr
                )
            )
        return outputs, source_fallback

    end = max(0, int(output_frame_end if output_frame_end is not None else plan.total_frames))
    if images_out and hasattr(images_out[0], "shape"):
        end = max(end, int(images_out[0].shape[0]))
    extracted = extract_timeline_audio(timeline, 0, end, fps) if end > 0 else None
    if mode == AUDIO_MODE_SOURCE and not _audio_has_samples(extracted):
        # Multi-segment source: use each segment's own reference audio, joined
        # along the timeline (segments without a ref audio become silence).
        refs = [_ref_audio_for_segment(seg) for seg in (getattr(plan, "segments", None) or [])]
        if any(a is not None for a in refs):
            extracted = _merge_generated_segment_audios(
                plan, refs, total_frames=end, fps=fps,
            )
    if mode == AUDIO_MODE_SOURCE and not _audio_has_samples(extracted):
        hint = diagnose_source_audio_failure(timeline, 0, end, fps)
        # Soft fallback after a successful video run: silent only (no model audio).
        log.warning(
            "声音=使用原声，但源视频无音轨（%s）；已回退为静音。",
            hint,
        )
        extracted = None
        source_fallback = "silent"
    merged = _coerce_audio_output(extracted, sample_rate=silent_sample_rate)
    sr = int(merged.get("sample_rate") or silent_sample_rate)
    merged = _pad_or_trim_audio_to_frames(
        merged, frame_count=end, fps=fps, sample_rate=sr
    )
    out = [merged] if len(images_out) == 1 else [empty_audio_dict(silent_sample_rate) for _ in images_out]
    return out, source_fallback


def source_audio_report_note(
    plan,
    audio_out: list,
    *,
    export_segments: bool,
    output_frame_end: int | None = None,
    used_generated_audio: bool = False,
    audio_mode: str | None = None,
    source_fallback: str | None = None,
) -> str:
    mode = audio_mode or resolve_audio_mode(plan)
    if mode == AUDIO_MODE_MUTE:
        return "\n\nAudio: muted (silent output)."
    if mode == AUDIO_MODE_SOURCE:
        if source_fallback == "silent" or not any(_audio_has_samples(a) for a in audio_out):
            fps = float(plan.frame_rate or 24.0)
            timeline = plan.raw or {}
            if export_segments and plan.segments:
                if plan.run_indices is not None:
                    seg_indices = sorted(plan.run_indices)
                else:
                    seg_indices = list(range(len(plan.segments)))
                seg = plan.segments[seg_indices[0]] if seg_indices else plan.segments[0]
                start, end = int(seg.start_frame), int(seg.end_frame)
            else:
                start = 0
                end = max(
                    0,
                    int(output_frame_end if output_frame_end is not None else plan.total_frames),
                )
            hint = diagnose_source_audio_failure(timeline, start, end, fps)
            return f"\n\nSource audio: none — {hint} (fell back to silent)."
        return (
            "\n\nSource audio: extracted from input video "
            "(frame-aligned PCM cut, length = picture / timeline fps)."
        )
    if used_generated_audio and any(_audio_has_samples(a) for a in audio_out):
        return (
            "\n\nGenerated audio: decoded from MiniMax H3 AV latent "
            "(frame-aligned to picture / timeline fps)."
        )
    if mode != AUDIO_MODE_SOURCE and not task_passes_source_audio(plan.global_task_key):
        return ""
    if mode != AUDIO_MODE_SOURCE and any(_audio_has_samples(a) for a in audio_out):
        return (
            "\n\nSource audio: extracted from input video "
            "(frame-aligned PCM cut, length = picture / timeline fps)."
        )

    fps = float(plan.frame_rate or 24.0)
    timeline = plan.raw or {}
    if export_segments and plan.segments:
        if plan.run_indices is not None:
            seg_indices = sorted(plan.run_indices)
        else:
            seg_indices = list(range(len(plan.segments)))
        seg = plan.segments[seg_indices[0]] if seg_indices else plan.segments[0]
        start, end = int(seg.start_frame), int(seg.end_frame)
    else:
        start = 0
        end = max(0, int(output_frame_end if output_frame_end is not None else plan.total_frames))
    hint = diagnose_source_audio_failure(timeline, start, end, fps)
    return f"\n\nSource audio: none — {hint}."
