"""One source audio track, addressed by absolute video-frame boundaries."""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import re

import av
import torch
import folder_paths

MASTER_KEY = "masterAudio"
MASTER_MODE = "master"
MASTER_TASKS = frozenset({"r2v", "mixed"})
MASTER_FPS = 24
MASTER_EXTS = frozenset({".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"})
MAX_SECONDS = 15 * 60
MAX_FILE_BYTES = 160 * 1024 * 1024


def master_record(timeline):
    record = (timeline or {}).get(MASTER_KEY)
    return record if isinstance(record, dict) and record.get("audioFile") else None


def resolve_master_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError("无效的全局音轨路径。")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in MASTER_EXTS:
        raise ValueError("全局音轨必须是 ComfyUI input 内支持的音频文件。")
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("找不到全局音轨，请重新上传或导入导演包。")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("全局音轨文件不能超过 160 MiB。")
    return path


def master_file_signature(timeline_data):
    try:
        data = json.loads(timeline_data or "{}")
        record = master_record(data) if isinstance(data, dict) else None
        if record:
            path = resolve_master_path(record["audioFile"])
            stat = path.stat()
            return f"{record['audioFile']}:{stat.st_size}:{stat.st_mtime_ns}"
    except (ValueError, OSError):
        return "missing master"
    return ""


def decode_master(path):
    """Decode in source sample rate, retaining mono/stereo. No fades or time scaling."""
    chunks = []
    count = 0
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("文件没有可解码的音轨。")
        stream = container.streams.audio[0]
        rate = int(stream.codec_context.sample_rate or 0)
        if not 8000 <= rate <= 192000:
            raise ValueError("全局音轨采样率须在 8–192 kHz 之间。")
        resampler = av.AudioResampler(
            format="fltp", rate=rate,
            layout="mono" if stream.codec_context.channels == 1 else "stereo",
        )
        def collect(frames):
            nonlocal count
            for frame in frames:
                count += frame.samples
                if count > min(MAX_SECONDS * rate, (256 * 1024 * 1024) // (4 * len(frame.layout.channels))):
                    raise ValueError("全局音轨超过 15 分钟或解码音频过大。")
                chunks.append(torch.from_numpy(frame.to_ndarray()))

        for frame in container.decode(audio=0):
            collect(resampler.resample(frame))
        collect(resampler.resample(None))
    if not chunks or count <= 0:
        raise ValueError("全局音轨为空。")
    waveform = torch.cat(chunks, dim=1).unsqueeze(0)
    if not torch.isfinite(waveform).all():
        raise ValueError("全局音轨含无效采样。")
    return {"waveform": waveform, "sample_rate": rate}


def master_peaks(audio, limit=24000):
    """Return peak amplitude and RMS per bucket in full-scale PCM units."""
    wave = audio["waveform"]
    count = int(wave.shape[-1])
    stride = max(1, math.ceil(count / limit))
    full = count // stride * stride
    blocks = wave[..., :full].reshape(wave.shape[1], -1, stride)
    # Include all channels; opposite-phase stereo should not appear silent.
    peaks = torch.maximum(blocks.amax(dim=(0, 2)), -blocks.amin(dim=(0, 2)))
    # Bound the temporary squared PCM buffer for long/high-rate tracks.
    batch = max(1, 1048576 // (stride * wave.shape[1]))
    rms = torch.cat([part.square().mean(dim=(0, 2)).sqrt() for part in blocks.split(batch, dim=1)])
    if full < count:
        tail = wave[..., full:]
        peaks = torch.cat((peaks, torch.maximum(tail.max(), -tail.min()).view(1)))
        rms = torch.cat((rms, tail.square().mean().sqrt().view(1)))

    # Finer than the legacy byte envelope so faint sounds survive visual zoom.
    scale = 32767
    def encode(values):
        return values.clamp(0, 1).mul(scale).round().to(torch.int16).tolist()

    return {"duration": count / audio["sample_rate"], "samples": count, "stride": stride,
            "scale": scale, "peaks": encode(peaks), "rms": encode(rms)}


def slice_master(audio, start_frame, end_frame, fps):
    """Round absolute boundaries once, so neighboring windows share a sample edge."""
    rate = audio["sample_rate"]
    start = round(int(start_frame) * rate / fps)
    end = round(int(end_frame) * rate / fps)
    wave = audio["waveform"][..., start:end]
    if wave.shape[-1] < end - start:
        wave = torch.nn.functional.pad(wave, (0, end - start - wave.shape[-1]))
    return {"waveform": wave, "sample_rate": rate}


def attach_master(plan, record, *, decode=True):
    if not record or plan.global_task_key not in MASTER_TASKS:
        return
    if float(plan.frame_rate) != MASTER_FPS:
        raise ValueError("全局音轨模式请使用 24 fps（H3 的原生口型时钟）。")
    from .plan import SegmentRefAudio

    path = resolve_master_path(record["audioFile"])
    if decode:
        plan.master_audio = decode_master(path)
    stat = path.stat()
    plan.master_audio_signature = f"{record['audioFile']}:{stat.st_size}:{stat.st_mtime_ns}"
    # H3 continuity can discard/extend frames at boundaries; fixed song time must win.
    plan.continuity_enabled = False
    for seg in plan.segments:
        if seg.task_key == "r2v":
            reference = (slice_master(plan.master_audio, seg.start_frame, seg.end_frame, plan.frame_rate)
                         if decode else None)
            if reference is not None and reference["waveform"].shape[1] == 1:
                # H3 audio VAE is stereo; keep the mono source unchanged at export.
                reference["waveform"] = reference["waveform"].repeat(1, 2, 1)
            seg.ref_audios = [SegmentRefAudio(index=0, audio=reference, audio_file=record["audioFile"])]
            # Do not let a paired reference-video soundtrack become <Audio 1>.
            seg.ref_video_audios = []


def master_prompt(prompt):
    # The user keeps control of visuals; previous per-shot audio labels no longer apply.
    prompt = re.sub(r"<audio\s+\d+\s*>\s*(?:fully_copy|reference)?", "", prompt, flags=re.I).strip()
    return f"<Audio 1> fully_copy. Match the timing and lip movements to <Audio 1>.\n{prompt}"


def export_master_audio(plan, images_out, split, released_slots=()):
    if not split:
        if len(images_out) != 1 or int(images_out[0].shape[0]) != plan.total_frames:
            raise ValueError("全局音轨导出失败：整片帧数与时间轴不符。")
        return [slice_master(plan.master_audio, 0, plan.total_frames, plan.frame_rate)]
    segments = [s for s in plan.segments if plan.run_indices is None or s.index in plan.run_indices]
    released = set(released_slots)
    segments = [s for i, s in enumerate(segments) if i not in released]
    if len(segments) != len(images_out) or any(
        int(images.shape[0]) != seg.frame_count for seg, images in zip(segments, images_out)
    ):
        raise ValueError("全局音轨导出失败：无法把视频批次与分镜时间对齐。")
    return [slice_master(plan.master_audio, seg.start_frame, seg.end_frame, plan.frame_rate)
            for seg in segments]
