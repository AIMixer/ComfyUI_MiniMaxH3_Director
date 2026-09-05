"""Shared MiniMax H3 timed-guide model and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch


# MiniMax H3 keyframe positions are always pixel-frame indices on a native
# 24 fps timebase. Director's editable/output FPS is intentionally unrelated.
H3_NATIVE_FPS = 24


@dataclass
class SegmentTimedGuide:
    """One image guide anchored inside a single H3 segment sampling."""

    id: str
    frame_index: int
    tensor: torch.Tensor | None = None
    image_file: str = ""
    image_identity: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class SegmentTimedAudioGuide:
    """One standalone audio guide anchored inside a single H3 segment."""

    id: str
    frame_index: int
    audio: dict[str, Any] | None = None
    audio_file: str = ""
    audio_path: str = ""
    audio_identity: str = ""
    source_duration_sec: float = 0.0
    effective_duration_sec: float = 0.0
    meta: dict = field(default_factory=dict)


def guide_display_seconds(frame_index: int) -> float:
    return int(frame_index) / float(H3_NATIVE_FPS)


def parse_timed_guides(
    raw_guides,
    *,
    load_image: Callable[[dict], torch.Tensor | None],
    image_identity: Callable[[dict], tuple[str, str]],
) -> list[SegmentTimedGuide]:
    """Parse timeline JSON without coupling the shared model to ComfyUI paths."""
    if not isinstance(raw_guides, list):
        raise ValueError("AddGuide timedGuides must be an array.")
    guides: list[SegmentTimedGuide] = []
    for raw in raw_guides:
        if not isinstance(raw, dict):
            raise ValueError("Each AddGuide timedGuides item must be an object.")
        guide_id = str(raw.get("id") or raw.get("guideId") or "").strip()
        frame_raw = raw.get("frameIndex", raw.get("frame_index"))
        if isinstance(frame_raw, bool) or frame_raw is None:
            raise ValueError(
                f"AddGuide Guide '{guide_id or '?'}' needs an integer frameIndex."
            )
        try:
            frame_index = int(frame_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"AddGuide Guide '{guide_id or '?'}' needs an integer frameIndex."
            ) from exc
        if isinstance(frame_raw, float) and not frame_raw.is_integer():
            raise ValueError(
                f"AddGuide Guide '{guide_id or '?'}' frameIndex must be an integer."
            )
        if isinstance(frame_raw, str) and str(frame_index) != frame_raw.strip():
            raise ValueError(
                f"AddGuide Guide '{guide_id or '?'}' frameIndex must be an integer."
            )
        image_raw = raw.get("image") if isinstance(raw.get("image"), dict) else raw
        image_file, identity = image_identity(image_raw)
        guides.append(
            SegmentTimedGuide(
                id=guide_id,
                frame_index=frame_index,
                tensor=load_image(image_raw),
                image_file=image_file,
                image_identity=identity,
                meta=dict(image_raw),
            )
        )
    return guides


def timed_guides_fingerprint(guides: Iterable[SegmentTimedGuide]) -> list[dict]:
    """Stable cache material ordered by the guide's actual H3 frame."""
    return [
        {
            "id": str(guide.id or ""),
            "frame_index": int(guide.frame_index),
            "image": str(guide.image_identity or guide.image_file or "missing"),
        }
        for guide in sorted(
            list(guides), key=lambda item: (int(item.frame_index), str(item.id))
        )
    ]


def parse_timed_audio_guides(
    raw_guides,
    *,
    audio_source: Callable[[dict], tuple[str, str, str, float]],
) -> list[SegmentTimedAudioGuide]:
    """Parse independent timeline ``timedAudioGuides`` records lazily."""
    if not isinstance(raw_guides, list):
        raise ValueError("AddGuide timedAudioGuides must be an array.")
    guides: list[SegmentTimedAudioGuide] = []
    for raw in raw_guides:
        if not isinstance(raw, dict):
            raise ValueError("Each AddGuide timedAudioGuides item must be an object.")
        guide_id = str(raw.get("id") or raw.get("guideId") or "").strip()
        frame_raw = raw.get("frameIndex", raw.get("frame_index"))
        if isinstance(frame_raw, bool) or frame_raw is None:
            raise ValueError(
                f"AddGuide Audio Guide '{guide_id or '?'}' needs an integer frameIndex."
            )
        try:
            frame_index = int(frame_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"AddGuide Audio Guide '{guide_id or '?'}' needs an integer frameIndex."
            ) from exc
        if isinstance(frame_raw, float) and not frame_raw.is_integer():
            raise ValueError(
                f"AddGuide Audio Guide '{guide_id or '?'}' frameIndex must be an integer."
            )
        if isinstance(frame_raw, str) and str(frame_index) != frame_raw.strip():
            raise ValueError(
                f"AddGuide Audio Guide '{guide_id or '?'}' frameIndex must be an integer."
            )
        audio_raw = raw.get("audio") if isinstance(raw.get("audio"), dict) else raw
        audio_file, audio_path, identity, duration = audio_source(audio_raw)
        guides.append(
            SegmentTimedAudioGuide(
                id=guide_id,
                frame_index=frame_index,
                audio_file=audio_file,
                audio_path=audio_path,
                audio_identity=identity,
                source_duration_sec=max(0.0, float(duration or 0.0)),
                meta=dict(audio_raw),
            )
        )
    return guides


def audio_guide_effective_duration(
    guide: SegmentTimedAudioGuide,
    *,
    next_frame_index: int | None,
    frame_count: int,
) -> float:
    """Duration handed to H3: source, next AG, and segment end all cap it."""
    start = int(guide.frame_index)
    remaining = max(0, int(frame_count) - start) / float(H3_NATIVE_FPS)
    handoff = remaining
    if next_frame_index is not None:
        handoff = max(0, int(next_frame_index) - start) / float(H3_NATIVE_FPS)
    source = max(0.0, float(guide.source_duration_sec or 0.0))
    return min(source, handoff, remaining)


def timed_audio_guides_fingerprint(
    guides: Iterable[SegmentTimedAudioGuide], *, frame_count: int
) -> list[dict]:
    """Stable first-pass cache material including AG handoff effects."""
    ordered = sorted(
        list(guides), key=lambda item: (int(item.frame_index), str(item.id))
    )
    out: list[dict] = []
    for index, guide in enumerate(ordered):
        next_frame = ordered[index + 1].frame_index if index + 1 < len(ordered) else None
        effective = audio_guide_effective_duration(
            guide, next_frame_index=next_frame, frame_count=frame_count
        )
        out.append(
            {
                "id": str(guide.id or ""),
                "frame_index": int(guide.frame_index),
                "audio": str(guide.audio_identity or guide.audio_file or "missing"),
                "source_duration_sec": round(float(guide.source_duration_sec or 0.0), 6),
                "effective_duration_sec": round(effective, 6),
            }
        )
    return out


def trim_audio_guide(audio: dict[str, Any], duration_sec: float) -> dict[str, Any]:
    """Return a real PCM crop for the official AddGuide audio encoder."""
    wave = audio.get("waveform") if isinstance(audio, dict) else None
    sample_rate = int(audio.get("sample_rate") or 0) if isinstance(audio, dict) else 0
    if not isinstance(wave, torch.Tensor) or wave.ndim < 1 or sample_rate <= 0:
        raise ValueError("Audio Guide could not be decoded into a valid AUDIO value.")
    sample_count = max(0, int(round(max(0.0, float(duration_sec)) * sample_rate)))
    if sample_count <= 0:
        raise ValueError("Audio Guide has no usable duration at its frame position.")
    return {
        **audio,
        "waveform": wave[..., :sample_count].clone(),
        "sample_rate": sample_rate,
    }


def validate_timed_audio_guides(
    guides: Iterable[SegmentTimedAudioGuide],
    *,
    frame_count: int,
    segment_number: int | None = None,
) -> list[SegmentTimedAudioGuide]:
    """Validate audio guides independently and return them in H3 frame order."""
    prefix = f"AddGuide segment #{segment_number}" if segment_number else "AddGuide segment"
    count = int(frame_count)
    if count <= 0:
        raise ValueError(f"{prefix} has no valid H3 frames.")
    ordered = sorted(list(guides), key=lambda guide: (int(guide.frame_index), guide.id))
    ids: set[str] = set()
    frames: set[int] = set()
    last = count - 1
    for position, guide in enumerate(ordered, 1):
        if not guide.id:
            raise ValueError(f"{prefix} Audio Guide {position} is missing a stable id.")
        if guide.id in ids:
            raise ValueError(f"{prefix} has duplicate Audio Guide id '{guide.id}'.")
        ids.add(guide.id)
        frame = int(guide.frame_index)
        if frame < 0 or frame >= count:
            raise ValueError(
                f"{prefix} Audio Guide {position} is at F{frame}, outside F0..F{last}."
            )
        if frame in frames:
            raise ValueError(f"{prefix} has more than one Audio Guide at F{frame}.")
        frames.add(frame)
        if not guide.audio_file or not guide.audio_path:
            raise ValueError(
                f"{prefix} Audio Guide {position} at F{frame} has no audio."
            )
        if float(guide.source_duration_sec or 0.0) <= 0:
            raise ValueError(
                f"{prefix} Audio Guide {position} at F{frame} has no usable audio duration."
            )
    return ordered


def choose_default_guide_frame(
    frame_count: int,
    occupied: Iterable[int],
    *,
    first_present: bool = False,
    last_present: bool = False,
) -> int | None:
    """Choose the midpoint of the earliest largest gap between H3 endpoints.

    F0 and F(frame_count-1) are always layout boundaries. Interior frames are
    preferred; an unreserved endpoint is used only when no interior frame is
    left. This keeps repeated additions distributed across the whole segment.
    """
    count = max(0, int(frame_count))
    if count <= 0:
        return None
    last = count - 1
    used: set[int] = set()
    for value in occupied:
        try:
            frame = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= frame < count:
            used.add(frame)
    anchors = sorted({0, last, *used})
    best: tuple[int, int, int] | None = None
    for left, right in zip(anchors, anchors[1:]):
        free = right - left - 1
        if free <= 0:
            continue
        candidate = (left + right) // 2
        rank = (-free, left, candidate)
        if best is None or rank < best:
            best = rank
    if best is not None:
        return best[2]

    endpoint_candidates = []
    if not first_present and 0 not in used:
        endpoint_candidates.append(0)
    if not last_present and last not in used:
        endpoint_candidates.append(last)
    return min(endpoint_candidates) if endpoint_candidates else None


def validate_timed_guides(
    guides: Iterable[SegmentTimedGuide],
    *,
    frame_count: int,
    first_present: bool,
    last_present: bool,
    segment_number: int | None = None,
) -> list[SegmentTimedGuide]:
    """Validate and return guides sorted by their true 0-based frame index."""
    prefix = f"AddGuide segment #{segment_number}" if segment_number else "AddGuide segment"
    count = int(frame_count)
    if count <= 0:
        raise ValueError(f"{prefix} has no valid H3 frames.")

    ordered = sorted(list(guides), key=lambda guide: (int(guide.frame_index), guide.id))
    if not ordered:
        raise ValueError(f"{prefix} requires at least one intermediate Guide image.")

    ids: set[str] = set()
    frames: set[int] = set()
    last = count - 1
    for position, guide in enumerate(ordered, 1):
        if not guide.id:
            raise ValueError(f"{prefix} Guide {position} is missing a stable id.")
        if guide.id in ids:
            raise ValueError(f"{prefix} has duplicate Guide id '{guide.id}'.")
        ids.add(guide.id)

        frame = int(guide.frame_index)
        if frame < 0 or frame >= count:
            raise ValueError(
                f"{prefix} Guide {position} is at F{frame}, outside F0..F{last}."
            )
        if first_present and frame == 0:
            raise ValueError(f"{prefix} Guide {position} at F0 conflicts with First Frame.")
        if last_present and frame == last:
            raise ValueError(
                f"{prefix} Guide {position} at F{last} conflicts with Last Frame."
            )
        if frame in frames:
            raise ValueError(f"{prefix} has more than one Guide at F{frame}.")
        frames.add(frame)

        tensor = guide.tensor
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() <= 0:
            raise ValueError(f"{prefix} Guide {position} at F{frame} has no image.")
        if tensor.ndim != 4:
            raise ValueError(
                f"{prefix} Guide {position} at F{frame} must be one IMAGE tensor."
            )
        if int(tensor.shape[0]) != 1:
            raise ValueError(
                f"{prefix} Guide {position} at F{frame} is an IMAGE batch "
                f"({int(tensor.shape[0])} frames); P1 supports one image only."
            )

    return ordered
