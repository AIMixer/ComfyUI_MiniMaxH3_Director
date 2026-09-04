"""Shared MiniMax H3 timed-guide model and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

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
