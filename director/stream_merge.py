"""大时间轴磁盘流式拼接。"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import torch

from ..lib.image_prep import pad_frames_to_canvas
from .plan import DirectorPlan
from .segment_continuity import join_continuous_pair

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.stream_merge")

STREAM_MERGE_RAM_LIMIT = 2 * 1024 * 1024 * 1024


def _stream_root() -> Path:
    import folder_paths

    root = Path(folder_paths.get_temp_directory()) / "minimax_director_merge"
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 48 * 60 * 60
    for old in root.glob("*.raw"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass
    return root


def stream_merge_required(frame_counts: list[int], shape: tuple[int, int, int]) -> bool:
    """预计完整内存拼接超过保守上限时返回 True。"""
    h, w, c = (int(shape[0]), int(shape[1]), int(shape[2]))
    frames = sum(max(0, int(n)) for n in frame_counts)
    return int(frames) * max(1, h) * max(1, w) * max(1, c) * 4 >= STREAM_MERGE_RAM_LIMIT


class FrameBatchAccumulator:
    """将完成接缝处理的批次追加到 raw 文件，并返回 memmap 张量。"""

    def __init__(self, plan: DirectorPlan, *, label: str, node_id: str | None = None):
        self._plan = plan
        self._label = str(label or "merge")
        self._current: torch.Tensor | None = None
        self._current_index: int | None = None
        self._frames = 0
        self._shape: tuple[int, int, int] | None = None
        self._closed = False
        safe_node = "".join(ch for ch in str(node_id or "run") if ch.isalnum())[:32] or "run"
        self.path = _stream_root() / f"{self._label}_{safe_node}_{uuid4().hex}.raw"
        self._file = self.path.open("wb", buffering=1024 * 1024)

    def push(self, batch: torch.Tensor, source_index: int) -> int | None:
        """加入一个源批次；若有批次已写盘，则返回其源索引。"""
        if self._closed:
            raise RuntimeError("FrameBatchAccumulator is closed")
        if not isinstance(batch, torch.Tensor) or batch.ndim != 4:
            raise ValueError(
                f"Expected NHWC frames tensor, got {type(batch)} "
                f"shape={getattr(batch, 'shape', None)}"
            )
        if self._current is None:
            self._current = batch
            self._current_index = int(source_index)
            return None

        left, body = join_continuous_pair(self._current, batch, self._plan)
        self._write(left)
        flushed = self._current_index
        self._current = body
        self._current_index = int(source_index)
        return flushed

    def finish(self) -> tuple[torch.Tensor, int]:
        """写入最后一批，返回磁盘支持的只读 IMAGE 张量。"""
        if self._closed:
            raise RuntimeError("FrameBatchAccumulator is closed")
        if self._current is None or self._current_index is None:
            raise ValueError("No frames to merge")
        self._write(self._current)
        flushed = int(self._current_index)
        self._current = None
        self._current_index = None
        self._close_file()
        if self._shape is None or self._frames <= 0:
            raise ValueError("No frames were written")
        h, w, c = self._shape
        array = np.memmap(
            str(self.path),
            dtype=np.float32,
            mode="r+",
            shape=(self._frames, h, w, c),
        )
        log.info(
            "Director %s merge streamed to disk: %d frames, %s",
            self._label,
            self._frames,
            self.path,
        )
        return torch.from_numpy(array), flushed

    def abort(self) -> None:
        self._current = None
        self._current_index = None
        self._close_file()
        try:
            self.path.unlink()
        except OSError:
            pass

    def _close_file(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
        except OSError:
            pass
        try:
            self._file.close()
        except OSError:
            pass

    def _write(self, batch: torch.Tensor) -> None:
        if not isinstance(batch, torch.Tensor) or batch.ndim != 4:
            raise ValueError(
                f"Expected NHWC frames tensor, got {type(batch)} "
                f"shape={getattr(batch, 'shape', None)}"
            )
        batch = batch.detach()
        if int(batch.shape[0]) <= 0:
            return
        if batch.dtype != torch.float32:
            batch = batch.float()
        if getattr(batch, "device", None) is not None and batch.device.type != "cpu":
            batch = batch.cpu()

        h, w, c = int(batch.shape[1]), int(batch.shape[2]), int(batch.shape[3])
        if self._shape is None:
            self._shape = (h, w, c)
        target_h, target_w, target_c = self._shape
        if (h, w) != (target_h, target_w):
            batch = pad_frames_to_canvas(batch, target_w, target_h)
        if int(batch.shape[3]) != target_c:
            batch = batch[..., :target_c]
        if not batch.is_contiguous():
            batch = batch.contiguous()
        batch.numpy().tofile(self._file)
        self._frames += int(batch.shape[0])


def concat_continuous_chunks_to_disk(
    chunks: list[torch.Tensor],
    segments: list,
    plan: DirectorPlan,
    *,
    label: str = "merge",
    node_id: str | None = None,
    on_release: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """沿用原接缝逻辑合并批次，并在写盘后尽早释放源数据。"""
    del segments
    if not chunks:
        raise ValueError("concat_continuous_chunks_to_disk: no chunks")

    accumulator = FrameBatchAccumulator(plan, label=label, node_id=node_id)
    try:
        for index in range(len(chunks)):
            chunk = chunks[index]
            flushed = accumulator.push(chunk, index)
            del chunk
            if flushed is not None and on_release is not None:
                on_release(flushed)
        combined, flushed = accumulator.finish()
    except Exception:
        accumulator.abort()
        raise

    if on_release is not None:
        on_release(flushed)
    return combined
