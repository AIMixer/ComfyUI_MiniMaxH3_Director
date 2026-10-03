"""WebSocket progress updates for MiniMax H3 Director multi-segment runs."""

from __future__ import annotations

import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

DIRECTOR_PHASES = (
    "prepare",
    "context_encode",
    "sample",
    "upscale",
    "refine",
    "decode",
)

PHASE_LABELS = {
    "prepare": "准备片段",
    "context_encode": "H3 条件编码",
    "sample": "采样",
    "upscale": "放大",
    "refine": "精修采样",
    "decode": "AV 解码",
    "plan": "解析时间轴 / 加载视频",
    "finish": "全部完成",
}


def _phase_index(phase: str) -> int:
    try:
        return DIRECTOR_PHASES.index(phase)
    except ValueError:
        return 0


def report_director_progress(
    node_id: str | None,
    *,
    segment_index: int,
    segment_total: int,
    phase: str,
    phase_value: float = 0,
    phase_max: float = 1,
    frames_label: str = "",
    task_key: str = "",
    timeline_segment_index: int | None = None,
    timeline_segment_total: int | None = None,
) -> None:
    if not node_id:
        return

    phases_per = len(DIRECTOR_PHASES)
    overall_max = max(1, segment_total * phases_per)
    phase_fraction = max(0.0, min(1.0, phase_value / max(phase_max, 1)))
    overall_value = min(
        overall_max,
        segment_index * phases_per + _phase_index(phase) + phase_fraction,
    )

    remaining_segments = max(0, segment_total - segment_index - 1)
    if phase == "finish":
        overall_value = overall_max
        remaining_segments = 0

    timeline_seg = (
        timeline_segment_index + 1
        if timeline_segment_index is not None
        else segment_index + 1
    )
    timeline_total = timeline_segment_total if timeline_segment_total is not None else segment_total
    partial_run = (
        timeline_segment_total is not None
        and segment_total < timeline_segment_total
    )

    payload = {
        "node_id": str(node_id),
        "segment": segment_index + 1,
        "segment_total": segment_total,
        "timeline_segment": timeline_seg,
        "timeline_segment_total": timeline_total,
        "partial_run": partial_run,
        "phase": phase,
        "phase_label": PHASE_LABELS.get(phase, phase),
        "phase_value": phase_value,
        "phase_max": phase_max,
        "overall_value": overall_value,
        "overall_max": overall_max,
        "remaining_segments": remaining_segments,
        "frames_label": frames_label,
        "task_key": task_key,
    }

    try:
        from server import PromptServer

        srv = PromptServer.instance
        if srv:
            srv.send_sync("minimax_director_progress", payload, srv.client_id)
            srv.send_progress_text("", str(node_id))
    except Exception as exc:
        log.debug("Director progress send skipped: %s", exc)

    try:
        from comfy_execution.progress import get_progress_state

        get_progress_state().update_progress(str(node_id), overall_value, overall_max)
    except Exception:
        pass


def report_director_segment_preview(
    node_id: str | None,
    *,
    segment_index: int,
    image_b64: str,
    width: int = 0,
    height: int = 0,
    frames: list[str] | None = None,
    fps: float = 24.0,
    live: bool = False,
    step: int | None = None,
    total_steps: int | None = None,
    mime: str | None = None,
) -> None:
    if not node_id or not image_b64:
        return
    payload = {
        "node_id": str(node_id),
        "segment_index": segment_index,
        "image_b64": image_b64,
        "width": width,
        "height": height,
        "live": bool(live),
    }
    if mime:
        payload["mime"] = str(mime)
    if frames:
        payload["frames"] = frames
        payload["fps"] = fps
    elif fps and mime in ("image/webp", "video/mp4"):
        payload["fps"] = fps
    if step is not None:
        payload["step"] = int(step)
    if total_steps is not None:
        payload["total_steps"] = int(total_steps)
    try:
        from server import PromptServer

        srv = PromptServer.instance
        if srv:
            srv.send_sync("minimax_director_preview", payload, srv.client_id)
    except Exception as exc:
        log.debug("Director preview send skipped: %s", exc)


def report_director_video(
    node_id: str | None,
    *,
    kind: str,
    subfolder: str = "",
    filename: str = "",
    fps: float = 24.0,
    frame_count: int = 0,
    segment_index: int | None = None,
    segments: list[dict[str, int]] | None = None,
    track: list[dict] | None = None,
    entries: list[dict] | None = None,
) -> None:
    """推送已导出的片段 / 完整视频播放列表，让导演台内嵌预览立即播放。

    kind:
      * ``"playlist"`` — 出片预览播放列表：``entries`` 为时间轴每段一条
        ``{"index","frames","status","subfolder","filename"}``；missing 段用代码生成的
        占位片补位。前端 <video> 逐段连播（不再拼 merged_latest.mp4 → 零磁盘累积）。
      * ``"segment"`` — 单段刚写完（前端当成只有一条的播放列表）。
    subfolder/filename 对应 output 目录下的 ``minimax_seg_export/<ts>/<file>``，
    前端用 ``/view?filename=..&subfolder=..&type=output`` 播放。

    ``segments`` = 该视频的实际构成 ``[{"index": 段序号, "frames": 帧数}]``。
    ``track`` = 时间轴**每一段**的渲染状态（``status`` ∈ ``"ok"/"missing"``）。
    ``entries`` = 完整播放列表（kind="playlist" 时使用，可独立成事件、无需 filename）。
    """
    if not node_id:
        return
    if not filename and not entries:
        return
    payload = {
        "node_id": str(node_id),
        "kind": str(kind),
        "subfolder": str(subfolder),
        "filename": str(filename),
        "fps": float(fps or 24.0),
        "frame_count": int(frame_count or 0),
    }
    if segment_index is not None:
        payload["segment_index"] = int(segment_index)
    if segments:
        payload["segments"] = [
            {"index": int(s.get("index", -1)), "frames": int(s.get("frames", 0) or 0)}
            for s in segments
        ]
    if track:
        payload["track"] = [
            {
                "index": int(t.get("index", -1)),
                "frames": int(t.get("frames", 0) or 0),
                "status": str(t.get("status", "ok") or "ok"),
            }
            for t in track
        ]
    if entries:
        payload["entries"] = [
            {
                "index": int(e.get("index", -1)),
                "frames": int(e.get("frames", 0) or 0),
                "status": str(e.get("status", "ok") or "ok"),
                "subfolder": str(e.get("subfolder") or ""),
                "filename": str(e.get("filename") or ""),
            }
            for e in entries
        ]
    try:
        from server import PromptServer

        srv = PromptServer.instance
        if srv:
            srv.send_sync("minimax_director_video", payload, srv.client_id)
    except Exception as exc:
        log.debug("Director video event send skipped: %s", exc)


def report_director_finish(node_id: str | None, segment_total: int) -> None:
    report_director_progress(
        node_id,
        segment_index=max(0, segment_total - 1),
        segment_total=max(1, segment_total),
        phase="finish",
        phase_value=1,
        phase_max=1,
    )


def report_director_planning(
    node_id: str | None,
    segment_total: int = 1,
    *,
    timeline_segment_total: int | None = None,
) -> None:
    report_director_progress(
        node_id,
        segment_index=0,
        segment_total=max(1, segment_total),
        phase="plan",
        phase_value=0,
        phase_max=1,
        timeline_segment_total=timeline_segment_total,
    )
