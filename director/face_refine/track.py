"""Per-frame face track + crop.

Adapted from ComfyUI-H3-FaceRefine (MIT, Carasibana): detection, gap fill,
trajectory smoothing, and sub-pixel affine crop. Identity / SAM / cut-picker
UI are omitted; this pass follows one subject (largest or most central face)
with nearest-box continuity.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.face_refine")

_DETECTOR_CACHE: dict[str, Any] = {}


def load_detector(name: str):
    if name in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[name]
    import folder_paths

    path = None
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path:
            break
    if path is None:
        base = getattr(folder_paths, "models_dir", "models")
        for sub in ("ultralytics/bbox", "ultralytics", "ultralytics/segm"):
            cand = os.path.join(base, *sub.split("/"), name)
            if os.path.isfile(cand):
                path = cand
                break
    if path is None:
        raise FileNotFoundError(
            f"FaceRefine 检测器 '{name}' 未找到。请将 face_yolov8m.pt 放到 "
            "models/ultralytics/bbox/（或安装 ultralytics 后放入该目录）。"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "FaceRefine 需要 ultralytics。请执行: pip install ultralytics"
        ) from exc
    model = YOLO(path)
    _DETECTOR_CACHE[name] = model
    return model


def _to_bgr_u8(img: torch.Tensor) -> np.ndarray:
    arr = (img[..., :3].clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    return arr[..., ::-1].copy()


def _interp_gaps(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = len(vals)
    idx = np.arange(n)
    if not valid.any():
        return np.zeros(n, dtype=np.float64)
    return np.interp(idx, idx[valid], vals[valid])


def _smooth(vals: np.ndarray, window: int, method: str = "gaussian") -> np.ndarray:
    if window <= 1 or len(vals) < 3:
        return vals
    window = min(int(window), len(vals))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return vals
    pad = window // 2
    padded = np.pad(vals, pad, mode="reflect")
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter

            polyorder = 2 if window > 3 else 1
            return np.asarray(savgol_filter(padded, window, polyorder))[pad : pad + len(vals)]
        except Exception:
            method = "gaussian"
    if method == "gaussian":
        x = np.arange(window, dtype=np.float64) - pad
        sigma = max(window / 6.0, 0.5)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kernel /= kernel.sum()
    else:
        kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")[: len(vals)]


def affine_crop(img: torch.Tensor, box: tuple, cw: int, ch: int) -> torch.Tensor:
    """Sub-pixel crop+resize. img [1,H,W,C] -> [1,ch,cw,C]."""
    x, y, bw, bh = box
    _, height, width, _ = img.shape
    src = img[..., :3].movedim(-1, 1).float()
    theta = torch.tensor(
        [
            [
                [bw / width, 0.0, (2.0 * x + bw) / width - 1.0],
                [0.0, bh / height, (2.0 * y + bh) / height - 1.0],
            ]
        ],
        dtype=torch.float32,
        device=src.device,
    )
    grid = F.affine_grid(theta, (1, 3, int(ch), int(cw)), align_corners=False)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.movedim(1, -1).to(img.dtype)


def gaussian_blur_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    if feather <= 0:
        return mask
    k = 2 * int(feather) + 1
    shortest = min(mask.shape[-2], mask.shape[-1])
    if shortest <= k:
        k = max(3, int(shortest / 2) | 1)
    sigma = max(k / 6.0, 0.5)
    x = torch.arange(k, device=mask.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(mask.dtype)
    pad = k // 2
    m = F.conv2d(F.pad(mask, (pad, pad, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    m = F.conv2d(F.pad(m, (0, 0, pad, pad), mode="replicate"), g.view(1, 1, k, 1))
    return m


def face_region_mask(
    ch: int,
    cw: int,
    face_rect,
    dilation: int,
    feather: int,
    shape: str,
    device,
    dtype,
) -> torch.Tensor:
    m = torch.zeros((1, 1, int(ch), int(cw)), device=device, dtype=torch.float32)
    fx, fy, fwd, fhd = face_rect
    fx -= dilation
    fy -= dilation
    fwd += 2 * dilation
    fhd += 2 * dilation
    if shape == "ellipse":
        yy = torch.arange(ch, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(cw, device=device, dtype=torch.float32).view(1, -1)
        ccx, ccy = fx + fwd / 2.0, fy + fhd / 2.0
        rx, ry = max(fwd / 2.0, 1.0), max(fhd / 2.0, 1.0)
        m[0, 0] = (((xx - ccx) / rx) ** 2 + ((yy - ccy) / ry) ** 2 <= 1.0).float()
    else:
        x0 = max(0, int(round(fx)))
        y0 = max(0, int(round(fy)))
        x1 = min(int(cw), int(round(fx + fwd)))
        y1 = min(int(ch), int(round(fy + fhd)))
        if x1 > x0 and y1 > y0:
            m[0, 0, y0:y1, x0:x1] = 1.0
    return gaussian_blur_mask(m, feather).clamp(0, 1).to(dtype)


def feather_edge_mask(h: int, w: int, feather: int, device, dtype) -> torch.Tensor:
    m = torch.ones((h, w), device=device, dtype=dtype)
    f = int(max(0, min(feather, min(h, w) // 2 - 1)))
    if f <= 0:
        return m
    ramp = 0.5 - 0.5 * torch.cos(
        torch.linspace(0, np.pi, f + 2, device=device, dtype=dtype)[1:-1]
    )
    m[:f, :] *= ramp.view(-1, 1)
    m[h - f :, :] *= ramp.flip(0).view(-1, 1)
    m[:, :f] *= ramp.view(1, -1)
    m[:, w - f :] *= ramp.flip(0).view(1, -1)
    return m


def _rank_box(box, frame_w: int, frame_h: int, select: str) -> float:
    x0, y0, x1, y1 = box
    h = max(1.0, y1 - y0)
    if select == "centre_most":
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        dx, dy = cx - frame_w * 0.5, cy - frame_h * 0.5
        return -float(dx * dx + dy * dy)
    return h


def track_and_crop(
    images: torch.Tensor,
    pack: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], str]:
    """Return (crops [K,ch,cw,3], transform, report)."""
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("FaceRefine 需要 IMAGE 视频帧 [N,H,W,C]。")
    frames = images[..., :3].contiguous()
    n_frames, height, width, _ = frames.shape
    detector = load_detector(str(pack.get("detector") or "face_yolov8m.pt"))
    conf = float(pack.get("confidence") or 0.35)
    crop_factor = float(pack.get("crop_factor") or 2.5)
    select = str(pack.get("select") or "largest_face")
    canvas_w = int(pack.get("canvas_width") or 768)
    canvas_h = int(pack.get("canvas_height") or 768)
    canvas_mode = str(pack.get("canvas_mode") or "manual")

    cx = np.zeros(n_frames, dtype=np.float64)
    cy = np.zeros(n_frames, dtype=np.float64)
    sz = np.zeros(n_frames, dtype=np.float64)
    fw = np.zeros(n_frames, dtype=np.float64)
    valid = np.zeros(n_frames, dtype=bool)

    lock = None  # (cx, cy, h)
    found = 0
    for i in range(n_frames):
        try:
            res = detector.predict(_to_bgr_u8(frames[i]), conf=conf, verbose=False)[0]
            boxes = res.boxes.xyxy.tolist() if len(res.boxes) else []
        except Exception as exc:
            log.debug("Face detect frame %d failed: %s", i, exc)
            boxes = []
        if not boxes:
            continue
        if lock is None:
            pick = max(boxes, key=lambda b: _rank_box(b, width, height, select))
        else:
            lx, ly, lh = lock
            reach = 0.75 * max(lh, 8.0)

            def _dist(b):
                pcx, pcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
                return (pcx - lx) ** 2 + (pcy - ly) ** 2

            pick = min(boxes, key=_dist)
            pcx, pcy = (pick[0] + pick[2]) * 0.5, (pick[1] + pick[3]) * 0.5
            if _dist(pick) ** 0.5 > reach:
                continue
        x0, y0, x1, y1 = pick
        pcx, pcy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        ph, pw = max(1.0, y1 - y0), max(1.0, x1 - x0)
        cx[i], cy[i], sz[i], fw[i] = pcx, pcy, ph, pw
        valid[i] = True
        lock = (pcx, pcy, ph)
        found += 1

    if found == 0:
        raise ValueError(
            "FaceRefine 未检测到人脸。请换检测器、降低 confidence，或确认成片里有可见的脸。"
        )

    raw_cx = _interp_gaps(cx, valid)
    raw_cy = _interp_gaps(cy, valid)
    raw_sz = _interp_gaps(sz, valid)
    raw_fw = _interp_gaps(fw, valid)
    sm_cx = _smooth(raw_cx, 21)
    sm_cy = _smooth(raw_cy, 21)
    sm_sz = _smooth(raw_sz, 51)
    sm_fw = _smooth(raw_fw, 51)

    if canvas_mode != "manual":
        need = float(min(sm_sz.max() * crop_factor, height))
        from ...lib.image_prep import ensure_minimax_canvas

        snapped = int(np.ceil(need / 32.0) * 32)
        snapped = max(512, min(snapped, 768 if canvas_mode == "auto_capped_768" else 1344))
        canvas_w, canvas_h = ensure_minimax_canvas(snapped, snapped)

    aspect = canvas_w / float(canvas_h)
    boxes: list[tuple[float, float, float, float]] = []
    crops = torch.zeros((n_frames, canvas_h, canvas_w, 3), dtype=frames.dtype)
    for i in range(n_frames):
        bh = sm_sz[i] * crop_factor
        bw = bh * aspect
        if bw > width:
            bw, bh = float(width), float(width) / aspect
        if bh > height:
            bh, bw = float(height), float(height) * aspect
        x = min(max(sm_cx[i] - bw / 2.0, 0.0), max(0.0, width - bw))
        y = min(max(sm_cy[i] - bh / 2.0, 0.0), max(0.0, height - bh))
        box = (float(x), float(y), float(bw), float(bh))
        boxes.append(box)
        crops[i : i + 1] = affine_crop(frames[i : i + 1], box, canvas_w, canvas_h).to(crops.dtype)

    weights = np.clip(_smooth(valid.astype(np.float64), 11), 0.0, 1.0)
    face_rect = []
    for i, box in enumerate(boxes):
        bw, bh = max(box[2], 1e-6), max(box[3], 1e-6)
        face_rect.append(
            (
                float(canvas_w) * 0.5 - 0.5 * float(sm_fw[i]) / bw * canvas_w,
                float(canvas_h) * 0.5 - 0.5 * float(sm_sz[i]) / bh * canvas_h,
                float(sm_fw[i]) / bw * canvas_w,
                float(sm_sz[i]) / bh * canvas_h,
            )
        )
    transform = {
        "boxes": boxes,
        "canvas": (int(canvas_w), int(canvas_h)),
        "src_size": (int(width), int(height)),
        "frames": int(n_frames),
        "source": list(range(n_frames)),
        "source_frames": int(n_frames),
        "weights": [float(w) for w in weights],
        "detected": [bool(v) for v in valid],
        "face_rect": face_rect,
        "crop_factor": float(crop_factor),
    }
    report = (
        f"FaceRefine track: {found}/{n_frames} faces, select={select}, "
        f"canvas={canvas_w}x{canvas_h} ({canvas_mode}), crop×{crop_factor:g}"
    )
    return crops, transform, report
