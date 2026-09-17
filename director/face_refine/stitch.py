"""Warp refined crops back onto source frames.

Adapted from ComfyUI-H3-FaceRefine H3FaceStitch (MIT).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .track import face_region_mask, feather_edge_mask

# Opening/tail frames where FaceRefine paste is ramped off at a continuity cut.
# Independent per-segment face r2v restarts at the export seam and pops hair /
# identity; motion-context pixels already match. ~0.5s @ 24fps.
FACE_REFINE_SEAM_FADE_FRAMES = 12


def stitch_faces(
    base_images: torch.Tensor,
    refined_crops: torch.Tensor,
    transform: dict,
    pack: dict,
) -> torch.Tensor:
    boxes = transform["boxes"]
    weights = transform.get("weights")
    source = transform.get("source") or list(range(min(len(boxes), refined_crops.shape[0])))
    count = min(len(boxes), refined_crops.shape[0], len(source), base_images.shape[0])
    if count <= 0:
        return base_images[..., :3].clone()

    try:
        import comfy.model_management as mm

        dev = mm.get_torch_device()
    except Exception:
        dev = base_images.device

    paste = str(pack.get("paste_region") or "face_only")
    dilation = int(pack.get("mask_dilation") or 16)
    feather = int(pack.get("feather") or 24)
    colour_match = float(pack.get("colour_match") or 1.0)
    blend = float(pack.get("blend") or 1.0)
    cw, ch = transform["canvas"]
    width, height = transform["src_size"]
    face_rects = transform.get("face_rect")
    dt = base_images.dtype
    out = base_images[..., :3].clone()
    per_frame_mb = (height * width * 3 * 4) / 2 ** 20
    chunk = max(1, min(32, int(1024 / max(per_frame_mb, 1e-6))))

    for c0 in range(0, count, chunk):
        try:
            import comfy.model_management as mm

            mm.throw_exception_if_processing_interrupted()
        except Exception:
            pass
        c1 = min(c0 + chunk, count)
        n = c1 - c0
        bh_mid = float(boxes[(c0 + c1 - 1) // 2][3])
        f_can = int(round(feather * (ch / max(bh_mid, 1.0))))
        f_can = max(1, min(f_can, ch // 3))
        if paste == "full_crop":
            one = feather_edge_mask(ch, cw, f_can, dev, torch.float32)
            mask_can = one.view(1, 1, ch, cw).expand(n, 1, ch, cw)
        else:
            mask_can = torch.cat(
                [
                    face_region_mask(
                        ch,
                        cw,
                        face_rects[i] if face_rects and i < len(face_rects) else (
                            cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5
                        ),
                        dilation,
                        f_can,
                        "ellipse" if paste == "face_ellipse" else "rect",
                        dev,
                        torch.float32,
                    )
                    for i in range(c0, c1)
                ],
                dim=0,
            )
        th = torch.empty((n, 2, 3), dtype=torch.float32, device=dev)
        for j, i in enumerate(range(c0, c1)):
            x, y, bw, bh = (float(v) for v in boxes[i])
            th[j, 0, 0] = width / bw
            th[j, 0, 1] = 0.0
            th[j, 0, 2] = (width - 2.0 * x) / bw - 1.0
            th[j, 1, 0] = 0.0
            th[j, 1, 1] = height / bh
            th[j, 1, 2] = (height - 2.0 * y) / bh - 1.0
        grid = F.affine_grid(th, (n, 3, int(height), int(width)), align_corners=False)
        patch_can = refined_crops[c0:c1, ..., :3].to(dev).movedim(-1, 1).float()
        patch = F.grid_sample(patch_can, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        m = F.grid_sample(mask_can.to(dev), grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0, 1)
        patch = patch.movedim(1, -1)
        m = m.movedim(1, -1)
        dst = torch.as_tensor(source[c0:c1], dtype=torch.long, device=out.device)
        base = out[dst].to(dev).float()
        if colour_match > 0.0:
            wsum = m.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            bmu = (base * m).sum(dim=(1, 2), keepdim=True) / wsum
            pmu = (patch * m).sum(dim=(1, 2), keepdim=True) / wsum
            bsd = (((base - bmu) ** 2 * m).sum(dim=(1, 2), keepdim=True) / wsum).sqrt().clamp_min(1e-6)
            psd = (((patch - pmu) ** 2 * m).sum(dim=(1, 2), keepdim=True) / wsum).sqrt().clamp_min(1e-6)
            adj = (patch - pmu) * (bsd / psd) + bmu
            patch = (patch + (adj - patch) * colour_match).clamp(0, 1)
        wv = torch.full((n, 1, 1, 1), float(blend), device=dev, dtype=torch.float32)
        if weights is not None:
            for j, i in enumerate(range(c0, c1)):
                if i < len(weights):
                    wv[j] *= float(weights[i])
        mm_ = m * wv
        out[dst] = ((1.0 - mm_) * base + mm_ * patch).to(out.device, dt)
    return out


def fade_stitch_at_seams(
    stitched: torch.Tensor,
    base: torch.Tensor,
    *,
    head_frames: int = 0,
    tail_frames: int = 0,
) -> torch.Tensor:
    """Lerp FaceRefine paste back to the unstitched clip at continuity edges.

    ``head_frames``: first N export frames (incoming cut) → base.
    ``tail_frames``: last N export frames (outgoing cut) → base.
    """
    if (
        stitched is None
        or base is None
        or stitched is base
        or not isinstance(stitched, torch.Tensor)
        or not isinstance(base, torch.Tensor)
        or stitched.ndim != 4
        or base.ndim != 4
    ):
        return stitched
    n = min(int(stitched.shape[0]), int(base.shape[0]))
    if n < 2:
        return stitched
    head = max(0, min(int(head_frames or 0), n))
    tail = max(0, min(int(tail_frames or 0), n))
    if head + tail > n:
        tail = max(0, n - head)
    if head < 1 and tail < 1:
        return stitched
    out = stitched[:n, ..., :3].clone()
    src = base[:n, ..., :3].to(device=out.device, dtype=out.dtype)
    if head:
        t = torch.arange(head, device=out.device, dtype=out.dtype) / float(head)
        w = (0.5 * (1.0 + torch.cos(t * math.pi))).view(head, 1, 1, 1)
        out[:head] = src[:head] * w + out[:head] * (1.0 - w)
    if tail:
        t = torch.arange(tail, device=out.device, dtype=out.dtype) / float(max(tail, 1))
        w = (0.5 * (1.0 - torch.cos(t * math.pi))).view(tail, 1, 1, 1)
        out[-tail:] = src[-tail:] * w + out[-tail:] * (1.0 - w)
    if n < int(stitched.shape[0]):
        merged = stitched.clone()
        merged[:n] = out.to(device=stitched.device, dtype=stitched.dtype)
        return merged
    return out.to(device=stitched.device, dtype=stitched.dtype)
