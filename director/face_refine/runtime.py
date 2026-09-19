"""Run one face-refine pass on decoded Director frames. Imported only when wired."""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..frame_align import minimax_align_frame_count
from .inject import inject_video_latent
from .stitch import stitch_faces
from .track import track_and_crop

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.face_refine")


def _pad_frames(frames: torch.Tensor, length: int) -> torch.Tensor:
    if frames.shape[0] >= length:
        return frames[:length]
    last = frames[-1:].expand(length - frames.shape[0], -1, -1, -1)
    return torch.cat([frames, last], dim=0)


def _decode_video(samples, vae) -> torch.Tensor:
    from nodes import VAEDecode

    images, = VAEDecode().decode(vae, samples)
    return images


def apply_segment_face_refine(
    *,
    frames: torch.Tensor,
    plan,
    seg,
    pack: dict[str, Any],
    model,
    vae,
    audio_vae,
    clip,
    seed: int,
    cfg: float,
    shift_video: float,
    shift_audio: float,
    shift_cache=None,
) -> tuple[torch.Tensor, str]:
    """Return (stitched frames [N,...], report). Caller keeps a copy of `frames` for compare."""
    from ...nodes.conditioning import run_minimax_conditioning
    from ..core_sampling import sample_single_stage
    from ..plan import (
        official_ref_image_size,
        ref_audios_to_dict,
        ref_image_long_preset_px,
        refs_to_kwargs_for_context,
        resolve_ref_image_size,
    )
    from ..refine_pack import parse_refine_sigmas
    from ...lib.image_prep import assert_minimax_canvas, limit_ref_image_dict

    if frames is None or not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("FaceRefine 需要解码后的视频帧。")
    base = frames[..., :3].contiguous().float().cpu()
    n_src = int(base.shape[0])
    crops, transform, track_note = track_and_crop(base, pack)
    if crops is None or transform is None:
        return base[:n_src].contiguous(), track_note
    canvas_w, canvas_h = transform["canvas"]
    assert_minimax_canvas(int(canvas_w), int(canvas_h))
    gen_len = minimax_align_frame_count(int(crops.shape[0]))
    crop_in = _pad_frames(crops, gen_len)

    ref_kwargs = refs_to_kwargs_for_context(seg.task_key, seg.refs)
    ref_images = {}
    for key, tensor in ref_kwargs.items():
        if tensor is None:
            continue
        idx = key.removeprefix("reference_image_")
        ref_images[f"ref_image_{idx}"] = tensor[:1] if tensor.ndim == 4 else tensor
    if not ref_images:
        ref_images = None
    long_px = ref_image_long_preset_px(resolve_ref_image_size(seg, plan))
    if ref_images and long_px:
        ref_images, _ = limit_ref_image_dict(ref_images, long_px, edge="long")
    # Face crop is the video stream (via inject). Skip segment ref_videos:
    # a 1-frame IMAGE wired as video trips official "need at least 5 frames",
    # and a full-scene clip fights the close-up canvas.
    ref_audios = ref_audios_to_dict(
        getattr(seg, "ref_audios", None) or [],
        cache=getattr(plan, "audio_decode_cache", None),
    )
    size_arg = official_ref_image_size(resolve_ref_image_size(seg, plan))
    if audio_vae is None:
        raise ValueError("FaceRefine 需要 audio_vae（与导演台 r2v 相同）。")

    positive, _negative, latent, cond_hint = run_minimax_conditioning(
        clip=clip,
        vae=vae,
        audio_vae=audio_vae,
        prompt=seg.prompt or plan.global_prompt or "a person, face close-up",
        width=int(canvas_w),
        height=int(canvas_h),
        length=int(gen_len),
        task_key="r2v",
        ref_images=ref_images,
        ref_videos=None,
        ref_audios=ref_audios,
        ref_image_size=size_arg,
    )
    latent = inject_video_latent(latent, crop_in, vae)
    sigma_t = None
    if pack.get("has_sigmas_tensor") and pack.get("sigmas_tensor") is not None:
        sigma_t = pack.get("sigmas_tensor")
    elif pack.get("sigmas_parsed"):
        sigma_t = parse_refine_sigmas(pack.get("sigmas_parsed"), fallback=False)
    fr_seed = int(seed)
    if str(pack.get("seed_mode") or "inherit") == "offset":
        fr_seed = int(seed) + 1 + int(getattr(seg, "index", 0) or 0)
    sampled = sample_single_stage(
        model=model,
        positive=positive,
        negative=[],
        latent=latent,
        seed=fr_seed,
        cfg=float(cfg),
        steps=int(pack.get("steps") or 8),
        sampler_name=str(pack.get("sampler") or "euler"),
        scheduler=str(pack.get("scheduler") or "simple"),
        shift_video=float(shift_video),
        shift_audio=float(shift_audio),
        denoise=float(pack.get("denoise") or 0.40),
        sigmas=sigma_t,
        phase_name="sample",
        shift_cache=shift_cache,
    )
    refined = _decode_video(sampled, vae)
    if refined.ndim != 4:
        raise RuntimeError(f"FaceRefine decode 返回了非视频张量: {tuple(getattr(refined, 'shape', ()))}")
    refined = refined[: int(crops.shape[0]), ..., :3].float().cpu()
    if refined.shape[0] < crops.shape[0]:
        refined = _pad_frames(refined, crops.shape[0])
    stitched = stitch_faces(base, refined, transform, pack)
    stitched = stitched[:n_src].contiguous().cpu().float()
    note = (
        f"{track_note}; sample {pack.get('sampler')} steps={pack.get('steps')} "
        f"denoise={float(pack.get('denoise') or 0):.2f} seed={fr_seed} | {cond_hint}"
    )
    return stitched, note
