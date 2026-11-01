"""Run one face-refine pass on decoded Director frames. Imported only when wired."""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..frame_align import minimax_align_frame_count
from ..h3_x2_decode import decode_video_latent
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
    return decode_video_latent(vae, samples, context="director.face_refine")


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
    from ..plan import (
        official_ref_image_size,
        ref_audios_to_dict,
        ref_image_long_preset_px,
        refs_to_kwargs_for_context,
        resolve_ref_image_size,
    )
    from ...lib.image_prep import limit_ref_image_dict

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
    def prepare_latent(latent, start, end):
        from copy import copy
        from ..audio_freeze import apply_first_pass_audio_freeze, freeze_requested

        if not freeze_requested(plan)[0]:
            return

        shot = copy(seg)
        shot.start_frame = int(seg.start_frame) + start
        shot.end_frame = int(seg.start_frame) + end
        shot_audios = {}
        fps = float(plan.frame_rate)
        for key, audio in (ref_audios or {}).items():
            audio = dict(audio)
            sr = int(audio["sample_rate"])
            audio["waveform"] = audio["waveform"][..., round(start * sr / fps):round(end * sr / fps)]
            shot_audios[key] = audio
        apply_first_pass_audio_freeze(
            plan, shot, latent=latent, audio_vae=audio_vae,
            ref_audios=shot_audios, ref_video_audios=None, trim_frames=0,
        )

    return refine_face_video(
        frames=frames, pack=pack, model=model, vae=vae, audio_vae=audio_vae, clip=clip,
        seed=seed, cfg=cfg, shift_video=shift_video, shift_audio=shift_audio,
        shift_cache=shift_cache, ref_images=ref_images, ref_audios=ref_audios,
        prompt=seg.prompt or plan.global_prompt or "a person, face close-up",
        ref_image_size=size_arg, segment_index=int(getattr(seg, "index", 0)),
        prepare_latent=prepare_latent,
    )


def refine_face_video(
    *, frames, pack, model, vae, audio_vae, clip, seed=42, cfg=1.0,
    shift_video=12.0, shift_audio=3.0, shift_cache=None,
    ref_images=None, ref_audios=None, prompt="a person, face close-up",
    ref_image_size="match", segment_index=0, prepare_latent=None,
):
    """Shared postprocess for Director and already-rendered videos; never changes audio."""
    from ...nodes.conditioning import run_minimax_conditioning
    from ..core_sampling import sample_single_stage
    from ..refine_pack import parse_refine_sigmas
    from ...lib.image_prep import assert_minimax_canvas
    from .strength import apply_face_strength

    if frames is None or not isinstance(frames, torch.Tensor) or frames.ndim != 4 or len(frames) == 0:
        raise ValueError("FaceRefine 需要非空的视频帧 IMAGE。")
    if audio_vae is None:
        raise ValueError("FaceRefine 需要 H3 audio_vae。")
    base = frames[..., :3].detach().float().cpu().contiguous()
    crops, transform, track_note = track_and_crop(base, pack)
    if crops is None:
        return base, track_note
    canvas_w, canvas_h = transform["canvas"]
    assert_minimax_canvas(int(canvas_w), int(canvas_h))
    if pack.get("refine_model") is not None:
        model = pack["refine_model"]
    if pack.get("reference_image") is not None:
        ref_images = {"ref_image_0": pack["reference_image"][:1]}
    prompt = pack.get("prompt") or prompt
    sigma_t = None
    if pack.get("has_sigmas_tensor") and pack.get("sigmas_tensor") is not None:
        sigma_t = pack.get("sigmas_tensor")
    elif pack.get("sigmas_parsed"):
        sigma_t = parse_refine_sigmas(pack.get("sigmas_parsed"), fallback=False)
    fr_seed = int(seed)
    if str(pack.get("seed_mode") or "inherit") == "offset":
        fr_seed = int(seed) + 1 + int(segment_index)
    refined = crops.clone()
    notes = []
    for a, b in transform.get("shot_ranges") or [(0, len(crops))]:
        if not any(transform["detected"][a:b]):
            notes.append(f"shot {a}:{b} skipped: no face")
            continue
        gen_len = minimax_align_frame_count(b - a)
        crop_in = _pad_frames(crops[a:b], gen_len)
        if a:
            from ..vram_cleanup import cleanup_segment_vram
            if shift_cache is not None:
                shift_cache.clear()
            cleanup_segment_vram(enabled=True, unload_models=True)
        positive, _, latent, _ = run_minimax_conditioning(
            clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt,
            width=int(canvas_w), height=int(canvas_h), length=gen_len,
            task_key="r2v", ref_images=ref_images, ref_videos=None,
            ref_audios=ref_audios, ref_image_size=ref_image_size,
        )
        latent = inject_video_latent(latent, crop_in, vae)
        if prepare_latent is not None:
            prepare_latent(latent, a, b)
        sample_model = model
        strength_note = "constant strength"
        if pack.get("adaptive_strength"):
            shot_transform = {
                "face_sizes": transform["face_sizes"][a:b] + [transform["face_sizes"][b-1]] * (gen_len - (b-a)),
                "detected": transform["detected"][a:b] + [transform["detected"][b-1]] * (gen_len - (b-a)),
                "shot_ranges": [(0, b - a)],
            }
            sample_model, latent, strength_note = apply_face_strength(
                model, latent, shot_transform, pack,
            )
        sampled = sample_single_stage(
            model=sample_model,
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
        decoded = _decode_video(sampled, vae)
        if decoded.ndim != 4 or decoded.shape[0] < b - a:
            raise RuntimeError("FaceRefine 解码帧数不足或形状无效。")
        refined[a:b] = decoded[:b-a, ..., :3].float().cpu()
        notes.append(f"shot {a}:{b}: {strength_note}")
        del sampled, decoded, latent, positive
    stitched = stitch_faces(base, refined, transform, pack)
    stitched = stitched[:len(base)].contiguous().cpu().float()
    note = (
        f"{track_note}; sample {pack.get('sampler')} steps={pack.get('steps')} "
        f"denoise={float(pack.get('denoise') or 0):.2f} seed={fr_seed}; " + "; ".join(notes)
    )
    return stitched, note
