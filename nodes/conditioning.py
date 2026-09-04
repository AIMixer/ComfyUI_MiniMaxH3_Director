"""MiniMax H3 conditioning — delegates to ComfyUI official MiniMaxH3 nodes."""

from __future__ import annotations

from ..lib.ref_images import MAX_REFERENCE_IMAGES, REF_IMAGE_KEY_PREFIX, flatten_reference_kwargs
from ..lib.task_modes import TASK_DESCRIPTIONS, infer_task


def _shared_optional_inputs() -> dict:
    return {
        "first_frame": (
            "IMAGE",
            {"tooltip": "Optional first keyframe (i2v / fl2v)."},
        ),
        "last_frame": (
            "IMAGE",
            {"tooltip": "Optional last keyframe (fl2v)."},
        ),
        **{
            f"{REF_IMAGE_KEY_PREFIX}{index}": (
                "IMAGE",
                {
                    "tooltip": (
                        f"Reference image for <Picture {index + 1}> in prompt (r2v). "
                        "Native aspect; H3 ref_image_size applies at encode time."
                    ),
                },
            )
            for index in range(MAX_REFERENCE_IMAGES)
        },
        "ref_image_size": (
            ["match", "max"],
            {
                "default": "match",
                "tooltip": "Reference image sizing for MiniMaxH3ReferenceToVideo.",
            },
        ),
    }


def _load_minimax_nodes():
    try:
        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
        )
    except ImportError as exc:
        raise RuntimeError(
            "MiniMaxH3Director requires ComfyUI official MiniMax H3 nodes "
            "(comfy_extras.nodes_minimax_h3). Upgrade to ComfyUI with PR #15224 merged."
        ) from exc
    return MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo


def _load_minimax_addguide_node():
    """Load AddGuide only when that task is executed (older ComfyUI stays usable)."""
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "MiniMax H3 AddGuide requires a ComfyUI version that provides "
            "MiniMaxH3AddGuide. Please update ComfyUI."
        ) from exc
    return MiniMaxH3AddGuide


def _unpack_positive_latent(out):
    args = None
    if hasattr(out, "args"):
        args = out.args
    elif isinstance(out, (tuple, list)):
        args = out
    if args and len(args) >= 2:
        return args[0], args[1]
    raise RuntimeError(f"MiniMax H3 conditioning returned unexpected output: {type(out)!r}")


def _unpack_positive(out):
    args = out.args if hasattr(out, "args") else out
    if isinstance(args, (tuple, list)) and args:
        return args[0]
    raise RuntimeError(f"MiniMax H3 AddGuide returned unexpected output: {type(out)!r}")


def apply_minimax_timed_guides(positive, latent, *, vae, timed_guides):
    """Chain official MiniMaxH3AddGuide calls into one segment conditioning."""
    guides = sorted(
        list(timed_guides or []),
        key=lambda guide: (int(getattr(guide, "frame_index", 0)), str(getattr(guide, "id", ""))),
    )
    if not guides:
        return positive
    MiniMaxH3AddGuide = _load_minimax_addguide_node()
    guided = positive
    for guide in guides:
        guided = _unpack_positive(
            MiniMaxH3AddGuide.execute(
                guided,
                latent,
                int(guide.frame_index),
                vae=vae,
                image=guide.tensor,
            )
        )
    return guided


def _reference_images_dict_from_kwargs(kwargs: dict) -> dict | None:
    nested = kwargs.get("ref_images")
    if isinstance(nested, dict) and nested:
        out = {k: v for k, v in nested.items() if v is not None}
        return out or None

    refs = flatten_reference_kwargs(kwargs)
    out: dict[str, object] = {}
    for key, value in refs.items():
        if value is None:
            continue
        idx = key.removeprefix(REF_IMAGE_KEY_PREFIX)
        out[f"ref_image_{idx}"] = value
    return out or None


def _reference_videos_dict(ref_videos: dict | None) -> dict | None:
    if not ref_videos:
        return None
    out = {k: v for k, v in ref_videos.items() if v is not None}
    return out or None


def _task_hint(task_key: str, ref_images, ref_videos) -> str:
    ref_image_count = len(ref_images or {})
    ref_video_count = len(ref_videos or {})
    mode = infer_task(ref_image_count, ref_video_count)
    hint = f"{task_key or mode.value} — {TASK_DESCRIPTIONS[mode]} (MiniMax H3)"
    if ref_image_count or ref_video_count:
        hint += f" (~{ref_image_count} ref image(s), {ref_video_count} ref video(s))"
    return hint


def run_minimax_conditioning(
    *,
    clip,
    vae,
    audio_vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    first_frame=None,
    last_frame=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
    ref_image_size: str = "match",
    **kwargs,
):
    """Build positive conditioning + AV latent via official MiniMax H3 nodes."""
    MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo = _load_minimax_nodes()

    ref_images = ref_images or _reference_images_dict_from_kwargs(kwargs)
    ref_videos = _reference_videos_dict(ref_videos)

    use_reference = (
        task_key in {"r2v", "v2v", "rv2v"}
        or ref_images
        or ref_videos
        or ref_audios
        or ref_video_audios
    )

    if use_reference:
        if audio_vae is None:
            raise ValueError("MiniMax H3 r2v/v2v/rv2v / reference conditioning requires audio_vae.")
        out = MiniMaxH3ReferenceToVideo.execute(
            clip,
            vae,
            audio_vae,
            prompt,
            width,
            height,
            length,
            ref_image_size,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )
    else:
        out = MiniMaxH3ImageToVideo.execute(
            clip,
            vae,
            prompt,
            width,
            height,
            length,
            first_frame=first_frame,
            last_frame=last_frame,
        )

    positive, latent = _unpack_positive_latent(out)
    hint = _task_hint(task_key, ref_images, ref_videos)
    return positive, [], latent, hint


class MiniMaxH3DirectorConditioning:
    """Thin wrapper around official MiniMax H3 conditioning (positive + latent)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "Required for r2v / v2v / rv2v / reference video+audio."}),
                **_shared_optional_inputs(),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3"

    def apply(self, clip, vae, prompt, width, height, length, audio_vae=None, **kwargs):
        positive, _, latent, _ = run_minimax_conditioning(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            task_key="r2v" if kwargs.get("ref_images") or any(
                k.startswith(REF_IMAGE_KEY_PREFIX) for k in kwargs
            ) else "t2v",
            **kwargs,
        )
        return positive, latent


class MiniMaxH3DirectorPlannerConditioning:
    """Official MiniMax H3 conditioning plus task_mode string for planning UIs."""

    @classmethod
    def INPUT_TYPES(cls):
        base = MiniMaxH3DirectorConditioning.INPUT_TYPES()
        return base

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "task_mode")
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3"

    def apply(self, clip, vae, prompt, width, height, length, audio_vae=None, **kwargs):
        positive, _, latent, hint = run_minimax_conditioning(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            task_key="r2v" if kwargs.get("ref_images") or any(
                k.startswith(REF_IMAGE_KEY_PREFIX) for k in kwargs
            ) else "t2v",
            **kwargs,
        )
        return positive, latent, hint
