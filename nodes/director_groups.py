"""Graph packer nodes: Image to Video / Reference to Video groups for Director."""

from __future__ import annotations

from ..director.external_groups import (
    MMX_DIR_GROUP,
    pack_i2v_group,
    pack_r2v_group,
)
from ..director.fl2v_timeline import DEFAULT_FL2V_DURATION_SEC
from ..lib.ref_audios import MAX_REFERENCE_AUDIOS
from ..lib.ref_images import MAX_REFERENCE_IMAGES
from ..lib.ref_videos import MAX_REFERENCE_VIDEOS

_CATEGORY = "MiniMaxH3/Director Groups"


def _i2v_inputs() -> dict:
    return {
        "required": {
            "prompt": (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for this group (can connect an expand/rewrite node).",
                },
            ),
            "duration_sec": (
                "FLOAT",
                {
                    "default": DEFAULT_FL2V_DURATION_SEC,
                    "min": 0.2,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Clip duration in seconds (snapped to MiniMax 17k+5 frames @ fps).",
                },
            ),
        },
        "optional": {
            "first_frame": (
                "IMAGE",
                {"tooltip": "Optional first frame → i2v / fl2v. Official FL2VA also allows last-only."},
            ),
            "last_frame": (
                "IMAGE",
                {"tooltip": "Optional last frame → fl2v (alone or with first_frame)."},
            ),
            "model": (  # [per-group-lora]
                "MODEL",
                {"tooltip": "Optional per-group MODEL (own LoRA stack). "
                            "Unconnected -> the Director's own model input."},
            ),
        },
    }


def _autogrow_to_index_map(raw) -> dict[int, object]:
    """Convert Autogrow dict ({ref_image_0: …}) or int-key dict to {0: value, …}."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, object] = {}
    for key, val in raw.items():
        if val is None:
            continue
        if isinstance(key, int):
            idx = key
        else:
            try:
                idx = int(str(key).rsplit("_", 1)[-1])
            except ValueError:
                continue
        out[idx] = val
    return out


def _pack_r2v_kwargs(
    *,
    prompt="",
    duration_sec=DEFAULT_FL2V_DURATION_SEC,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
    model=None,  # [per-group-lora]
    **kwargs,
):
    """Pack R2V group from Autogrow dicts and/or legacy ref_*_N kwargs."""
    images = _autogrow_to_index_map(ref_images)
    videos = _autogrow_to_index_map(ref_videos)
    v_audios = _autogrow_to_index_map(ref_video_audios)
    audios = _autogrow_to_index_map(ref_audios)
    # Legacy V1 flat kwargs (ref_image_0, …) when Autogrow API is unavailable.
    for i in range(MAX_REFERENCE_IMAGES):
        val = kwargs.get(f"ref_image_{i}")
        if val is not None and i not in images:
            images[i] = val
    for i in range(MAX_REFERENCE_VIDEOS):
        val = kwargs.get(f"ref_video_{i}")
        if val is not None and i not in videos:
            videos[i] = val
        aud = kwargs.get(f"ref_video_audio_{i}")
        if aud is not None and i not in v_audios:
            v_audios[i] = aud
    for i in range(MAX_REFERENCE_AUDIOS):
        val = kwargs.get(f"ref_audio_{i}")
        if val is not None and i not in audios:
            audios[i] = val
    return pack_r2v_group(
        prompt=prompt,
        duration_sec=duration_sec,
        ref_images=images,
        ref_videos=videos,
        ref_video_audios=v_audios,
        ref_audios=audios,
        model=model,  # [per-group-lora]
    )


class MiniMaxH3DirectorGroupImageToVideo:
    """Pack one Image-to-Video group (t2v / i2v / fl2v) for the Director."""

    @classmethod
    def INPUT_TYPES(cls):
        return _i2v_inputs()

    RETURN_TYPES = (MMX_DIR_GROUP,)
    RETURN_NAMES = ("group",)
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Pack one MiniMax H3 Image to Video group for Director.i2v_groups. "
        "No frames = t2v; first only = i2v; last only / first+last = fl2v. "
        "Output size is controlled by the Director node (not this packer). "
        "Connect `group` to Director (single) or Groups Combine (batch)."
    )

    def pack(
        self,
        prompt="",
        duration_sec=DEFAULT_FL2V_DURATION_SEC,
        first_frame=None,
        last_frame=None,
        model=None,  # [per-group-lora]
        **_kwargs,
    ):
        group = pack_i2v_group(
            prompt=prompt,
            duration_sec=duration_sec,
            first_frame=first_frame,
            last_frame=last_frame,
            model=model,
        )
        return (group,)


try:
    from comfy_api.latest import io as comfy_io
except ImportError:  # pragma: no cover — only available inside ComfyUI
    comfy_io = None


def _combine_groups(groups_map) -> list:
    """Merge Autogrow / kwargs group slots into one ordered list."""
    from ..director.external_groups import normalize_groups_list

    if not groups_map:
        raise ValueError("Director Groups Combine: connect at least one group.")

    def _slot_index(name: str) -> int:
        try:
            return int(str(name).rsplit("_", 1)[-1])
        except ValueError:
            return 10**9

    out: list = []
    for name, g in sorted(groups_map.items(), key=lambda kv: _slot_index(kv[0])):
        if g is None:
            continue
        out.extend(normalize_groups_list(g))
    if not out:
        raise ValueError("Director Groups Combine: connect at least one group.")
    families = {
        g.get("family") or ("r2v" if g.get("kind") == "r2v" else "i2v") for g in out
    }
    if len(families) > 1:
        raise ValueError(
            "Director Groups Combine: do not mix Image to Video groups with "
            "Reference to Video groups in one list."
        )
    return out


if comfy_io is not None:
    _MMXDirGroup = comfy_io.Custom(MMX_DIR_GROUP)

    class MiniMaxH3DirectorGroupReferenceToVideo(comfy_io.ComfyNode):
        """Pack one R2V group — Autogrow refs (same UX as official MiniMaxH3ReferenceToVideo)."""

        @classmethod
        def define_schema(cls):
            return comfy_io.Schema(
                node_id="MiniMaxH3DirectorGroupReferenceToVideo",
                display_name="MiniMax H3 Director Group (Reference to Video)",
                category=_CATEGORY,
                description=(
                    "Pack one MiniMax H3 Reference to Video group for Director.r2v_groups. "
                    "Autogrow slots for images (≤9), videos (≤3), video audios (≤3), audios (≤3) "
                    "— same as official MiniMax H3 Reference to Video. "
                    "Output size is controlled by the Director node. "
                    "Connect `group` to Director (single) or Groups Combine (batch)."
                ),
                inputs=[
                    comfy_io.String.Input(
                        "prompt",
                        multiline=True,
                        default="",
                        tooltip="Prompt; use <Picture N> / <Video K> / <Audio J> tags.",
                    ),
                    comfy_io.Float.Input(
                        "duration_sec",
                        default=DEFAULT_FL2V_DURATION_SEC,
                        min=0.2,
                        max=120.0,
                        step=0.1,
                        tooltip="Clip duration in seconds (snapped to MiniMax 17k+5 frames @ fps).",
                    ),
                    comfy_io.Autogrow.Input(
                        "ref_images",
                        optional=True,
                        tooltip="Reference images → <Picture N>. Connect to grow another slot.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Image.Input(
                                "ref_image",
                                tooltip="Reference image (→ <Picture N>).",
                            ),
                            prefix="ref_image_",
                            min=0,
                            max=MAX_REFERENCE_IMAGES,
                        ),
                    ),
                    comfy_io.Autogrow.Input(
                        "ref_videos",
                        optional=True,
                        tooltip="Reference video frame batches → <Video K>.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Image.Input(
                                "ref_video",
                                tooltip="Reference video frames (IMAGE batch) → <Video K>.",
                            ),
                            prefix="ref_video_",
                            min=0,
                            max=MAX_REFERENCE_VIDEOS,
                        ),
                    ),
                    comfy_io.Autogrow.Input(
                        "ref_video_audios",
                        optional=True,
                        tooltip="Soundtrack paired with the same-numbered reference video.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Audio.Input(
                                "ref_video_audio",
                                tooltip="Audio paired with ref_video_N.",
                            ),
                            prefix="ref_video_audio_",
                            min=0,
                            max=MAX_REFERENCE_VIDEOS,
                        ),
                    ),
                    comfy_io.Autogrow.Input(
                        "ref_audios",
                        optional=True,
                        tooltip="Standalone reference audio → <Audio J>.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Audio.Input(
                                "ref_audio",
                                tooltip="Standalone reference audio → <Audio J>.",
                            ),
                            prefix="ref_audio_",
                            min=0,
                            max=MAX_REFERENCE_AUDIOS,
                        ),
                    ),
                    comfy_io.Model.Input(  # [per-group-lora]
                        "model",
                        optional=True,
                        tooltip="Optional per-group MODEL (own LoRA stack). "
                                "Unconnected -> the Director's own model input.",
                    ),
                ],
                outputs=[
                    _MMXDirGroup.Output("group"),
                ],
            )

        @classmethod
        def execute(
            cls,
            prompt="",
            duration_sec=DEFAULT_FL2V_DURATION_SEC,
            ref_images=None,
            ref_videos=None,
            ref_video_audios=None,
            ref_audios=None,
            model=None,  # [per-group-lora]
        ) -> comfy_io.NodeOutput:
            group = _pack_r2v_kwargs(
                prompt=prompt,
                duration_sec=duration_sec,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
                model=model,
            )
            return comfy_io.NodeOutput(group)

    class MiniMaxH3DirectorGroupsCombine(comfy_io.ComfyNode):
        """Fan-in Director groups via Autogrow slots (same UX as official R2V refs)."""

        @classmethod
        def define_schema(cls):
            return comfy_io.Schema(
                node_id="MiniMaxH3DirectorGroupsCombine",
                display_name="MiniMax H3 Director Groups Combine",
                category=_CATEGORY,
                description=(
                    "Combine multiple Director Group outputs into one list for the Director. "
                    "Connect a group to grow another empty slot (Autogrow, same as official "
                    "MiniMax H3 Reference to Video). Empty slots are skipped; order follows "
                    "connection index (group_0, group_1, …)."
                ),
                inputs=[
                    comfy_io.Autogrow.Input(
                        "groups",
                        optional=True,
                        tooltip="Director Group slots. Linking the last empty slot adds another.",
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=_MMXDirGroup.Input(
                                "group",
                                tooltip="One Director Group (Image to Video or Reference to Video).",
                            ),
                            prefix="group_",
                            # min=0 → one empty slot at start (same as official R2V refs).
                            # min=1 would pre-create two slots via frontend Autogrow init.
                            min=0,
                            max=100,
                        ),
                    ),
                ],
                outputs=[
                    _MMXDirGroup.Output("groups"),
                ],
            )

        @classmethod
        def execute(cls, groups=None) -> comfy_io.NodeOutput:
            return comfy_io.NodeOutput(_combine_groups(groups))

else:
    # Fallback for tooling / very old hosts without comfy_api Autogrow.
    class MiniMaxH3DirectorGroupReferenceToVideo:
        """Pack one R2V group (static slots when Autogrow API is unavailable)."""

        @classmethod
        def INPUT_TYPES(cls):
            opts: dict = {}
            for i in range(MAX_REFERENCE_IMAGES):
                opts[f"ref_image_{i}"] = (
                    "IMAGE",
                    {"tooltip": f"Reference image → <Picture {i + 1}>."},
                )
            for i in range(MAX_REFERENCE_VIDEOS):
                opts[f"ref_video_{i}"] = (
                    "IMAGE",
                    {
                        "tooltip": (
                            f"Reference video frames (IMAGE batch) → <Video {i + 1}>."
                        ),
                    },
                )
                opts[f"ref_video_audio_{i}"] = (
                    "AUDIO",
                    {"tooltip": f"Audio paired with ref_video_{i}."},
                )
            for i in range(MAX_REFERENCE_AUDIOS):
                opts[f"ref_audio_{i}"] = (
                    "AUDIO",
                    {"tooltip": f"Standalone reference audio → <Audio {i + 1}>."},
                )
            return {
                "required": {
                    "prompt": (
                        "STRING",
                        {"multiline": True, "default": "", "tooltip": "Prompt; use tags."},
                    ),
                    "duration_sec": (
                        "FLOAT",
                        {
                            "default": DEFAULT_FL2V_DURATION_SEC,
                            "min": 0.2,
                            "max": 120.0,
                            "step": 0.1,
                        },
                    ),
                },
                "optional": {
                    **opts,
                    "model": (  # [per-group-lora]
                        "MODEL",
                        {"tooltip": "Optional per-group MODEL (own LoRA stack)."},
                    ),
                },
            }

        RETURN_TYPES = (MMX_DIR_GROUP,)
        RETURN_NAMES = ("group",)
        FUNCTION = "pack"
        CATEGORY = _CATEGORY
        DESCRIPTION = (
            "Pack one MiniMax H3 Reference to Video group for Director.r2v_groups."
        )

        def pack(self, prompt="", duration_sec=DEFAULT_FL2V_DURATION_SEC, **kwargs):
            return (_pack_r2v_kwargs(prompt=prompt, duration_sec=duration_sec, **kwargs),)

    class MiniMaxH3DirectorGroupsCombine:
        """Fan-in groups (static slots when Autogrow API is unavailable)."""

        _MAX_SLOTS = 16

        @classmethod
        def INPUT_TYPES(cls):
            opts = {
                f"group_{i}": (MMX_DIR_GROUP, {"tooltip": f"Group slot {i}."})
                for i in range(cls._MAX_SLOTS)
            }
            return {"required": {}, "optional": opts}

        RETURN_TYPES = (MMX_DIR_GROUP,)
        RETURN_NAMES = ("groups",)
        FUNCTION = "combine"
        CATEGORY = _CATEGORY
        DESCRIPTION = (
            "Combine multiple Director Group outputs into one list for the Director. "
            "Empty slots are skipped; order follows group_0 → group_N."
        )

        def combine(self, **kwargs):
            slots = {
                k: v
                for k, v in kwargs.items()
                if k.startswith("group_") and v is not None
            }
            return (_combine_groups(slots),)
