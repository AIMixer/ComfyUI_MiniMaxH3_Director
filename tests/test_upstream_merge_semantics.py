"""Behavioral coverage for upstream audio slots, continuity export, and WebP."""

import base64
import importlib
import io
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def modules():
    # Load production modules without importing the ComfyUI node entry point.
    package = types.ModuleType("upstream_semantics_pkg")
    package.__path__ = [str(ROOT)]
    paths = types.ModuleType("folder_paths")
    paths.get_input_directory = lambda: str(ROOT)
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    utils = types.ModuleType("comfy.utils")
    utils.common_upscale = lambda tensor, *_args, **_kwargs: tensor
    with mock.patch.dict(sys.modules, {
        "upstream_semantics_pkg": package,
        "folder_paths": paths,
        "comfy": comfy,
        "comfy.utils": utils,
    }):
        yield tuple(importlib.import_module(f"upstream_semantics_pkg.director.{name}")
                    for name in ("plan", "h3_motion_context", "tae_preview"))


@pytest.mark.parametrize("task", ["r2v", "rv2v"])
def test_failed_reference_audio_preserves_surviving_slot_and_prompt(modules, task):
    plan, _, _ = modules
    pcm = {"waveform": torch.ones((1, 1, 24)), "sample_rate": 24000}
    slots = [
        plan.SegmentRefAudio(index=0, audio=None, audio_path="broken.wav"),
        plan.SegmentRefAudio(index=1, audio=None, audio_path="valid.wav"),
    ]
    cache = {}
    with mock.patch.object(plan, "load_reference_audio", side_effect=lambda path, **_: pcm if path == "valid.wav" else None) as decode:
        refs = plan.segment_ref_audios_for_context(task, slots)
        usable = plan.usable_ref_audio_indices(refs, cache=cache)
        prompt = plan.drop_unusable_audio_prompt_tags("Voice <Audio 1> then <Audio 2>.", usable)
        conditioning = plan.ref_audios_to_dict(refs, cache=cache)
    assert usable == [1]
    assert "<Audio 1>" not in prompt
    assert "<Audio 2>" in prompt
    assert set(conditioning) == {"ref_audio_1"}
    assert conditioning["ref_audio_1"] is pcm
    assert sum(call.args[0] == "valid.wav" for call in decode.call_args_list) == 1
    assert all(call.kwargs["cache"] is cache for call in decode.call_args_list)
    assert plan.segment_ref_audios_for_context("addguide", slots) == []


@pytest.mark.parametrize("keep_tail,expected", [(False, 90), (True, 102)])
def test_continuity_keep_tail_controls_export_and_next_handoff(modules, keep_tail, expected):
    _, motion, _ = modules
    exported = motion.continuity_export_len(
        trim_frames=22, sample_len=124, visible_frames=90,
        target_len=90, keep_tail=keep_tail,
    )
    assert exported == expected
    assert motion.handoff_end_frame(trim_frames=22, export_frames=exported) == 22 + expected


@pytest.mark.parametrize("keep_tail", [False, True])
def test_clean_segment_export_does_not_change_with_keep_tail(modules, keep_tail):
    _, motion, _ = modules
    assert motion.continuity_export_len(
        trim_frames=0, sample_len=124, visible_frames=90,
        target_len=107, keep_tail=keep_tail,
    ) == 107


def test_multiframe_preview_encodes_actual_looping_webp(modules):
    _, _, preview = modules
    frames = [Image.new("RGB", (16, 12), color) for color in ("red", "blue", "green")]
    encoded, mime, width, height = preview.encode_preview_payload(frames, fps=12)
    assert (mime, width, height) == ("image/webp", 16, 12)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as animation:
        assert animation.format == "WEBP"
        assert animation.is_animated
        assert animation.n_frames == 3
        assert animation.info["loop"] == 0
        animation.seek(1)
        animation.load()
        red, green, blue = animation.convert("RGB").getpixel((8, 6))
        assert blue > red + 100 and blue > green + 100
