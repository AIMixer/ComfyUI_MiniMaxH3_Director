"""Regression coverage for the AddGuide/upstream continuity merge."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_director_modules():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(ROOT)
    folder_paths.get_output_directory = lambda: str(ROOT)
    folder_paths.get_temp_directory = lambda: str(ROOT)
    folder_paths.get_annotated_filepath = lambda path: str(ROOT / path)

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.common_upscale = lambda tensor, *_args, **_kwargs: tensor

    package = types.ModuleType("addguide_merge_pkg")
    package.__path__ = [str(ROOT)]
    sys.modules.update(
        {
            "folder_paths": folder_paths,
            "comfy": comfy,
            "comfy.utils": comfy_utils,
            "addguide_merge_pkg": package,
        }
    )
    plan = importlib.import_module("addguide_merge_pkg.director.plan")
    gen = importlib.import_module("addguide_merge_pkg.director.gen_timeline")
    cache = importlib.import_module("addguide_merge_pkg.director.segment_cache")
    timed = importlib.import_module("addguide_merge_pkg.director.timed_guides")
    return plan, gen, cache, timed


plan_module, gen_module, cache_module, timed_module = load_director_modules()


def picture_guide(frame_index: int, identity: str = "image:a"):
    return timed_module.SegmentTimedGuide(
        id="guide",
        frame_index=frame_index,
        tensor=torch.zeros((1, 8, 8, 3)),
        image_file="guide.png",
        image_identity=identity,
    )


def build_plan(task_key: str, segment_keys: list[str], *, mode: str = "continue"):
    segments = [
        {
            "id": f"s{index}",
            "frameCount": 22,
            "prompt": f"segment {index}",
            "taskType": segment_key if task_key == "mixed" else "",
            "timedGuides": ([{"id": "guide", "frameIndex": 10}] if segment_key == "addguide" else []),
            "continuityFromPrev": True,
        }
        for index, segment_key in enumerate(segment_keys)
    ]
    timeline = {
        "timelineMode": "prompt_batch",
        "editMode": "segment",
        "global": {"taskType": task_key, "prompt": "global"},
        "output": {
            "mode": "fixed",
            "width": 864,
            "height": 480,
            "continuityEnabled": True,
            "continuityOverlapFrames": 22,
            "continuityMode": mode,
            "continuityRedraw": 0.72,
        },
        "segments": segments,
    }
    guide = picture_guide(10)
    with (
        mock.patch.object(gen_module, "_load_timed_guides", return_value=[guide]),
        mock.patch.object(gen_module, "validate_timed_guides", side_effect=lambda guides, **_kwargs: guides),
    ):
        return gen_module.build_gen_director_plan(
            timeline,
            global_task_type=task_key,
            global_prompt="global",
            total_frames=22,
            frame_rate=24,
            width=864,
            height=480,
            ref_max_size=864,
        )


class AddGuideContinuityMergeTests(unittest.TestCase):
    def test_pure_addguide_forces_continuity_off_even_in_continue_mode(self):
        built = build_plan("addguide", ["addguide"], mode="continue")
        self.assertFalse(built.continuity_enabled)
        self.assertEqual(built.continuity_overlap_frames, 0)
        self.assertEqual(built.continuity_mode, "continue")

    def test_addguide_never_consumes_previous_segment(self):
        built = build_plan("mixed", ["t2v", "addguide"], mode="continue")
        self.assertFalse(built.segments[1].continuity_from_prev)

    def test_segment_after_addguide_can_use_continue_mode(self):
        built = build_plan("mixed", ["addguide", "t2v"], mode="continue")
        self.assertTrue(built.continuity_enabled)
        self.assertEqual(built.continuity_mode, "continue")
        self.assertTrue(built.segments[1].continuity_from_prev)

    def test_cache_tracks_continue_settings_and_addguide_timed_guides(self):
        built = build_plan("mixed", ["addguide", "t2v"], mode="guide")
        normal = built.segments[1]
        guide_fp = cache_module.first_pass_cache_fingerprint(normal, built)
        built.continuity_mode = "continue"
        built.continuity_redraw = 0.61
        continue_fp = cache_module.first_pass_cache_fingerprint(normal, built)
        built.continuity_redraw = 0.83
        redraw_fp = cache_module.first_pass_cache_fingerprint(normal, built)
        self.assertNotEqual(guide_fp, continue_fp)
        self.assertNotEqual(continue_fp, redraw_fp)

        addguide = built.segments[0]
        first = cache_module.first_pass_cache_fingerprint(addguide, built)
        addguide.timed_guides = [picture_guide(11)]
        moved = cache_module.first_pass_cache_fingerprint(addguide, built)
        addguide.timed_guides = [picture_guide(11, identity="image:b")]
        replaced = cache_module.first_pass_cache_fingerprint(addguide, built)
        self.assertNotEqual(first, moved)
        self.assertNotEqual(moved, replaced)


if __name__ == "__main__":
    unittest.main()
