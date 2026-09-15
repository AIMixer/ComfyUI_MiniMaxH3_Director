"""Regression tests for generation-only segment source resolution."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_segment_runtime():
    package = types.ModuleType("segment_runtime_pkg")
    package.__path__ = []
    lib = types.ModuleType("segment_runtime_pkg.lib")
    lib.__path__ = []
    director = types.ModuleType("segment_runtime_pkg.director")
    director.__path__ = []

    image_prep = types.ModuleType("segment_runtime_pkg.lib.image_prep")
    image_prep.fit_canvas = lambda frames, width, height: frames
    image_prep.fit_video_long_edge = lambda frames, long_edge: frames

    video_io = types.ModuleType("segment_runtime_pkg.lib.video_io")

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("generation-only segment must not decode timeline video")

    video_io.load_timeline_segment = fail_decode

    frame_align = types.ModuleType("segment_runtime_pkg.director.frame_align")
    frame_align.pad_or_trim_frames = lambda frames, target_len: frames[:target_len]
    plan = types.ModuleType("segment_runtime_pkg.director.plan")
    plan.DirectorPlan = object

    sys.modules.update(
        {
            "segment_runtime_pkg": package,
            "segment_runtime_pkg.lib": lib,
            "segment_runtime_pkg.director": director,
            "segment_runtime_pkg.lib.image_prep": image_prep,
            "segment_runtime_pkg.lib.video_io": video_io,
            "segment_runtime_pkg.director.frame_align": frame_align,
            "segment_runtime_pkg.director.plan": plan,
        }
    )
    spec = importlib.util.spec_from_file_location(
        "segment_runtime_pkg.director.segment_runtime",
        ROOT / "director/segment_runtime.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runtime = load_segment_runtime()


class GenerationSegmentSourceTests(unittest.TestCase):
    def setUp(self):
        self.plan = SimpleNamespace(
            raw={"timelineMode": "prompt_batch"},
            source_video=torch.full((2, 16, 16, 3), 0.5),
        )

    def test_addguide_nonzero_global_offset_never_decodes_source_video(self):
        seg = SimpleNamespace(
            task_key="addguide",
            source_clip=None,
            start_frame=124,
            end_frame=248,
        )

        result = runtime.resolve_segment_raw_clip(self.plan, seg)

        self.assertEqual(tuple(result.shape), (0, 16, 16, 3))

    def test_addguide_lookahead_never_decodes_source_video(self):
        seg = SimpleNamespace(
            task_key="addguide",
            source_clip=None,
            start_frame=124,
            end_frame=248,
        )

        result = runtime.resolve_segment_raw_clip_with_lookahead(
            self.plan, seg, end_extra=22
        )

        self.assertEqual(tuple(result.shape), (0, 16, 16, 3))


if __name__ == "__main__":
    unittest.main()
