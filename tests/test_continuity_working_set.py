from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
CUSTOM_NODES_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

from ComfyUI_MiniMaxH3_Director.director.executor_core import (
    _prune_continuity_working_set,
)


class ContinuityWorkingSetTests(unittest.TestCase):
    def test_keeps_only_direct_predecessor_and_future_entries(self):
        av_latents = {0: {"samples": "a"}, 1: {"samples": "b"}, 2: {"samples": "c"}}
        refine_passes = {0: [("p1", "a")], 1: [("p1", "b")], 2: [("p1", "c")]}

        _prune_continuity_working_set(2, av_latents, refine_passes)

        self.assertEqual(set(av_latents), {1, 2})
        self.assertEqual(set(refine_passes), {1, 2})

    def test_missing_predecessor_does_not_keep_older_segment(self):
        av_latents = {0: {"samples": "a"}, 1: {"samples": "b"}}
        refine_passes = {0: [("p1", "a")], 1: [("p1", "b")]}

        _prune_continuity_working_set(3, av_latents, refine_passes)

        self.assertEqual(av_latents, {})
        self.assertEqual(refine_passes, {})

    def test_does_not_receive_or_modify_final_output_collections(self):
        av_latents = {0: {"samples": "a"}, 1: {"samples": "b"}}
        refine_passes = {0: [("p1", "a")], 1: [("p1", "b")]}
        completed_outputs = {0: "final-a", 1: "final-b"}
        completed_pre_refine = {0: "pre-a", 1: "pre-b"}
        completed_audios = {0: "audio-a", 1: "audio-b"}

        _prune_continuity_working_set(2, av_latents, refine_passes)

        self.assertEqual(completed_outputs, {0: "final-a", 1: "final-b"})
        self.assertEqual(completed_pre_refine, {0: "pre-a", 1: "pre-b"})
        self.assertEqual(completed_audios, {0: "audio-a", 1: "audio-b"})


if __name__ == "__main__":
    unittest.main()
