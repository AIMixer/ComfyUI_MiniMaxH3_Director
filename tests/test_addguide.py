"""CPU-only tests for AddGuide timeline semantics and conditioning order."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


timed = load_module("addguide_timed", "director/timed_guides.py")
frame_align = load_module("addguide_frame_align", "director/frame_align.py")


def guide(identifier: str, frame: int, *, batch: int = 1, identity: str = "img:a"):
    return timed.SegmentTimedGuide(
        id=identifier,
        frame_index=frame,
        tensor=torch.zeros((batch, 8, 8, 3)),
        image_file=f"{identifier}.png",
        image_identity=identity,
    )


class DefaultFrameTests(unittest.TestCase):
    def test_repeated_additions_fill_earliest_largest_gap(self):
        self.assertEqual(timed.choose_default_guide_frame(243, []), 121)
        self.assertEqual(timed.choose_default_guide_frame(243, [121]), 60)
        self.assertEqual(timed.choose_default_guide_frame(243, [60, 121]), 181)

    def test_endpoints_are_fallback_slots_only(self):
        self.assertEqual(timed.choose_default_guide_frame(2, []), 0)
        self.assertEqual(
            timed.choose_default_guide_frame(
                2, [], first_present=True, last_present=True
            ),
            None,
        )

    def test_bad_occupied_values_do_not_break_placement(self):
        self.assertEqual(timed.choose_default_guide_frame(5, ["bad", 2]), 1)


class ValidationTests(unittest.TestCase):
    def validate(self, guides, *, first=False, last=False, count=243):
        return timed.validate_timed_guides(
            guides,
            frame_count=count,
            first_present=first,
            last_present=last,
            segment_number=2,
        )

    def test_one_and_multiple_guides_are_legal_and_sorted(self):
        one = self.validate([guide("a", 48)])
        self.assertEqual([item.frame_index for item in one], [48])
        multiple = self.validate([guide("c", 181), guide("a", 48), guide("b", 120)])
        self.assertEqual([item.frame_index for item in multiple], [48, 120, 181])
        self.assertEqual([item.id for item in multiple], ["a", "b", "c"])

    def test_first_and_last_endpoint_rules(self):
        self.assertEqual(self.validate([guide("a", 0)])[0].frame_index, 0)
        self.assertEqual(self.validate([guide("a", 242)])[0].frame_index, 242)
        with self.assertRaisesRegex(ValueError, "conflicts with First"):
            self.validate([guide("a", 0)], first=True)
        with self.assertRaisesRegex(ValueError, "conflicts with Last"):
            self.validate([guide("a", 242)], last=True)

    def test_duplicate_frame_id_range_and_missing_image_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "more than one Guide"):
            self.validate([guide("a", 48), guide("b", 48)])
        with self.assertRaisesRegex(ValueError, "duplicate Guide id"):
            self.validate([guide("same", 48), guide("same", 120)])
        with self.assertRaisesRegex(ValueError, "outside F0..F242"):
            self.validate([guide("a", 243)])
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            self.validate([])
        empty = guide("a", 48)
        empty.tensor = None
        with self.assertRaisesRegex(ValueError, "has no image"):
            self.validate([empty])

    def test_image_batch_is_not_silently_used_as_a_clip(self):
        with self.assertRaisesRegex(ValueError, "P1 supports one image only"):
            self.validate([guide("a", 48, batch=2)])

    def test_duration_boundary_uses_aligned_frame_count(self):
        frame_count = frame_align.minimax_align_frame_count(240)
        self.assertEqual(frame_count, 243)
        self.assertEqual(
            self.validate([guide("last", frame_count - 1)], count=frame_count)[0].frame_index,
            242,
        )
        with self.assertRaisesRegex(ValueError, "outside F0..F123"):
            self.validate([guide("old", 220)], count=124)


class ParseAndFingerprintTests(unittest.TestCase):
    @staticmethod
    def parse(raw):
        return timed.parse_timed_guides(
            raw,
            load_image=lambda image: torch.zeros((1, 4, 4, 3))
            if image.get("imageFile")
            else None,
            image_identity=lambda image: (
                image.get("imageFile", ""),
                f"identity:{image.get('imageFile', 'missing')}",
            ),
        )

    def test_timeline_json_preserves_stable_id_frame_and_nested_image(self):
        parsed = self.parse(
            [
                {
                    "id": "guide-stable",
                    "frameIndex": 48,
                    "image": {"imageFile": "guides/a.png", "width": 64},
                }
            ]
        )
        self.assertEqual(parsed[0].id, "guide-stable")
        self.assertEqual(parsed[0].frame_index, 48)
        self.assertEqual(parsed[0].image_file, "guides/a.png")
        self.assertEqual(parsed[0].meta["width"], 64)

    def test_parser_rejects_fractional_frame_and_non_array(self):
        with self.assertRaisesRegex(ValueError, "must be an array"):
            self.parse({})
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.parse(
                [{"id": "a", "frameIndex": 1.5, "image": {"imageFile": "a.png"}}]
            )

    def test_cache_fingerprint_is_sorted_and_sensitive_to_all_guide_data(self):
        a = guide("a", 48, identity="sha:a")
        b = guide("b", 120, identity="sha:b")
        baseline = timed.timed_guides_fingerprint([b, a])
        self.assertEqual([item["id"] for item in baseline], ["a", "b"])
        self.assertEqual(baseline, timed.timed_guides_fingerprint([a, b]))
        self.assertNotEqual(baseline, timed.timed_guides_fingerprint([guide("a", 49, identity="sha:a"), b]))
        self.assertNotEqual(baseline, timed.timed_guides_fingerprint([guide("a", 48, identity="sha:new"), b]))
        self.assertNotEqual(baseline, timed.timed_guides_fingerprint([a]))


class PackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        addon = sys.modules.setdefault("addguide_pkg", types.ModuleType("addguide_pkg"))
        addon.__path__ = []
        lib = sys.modules.setdefault("addguide_pkg.lib", types.ModuleType("addguide_pkg.lib"))
        lib.__path__ = []
        director = sys.modules.setdefault(
            "addguide_pkg.director", types.ModuleType("addguide_pkg.director")
        )
        director.__path__ = []
        if "addguide_pkg.lib.task_prompts" not in sys.modules:
            load_module("addguide_pkg.lib.task_prompts", "lib/task_prompts.py")

        cls.folder_paths = types.ModuleType("folder_paths")
        cls.folder_paths.get_input_directory = lambda: cls.input_dir
        cls.folder_paths.get_output_directory = lambda: cls.output_dir
        cls.folder_paths.get_temp_directory = lambda: cls.temp_dir
        sys.modules["folder_paths"] = cls.folder_paths
        aiohttp = types.ModuleType("aiohttp")
        aiohttp.web = types.SimpleNamespace()
        sys.modules["aiohttp"] = aiohttp
        cls.pack = load_module("addguide_pkg.director.pack", "director/pack.py")

    def test_pack_rewrites_sorted_guide_media_and_import_prefixes_nested_path(self):
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            self.__class__.input_dir = str(root / "input")
            self.__class__.output_dir = str(root / "output")
            self.__class__.temp_dir = str(root / "temp")
            source_dir = Path(self.input_dir) / "guides"
            source_dir.mkdir(parents=True)
            (source_dir / "early.png").write_bytes(b"early")
            (source_dir / "later.jpg").write_bytes(b"later")

            card = {
                "timedGuides": [
                    {
                        "id": "stable-later",
                        "frameIndex": 120,
                        "image": {"imageFile": "guides/later.jpg"},
                    },
                    {
                        "id": "stable-early",
                        "frameIndex": 48,
                        "image": {"imageFile": "guides/early.png"},
                    },
                ]
            }
            staging = root / "staging"
            missing = []
            sizes = []
            self.pack._explode_card(
                card, "asset_groups/01", staging, missing, False, sizes
            )

            self.assertEqual(missing, [])
            self.assertEqual(
                [item["id"] for item in card["timedGuides"]],
                ["stable-early", "stable-later"],
            )
            self.assertEqual(
                card["timedGuides"][0]["image"]["imageFile"],
                "asset_groups/01/guide_001.png",
            )
            self.assertEqual(
                card["timedGuides"][1]["image"]["imageFile"],
                "asset_groups/01/guide_002.jpg",
            )
            self.assertTrue((staging / "asset_groups/01/guide_001.png").is_file())
            self.assertTrue((staging / "asset_groups/01/guide_002.jpg").is_file())

            self.pack._prefix_pack_paths(card, "minimax_director_packs/imported")
            self.assertEqual(
                card["timedGuides"][0]["image"]["imageFile"],
                "minimax_director_packs/imported/asset_groups/01/guide_001.png",
            )
            self.assertEqual(card["timedGuides"][0]["frameIndex"], 48)
            self.assertEqual(card["timedGuides"][0]["id"], "stable-early")


class ConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        addon = types.ModuleType("addguide_pkg")
        addon.__path__ = []
        nodes = types.ModuleType("addguide_pkg.nodes")
        nodes.__path__ = []
        lib = types.ModuleType("addguide_pkg.lib")
        lib.__path__ = []
        sys.modules.update(
            {
                "addguide_pkg": addon,
                "addguide_pkg.nodes": nodes,
                "addguide_pkg.lib": lib,
            }
        )
        load_module("addguide_pkg.lib.ref_images", "lib/ref_images.py")
        load_module("addguide_pkg.lib.task_modes", "lib/task_modes.py")
        cls.conditioning = load_module(
            "addguide_pkg.nodes.conditioning", "nodes/conditioning.py"
        )

    def test_official_addguide_calls_are_chained_in_frame_order(self):
        calls = []

        class Output:
            def __init__(self, positive):
                self.args = (positive,)

        class FakeAddGuide:
            @staticmethod
            def execute(positive, latent, frame_idx, **kwargs):
                calls.append((frame_idx, latent, kwargs["vae"], kwargs["image"]))
                return Output([*positive, frame_idx])

        original_loader = self.conditioning._load_minimax_addguide_node
        self.conditioning._load_minimax_addguide_node = lambda: FakeAddGuide
        try:
            latent = object()
            vae = object()
            guides = [guide("later", 120), guide("earlier", 48)]
            result = self.conditioning.apply_minimax_timed_guides(
                [], latent, vae=vae, timed_guides=guides
            )
        finally:
            self.conditioning._load_minimax_addguide_node = original_loader

        self.assertEqual(result, [48, 120])
        self.assertEqual([item[0] for item in calls], [48, 120])
        self.assertTrue(all(item[1] is latent and item[2] is vae for item in calls))

    def test_missing_official_class_fails_only_when_addguide_is_loaded(self):
        expected = (
            "MiniMax H3 AddGuide requires a ComfyUI version that provides "
            "MiniMaxH3AddGuide. Please update ComfyUI."
        )
        with mock.patch.dict(sys.modules, {"comfy_extras": None}):
            with self.assertRaisesRegex(RuntimeError, "MiniMaxH3AddGuide") as caught:
                self.conditioning._load_minimax_addguide_node()
        self.assertEqual(str(caught.exception), expected)


if __name__ == "__main__":
    unittest.main()
