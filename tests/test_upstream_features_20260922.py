"""CPU behavioral checks shared by the integration and pristine upstream trees.

Set MINIMAX_UPSTREAM_SOURCE_ROOT to an extracted upstream checkout to run the
same tests against that source. Only ComfyUI host interfaces are stubbed; the
pack, conditioning resize, and semantic rewrite implementations are real.
"""

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


ROOT = Path(os.environ.get("MINIMAX_UPSTREAM_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
PACKAGE = "upstream_features_20260922_pkg"


class HostPackedLayout:
    """Capture the constructor contract without loading ComfyUI's model."""

    def __init__(self, text_len, latent_t, height, width, audio_t,
                 keyframes=None, refs=None, frame_count=None):
        self.signature = (text_len, latent_t, height, width, audio_t)
        self.latent_t = latent_t
        self.audio_t = audio_t
        self.keyframes = keyframes
        self.refs = refs
        self.frame_count = frame_count


@pytest.fixture(scope="module")
def features():
    stubs = {}
    # Skip plugin registration and SelfLift's GPU sampling import at package init.
    for suffix in ("", ".lib", ".director", ".director.selflift", ".director.face_refine"):
        module = types.ModuleType(PACKAGE + suffix)
        module.__path__ = [str(ROOT.joinpath(*suffix.strip(".").split("."))) if suffix else str(ROOT)]
        stubs[module.__name__] = module
    for name in ("comfy", "comfy.utils", "comfy.ldm", "comfy.ldm.minimax", "comfy.ldm.minimax.model"):
        module = types.ModuleType(name)
        module.__path__ = []
        stubs[name] = module
    stubs["comfy.utils"].common_upscale = mock.Mock(
        side_effect=AssertionError("These checks must not call the host image resizer")
    )
    stubs["comfy.ldm.minimax.model"].PackedLayout = HostPackedLayout
    with mock.patch.dict(sys.modules, stubs):
        yield SimpleNamespace(**{
            alias: importlib.import_module(f"{PACKAGE}.director.{name}")
            for alias, name in (
                ("refine", "refine_pack"),
                ("bridge", "semantic_bridge"),
                ("cond", "selflift.cond"),
                ("face", "face_refine.pack"),
            )
        })


@pytest.mark.parametrize("mode,expected", [
    ("inherit", [123, 123, 123]),
    ("offset", [124, 125, 126]),
    ("independent", [700, 701, 702]),
])
def test_refine_seed_modes_survive_pack_normalization(features, mode, expected):
    refine = features.refine
    raw = refine.pack_refine(seed_mode=mode, seed=700, passes=3)
    normalized = refine.normalize_refine_pack(raw, base_width=864, base_height=480)
    assert [refine.refine_seed_for(normalized, 123, index) for index in range(3)] == expected
    assert raw["seed"] == normalized["seed"] == 700


def test_independent_refine_seed_changes_refine_fingerprint_only(features):
    refine = features.refine
    plan = SimpleNamespace(seed=123, refine=refine.pack_refine(seed_mode="independent", seed=700))
    before = refine.refine_fingerprint(plan)
    plan.refine = refine.pack_refine(seed_mode="independent", seed=701)
    after = refine.refine_fingerprint(plan)
    assert plan.seed == 123
    assert before["refine_seed"] == 700
    assert after["refine_seed"] == 701
    assert {key for key in before if before[key] != after[key]} == {"refine_seed"}


@pytest.mark.parametrize("mode", ["upscale", "latent_upscale"])
@pytest.mark.parametrize("canvas", [(864, 480), (480, 864), (640, 640)])
def test_refine_upscale_uses_first_pass_aspect_over_legacy_target(features, mode, canvas):
    refine = features.refine
    width, height = canvas
    raw = refine.pack_refine(mode=mode, aspect_ratio="1:1", megapixels=1.8,
                             target_width=320, target_height=960)
    result = refine.normalize_refine_pack(raw, base_width=width, base_height=height)
    target_w, target_h = result["target_width"], result["target_height"]
    assert result["aspect_ratio"] == refine.FOLLOW_DIRECTOR_ASPECT
    assert target_w % 32 == target_h % 32 == 0
    assert target_w >= width and target_h >= height
    assert target_w / target_h == pytest.approx(width / height, rel=0.035)
    assert target_w * target_h == pytest.approx(1.8 * 1024 * 1024, rel=0.04)


def test_refine_small_megapixel_target_never_downscales(features):
    result = features.refine.normalize_refine_pack(
        features.refine.pack_refine(mode="upscale", megapixels=0.1),
        base_width=1536, base_height=864,
    )
    assert result["target_width"] >= 1536
    assert result["target_height"] >= 864


def test_refine_preserves_wired_sigmas_and_second_pass_model(features):
    sigmas = torch.tensor([0.85, 0.725, 0.4219, 0.0])
    model = object()
    refine = features.refine
    packed = refine.pack_refine(sigmas=sigmas, sample_model=model)
    result = refine.normalize_refine_pack(packed)
    assert result["sigmas_tensor"] is sigmas
    assert refine.refine_sigmas_override(result) == pytest.approx(sigmas.tolist())
    assert refine.refine_model_for(result, object()) is model
    assert result["has_sigmas_tensor"] is True


def test_optional_features_unwired_leave_cache_and_conditioning_unchanged(features):
    positive = [[torch.ones(1, 2, 5120), {"custom": object()}]]
    plan = SimpleNamespace()
    result, note = features.bridge.apply_semantic_bridge(positive, plan)
    assert result is positive and note is None
    assert features.bridge.semantic_bridge_fingerprint(plan) == {}
    assert features.face.face_refine_fingerprint(plan) == {}
    assert features.refine.normalize_refine_pack(None) is None
    assert features.bridge.normalize_semantic_bridge_pack(None) is None
    assert features.face.normalize_face_refine_pack(None) is None


@pytest.mark.parametrize("magnitude_match,expected", [(False, 2.5), (True, 3.0)])
def test_semantic_bridge_rewrites_hidden_preserving_metadata_and_idempotence(features, magnitude_match, expected):
    bridge = features.bridge
    hidden = torch.full((1, 2, 5120), 3.0)
    keyframes = [{"latent": torch.ones(1, 4, 2, 8, 12), "index": 5}]
    refs = {"audio": object(), "image": object()}
    meta = {"minimax_keyframes": keyframes, "refs": refs, "custom": object()}
    extra = object()
    second_item = [torch.ones(1, 1, 5120), {"negative": True}]
    positive = [[hidden, meta, extra], second_item]
    plan = SimpleNamespace(semantic_bridge=bridge.pack_semantic_bridge(
        adapter="test.pt", alpha=0.25, magnitude_match=magnitude_match))

    class ConstantStudent(torch.nn.Module):
        def forward(self, tokens):
            return torch.ones_like(tokens)

    with mock.patch.object(bridge, "resolve_semantic_bridge_path", return_value="test.pt"), \
         mock.patch.object(bridge, "load_semantic_student", return_value=ConstantStudent()) as load:
        result, note = bridge.apply_semantic_bridge(positive, plan, task_key="i2v")
        again, second_note = bridge.apply_semantic_bridge(result, plan, task_key="i2v")
    torch.testing.assert_close(result[0][0], torch.full_like(hidden, expected))
    torch.testing.assert_close(hidden, torch.full_like(hidden, 3.0))
    assert result[0][0].dtype == hidden.dtype
    assert result[0][1]["minimax_keyframes"] is keyframes
    assert result[0][1]["refs"] is refs
    assert result[0][1]["custom"] is meta["custom"]
    assert result[0][2] is extra and result[1] is second_item
    assert bridge.META_APPLIED_KEY not in meta
    assert result[0][1][bridge.META_APPLIED_KEY] is True
    assert note and again is result and second_note is None
    load.assert_called_once()


def test_semantic_bridge_missing_adapter_is_explicit_noop(features):
    positive = [[torch.ones(1, 2, 5120), {}]]
    plan = SimpleNamespace(semantic_bridge=features.bridge.pack_semantic_bridge(adapter="missing.pt"))
    with mock.patch.object(features.bridge, "resolve_semantic_bridge_path", return_value=None):
        result, note = features.bridge.apply_semantic_bridge(positive, plan)
    assert result is positive
    assert "adapter not found" in note


def test_selflift_resize_preserves_temporal_audio_refs_and_semantic_metadata(features):
    video = torch.arange(3.0).reshape(1, 1, 3, 1, 1).expand(1, 4, 3, 8, 12).clone()
    hidden = torch.ones(1, 2, 5120)
    refs = [{"kind": "ref_audio", "latent": torch.ones(1, 2, 11)}]
    keyframes = [{"latent": video, "index": 22, "strength": 0.7}]
    payload = {"keyframes": keyframes, "refs": refs, "frame_count": 39,
               "cond_video_latents": [video],
               "layout": HostPackedLayout(2, 3, 8, 12, 11)}
    cond = SimpleNamespace(cond=payload)
    shapes = SimpleNamespace(cond=[tuple(video.shape), (1, 2, 11)])
    meta = {"model_conds": {"payload": cond, "shapes": shapes},
            "minimax_keyframes": keyframes, "mmx_semantic_bridge": True,
            "custom_audio": refs}
    positive = [[hidden, meta]]
    result = features.cond.resize_positive_spatial(positive, 8, 12, 4, 6)
    result_meta = result[0][1]
    result_payload = result_meta["model_conds"]["payload"].cond
    for resized in (result_payload["keyframes"][0]["latent"],
                    result_payload["cond_video_latents"][0],
                    result_meta["minimax_keyframes"][0]["latent"]):
        assert tuple(resized.shape) == (1, 4, 3, 4, 6)
        torch.testing.assert_close(resized[0, 0, :, 0, 0], torch.arange(3.0))
    assert result_payload["layout"].signature == (2, 3, 4, 6, 11)
    assert result_payload["layout"].frame_count == 39
    assert result_payload["refs"] is refs
    assert result_meta["custom_audio"] is refs
    assert result_meta["mmx_semantic_bridge"] is True
    assert result_meta["minimax_keyframes"][0]["index"] == 22
    assert result_meta["minimax_keyframes"][0]["strength"] == 0.7
    assert result_meta["model_conds"]["shapes"].cond == [(1, 4, 3, 4, 6), (1, 2, 11)]
    assert result[0][0] is hidden
    assert cond.cond is payload and keyframes[0]["latent"].shape[-2:] == (8, 12)
    assert features.cond.resize_positive_spatial(positive, 8, 12, 8, 12) is positive


def test_selflift_av_resize_keeps_audio_and_resizes_video_mask(features):
    video = torch.ones(1, 4, 3, 8, 12)
    audio = torch.arange(22.0).reshape(1, 2, 11)
    mask = torch.ones(1, 1, 3, 8, 12)
    latent = {"samples": (video, audio), "noise_mask": mask, "custom": object()}
    result = features.cond.resize_av_video(latent, 4, 6)
    assert result["samples"][0].shape == (1, 4, 3, 4, 6)
    assert result["samples"][1] is audio
    assert result["noise_mask"].shape == (1, 1, 3, 4, 6)
    assert result["custom"] is latent["custom"]
    assert latent["samples"][0] is video and latent["noise_mask"] is mask


def test_face_refine_preserves_explicit_schedule_and_postprocess_configuration(features):
    sigmas = torch.tensor([0.7, 0.3, 0.0])
    packed = features.face.pack_face_refine(
        sigmas=sigmas, canvas_width=768, canvas_height=512,
        seed_mode="offset", select="centre_most", paste_region="face_ellipse",
        blend=0.8, feather=32,
    )
    normalized = features.face.normalize_face_refine_pack(packed)
    assert normalized["sigmas_tensor"] is sigmas
    assert normalized["has_sigmas_tensor"] is True
    for name, expected in {"canvas_width": 768, "canvas_height": 512,
                           "seed_mode": "offset", "select": "centre_most",
                           "paste_region": "face_ellipse", "blend": 0.8, "feather": 32}.items():
        assert normalized[name] == expected
    assert features.face.face_refine_enabled(SimpleNamespace(face_refine=normalized))
    assert features.face.normalize_face_refine_pack({"enabled": False}) is None
