"""Spatial tiling must crop the model-side denoise mask together with the latent.

Regression for: MiniMax H3 Director + SelfLift + 引导+重绘 + 引用上段
    RuntimeError: The size of tensor a (60) must match the size of tensor b (86)
    at non-singleton dimension 4
raised from comfy/ldm/minimax/model.py ``out[0] = out[0] * denoise_mask``.

The Sampler unpacks the packed noise mask with the *full* latent shapes once,
stores it in the conds, and every tiled model call then got a tile-sized ``x``
next to a full-canvas mask. No ComfyUI import needed: the tiny pack/unpack
helpers are stubbed with the same implementation as ``comfy.utils``.
"""
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import torch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PLUGIN_ROOT / "director" / "spatial_tiled_sampling.py"
    spec = importlib.util.spec_from_file_location("spatial_tiles_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack_latents(latents):
    shapes = []
    tensors = []
    for tensor in latents:
        shapes.append(tuple(tensor.shape))
        tensors.append(tensor.reshape(tensor.shape[0], 1, -1))
    return torch.cat(tensors, dim=-1), shapes


def _unpack_latents(combined, shapes):
    if len(shapes) > 1:
        out = []
        for shape in shapes:
            cut = math.prod(shape[1:])
            tens = combined[:, :, :cut]
            combined = combined[:, :, cut:]
            out.append(tens.reshape([tens.shape[0]] + list(shape)[1:]))
        return out
    return [combined]


class _Cond:
    def __init__(self, cond):
        self.cond = cond


class _Layout:
    """Stand-in for comfy.ldm.minimax.model.PackedLayout."""

    def __init__(self, text_len, latent_t, lat_h, lat_w, audio_t, frame_count=None,
                 keyframes=None, refs=None):
        self.signature = (text_len, latent_t, lat_h, lat_w, audio_t)
        self.segments = [(0, text_len, "text")]
        self.img_update = None


class _Base:
    def __init__(self, latent_shapes):
        self.latent_shapes = latent_shapes


class _Guider:
    def __init__(self, conds, base):
        self.conds = conds
        self.inner_model = base


class _FakeModel:
    """Mimics the KSamplerX0Inpaint -> DiT forward path, mask multiply included."""

    def __init__(self, guider, tile_shapes_box):
        self.inner_model = guider
        self._tile_box = tile_shapes_box
        self.calls = []

    def __call__(self, x, sigma, denoise_mask=None, **kwargs):
        shapes = self._tile_box["shapes"]
        streams = _unpack_latents(x, shapes)
        video = streams[0]
        cond = self.inner_model.conds["positive"][0][1]["model_conds"].get("denoise_mask")
        mask = cond.cond if cond is not None else None
        if denoise_mask is not None:
            packed_mask = _unpack_latents(denoise_mask, shapes)[0]
            self.calls.append(("tile", tuple(video.shape), tuple(packed_mask.shape)))
        # this line is what blew up in comfy/ldm/minimax/model.py
        out_video = video if mask is None else video * mask
        packed, _ = _pack_latents([out_video, streams[1]])
        return packed


class SpatialTileMaskTests(unittest.TestCase):
    def setUp(self):
        self.torch = torch
        self.module = _load_module()

        comfy = types.ModuleType("comfy")
        utils = types.ModuleType("comfy.utils")
        utils.pack_latents = _pack_latents
        utils.unpack_latents = _unpack_latents
        comfy.utils = utils

        ldm = types.ModuleType("comfy.ldm")
        minimax = types.ModuleType("comfy.ldm.minimax")
        model = types.ModuleType("comfy.ldm.minimax.model")
        model.PackedLayout = _Layout
        minimax.model = model
        ldm.minimax = minimax
        comfy.ldm = ldm

        self._saved = {name: sys.modules.get(name) for name in
                       ("comfy", "comfy.utils", "comfy.ldm", "comfy.ldm.minimax",
                        "comfy.ldm.minimax.model")}
        sys.modules.update({
            "comfy": comfy,
            "comfy.utils": utils,
            "comfy.ldm": ldm,
            "comfy.ldm.minimax": minimax,
            "comfy.ldm.minimax.model": model,
        })

    def tearDown(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _scene(self, *, height=48, width=86, steps=44, prefix=7):
        video = torch.zeros((1, 24, steps, height, width))
        audio = torch.zeros((1, 2, 2, 155))
        full_shapes = [tuple(video.shape), tuple(audio.shape)]
        mask = torch.ones((1, 1, steps, height, width))
        mask[:, :, :prefix] = 0.0  # 引用上段: pinned prefix rows
        packed_x, _ = _pack_latents([video, audio])
        # comfy.sampler_helpers.prepare_mask repeats the mask across latent channels
        # before the sampler packs it, so the packed stream is latent-sized.
        wide_mask = mask.expand(1, int(video.shape[1]), steps, height, width).contiguous()
        packed_mask, _ = _pack_latents([wide_mask, torch.ones((1, 2, 2, 155))])
        payload = {"layout": None, "keyframes": None, "refs": None,
                   "frame_count": None, "seed": 0, "audio_scale": 1.0}
        conds = {"positive": [[torch.zeros(1), {"model_conds": {
            "latent_shapes": _Cond(full_shapes),
            "minimax_payload": _Cond(payload),
            "denoise_mask": _Cond(mask),
        }}]]}
        base = _Base(full_shapes)
        tile_box = {}
        fake = _FakeModel(_Guider(conds, base), tile_box)
        return fake, base, packed_x, packed_mask, full_shapes, conds, tile_box

    def test_mask_cond_is_cropped_per_tile(self):
        fake, base, packed_x, packed_mask, full_shapes, conds, tile_box = self._scene()

        plan = self.module._plan_regions((1, 24, 44, 48, 86), 2, 128)
        self.assertEqual([(a, b) for a, b, *_ in plan["regions"]], [(0, 59), (35, 86)])

        real_unpack = self.module._crop_packed

        def crop_packed_spy(x, shapes, axis, start, end):
            packed, tile_shapes, pad = real_unpack(x, shapes, axis, start, end)
            tile_box["shapes"] = tile_shapes
            return packed, tile_shapes, pad

        self.module._crop_packed = crop_packed_spy
        self.addCleanup(setattr, self.module, "_crop_packed", real_unpack)

        # before the fix this raised: tensor a (60) vs tensor b (86) at dim 4
        out = self.module._tiled_model_call(
            fake, packed_x, torch.tensor(0.5),
            denoise_mask=packed_mask,
            n_tiles=2, overlap_pixels=128,
        )

        self.assertEqual(tuple(out.shape), tuple(packed_x.shape))
        self.assertEqual(len(fake.calls), 2)
        for _, video_shape, mask_shape in fake.calls:
            self.assertEqual(video_shape, mask_shape, "tile latent and tile mask must match")
        self.assertEqual([c[1][-1] for c in fake.calls], [60, 52])

        # conds and latent_shapes are restored for the next step / sampler
        restored = conds["positive"][0][1]["model_conds"]["denoise_mask"].cond
        self.assertEqual(tuple(restored.shape), (1, 1, 44, 48, 86))
        self.assertEqual(base.latent_shapes, full_shapes)

    def test_without_mask_cond_nothing_changes(self):
        fake, base, packed_x, packed_mask, full_shapes, conds, tile_box = self._scene()
        del conds["positive"][0][1]["model_conds"]["denoise_mask"]

        real_unpack = self.module._crop_packed

        def crop_packed_spy(x, shapes, axis, start, end):
            packed, tile_shapes, pad = real_unpack(x, shapes, axis, start, end)
            tile_box["shapes"] = tile_shapes
            return packed, tile_shapes, pad

        self.module._crop_packed = crop_packed_spy
        self.addCleanup(setattr, self.module, "_crop_packed", real_unpack)

        out = self.module._tiled_model_call(
            fake, packed_x, torch.tensor(0.5),
            denoise_mask=None, n_tiles=2, overlap_pixels=128,
        )
        self.assertEqual(tuple(out.shape), tuple(packed_x.shape))
        self.assertEqual(len(fake.calls), 0)


if __name__ == "__main__":
    unittest.main()
