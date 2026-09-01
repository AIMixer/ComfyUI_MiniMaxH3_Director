"""Independent ComfyUI MiniMax H3 runtime patches for Director segment continuity.

Enables interior keyframe time coordinates and keyframe+reference payload
coexistence so the Director can pin a previous segment's tail into the next
segment. Behavior is inspired by community Motion Context work; this module is
an original Apache-2.0 implementation for AIMixer/ComfyUI_MiniMaxH3_Director.

Does not copy third-party GPL sources. Refuses to stack on foreign H3 layout /
payload wrappers (including standalone Motion Context packs).
"""

from __future__ import annotations

import inspect
import logging

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.h3_context_patches")

# Director-owned markers (not shared with third-party packs).
CTX_FRAME_KEY = "director_context_index"
CTX_AUDIO_END_KEY = "director_context_audio_end"
LAYOUT_MARKER = "_h3_director_continuity_layout_patch"
PAYLOAD_MARKER = "_h3_director_continuity_payload_patch"

# Known foreign markers — stand down / refuse rather than double-wrap.
_FOREIGN_LAYOUT_MARKERS = (
    LAYOUT_MARKER,
    "_h3_motion_context_layout_patch",
)
_FOREIGN_PAYLOAD_MARKERS = (
    PAYLOAD_MARKER,
    "_h3_motion_context_payload_patch",
)

_layout_orig = None
_layout_applied = False
_payload_orig = None
_payload_applied = False

_REF_KINDS = ("ref_img", "ref_audio")


def _mm():
    import comfy.ldm.minimax.model as mm

    return mm


def layout_patch_applied() -> bool:
    return _layout_applied


def payload_patch_applied() -> bool:
    return _payload_applied


def _has_marker(fn, markers: tuple[str, ...]) -> str | None:
    if fn is None:
        return None
    for name in markers:
        if getattr(fn, name, False):
            return name
    return None


def _target_origin(layout) -> float:
    """Time coordinate where the target video segment begins."""
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "Director continuity: expected final layout segment to be target video "
            f"(got {kind!r} spanning {b - a} rows)."
        )
    return float(layout.position_ids[a, 0])


def _keyframe_time(mm, text_len: float, latent_t: int, frame_count: int | None, pixel_index: int) -> float:
    """Stock-compatible time for pixel frame ``pixel_index`` on the target clip."""
    p = int(pixel_index)
    if p == 0:
        return float(text_len)
    if frame_count is not None and p == int(frame_count) - 1:
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(text_len) + mm.FRAME_RESCALE * float(p)


def _expected_ref_kinds(block: dict) -> tuple[str, ...]:
    kind = block.get("kind")
    if kind == "image":
        return ("ref_img",)
    if kind == "audio":
        return ("ref_audio",) if int(block.get("ref_audio_t", 0) or 0) > 0 else ()
    if kind in ("video", "video_audio"):
        if int(block.get("ref_audio_t", 0) or 0) > 0:
            return ("ref_audio", "ref_img")
        return ("ref_img",)
    raise RuntimeError(f"Director continuity: unknown reference kind {kind!r}.")


def _ref_segments(layout, refs: list | None) -> dict[int, dict[str, tuple[int, int]]]:
    ref_segs = [(a, b, k) for a, b, k in layout.segments if k in _REF_KINDS]
    wanted = [(i, k) for i, blk in enumerate(refs or []) for k in _expected_ref_kinds(blk)]
    if len(wanted) != len(ref_segs):
        raise RuntimeError(
            "Director continuity: reference segment count mismatch "
            f"(expected {len(wanted)}, layout has {len(ref_segs)})."
        )
    out: dict[int, dict[str, tuple[int, int]]] = {}
    for (i, kind), (a, b, got) in zip(wanted, ref_segs):
        if got != kind:
            raise RuntimeError(
                f"Director continuity: reference block {i} expected {kind}, got {got}."
            )
        out.setdefault(i, {})[kind] = (a, b)
    return out


def _rewrite_keyframe_times(layout, text_len, latent_t, frame_count, keyframes, refs=None):
    del refs  # origin is read from the built layout
    mm = _mm()
    offset = _target_origin(layout) - float(text_len)
    if offset and any(kf.get(CTX_FRAME_KEY) is None for kf in keyframes):
        raise RuntimeError(
            "Director continuity: cannot mix unmarked stock keyframes with "
            "director context anchors when references shift the target origin."
        )
    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            f"Director continuity: expected {len(keyframes)} cond segments, "
            f"layout has {len(cond_spans)}."
        )
    for (a, b), kf in zip(cond_spans, keyframes):
        p = kf.get(CTX_FRAME_KEY)
        if p is None:
            continue
        layout.position_ids[a:b, 0] = _keyframe_time(
            mm, text_len, latent_t, frame_count, int(p)
        ) + offset


def _rewrite_audio_timeline(layout, text_len, refs):
    del text_len
    mm = _mm()
    marked = [i for i, r in enumerate(refs or []) if r.get(CTX_AUDIO_END_KEY) is not None]
    if len(marked) != 1:
        raise RuntimeError(
            "Director continuity: audio continuation requires exactly one marked "
            f"audio reference (found {len(marked)})."
        )
    idx = marked[0]
    blk = refs[idx]
    if blk.get("kind") != "audio":
        raise RuntimeError(
            f"Director continuity: {CTX_AUDIO_END_KEY} set on non-audio ref "
            f"{blk.get('kind')!r}."
        )
    rt = int(blk.get("ref_audio_t", 0) or 0)
    if rt <= 0:
        return
    seg = _ref_segments(layout, refs).get(idx, {}).get("ref_audio")
    if seg is None:
        raise RuntimeError("Director continuity: marked audio ref produced no layout rows.")
    a, b = seg
    if b - a != 2 * rt:
        raise RuntimeError(
            f"Director continuity: audio ref rows {b - a} != expected stereo {2 * rt}."
        )
    origin = _target_origin(layout)
    slot_start = float(layout.position_ids[a, 0])
    end_frame = float(blk[CTX_AUDIO_END_KEY])
    desired_start = origin + mm.FRAME_RESCALE * end_frame - float(rt)
    layout.position_ids[a:b, 0] = layout.position_ids[a:b, 0] + (desired_start - slot_start)


def _stock_layout_kwargs(keyframes=None, refs=None, frame_count=None):
    """Forward only kwargs the installed PackedLayout.__init__ actually accepts."""
    kwargs = {"keyframes": keyframes, "refs": refs}
    orig = _layout_orig
    if orig is None:
        return kwargs
    try:
        params = inspect.signature(orig).parameters
    except (TypeError, ValueError):
        return kwargs
    if "frame_count" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        kwargs["frame_count"] = frame_count
    return kwargs


def _call_layout_orig(
    obj,
    text_len,
    latent_t,
    latent_h,
    latent_w,
    audio_t,
    keyframes=None,
    refs=None,
    frame_count=None,
):
    kwargs = _stock_layout_kwargs(keyframes, refs, frame_count)
    try:
        _layout_orig(obj, text_len, latent_t, latent_h, latent_w, audio_t, **kwargs)
    except TypeError as exc:
        if "frame_count" not in str(exc) or "frame_count" not in kwargs:
            raise
        kwargs.pop("frame_count", None)
        _layout_orig(obj, text_len, latent_t, latent_h, latent_w, audio_t, **kwargs)


def _director_layout_init(
    self,
    text_len,
    latent_t,
    latent_h,
    latent_w,
    audio_t,
    keyframes=None,
    refs=None,
    frame_count=None,
):
    # Stock accepts only first/last; pass interior anchors as index 0, then rewrite.
    stock_keyframes = None
    if keyframes:
        stock_keyframes = []
        for kf in keyframes:
            entry = dict(kf)
            if CTX_FRAME_KEY in entry:
                entry["resolved_frame_index"] = 0
            stock_keyframes.append(entry)
    _call_layout_orig(
        self,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=stock_keyframes,
        refs=refs,
        frame_count=frame_count,
    )
    has_ctx_kf = bool(keyframes) and any(kf.get(CTX_FRAME_KEY) is not None for kf in keyframes)
    has_ctx_audio = bool(refs) and any(r.get(CTX_AUDIO_END_KEY) is not None for r in refs)
    if has_ctx_kf:
        _rewrite_keyframe_times(self, text_len, latent_t, frame_count, keyframes, refs)
    if has_ctx_audio:
        _rewrite_audio_timeline(self, text_len, refs)


setattr(_director_layout_init, LAYOUT_MARKER, True)


def _self_test_layout() -> None:
    mm = _mm()
    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))

    dummy = torch.zeros(1, 1, 1, 1, 1)

    def _kf(pixel_index: int, *, marked: bool = False) -> dict:
        entry = {
            "resolved_frame_index": 0 if marked else int(pixel_index),
            "latent": dummy,
        }
        if marked:
            entry[CTX_FRAME_KEY] = int(pixel_index)
        return entry

    def build(keyframes=None, refs=None, rewrite=False):
        lay = mm.PackedLayout.__new__(mm.PackedLayout)
        stock_kf = None
        if keyframes:
            stock_kf = []
            for kf in keyframes:
                entry = dict(kf)
                if rewrite and CTX_FRAME_KEY in entry:
                    entry["resolved_frame_index"] = 0
                stock_kf.append(entry)
        _call_layout_orig(
            lay,
            text_len,
            latent_t,
            lh,
            lw,
            audio_t,
            keyframes=stock_kf,
            refs=refs,
            frame_count=frame_count,
        )
        if rewrite:
            _rewrite_keyframe_times(lay, text_len, latent_t, frame_count, keyframes, refs)
        return lay

    stock = build(keyframes=[_kf(0), _kf(frame_count - 1)])
    ours = build(
        keyframes=[_kf(0, marked=True), _kf(frame_count - 1, marked=True)],
        rewrite=True,
    )
    if not torch.allclose(stock.position_ids, ours.position_ids, rtol=0, atol=1e-9):
        raise RuntimeError("Director continuity layout self-test: endpoint mismatch vs stock")

    run = [_kf(i, marked=True) for i in range(4)]
    interior = build(keyframes=run, rewrite=True)
    times = [float(interior.position_ids[a, 0]) for a, _, k in interior.segments if k == "cond"]
    if len(times) != 4 or any(times[i] >= times[i + 1] for i in range(3)):
        raise RuntimeError("Director continuity layout self-test: interior times not increasing")


def _classify_layout_owner() -> str | None:
    """Return None, 'ours', 'compatible_solattn', 'foreign_mc', or 'foreign_other'."""
    mm = _mm()
    init = getattr(getattr(mm, "PackedLayout", None), "__init__", None)
    if init is None:
        return None
    if getattr(init, LAYOUT_MARKER, False):
        return "ours"
    if getattr(init, "_h3_motion_context_layout_patch", False):
        return "foreign_mc"
    # SolAttn_triton only observes H3 PackedLayout construction. Its exact
    # wrapper can safely remain underneath Director's continuity wrapper.
    # Unknown wrappers are still rejected by the checks below.
    module = str(getattr(init, "__module__", "")).replace("\\", "/")
    qualname = getattr(init, "__qualname__", "")
    if (
        module.endswith("ComfyUI-SolAttn_triton._morton_h3")
        and qualname == "_patch_packed_layout.<locals>.__init__"
    ) or (
        module.endswith("sol_attn_minimax_v5")
        and qualname == "_patch_packed_layout.<locals>.__init__"
    ):
        return "compatible_solattn"
    if getattr(init, "__name__", "") in {"_patched_init", "_director_layout_init"}:
        return "foreign_other"
    if hasattr(init, "__wrapped__"):
        return "foreign_other"
    home = getattr(mm.PackedLayout, "__module__", None)
    where = getattr(init, "__module__", None)
    if home and where and where != home:
        return "foreign_other"
    return None


def ensure_layout_patch() -> bool:
    """Install layout patch on first continuity use. Returns True if usable."""
    global _layout_orig, _layout_applied
    if _layout_applied:
        return True
    owner = _classify_layout_owner()
    if owner == "ours":
        _layout_applied = True
        return True
    if owner == "foreign_mc":
        raise RuntimeError(
            "Director continuity: standalone ComfyUI-H3-Motion-Context (or a fork) "
            "already patched MiniMax H3 layout. Disable that custom node pack and "
            "restart ComfyUI — both packs cannot own PackedLayout.__init__."
        )
    if owner == "compatible_solattn":
        log.info("Director continuity: composing with SolAttn H3 layout observer")
    if owner == "foreign_other":
        raise RuntimeError(
            "Director continuity: another pack already patched MiniMax H3 "
            "PackedLayout.__init__. Disable the other pack and restart ComfyUI."
        )
    mm = _mm()
    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        raise RuntimeError("Director continuity: MiniMax H3 model module incomplete.")
    _layout_orig = mm.PackedLayout.__init__
    try:
        _self_test_layout()
    except Exception as exc:
        _layout_orig = None
        raise RuntimeError(
            f"Director continuity: layout self-test failed ({exc}). "
            "Interior keyframe anchors unavailable."
        ) from exc
    mm.PackedLayout.__init__ = _director_layout_init
    _layout_applied = True
    log.info("Director continuity: interior keyframe anchors enabled")
    return True


def _director_extra_conds(self, **kwargs):
    out = _payload_orig(self, **kwargs)
    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out
    if not (
        any(CTX_FRAME_KEY in kf for kf in keyframes)
        or any(CTX_AUDIO_END_KEY in r for r in refs)
    ):
        return out
    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        log.warning("Director continuity: could not reach H3 payload for keyframe+ref merge")
        return out
    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [r["latent"] for r in refs if "latent" in r]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [
        r["audio_latent"] for r in refs if r.get("audio_latent") is not None
    ]
    fc = kwargs.get("minimax_frame_count", None)
    if fc is not None:
        payload["frame_count"] = fc
    return out


setattr(_director_extra_conds, PAYLOAD_MARKER, True)


def _classify_payload_owner() -> str | None:
    import comfy.model_base as model_base

    cls = getattr(model_base, "MiniMaxH3", None)
    fn = getattr(cls, "extra_conds", None) if cls is not None else None
    if fn is None:
        return None
    if getattr(fn, PAYLOAD_MARKER, False):
        return "ours"
    if getattr(fn, "_h3_motion_context_payload_patch", False):
        return "foreign_mc"
    if getattr(fn, "__name__", "") in {"_patched_extra_conds", "_director_extra_conds"}:
        return "foreign_other"
    if hasattr(fn, "__wrapped__"):
        return "foreign_other"
    home = getattr(cls, "__module__", None)
    where = getattr(fn, "__module__", None)
    if home and where and where != home:
        return "foreign_other"
    return None


def ensure_payload_patch() -> bool:
    """Install payload merge patch when audio refs coexist with keyframes."""
    global _payload_orig, _payload_applied
    if _payload_applied:
        return True
    owner = _classify_payload_owner()
    if owner == "ours":
        _payload_applied = True
        return True
    if owner == "foreign_mc":
        raise RuntimeError(
            "Director continuity: standalone ComfyUI-H3-Motion-Context already patched "
            "MiniMaxH3.extra_conds. Disable that pack and restart ComfyUI."
        )
    if owner == "foreign_other":
        raise RuntimeError(
            "Director continuity: another pack already patched MiniMaxH3.extra_conds. "
            "Disable the other pack and restart ComfyUI."
        )
    import comfy.model_base as model_base

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        raise RuntimeError("Director continuity: MiniMaxH3.extra_conds not found.")
    _payload_orig = cls.extra_conds
    cls.extra_conds = _director_extra_conds
    _payload_applied = True
    log.info("Director continuity: keyframe/ref coexistence enabled")
    return True
