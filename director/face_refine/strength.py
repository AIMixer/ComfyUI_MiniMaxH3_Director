"""Face-size strength ramp. Adapted from Carasibana/ComfyUI-H3-FaceRefine (MIT).
Copyright (c) 2026 Carasibana. See UPSTREAM_LICENSE in this directory.
"""
import numpy as np
import torch
import torch.nn.functional as F
import comfy.nested_tensor
from .track import _smooth

def _renoised_inpaint(model):
    """Blend held frames toward an original re-noised to the CURRENT sigma.

    H3 hands a preserved region back as 0.999*clean + 0.001*noise, and tells the model
    through per-token timesteps that it sits at H3's conditioning timestep. With that
    telling suppressed the injected latent is far cleaner than the step it lands in, and
    the sampler mixes (1 - mask) of it in on EVERY step, so the disagreement grows with
    the ramp.

    Re-noising puts those frames at the sigma the sampler is actually on, so there is
    nothing left to declare - the latent is simply true. This is what models without a
    conditioning-timestep scheme already do for a masked region. The audio half keeps
    H3's own rescale untouched.
    """
    m = model.clone()
    base = getattr(m, "model", None)
    if base is None or not hasattr(base, "scale_latent_inpaint"):
        return m, "this model reads no denoise mask; nothing to re-noise"
    orig = base.scale_latent_inpaint

    def _fn(sigma, noise, latent_image, x=None, denoise_mask=None, **kwargs):
        import comfy.utils
        import comfy.ldm.minimax.model as _mmx
        shapes = getattr(base, "latent_shapes", None)
        if shapes is None or len(shapes) < 2:
            return orig(sigma=sigma, noise=noise, latent_image=latent_image, **kwargs)
        cleans = comfy.utils.unpack_latents(latent_image, shapes)
        noises = comfy.utils.unpack_latents(noise, shapes)
        ms = base.model_sampling
        s = sigma.reshape([sigma.shape[0]] + [1] * (cleans[0].ndim - 1))
        cleans[0] = ms.noise_scaling(s, noises[0], cleans[0])
        scale = base.audio_scale()
        if scale != 1.0:
            sigma_v = sigma.clamp(min=1e-6)
            sigma_a = _mmx.time_shift_sigma(sigma_v, ms.shift, ms.audio_shift)
            factor = (sigma_v / sigma_a) / scale
            cleans[1] = cleans[1] * factor.view(
                factor.shape[:1] + (1,) * (cleans[1].ndim - 1)).to(cleans[1].dtype)
        return comfy.utils.pack_latents(cleans)[0]

    m.add_object_patch("scale_latent_inpaint", _fn)
    return m, "held frames re-noised to the sigma the sampler is on"


def _mask_via_sampler_only(model):
    """Return a clone that keeps this node's per-frame video mask out of the model.

    A mask otherwise reaches the result twice: the sampler blends by it, and H3 also
    shifts each token's timestep by it. The model then predicts at a timestep the latent
    it was handed does not sit at, and that disagreement prints as a repeating grid at
    latent-cell size, for a uniform mask as much as a varying one.

    Only the VIDEO entry is withheld. The audio entry is how the model is told the audio
    is being held, which is what MiniMaxH3NativeAudioLock relies on to drive lipsync;
    this node writes nothing to the audio side and must not disturb what reads it.

    Pair with _renoised_inpaint: withholding the timestep leaves the held frames cleaner
    than the step they sit on, and that re-noises them to match.
    """
    m = model.clone()
    base = getattr(m, "model", None)
    if base is None or not hasattr(base, "_denoise_mask_conds"):
        return m, "this model reads no denoise mask; nothing to keep out of it"

    orig_conds = base._denoise_mask_conds

    def _audio_only_conds(denoise_mask, latent_shapes):
        out = dict(orig_conds(denoise_mask, latent_shapes))
        out.pop("denoise_mask", None)          # video: withheld
        return out                             # audio: passed through untouched

    m.add_object_patch("_denoise_mask_conds", _audio_only_conds)
    return m, "video mask applied by the sampler only; the audio lock reaches the model intact"



def apply_face_strength(model, latent, transform, pack):
    """Apply the video mask through the sampler, preserving the joint audio mask."""
    patched, note = _mask_via_sampler_only(model)
    patched, renote = _renoised_inpaint(patched)
    members = list(latent["samples"].unbind())
    video = members[0]
    face = np.asarray(transform["face_sizes"], dtype=np.float64)
    lo = float(pack["face_px_small"])
    hi = max(lo + 1.0, float(pack["face_px_large"]))
    t = np.clip((face - lo) / (hi - lo), 0.0, 1.0)
    strength = 1.0 + (float(pack["large_face_multiplier"]) - 1.0) * t
    for a, b in transform.get("shot_ranges") or [(0, len(strength))]:
        strength[a:b] = _smooth(strength[a:b], 9)
    detected = np.asarray(transform["detected"], dtype=bool)
    strength[~detected] = 0.0
    s = torch.from_numpy(strength.copy()).float().view(1, 1, -1)
    s = F.interpolate(s, size=video.shape[-3], mode="linear", align_corners=True)
    vmask = s.reshape(1, 1, video.shape[-3], 1, 1).to(video).expand_as(video).contiguous()
    previous = latent.get("noise_mask")
    if previous is not None:
        masks = list(previous.unbind())
        masks[0] = vmask
    else:
        masks = [vmask] + [torch.zeros_like(member) for member in members[1:]]
    out = dict(latent)
    out["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(masks))
    return patched, out, (
        f"face-size multiplier {strength.min():.2f}–{strength.max():.2f}; {note}; {renote}"
    )
