"""Run a Director segment through the installed selflift-Avatar sampler."""

from __future__ import annotations

import torch

from ..core_sampling import _unpack_node_output


def sample_avatar_stage(*, model, positive, negative, latent, vae, seed, cfg, steps,
                        scheduler, sigmas, pack, shift_video, shift_audio, shift_cache,
                        after_shift=None):
    if after_shift is not None:
        raise ValueError("原版 SelfLift 不支持导演台的动态续写遮罩；请将段间衔接改为引导模式。")

    from nodes import ConditioningZeroOut, NODE_CLASS_MAPPINGS
    from comfy_extras.nodes_custom_sampler import BasicScheduler

    sampler_class = NODE_CLASS_MAPPINGS.get("SelfLiftAvatarH3Sampler")
    if sampler_class is None:
        raise RuntimeError("需要安装并启用 selflift-Avatar 的 SelfLiftAvatarH3Sampler。")
    if pack.get("sampler") is None:
        raise ValueError("原版 SelfLift 桥接节点需要连接 Euler 采样器。")

    model = shift_cache.get(model, shift_video, shift_audio)
    high_model = pack.get("sample_model")
    if high_model is not None:
        high_model = shift_cache.get(high_model, shift_video, shift_audio)
    if sigmas is None:
        sigmas = _unpack_node_output(BasicScheduler.execute(model, scheduler, int(steps), 1.0))[0]
    else:
        sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
    if not negative:
        negative = ConditioningZeroOut().zero_out(positive)[0]

    result = sampler_class().sample(
        low_res_model=model, high_res_model=high_model,
        positive=positive, negative=negative, vae=vae, latent_image=latent,
        sampler=pack["sampler"], sigmas=sigmas, seed=seed, cfg=cfg,
        transition_step=pack["transition_step"], lowres_scale=pack["lowres_scale"],
        rho=pack["rho"], w_min=pack["w_min"], w_max=pack["w_max"],
        upscaler_model=pack["h3_latent_model"] or "none",
        highres_tiling=pack["enable_tiling"], tiling_mode=pack["avatar_tiling_mode"],
        tiling_tiles=pack["tile_count"], tiling_axis=pack["avatar_tiling_axis"],
        highres_tiling_mode=pack["avatar_tiling_method"],
        highres_token_chunk=pack["avatar_token_chunk"],
        highres_weight_cache_gb=pack["avatar_weight_cache_gb"],
    )
    return result["result"][0]
