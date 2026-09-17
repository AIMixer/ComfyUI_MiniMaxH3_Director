"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")


def _evict_dead_loaded_models() -> int:
    """Pop Comfy LoadedModel slots that ``free_memory`` will skip forever.

    ``is_dead()`` means the ModelPatcher weakref is gone while the shared
    MiniMaxH3 module is still alive (graph MODEL / Sage cycle). Those slots
    log ``Potential memory leak detected with model MiniMaxH3`` and then sit
    in ``current_loaded_models``, so later unloads cannot touch them.
    Evicting the slot does not copy weights; it restores unload bookkeeping.
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    models = getattr(mm, "current_loaded_models", None)
    if not models:
        return 0
    evicted = 0
    for i in range(len(models) - 1, -1, -1):
        cur = models[i]
        try:
            if not cur.is_dead():
                continue
            name = "?"
            try:
                real = cur.real_model()
                name = type(real).__name__ if real is not None else "?"
            except Exception:
                pass
            models.pop(i)
            evicted += 1
            log.info("MiniMax H3 Director: evicted dead LoadedModel slot (%s)", name)
        except Exception:
            continue
    return evicted


def _loaded_slot_for(patcher):
    """把 ModelPatcher 映射到 ``current_loaded_models`` 里的那个 LoadedModel。

    必须返回**同一个对象**，不能传 ModelPatcher 充数 —— Comfy 的
    ``LoadedModel.__eq__`` 是 ``self.model is other.model``，拿 ModelPatcher
    去比会去读它的 ``.model`` 属性（那是内层 BaseModel），永远比不中。

    LoRA patch 会把模型变成**克隆**（对象不同但 ``clone_base_uuid`` 相同），
    所以除了 ``is`` 还要按 uuid 兜一层 —— 一条链上挂了 LoRA 时，
    ``current_loaded_models`` 里存的是克隆，只靠 ``is`` 是匹配不到的。
    """
    if patcher is None:
        return None
    try:
        import comfy.model_management as mm
    except Exception:
        return None
    base_uuid = getattr(patcher, "clone_base_uuid", None)
    for slot in list(getattr(mm, "current_loaded_models", ()) or ()):
        try:
            inner = getattr(slot, "model", None)
            if inner is patcher:
                return slot
            if base_uuid is not None and getattr(inner, "clone_base_uuid", None) == base_uuid:
                return slot
        except Exception:
            continue
    return None


def unload_except(keep_patchers) -> int:
    """卸载所有已加载模型，**除了** ``keep_patchers`` 里的那些。返回卸掉几个。

    为什么要有这个：``mm.unload_all_models()`` 在段与段之间会把**下一个阶段
    正要用的模型**也一起赶走。内存充裕时无所谓；内存紧张时，被赶走的权重会
    落到页面文件，下次装载要再从硬盘读回来 —— 而这一趟远比直接读模型文件慢。

    传空列表时行为与 ``unload_all_models()`` **完全一致**，所以是向后兼容的。
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return 0

    keep = []
    for patcher in keep_patchers or ():
        slot = _loaded_slot_for(patcher)
        if slot is not None and slot not in keep:
            keep.append(slot)

    try:
        devices = list(mm.get_all_torch_devices())
    except Exception:
        try:
            devices = [mm.get_torch_device()]
        except Exception:
            return 0

    freed = 0
    for device in devices:
        try:
            freed += len(mm.free_memory(1e30, device, keep_loaded=keep))
        except Exception as exc:
            # 选择性卸载失败不算致命：退回全卸，最多慢一点，不能让它把流程打断
            log.warning("selective unload failed on %s (%s); falling back to unload_all", device, exc)
            try:
                mm.unload_all_models()
            except Exception:
                pass
            return freed
    return freed



# ---------------------------------------------------------------- 阶段调度

#: 每个阶段**真正会用到**的模型角色。
#:
#: 这张表来自代码在各阶段的实际行为，与机器配置无关 —— 换任何模型组合都成立：
#:
#:   context_encode  文本编码器编提示词；视频 VAE 编参考图；音频 VAE 编参考音频
#:   sample          扩散模型采样；二采模型 / 放大网做 refine；VAE 出画中画预览
#:   decode          视频 VAE + 音频 VAE
#:
#: **表里没有的角色，就是该阶段可以安全卸掉的。** 至于卸了值不值（读盘 vs 腾内存），
#: 由 should_evict() 按物理内存自动判断，不在这里写死。
PHASE_MODEL_ROLES = {
    "context_encode": ("clip", "vae", "audio_vae"),
    "sample": ("model", "refine_model", "upscale_model", "vae"),
    "decode": ("vae", "audio_vae"),
}

#: 已加载模型的总量超过物理内存这个比例时，才值得卸载。
#: 低于它说明内存本来就够，卸载只会白白多读几次盘。
EVICT_RAM_RATIO = 0.65


def phase_keep_pool(phase, pool):
    """该阶段要保留的模型列表。``pool`` 是 {角色名: ModelPatcher}。"""
    return [pool[r] for r in PHASE_MODEL_ROLES.get(phase, ()) if pool.get(r) is not None]


def _loaded_model_bytes():
    """当前已加载模型的总体积（按 model_memory 算，含 offload 部分）。"""
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    total = 0
    for slot in list(getattr(mm, "current_loaded_models", ()) or ()):
        try:
            total += int(slot.model_memory())
        except Exception:
            continue
    return total


def _ram_total():
    try:
        import torch
        import comfy.model_management as mm
        return int(mm.get_total_memory(torch.device("cpu")))
    except Exception:
        return 0


def _patcher_bytes(patcher) -> int:
    try:
        return int(patcher.model_size())
    except Exception:
        return 0


def should_evict(pool=None) -> bool:
    """值不值得按阶段卸？内存宽裕时一个都不卸。

    ⚠ 判据是**整个工作流要用的模型总量**，不是"当前已加载了多少"。

    为什么不能用当前加载量判断：某个阶段结束时场上可能只剩下编码器和 VAE，
    看着很宽裕；但下一个阶段要装的生成模型还没算进去，一装就会溢出 ——
    用当前值判会得出"不用卸"，然后一装就爆。
    模型总量是工作流的固有属性，不随加载状态变化，所以它才是对的判据。

    拿不到内存数据时按保守来（卸）。
    """
    ram = _ram_total()
    if ram <= 0:
        return True
    total = 0
    for patcher in (pool or {}).values():
        if patcher is not None:
            total += _patcher_bytes(patcher)
    if total <= 0:
        total = _loaded_model_bytes()
    if total <= 0:
        return False
    return total > ram * EVICT_RAM_RATIO


def cleanup_segment_vram(
    *,
    enabled: bool = True,
    unload_models: bool = True,
    keep=(),
    adaptive: bool = True,
    pool=None,
) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache.

    ``keep`` 是**要保留在场**的模型（传 ModelPatcher 即可，内部会映射到
    LoadedModel）。典型用法是按阶段只保留该阶段用得上的模型，其余卸掉 ——
    这样相邻阶段所需的大模型不会同时挤在内存里。
    """
    if not enabled:
        return
    if unload_models and adaptive and not should_evict(pool):
        log.debug(
            "MiniMax H3 Director: skip segment cleanup (loaded models fit in RAM)"
        )
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        if unload_models:
            if keep:
                unload_except(keep)
            else:
                mm.unload_all_models()
            mm.cleanup_models()
        _evict_dead_loaded_models()
        gc.collect()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    if unload_models:
        kept = len(tuple(keep or ()))
        log.debug(
            "MiniMax H3 Director: segment VRAM cleanup (unloaded except %d kept, cache cleared)",
            kept,
        )
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")
