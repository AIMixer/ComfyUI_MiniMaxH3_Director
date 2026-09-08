"""In-sampling audio preview for MiniMax H3 (fl2va joint diffusion).

The per-step sampler callback receives ``x0`` (the predicted clean AV latent),
which is a NestedTensor whose last stream is the audio latent — the same stream
``VAEDecodeAudio`` decodes for the final clip. This module decodes that stream
with the official audio VAE and encodes a PCM16 WAV (base64) so the UI can offer
a manual "listen to the current step" preview, starting from the first step.

Design notes:
- There is no tiny audio decoder analogous to the video TAE; we must run the full
  fp32 audio VAE, so decoding is throttled (first/last step always, plus an
  adaptive stride in between) to avoid slowing sampling.
- Early-step audio is not yet converged and sounds noisy (the same way early video
  frames are blurry), but is enough to judge "any sound / voice vs music / energy
  contour" so a bad run can be aborted early.
- The caller MUST pass ``x0`` through ``inner_model.process_latent_out()`` before
  calling us: during sampling the audio stream is carried scaled by ``audio_scale``
  in model space and is only divided back to VAE space at the end of sampling
  (see comfy/samplers.py / comfy/model_base.py). Decoding the raw model-space
  stream yields audio that is ~audio_scale times too loud, clipped, and does not
  match the final clip. (The video TAE preview is unaffected — it consumes
  model-space latent directly.)
- Decoding goes through the same ``VAEDecodeAudio`` node path as the final clip so
  the AV NestedTensor audio stream is unbound correctly.
- Everything here is best-effort: any failure returns None and never breaks sampling.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.audio_preview")


# 自适应预览步距：首步与末步必发；中间步距随总步数缩放。
# 少步快速迭代（如 6 步）几乎每步都推，便于采样前期就试听、发现音频不对立即停掉重开；
# 长任务按约 15% 步距稀疏推，控制 fp32 音频 VAE 的反复解码开销。
def _preview_stride(total: int) -> int:
    return max(1, int(round(total * 0.15)))


def should_emit_audio_preview(step: int, total_steps: int) -> bool:
    """首步、末步必发；中间按自适应步距发射。

    早期步骤音频尚未收敛、偏噪（扩散本质，画面预览早期同样是糊的），但足以
    判断「有无声音 / 人声还是音乐 / 能量轮廓」，供少步快速迭代时及早止损。
    """
    try:
        total = max(1, int(total_steps))
        s = int(step)
        last = total - 1
        if s <= 0 or s >= last:
            return True
        return s % _preview_stride(total) == 0
    except Exception:
        return False


def _import_vaedecodeaudio_node():
    """成片同款节点类（VAEDecodeAudio.execute），与 executor_core._decode_av_latent
    走完全一致的解码路径。优先用它——它在各 ComfyUI 版本下都能正确 unbind AV
    NestedTensor 的音频 stream、跑 audio VAE 并做官方响度归一化。"""
    try:
        from comfy_extras.nodes_audio import VAEDecodeAudio

        return VAEDecodeAudio
    except ImportError:  # older ComfyUI layouts
        try:
            from comfy_extras.nodes_lt import VAEDecodeAudio  # type: ignore

            return VAEDecodeAudio
        except Exception:
            return None


def _import_vae_decode_audio():
    """旧版/兜底用的函数式入口。注意：不同 ComfyUI 版本下该函数对 AV NestedTensor
    的处理可能与节点类不一致，仅作为节点类不可用时的后备。"""
    try:
        from comfy_extras.nodes_audio import vae_decode_audio

        return vae_decode_audio
    except ImportError:  # older ComfyUI layouts
        try:
            from comfy_extras.nodes_lt import vae_decode_audio  # type: ignore

            return vae_decode_audio
        except Exception:
            return None


def decode_preview_audio_dict(x0: Any, audio_vae: Any) -> dict | None:
    """Decode the current-step AV latent ``x0`` to a ComfyUI AUDIO dict.

    ``x0`` 必须已经过 ``process_latent_out`` 除回 VAE 空间（见模块 docstring）。
    解码与成片同路径：VAEDecodeAudio 节点会 unbind NestedTensor 最后一个
    stream（音频），再 ``vae.decode`` + 官方响度归一化。
    """
    if audio_vae is None or x0 is None:
        return None
    audio = None
    node_cls = _import_vaedecodeaudio_node()
    if node_cls is not None:
        try:
            out = node_cls.execute(audio_vae, {"samples": x0})
            # 节点返回 (AUDIO dict,)；兼容直接返回 dict 的版本。
            audio = out[0] if isinstance(out, (tuple, list)) else out
        except Exception as exc:
            log.debug("VAEDecodeAudio.execute preview path failed: %s", exc)
    if not isinstance(audio, dict):
        fn = _import_vae_decode_audio()
        if fn is not None:
            try:
                audio = fn(audio_vae, {"samples": x0})
            except Exception as exc:
                log.debug("Audio preview decode skipped: %s", exc)
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    if not isinstance(waveform, torch.Tensor) or waveform.numel() <= 0:
        return None
    sr = int(audio.get("sample_rate") or 32000)
    return {"waveform": waveform, "sample_rate": sr}


def _waveform_to_wav_b64(audio: dict) -> tuple[str, int] | None:
    """Encode a ComfyUI AUDIO dict ([1,C,T] tensor) as PCM16 WAV base64."""
    try:
        wf = audio.get("waveform")
        sr = int(audio.get("sample_rate") or 32000)
        if not isinstance(wf, torch.Tensor) or wf.numel() <= 0:
            return None
        t = wf.detach().cpu().float()
        if t.ndim == 3:
            t = t[0]  # [C, T]
        if t.ndim != 2:
            return None
        # Keep at most stereo (H3 audio is mono/stereo).
        if t.shape[0] > 2:
            t = t[:2]
        ch = int(t.shape[0])
        n = int(t.shape[-1])
        if n <= 0 or ch <= 0:
            return None
        pcm = t.clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16)
        # [C, T] -> [T, C] interleaved -> little-endian PCM16 bytes
        arr = pcm.numpy().transpose(1, 0).reshape(-1).astype("<i2", copy=False)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(ch)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(arr.tobytes())
        return base64.b64encode(buf.getvalue()).decode("ascii"), sr
    except Exception as exc:
        log.debug("Audio preview WAV encode skipped: %s", exc)
        return None


def x0_to_audio_preview_b64(x0: Any, audio_vae: Any) -> tuple[str, int] | None:
    """Full path: current-step ``x0`` (VAE-space) -> (WAV base64, sample_rate)."""
    audio = decode_preview_audio_dict(x0, audio_vae)
    if audio is None:
        return None
    return _waveform_to_wav_b64(audio)
