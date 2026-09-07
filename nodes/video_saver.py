"""MiniMaxH3VideoSaver — 流式视频落盘节点（内置 SaveVideo 的兜底替代）。

多段长片/高分辨率成片在 ComfyUI-XS 内置 SaveVideo 的 PyAV 音频混流阶段
会因系统内存耗尽抛 ``av.error.MemoryError: [Errno 12] Cannot allocate
memory``。本节点从机制上绕开该路径：

- 视频：分块转 uint8 后逐块喂给 imageio-ffmpeg（libx264/yuv420p），
  随写随弃，不复制全片 float32；
- 音频：torch 线性重采样到 44100 Hz 后直接写 int16 WAV，再用
  imageio-ffmpeg 自带的 ffmpeg 做流式混流（``-c:v copy``），
  完全绕开 PyAV 的大缓冲重采样；
- 无音频（mute/空轨）时输出纯视频 mp4，不生成静音轨。
"""

from __future__ import annotations

import logging
import os
import subprocess
import wave

import torch

import folder_paths

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.video_saver")

VIDEO_CHUNK_FRAMES = 64
MUX_SAMPLE_RATE = 44100


class MiniMaxH3VideoSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01},
                ),
                "filename_prefix": ("STRING", {"default": "MiniMaxH3"}),
                "crf": ("INT", {"default": 18, "min": 10, "max": 30}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_video"
    CATEGORY = "MiniMaxH3"

    @staticmethod
    def _as_batch(images) -> torch.Tensor:
        if isinstance(images, (list, tuple)):
            parts = [t for t in images if t is not None]
            if not parts:
                raise ValueError("MiniMaxH3VideoSaver: 空视频输入")
            return torch.cat(parts, dim=0)
        return images

    @classmethod
    def _resample_audio(cls, audio: dict) -> torch.Tensor | None:
        wf = (audio or {}).get("waveform")
        if not isinstance(wf, torch.Tensor) or wf.numel() <= 0:
            return None
        sr = int((audio or {}).get("sample_rate") or 32000)
        if sr <= 0:
            sr = 32000
        w = wf.float()
        if w.ndim == 1:
            w = w.unsqueeze(0)
        mono = w.mean(dim=0, keepdim=True)
        if sr != MUX_SAMPLE_RATE:
            target = max(1, int(round(mono.shape[-1] * MUX_SAMPLE_RATE / float(sr))))
            mono = torch.nn.functional.interpolate(
                mono.unsqueeze(0), size=target, mode="linear", align_corners=False
            )[0]
        return mono.clamp(-1.0, 1.0)

    @classmethod
    def _write_wav(cls, samples: torch.Tensor, path: str) -> bool:
        try:
            pcm = (
                (samples[0] * 32767.0).round().clamp(-32768, 32767)
                .to(torch.int16).numpy()
            )
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(MUX_SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())
            return True
        except Exception as exc:
            log.warning("MiniMaxH3VideoSaver: WAV 写入失败（%s），输出纯视频。", exc)
            return False

    @classmethod
    def _mux_audio(cls, video_path: str, wav_path: str) -> bool:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            log.warning("MiniMaxH3VideoSaver: imageio_ffmpeg 不可用（%s），跳过混流。", exc)
            return False
        final = video_path + ".mux.mp4"
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path, "-i", wav_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", final,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
            os.replace(final, video_path)
            return True
        except Exception as exc:
            log.warning("MiniMaxH3VideoSaver: 音频混流失败（%s），保留纯视频。", exc)
            return False

    def save_video(
        self,
        images,
        fps,
        filename_prefix,
        crf,
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        del prompt, extra_pnginfo
        import imageio

        batch = self._as_batch(images).cpu()
        total = int(batch.shape[0])
        fps = float(fps or 24.0) or 24.0

        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(), 1, 1
            )
        )
        out_name = f"{filename}_{counter:05d}_.mp4"
        video_path = os.path.join(full_output_folder, out_name)

        writer = imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_params=["-crf", str(int(crf)), "-preset", "medium"],
        )
        try:
            for start in range(0, total, VIDEO_CHUNK_FRAMES):
                chunk = batch[start : start + VIDEO_CHUNK_FRAMES]
                if chunk.dtype != torch.uint8:
                    chunk = (
                        chunk.float().clamp(0.0, 1.0).mul(255.0)
                        .round().clamp(0, 255).to(torch.uint8)
                    )
                arr = chunk[..., :3].numpy()
                del chunk
                writer.append_data(arr)
                del arr
        finally:
            writer.close()
        del batch

        samples = self._resample_audio(audio)
        if samples is not None:
            wav_path = video_path + ".audio.wav"
            if self._write_wav(samples, wav_path):
                self._mux_audio(video_path, wav_path)
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except OSError:
                pass

        log.info(
            "MiniMaxH3VideoSaver: 已写入 %s（%d 帧 @%.2ffps, crf=%d）",
            out_name,
            total,
            fps,
            int(crf),
        )
        return {
            "ui": {
                "video": [
                    {"filename": out_name, "subfolder": subfolder, "type": "output"}
                ]
            }
        }


NODE_CLASS_MAPPINGS = {"MiniMaxH3VideoSaver": MiniMaxH3VideoSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3VideoSaver": "MiniMax H3 Video Saver"}
