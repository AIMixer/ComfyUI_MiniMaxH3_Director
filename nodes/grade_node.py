"""MiniMaxH3Grade：生成后的手工调色节点（v52，纯下游后期制作）。

不触碰采样/二采/缓存/版本号。输入导演台输出的 IMAGE（一采或二采成片均可），
面板内预览播放 + 逐段/全局手工参数 + 自动优化建议 + 撤销栈。
"""

from __future__ import annotations

import json
import os
import time

import torch

from ..director.grade_core import (
    apply_grade,
    auto_analyze,
    cache_grade_input,
    parse_timeline_segments,
    resolve_effective_segments,
    save_preview_video,
    write_mp4_imageio,
)

_CATEGORY = "MiniMaxH3"


class MiniMaxH3Grade:
    """生成后手工调色节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "导演台输出的图像（一采或二采成片均可）。"}),
                "grade_spec": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Internal — 调色参数（面板生成，随工作流保存）。",
                    },
                ),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "preview": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "写低分辨率预览 mp4（节点面板内播放）。"},
                ),
            },
            "optional": {
                "timeline_data": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "可选：导演台时间轴数据（用于段边界定位）。"},
                ),
                "audio": ("AUDIO", {"tooltip": "音频透传（不处理）。"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = False

    def execute(self, images, grade_spec, fps, preview, timeline_data="", audio=None, unique_id=None):
        spec: dict = {}
        try:
            parsed = json.loads(grade_spec or "{}")
            if isinstance(parsed, dict):
                spec = parsed
        except Exception:
            spec = {}
        spec.setdefault("timeline_data", timeline_data or "")
        # 段边界：以本次时间轴解析为准写回 spec，并通过 ui.segments 同步给面板。
        segs, _fps = parse_timeline_segments(timeline_data or "")
        if segs:
            spec["segments"] = segs

        ui: dict = {}
        t0 = time.perf_counter()
        out = apply_grade(images, spec)

        # 实际生效的分段（segment_bounds > 时间轴 > 整段）→ 同步给面板。
        # 若对不上（pin 相位对齐裁尾 / 旧画面 + 新时间轴），明确警告而不是静默退化。
        total = int(out.shape[0])
        eff_segs, fell_back = resolve_effective_segments(total, spec)
        if fell_back:
            ui["warning"] = (
                f"警告：提供的分段帧数与输入画面 {total} 帧不一致——"
                "导演台 pin 相位对齐会裁掉接缝尾部数帧，或时间轴刚改过还没重跑。"
                "本次按整段处理（无分段参数/接缝尖峰修复）。"
                "完整运行一次工作流后点「队列出片」，面板会自动同步真实分段。"
            )
        elif eff_segs and len(eff_segs) > 1:
            ui["segments"] = eff_segs

        req = spec.get("auto_requests")
        if isinstance(req, dict) and req.get("mode"):
            analysis = auto_analyze(images, spec)
            ui["auto_results"] = json.dumps(analysis, ensure_ascii=False)
        else:
            analysis = None

        if preview:
            # 预览文件 LRU 轮换：只保留最近 3 个（preview_<id>_0/1/2.mp4），
            # 新计算覆盖最早的，避免反复调色把硬盘堆满。
            slot = int(spec.get("preview_slot") or 0) % 3
            name = save_preview_video(
                out, fps or 24.0, f"minimax_grade_preview_{unique_id or 'x'}_{slot}.mp4"
            )
            if name:
                entry = {"filename": name, "type": "output", "format": "video/mp4", "subfolder": ""}
                ui["videos"] = [entry]
                # 旧前端用 animated 键播放视频；新前端识别 videos。
                ui["animated"] = [entry]
        # 队列出片时的"下一轮参考副本"：把调色后的全分辨率画面写进 input 目录，
        # 供二采/三采（导演台 passes）直接当参考视频引用。
        if bool(spec.get("save_ref_copy")):
            try:
                import folder_paths

                ref_name = f"minimax_grade_ref_{unique_id or 'x'}.mp4"
                ref_path = os.path.join(folder_paths.get_input_directory(), ref_name)
                if write_mp4_imageio(out, ref_path, fps or 24.0):
                    ui["ref_copy"] = {"path": ref_name, "full": ref_path}
                    log = __import__("logging").getLogger(
                        "ComfyUI-MiniMaxH3-Director.director.grade"
                    )
                    log.info("调色：下一轮参考副本已写入 input\\%s", ref_name)
            except Exception as exc:
                __import__("logging").getLogger(
                    "ComfyUI-MiniMaxH3-Director.director.grade"
                ).warning("调色：参考副本写入失败（%s）。", exc)
        elapsed = time.perf_counter() - t0
        # 缓存原始输入帧（fp16 覆盖写）——供实时调色路由绕过队列重算。
        cache_grade_input(images, unique_id)
        __import__("logging").getLogger(
            "ComfyUI-MiniMaxH3-Director.director.grade"
        ).info(
            "调色：完成 %d 帧（%.1fs）%s",
            int(out.shape[0]),
            elapsed,
            f"，自动分析：{analysis['notes']}" if analysis else "",
        )
        return (out, audio, ui)
