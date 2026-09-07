"""MiniMaxH3Grade 调色引擎（v52）：生成后的手工调色 + 自动优化建议 + 尖峰修复。

纯像素域后处理，位于生成管线之外（不碰采样/二采/缓存/版本号）：
- 手工参数两级：全片统一 + 每段覆盖（bright/contrast/sat/hue 叠加，soften 取大）。
- 自动优化（把建议值返回给前端写入滑块，不直接改画面）：
  1) 接缝漂移（全局/每段）：同内容相邻帧测量（亮度/对比度/饱和度/色相旋转）；
  2) 对比度均衡（全局/每段）：整段平均对比度 vs 前段（或段1），带可信区间
     0.80–1.25 双重验证（超范围判定为提示词作用，不给建议）。
- 接缝单帧尖峰修复：检测段边界单帧亮度离群并拉回局部趋势（手工无法修）。
"""

from __future__ import annotations

import json
import math
import os
import shutil
from typing import Any

import torch

from .h3_motion_context import (
    _gaussian_blur_batch,
    _rgb_to_yuv,
    _uv_affine,
    _yuv_to_rgb,
)

log = __import__("logging").getLogger("ComfyUI-MiniMaxH3-Director.director.grade")

# ── 常量 ─────────────────────────────────────────────────────────────────
GRADE_PARAM_KEYS = ("bright", "contrast", "sat", "hue", "soften")
GRADE_DEFAULTS = {"bright": 0.0, "contrast": 1.0, "sat": 1.0, "hue": 0.0, "soften": 0.0}
# 对比度均衡可信区间：超出判定为提示词/内容作用。
CONTRAST_CREDIBLE_MIN = 0.80
CONTRAST_CREDIBLE_MAX = 1.25
# 单帧尖峰阈值（Y 均值离群量）。
SPIKE_THRESHOLD = 0.02


def parse_timeline_segments(timeline_data: str) -> tuple[list[int], float]:
    """从 timeline_data 提取每段帧数与 fps（尽力而为）。"""
    segs: list[int] = []
    fps = 24.0
    try:
        data = json.loads(timeline_data or "{}") if isinstance(timeline_data, str) else (timeline_data or {})
        if isinstance(data, dict):
            fps = float(data.get("fps") or data.get("frameRate") or 24.0) or 24.0
            rows = None
            for key in ("segments", "shots", "r2vGroups", "fl2vGroups"):
                v = data.get(key)
                if isinstance(v, list) and v:
                    rows = v
                    break
            if rows is None and isinstance(data.get("output"), dict):
                out = data["output"]
                if isinstance(out.get("segments"), list):
                    rows = out["segments"]
            if rows:
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    fc = row.get("frameCount") or row.get("length") or row.get("frames")
                    if fc is None:
                        dur = row.get("durationSec")
                        if dur is not None:
                            fc = max(1, int(round(float(dur) * fps)))
                    if fc is not None:
                        segs.append(max(1, int(fc)))
    except Exception as exc:
        log.warning("grade: timeline_data parse failed (%s); 按整段处理。", exc)
    return segs, fps


def _merge_batches(images) -> torch.Tensor:
    """把 IMAGE 输入（单批或列表）合并为 [F,H,W,C] float。"""
    if isinstance(images, (list, tuple)):
        parts = [t for t in images if t is not None]
        if not parts:
            raise ValueError("grade: 空输入")
        return torch.cat([t.float() for t in parts], dim=0)
    return images.float()


def _resolve_boundaries(total_frames: int, segs: list[int]) -> list[int]:
    """边界列表 [b0, b1, ..., bn]（b0=0, bn=total）。segs 缺失或不匹配时退化为单段。"""
    if not segs or sum(segs) != total_frames:
        return [0, total_frames]
    out = [0]
    acc = 0
    for s in segs:
        acc += s
        out.append(acc)
    return out


def _safe_segments(spec: dict) -> list[int] | None:
    """spec["segments"] 的安全读取（防手工编辑的脏值）。"""
    raw = spec.get("segments")
    if not isinstance(raw, list):
        return None
    out = []
    try:
        for x in raw:
            out.append(max(1, int(x)))
    except Exception:
        return None
    return out or None


def resolve_effective_segments(total: int, spec: dict) -> tuple[list[int], bool]:
    """确定最终生效的段边界（每段帧数）。

    优先级：导演台侧磁盘副载（pin 裁切后真实帧数，同轮运行即可用）
    > spec.segment_bounds（前端从 report 同步）> spec.segments（时间轴）
    > 整段。返回 (segments, fell_back)：fell_back=True 表示提供了分段但
    帧数合计与输入不符（pin 相位对齐裁尾 / 旧缓存），已按整段处理。
    """
    # 1) 磁盘副载：与用户侧分段做一致性对照（段数相同且每段差 ≤5 帧视为
    #    同一时间轴的裁切差异；否则可能是换过时间轴但总帧数恰好相同）。
    try:
        from . import drift_stats as _ds

        data = _ds.read_grade_bounds()
        if isinstance(data, dict) and int(data.get("frames") or 0) == total:
            side = _safe_segments({"segments": data.get("segments")})
            if side and sum(side) == total:
                ok = True
                for key in ("segment_bounds", "segments"):
                    ref = _safe_segments({"segments": spec.get(key)})
                    if ref:
                        ok = len(ref) == len(side) and all(
                            abs(a - b) <= 5 for a, b in zip(ref, side)
                        )
                        break
                if ok:
                    return side, False
    except Exception:
        pass
    # 2) spec 内提供的分段。
    had = False
    for key in ("segment_bounds", "segments"):
        ss = _safe_segments({"segments": spec.get(key)})
        if ss:
            if sum(ss) == total:
                return ss, False
            had = True
    return [total], had


def _frame_params(spec: dict, boundaries: list[int]) -> list[dict]:
    """把两级参数展开成每段生效参数（bright/hue 相加，contrast/sat 相乘，soften 取大）。"""
    g = dict(GRADE_DEFAULTS)
    gspec = spec.get("global")
    if isinstance(gspec, dict):
        g.update(gspec)
    per = spec.get("per_segment")
    per = per if isinstance(per, list) else []
    n = len(boundaries) - 1
    out = []
    for i in range(n):
        o = dict(g)
        row = per[i] if i < len(per) and isinstance(per[i], dict) else {}
        for k in ("bright", "hue"):
            o[k] = float(g.get(k, 0.0) or 0.0) + float(row.get(k, 0.0) or 0.0)
        for k in ("contrast", "sat"):
            o[k] = float(g.get(k, 1.0) or 1.0) * float(row.get(k, 1.0) or 1.0)
        o["soften"] = max(float(g.get("soften", 0.0) or 0.0), float(row.get("soften", 0.0) or 0.0))
        out.append(o)
    return out


def _spike_deviations(f: torch.Tensor, boundaries: list[int], total: int) -> dict:
    """两遍法第一步：按帧均值检测接缝单帧尖峰。

    返回 {frame_idx: (dev, seg_idx)}。检测基于原始帧的帧均值；
    校正时按所在段对比度换算（对比度 c 会把偏差放大 c 倍，亮度偏移相互抵消）。
    """
    import bisect

    means: list[torch.Tensor] = []
    CHUNK = 64
    for s0 in range(0, total, CHUNK):
        y = _rgb_to_yuv(f[s0 : s0 + CHUNK])[..., 0]
        means.append(y.mean(dim=(1, 2)))
    ym = torch.cat(means)  # [total]
    devs: dict = {}
    for b in boundaries[1:-1]:
        for idx, prev_i, next_i in ((b, b - 1, b + 1), (b - 1, b - 2, b)):
            if prev_i < 0 or next_i >= total:
                continue
            y_prev = float(ym[prev_i].item())
            y_cur = float(ym[idx].item())
            y_next = float(ym[next_i].item())
            local = (y_prev + y_next) * 0.5
            dev = y_cur - local
            if abs(dev) > SPIKE_THRESHOLD and abs(y_prev - y_next) < SPIKE_THRESHOLD:
                seg_idx = bisect.bisect_right(boundaries, idx) - 1
                devs[idx] = (dev, max(0, seg_idx))
                log.info("调色：接缝单帧尖峰已修复 @%d（偏差 %+.3f）", idx, dev)
    return devs


def _apply_grade_chunk(
    chunk: torch.Tensor,
    s0: int,
    s1: int,
    boundaries: list[int],
    params: list[dict],
    hue_sin: list[float],
    hue_cos: list[float],
    devs: dict,
) -> torch.Tensor:
    """对 [s0, s1) 帧块应用逐段调色 + 尖峰修正，返回调色后 RGB 块。"""
    yuv = _rgb_to_yuv(chunk[..., :3])
    y = yuv[..., 0]
    uv = yuv[..., 1:]
    for i, p in enumerate(params):
        bs, be = boundaries[i], boundaries[i + 1]
        a0, a1 = max(bs, s0), min(be, s1)
        if a1 <= a0:
            continue
        sy = y[a0 - s0 : a1 - s0]
        suv = uv[a0 - s0 : a1 - s0]
        # 对比度：以 0.5 为中心缩放（不改均值）。
        c = float(p["contrast"])
        if abs(c - 1.0) > 1e-4:
            sy = (sy - 0.5) * c + 0.5
        # 亮度偏移。
        b = float(p["bright"])
        if abs(b) > 1e-4:
            sy = sy + b
        # 饱和度 + 色相旋转（U/V 平面）。
        s = float(p["sat"])
        if abs(s - 1.0) > 1e-4 or abs(p["hue"]) > 1e-2:
            a = s * hue_cos[i]
            d = s * hue_sin[i]
            u2 = suv[..., 0] * a - suv[..., 1] * d
            v2 = suv[..., 0] * d + suv[..., 1] * a
            suv = torch.stack([u2, v2], dim=-1)
        # 软化（luma-only 高斯去锐化）。
        w = float(p["soften"])
        if w > 1e-3:
            weight = min(0.85, w * 0.5)
            blur = _gaussian_blur_batch(sy.unsqueeze(-1))[..., 0]
            sy = sy * (1.0 - weight) + blur * weight
        y[a0 - s0 : a1 - s0] = sy
        uv[a0 - s0 : a1 - s0] = suv

    # 接缝单帧尖峰修正（两遍法第二步）：按段对比度换算偏差后拉回局部趋势。
    for idx, (dev, seg_idx) in devs.items():
        if s0 <= idx < s1:
            scale = float(params[seg_idx]["contrast"]) if seg_idx < len(params) else 1.0
            y[idx - s0] = y[idx - s0] - dev * scale

    out_yuv = torch.cat([y.unsqueeze(-1), uv], dim=-1)
    rgb = _yuv_to_rgb(out_yuv).clamp(0.0, 1.0)
    out = chunk.clone()
    out[..., :3] = rgb
    return out


def apply_grade(images, spec: dict) -> torch.Tensor:
    """按 spec 对手工参数做逐段调色 + 单帧尖峰修复。

    显存策略：二采成片（1376×768 × 数百帧 ≈ 数 GB）在显存里再复制一份
    必然 OOM——按 64 帧分块处理、结果直接落 CPU 输出（下游编码器同样可用），
    显存峰值只增加一个块的大小。
    """
    f = _merge_batches(images)
    total = int(f.shape[0])
    segs, _fell_back = resolve_effective_segments(total, spec)
    boundaries = _resolve_boundaries(total, segs)
    params = _frame_params(spec, boundaries)
    hue_sin = [math.sin(math.radians(p["hue"])) for p in params]
    hue_cos = [math.cos(math.radians(p["hue"])) for p in params]

    big = int(f.shape[1]) * int(f.shape[2]) * total > 200_000_000
    out_device = "cpu" if (f.is_cuda and big) else f.device
    out = torch.empty_like(f, device=out_device)

    devs = _spike_deviations(f, boundaries, total) if bool(spec.get("spike_fix", True)) else {}
    CHUNK = 64
    for s0 in range(0, total, CHUNK):
        s1 = min(s0 + CHUNK, total)
        out[s0:s1] = _apply_grade_chunk(
            f[s0:s1], s0, s1, boundaries, params, hue_sin, hue_cos, devs
        ).to(out_device)
    return out


def _seam_suggestion(head: torch.Tensor, tail: torch.Tensor) -> dict:
    """同内容相邻帧测量 → 手工参数建议（bright/contrast/sat/hue）。

    建议值的符号与 apply_grade 的滑块一致：作用到 head 段后与 tail 对齐。
    """
    yh = _rgb_to_yuv(head)[..., 0]
    yt = _rgb_to_yuv(tail)[..., 0]
    bright = float(yt.mean().item() - yh.mean().item())
    contrast = float(yt.std().item() / max(yh.std().item(), 1e-4))
    uvh = _rgb_to_yuv(head)[..., 1:].reshape(-1, 2)
    uvt = _rgb_to_yuv(tail)[..., 1:].reshape(-1, 2)
    sat = float(uvt.std(0).mean().item() / max(uvh.std(0).mean().item(), 1e-4))
    A, _o = _uv_affine(uvh, uvt)
    # A 把 head 映射到 tail（tail = A·head，旋转角 θ）；滑块 hue=+θ 即此变换，
    # 与后端 tone_compensate 应用方向一致（uv2 = uv @ A.T）。
    hue = math.degrees(math.atan2(float(A[1, 0]) - float(A[0, 1]), float(A[0, 0]) + float(A[1, 1])))
    return {
        "bright": max(-0.10, min(0.10, bright)),
        "contrast": max(0.80, min(1.25, contrast)),
        "sat": max(0.80, min(1.25, sat)),
        "hue": max(-8.0, min(8.0, hue)),
    }


def auto_analyze(images, spec: dict) -> dict:
    """自动优化分析。返回 {"suggestions": ..., "notes": [...]}（建议值由前端写入手工参数）。"""
    f = _merge_batches(images)
    total = int(f.shape[0])
    segs, _fell_back = resolve_effective_segments(total, spec)
    boundaries = _resolve_boundaries(total, segs)
    n = len(boundaries) - 1
    req = spec.get("auto_requests") or {}
    mode = req.get("mode", "")
    target = int(req.get("target", -1))  # -1 = 全局
    notes: list[str] = []
    per_seg = [dict(GRADE_DEFAULTS) for _ in range(n)]
    global_sugg = dict(GRADE_DEFAULTS)
    SEAM_KEYS = ("bright", "contrast", "sat", "hue")
    result = {"suggestions": {"global": global_sugg, "per_segment": per_seg}, "notes": notes}

    if mode == "seam":
        # 接缝漂移：各段入口接缝的实测建议。
        suggs = []  # (k, 建议) — 与段号对应，避免缺帧时错位。
        for k in range(1, n):
            b = boundaries[k]
            if b - 2 < 0 or b + 2 > total:
                continue
            suggs.append((k, _seam_suggestion(f[b : b + 2], f[b - 2 : b])))
        if not suggs:
            notes.append("无可用接缝（段数不足）")
        elif target < 0:
            for key in SEAM_KEYS:
                global_sugg[key] = sum(s[key] for _k, s in suggs) / len(suggs)
            notes.append(f"全局接缝漂移建议 = {len(suggs)} 个接缝均值")
        else:
            hit = next((s for k, s in suggs if k == target), None)
            if hit is not None:
                per_seg[target] = dict(GRADE_DEFAULTS)
                for key in SEAM_KEYS:
                    per_seg[target][key] = hit[key]
                notes.append(f"段 {target + 1} 接缝漂移建议已生成")
            else:
                notes.append("该段无入口接缝可分析（段 1 为参考段）")
    elif mode == "contrast":
        # 对比度均衡（整段平均对比度，可信区间 0.80–1.25 + 接缝交叉验证）。
        levels = []
        for k in range(n):
            s0, s1 = boundaries[k], boundaries[k + 1]
            y = 0.299 * f[s0:s1, ..., 0] + 0.587 * f[s0:s1, ..., 1] + 0.114 * f[s0:s1, ..., 2]
            levels.append(float(y.std(dim=(1, 2)).mean().item()) or 1e-4)
        ref = levels[0]
        for k in range(1, n):
            if target >= 0 and k != target:
                continue
            ratio = ref / levels[k] if target < 0 else levels[k - 1] / levels[k]
            if not (CONTRAST_CREDIBLE_MIN <= ratio <= CONTRAST_CREDIBLE_MAX):
                notes.append(
                    f"段 {k + 1}：对比度比 {ratio:.2f} 超出可信区间 0.80–1.25"
                    "（判定为提示词/内容作用），跳过"
                )
                continue
            # 接缝交叉验证：同内容相邻帧的对比度漂移应与整段比同向同量级。
            b = boundaries[k]
            if b - 2 >= 0 and b + 2 <= total:
                seam_ratio = float(
                    (0.299 * f[b : b + 2, ..., 0] + 0.587 * f[b : b + 2, ..., 1] + 0.114 * f[b : b + 2, ..., 2])
                    .std()
                    .item()
                    / max(
                        (0.299 * f[b - 2 : b, ..., 0] + 0.587 * f[b - 2 : b, ..., 1] + 0.114 * f[b - 2 : b, ..., 2])
                        .std()
                        .item(),
                        1e-4,
                    )
                )
                if abs(ratio - seam_ratio) > 2.5 * max(abs(seam_ratio - 1.0), 0.02):
                    notes.append(
                        f"段 {k + 1}：整段对比度比 {ratio:.2f} 与接缝级 {seam_ratio:.2f} "
                        "严重不符（>2.5×），判定为内容作用，跳过"
                    )
                    continue
            per_seg[k]["contrast"] = round(max(0.80, min(1.25, ratio)), 3)
            notes.append(f"段 {k + 1} 对比度均衡建议已生成")
    elif mode == "drift_report":
        # 各段间漂移参考表：原始实测值（不限幅），与后端日志"色调角"同口径。
        rows = []
        for k in range(1, n):
            b = boundaries[k]
            if b - 2 < 0 or b + 2 > total:
                continue
            head = f[b : b + 2]
            tail = f[b - 2 : b]
            yh = _rgb_to_yuv(head)[..., 0]
            yt = _rgb_to_yuv(tail)[..., 0]
            bright = float(yt.mean().item() - yh.mean().item())
            contrast = float(yt.std().item() / max(yh.std().item(), 1e-4))
            uvh = _rgb_to_yuv(head)[..., 1:].reshape(-1, 2)
            uvt = _rgb_to_yuv(tail)[..., 1:].reshape(-1, 2)
            sat = float(uvt.std(0).mean().item() / max(uvh.std(0).mean().item(), 1e-4))
            hue = math.degrees(
                math.atan2(float(uvh.mean(0)[1]), float(uvh.mean(0)[0]))
                - math.atan2(float(uvt.mean(0)[1]), float(uvt.mean(0)[0]))
            )
            rows.append(
                {
                    "seg": k + 1,
                    "bright": round(bright * 100.0, 2),
                    "contrast": round(contrast, 3),
                    "sat": round(sat, 3),
                    "hue": round(hue, 1),
                    "hue_flag": abs(hue) >= 15.0,
                }
            )
        result["drift_report"] = rows
        if not rows:
            notes.append("无可用接缝（段数不足）")
    else:
        notes.append("未识别的自动优化模式")

    return result


def _grade_cache_dir() -> str:
    """调色台持久文件目录 = ComfyUI output\\temp。

    插件目录在部署新版本时会被整体替换；ComfyUI 临时目录可能被衍生版
    清理。output\\temp 与插件更新无关、重启保留，且用户可见可控。
    """
    try:
        import folder_paths

        base = os.path.join(folder_paths.get_output_directory(), "temp")
        os.makedirs(base, exist_ok=True)
        return base
    except Exception:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _grade_legacy_temp_dir() -> str | None:
    """旧位置（ComfyUI temp 目录）——仅用于迁移。"""
    try:
        import folder_paths

        return folder_paths.get_temp_directory()
    except Exception:
        return None


def _grade_input_cache_path(unique_id: str) -> str:
    return os.path.join(_grade_cache_dir(), f"minimax_grade_input_{unique_id or 'x'}.pt")


# ── 参数库（保存/读取/删除调色参数预设，命名 = 日期 + 当天次数） ─────
_PRESETS_FILE = "minimax_grade_presets.json"


def _presets_path() -> str:
    return os.path.join(_grade_cache_dir(), _PRESETS_FILE)


def _migrate_from_legacy(path: str) -> bool:
    """老位置（ComfyUI temp）有文件而新位置没有时，搬到新位置。"""
    try:
        legacy_dir = _grade_legacy_temp_dir()
        if not legacy_dir:
            return False
        legacy = os.path.join(legacy_dir, os.path.basename(path))
        if os.path.exists(legacy) and not os.path.exists(path):
            shutil.move(legacy, path)
            log.info("调色：已从旧位置迁移 %s → output\\temp", os.path.basename(path))
            return True
    except Exception:
        pass
    return False


def _load_presets() -> dict:
    _migrate_from_legacy(_presets_path())
    try:
        with open(_presets_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("presets"), list):
            return data
    except Exception:
        pass
    return {"presets": []}


def _write_presets(data: dict) -> None:
    try:
        with open(_presets_path(), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
    except Exception as exc:
        log.info("调色：参数库写盘失败（%s）。", exc)


def list_grade_presets() -> list[dict]:
    """全部已保存参数（含 params）。"""
    return [dict(p) for p in _load_presets().get("presets", [])]


def save_grade_preset(params: dict) -> dict:
    """保存一组调色参数；命名 = 日期 + 当天保存次数（YYYYMMDD_NN）。"""
    import datetime as _dt

    data = _load_presets()
    today = _dt.date.today().strftime("%Y%m%d")
    n = 1
    for p in data["presets"]:
        nm = str(p.get("name") or "")
        if nm.startswith(today + "_"):
            try:
                n = max(n, int(nm.rsplit("_", 1)[1]) + 1)
            except Exception:
                pass
    name = f"{today}_{n:02d}"
    entry = {
        "name": name,
        "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "global": dict(params.get("global") or {}),
            "per_segment": [dict(r or {}) for r in (params.get("per_segment") or [])],
            "spike_fix": bool(params.get("spike_fix", True)),
        },
    }
    data["presets"].append(entry)
    _write_presets(data)
    return entry


def delete_grade_preset(name: str) -> bool:
    """删除指定名称的参数预设。"""
    data = _load_presets()
    before = len(data["presets"])
    data["presets"] = [p for p in data["presets"] if str(p.get("name") or "") != str(name)]
    if len(data["presets"]) != before:
        _write_presets(data)
        return True
    return False


def cache_grade_input(images, unique_id: str) -> None:
    """把调色台最近一次真实执行的原始输入帧落盘（fp16，覆盖写）。

    供实时调色路由在不走 ComfyUI 队列的情况下重算（ComfyUI-XS 等衍生版
    不支持部分 prompt 重执行）。写失败静默：实时调色不可用但不影响主流程。
    """
    try:
        f = _merge_batches(images)
        path = _grade_input_cache_path(unique_id)
        torch.save(f.half().cpu(), path)
        log.info(
            "调色：输入帧已缓存（%.1f MB）%s",
            os.path.getsize(path) / 1048576.0,
            os.path.basename(path),
        )
    except Exception as exc:
        log.info("调色：输入帧缓存失败（%s），实时调色将不可用。", exc)


def load_grade_input(unique_id: str) -> torch.Tensor | None:
    """读回调色台缓存的原始输入帧；不存在/损坏返回 None。"""
    path = _grade_input_cache_path(unique_id)
    _migrate_from_legacy(path)
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True).float()
    except Exception as exc:
        log.warning("调色：输入帧缓存读取失败（%s）。", exc)
        return None


def _even(n: int) -> int:
    """yuv420p 要求偶数边长。"""
    return max(2, int(n)) & ~1


def _write_preview_native(f: torch.Tensor, name_no_ext: str, fps: float) -> bool:
    """官方 ComfyUI 路径（comfy_extras.nodes_video.save_video）。失败/不存在返回 False。"""
    try:
        from comfy_extras.nodes_video import save_video
    except Exception:
        return False
    try:
        save_video(f.cpu(), name_no_ext, fps, codec="libx264", pix_fmt="yuv420p", crf=19)
        return True
    except Exception as exc:
        log.info("调色：官方 save_video 不可用（%s），改用 imageio 直写。", exc)
        return False


def _write_preview_imageio(f: torch.Tensor, path: str, fps: float) -> bool:
    """imageio + imageio-ffmpeg 直写 mp4（VideoHelperSuite 环境必有；绕开
    ComfyUI-XS 等衍生版缺 save_video 的问题）。"""
    try:
        import imageio
    except Exception as exc:
        log.info("调色：imageio 不可用（%s）。", exc)
        return False
    try:
        arr = (f.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)[..., :3]
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        try:
            for i in range(int(arr.shape[0])):
                writer.append_data(arr[i].numpy())
        finally:
            writer.close()
        return True
    except Exception as exc:
        log.warning("调色：imageio 预览写入失败（%s）。", exc)
        return False


def write_mp4_imageio(images, path: str, fps: float) -> bool:
    """全分辨率写 mp4（imageio 直写，可写任意目录）。

    供两类用途：面板预览（调用方先降采样）与"下一轮二采/三采参考副本"
    （写 input 目录、全分辨率）。
    """
    f = _merge_batches(images)
    return _write_preview_imageio(f.cpu(), path, fps)


def save_preview_video(images, fps: float, filename: str) -> str | None:
    """写低分辨率预览 mp4 到输出目录（失败返回 None，不影响主输出）。

    写入链：官方 save_video → imageio/ffmpeg。任一成功即返回文件名。
    """
    try:
        f = _merge_batches(images)
        # 预览降采样：最长边 640（偶数对齐，yuv420p 需要）。
        h, w = int(f.shape[1]), int(f.shape[2])
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            from comfy.utils import common_upscale

            f = common_upscale(
                f.permute(0, 3, 1, 2),
                _even(w * scale),
                _even(h * scale),
                "lanczos",
                "disabled",
            ).permute(0, 2, 3, 1)
        base = str(filename).removesuffix(".mp4")
        fps = fps or 24.0
        if _write_preview_native(f, base, fps):
            log.info("调色：预览视频已写入（官方 save_video）%s", os.path.basename(base + ".mp4"))
            return os.path.basename(base + ".mp4")
        import folder_paths

        out_dir = folder_paths.get_output_directory()
        path = os.path.join(out_dir, os.path.basename(base + ".mp4"))
        if _write_preview_imageio(f, path, fps):
            log.info("调色：预览视频已写入（imageio）%s", os.path.basename(base + ".mp4"))
            return os.path.basename(base + ".mp4")
        log.warning("调色：预览视频两种写入方式均失败（预览不可用，不影响主输出）。")
        return None
    except Exception as exc:
        log.warning("调色：预览视频写入失败（%s）；不影响主输出。", exc)
        return None
