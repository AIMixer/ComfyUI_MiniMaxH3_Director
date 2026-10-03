"""Incremental per-segment MP4 export for「分段导出」runs.

Best-effort: encode failures must never abort generation. Each run uses a
timestamp folder: ``output/minimax_seg_export/<YYYYMMDD_HHMMSS>/``.

Files:
  ``seg_XXXX.mp4`` — final clip (last refine pass / no Refine; FaceRefine stitch if wired)
  ``seg_XXXX_pre.mp4`` — first pass (一采), only when Refine ran
  ``seg_XXXX_pN.mp4`` — refine pass N (分段导出且次数>1)
  ``seg_XXXX_facepre.mp4`` — before FaceRefine stitch, only when「输出修脸前」is on
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import folder_paths
import torch

from ..lib.task_prompts import resolve_task_key
from .audio_export import prepare_segment_audio_for_file_export
from .plan import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.mp4_export")

VIDEO_EXPORT_TASKS = frozenset({"t2v", "i2v", "r2v", "fl2v", "v2v", "rv2v"})


def new_segment_mp4_run_dir(plan: DirectorPlan) -> Path | None:
    """Create ``minimax_seg_export/<YYYYMMDD_HHMMSS>/`` for one Director execute.

    Returns None when not in segments mode or the output dir is unavailable.
    """
    if getattr(plan, "export_mode", "all") != "segments":
        return None
    try:
        base = Path(folder_paths.get_output_directory()) / "minimax_seg_export"
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = base / stamp
        if root.exists():
            # Same-second collision (rare): append a short suffix.
            for i in range(1, 1000):
                candidate = base / f"{stamp}_{i:03d}"
                if not candidate.exists():
                    root = candidate
                    break
        root.mkdir(parents=True, exist_ok=False)
        log.info("MiniMax H3 Director segment mp4 run dir: %s", root)
        return root
    except OSError as exc:
        log.warning("Segment mp4 export dir unavailable (%s); skipped.", exc)
        return None


def _safe_mp4_suffix(suffix: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(suffix or ""))


def segment_mp4_path(run_dir: Path, seg: SegmentPlan, *, suffix: str = "") -> Path:
    tag = f"_{_safe_mp4_suffix(suffix)}" if _safe_mp4_suffix(suffix) else ""
    return Path(run_dir) / f"seg_{int(seg.index):04d}{tag}.mp4"


def mp4_export_kind(path: str | None) -> str:
    name = Path(str(path or "")).name
    if name.endswith("_facepre.mp4"):
        return "修脸前 mp4"
    if name.endswith("_pre.mp4"):
        return "一采 mp4"
    m = re.search(r"_p(\d+)\.mp4$", name)
    if m:
        return f"第{m.group(1)}轮精修 mp4"
    return "mp4"


def _suffix_log_label(suffix: str) -> str:
    tag = str(suffix or "")
    if tag == "pre":
        return "first-pass "
    if tag == "facepre":
        return "pre-face "
    return ""


def _pre_frames_distinct(pre_frames, frames) -> bool:
    if pre_frames is None or frames is None:
        return False
    if pre_frames is frames:
        return False
    if not isinstance(pre_frames, torch.Tensor) or pre_frames.ndim != 4:
        return False
    return int(pre_frames.shape[0]) > 0


def maybe_export_segment_mp4(
    run_dir: Path | None,
    plan: DirectorPlan,
    seg: SegmentPlan,
    frames: torch.Tensor,
    audio_dict: dict[str, Any] | None = None,
    *,
    suffix: str = "",
) -> str | None:
    """Write one segment mp4 into ``run_dir``. Never raises.

    ``suffix="pre"`` writes the first-pass clip (``seg_XXXX_pre.mp4``).
    ``suffix="facepre"`` writes the clip before FaceRefine stitch.
    ``suffix="p2"`` writes refine pass 2 (``seg_XXXX_p2.mp4``).

    Returns the absolute path string on success, otherwise None.
    """
    if run_dir is None or getattr(plan, "export_mode", "all") != "segments":
        return None
    task = str(getattr(seg, "task_key", "") or getattr(plan, "global_task_key", "") or "")
    if task not in VIDEO_EXPORT_TASKS:
        return None
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or int(frames.shape[0]) <= 0:
        return None

    dest = segment_mp4_path(run_dir, seg, suffix=suffix)

    try:
        from ..lib.video_export import write_frames_to_mp4

        audio = prepare_segment_audio_for_file_export(
            plan,
            seg,
            audio_dict=audio_dict,
            frame_count=int(frames.shape[0]),
        )
        path = write_frames_to_mp4(
            dest,
            frames.detach().cpu().float(),
            fps=float(getattr(plan, "frame_rate", 24) or 24),
            audio=audio,
        )
        log.info(
            "MiniMax H3 Director segment #%d %smp4 saved: %s",
            int(seg.index) + 1,
            _suffix_log_label(suffix),
            path,
        )
        # 仅 final 片段（suffix="")登记到任务表；pre/facepre/pN 是中间产物，不进表。
        if suffix == "":
            try:
                register_segment_clip(plan, seg, str(path), int(frames.shape[0]))
            except Exception:
                pass
        return str(path)
    except Exception as exc:
        log.warning(
            "Segment #%d %smp4 export failed (generation continues): %s",
            int(seg.index) + 1,
            _suffix_log_label(suffix),
            exc,
        )
        return None


def maybe_export_segment_mp4s(
    run_dir: Path | None,
    plan: DirectorPlan,
    seg: SegmentPlan,
    frames: torch.Tensor,
    audio_dict: dict[str, Any] | None = None,
    *,
    pre_frames: torch.Tensor | None = None,
    pre_face_frames: torch.Tensor | None = None,
) -> list[str]:
    """Write final clip, plus first-pass / pre-face when those tensors differ."""
    paths: list[str] = []
    final_path = maybe_export_segment_mp4(
        run_dir, plan, seg, frames, audio_dict,
    )
    if final_path:
        paths.append(final_path)
    if _pre_frames_distinct(pre_frames, frames):
        pre_path = maybe_export_segment_mp4(
            run_dir, plan, seg, pre_frames, audio_dict, suffix="pre",
        )
        if pre_path:
            paths.append(pre_path)
    if _pre_frames_distinct(pre_face_frames, frames):
        face_path = maybe_export_segment_mp4(
            run_dir, plan, seg, pre_face_frames, audio_dict, suffix="facepre",
        )
        if face_path:
            paths.append(face_path)
    return paths


def copy_segment_mp4_suffix(
    run_dir: Path | None,
    plan: DirectorPlan,
    seg: SegmentPlan,
    *,
    dest_suffix: str,
) -> str | None:
    """Copy ``seg_XXXX.mp4`` to ``seg_XXXX_<suffix>.mp4``. Never raises."""
    if run_dir is None or getattr(plan, "export_mode", "all") != "segments":
        return None
    tag = _safe_mp4_suffix(dest_suffix)
    if not tag:
        return None
    src = segment_mp4_path(run_dir, seg)
    dest = segment_mp4_path(run_dir, seg, suffix=tag)
    try:
        if not src.is_file():
            return None
        shutil.copy2(src, dest)
        log.info(
            "MiniMax H3 Director segment #%d copied %s → %s",
            int(seg.index) + 1,
            src.name,
            dest.name,
        )
        return str(dest)
    except Exception as exc:
        log.warning(
            "Segment #%d copy to %s failed: %s",
            int(seg.index) + 1,
            dest.name,
            exc,
        )
        return None


def _filter_concat_cmd(
    ffmpeg: str,
    clips: list[str],
    probes: list[tuple],
    dest: Path,
) -> list[str] | None:
    """用 concat **滤镜**（而非 demuxer）拼片 —— 输入参数不齐时唯一能保证时长的路径。

    每一路先 ``scale``+``pad``+``fps`` 统一到同一画布、``aresample`` 统一采样率，再
    ``concat`` 合成：逐路时长严格相加等于成片时长。直接把这些输入喂给 concat demuxer
    时，**采样率混用会让音轨时基算错**（实测被拉成 1.378 倍，88.2s 变 121.6s）。

    画布取**面积最大**的那一路（不把高分辨率素材降级），帧率/采样率取出现次数最多的。
    返回完整命令；探测不到视频流（无从定画布）返回 ``None``，由调用方退回 demuxer 路径。
    """
    vkeys = [pk[0] for pk in probes if pk and pk[0]]
    if not vkeys:
        return None
    w, h = max(((int(k[1]), int(k[2])) for k in vkeys), key=lambda wh: wh[0] * wh[1])
    fps_list = [float(k[4]) for k in vkeys if k[4]]
    fps = max(set(fps_list), key=fps_list.count) if fps_list else 24.0
    rates = [int(pk[1][1]) for pk in probes if pk and pk[1] and pk[1][1] > 0]
    rate = max(set(rates), key=rates.count) if rates else 48000
    parts: list[str] = []
    refs: list[str] = []
    for i in range(len(clips)):
        parts.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:g},format=yuv420p[v{i}]"
        )
        parts.append(
            f"[{i}:a]aresample={rate},aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
        )
        refs.append(f"[v{i}][a{i}]")
    graph = ";".join(parts) + ";" + "".join(refs) + (
        f"concat=n={len(clips)}:v=1:a=1[outv][outa]"
    )
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
    for p in clips:
        cmd += ["-i", str(p)]
    cmd += [
        "-filter_complex", graph,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(dest),
    ]
    return cmd


def _with_silent_audio(
    ffmpeg: str, clips: list[str], probes: list[tuple]
) -> tuple[list[str], list[Path]] | None:
    """给「完全没有音轨」的片段各补一条静音音轨 → ``(新片段列表, 需清理的临时文件)``。

    为什么必须补：concat 的两种实现都要求**每一路都有同构的流**。
      * concat **滤镜**：滤镜图对每路写 ``[i:a]aresample=...``，无音轨的那一路会让
        ffmpeg 直接报 ``Error binding filtergraph inputs/outputs: Invalid argument``
        （本机 ffmpeg 7.1 实测复现），整条时间轴拼不出来 —— 预览空白、也拿不到整片；
      * concat **demuxer**：``-c:a aac`` 在输入流不齐时同样会失败。

    视频流 ``-c:v copy`` 原样搬运，只新增一条静音音轨 —— **零重编码**。
    采样率取既有片段里出现最多的那个（都没有就退回 :data:`_PLACEHOLDER_AUDIO_RATE`），
    补齐后 ``uniform_audio`` 判定为真，能走最省钱的 demuxer + copy 路。

    任一路补齐失败就整体放弃（返回 ``None``），由调用方退回原有兜底逻辑。
    """
    rates = [int(pk[1][1]) for pk in probes if pk and pk[1] and pk[1][1] > 0]
    rate = max(set(rates), key=rates.count) if rates else _PLACEHOLDER_AUDIO_RATE
    out: list[str] = []
    temps: list[Path] = []

    def _drop_temp(p: Path) -> None:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    try:
        base = Path(folder_paths.get_temp_directory()) / "mmx_concat_fixup"
        base.mkdir(parents=True, exist_ok=True)
        for i, (p, pk) in enumerate(zip(clips, probes)):
            if pk and pk[1] is not None:
                out.append(p)
                continue
            tmp = base / f"{Path(p).stem}.{os.getpid()}.a{i}.mp4"
            cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", str(p),
                "-f", "lavfi", "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={int(rate)}",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart", str(tmp),
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=600)
            if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
                err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
                log.warning(
                    "Silent-audio fixup failed for %s [%d]: %s",
                    p, i, err[-200:] or f"code={proc.returncode}",
                )
                for t in temps:
                    _drop_temp(t)
                _drop_temp(tmp)
                return None
            out.append(str(tmp))
            temps.append(tmp)
        return out, temps
    except Exception as exc:
        log.warning("Silent-audio fixup skipped: %s", exc)
        for t in temps:
            _drop_temp(t)
        return None


def _concat_clips_to(
    paths: list[str],
    dest: Path,
    *,
    label: str = "concat",
    timeout: int = 3600,
) -> bool:
    """把若干 mp4 按给定顺序 ffmpeg concat 成 ``dest``（普通 mp4 + faststart，可随机 seek）。

    三条路径：

    * 视频参数一致 → concat demuxer + ``-c:v copy``（近乎零开销）；
    * 视频参数不一致 → concat 滤镜全重编码（分辨率/编码不同也不至于拼出坏文件）；
    * **音频采样率不一致** → 同样强制走 concat 滤镜。concat demuxer 在这种输入下会把
      音轨时基算错（实测 MiniMax H3 真实段 32 kHz + 占位片 44.1 kHz → 音轨被拉成
      1.378 倍，成片 121.6s 而视频只有 88.2s），且 ``-ar`` / ``aresample`` 都救不回来。

    * 先写 ``.tmp.mp4`` 再 ``os.replace`` 原子发布，中途失败不会留下半个产物；
    * concat 清单与临时文件在 finally 里清理（Windows 上用 ``/`` 分隔，反斜杠会被当转义）。

    出片预览文件与「分段导出」的每段 mp4 都走这里，两处逻辑必须同源，
    否则又会出现「预览是跨运行整条时间轴、导出却只有本次几段」这类分叉 bug。

    任何失败返回 False（best-effort，不抛）。
    """
    try:
        ffmpeg = _ffmpeg_bin_or_none()
        if not ffmpeg:
            log.warning("%s skipped: ffmpeg unavailable.", label)
            return False
        clips = [
            p for p in (paths or [])
            if p and os.path.isfile(p) and os.path.getsize(p) > 0
        ]
        if not clips:
            log.warning("%s skipped: no usable clip.", label)
            return False
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        concat_path = dest.parent / f".{dest.stem}.{os.getpid()}.concat.txt"
        publish = dest.parent / f".{dest.stem}.{os.getpid()}.tmp.mp4"
        extra_temps: list[Path] = []
        try:
            probes = [_probe_stream_keys(ffmpeg, p) for p in clips]
            # 🔴 有片段**完全没有音轨**时：concat 滤镜图里的 ``[i:a]`` 会引用不存在的
            # 流，ffmpeg 直接报 ``Error binding filtergraph inputs/outputs: Invalid
            # argument``（本机 7.1 实测复现），整条时间轴拼不出来 —— 预览空白、
            # 整片也出不来。先给这些片段补一条静音音轨（视频 ``-c:v copy``，
            # 零重编码），让每一路都有同构的流，再交给 concat 的任一实现。
            if any(pk[1] is None for pk in probes):
                fixed = _with_silent_audio(ffmpeg, clips, probes)
                if fixed:
                    clips, extra_temps = fixed
                    probes = [_probe_stream_keys(ffmpeg, p) for p in clips]
            concat_path.write_text(
                "".join(f"file '{p.replace(chr(92), '/')}'\n" for p in clips),
                encoding="utf-8",
            )
            vkeys = {pk[0] for pk in probes if pk[0]}
            arates = [int(pk[1][1]) for pk in probes if pk[1] and pk[1][1] > 0]
            # 只有「所有片段都探到视频流且参数完全一致」才敢走 demuxer + copy。
            # 探测全失败（vkeys 为空）是「没有信息」，不能当成「一致」。
            copy_ok = len(vkeys) == 1
            # 采样率**不统一**时不能让 concat demuxer 接手（音轨时基会被算成 1.378 倍，
            # 见 _probe_stream_keys）；全部探不出音轨时交回 demuxer 的原有逻辑。
            uniform_audio = len(arates) == len(clips) and len(set(arates)) == 1
            cmd: list[str] | None = None
            mode = "copy" if copy_ok else "re-encode"
            if not uniform_audio:
                cmd = _filter_concat_cmd(ffmpeg, clips, probes, publish)
                if cmd is not None:
                    mode = "filter"
            if cmd is None:
                cmd = [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-f", "concat", "-safe", "0", "-i", str(concat_path),
                ]
                if copy_ok and uniform_audio:
                    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
                    mode = "copy"
                else:
                    cmd += [
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    ]
                    mode = "re-encode"
                cmd += ["-movflags", "+faststart", str(publish)]
            try:
                if publish.exists():
                    publish.unlink()
            except OSError:
                pass
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if proc.returncode != 0 or not publish.is_file() or publish.stat().st_size <= 0:
                err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
                log.warning("%s failed (code=%s): %s", label, proc.returncode, err)
                return False
            os.replace(publish, dest)
            log.info(
                "%s → %s (%d clip(s), %s)",
                label, dest, len(clips), mode,
            )
            return True
        finally:
            for tmp in (concat_path, publish, *extra_temps):
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
    except Exception as exc:
        log.warning("%s skipped: %s", label, exc)
        return False


def resolve_preview_entries(
    *,
    node_id: str | None = None,
    plan: DirectorPlan | None = None,
    shape=None,
    task_key: str = "",
    fps: float = 0.0,
    width: int = 0,
    height: int = 0,
) -> tuple[list[dict], str]:
    """出片预览与产物导出**唯一取源入口** —— 两条路走同一个函数、同一个优先级。

    它们各自再写一份取源代码，就必然分叉（历史上「页面看到的」与「导出的」对不上，
    根因都是这个）。所以这里把顺序钉死，两侧只提供各自手上有的东西：

      ① ``node_id`` 的落盘快照 —— 但必须是**同一条时间轴**（指纹一致才算命中）。
         命中即「页面上正在播的那批条目」，两处拿到的是同一个文件；
      ② 有 ``plan``（节点执行侧）→ 按当前时间轴重建（:func:`build_segment_playlist`），
         顺手把结果与指纹写回快照，供页面侧共用；
      ③ 只有 ``shape``（HTTP 侧、前端送上来）→ 按形状展开（:func:`build_preview_playlist_for_shape`），
         同样写回快照。② 与 ③ 的输入同源（都来自 ``timeline_data.segments``），
         所以同一时刻两侧得到**同一批条目**；
      ④ 兜底：只有任务表（既无 plan 又无形状，例如旧前端 / 后端刚重启还没跑过）——
         段数会塌成「登记过的那几段」，但这已经是当时唯一的信息，两侧一致。

    Returns ``(entries, source)``；``source`` ∈ ``snapshot`` / ``plan`` / ``shape`` /
    ``manifests`` / ``none``。
    """
    key = ""
    if plan is not None:
        key = shape_key_from_plan(plan)
    elif shape:
        key = compute_shape_key(shape, task_key=task_key)

    # ① 最近一次展开的快照（须与当前时间轴同源）。
    if node_id:
        items = get_preview_items(node_id, expect_shape_key=key or None)
        if items:
            return items, "snapshot"

    # ② 节点执行侧：当前 plan 是这条时间轴的权威描述。
    if plan is not None:
        items = build_segment_playlist(plan)
        if items:
            if node_id:
                try:
                    set_preview_items(node_id, items, shape_key=key)
                except Exception:
                    pass
            return items, "plan"

    # ③ HTTP 侧：只有形状，按形状展开（与 ② 同一份数据来源）。
    if shape:
        items = build_preview_playlist_for_shape(
            shape, task_key=task_key, fps=fps, width=width, height=height
        )
        if items:
            if node_id:
                try:
                    set_preview_items(node_id, items, shape_key=key)
                except Exception:
                    pass
            return items, "shape"

    # ④ 兜底：任务表。
    items = build_preview_playlist_from_manifests()
    return items, ("manifests" if items else "none")


# ---------------------------------------------------------------------------
# 段落片段「任务表」：按内容指纹 + 段索引登记，跨运行可靠复用
#
# 旧方案扫所有运行目录的 mtime 挑「最新片段」不可靠：半写的段、同段不同内容、
# 旧运行残留都会被误选。改为维护一张文件任务表——
#   * 每个任务类型（r2v / i2v / t2v / fl2v / v2v / rv2v）各一张 manifest；
#   * key = 文件指纹(prompt + 负向 + 参考素材标识 + 时长帧数) + "#" + 段索引；
#   * 段 mp4 写盘成功即登记「最新有效片段」绝对路径；
#   * 拼接「未重渲染的段」时按当前 plan 每段指纹查表，命中且文件仍有效才用。
# 不同内容 / 没写完的段永远不会被拼进来。
# ---------------------------------------------------------------------------

def _manifest_dir() -> Path:
    try:
        base = Path(folder_paths.get_output_directory()) / "minimax_seg_manifests"
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        return Path(folder_paths.get_output_directory())


def _manifest_path(task_key: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9]", "", str(task_key or "unknown")) or "unknown"
    return _manifest_dir() / f"{safe}.json"


def _load_manifest(task_key: str) -> dict:
    try:
        p = _manifest_path(task_key)
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_manifest_file(mf: Path) -> dict:
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(task_key: str, data: dict) -> None:
    try:
        p = _manifest_path(task_key)
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as exc:
        log.warning("Segment manifest save failed (%s): %s", task_key, exc)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _ref_identities(seg: SegmentPlan) -> list[str]:
    """参考素材的稳定标识：文件名（去目录），按类型前缀分组后排序。"""
    ids: list[str] = []
    for r in getattr(seg, "refs", []) or []:
        f = str(getattr(r, "image_file", "") or "").replace("\\", "/").strip()
        if f:
            ids.append("img:" + f.rsplit("/", 1)[-1])
    for r in getattr(seg, "ref_videos", []) or []:
        f = str(getattr(r, "video_file", "") or "").replace("\\", "/").strip()
        if f:
            ids.append("vid:" + f.rsplit("/", 1)[-1])
    for r in getattr(seg, "ref_audios", []) or []:
        f = str(getattr(r, "audio_file", "") or getattr(r, "audio_path", "") or "").replace(
            "\\", "/"
        ).strip()
        if f:
            ids.append("aud:" + f.rsplit("/", 1)[-1])
    return sorted(ids)


def compute_segment_fingerprint(seg: SegmentPlan) -> str:
    """文件指纹 = 提示词 + 负向 + 参考素材标识 + 时长(帧数) 的稳定 sha256 前 16 位。"""
    parts = [
        "p=" + _norm_ws(getattr(seg, "prompt", "")),
        "n=" + _norm_ws(getattr(seg, "negative_prompt", "")),
        "r=" + "|".join(_ref_identities(seg)),
        "d=" + str(int(getattr(seg, "frame_count", 0) or 0)),
    ]
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def register_segment_clip(
    plan: DirectorPlan,
    seg: SegmentPlan,
    abs_path: str,
    frames: int,
) -> None:
    """段 mp4 写盘成功后登记「最新运行片段」到该任务类型的 manifest（内容指纹 + 段索引）。"""
    if not abs_path or not os.path.isfile(abs_path) or os.path.getsize(abs_path) <= 0:
        return
    task = str(getattr(seg, "task_key", "") or getattr(plan, "global_task_key", "") or "").strip()
    if not task:
        return
    try:
        fp = compute_segment_fingerprint(seg)
        idx = int(seg.index)
        key = f"{fp}#{idx}"
        data = _load_manifest(task)
        # 段内容被修改 → 旧指纹的登记项已失效。清掉同段索引下、指纹不同的旧条目，
        # 避免 manifest 累积死数据，也避免无 plan 兜底路径误选到旧内容。
        for old in [k for k in data if k != key and k.split("#", 1)[-1] == str(idx)]:
            data.pop(old, None)
        data[key] = {
            "fingerprint": fp,
            "segment_index": idx,
            "ui_index": int(seg.timeline_index),
            "task_key": task,
            "run_dir": os.path.basename(os.path.dirname(abs_path)),
            "filename": os.path.basename(abs_path),
            "abs": os.path.abspath(abs_path),
            "frames": int(frames or 0),
            "mtime": os.path.getmtime(abs_path),
            "prompt": (getattr(seg, "prompt", "") or "")[:200],
            "refs": _ref_identities(seg),
        }
        _save_manifest(task, data)
    except Exception as exc:
        log.warning(
            "Segment manifest register failed (%s #%d): %s", task, int(seg.index), exc
        )


# ---------------------------------------------------------------------------
# 手动「恢复素材」：把现成的 mp4 直接绑到时间轴某一段（**不依赖提示词指纹**）。
#
# 任务表靠内容指纹（提示词 + 负向 + 参考素材 + 帧数）定位段 —— 这对「同一版参数
# 重新跑出来的段」是准确的；但用户的旧素材常常是在**改过时长/提示词的版本**下生成的
# （实测同一条时间轴的同一段既有 124 帧版、又有 136 帧版），一改指纹必然对不上，
# 那些段在预览里就全成了红斜纹占位片，而素材其实好好躺在磁盘上。
#
# 指纹证明不了的事只能由人担保：用户在标红段上选一个 mp4、指定「它就属于第 N 段」，
# 于是写一条 ``bind#<idx>`` 记录。查询优先级 **低于内容指纹、高于占位片**：
#   · 该段真跑过一次生成 → register_segment_clip 会清掉同段索引下的旧条目，手动
#     绑定被真实产物覆盖（机器算出来的优先，符合直觉）；
#   · 没跑过 → 手动绑定生效，红斜纹变回可播放的真实素材。
# ---------------------------------------------------------------------------

BIND_KEY_PREFIX = "bind#"
RECOVERED_DIR_PREFIX = "recovered_"

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _probe_clip_seconds(path: str) -> float:
    """``ffmpeg -i`` 读容器时长（秒）；读不到返回 0.0（不影响绑定，只是帧数记 0）。"""
    ffmpeg = _ffmpeg_bin_or_none()
    if not ffmpeg or not path:
        return 0.0
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, timeout=60
        )
        m = _DURATION_RE.search(proc.stderr.decode("utf-8", "replace"))
        if not m:
            return 0.0
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        return 0.0


def _probe_clip_frames(path: str, frame_rate: float = 0.0) -> int:
    """片段真实帧数 = 时长 × 帧率（两者都尽量从文件本身读，读不到才退回 24fps）。"""
    fps = float(frame_rate or 0)
    if not fps:
        probe = _probe_video_key(_ffmpeg_bin_or_none(), str(path))
        if probe and probe[4]:
            fps = float(probe[4])
    seconds = _probe_clip_seconds(path)
    if seconds <= 0:
        return 0
    return int(round(seconds * (fps or 24.0)))


_RECOVERED_MAX_DIRS = 40  # recovered_*/ 目录上限（超了才回收最旧的「孤儿」）


def _referenced_clip_paths() -> set[str]:
    """所有**仍被引用**的素材绝对路径（任务表登记 + 预览快照条目）。

    用于回收 ``recovered_*/`` 时判断「这个文件还有没有人指向它」。任何一层 dict
    里的 ``abs`` 都算引用 —— 两个目录的 JSON 结构不同（manifest 是 ``{key: {...}}``、
    快照是 ``{"entries": [...]}``），递归扫最省事也最不容易漏。
    """
    refs: set[str] = set()
    base = Path(folder_paths.get_output_directory())
    for sub in ("minimax_seg_manifests", "minimax_preview_items"):
        d = base / sub
        try:
            files = list(d.glob("*.json"))
        except OSError:
            continue
        for f in files:
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            stack = [raw]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    p = cur.get("abs")
                    if isinstance(p, str) and p:
                        try:
                            refs.add(os.path.normcase(os.path.abspath(p)))
                        except OSError:
                            pass
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)
    return refs


def prune_recovered_clips(keep_dir: str | None = None) -> int:
    """回收 ``minimax_seg_export/recovered_*/`` 里**没人引用**的旧目录。

    为什么需要：每次「恢复素材」都会新建一个 ``recovered_<时间戳>/`` 并 copy 一份
    mp4 —— 反复恢复同一段就会无限堆积。项目里预览缓存 / 占位片都有 prune 兜底，
    这条不能漏（否则只增不减）。

    🔴 **被引用的文件永不删除**：任务表登记（``bind#<idx>``）或预览快照里的 ``abs``
    只要还指向它，删掉就会让对应段变 missing —— 那是用户恢复的资产。
    只有「目录数超上限」时才动手，且只删最旧、且**整目录都无引用**的那些。
    """
    base = Path(folder_paths.get_output_directory()) / "minimax_seg_export"
    try:
        dirs = [
            d for d in base.iterdir()
            if d.is_dir() and d.name.startswith(RECOVERED_DIR_PREFIX)
        ]
    except OSError:
        return 0
    if len(dirs) <= _RECOVERED_MAX_DIRS:
        return 0
    try:
        dirs.sort(key=lambda d: d.stat().st_mtime)
    except OSError:
        return 0
    refs = _referenced_clip_paths()
    keep_name = Path(keep_dir).name if keep_dir else ""
    removed = 0
    alive = len(dirs)
    # 从最旧往后扫：跳过 keep 与被引用的，继续找可删的，**直到真正降到上限**
    # （只删「最旧 N 个」的话，一旦它们恰好都被引用就一个也删不掉，目录照样涨）。
    for d in dirs:
        if alive <= _RECOVERED_MAX_DIRS:
            break
        if keep_name and d.name == keep_name:
            continue
        try:
            files = [f for f in d.glob("*.mp4")]
        except OSError:
            continue
        used = False
        for f in files:
            try:
                if os.path.normcase(os.path.abspath(str(f))) in refs:
                    used = True
                    break
            except OSError:
                used = True
                break
        if used:
            continue
        try:
            shutil.rmtree(str(d))
            removed += 1
            alive -= 1
        except OSError:
            continue
    if removed:
        log.info("Recovered clip prune: removed %d orphan dir(s).", removed)
    return removed


def import_recovered_clip(src_path: str, segment_index: int) -> str | None:
    """把一份 mp4 收进 ``output/minimax_seg_export/recovered_<ts>/seg_XXXX.mp4``。

    收进来而不是原地引用，有两个理由：
      ① 该目录与正常产物同构，``_view_ref`` 能直接算出 /view 引用（input/ 下的文件
         算不出来，会落到错误的 subfolder）；
      ② 源文件可能在 ``input/`` 或临时目录，被清理/移动后预览就断了。
    """
    try:
        src = Path(str(src_path or ""))
        if not src.is_file() or src.stat().st_size <= 0:
            return None
        idx = max(0, int(segment_index))
        base = Path(folder_paths.get_output_directory()) / "minimax_seg_export"
        dest_dir = base / f"{RECOVERED_DIR_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"seg_{idx:04d}.mp4"
        n = 1
        while dest.exists():
            dest = dest_dir / f"seg_{idx:04d}_{n}.mp4"
            n += 1
        shutil.copy2(str(src), str(dest))
        # 顺手回收没人引用的旧目录（这条链只增不减会撑爆磁盘）；本目录 keep 永不删。
        try:
            prune_recovered_clips(keep_dir=str(dest_dir))
        except Exception:
            pass
        return str(dest)
    except Exception as exc:
        log.warning("Import recovered clip failed (%s): %s", src_path, exc)
        return None


def bind_segment_clip(
    task_key: str,
    segment_index: int,
    abs_path: str,
    *,
    frames: int = 0,
    note: str = "",
) -> dict | None:
    """把 ``abs_path`` 这份 mp4 手动绑定到 ``task_key`` 表的第 ``segment_index`` 段。"""
    task = str(task_key or "").strip()
    if not task or not abs_path:
        return None
    p = Path(str(abs_path))
    if not p.is_file() or p.stat().st_size <= 0:
        return None
    try:
        idx = max(0, int(segment_index))
        record = {
            "fingerprint": "",
            "manual": True,
            "segment_index": idx,
            "ui_index": idx,
            "task_key": task,
            "run_dir": os.path.basename(os.path.dirname(str(p))),
            "filename": os.path.basename(str(p)),
            "abs": os.path.abspath(str(p)),
            "frames": int(frames or 0) or _probe_clip_frames(str(p)),
            # 用「现在」而不是文件 mtime：无 plan 兜底路径按 mtime 取每段最新一条，
            # 手动绑定是用户刚做的决定，理应压过历史上任何一条登记。
            "mtime": time.time(),
            "prompt": "",
            "refs": [],
            "note": str(note or "")[:120],
        }
        data = _load_manifest(task)
        data[f"{BIND_KEY_PREFIX}{idx}"] = record
        _save_manifest(task, data)
        log.info(
            "Segment #%d manually bound to %s (%s)", idx, record["filename"], task
        )
        return record
    except Exception as exc:
        log.warning("Bind segment clip failed (%s #%s): %s", task, segment_index, exc)
        return None


def unbind_segment_clip(task_key: str, segment_index: int) -> bool:
    """撤销某段的手动绑定（只删 ``bind#<idx>``，不动指纹登记）。"""
    task = str(task_key or "").strip()
    if not task:
        return False
    try:
        key = f"{BIND_KEY_PREFIX}{max(0, int(segment_index))}"
        data = _load_manifest(task)
        if key not in data:
            return False
        data.pop(key, None)
        _save_manifest(task, data)
        return True
    except Exception as exc:
        log.warning("Unbind segment clip failed (%s #%s): %s", task, segment_index, exc)
        return False


def _ffmpeg_bin_or_none() -> str | None:
    try:
        from ..lib.video_export import _ffmpeg_bin

        return _ffmpeg_bin()
    except Exception:
        return None


def _view_ref(abs_path: str) -> tuple[str, str]:
    """绝对路径 → ComfyUI /view 用的 (subfolder, filename)（相对 output 根目录）。"""
    try:
        out = Path(folder_paths.get_output_directory()).resolve()
        p = Path(str(abs_path)).resolve()
        rel = p.relative_to(out)
        parts = rel.parts
        if len(parts) <= 1:
            return "", (parts[-1] if parts else Path(str(abs_path)).name)
        return "/".join(parts[:-1]), parts[-1]
    except Exception:
        return "minimax_seg_export", Path(str(abs_path)).name


# ---------------------------------------------------------------------------
# 缺失段「占位片」：代码生成（ffmpeg lavfi color + drawtext），按 key 缓存复用。
#
# 出片预览改为「播放列表逐段连播」，不再为「完整视频」反复 concat 落盘 ——
# 缺段直接用一小段文字提示视频补位，全程在 <video> 里连续播放：既不黑屏、
# 也不因拼接坏文件整块消失。占位片按 (段号, 帧数, fps, 分辨率) 命名缓存于共享
# 目录，同一段重复预览直接复用，不随运行次数累积。
# ---------------------------------------------------------------------------

_PLACEHOLDER_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _placeholder_dir() -> Path:
    base = Path(folder_paths.get_output_directory()) / "minimax_seg_placeholder"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _pick_placeholder_font() -> str | None:
    for f in _PLACEHOLDER_FONT_CANDIDATES:
        try:
            if os.path.isfile(f):
                return f
        except Exception:
            continue
    return None


_PLACEHOLDER_MAX_FILES = 80
_PLACEHOLDER_TTL_SEC = 7 * 24 * 3600  # 7 天没被用过 → 回收
# 占位片静音轨的采样率 —— **必须与真实段一致**。concat demuxer 在采样率混用时会把
# 音轨时基算错（实测 32 kHz 真实段 + 44.1 kHz 占位片 → 音轨被拉成 1.378 倍）。
# MiniMax H3 原生 AAC 是 32 kHz（实测真实段全为 32000 Hz），所以占位片跟它对齐：
# 参数一致才走 `-c:v copy` 的快路径（预览才不卡）。万一仍不一致（例如手动恢复的素材
# 是 48 kHz），`_concat_clips_to` 会自动改走 concat 滤镜兜住正确性，只是慢一点。
_PLACEHOLDER_AUDIO_RATE = 32000


def prune_placeholder_clips(keep: str | None = None) -> int:
    """占位片回收：名字里带分辨率/帧数，换输出设置就会生成新的一批 —— 不回收会慢慢堆积。

    规则：半写残留（``.<名>.<pid>.tmp.mp4``）与超过 TTL 的立即删；总量超上限时按
    mtime 从最旧删到上限；``keep``（本次要用的那一个）永不删。返回删除数量。
    """
    removed = 0
    try:
        base = _placeholder_dir()
        now = time.time()
        keep_p = str(keep or "")
        keep_name = os.path.basename(keep_p) if keep_p else ""
        alive: list[tuple[float, Path]] = []
        for f in base.iterdir():
            try:
                if not f.is_file():
                    continue
                st = f.stat()
            except OSError:
                continue
            if keep_name and f.name == keep_name:
                alive.append((st.st_mtime, f))
                continue
            stale = f.name.startswith(".") or (now - st.st_mtime) > _PLACEHOLDER_TTL_SEC
            if not stale:
                alive.append((st.st_mtime, f))
                continue
            try:
                f.unlink()
                removed += 1
            except OSError:
                continue
        if len(alive) > _PLACEHOLDER_MAX_FILES:
            alive.sort(key=lambda x: x[0], reverse=True)
            for _mt, f in alive[_PLACEHOLDER_MAX_FILES:]:
                if keep_name and f.name == keep_name:
                    continue
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    continue
    except Exception:
        pass
    return removed


def ensure_placeholder_clip(
    index: int, frames: int, fps: float, width: int, height: int
) -> str | None:
    """生成「第 N 段缺失」占位片（黑底 + 文字）。按 key 缓存，失败返回 None（预览跳过该段）。"""
    try:
        ffmpeg = _ffmpeg_bin_or_none()
        if not ffmpeg:
            return None
        fps = float(fps or 24) or 24.0
        fr = int(frames or 0) or int(max(1, round(fps)))  # 无帧数时占 1 秒
        w = int(width or 0)
        h = int(height or 0)
        if w < 16 or h < 16:
            w, h = 1280, 720
        w -= w % 2
        h -= h % 2
        dur = max(0.2, fr / fps)
        # _a1 = 带静音音轨的版本。流式 concat 时若占位段缺音轨而真实段有音轨，
        # `-c:a aac` 会因输入流不齐而失败 —— 所以占位片必须自带一条静音音轨。
        # 音轨采样率跟真实段对齐（名字里带上，换率会生成新文件，不会拿到旧的 44.1k）。
        name = (
            f"missing_{int(index):04d}_{fr}f_{int(round(fps))}_{w}x{h}"
            f"_a1_{_PLACEHOLDER_AUDIO_RATE}.mp4"
        )
        dest = _placeholder_dir() / name
        try:
            if dest.is_file() and dest.stat().st_size > 0:
                return str(dest)
        except OSError:
            pass
        font = _pick_placeholder_font()
        if font:
            # ffmpeg 滤镜串里 ':' 是选项分隔符 —— Windows 盘符（C:）必须转义成 '\:'，
            # 否则 `fontfile=C:/...` 会被截成 fontfile=C + 一个非法选项（实测解析报错）。
            fesc = str(font).replace("\\", "/").replace(":", "\\:")
            fs1 = max(18, h // 12)
            fs2 = max(14, h // 20)
            gap = max(6, h // 20)
            vf = (
                f"drawtext=fontfile='{fesc}':text='第 {int(index) + 1} 段缺失':"
                f"fontcolor=0xE6E6E6:fontsize={fs1}:x=(w-text_w)/2:y=(h-text_h)/2-{gap},"
                f"drawtext=fontfile='{fesc}':text='需重新生成':"
                f"fontcolor=0xFF8A7A:fontsize={fs2}:x=(w-text_w)/2:y=(h-text_h)/2+{gap}"
            )
        else:
            vf = "null"
        publish = _placeholder_dir() / f".{name}.{os.getpid()}.tmp.mp4"
        try:
            if publish.exists():
                publish.unlink()
        except OSError:
            pass
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"color=c=0x141b26:s={w}x{h}:d={dur:.3f}:r={fps:.6f}",
            # 静音音轨：与真实段（h264 + aac）保持流结构**和采样率**一致，
            # 流式 concat 才能 copy 成功、时长才不会被算错。
            "-f", "lavfi", "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={_PLACEHOLDER_AUDIO_RATE}",
            "-shortest",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(publish),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0 or not publish.is_file() or publish.stat().st_size <= 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            log.warning("Placeholder clip gen failed (code=%s): %s", proc.returncode, err)
            return None
        os.replace(publish, dest)
        log.info("Missing-segment placeholder clip → %s", dest)
        # 只在「新建了一个」时回收（缓存命中直接 return，不天天扫目录）。
        prune_placeholder_clips(keep=str(dest))
        return str(dest)
    except Exception as exc:
        log.warning("Placeholder clip skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 工作流身份指纹（shape_key）
#
# 预览快照按 **节点 id** 命名，而节点 id 只在单个工作流内唯一 —— 多个工作流各放一个
# 导演节点时（id 都可能正好是 5），一个工作流的快照会被另一个当成自己的用掉，产物就
# 可能导出别条时间轴的内容。浏览器端手上就有「当前这条时间轴长什么样」，所以把
# **段形状 + 任务类型** 压成一个短指纹：写快照时存下来、读快照时校验。
#   * 形状/任务不同 → 指纹不同 → 不复用别的工作流的快照（各自重新展开，互不污染）；
#   * 形状/任务相同 → 指纹相同 → 正常命中（同一时间轴）。
# 前端 ``_timelineShape()`` 与后端 ``plan.segments`` 都源自 ``timeline_data.segments``，
# 实测 16/16 段逐段一致（``frameCount == end_frame - start_frame``），两边算得出同一个值。
# ---------------------------------------------------------------------------

def compute_shape_key(segments, *, task_key: str = "") -> str:
    """段形状（``index:frames`` 序列）+ 任务类型 → 稳定短指纹；无法识别时返回 ``""``。

    ``segments`` 接受 ``[{"index": i, "frames": n}, ...]`` 或 ``[[i, n], ...]``；
    顺序无关（先按段号排序），帧数取绝对值，任务键经 :func:`resolve_task_key` 归一 ——
    保证「前端送的标签」与「plan 的 global_task_key」算出同一个值。
    """
    try:
        pairs: list[tuple[int, int]] = []
        for raw in segments or []:
            if isinstance(raw, dict):
                idx = int(raw.get("index", -1))
                frames = int(raw.get("frames") or 0)
            else:
                idx, frames = int(raw[0]), int(raw[1] or 0)
            if idx < 0:
                continue
            pairs.append((idx, max(0, frames)))
        if not pairs:
            return ""
        pairs.sort()
        shape = "|".join(f"{i}:{f}" for i, f in pairs)
        key = ""
        text = str(task_key or "").strip()
        if text:
            try:
                key = resolve_task_key(text)
            except Exception:
                key = text
        raw = f"t={key}\nshape={shape}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def shape_key_from_plan(plan) -> str:
    """``DirectorPlan`` → 与前端形状**同源**的身份指纹。"""
    segs: list[dict] = []
    for seg in getattr(plan, "segments", []) or []:
        try:
            segs.append(
                {
                    "index": int(getattr(seg, "index", -1)),
                    "frames": int(getattr(seg, "frame_count", 0) or 0),
                }
            )
        except Exception:
            continue
    return compute_shape_key(segs, task_key=str(getattr(plan, "global_task_key", "") or ""))


def _entry_ok(idx: int, frames: int, abs_path: str, *, source: str = "manifest") -> dict:
    sub, fn = _view_ref(abs_path)
    return {
        "index": int(idx),
        "frames": int(frames or 0),
        "status": "ok",
        "subfolder": sub,
        "filename": fn,
        # "manifest" = 内容指纹命中任务表（段内容与当前 plan 一致，可信）；
        # "bound" = 用户手动「恢复素材」绑定的现成 mp4（不校验提示词，由人担保）。
        # （历史值 "scan" 已废弃：按段号扫盘会把别条时间轴的段拼进来。）
        "source": str(source or "manifest"),
        # 内部字段：流式拼接要用绝对路径。progress.py 下发时会按白名单过滤，不会外泄。
        "abs": str(abs_path),
    }


def _entry_missing(idx: int, frames: int, fps: float, width: int, height: int) -> dict:
    entry = {"index": int(idx), "frames": int(frames or 0), "status": "missing"}
    ph = ensure_placeholder_clip(idx, frames, fps, width, height)
    if ph:
        sub, fn = _view_ref(ph)
        entry["subfolder"] = sub
        entry["filename"] = fn
        entry["abs"] = str(ph)
    return entry


def build_segment_playlist(plan: DirectorPlan) -> list[dict]:
    """当前 plan → 出片预览播放列表：时间轴**每段一条**。

    每段按优先级三级取源，**唯一自动判据是「当前任务类型 + 当前提示词指纹」**：
      ① 任务表指纹命中 —— 该段的 ``task_key`` 表里、用它自己的内容指纹
         （提示词 + 负向 + 参考素材 + 帧数）登记过，且文件仍在 → 用这份真实片段；
      ② 指纹未命中 → 查该段的手动绑定 ``bind#<idx>``（用户在标红段上「恢复素材」
         上传/指定的现成 mp4）→ 用这份。改过时长/提示词的旧素材只能靠这条找回；
      ③ 都没有 → 代码生成的「第 N 段缺失」占位片（时长按**当前** plan 的帧数，
         所以整条时间轴的总时长仍然精确等于当前时间轴，不会被旧素材带偏）。

    🔴 **不要**退化成「按段号扫磁盘取最新同名段」。磁盘上 ``minimax_seg_export/``
    里躺着多**条**时间轴的产物（同一天可能既有 16 段的、又有 60 段的），它们段号相同
    但属于完全不同的提示词分组；按段号捞最新会把别的分组的段拼进来 ——
    实测用户当前时间轴 83.38s，却被拼成了 5:39（60 段 × 5.65s，来自另一条时间轴）。
    段号不承载任何语义，能证明「这段属于当前提示词」的只有内容指纹。

    占位片的尺寸/帧率取「**已命中素材的实际参数**」，与预览侧
    （:func:`build_preview_playlist_for_shape`）保持一致 —— 两侧不一致时同一条时间轴会
    拼出两个文件（预览文件的 cache key 含尺寸），「预览 = 产物」就不成立了。

    每条结构::

        {"index", "frames", "status": "ok"|"missing", "source",
         "subfolder", "filename"}   # subfolder/filename 可能缺省（占位生成失败时）
    """
    try:
        fps = float(getattr(plan, "frame_rate", 24) or 24)
        w = int(getattr(plan, "width", 0) or 0)
        h = int(getattr(plan, "height", 0) or 0)
        default_task = str(getattr(plan, "global_task_key", "") or "")
        # 同一轮里所有段共用一张任务表，读一次就够（16 段本来会读 16 次同一个文件）。
        manifests: dict[str, dict] = {}
        # 两阶段：先定「命中真实素材」的段，再用它们探到的实际参数补缺失段占位片。
        #   为什么不能一手一脚地补：占位片的尺寸进了预览文件的 cache key，而**预览侧**
        #   （``build_preview_playlist_for_shape``）用的是「已命中素材的实际分辨率」。
        #   这里若直接用 plan 的画布尺寸（如 384×288），同一条时间轴两边会拼出**两个文件**
        #   （素材是 1184×896 时尤其明显），「预览 = 产物」就不成立了。
        entries: list[dict | None] = []
        pending: dict[int, tuple[int, int]] = {}  # entries 下标 → (段号, 帧数)
        ok_paths: list[str] = []
        for seg in getattr(plan, "segments", []) or []:
            idx = int(getattr(seg, "index", -1))
            frames = int(getattr(seg, "frame_count", 0) or 0)
            # 与 register_segment_clip 保持同样的 task 回退规则，否则查询与登记会
            # 落到不同的表上（登记进 r2v.json、却去 unknown.json 里找 → 全判缺失）。
            task = str(getattr(seg, "task_key", "") or default_task)
            hit: tuple[int, str, str] | None = None
            table: dict = {}
            if task and idx >= 0:
                if task not in manifests:
                    manifests[task] = _load_manifest(task)
                table = manifests[task]
                fp = compute_segment_fingerprint(seg)
                rec = table.get(f"{fp}#{idx}")
                if rec:
                    p = Path(str(rec.get("abs") or ""))
                    if p.is_file() and p.stat().st_size > 0:
                        hit = (int(rec.get("frames") or frames), str(p), "manifest")
            # ② 手动「恢复素材」绑定：指纹对不上的旧素材靠这条命中（见 BIND_KEY_PREFIX 注）。
            if hit is None and table:
                rec = table.get(f"{BIND_KEY_PREFIX}{idx}")
                if rec:
                    p = Path(str(rec.get("abs") or ""))
                    if p.is_file() and p.stat().st_size > 0:
                        hit = (int(rec.get("frames") or frames), str(p), "bound")
            if hit:
                entries.append(_entry_ok(idx, hit[0], hit[1], source=hit[2]))
                ok_paths.append(hit[1])
            else:
                pending[len(entries)] = (idx, frames)
                entries.append(None)
        if pending:
            pfps, pw, ph = _probe_playlist_params(ok_paths)
            pfps = float(pfps or 0)
            mfps = float(fps or 0) or pfps or 24.0
            mw = int(w or 0) or pw
            mh = int(h or 0) or ph
            for pos, (idx, frames) in pending.items():
                entries[pos] = _entry_missing(idx, frames, mfps, mw, mh)
        return [e for e in entries if e]
    except Exception as exc:
        log.warning("Build segment playlist skipped: %s", exc)
        return []


def _latest_manifest_records(task_key: str | None = None) -> dict[int, dict]:
    """``{段号: 该段最近一次登记记录}``。

    ``task_key`` 给定时**只读那一张任务表** —— 段号只在同一任务类型（同一提示词分组）
    内才有意义；跨表按段号取会拿到别的分组的素材（实测把 16 段时间轴拼成 60 段 / 5:39）。
    """
    latest: dict[int, dict] = {}
    try:
        mdir = _manifest_dir()
        if not mdir.is_dir():
            return latest
        if task_key:
            files = [_manifest_path(task_key)]
        else:
            files = [f for f in sorted(mdir.glob("*.json")) if not f.name.endswith(".tmp")]
        for mf in files:
            for rec in _load_manifest_file(mf).values():
                if not isinstance(rec, dict):
                    continue
                idx = int(rec.get("segment_index", -1))
                if idx < 0:
                    continue
                prev = latest.get(idx)
                if prev is None or float(rec.get("mtime", 0)) > float(prev.get("mtime", 0)):
                    latest[idx] = rec
    except Exception as exc:
        log.warning("Load manifest records skipped: %s", exc)
    return latest


def _probe_playlist_params(paths: list[str]) -> tuple[float, int, int]:
    """从任意一个有效段上探 ``(fps, w, h)`` —— 占位片按同一参数生成，
    concat 才能走 ``-c:v copy``（参数不一致只能退化成全重编码，慢）。"""
    fps, w, h = 24.0, 0, 0
    for p in paths:
        probe = _probe_video_key(_ffmpeg_bin_or_none(), p)
        if probe:
            _codec, pw, ph, _pf, pfps = probe
            w, h = int(pw), int(ph)
            if pfps:
                fps = float(pfps)
            break
    return fps, w, h


def build_preview_playlist_for_shape(
    shape,
    *,
    task_key: str = "",
    fps: float = 0.0,
    width: int = 0,
    height: int = 0,
) -> list[dict]:
    """按**当前时间轴的段形状**展开播放列表 —— 页面刷新 / 后端重启后恢复的正路。

    ``shape`` 是前端下发的 ``[{"index": i, "frames": n}, ...]``，就是导演台时间轴上
    现有的每一段。段数与每段帧数**只有浏览器端持有**，所以这里一个都不用猜：

      ① 段号在该任务类型的任务表里有有效登记 → 用那份真实片段（``manifest``）；
      ② 手动「恢复素材」绑定过该段 → 用绑定文件（``bound``）；
      ③ 都没有 → 按时长补「第 N 段缺失」占位片，整条时间轴仍然连续、不黑屏。

    两条曾经的错路（都别回头）：
      * 只读任务表 → 段数只等于登记过的段数（16 段的时间轴刷新后只剩 17 秒）；
      * 按段号扫盘补段 → ``minimax_seg_export/`` 里躺着多**条**时间轴的产物，
        段号相同但属于不同的提示词分组，实测把 83.38s 的当前时间轴拼成 5:39。
    """
    try:
        recs = _latest_manifest_records(task_key or None)
        items: list[tuple[int, int]] = []
        for raw in shape or []:
            if isinstance(raw, dict):
                idx = int(raw.get("index", -1))
                frames = int(raw.get("frames") or 0)
            else:
                try:
                    idx, frames = int(raw[0]), int(raw[1] or 0)
                except Exception:
                    continue
            if idx < 0:
                continue
            items.append((idx, frames))
        if not items:
            return []
        ok_paths: dict[int, str] = {}
        sources: dict[int, str] = {}
        for idx, _frames in items:
            rec = recs.get(idx)
            if not rec:
                continue
            p = Path(str(rec.get("abs") or ""))
            try:
                if p.is_file() and p.stat().st_size > 0:
                    ok_paths[idx] = str(p)
                    sources[idx] = "bound" if rec.get("manual") else "manifest"
            except OSError:
                continue
        pfps, pw, ph = _probe_playlist_params(list(ok_paths.values()))
        fps = float(fps or 0) or pfps
        w = int(width or 0) or pw
        h = int(height or 0) or ph
        entries: list[dict] = []
        for idx, frames in items:
            if idx in ok_paths:
                # 用**素材实际帧数**（登记值），不是时间轴标称值 —— 预览里每条的时长
                # 必须和拼出来的视频一致，否则播放头/拖动位置会一路漂。
                real = int(recs[idx].get("frames") or 0) or frames
                entries.append(_entry_ok(idx, real, ok_paths[idx], source=sources[idx]))
            else:
                entries.append(_entry_missing(idx, frames, fps, w, h))
        return entries
    except Exception as exc:
        log.warning("Build playlist for shape skipped: %s", exc)
        return []


def build_preview_playlist_from_manifests() -> list[dict]:
    """旧降级路（既没有 plan、也没有前端时间轴形状时）：只按任务表登记过的段建列表。

    ⚠️ 段数只等于「登记过的段数」—— 16 段的时间轴在这里只会返回跑过的 3 段。
    页面刷新走 ``/preview_plan``（前端把当前时间轴形状送上来），不要再依赖这条路。
    """
    recs = _latest_manifest_records(None)
    if not recs:
        return []
    return build_preview_playlist_for_shape(
        [{"index": i, "frames": int(recs[i].get("frames") or 0)} for i in sorted(recs)]
    )


# ---------------------------------------------------------------------------
# 出片预览「预览文件」：按**内容指纹**落一份 faststart mp4，并**强制回收**。
#
# 演进（2026-10-03 三次改版）：
#   ① 跨运行拼 merged_latest.mp4 —— 每次运行/刷新都重拼，磁盘累积；缺段拼坏就黑屏。
#   ② 前端播放列表逐段连播 —— 段间换 <video> 源有可见闪烁。
#   ③ 流式（fragmented mp4 边拼边吐，不落盘）—— 不闪了，但 fMP4 不可随机 seek：
#      每次拖动都要重拉一条 ffmpeg 流，1–3 秒延迟；且 duration 未知、原生时间条不准。
#   ④ 现在：落一份**普通 mp4 + faststart**（moov 前置），seek 交给浏览器走 HTTP range
#      （毫秒级）、duration 正确。落盘带来的累积风险由 cache key + prune 兜住：
#       · key = 所有输入（路径 + size + mtime_ns）+ 输出参数的 sha256 → 内容没变就命中
#         既有文件，**不重拼**；改过一段 → 新 key → 新文件，旧的成孤儿被回收；
#       · 每次构建前 prune：TTL 过期 / 数量上限 / 总大小上限 / 半写 .tmp 残留，全清，
#         并且**永不删除正在服务的那一份**（keep）。
# ---------------------------------------------------------------------------

_VIDEO_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?: Video: ([A-Za-z0-9_]+).*?, (\d{2,5})x(\d{2,5})"
)
_PIXFMT_RE = re.compile(r"\b(yuvj?\d+p|nv12|nv21|rgb24|bgr24|gray|gbrp)\b")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Audio: ([A-Za-z0-9_]+)")
_AUDIO_RATE_RE = re.compile(r"Audio: [^\n]*?, (\d{2,6}) Hz")


def _probe_stream_keys(
    ffmpeg: str | None, path: str
) -> tuple[tuple | None, tuple | None]:
    """一次 ``ffmpeg -i`` 同时取回视频 / 音频关键参数 → ``(vkey, akey)``。

    * ``vkey`` = ``(codec, w, h, pix_fmt, fps)``，无视频流为 ``None``；
    * ``akey`` = ``(codec, sample_rate, channels)``，无音轨为 ``None``。

    本机只有 imageio-ffmpeg 提供的 ffmpeg（无 ffprobe），所以用 ``ffmpeg -i <file>``：
    不带输出参数它会立刻报错退出，但 stderr 里已含完整流信息，比解码全片快得多。

    🔴 **音频采样率必须和视频参数一起看。** ffmpeg 的 concat *demuxer* 在输入采样率
    不一致时会把音轨时基算错：实测 MiniMax H3 真实段（32 kHz）与占位片（44.1 kHz）
    混拼，音轨被拉成 **1.378 倍**（88.2s 的片子变成 121.6s），
    而 ``-ar`` / ``aresample=async`` / 统一分辨率全都救不回来 —— 只能提前识别出来，
    改走 concat **滤镜**（见 :func:`_filter_concat_cmd`）。
    """
    if not ffmpeg:
        return None, None
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, timeout=20
        )
        txt = (proc.stderr or b"").decode("utf-8", errors="replace")
        vkey = None
        m = _VIDEO_STREAM_RE.search(txt)
        if m:
            nl = txt.find("\n", m.start())
            line = txt[m.start(): nl if nl != -1 else len(txt)]
            pm = _PIXFMT_RE.search(line)
            fm = _FPS_RE.search(line)
            vkey = (
                m.group(1).lower(),
                int(m.group(2)),
                int(m.group(3)),
                (pm.group(1).lower() if pm else ""),
                (float(fm.group(1)) if fm else 0.0),
            )
        akey = None
        am = _AUDIO_STREAM_RE.search(txt)
        if am:
            nl = txt.find("\n", am.start())
            aline = txt[am.start(): nl if nl != -1 else len(txt)]
            rm = _AUDIO_RATE_RE.search(aline)
            akey = (
                am.group(1).lower(),
                (int(rm.group(1)) if rm else 0),
                0,
            )
        return vkey, akey
    except Exception:
        return None, None


def _probe_video_key(ffmpeg: str | None, path: str) -> tuple | None:
    """读视频流关键参数 ``(codec, w, h, pix_fmt, fps)``（音频参数见
    :func:`_probe_stream_keys`，同一次 ``ffmpeg -i`` 一并取回）。
    """
    return _probe_stream_keys(ffmpeg, path)[0]


# 预览缓存容量闸门 —— 「落盘可以，但不能持续累积」的硬保证：
#   * 同一内容永不重复生成（按内容指纹命名，命中即复用）；
#   * 每次构建前 prune：TTL 过期 → 先清；再按「最久未用」清到数量/总大小上限以内；
#   * 半写残留（.tmp）超过 1 小时直接清；当前正在服务的那一份永远不动。
_PREVIEW_CACHE_MAX_FILES = 12
_PREVIEW_CACHE_MAX_BYTES = 1536 * 1024 * 1024  # 1.5 GB
_PREVIEW_CACHE_TTL_SEC = 3 * 24 * 3600  # 3 天没被访问过 → 回收
_PREVIEW_TMP_STALE_SEC = 3600  # 崩溃留下的半成品，超过 1 小时清掉

_PREVIEW_LOCKS: dict[str, threading.Lock] = {}
_PREVIEW_LOCKS_GUARD = threading.Lock()


def _preview_cache_dir() -> Path:
    """预览文件缓存目录（在 ComfyUI output 下，才能被 /view 读取）。"""
    base = Path(folder_paths.get_output_directory()) / "minimax_preview_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _preview_lock(key: str) -> threading.Lock:
    """每个内容指纹一把锁：并发请求同一个 key 时只跑一次 ffmpeg。"""
    with _PREVIEW_LOCKS_GUARD:
        lk = _PREVIEW_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _PREVIEW_LOCKS[key] = lk
        return lk


def compute_preview_cache_key(
    items: list[dict], width: int = 0, height: int = 0, fps: float = 0.0
) -> str:
    """播放列表内容 → 缓存 key（sha256 前 16 位）。

    参与计算的是**每个输入文件的实际状态**（绝对路径 + 字节数 + 纳秒 mtime）与输出
    参数：任何一段被重新生成 / 被删 / 换成占位片，key 都会变 → 走新缓存文件；旧文件
    成为孤儿，由 :func:`prune_preview_cache` 回收。
    """
    h = hashlib.sha256()
    h.update(f"v1|{int(width or 0)}x{int(height or 0)}|{float(fps or 0):.6f}\n".encode())
    for it in items or []:
        p = str(it.get("abs") or "")
        try:
            st = os.stat(p)
            sig = f"{int(st.st_size)}:{int(st.st_mtime_ns)}"
        except OSError:
            sig = "-"
        h.update(
            f"{int(it.get('index', -1))}|{it.get('status')}|{p}|{sig}\n".encode("utf-8", "replace")
        )
    return h.hexdigest()[:16]


def prune_preview_cache(keep: set[str] | None = None) -> dict[str, int]:
    """回收预览缓存：「清 .tmp 残留 → TTL 过期 → 数量/总大小超限」。

    ``keep`` 里的文件名（正在服务的那一份）永不删除。超限时按 mtime 最旧优先删。
    返回 ``{"removed", "freed", "kept", "size"}``。任何异常都只记日志、不影响预览。
    """
    keep_names = {str(k) for k in (keep or set())}
    stats = {"removed": 0, "freed": 0, "kept": 0, "size": 0}
    try:
        base = _preview_cache_dir()
        now = time.time()
        live: list[tuple[float, int, Path]] = []
        for f in base.iterdir():
            try:
                if not f.is_file():
                    continue
                st = f.stat()
            except OSError:
                continue
            name = f.name
            # 半写残留：`.<name>.<pid>.tmp.mp4` / `.concat.txt` —— 只清够旧的，避免
            # 误删别的进程正在写的文件。
            if name.startswith(".") or ".tmp." in name:
                if now - st.st_mtime > _PREVIEW_TMP_STALE_SEC:
                    try:
                        f.unlink()
                        stats["removed"] += 1
                        stats["freed"] += int(st.st_size)
                    except OSError:
                        pass
                continue
            if not name.endswith(".mp4"):
                continue
            live.append((float(st.st_mtime), int(st.st_size), f))

        # ① TTL：太久没人用过 → 回收（保护在用）
        survivors: list[tuple[float, int, Path]] = []
        for mt, sz, f in live:
            if f.name in keep_names:
                survivors.append((mt, sz, f))
                continue
            if now - mt > _PREVIEW_CACHE_TTL_SEC:
                try:
                    f.unlink()
                    stats["removed"] += 1
                    stats["freed"] += sz
                    continue
                except OSError:
                    pass
            survivors.append((mt, sz, f))

        # ② 数量 / 总大小：最久未用先删（保护在用）
        survivors.sort(key=lambda e: e[0])
        count = len(survivors)
        total = sum(sz for _, sz, _ in survivors)
        for mt, sz, f in survivors:
            if count <= _PREVIEW_CACHE_MAX_FILES and total <= _PREVIEW_CACHE_MAX_BYTES:
                break
            if f.name in keep_names:
                continue
            try:
                f.unlink()
            except OSError:
                continue
            count -= 1
            total -= sz
            stats["removed"] += 1
            stats["freed"] += sz

        stats["kept"] = count
        stats["size"] = int(total)
        if stats["removed"]:
            log.info(
                "Preview cache pruned: removed %d file(s), freed %.1f MB; kept %d (%.1f MB)",
                stats["removed"], stats["freed"] / 1048576, count, total / 1048576,
            )
    except Exception as exc:
        log.warning("Preview cache prune skipped: %s", exc)
    return stats


def build_preview_file(items: list[dict]) -> tuple[str | None, str | None]:
    """播放列表 → 一份**可随机 seek** 的 faststart mp4（按内容指纹缓存）。

    * 内容没变 → 命中既有文件，零 ffmpeg 开销（只刷新 mtime，LRU 不会误回收）；
    * 内容变了 → 重新 concat 到 `.tmp.mp4` 再 ``os.replace`` 原子发布；
    * 参数一致走 ``-c:v copy``（近乎零开销），不一致才全重编码兜底；
    * 构建前先 :func:`prune_preview_cache`（keep 当前文件）→ 磁盘不会持续累积。

    返回 ``(绝对路径, 缓存文件名)``；无可用片段 / ffmpeg 失败返回 ``(None, None)``。
    """
    try:
        paths = [str(it.get("abs") or "") for it in (items or [])]
        paths = [p for p in paths if p and os.path.isfile(p) and os.path.getsize(p) > 0]
        if not paths:
            return None, None
        ffmpeg = _ffmpeg_bin_or_none()
        if not ffmpeg:
            return None, None
        key = compute_preview_cache_key(items)
        name = f"preview_{key}.mp4"
        dest = _preview_cache_dir() / name

        def _hit() -> tuple[str, str] | None:
            try:
                if dest.is_file() and dest.stat().st_size > 0:
                    os.utime(dest, None)  # 刷新 mtime = 「刚被用过」
                    return str(dest), name
            except OSError:
                pass
            return None

        got = _hit()
        if got:
            return got
        # 并发（预热线程 + 前端请求）同一 key：串行化，后来者等锁再查一次即可命中。
        with _preview_lock(key):
            got = _hit()
            if got:
                return got
            prune_preview_cache(keep={name})
            if not _concat_clips_to(paths, dest, label="Preview file build"):
                return None, None
            return str(dest), name
    except Exception as exc:
        log.warning("Preview file build skipped: %s", exc)
        return None, None


# 最近一次运行下发的播放列表（带绝对路径），供 /preview_file 路由优先使用。
# **内存 + 落盘双份** —— 只放内存的话，ComfyUI 重启 / 页面刷新后缓存没了，就只剩
# 「从任务表反推」这条降级路：它拼不出「上次那条完整时间轴」，段数/时长都对不上，
# 严重时还会把别的提示词分组的同段号素材混进来。落盘后刷新仍拿到上次的完整列表。
_PREVIEW_ITEMS: dict[str, list[dict]] = {}
_PREVIEW_ITEMS_DIRNAME = "minimax_preview_items"
_PREVIEW_ITEMS_KEEP = 20  # 快照文件上限（每个 node_id 一个，很小）


def _sanitize_node_id(node_id: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", str(node_id or "")) or "unknown"


def _preview_items_dir() -> Path:
    base = Path(folder_paths.get_output_directory()) / _PREVIEW_ITEMS_DIRNAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _preview_items_path(node_id: str | None) -> Path | None:
    try:
        return _preview_items_dir() / f"{_sanitize_node_id(node_id)}.json"
    except OSError:
        return None


def _stored_shape_key(node_id: str | None) -> str:
    """已存快照的身份指纹（内存优先，其次落盘）；没有则 ``""``。"""
    try:
        mem = _PREVIEW_ITEMS.get(str(node_id))
        if isinstance(mem, dict):
            return str(mem.get("shape_key") or "")
    except Exception:
        pass
    p = _preview_items_path(node_id)
    if p is None or not p.is_file():
        return ""
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(raw.get("shape_key") or "") if isinstance(raw, dict) else ""


def set_preview_items(
    node_id: str | None, items: list[dict], *, shape_key: str | None = None
) -> None:
    """记住最近一次下发的播放列表（内存 + 落盘快照）。

    ``shape_key`` = 这批条目所属**时间轴**的身份指纹（见 :func:`compute_shape_key`）。
    不传时**沿用已存值** —— 调用方（如「恢复素材」只改一段）不必关心指纹，
    但不能把别人的指纹清掉，否则同一份快照会被判成「不是这条时间轴」而白重展开一次。
    """
    if not node_id or not items:
        return
    try:
        data = [dict(it) for it in items]
    except Exception:
        return
    if shape_key is None:
        key = _stored_shape_key(node_id)
    else:
        key = str(shape_key or "")
    try:
        _PREVIEW_ITEMS[str(node_id)] = {"shape_key": key, "entries": list(data)}
    except Exception:
        pass
    p = _preview_items_path(node_id)
    if p is None:
        return
    try:
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(
                {"ts": time.time(), "shape_key": key, "entries": data},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, p)
        _prune_preview_items(keep=p)
    except Exception as exc:
        log.warning("Preview items save failed: %s", exc)


def _prune_preview_items(keep: Path | None = None) -> None:
    """快照文件按 mtime 只留最近 N 个（含本次写入的那个）。"""
    try:
        base = _preview_items_dir()
        files = [f for f in base.iterdir() if f.is_file() and f.suffix == ".json"]
        if len(files) <= _PREVIEW_ITEMS_KEEP:
            return
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[_PREVIEW_ITEMS_KEEP:]:
            if keep is not None and f == keep:
                continue
            try:
                f.unlink()
            except OSError:
                continue
    except Exception:
        pass


def get_preview_items(node_id: str | None, *, expect_shape_key: str | None = None) -> list[dict]:
    """播放列表：内存 → 落盘快照 → ``[]``（由调用方走任务表降级）。

    快照里的 ``abs`` 可能已被删 —— 缺失段**只补占位片**，绝不去磁盘上按段号另找一个
    顶上（那是另一条时间轴的素材）。这样整条时间轴的段数与时长始终与上次运行时一致。

    ``expect_shape_key`` 给定时只认**同一条时间轴**的快照：存的指纹对不上（或存的是
    没有指纹的旧快照）→ 返回 ``[]``，由调用方按当前形状重新展开。这是「多个工作流
    共用同一个 node id」时防止互相复用的那道门 —— 宁可重展开一次，也绝不把别条
    时间轴的素材当成自己的。
    """
    want = str(expect_shape_key or "")
    try:
        mem = _PREVIEW_ITEMS.get(str(node_id))
    except Exception:
        mem = None
    if isinstance(mem, dict):
        if want and str(mem.get("shape_key") or "") != want:
            return []
        got = mem.get("entries")
        if isinstance(got, list) and got:
            return [dict(it) for it in got]
    elif isinstance(mem, list) and mem:
        # 进程内旧结构（裸列表、无指纹）：要校验时只能判为「来源不明」。
        return [] if want else [dict(it) for it in mem]
    p = _preview_items_path(node_id)
    if p is None or not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    stored_key = str(raw.get("shape_key") or "")
    if want and stored_key != want:
        return []
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        return []
    # 从仍有效的段上探分辨率/帧率，让占位片与真实段参数一致（concat 才能走 copy）。
    fps, w, h = 24.0, 0, 0
    for it in entries:
        pth = str(it.get("abs") or "")
        if pth and os.path.isfile(pth) and os.path.getsize(pth) > 0:
            probe = _probe_video_key(_ffmpeg_bin_or_none(), pth)
            if probe:
                _codec, pw, ph, _pf, pfps = probe
                w, h = int(pw), int(ph)
                if pfps:
                    fps = float(pfps)
            break
    out: list[dict] = []
    for it in entries:
        idx = int(it.get("index", -1))
        frames = int(it.get("frames") or 0)
        pth = str(it.get("abs") or "")
        stored = str(it.get("status") or "").lower()
        if stored == "missing":
            # 🔴 缺失段的 ``abs`` 指向的是**占位片**（``missing_XXXX_*.mp4``），文件当然存在 ——
            # 不能据此判成 ok，否则刷新后 missing 被洗成 ok：页面不再标红、用户以为素材齐了，
            # 而执行时下发（report_director_video）的又是原始 missing，状态自相矛盾。
            # 这里原样沿用快照自带的占位片路径（不重新生成），预览画面不因「谁来读」而改变。
            entry = {"index": idx, "frames": frames, "status": "missing"}
            if pth and os.path.isfile(pth) and os.path.getsize(pth) > 0:
                sub, fn = _view_ref(pth)
                entry["subfolder"] = sub
                entry["filename"] = fn
                entry["abs"] = pth
            else:
                entry = _entry_missing(idx, frames, fps, w, h)
            out.append(entry)
        elif pth and os.path.isfile(pth) and os.path.getsize(pth) > 0:
            out.append(_entry_ok(idx, frames, pth, source=str(it.get("source") or "snapshot")))
        else:
            out.append(_entry_missing(idx, frames, fps, w, h))
    try:
        _PREVIEW_ITEMS[str(node_id)] = {"shape_key": stored_key, "entries": list(out)}
    except Exception:
        pass
    return out


def is_released_poster(tensor, expected_frames: int) -> bool:
    """True when IMAGE slot was replaced by a 1-frame stand-in after mp4 flush."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        return False
    expected = int(expected_frames or 0)
    got = int(tensor.shape[0])
    return expected > 1 and got < expected


def released_output_slots(segment_outputs: list, frame_counts: list[int] | None) -> list[int]:
    """Indexes whose IMAGE slot is a poster; full clip is already on disk."""
    counts = frame_counts or []
    out: list[int] = []
    for pos, tensor in enumerate(segment_outputs):
        expected = int(counts[pos]) if pos < len(counts) else 0
        if is_released_poster(tensor, expected):
            out.append(pos)
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            out.append(pos)
            continue
        # 1x1 / odd-tiny leftovers encode to H.264 that Movies & TV rejects (0x80004005).
        if int(tensor.shape[1]) < 2 or int(tensor.shape[2]) < 2:
            out.append(pos)
    return out
