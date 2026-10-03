"""HTTP routes for MiniMax H3 Director (chunked video upload)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import uuid

import folder_paths
from aiohttp import web
from server import PromptServer

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_upload_chunks")
REF_AUDIO_CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_ref_audio_chunks")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".mts", ".ts"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
_WIN_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WIN_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)
_SAFE_EXT = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_ROUTES_REGISTERED = False


def _safe_basename(name: str) -> str:
    """Keep CJK names; only strip path pieces and Windows-illegal characters."""
    base = os.path.basename(str(name or "upload.bin").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if not _SAFE_EXT.fullmatch(ext):
        ext = ".bin"
    if ext == ".jpeg":
        ext = ".jpg"
    stem = _WIN_ILLEGAL.sub("_", stem).rstrip(" .")[:80]
    if not stem or _WIN_RESERVED.match(stem):
        stem = "upload"
    return f"{stem}{ext}"


def _get_media_exts(kind: str) -> set[str]:
    kind = str(kind or "").strip().lower()
    if kind == "image":
        return IMAGE_EXTS
    if kind == "video":
        return VIDEO_EXTS
    if kind == "audio":
        return AUDIO_EXTS
    if kind == "reference_audio":
        return AUDIO_EXTS | VIDEO_EXTS
    raise ValueError("kind must be image, video, audio or reference_audio")


def _peek_image_size(path: str) -> tuple[int, int]:
    """Read width/height from the image header without decoding pixels."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            return int(w or 0), int(h or 0)
    except Exception:
        return 0, 0


def _list_input_media(kind: str) -> list[dict]:
    input_dir = folder_paths.get_input_directory()
    exts = _get_media_exts(kind)
    peek_video = None
    if kind == "video":
        from ..lib.video_io import peek_video_size as peek_video
    items: list[dict] = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            abs_path = os.path.join(root, name)
            try:
                stat = os.stat(abs_path)
            except OSError:
                continue
            try:
                rel_path = os.path.relpath(abs_path, input_dir).replace("\\", "/")
            except ValueError:
                continue
            if rel_path.startswith(".."):
                continue
            subfolder = os.path.dirname(rel_path).replace("\\", "/")
            if subfolder == ".":
                subfolder = ""
            width, height = (0, 0)
            if ext in IMAGE_EXTS:
                width, height = _peek_image_size(abs_path)
            elif peek_video is not None and ext in VIDEO_EXTS:
                try:
                    width, height = peek_video(abs_path)
                except Exception:
                    width, height = 0, 0
            items.append(
                {
                    "name": name,
                    "fileName": name,
                    "relPath": rel_path,
                    "subfolder": subfolder,
                    "type": "input",
                    "modified": float(stat.st_mtime),
                    "width": width,
                    "height": height,
                    "mediaKind": "video" if ext in VIDEO_EXTS else (
                        "audio" if ext in AUDIO_EXTS else "image"
                    ),
                }
            )
    items.sort(key=lambda item: (-item["modified"], item["relPath"]))
    return items


async def minimax_upload_video_chunk(request):
    try:
        post = await request.post()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid upload: {exc}")

    upload_id = str(post.get("upload_id") or "").strip()
    filename = _safe_basename(post.get("filename"))
    chunk_field = post.get("chunk")
    if not upload_id or chunk_field is None:
        return web.Response(status=400, text="Missing upload_id or chunk.")

    if ".." in upload_id or "/" in upload_id or "\\" in upload_id:
        return web.Response(status=400, text="Invalid upload_id.")

    try:
        chunk_index = int(post.get("chunk_index", 0))
        total_chunks = int(post.get("total_chunks", 1))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid chunk index.")

    if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
        return web.Response(status=400, text="Chunk index out of range.")

    session_dir = os.path.join(CHUNK_ROOT, upload_id)
    os.makedirs(session_dir, exist_ok=True)
    part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")

    with open(part_path, "wb") as out:
        while True:
            block = chunk_field.file.read(1024 * 1024)
            if not block:
                break
            out.write(block)

    if chunk_index + 1 < total_chunks:
        return web.json_response({"status": "ok", "chunk_index": chunk_index})

    input_dir = folder_paths.get_input_directory()
    out_path = os.path.join(input_dir, filename)
    if os.path.exists(out_path):
        stem, ext = os.path.splitext(filename)
        for n in range(1, 1000):
            candidate = f"{stem}_{n}{ext}"
            candidate_path = os.path.join(input_dir, candidate)
            if not os.path.exists(candidate_path):
                out_path = candidate_path
                filename = candidate
                break

    with open(out_path, "wb") as out:
        for i in range(total_chunks):
            part = os.path.join(session_dir, f"{i:06d}.part")
            if not os.path.isfile(part):
                shutil.rmtree(session_dir, ignore_errors=True)
                return web.Response(status=400, text=f"Missing chunk {i}.")
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)

    shutil.rmtree(session_dir, ignore_errors=True)
    log.info("MiniMax H3 Director uploaded video to input/: %s", filename)
    return web.json_response({"name": filename, "subfolder": "", "type": "input"})


def _reference_audio_result(path: str, *, reused: bool, source_kind: str) -> dict:
    name = os.path.basename(path)
    return {
        "name": name,
        "fileName": name,
        "relPath": name,
        "subfolder": "",
        "type": "input",
        "reused": bool(reused),
        "sourceKind": source_kind,
    }


def _files_identical(first: str, second: str) -> bool:
    """Match ComfyUI upload dedupe without assigning content-derived filenames."""
    try:
        if os.path.getsize(first) != os.path.getsize(second):
            return False
        with open(first, "rb") as left, open(second, "rb") as right:
            while True:
                left_block = left.read(4 * 1024 * 1024)
                right_block = right.read(4 * 1024 * 1024)
                if left_block != right_block:
                    return False
                if not left_block:
                    return True
    except OSError:
        return False


def _place_in_input_like_comfy_upload(temp_path: str, filename: str) -> tuple[str, bool]:
    """Use ComfyUI's non-overwrite rule: reuse identical, otherwise append ` (n)`."""
    input_dir = folder_paths.get_input_directory()
    filename = _safe_basename(filename)
    stem, ext = os.path.splitext(filename)
    candidate_name = filename
    index = 1
    while True:
        candidate_path = os.path.join(input_dir, candidate_name)
        if not os.path.exists(candidate_path):
            os.replace(temp_path, candidate_path)
            return candidate_path, False
        if _files_identical(candidate_path, temp_path):
            os.remove(temp_path)
            return candidate_path, True
        candidate_name = f"{stem} ({index}){ext}"
        index += 1


def _prepare_reference_audio(source_path: str, display_name: str) -> dict:
    """Extract a video's first audio stream and place it directly in input/."""
    if not os.path.isfile(source_path) or os.path.getsize(source_path) <= 0:
        raise ValueError("Reference audio source is empty or missing.")
    ext = os.path.splitext(display_name or source_path)[1].lower()
    if ext not in VIDEO_EXTS:
        raise ValueError("Selected source is not a supported video.")

    safe_name = _safe_basename(display_name or os.path.basename(source_path))
    safe_stem = os.path.splitext(safe_name)[0] or "reference_audio"
    output_name = f"{safe_stem}.flac"
    output_dir = folder_paths.get_input_directory()

    from ..lib.audio_io import _ffmpeg_bin

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is unavailable; cannot extract audio from video.")
    tmp_path = os.path.join(output_dir, f".minimax_ref_audio_{uuid.uuid4().hex}.flac")
    args = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        source_path,
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        "-y",
        tmp_path,
    ]
    try:
        result = subprocess.run(args, capture_output=True, check=False)
        if result.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) <= 0:
            error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or "The selected video has no decodable audio stream.")
        output_path, reused = _place_in_input_like_comfy_upload(tmp_path, output_name)
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    return _reference_audio_result(output_path, reused=reused, source_kind="video")


async def minimax_extract_reference_audio(request):
    """Extract an existing input video's audio immediately into input/."""
    try:
        body = await request.json()
        video_file = str(body.get("videoFile") or body.get("relPath") or "").strip()
        if not video_file:
            return web.Response(status=400, text="Missing videoFile.")
        from ..lib.video_io import resolve_video_path

        clip = {
            "videoFile": video_file,
            "fileName": str(body.get("fileName") or os.path.basename(video_file)),
            "subfolder": str(body.get("subfolder") or ""),
            "type": str(body.get("type") or "input"),
        }
        source_path = resolve_video_path(clip)
        if os.path.splitext(source_path)[1].lower() not in VIDEO_EXTS:
            return web.Response(status=400, text="Selected source is not a supported video.")
        result = await asyncio.to_thread(
            _prepare_reference_audio,
            source_path,
            clip["fileName"] or os.path.basename(source_path),
        )
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director reference audio extraction failed: %s", exc)
        return web.Response(status=400, text=str(exc))


async def minimax_prepare_reference_audio_chunk(request):
    """Receive large local audio/video; store audio or extract video audio into input/."""
    session_dir = ""
    try:
        post = await request.post()
        upload_id = str(post.get("upload_id") or "").strip()
        filename = _safe_basename(post.get("filename"))
        chunk_field = post.get("chunk")
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", upload_id) or chunk_field is None:
            return web.Response(status=400, text="Invalid reference audio upload.")
        try:
            chunk_index = int(post.get("chunk_index", 0))
            total_chunks = int(post.get("total_chunks", 1))
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid chunk index.")
        if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
            return web.Response(status=400, text="Chunk index out of range.")
        source_ext = os.path.splitext(filename)[1].lower()
        if source_ext not in AUDIO_EXTS | VIDEO_EXTS:
            return web.Response(status=400, text="Unsupported reference audio source format.")

        session_dir = os.path.join(REF_AUDIO_CHUNK_ROOT, upload_id)
        os.makedirs(session_dir, exist_ok=True)
        part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")
        with open(part_path, "wb") as out:
            while True:
                block = chunk_field.file.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if chunk_index + 1 < total_chunks:
            response = web.json_response({"status": "ok", "chunk_index": chunk_index})
            session_dir = ""
            return response

        source_path = os.path.join(session_dir, filename)
        with open(source_path, "wb") as out:
            for index in range(total_chunks):
                part = os.path.join(session_dir, f"{index:06d}.part")
                if not os.path.isfile(part):
                    raise ValueError(f"Missing chunk {index}.")
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out)
        if source_ext in AUDIO_EXTS:
            output_path, reused = await asyncio.to_thread(
                _place_in_input_like_comfy_upload,
                source_path,
                filename,
            )
            result = _reference_audio_result(output_path, reused=reused, source_kind="audio")
        else:
            result = await asyncio.to_thread(_prepare_reference_audio, source_path, filename)
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director local reference audio preparation failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    finally:
        if session_dir:
            shutil.rmtree(session_dir, ignore_errors=True)


async def minimax_probe_video(request):
    try:
        if request.can_read_body and request.content_type == "application/json":
            body = await request.json()
        else:
            body = dict(request.query)
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid request: {exc}")

    video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
    if not video_file:
        return web.Response(status=400, text="Missing videoFile.")

    from ..lib.video_io import probe_video_clip

    clip = {
        "videoFile": video_file,
        "fileName": os.path.basename(video_file),
        "subfolder": str(body.get("subfolder") or "").strip(),
        "type": str(body.get("type") or "input").strip() or "input",
    }
    try:
        info = probe_video_clip(clip)
    except Exception as exc:
        log.warning("MiniMax H3 Director video probe failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    return web.json_response(info)


async def minimax_list_input_media(request):
    try:
        kind = str(request.query.get("kind") or "").strip().lower()
        if not kind:
            return web.Response(status=400, text="Missing kind.")
        items = _list_input_media(kind)
    except ValueError as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Director list input media failed: %s", exc)
        return web.Response(status=500, text=str(exc))
    return web.json_response({"items": items})


async def minimax_detect_shots(request):
    """Detect shot boundaries with PySceneDetect; return logical cut frames."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    from ..lib.shot_detect import (
        detect_timeline_shot_cuts,
        scenedetect_available,
        scenedetect_install_hint,
    )

    if not scenedetect_available():
        return web.Response(
            status=400,
            text=(
                "PySceneDetect is not installed in ComfyUI's Python "
                f"({__import__('sys').executable}). "
                f"Run: {scenedetect_install_hint()}"
            ),
        )

    try:
        frame_rate = float(body.get("frameRate") or body.get("frame_rate") or 24)
    except (TypeError, ValueError):
        frame_rate = 24.0
    try:
        total_frames = int(body.get("totalFrames") or body.get("total_frames") or 0)
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid totalFrames.")

    sensitivity = str(body.get("sensitivity") or "medium").strip().lower()
    try:
        min_shot_frames = int(body.get("minShotFrames") or body.get("min_shot_frames") or 12)
    except (TypeError, ValueError):
        min_shot_frames = 12

    clips_in = body.get("clips")
    clips: list[dict] = []
    if isinstance(clips_in, list) and clips_in:
        for item in clips_in:
            if not isinstance(item, dict):
                continue
            video_file = str(item.get("videoFile") or item.get("video_file") or "").strip()
            if not video_file:
                continue
            clips.append(
                {
                    "videoFile": video_file,
                    "fileName": os.path.basename(video_file),
                    "subfolder": str(item.get("subfolder") or "").strip(),
                    "type": str(item.get("type") or "input").strip() or "input",
                    "logicalStart": item.get("logicalStart", item.get("logical_start", 0)),
                    "logicalEnd": item.get("logicalEnd", item.get("logical_end", total_frames)),
                    "nativeFps": item.get("nativeFps", item.get("native_fps")),
                }
            )
    else:
        video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
        if not video_file:
            return web.Response(status=400, text="Missing clips[] or videoFile.")
        clips.append(
            {
                "videoFile": video_file,
                "fileName": os.path.basename(video_file),
                "subfolder": str(body.get("subfolder") or "").strip(),
                "type": str(body.get("type") or "input").strip() or "input",
                "logicalStart": 0,
                "logicalEnd": total_frames,
                "nativeFps": body.get("nativeFps", body.get("native_fps")),
            }
        )

    if total_frames <= 0:
        return web.Response(status=400, text="totalFrames must be > 0.")

    try:
        result = detect_timeline_shot_cuts(
            clips,
            frame_rate=frame_rate,
            total_frames=total_frames,
            sensitivity=sensitivity,
            min_shot_frames=min_shot_frames,
        )
    except ImportError as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Director shot detect failed: %s", exc)
        return web.Response(status=400, text=str(exc))

    return web.json_response(result)


async def minimax_first_pass_cache_status(request):
    """Compare stored first-pass metadata with the Director's current inputs."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .external_groups import external_witness_from_timeline_data
        from .plan import build_director_plan
        from .segment_cache import inspect_first_pass_cache

        plan = build_director_plan(
            str(timeline_data),
            global_task_type=str(body.get("task_type") or ""),
            global_prompt=str(body.get("global_prompt") or ""),
            total_frames=int(body.get("total_frames") or 124),
            frame_rate=float(body.get("frame_rate") or 24.0),
            width=int(body.get("width") or 864),
            height=int(body.get("height") or 480),
            ref_max_size=int(body.get("ref_max_size") or 864),
        )
        plan.sample_seed = int(body.get("seed") or 0)
        plan.sample_cfg = float(body.get("cfg") or 1.0)
        plan.sample_steps = int(body.get("steps") or 25)
        plan.sample_sampler = str(body.get("sampler") or "")
        plan.sample_scheduler = str(body.get("scheduler") or "")
        plan.sample_sigmas_linked = bool(body.get("sigmas_linked"))
        plan.sample_shift_video = float(body.get("shift_video") or 12.0)
        plan.sample_shift_audio = float(body.get("shift_audio") or 3.0)
        from .selflift.pack import normalize_selflift_pack
        from .semantic_bridge import normalize_semantic_bridge_pack
        from .refine_pack import normalize_refine_pack

        plan.selflift = normalize_selflift_pack(body.get("selflift"))
        plan.semantic_bridge = normalize_semantic_bridge_pack(body.get("semantic_bridge"))
        plan.refine = normalize_refine_pack(
            body.get("refine"),
            base_width=int(getattr(plan, "width", 0) or 0),
            base_height=int(getattr(plan, "height", 0) or 0),
        )
        # Graph-wired i2v_groups / r2v_groups never reach this route as values,
        # so the panel ships a witness of that wiring inside timeline_data.
        witness = external_witness_from_timeline_data(timeline_data)
        if witness:
            plan.external_groups_witness = witness
        return web.json_response(
            inspect_first_pass_cache(node_id, plan, external_groups=witness)
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director first-pass cache inspection failed: %s", exc)
        return web.json_response(
            {"exists": False, "matches": False, "error": str(exc)},
            status=400,
        )


async def minimax_clear_segment_cache(request):
    """Delete first-pass (.pre.*) or final segment cache files."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    kind = str(body.get("kind") or "final").strip().lower()
    if kind not in {"first_pass", "final", "all"}:
        return web.Response(status=400, text="kind must be first_pass, final or all.")

    try:
        from .segment_cache import clear_segment_cache

        removed = clear_segment_cache(node_id, kind=kind)
        return web.json_response({"removed": removed, "kind": kind})
    except Exception as exc:
        log.warning("MiniMax H3 Director clear segment cache failed: %s", exc)
        return web.Response(status=500, text=str(exc))


async def minimax_latest_seg_export(request=None):
    """返回最近一次分段导出的目录与文件（供前端刷新页面后恢复出片预览）。

    按目录 mtime 取最新一个 ``output/minimax_seg_export/<ts>/``。
    只报每段的 ``seg_XXXX.mp4``；整片不在这里产出 —— 合并成片由出片预览统一提供
    （``minimax_preview_cache/`` 下按内容指纹缓存，见 ``build_preview_file``）。
    """
    try:
        base = os.path.join(folder_paths.get_output_directory(), "minimax_seg_export")
        empty = {"run_dir": "", "files": [], "seg_indexes": []}
        if not os.path.isdir(base):
            return web.json_response(empty)
        runs = sorted(
            (os.path.join(base, name) for name in os.listdir(base)
             if os.path.isdir(os.path.join(base, name))),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if not runs:
            return web.json_response(empty)
        run = runs[0]
        files = sorted(
            name for name in os.listdir(run)
            if name.lower().endswith(".mp4") and os.path.isfile(os.path.join(run, name))
        )
        # 目录里有哪些段（按 seg_XXXX.mp4 文件名）——刷新后素材轴要据此复原，
        # 否则「选择运行」过的时间轴会按全量段去画，播放头错位。
        seg_indexes: list[int] = []
        for name in files:
            m = re.match(r"^seg_(\d{4})\.mp4$", name)
            if m:
                seg_indexes.append(int(m.group(1)))
        return web.json_response({
            "run_dir": os.path.basename(run),
            "files": files,
            "seg_indexes": sorted(seg_indexes),
        })
    except Exception as exc:
        log.warning("MiniMax H3 Director latest_seg_export failed: %s", exc)
        return web.Response(status=500, text=str(exc))


async def minimax_stitch_latest(request=None):
    """出片预览恢复：返回「播放列表」条目（每段一条 /view 引用）。

    预览已改为前端逐段连播、不再落盘拼接 merged_latest.mp4。本路由从任务表取
    每段「最新登记」的有效片段，文件已删 / 0 字节的段用代码生成的占位片补位并标
    missing —— 供前端「出片预览」整条时间轴恢复（删掉一段后刷新，那段仍在素材轴上）。
    """
    try:
        from .segment_mp4_export import build_preview_playlist_from_manifests

        entries = build_preview_playlist_from_manifests()
        return web.json_response({"entries": entries})
    except Exception as exc:
        log.warning("MiniMax H3 Director stitch_latest failed: %s", exc)
        return web.Response(status=500, text=str(exc))


async def minimax_preview_plan(request=None):
    """出片预览恢复（首选路）：前端把**当前时间轴的段形状**送上来，后端按它展开播放列表。

    body::

        {"node_id": 12,
         "task_type": "r2v — 参考主体生视频(Reference to Video)",
         "frame_rate": 24, "width": 640, "height": 480,
         "segments": [{"index": 0, "frames": 124}, ...]}

    返回 ``{"entries": [...]}``：段号在该任务类型的任务表里有有效登记 → 真实片段；
    手动「恢复素材」绑定过 → 绑定文件；其余 → 「第 N 段缺失」占位片（时长按时间轴帧数）。

    为什么必须由前端给形状：后端在没有 plan 时**不知道当前时间轴有几段、每段多少帧**。
    只读任务表 → 段数只等于登记过的段数（16 段的时间轴刷新后只剩 17 秒，缺段全没了）；
    按段号扫盘补 → 会把别条时间轴的素材拼进来（实测 83.38s 被拼成 5:39）。浏览器端
    手上就有这条时间轴，送上来最准，也不用任何猜测。

    成功后顺带把列表记进本进程缓存 + 落盘快照，后续 ``/preview_file`` 直接复用。
    """
    try:
        payload: dict = {}
        if request is not None:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
        node_id = payload.get("node_id")
        shape = payload.get("segments") or payload.get("shape") or []
        if not isinstance(shape, list) or not shape:
            return web.json_response({"entries": []})

        def _num(v) -> float:
            try:
                return float(v)
            except Exception:
                return 0.0

        from .segment_mp4_export import (
            build_preview_playlist_for_shape,
            compute_shape_key,
            set_preview_items,
        )

        task_key = _resolve_task_key_from_payload(payload)
        entries = await asyncio.to_thread(
            build_preview_playlist_for_shape,
            shape,
            task_key=task_key,
            fps=_num(payload.get("frame_rate") or payload.get("fps")),
            width=int(_num(payload.get("width"))),
            height=int(_num(payload.get("height"))),
        )
        if entries and node_id not in (None, ""):
            try:
                # 连同「工作流身份指纹」一起存：多工作流共用同一 node id 时，读的一方按
                # 指纹校验，不会把别条时间轴的快照当成自己的（见 compute_shape_key）。
                set_preview_items(
                    str(node_id), entries, shape_key=compute_shape_key(shape, task_key=task_key)
                )
            except Exception:
                pass
        return web.json_response({"entries": entries, "task": task_key})
    except Exception as exc:
        log.warning("MiniMax H3 Director preview_plan failed: %s", exc)
        return web.Response(status=500, text=str(exc))


async def minimax_preview_file(request=None):
    """出片预览「预览文件」：整条时间轴的段（含缺失段占位片）拼成**一份可随机 seek 的
    faststart mp4**，按内容指纹缓存在 ``output/minimax_preview_cache/``，返回它的
    ``/view`` 引用（前端挂到 ``<video src>``）。

    * 内容未变 → 命中既有文件（零 ffmpeg 开销）；改过任意一段 → 新 key → 新文件；
    * 普通 mp4（非 fMP4）→ 浏览器走 HTTP range 随机 seek，**拖动毫秒级**，duration 正确；
    * 落盘累积由 ``prune_preview_cache`` 兜住：TTL / 数量上限 / 总大小上限 / 半写残留，
      每次构建前回收，且永不删除正在服务的那一份。

    query：
      * ``node_id`` —— 命中该节点的落盘快照（最近一次展开的播放列表）；
      * ``shape``（``"0:124,1:124"``）/ ``task_type`` / ``frame_rate`` —— **当前时间轴的
        段形状**，前端手上就有。有了它才算得出「工作流身份指纹」：多个工作流共用同一个
        node id 时，别条时间轴的快照不会被复用（指纹不符 → 按形状重新展开）。
        不带形状时退回旧行为（快照 → 任务表）。

    取源统一走 ``resolve_preview_entries`` —— 页面预览与「整片产物」同一个函数、同一个
    顺序，保证「页面播的」与「导出的」是同一个文件。
    """
    node_id = None
    shape: list[dict] = []
    task_key = ""
    fps = 0.0
    width = 0
    height = 0
    try:
        if request is not None:
            q = request.rel_url.query
            node_id = q.get("node_id")
            shape = _parse_shape_query(q.get("shape"))
            if q.get("task") or q.get("task_type"):
                task_key = _resolve_task_key_from_payload(
                    {"task": q.get("task") or "", "task_type": q.get("task_type") or ""}
                )
            fps = _query_float(q.get("frame_rate") or q.get("fps"))
            width = int(_query_float(q.get("width")))
            height = int(_query_float(q.get("height")))
    except Exception:
        node_id, shape = None, []
    try:
        from .segment_mp4_export import build_preview_file, resolve_preview_entries

        items, src = await asyncio.to_thread(
            resolve_preview_entries,
            node_id=node_id,
            shape=shape,
            task_key=task_key,
            fps=fps,
            width=width,
            height=height,
        )
        if not items:
            return web.json_response({"filename": "", "subfolder": "", "reason": "no_segment"})
        # concat 可能要几秒（copy 模式通常 <1s）→ 丢线程池，别阻塞 aiohttp 事件循环。
        path, name = await asyncio.to_thread(build_preview_file, items)
        if not path or not name:
            return web.json_response({"filename": "", "subfolder": "", "reason": "build_failed"})
        return web.json_response({
            "filename": name,
            "subfolder": "minimax_preview_cache",
            "source": src,
        })
    except Exception as exc:
        log.warning("MiniMax H3 Director preview_file failed: %s", exc)
        return web.Response(status=500, text=str(exc))


def _resolve_task_key_from_payload(payload: dict) -> str:
    """请求体 → 任务键。前端送的是任务类型标签（如「r2v — 参考主体生视频」），
    与后端登记侧共用 ``resolve_task_key``，避免两边切出不同的键名。"""
    explicit = str(payload.get("task") or "").strip()
    if explicit:
        return explicit
    label = str(payload.get("task_type") or payload.get("taskType") or "").strip()
    if not label:
        return ""
    try:
        from ..lib.task_prompts import resolve_task_key

        return resolve_task_key(label)
    except Exception:
        return label.split(" — ", 1)[0].split(" - ", 1)[0].strip()


def _query_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_shape_query(raw) -> list[dict]:
    """``shape`` 查询参数 → ``[{"index": i, "frames": n}, ...]``。

    紧凑串（前端用，URL 短）::

        "0:124,1:124,8:141"

    也接受 JSON 数组（便于手工调试 / 旧调用方）::

        '[{"index": 0, "frames": 124}, ...]'   或   '[[0, 124], ...]'
    """
    text = str(raw or "").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            data = json.loads(text)
        except Exception:
            return []
        out: list[dict] = []
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict):
                out.append({"index": int(item.get("index", -1)), "frames": int(item.get("frames") or 0)})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append({"index": int(item[0]), "frames": int(item[1] or 0)})
        return [e for e in out if e["index"] >= 0]
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 2:
            continue
        try:
            out.append({"index": int(parts[0]), "frames": int(float(parts[1]))})
        except Exception:
            continue
    return [e for e in out if e["index"] >= 0]


def _resolve_local_media(payload: dict) -> str | None:
    """请求体 → 本机文件路径。支持 ``path``（本机绝对路径）或
    ``filename`` + ``type`` + ``subfolder``（刚通过 upload_chunk 上传的文件）。"""
    raw = str(payload.get("path") or "").strip()
    if raw and os.path.isfile(raw):
        return raw
    name = _safe_basename(payload.get("filename"))
    if not name:
        return None
    kind = str(payload.get("type") or "input").strip().lower()
    if kind == "output":
        base = folder_paths.get_output_directory()
    elif kind == "temp":
        base = folder_paths.get_temp_directory()
    else:
        base = folder_paths.get_input_directory()
    sub = str(payload.get("subfolder") or "").strip().replace("\\", "/").strip("/")
    if sub and (".." in sub.split("/")):
        return None
    cand = os.path.join(base, sub, name) if sub else os.path.join(base, name)
    return cand if os.path.isfile(cand) else None


async def minimax_recover_segment(request):
    """标红段「恢复素材」：把一份现成的 mp4 绑定到时间轴第 N 段。

    为什么需要：任务表按内容指纹（提示词 + 负向 + 参考素材 + 帧数）定位段，改过
    时长/提示词的旧素材指纹必然对不上 —— 那些段在预览里全成了红斜纹占位片，而素材
    其实还在。指纹证明不了的事由人担保：用户指定「这个文件就是第 N 段」。

    body(JSON)：``index``（段号，0 基）、``task_type``（或 ``task``）、``node_id``（可选，
    用于就地替换本进程已下发的播放列表）、来源二选一：``path``（本机绝对路径）
    或 ``filename`` + ``type`` + ``subfolder``（upload_chunk 刚传上来的）。
    """
    try:
        payload = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body.")
    if not isinstance(payload, dict):
        return web.Response(status=400, text="Invalid JSON body.")
    try:
        idx = int(payload.get("index"))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Missing segment index.")
    if idx < 0:
        return web.Response(status=400, text="Invalid segment index.")
    task = _resolve_task_key_from_payload(payload)
    if not task:
        return web.Response(status=400, text="Missing task type.")
    src = _resolve_local_media(payload)
    if not src:
        return web.Response(status=400, text="Media file not found.")
    try:
        from .segment_mp4_export import (
            _entry_ok,
            bind_segment_clip,
            get_preview_items,
            import_recovered_clip,
            set_preview_items,
        )

        dest = await asyncio.to_thread(import_recovered_clip, src, idx)
        if not dest:
            return web.Response(status=500, text="Failed to import clip.")
        record = await asyncio.to_thread(bind_segment_clip, task, idx, dest)
        if not record:
            return web.Response(status=500, text="Failed to bind clip.")
        entry = _entry_ok(idx, int(record.get("frames") or 0), dest, source="bound")
        # 本进程已下发的播放列表也换掉这一段 —— 否则 /preview_file 仍按旧列表拼，
        # 用户会看到「红块变绿了但预览里还是占位片」。
        node_id = payload.get("node_id")
        items = get_preview_items(node_id)
        if items:
            for i, e in enumerate(items):
                if int(e.get("index", -1)) == idx:
                    items[i] = entry
                    break
            set_preview_items(node_id, items)
        return web.json_response(
            {
                "ok": True,
                "task": task,
                "index": idx,
                "frames": int(record.get("frames") or 0),
                "filename": entry.get("filename") or "",
                "subfolder": entry.get("subfolder") or "",
            }
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director recover_segment failed: %s", exc)
        return web.Response(status=500, text=str(exc))


async def minimax_unrecover_segment(request):
    """撤销某段的「恢复素材」绑定（只删 ``bind#<idx>``，不动指纹登记）。"""
    try:
        payload = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON body.")
    if not isinstance(payload, dict):
        return web.Response(status=400, text="Invalid JSON body.")
    try:
        idx = int(payload.get("index"))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Missing segment index.")
    task = _resolve_task_key_from_payload(payload)
    if not task:
        return web.Response(status=400, text="Missing task type.")
    try:
        from .segment_mp4_export import unbind_segment_clip

        removed = await asyncio.to_thread(unbind_segment_clip, task, idx)
        return web.json_response({"ok": True, "removed": bool(removed), "index": idx})
    except Exception as exc:
        log.warning("MiniMax H3 Director unrecover_segment failed: %s", exc)
        return web.Response(status=500, text=str(exc))


def _register_route(routes, method: str, path: str, handler) -> None:
    if hasattr(routes, "add_route"):
        routes.add_route(method, path, handler)
    elif method == "POST" and hasattr(routes, "post"):
        routes.post(path)(handler)
    elif method == "GET" and hasattr(routes, "get"):
        routes.get(path)(handler)
    else:
        raise AttributeError("Unsupported ComfyUI route table API")


def register_routes() -> bool:
    """Register MiniMax H3 Director HTTP routes on the ComfyUI PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    server = PromptServer.instance
    if server is None:
        log.warning("MiniMax H3 Director: PromptServer not ready, HTTP routes not registered")
        return False

    routes = server.routes
    _register_route(routes, "POST", "/minimax/director/upload_chunk", minimax_upload_video_chunk)
    _register_route(
        routes,
        "POST",
        "/minimax/director/extract_reference_audio",
        minimax_extract_reference_audio,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/prepare_reference_audio_chunk",
        minimax_prepare_reference_audio_chunk,
    )
    _register_route(routes, "POST", "/minimax/director/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director/list_input_media", minimax_list_input_media)
    _register_route(routes, "POST", "/minimax/director/detect_shots", minimax_detect_shots)
    _register_route(
        routes,
        "POST",
        "/minimax/director/first_pass_cache_status",
        minimax_first_pass_cache_status,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/clear_segment_cache",
        minimax_clear_segment_cache,
    )
    _register_route(routes, "GET", "/minimax/director/latest_seg_export", minimax_latest_seg_export)
    _register_route(routes, "GET", "/minimax/director/stitch_latest", minimax_stitch_latest)
    _register_route(routes, "POST", "/minimax/director/preview_plan", minimax_preview_plan)
    _register_route(routes, "GET", "/minimax/director/preview_file", minimax_preview_file)
    _register_route(
        routes, "POST", "/minimax/director/recover_segment", minimax_recover_segment
    )
    _register_route(
        routes, "POST", "/minimax/director/unrecover_segment", minimax_unrecover_segment
    )
    from .pack import minimax_download_pack, minimax_export_pack, minimax_import_pack

    _register_route(routes, "POST", "/minimax/director/export_pack", minimax_export_pack)
    _register_route(routes, "GET", "/minimax/director/download_pack", minimax_download_pack)
    _register_route(routes, "POST", "/minimax/director/import_pack", minimax_import_pack)
    from .lora_previews import minimax_lora_preview, minimax_lora_previews

    _register_route(routes, "GET", "/minimax/director/lora_previews", minimax_lora_previews)
    _register_route(routes, "GET", "/minimax/director/lora_preview", minimax_lora_preview)
    _ROUTES_REGISTERED = True
    log.info("MiniMax H3 Director HTTP routes registered")
    return True
