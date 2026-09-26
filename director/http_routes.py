"""HTTP routes for MiniMax H3 Director (chunked video upload)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

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

# --- Merge progress store (polled by the frontend progress bar) ----------------
# key: job_id -> {"percent": float, "message": str, "done": bool, "error": str|None, "ttl": int}
_MERGE_PROGRESS: dict[str, dict] = {}
_MERGE_PROGRESS_STALE_MS = 600_000  # keep an entry for 10 min after creation


def _progress_set(job_id: str, percent: float, message: str, done: bool = False, error: str | None = None) -> None:
    if not job_id:
        return
    _MERGE_PROGRESS[job_id] = {
        "percent": round(max(0.0, min(100.0, float(percent))), 1),
        "message": str(message or ""),
        "done": bool(done),
        "error": error,
        "ttl": int(time.time() * 1000),
    }


def _progress_responder(job_id: str):
    def _rp(percent: float, message: str) -> None:
        _progress_set(job_id, percent, message)
    return _rp


async def minimax_merge_progress(request):
    """GET /minimax/director/merge_progress?job_id=... -> current merge progress."""
    job_id = str(request.query.get("job_id") or "").strip()
    if not job_id:
        return web.json_response({"error": "missing job_id"}, status=400)
    entry = _MERGE_PROGRESS.get(job_id)
    if not entry:
        return web.json_response({"error": "unknown job_id"}, status=404)
    now = int(time.time() * 1000)
    if now - entry.get("ttl", 0) > _MERGE_PROGRESS_STALE_MS:
        _MERGE_PROGRESS.pop(job_id, None)
        return web.json_response({"error": "expired job_id"}, status=404)
    return web.json_response(entry)



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
        from .segment_cache import inspect_first_pass_cache, resolve_project_id

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
        # Different projects must not share segment cache; fall back to node_id
        # when the timeline has no usable projectId (old workflows / parse failure).
        cache_key = resolve_project_id(getattr(plan, "raw", None), node_id)
        return web.json_response(
            inspect_first_pass_cache(cache_key, plan, external_groups=witness)
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

    # Optional project scoping: prefer timeline_data's projectId, then an explicit
    # project_id field, then fall back to node_id. Missing fields are not an error.
    timeline = body.get("timeline_data")
    timeline_dict = timeline if isinstance(timeline, dict) else None
    if timeline_dict is None and isinstance(timeline, str) and timeline.strip():
        try:
            parsed = json.loads(timeline)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            timeline_dict = parsed
    if timeline_dict is not None:
        key_source = timeline_dict
    elif str(body.get("project_id") or "").strip():
        key_source = {"projectId": body.get("project_id")}
    else:
        key_source = None

    try:
        from .segment_cache import clear_segment_cache, resolve_project_id

        cache_key = resolve_project_id(key_source, node_id)
        removed = clear_segment_cache(cache_key, kind=kind)
        return web.json_response({"removed": removed, "kind": kind})
    except Exception as exc:
        log.warning("MiniMax H3 Director clear segment cache failed: %s", exc)
        return web.Response(status=500, text=str(exc))


_SEG_FILE_RE = re.compile(r"^seg_\d{4}\.mp4$", re.I)
# Any per-segment version: ``seg_0001.mp4``(成片/二采) / ``_pre``(一采) /
# ``_p2``(第2轮精修副本) / ``_facepre``(修脸前)。
_SEG_ANY_RE = re.compile(r"^seg_(\d{4})(?:_([A-Za-z0-9]+))?\.mp4$", re.I)
RUN_META_NAME = "_run_meta.json"
# 同一运行目录内的版本优先级：成片(二采) > 一采 > 修脸前。
_VER_FINAL, _VER_PRE, _VER_FACEPRE = 3, 2, 1


def _seg_export_root() -> str:
    """Directory holding per-run segment exports (``<output>/minimax_seg_export``)."""
    return os.path.join(folder_paths.get_output_directory(), "minimax_seg_export")


def _read_run_meta(run_dir: str) -> dict:
    """Run-level metadata written by the node; ``{}`` for legacy folders."""
    try:
        with open(os.path.join(run_dir, RUN_META_NAME), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _version_rank(suffix: str) -> int:
    """Version priority inside one run (higher wins)."""
    tag = str(suffix or "").lower()
    if not tag:
        return _VER_FINAL
    if tag == "facepre":
        return _VER_FACEPRE
    if tag == "pre":
        return _VER_PRE
    if re.fullmatch(r"p\d+", tag):
        return _VER_FINAL  # 第N轮精修副本 == 最终二采的别名
    return _VER_PRE


def _version_kind(suffix: str) -> str:
    tag = str(suffix or "").lower()
    if not tag:
        return "final"
    if tag == "pre":
        return "pre"
    if tag == "facepre":
        return "facepre"
    if re.fullmatch(r"p\d+", tag):
        return "pass"
    return "other"


def _scan_run(run_dir: str) -> dict[int, list[dict]]:
    """All per-segment versions in one run → ``{seg_index: [entry, ...]}``.

    Entries are sorted so ``entries[0]`` is what this run contributes by default
    (成片/二采 first, then 一采, then 修脸前).
    """
    out: dict[int, list[dict]] = {}
    try:
        names = os.listdir(run_dir)
    except OSError:
        return out
    for name in sorted(names):
        m = _SEG_ANY_RE.match(name)
        if not m:
            continue
        abs_path = os.path.join(run_dir, name)
        if not os.path.isfile(abs_path):
            continue
        try:
            stat = os.stat(abs_path)
        except OSError:
            continue
        suffix = (m.group(2) or "").lower()
        out.setdefault(int(m.group(1)), []).append({
            "name": name,
            "suffix": suffix,
            "rank": _version_rank(suffix),
            "mtime": float(stat.st_mtime),
            "size": int(stat.st_size),
        })
    for entries in out.values():
        entries.sort(key=lambda e: (-e["rank"], -e["mtime"], e["name"]))
    return out


def _collect_runs() -> list[dict]:
    """Every run folder holding at least one segment version, oldest → newest.

    Runs may live under the legacy flat layout ``minimax_seg_export/<run>/`` or
    the per-project layout ``minimax_seg_export/<project_id>/<run>/``. The run
    clock that「取最新」compares on is the run folder's mtime, so a later run
    always wins regardless of which project layer it sits under.
    """
    root = _seg_export_root()
    runs: list[dict] = []
    if not os.path.isdir(root):
        return runs

    def scan_run(run_dir: str, run_name: str) -> dict | None:
        seg_map = _scan_run(run_dir)
        if not seg_map:
            return None
        meta = _read_run_meta(run_dir)
        finals = sorted(f for f in os.listdir(run_dir) if _SEG_FILE_RE.match(f))
        return {
            "dir": run_name,
            "path": run_dir,
            "segs": seg_map,
            "projectId": str(meta.get("projectId") or ""),
            "projectName": str(meta.get("projectName") or ""),
            "createdAt": str(meta.get("createdAt") or ""),
            "legacy": not bool(meta),
            "finalCount": len(finals),
            "totalCount": sum(len(v) for v in seg_map.values()),
            "first": finals[0] if finals else None,
            "last": finals[-1] if finals else None,
            "mtime": os.path.getmtime(run_dir),
        }

    for name in os.listdir(root):
        child = os.path.join(root, name)
        if not os.path.isdir(child):
            continue
        # Legacy flat run: seg files live directly under this folder.
        if _scan_run(child):
            run = scan_run(child, name)
            if run:
                runs.append(run)
            continue
        # Project layer: each subdirectory is a run.
        for sub in os.listdir(child):
            sub_dir = os.path.join(child, sub)
            if not os.path.isdir(sub_dir):
                continue
            run = scan_run(sub_dir, f"{name}/{sub}")
            if run:
                runs.append(run)
    runs.sort(key=lambda item: (item["mtime"], item["dir"]))
    return runs


def _public_run(run: dict) -> dict:
    """JSON-safe run summary (drops the internal seg map / absolute path)."""
    return {
        "dir": run["dir"],
        "count": len(run["segs"]),
        "finalCount": run["finalCount"],
        "versions": run["totalCount"],
        "segs": sorted(run["segs"].keys()),
        "first": run["first"],
        "last": run["last"],
        "mtime": run["mtime"],
        "projectId": run["projectId"],
        "projectName": run["projectName"],
        "createdAt": run["createdAt"],
        "legacy": run["legacy"],
    }


def _run_url(run_name: str, file_name: str) -> str:
    from urllib.parse import quote

    sub = quote(f"minimax_seg_export/{run_name}", safe="")
    return f"/view?filename={quote(file_name, safe='')}&subfolder={sub}&type=output"


def _select_latest(runs: list[dict], allowed: set[str] | None = None) -> list[dict]:
    """Per-segment「取最新」: for each segment index, the newest run's best version.

    「最新」is judged by run-folder time first, so a later run that only produced
    一采 still beats an earlier run's 二采（二采新就用二采，一采新就用一采）。
    Inside one run the 成片/二采 always beats that run's own 一采。
    Result is sorted by segment index = time-line order.
    """
    best: dict[int, tuple] = {}
    for run in runs:
        if allowed is not None and run["dir"] not in allowed:
            continue
        for idx, entries in run["segs"].items():
            # 修脸前(_facepre) 是调试中间件，只允许在挑段面板里手动选，不进自动取最新。
            candidates = [e for e in entries if e["suffix"] != "facepre"]
            if not candidates:
                continue
            top = candidates[0]
            score = (run["mtime"], top["rank"], top["mtime"])
            cur = best.get(idx)
            if cur is None or score > cur[0]:
                best[idx] = (score, {
                    "seg": idx,
                    "run": run["dir"],
                    "name": top["name"],
                    "suffix": top["suffix"],
                    "rank": top["rank"],
                    "mtime": top["mtime"],
                    "path": os.path.join(run["path"], top["name"]),
                })
    return [best[idx][1] for idx in sorted(best)]


async def minimax_list_segment_runs(request):
    """List segment-export run folders available for a manual merge."""
    try:
        runs = _collect_runs()
        project_id = str(request.rel_url.query.get("project_id") or "").strip()
        if project_id:
            # Legacy folders carry no owner; keep them visible rather than hiding
            # every pre-upgrade run, but never mix in a *known* different project.
            runs = [
                r for r in runs
                if r["legacy"] or not r["projectId"] or r["projectId"] == project_id
            ]
        public = [_public_run(r) for r in reversed(runs)]
    except Exception as exc:
        log.warning("MiniMax H3 Director list segment runs failed: %s", exc)
        return web.Response(status=500, text=str(exc))
    return web.json_response({"runs": public})


async def minimax_list_segment_versions(request):
    """Every version of every segment across runs, for the manual picker."""
    try:
        runs = _collect_runs()
        project_id = str(request.rel_url.query.get("project_id") or "").strip()
        if project_id:
            runs = [
                r for r in runs
                if r["legacy"] or not r["projectId"] or r["projectId"] == project_id
            ]
        run_path = str(request.rel_url.query.get("run") or "").strip()
        if run_path:
            run_key = run_path.rstrip("/\\")
            runs = [r for r in runs if (r["dir"] or "").rstrip("/\\") == run_key]
        auto = {item["seg"]: item for item in _select_latest(runs)}
        grouped: dict[int, list[dict]] = {}
        for run in runs:
            for idx, entries in run["segs"].items():
                for entry in entries:
                    picked = auto.get(idx) or {}
                    grouped.setdefault(idx, []).append({
                        "run": run["dir"],
                        "name": entry["name"],
                        "suffix": entry["suffix"],
                        "kind": _version_kind(entry["suffix"]),
                        "mtime": entry["mtime"],
                        "size": entry["size"],
                        "url": _run_url(run["dir"], entry["name"]),
                        "auto": bool(
                            picked.get("run") == run["dir"]
                            and picked.get("name") == entry["name"]
                        ),
                    })
        out = []
        for idx in sorted(grouped):
            versions = grouped[idx]
            versions.sort(key=lambda v: (v["run"], -v["mtime"]), reverse=True)
            out.append({"index": idx, "versions": versions})
    except Exception as exc:
        log.warning("MiniMax H3 Director list segment versions failed: %s", exc)
        return web.Response(status=500, text=str(exc))
    return web.json_response({"segments": out})


def _safe_child(root: str, name: str) -> str | None:
    """Resolve ``root/name`` and refuse anything that escapes ``root``."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, name))
    if candidate != root_real and not candidate.startswith(root_real + os.sep):
        return None
    return candidate


async def minimax_delete_segment_export(request):
    """Delete a whole segment-export run folder, or one seg file inside it."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    root = _seg_export_root()
    run_name = str(body.get("run") or "").strip()
    file_name = str(body.get("file") or "").strip()
    run_dir = _safe_child(root, run_name)
    if not run_dir or not os.path.isdir(run_dir):
        return web.Response(status=404, text=f"分段导出目录不存在：{run_name}")

    try:
        if file_name:
            if not _SEG_ANY_RE.match(file_name):
                return web.Response(status=400, text="只能删除 seg_XXXX...mp4 文件。")
            target = _safe_child(run_dir, file_name)
            if not target or not os.path.isfile(target):
                return web.Response(status=404, text=f"分段文件不存在：{file_name}")
            os.remove(target)
            log.info("MiniMax H3 Director deleted segment file: %s", target)
            return web.json_response({
                "ok": True, "removed": "file", "run": run_name, "file": file_name,
            })

        if os.listdir(run_dir):
            shutil.rmtree(run_dir)
        else:
            os.rmdir(run_dir)
        log.info("MiniMax H3 Director deleted segment run dir: %s", run_dir)
        return web.json_response({"ok": True, "removed": "run", "run": run_name})
    except Exception as exc:
        log.warning("MiniMax H3 Director delete segment export failed: %s", exc)
        return web.Response(status=500, text=str(exc))


async def minimax_merge_segments(request):
    """Merge segments into one MP4, in timeline (segment-index) order.

    ``mode``:
      ``latest`` (default) — for every segment index, take the newest version
                  across all runs（二采新就用二采，一采新就用一采）.
      ``single``  — merge one run folder only (legacy behaviour).
      ``custom``  — merge exactly the ``picks`` chosen in the manual picker.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    runs = _collect_runs()
    if not runs:
        return web.Response(status=404, text="没有找到分段导出的目录。")
    by_dir = {r["dir"]: r for r in runs}

    mode = str(body.get("mode") or "").strip().lower()
    run_name = str(body.get("run") or "").strip()
    if mode not in {"latest", "single", "custom"}:
        mode = "single" if run_name else "latest"

    run_filter = body.get("runs")
    allowed = None
    if isinstance(run_filter, list) and run_filter:
        allowed = {str(x) for x in run_filter}

    if mode == "single":
        if not run_name:
            run_name = runs[-1]["dir"]
        if run_name not in by_dir:
            return web.Response(status=404, text=f"分段导出目录不存在：{run_name}")
        chosen = _select_latest([by_dir[run_name]])
    elif mode == "custom":
        raw_picks = body.get("picks")
        if not isinstance(raw_picks, list) or not raw_picks:
            return web.Response(status=400, text="custom 模式需要 picks 列表。")
        chosen = []
        for pick in raw_picks:
            if not isinstance(pick, dict):
                continue
            p_run = str(pick.get("run") or "").strip()
            p_name = str(pick.get("name") or "").strip()
            run = by_dir.get(p_run)
            if run is None or not _SEG_ANY_RE.match(p_name):
                continue
            target = _safe_child(run["path"], p_name)
            if not target or not os.path.isfile(target):
                continue
            seg_idx = next(
                (i for i, es in run["segs"].items() if any(e["name"] == p_name for e in es)),
                None,
            )
            if seg_idx is None:
                continue
            chosen.append({
                "seg": seg_idx,
                "run": p_run,
                "name": p_name,
                "suffix": str(pick.get("suffix") or ""),
                "rank": _version_rank(str(pick.get("suffix") or "")),
                "mtime": os.path.getmtime(target),
                "path": target,
            })
        if not chosen:
            return web.Response(status=400, text="所选的段都不可用。")
        chosen.sort(key=lambda c: (c["seg"], c["mtime"]))
    else:
        chosen = _select_latest(runs, allowed)

    if not chosen:
        return web.Response(status=404, text="没有可用于合并的分段文件。")

    try:
        fps = float(body.get("fps") or 24.0)
    except (TypeError, ValueError):
        fps = 24.0
    if fps <= 0:
        fps = 24.0
    seam_blending = body.get("seam_blending") is not False
    continuity = body.get("continuity") is not False

    from .deferred_merge import deferred_merge_with_seam_reencode

    job_id = str(body.get("job_id") or "").strip()
    if job_id:
        _progress_set(job_id, 0.0, "init")
        _progress_responder_fn = _progress_responder(job_id)
    else:
        _progress_responder_fn = None

    try:
        result = await asyncio.to_thread(
            deferred_merge_with_seam_reencode,
            segment_mp4_paths=[Path(c["path"]) for c in chosen],
            fps=fps,
            seam_blending_enabled=seam_blending,
            continuity_enabled=continuity,
            preview_only_frames=1,
            progress_reporter=_progress_responder_fn,
        )
        if job_id:
            _progress_set(job_id, 100.0, "done", done=True)
    except Exception as exc:
        if job_id:
            _progress_set(job_id, 100.0, "error", done=True, error=str(exc))
        log.warning("MiniMax H3 Director manual merge failed: %s", exc)
        return web.Response(status=500, text=str(exc)[-1200:])

    merged_path = Path(result.merged_video_path)
    if not merged_path.exists():
        return web.Response(status=500, text="合并完成但没有生成输出文件。")

    out_fps = float(result.fps or fps)
    total_frames = int(result.total_frames or 0)
    sources = [
        {
            "seg": c["seg"],
            "segNo": c["seg"] + 1,
            "run": c["run"],
            "name": c["name"],
            "kind": _version_kind(c["suffix"]),
        }
        for c in chosen
    ]
    used_runs = sorted({c["run"] for c in chosen})
    return web.json_response({
        "ok": True,
        "mode": mode,
        "run": run_name if mode == "single" else ", ".join(used_runs),
        "runs": used_runs,
        "segments": len(chosen),
        "sources": sources,
        "preCount": sum(1 for s in sources if s["kind"] == "pre"),
        "firstSeg": chosen[0]["seg"] + 1,
        "lastSeg": chosen[-1]["seg"] + 1,
        "name": merged_path.name,
        "subfolder": merged_path.parent.name,
        "path": str(merged_path),
        "mtime": os.path.getmtime(merged_path),
        "total_frames": total_frames,
        "duration_s": (total_frames / out_fps) if out_fps > 0 else 0.0,
        "fps": out_fps,
        "sample_rate": int(result.sample_rate or 0),
    })


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
    _register_route(
        routes,
        "GET",
        "/minimax/director/list_segment_runs",
        minimax_list_segment_runs,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/merge_segments",
        minimax_merge_segments,
    )
    _register_route(
        routes,
        "GET",
        "/minimax/director/merge_progress",
        minimax_merge_progress,
    )
    _register_route(
        routes,
        "GET",
        "/minimax/director/list_segment_versions",
        minimax_list_segment_versions,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/list_segment_versions",
        minimax_list_segment_versions,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/delete_segment_export",
        minimax_delete_segment_export,
    )
    from .pack import minimax_download_pack, minimax_export_pack, minimax_import_pack

    _register_route(routes, "POST", "/minimax/director/export_pack", minimax_export_pack)
    _register_route(routes, "GET", "/minimax/director/download_pack", minimax_download_pack)
    _register_route(routes, "POST", "/minimax/director/import_pack", minimax_import_pack)
    _ROUTES_REGISTERED = True
    log.info("MiniMax H3 Director HTTP routes registered")
    return True
