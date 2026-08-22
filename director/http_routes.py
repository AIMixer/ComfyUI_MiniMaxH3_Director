"""HTTP routes for MiniMax H3 Director (chunked video upload)."""

from __future__ import annotations

import logging
import os
import re
import shutil

import folder_paths
from aiohttp import web
from server import PromptServer

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_upload_chunks")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".mts", ".ts"}
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
    raise ValueError("kind must be image or video")


def _list_input_media(kind: str) -> list[dict]:
    input_dir = folder_paths.get_input_directory()
    exts = _get_media_exts(kind)
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
            rel_path = os.path.relpath(abs_path, input_dir).replace("\\", "/")
            subfolder = os.path.dirname(rel_path).replace("\\", "/")
            if subfolder == ".":
                subfolder = ""
            items.append(
                {
                    "name": name,
                    "fileName": name,
                    "relPath": rel_path,
                    "subfolder": subfolder,
                    "type": "input",
                    "modified": float(stat.st_mtime),
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
    _register_route(routes, "POST", "/minimax/director/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director/list_input_media", minimax_list_input_media)
    _register_route(routes, "POST", "/minimax/director/detect_shots", minimax_detect_shots)
    _ROUTES_REGISTERED = True
    log.info("MiniMax H3 Director HTTP routes registered")
    return True