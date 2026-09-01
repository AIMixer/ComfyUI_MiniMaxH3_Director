"""MiniMax H3 Director pack (script + assets) import/export.

Zip layout uses ASCII paths that match the English UI:
  shared_params/  Asset group folders  PictureN / VideoN / AudioN
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import folder_paths
from aiohttp import web

from ..lib.task_prompts import resolve_task_key, task_type_option_label, TASK_PROMPT_BY_KEY

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.pack")

PACK_FORMAT = "minimax-h3-director-pack"
PACK_VERSION = 1
PACK_PREFIXES = ("shared_params/", "asset_groups/", "source_video/", "extra/")
ASCII_PATH_RE = re.compile(r"^[A-Za-z0-9_./]+$")
SAFE_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")
PICTURE_FILE_RE = re.compile(r"^Picture([1-9])(\.[A-Za-z0-9]{1,8})$", re.I)
VIDEO_FILE_RE = re.compile(r"^Video([1-3])(\.[A-Za-z0-9]{1,8})$", re.I)
AUDIO_FILE_RE = re.compile(r"^Audio([1-3])(\.[A-Za-z0-9]{1,8})$", re.I)
START_FILE_RE = re.compile(r"^start(\.[A-Za-z0-9]{1,8})$", re.I)
END_FILE_RE = re.compile(r"^end(\.[A-Za-z0-9]{1,8})$", re.I)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".mts", ".ts"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS

MAX_UNCOMPRESSED = 16 * 1024 * 1024 * 1024
MAX_ZIP_ENTRIES = 8000
MAX_SINGLE_FILE = 8 * 1024 * 1024 * 1024
PACK_EXPORT_TTL_SEC = 60 * 60
ZIP_STREAM_CHUNK = 1024 * 1024
# POST JSON already works; keep small packs on that path instead of FileResponse.
INLINE_JSON_MAX = 48 * 1024 * 1024

IMAGE_KEYS = ("imageFile", "image_file")
AUDIO_KEYS = ("audioFile", "audio_file")
VIDEO_KEYS = ("videoFile", "video_file")
PREVIEW_KEYS = ("previewImageFile", "preview_image_file")
PAIRED_AUDIO_KEYS = ("pairedAudioFile", "paired_audio_file")


def _pack_export_root() -> Path:
    root = Path(folder_paths.get_temp_directory()) / "minimax_director_pack_export"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _purge_pack_exports(*, keep: str | None = None) -> None:
    """Remove export zips older than TTL. Never delete a file that may still be downloading."""
    root = _pack_export_root()
    now = time.time()
    for path in root.glob("*.mmxpack.zip"):
        if keep and path.name == keep:
            continue
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age >= PACK_EXPORT_TTL_SEC:
            _unlink_quiet(path)


def _is_under_dir(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _zip_headers(download_name: str, size: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/zip",
        "Content-Disposition": f'attachment; filename="{download_name}"',
        "Content-Length": str(int(size)),
        "Cache-Control": "no-store",
    }
    if extra:
        headers.update(extra)
    return headers


async def _send_zip_file(
    request,
    path: Path,
    download_name: str,
    extra_headers: dict[str, str] | None = None,
    *,
    unlink_after: bool = False,
):
    """Stream a zip without aiohttp FileResponse.

    FileResponse/sendfile on Windows often yields HTTP 200 with an empty body,
    which browsers save as a 0-byte .zip that unzip tools reject.
    """
    try:
        size = int(path.stat().st_size)
    except OSError:
        return web.Response(status=404, text="Pack not found.")
    if size <= 0:
        return web.Response(status=404, text="Pack not found.")
    headers = _zip_headers(download_name, size, extra_headers)
    resp = web.StreamResponse(status=200, headers=headers)
    await resp.prepare(request)
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(ZIP_STREAM_CHUNK)
                if not chunk:
                    break
                await resp.write(chunk)
        await resp.write_eof()
        return resp
    finally:
        if unlink_after:
            _unlink_quiet(path)


def _input_dir() -> Path:
    return Path(folder_paths.get_input_directory())


def _is_ascii_pack_path(rel: str) -> bool:
    s = str(rel or "").replace("\\", "/").strip()
    if not s or s.startswith("/") or ".." in s.split("/"):
        return False
    return bool(ASCII_PATH_RE.fullmatch(s))


def _safe_ext(path: Path | str, fallback: str = ".bin") -> str:
    ext = Path(str(path)).suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if not SAFE_EXT_RE.fullmatch(ext) or ext not in MEDIA_EXTS | {".json"}:
        return fallback
    return ext


def _posix(rel: Path | str) -> str:
    return str(rel).replace("\\", "/").lstrip("/")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _media_bases() -> list[tuple[str, Path]]:
    bases: list[tuple[str, Path]] = [("input", _input_dir())]
    try:
        bases.append(("output", Path(folder_paths.get_output_directory())))
    except Exception:
        pass
    try:
        bases.append(("temp", Path(folder_paths.get_temp_directory())))
    except Exception:
        pass
    return bases


def resolve_media_path(
    rel: str,
    *,
    subfolder: str = "",
    type_name: str = "input",
) -> Path | None:
    raw = str(rel or "").replace("\\", "/").strip()
    if not raw:
        return None
    sub = str(subfolder or "").replace("\\", "/").strip().strip("/")
    if sub and "/" not in raw and not raw.startswith(sub + "/"):
        raw = f"{sub}/{raw}"
    type_name = str(type_name or "input").strip() or "input"
    candidates: list[Path] = []
    for kind, base in _media_bases():
        if kind == type_name or not type_name:
            candidates.append(base / raw.replace("/", os.sep))
    for _kind, base in _media_bases():
        p = base / raw.replace("/", os.sep)
        if p not in candidates:
            candidates.append(p)
        candidates.append(base / Path(raw).name)
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _item_rel(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        val = item.get(key)
        if val:
            return str(val).replace("\\", "/").strip()
    return ""


def _copy_file(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    shutil.copy2(src, dest)
    return dest


def _set_media_fields(item: dict, pack_rel: str, primary_key: str) -> None:
    name = Path(pack_rel).name
    parent = _posix(Path(pack_rel).parent)
    item[primary_key] = pack_rel
    if primary_key == "imageFile":
        item.pop("image_file", None)
        item["fileName"] = name
    elif primary_key == "audioFile":
        item.pop("audio_file", None)
        item["fileName"] = name
    elif primary_key == "videoFile":
        item.pop("video_file", None)
        item["fileName"] = name
    item["type"] = "input"
    item["subfolder"] = "" if parent in (".", "") else parent
    item["imageB64"] = ""
    item.pop("previewB64", None)


def _rewrite_one(
    item: dict | None,
    keys: tuple[str, ...],
    dest_rel: str,
    staging: Path,
    missing: list[str],
    *,
    dry_run: bool,
    sizes: list[int],
) -> str | None:
    if not isinstance(item, dict):
        return None
    rel = _item_rel(item, keys)
    if not rel:
        return None
    if rel.replace("\\", "/").startswith(PACK_PREFIXES):
        return rel.replace("\\", "/")
    src = resolve_media_path(rel, subfolder=str(item.get("subfolder") or ""), type_name=str(item.get("type") or "input"))
    if src is None:
        missing.append(rel)
        for key in keys:
            if key in item:
                item[key] = ""
        return None
    try:
        sizes.append(int(src.stat().st_size))
    except OSError:
        sizes.append(0)
    if not dry_run:
        _copy_file(src, staging / dest_rel.replace("/", os.sep))
    _set_media_fields(item, dest_rel, keys[0])
    return dest_rel


def _slot_index(item: dict, fallback: int) -> int:
    raw = item.get("index", item.get("slot", fallback))
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = fallback
    return n


def _rewrite_image_list(refs: list, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(refs, list):
        return
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        idx = _slot_index(ref, i)
        if idx < 0 or idx > 8:
            continue
        src = resolve_media_path(
            _item_rel(ref, IMAGE_KEYS),
            subfolder=str(ref.get("subfolder") or ""),
            type_name=str(ref.get("type") or "input"),
        )
        ext = _safe_ext(src or _item_rel(ref, IMAGE_KEYS), ".png")
        if ext not in IMAGE_EXTS:
            ext = ".png"
        dest = f"{folder}/Picture{idx + 1}{ext}"
        _rewrite_one(ref, IMAGE_KEYS, dest, staging, missing, dry_run=dry_run, sizes=sizes)


def _rewrite_audio_list(refs: list, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(refs, list):
        return
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        idx = _slot_index(ref, i)
        if idx < 0 or idx > 2:
            continue
        src = resolve_media_path(
            _item_rel(ref, AUDIO_KEYS),
            subfolder=str(ref.get("subfolder") or ""),
            type_name=str(ref.get("type") or "input"),
        )
        ext = _safe_ext(src or _item_rel(ref, AUDIO_KEYS), ".wav")
        if ext not in AUDIO_EXTS:
            ext = ".wav"
        dest = f"{folder}/Audio{idx + 1}{ext}"
        _rewrite_one(ref, AUDIO_KEYS, dest, staging, missing, dry_run=dry_run, sizes=sizes)


def _rewrite_video_list(refs: list, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(refs, list):
        return
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        idx = _slot_index(ref, i)
        if idx < 0 or idx > 2:
            continue
        src = resolve_media_path(
            _item_rel(ref, VIDEO_KEYS),
            subfolder=str(ref.get("subfolder") or ""),
            type_name=str(ref.get("type") or "input"),
        )
        ext = _safe_ext(src or _item_rel(ref, VIDEO_KEYS), ".mp4")
        if ext not in VIDEO_EXTS:
            ext = ".mp4"
        dest = f"{folder}/Video{idx + 1}{ext}"
        _rewrite_one(ref, VIDEO_KEYS, dest, staging, missing, dry_run=dry_run, sizes=sizes)
        preview_src = resolve_media_path(
            _item_rel(ref, PREVIEW_KEYS),
            subfolder=str(ref.get("subfolder") or ""),
            type_name=str(ref.get("type") or "input"),
        )
        if preview_src or _item_rel(ref, PREVIEW_KEYS):
            pext = _safe_ext(preview_src or _item_rel(ref, PREVIEW_KEYS), ".jpg")
            if pext not in IMAGE_EXTS:
                pext = ".jpg"
            _rewrite_one(
                ref, PREVIEW_KEYS, f"{folder}/Video{idx + 1}_preview{pext}",
                staging, missing, dry_run=dry_run, sizes=sizes,
            )
        if _item_rel(ref, PAIRED_AUDIO_KEYS):
            aext = _safe_ext(_item_rel(ref, PAIRED_AUDIO_KEYS), ".flac")
            if aext not in AUDIO_EXTS:
                aext = ".flac"
            _rewrite_one(
                ref, PAIRED_AUDIO_KEYS, f"{folder}/Video{idx + 1}_audio{aext}",
                staging, missing, dry_run=dry_run, sizes=sizes,
            )


def _rewrite_image_ref(item: dict | None, dest_stem: str, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(item, dict):
        return
    rel = _item_rel(item, IMAGE_KEYS)
    if not rel:
        return
    src = resolve_media_path(rel, subfolder=str(item.get("subfolder") or ""), type_name=str(item.get("type") or "input"))
    ext = _safe_ext(src or rel, ".jpg")
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
    _rewrite_one(item, IMAGE_KEYS, f"{folder}/{dest_stem}{ext}", staging, missing, dry_run=dry_run, sizes=sizes)


def _rewrite_video_media(item: dict | None, dest_stem: str, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(item, dict):
        return
    rel = _item_rel(item, VIDEO_KEYS) or str(item.get("fileName") or "").replace("\\", "/").strip()
    if not rel:
        return
    src = resolve_media_path(rel, subfolder=str(item.get("subfolder") or ""), type_name=str(item.get("type") or "input"))
    ext = _safe_ext(src or rel, ".mp4")
    if ext not in VIDEO_EXTS:
        ext = ".mp4"
    _rewrite_one(item, VIDEO_KEYS, f"{folder}/{dest_stem}{ext}", staging, missing, dry_run=dry_run, sizes=sizes)


def _group_json(seg: dict) -> dict:
    out = {
        "id": seg.get("id") or "",
        "prompt": seg.get("prompt") or "",
        "negativePrompt": seg.get("negativePrompt") or "",
        "durationSec": seg.get("durationSec"),
        "frameCount": seg.get("frameCount") if seg.get("frameCount") is not None else seg.get("length"),
        "length": seg.get("length") if seg.get("length") is not None else seg.get("frameCount"),
        "start": seg.get("start"),
        "taskType": seg.get("taskType") or "",
        "refs": seg.get("refs") or [],
        "refAudios": seg.get("refAudios") or seg.get("ref_audios") or [],
        "refVideos": seg.get("refVideos") or seg.get("ref_videos") or [],
        "continuityFromPrev": seg.get("continuityFromPrev", seg.get("continuity_from_prev")),
        "refImageSize": seg.get("refImageSize") or seg.get("ref_image_size"),
    }
    if isinstance(seg.get("genImage"), dict):
        out["genImage"] = {
            "imageFile": seg["genImage"].get("imageFile") or "",
            "fileName": seg["genImage"].get("fileName") or "",
        }
    if seg.get("imageFile"):
        out["imageFile"] = seg.get("imageFile")
    for key in ("startImage", "endImage"):
        val = seg.get(key)
        if isinstance(val, dict) and (val.get("imageFile") or val.get("image_file")):
            out[key] = {
                "imageFile": val.get("imageFile") or val.get("image_file") or "",
                "width": val.get("width") or 0,
                "height": val.get("height") or 0,
            }
        else:
            out[key] = None
    return out


def _explode_card(card: dict, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    _rewrite_image_list(card.get("refs") or [], folder, staging, missing, dry_run, sizes)
    _rewrite_audio_list(card.get("refAudios") or card.get("ref_audios") or [], folder, staging, missing, dry_run, sizes)
    _rewrite_video_list(card.get("refVideos") or card.get("ref_videos") or [], folder, staging, missing, dry_run, sizes)
    if isinstance(card.get("genImage"), dict):
        _rewrite_image_ref(card["genImage"], "start", folder, staging, missing, dry_run, sizes)
        if card["genImage"].get("imageFile"):
            card["imageFile"] = card["genImage"]["imageFile"]
    elif card.get("imageFile"):
        dummy = {"imageFile": card.get("imageFile"), "subfolder": card.get("subfolder") or "", "type": card.get("type") or "input"}
        _rewrite_image_ref(dummy, "start", folder, staging, missing, dry_run, sizes)
        card["imageFile"] = dummy.get("imageFile") or card.get("imageFile")
    _rewrite_image_ref(card.get("startImage"), "start", folder, staging, missing, dry_run, sizes)
    _rewrite_image_ref(card.get("endImage"), "end", folder, staging, missing, dry_run, sizes)


def _ascii_extra_name(src: Path, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", src.stem).strip("_")[:40] or "media"
    digest = hashlib.sha1(str(src).encode("utf-8", "replace")).hexdigest()[:8]
    if not re.fullmatch(r"[A-Za-z0-9_]+", stem):
        stem = "media"
    ext = _safe_ext(src, ".bin")
    name = f"{stem}_{digest}{ext}"
    n = 1
    while name in used:
        name = f"{stem}_{digest}_{n}{ext}"
        n += 1
    used.add(name)
    return f"extra/{name}"


def _walk_rewrite_leftovers(
    obj: Any,
    staging: Path,
    missing: list[str],
    dry_run: bool,
    sizes: list[int],
    extra_used: set[str],
) -> None:
    if isinstance(obj, list):
        for item in obj:
            _walk_rewrite_leftovers(item, staging, missing, dry_run, sizes, extra_used)
        return
    if not isinstance(obj, dict):
        return
    for keys, fallback_ext, allowed in (
        (IMAGE_KEYS, ".png", IMAGE_EXTS),
        (AUDIO_KEYS, ".wav", AUDIO_EXTS),
        (VIDEO_KEYS, ".mp4", VIDEO_EXTS),
        (PREVIEW_KEYS, ".jpg", IMAGE_EXTS),
        (PAIRED_AUDIO_KEYS, ".flac", AUDIO_EXTS),
    ):
        rel = _item_rel(obj, keys)
        if not rel or rel.replace("\\", "/").startswith(PACK_PREFIXES):
            continue
        src = resolve_media_path(rel, subfolder=str(obj.get("subfolder") or ""), type_name=str(obj.get("type") or "input"))
        ext = _safe_ext(src or rel, fallback_ext)
        if ext not in allowed:
            ext = fallback_ext
        dest = _ascii_extra_name(src or Path(rel), extra_used)
        # keep intended extension
        dest = str(Path(dest).with_suffix(ext)).replace("\\", "/")
        extra_used.add(Path(dest).name)
        _rewrite_one(obj, keys, dest, staging, missing, dry_run=dry_run, sizes=sizes)
    for key, val in obj.items():
        if key in IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS + PREVIEW_KEYS + PAIRED_AUDIO_KEYS:
            continue
        if isinstance(val, (dict, list)):
            _walk_rewrite_leftovers(val, staging, missing, dry_run, sizes, extra_used)


def _task_combo(value: str) -> str:
    key = resolve_task_key(value or "t2v")
    spec = TASK_PROMPT_BY_KEY.get(key)
    if spec is None or key == "default":
        spec = TASK_PROMPT_BY_KEY["t2v"]
        key = "t2v"
    return task_type_option_label(spec)


def build_export_pack(timeline: dict, widgets: dict | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    if not isinstance(timeline, dict):
        raise ValueError("timeline must be an object")
    data = copy.deepcopy(timeline)
    widgets = widgets if isinstance(widgets, dict) else {}
    missing: list[str] = []
    sizes: list[int] = []
    staging: Path | None = None
    if not dry_run:
        staging = Path(tempfile.mkdtemp(prefix="mmx_pack_"))

    global_block = data.setdefault("global", {})
    if not isinstance(global_block, dict):
        global_block = {}
        data["global"] = global_block
    shared_folder = "shared_params"
    _rewrite_image_list(global_block.get("refs") or [], shared_folder, staging or Path("."), missing, dry_run, sizes)
    _rewrite_audio_list(global_block.get("refAudios") or global_block.get("ref_audios") or [], shared_folder, staging or Path("."), missing, dry_run, sizes)
    _rewrite_video_list(global_block.get("refVideos") or global_block.get("ref_videos") or [], shared_folder, staging or Path("."), missing, dry_run, sizes)
    if isinstance(global_block.get("referenceVideo"), dict):
        _rewrite_video_media(global_block["referenceVideo"], "reference_video", shared_folder, staging or Path("."), missing, dry_run, sizes)
    if isinstance(global_block.get("genImage"), dict):
        _rewrite_image_ref(global_block["genImage"], "start", shared_folder, staging or Path("."), missing, dry_run, sizes)

    task_key = resolve_task_key(
        str(widgets.get("task_type") or widgets.get("taskType") or global_block.get("taskType") or "t2v")
    )
    cards: list[dict] = []
    if task_key == "fl2v" and isinstance(data.get("shots"), list) and data["shots"]:
        cards = [c for c in data["shots"] if isinstance(c, dict)]
    elif isinstance(data.get("segments"), list) and data["segments"]:
        cards = [c for c in data["segments"] if isinstance(c, dict)]

    for i, card in enumerate(cards):
        folder = f"asset_groups/{i + 1:02d}"
        _explode_card(card, folder, staging or Path("."), missing, dry_run, sizes)

    _rewrite_video_media(data.get("video") if isinstance(data.get("video"), dict) else None, "clip_1", "source_video", staging or Path("."), missing, dry_run, sizes)
    clips = data.get("videoClips")
    if isinstance(clips, list):
        for i, clip in enumerate(clips):
            stem = "clip_1" if i == 0 else f"clip_{i + 1}"
            _rewrite_video_media(clip if isinstance(clip, dict) else None, stem, "source_video", staging or Path("."), missing, dry_run, sizes)

    extra_used: set[str] = set()
    _walk_rewrite_leftovers(data, staging or Path("."), missing, dry_run, sizes, extra_used)

    total_bytes = int(sum(sizes))
    unique_missing = []
    seen_m: set[str] = set()
    for item in missing:
        if item not in seen_m:
            seen_m.add(item)
            unique_missing.append(item)

    pack_meta = {
        "format": PACK_FORMAT,
        "formatVersion": PACK_VERSION,
        "taskType": task_key,
        "widgets": {
            k: widgets[k]
            for k in ("steps", "sampler", "scheduler", "cfg", "shift_video", "shift_audio", "seed", "task_type")
            if k in widgets
        },
        "output": data.get("output") if isinstance(data.get("output"), dict) else {},
    }
    shared_json = {
        "commonEnabled": bool(global_block.get("commonEnabled") or global_block.get("common_enabled")),
        "commonCollapsed": bool(global_block.get("commonCollapsed") or global_block.get("common_collapsed")),
        "prompt": global_block.get("prompt") or "",
        "refs": global_block.get("refs") or [],
        "refAudios": global_block.get("refAudios") or global_block.get("ref_audios") or [],
        "refVideos": global_block.get("refVideos") or global_block.get("ref_videos") or [],
    }

    result: dict[str, Any] = {
        "missing": unique_missing,
        "totalBytes": total_bytes,
        "fileCount": len(sizes),
        "taskType": task_key,
    }
    if dry_run:
        return result

    assert staging is not None
    try:
        _write_json(staging / "pack.json", pack_meta)
        _write_json(staging / "shared_params" / "shared_params.json", shared_json)
        (staging / "shared_params").mkdir(exist_ok=True)
        (staging / "asset_groups").mkdir(exist_ok=True)
        for i, card in enumerate(cards):
            folder = staging / "asset_groups" / f"{i + 1:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            _write_json(folder / "group.json", _group_json(card))
        _write_json(staging / "timeline.json", data)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        download_name = f"MiniMaxH3Director-{task_key}-{stamp}.mmxpack.zip"
        stored_name = f"{uuid.uuid4().hex}.mmxpack.zip"
        _purge_pack_exports(keep=stored_name)
        zip_path = _pack_export_root() / stored_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in staging.rglob("*"):
                if not path.is_file():
                    continue
                rel = _posix(path.relative_to(staging))
                if not _is_ascii_pack_path(rel):
                    raise ValueError(f"Non-ASCII pack path: {rel}")
                zf.write(path, rel)
        zip_bytes = int(zip_path.stat().st_size)
        if zip_bytes <= 0:
            _unlink_quiet(zip_path)
            raise ValueError("Export produced an empty pack zip.")
        result["filename"] = stored_name
        result["downloadName"] = download_name
        result["bytes"] = zip_bytes
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_zip_entry(name: str, info: zipfile.ZipInfo, uncompressed_total: int) -> str:
    rel = name.replace("\\", "/").strip()
    if not rel or rel.endswith("/"):
        return ""
    if rel.startswith("__MACOSX/") or rel.startswith("."):
        return ""
    parts = [p for p in rel.split("/") if p]
    if any(p == ".." for p in parts):
        raise ValueError(f"Unsafe path in pack: {name}")
    if any(p.startswith(".") for p in parts):
        return ""
    if not _is_ascii_pack_path(rel):
        raise ValueError(f"Pack paths must be ASCII: {name}")
    if info.file_size > MAX_SINGLE_FILE:
        raise ValueError("A file in the pack exceeds the size limit.")
    if uncompressed_total + info.file_size > MAX_UNCOMPRESSED:
        raise ValueError("Pack uncompressed size exceeds the limit.")
    return rel


def extract_pack_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise ValueError("Pack has too many files.")
        total = 0
        kept: list[tuple[str, zipfile.ZipInfo]] = []
        for info in infos:
            rel = _validate_zip_entry(info.filename, info, total)
            if not rel:
                continue
            total += max(0, int(info.file_size or 0))
            kept.append((rel, info))
        if not kept:
            raise ValueError("Pack zip is empty.")
        for rel, info in kept:
            target = dest / rel.replace("/", os.sep)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _scan_slot_files(folder: Path) -> dict[str, list[dict]]:
    refs: list[dict] = []
    audios: list[dict] = []
    videos: list[dict] = []
    start = None
    end = None
    if not folder.is_dir():
        return {"refs": refs, "refAudios": audios, "refVideos": videos, "startImage": start, "endImage": end}
    pack_folder = _posix(folder.name if folder.parent.name != "asset_groups" else f"asset_groups/{folder.name}")
    if folder.name == "shared_params":
        pack_folder = "shared_params"
    elif folder.parent.name == "asset_groups":
        pack_folder = f"asset_groups/{folder.name}"
    for path in folder.iterdir():
        if not path.is_file():
            continue
        name = path.name
        m = PICTURE_FILE_RE.fullmatch(name)
        if m:
            idx = int(m.group(1)) - 1
            rel = f"{pack_folder}/{name}"
            refs.append({"index": idx, "imageFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder})
            continue
        m = AUDIO_FILE_RE.fullmatch(name)
        if m:
            idx = int(m.group(1)) - 1
            rel = f"{pack_folder}/{name}"
            audios.append({"index": idx, "audioFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder})
            continue
        m = VIDEO_FILE_RE.fullmatch(name)
        if m:
            idx = int(m.group(1)) - 1
            rel = f"{pack_folder}/{name}"
            videos.append({"index": idx, "videoFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder})
            continue
        if START_FILE_RE.fullmatch(name):
            rel = f"{pack_folder}/{name}"
            start = {"imageFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder}
        elif END_FILE_RE.fullmatch(name):
            rel = f"{pack_folder}/{name}"
            end = {"imageFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder}
    refs.sort(key=lambda r: int(r["index"]))
    audios.sort(key=lambda r: int(r["index"]))
    videos.sort(key=lambda r: int(r["index"]))
    return {"refs": refs, "refAudios": audios, "refVideos": videos, "startImage": start, "endImage": end}


def _merge_refs(json_refs: list | None, scanned: list) -> list:
    by_idx: dict[int, dict] = {}
    for item in scanned:
        by_idx[int(item["index"])] = dict(item)
    for item in json_refs or []:
        if not isinstance(item, dict):
            continue
        idx = _slot_index(item, -1)
        if idx < 0:
            continue
        merged = {**by_idx.get(idx, {}), **item}
        if not _item_rel(merged, IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS):
            if idx in by_idx:
                merged = {**item, **by_idx[idx]}
        by_idx[idx] = merged
    return [by_idx[k] for k in sorted(by_idx)]


def _assemble_timeline(extracted: Path, pack_meta: dict) -> dict:
    shared_path = extracted / "shared_params" / "shared_params.json"
    shared = _read_json(shared_path) if shared_path.is_file() else {}
    scanned_shared = _scan_slot_files(extracted / "shared_params")
    global_block = {
        "taskType": _task_combo(str(pack_meta.get("taskType") or "t2v")),
        "prompt": shared.get("prompt") or "",
        "commonEnabled": bool(shared.get("commonEnabled")),
        "commonCollapsed": bool(shared.get("commonCollapsed")),
        "refs": _merge_refs(shared.get("refs"), scanned_shared["refs"]),
        "refAudios": _merge_refs(shared.get("refAudios") or shared.get("ref_audios"), scanned_shared["refAudios"]),
        "refVideos": _merge_refs(shared.get("refVideos") or shared.get("ref_videos"), scanned_shared["refVideos"]),
        "referenceVideo": {},
        "continuousReference": False,
    }
    groups_root = extracted / "asset_groups"
    group_dirs = sorted(
        [p for p in groups_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    ) if groups_root.is_dir() else []
    segments: list[dict] = []
    shots: list[dict] = []
    cursor = 0
    for i, gdir in enumerate(group_dirs):
        gj = gdir / "group.json"
        raw = _read_json(gj) if gj.is_file() else {}
        scanned = _scan_slot_files(gdir)
        fc = int(raw.get("frameCount") or raw.get("length") or 124)
        dur = raw.get("durationSec")
        start_img = raw.get("startImage") if isinstance(raw.get("startImage"), dict) else scanned["startImage"]
        end_img = raw.get("endImage") if isinstance(raw.get("endImage"), dict) else scanned["endImage"]
        gen = raw.get("genImage") if isinstance(raw.get("genImage"), dict) else None
        if scanned["startImage"] and not (gen and gen.get("imageFile")) and not (start_img and start_img.get("imageFile")):
            gen = {"imageFile": scanned["startImage"]["imageFile"], "fileName": scanned["startImage"]["fileName"]}
        seg = {
            "id": raw.get("id") or f"g{i}",
            "start": raw.get("start") if raw.get("start") is not None else cursor,
            "length": fc,
            "frameCount": fc,
            "durationSec": dur,
            "prompt": raw.get("prompt") or "",
            "negativePrompt": raw.get("negativePrompt") or "",
            "taskType": raw.get("taskType") or "",
            "refs": _merge_refs(raw.get("refs"), scanned["refs"]),
            "refAudios": _merge_refs(raw.get("refAudios") or raw.get("ref_audios"), scanned["refAudios"]),
            "refVideos": _merge_refs(raw.get("refVideos") or raw.get("ref_videos"), scanned["refVideos"]),
            "continuityFromPrev": raw.get("continuityFromPrev", raw.get("continuity_from_prev")),
            "refImageSize": raw.get("refImageSize") or raw.get("ref_image_size"),
            "genImage": gen or {"imageFile": ""},
            "imageFile": (gen or {}).get("imageFile") or raw.get("imageFile") or "",
            "startImage": start_img,
            "endImage": end_img,
        }
        segments.append(seg)
        shots.append({
            "id": seg["id"],
            "durationSec": dur,
            "prompt": seg["prompt"],
            "negativePrompt": seg["negativePrompt"],
            "continuityFromPrev": seg["continuityFromPrev"],
            "startImage": start_img,
            "endImage": end_img,
        })
        cursor += fc
    task_key = resolve_task_key(str(pack_meta.get("taskType") or global_block["taskType"]))
    output = pack_meta.get("output") if isinstance(pack_meta.get("output"), dict) else {}
    mode = "fl2v" if task_key == "fl2v" else ("video" if task_key in ("v2v", "rv2v") else "prompt_batch")
    timeline = {
        "version": 5,
        "timelineMode": mode,
        "editMode": "segment" if mode != "video" else "global",
        "frameRate": output.get("frameRate") or 24,
        "totalFrames": cursor or 124,
        "global": global_block,
        "output": output or {
            "mode": "fixed",
            "width": 864,
            "height": 480,
            "exportMode": "all",
            "audioMode": "generate",
            "refImageSize": "match",
            "continuityEnabled": False,
            "continuityOverlapFrames": 22,
        },
        "segments": segments or [{
            "id": "g0",
            "start": 0,
            "length": 124,
            "frameCount": 124,
            "prompt": "",
            "refs": [],
            "refAudios": [],
            "refVideos": [],
            "genImage": {"imageFile": ""},
        }],
        "video": {"fileName": "", "videoFile": "", "subfolder": "", "type": "input", "frames": [], "frameMap": []},
        "videoClips": [],
        "runSelectEnabled": False,
        "runSelection": [],
    }
    if task_key == "fl2v":
        timeline["shots"] = shots
        timeline["keyframes"] = []
    src_dir = extracted / "source_video"
    if src_dir.is_dir():
        clips = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        video_clips = []
        for i, path in enumerate(clips):
            rel = f"source_video/{path.name}"
            rec = {"videoFile": rel, "fileName": path.name, "type": "input", "subfolder": "source_video", "frames": []}
            video_clips.append(rec)
            if i == 0:
                timeline["video"] = dict(rec)
        if video_clips:
            timeline["videoClips"] = video_clips
    return timeline


def _prefix_pack_paths(obj: Any, prefix: str) -> None:
    if isinstance(obj, list):
        for item in obj:
            _prefix_pack_paths(item, prefix)
        return
    if not isinstance(obj, dict):
        return
    for key in IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS + PREVIEW_KEYS + PAIRED_AUDIO_KEYS:
        val = obj.get(key)
        if not val:
            continue
        rel = str(val).replace("\\", "/").strip().lstrip("/")
        if rel.startswith("minimax_director_packs/"):
            continue
        if rel.startswith(PACK_PREFIXES):
            new_rel = f"{prefix}/{rel}"
            obj[key] = new_rel
            obj["type"] = "input"
            parent = str(Path(new_rel).parent).replace("\\", "/")
            obj["subfolder"] = "" if parent in (".", "") else parent
            obj["fileName"] = Path(new_rel).name
    for key, val in obj.items():
        if key in IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS + PREVIEW_KEYS + PAIRED_AUDIO_KEYS:
            continue
        if isinstance(val, (dict, list)):
            _prefix_pack_paths(val, prefix)


def _copy_tree_media(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = _posix(path.relative_to(src))
        if rel.endswith(".json"):
            continue
        if not _is_ascii_pack_path(rel):
            continue
        ext = path.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        if ext not in MEDIA_EXTS:
            continue
        target = dest / rel.replace("/", os.sep)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _collect_missing_media(obj: Any, dest: Path, rel_prefix: str, missing: list[str]) -> None:
    if isinstance(obj, list):
        for item in obj:
            _collect_missing_media(item, dest, rel_prefix, missing)
        return
    if not isinstance(obj, dict):
        return
    prefix = rel_prefix.strip("/") + "/"
    for keys in (IMAGE_KEYS, AUDIO_KEYS, VIDEO_KEYS, PREVIEW_KEYS, PAIRED_AUDIO_KEYS):
        rel = _item_rel(obj, keys)
        if not rel:
            continue
        rel = rel.replace("\\", "/").strip().lstrip("/")
        if rel.startswith(prefix):
            inner = rel[len(prefix):]
        elif rel.startswith(PACK_PREFIXES):
            inner = rel
        else:
            continue
        path = dest / inner.replace("/", os.sep)
        if path.is_file():
            continue
        missing.append(rel)
        for key in keys:
            if key in obj:
                obj[key] = ""
    for key, val in obj.items():
        if key in IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS + PREVIEW_KEYS + PAIRED_AUDIO_KEYS:
            continue
        if isinstance(val, (dict, list)):
            _collect_missing_media(val, dest, rel_prefix, missing)


def import_extracted_pack(extracted: Path) -> dict[str, Any]:
    pack_path = extracted / "pack.json"
    pack_meta: dict[str, Any] = {}
    if pack_path.is_file():
        pack_meta = _read_json(pack_path)
        if not isinstance(pack_meta, dict):
            pack_meta = {}
    fmt = str(pack_meta.get("format") or "")
    if pack_path.is_file() and fmt and fmt != PACK_FORMAT:
        raise ValueError(f"Unsupported pack format: {fmt}")
    version = int(pack_meta.get("formatVersion") or 1)
    if version > PACK_VERSION:
        raise ValueError(f"Pack formatVersion {version} is newer than this plugin.")

    timeline_path = extracted / "timeline.json"
    if timeline_path.is_file():
        timeline = _read_json(timeline_path)
        if not isinstance(timeline, dict):
            raise ValueError("timeline.json is invalid.")
    else:
        timeline = _assemble_timeline(extracted, pack_meta)

    pack_id = uuid.uuid4().hex[:12]
    rel_prefix = f"minimax_director_packs/{pack_id}"
    dest = _input_dir() / "minimax_director_packs" / pack_id
    _copy_tree_media(extracted, dest)
    _prefix_pack_paths(timeline, rel_prefix)
    missing: list[str] = []
    _collect_missing_media(timeline, dest, rel_prefix, missing)
    unique_missing: list[str] = []
    seen_m: set[str] = set()
    for item in missing:
        if item not in seen_m:
            seen_m.add(item)
            unique_missing.append(item)

    widgets = pack_meta.get("widgets") if isinstance(pack_meta.get("widgets"), dict) else {}
    task_type = widgets.get("task_type") or _task_combo(str(pack_meta.get("taskType") or (timeline.get("global") or {}).get("taskType") or "t2v"))
    widgets = {**widgets, "task_type": task_type}
    if isinstance(timeline.get("global"), dict):
        timeline["global"]["taskType"] = task_type

    return {
        "timeline": timeline,
        "widgets": widgets,
        "packId": pack_id,
        "missing": unique_missing,
    }


def _resolve_uploaded_zip(name: str, subfolder: str = "", type_name: str = "input") -> Path:
    path = resolve_media_path(name, subfolder=subfolder, type_name=type_name)
    if path is None or not path.is_file():
        raise ValueError("Uploaded pack zip was not found.")
    if path.suffix.lower() not in {".zip"}:
        raise ValueError("Pack must be a .zip file.")
    return path


async def minimax_export_pack(request):
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")
    timeline = body.get("timeline")
    if not isinstance(timeline, dict):
        return web.Response(status=400, text="Missing timeline object.")
    widgets = body.get("widgets") if isinstance(body.get("widgets"), dict) else {}
    dry_run = bool(body.get("dryRun") or body.get("dry_run"))
    try:
        result = build_export_pack(timeline, widgets, dry_run=dry_run)
    except Exception as exc:
        log.warning("Director pack export failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    if dry_run:
        return web.json_response(result)
    filename = str(result.get("filename") or "")
    if not re.fullmatch(r"[A-Za-z0-9]+\.mmxpack\.zip", filename):
        return web.Response(status=500, text="Export produced an invalid pack filename.")
    path = _pack_export_root() / filename
    download_name = str(result.get("downloadName") or filename)
    download_name = re.sub(r"[^A-Za-z0-9._-]+", "_", download_name) or filename
    extra = {
        "X-Pack-Download-Name": download_name,
        "X-Pack-Missing": json.dumps(result.get("missing") or [], ensure_ascii=True),
    }
    try:
        size = int(path.stat().st_size)
    except OSError:
        return web.Response(status=404, text="Pack not found.")
    if size <= 0:
        return web.Response(status=500, text="Export produced an empty pack zip.")
    # Small packs: JSON + base64. fetchApi POST JSON is reliable; GET FileResponse
    # on Windows is HTTP 200 with an empty body (0-byte .zip).
    if size <= INLINE_JSON_MAX:
        data = path.read_bytes()
        _unlink_quiet(path)
        result["zipB64"] = base64.b64encode(data).decode("ascii")
        result["downloadName"] = download_name
        result["bytes"] = size
        return web.json_response(result)
    return await _send_zip_file(request, path, download_name, extra, unlink_after=True)


async def minimax_download_pack(request):
    filename = str(request.query.get("filename") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]+\.mmxpack\.zip", filename):
        return web.Response(status=400, text="Invalid pack filename.")
    path = _pack_export_root() / filename
    if not path.is_file():
        return web.Response(status=404, text="Pack not found.")
    download_name = str(request.query.get("download") or filename)
    download_name = re.sub(r"[^A-Za-z0-9._-]+", "_", download_name) or filename
    return await _send_zip_file(request, path, download_name)


async def minimax_import_pack(request):
    extracted: Path | None = None
    upload_dir: Path | None = None
    input_zip: Path | None = None
    try:
        ctype = request.content_type or ""
        if "multipart" in ctype:
            post = await request.post()
            upload = post.get("pack")
            if upload is None or not hasattr(upload, "file"):
                return web.Response(status=400, text="Missing pack file.")
            upload_dir = Path(tempfile.mkdtemp(prefix="mmx_pack_up_"))
            zip_path = upload_dir / "pack.zip"
            with open(zip_path, "wb") as out:
                shutil.copyfileobj(upload.file, out)
        else:
            body = await request.json()
            zip_path = _resolve_uploaded_zip(
                str(body.get("filename") or body.get("name") or ""),
                subfolder=str(body.get("subfolder") or ""),
                type_name=str(body.get("type") or "input"),
            )
            if zip_path.suffix.lower() == ".zip" and _is_under_dir(zip_path, _input_dir()):
                input_zip = zip_path
        extracted = Path(tempfile.mkdtemp(prefix="mmx_pack_ex_"))
        extract_pack_zip(zip_path, extracted)
        result = import_extracted_pack(extracted)
        return web.json_response(result)
    except Exception as exc:
        log.warning("Director pack import failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    finally:
        if extracted is not None:
            shutil.rmtree(extracted, ignore_errors=True)
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        if input_zip is not None:
            _unlink_quiet(input_zip)
