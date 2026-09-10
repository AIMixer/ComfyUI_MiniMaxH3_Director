"""Preview images and videos for LoRAs, shown in the per-segment LoRA picker.

Previews are the sidecar files LoRA Manager and Civitai helpers keep next to a
LoRA: ``<name>.preview.<ext>`` or ``<name>.<ext>`` (png / jpg / jpeg / webp /
gif, or mp4 / webm), falling back to a local ``preview_url`` in LoRA Manager's
``<name>.metadata.json``. Only names ComfyUI lists under ``loras`` are looked
up, and a preview is served only when it sits inside a configured loras folder.
"""

from __future__ import annotations

import json
import logging
import os

import folder_paths
from aiohttp import web

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.lora_previews")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
VIDEO_EXTS = (".mp4", ".webm")
_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def _inside_lora_roots(path: str) -> bool:
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    for root in folder_paths.get_folder_paths("loras"):
        try:
            root_real = os.path.realpath(root)
            if os.path.commonpath([real, root_real]) == root_real:
                return True
        except ValueError:  # different drives on Windows
            continue
    return False


def _preview_for(lora_path: str) -> str | None:
    base = os.path.splitext(lora_path)[0]
    for ext in IMAGE_EXTS + VIDEO_EXTS:
        for cand in (f"{base}.preview{ext}", f"{base}{ext}"):
            if os.path.isfile(cand) and _inside_lora_roots(cand):
                return cand
    meta = f"{base}.metadata.json"
    if os.path.isfile(meta):
        try:
            with open(meta, encoding="utf-8") as fh:
                url = str(json.load(fh).get("preview_url") or "")
        except (OSError, ValueError, AttributeError):
            url = ""
        if (
            url
            and url.lower().endswith(IMAGE_EXTS + VIDEO_EXTS)
            and os.path.isfile(url)
            and _inside_lora_roots(url)
        ):
            return os.path.normpath(url)
    return None


def _kind(path: str) -> str:
    return "video" if path.lower().endswith(VIDEO_EXTS) else "image"


def find_lora_preview(name: str) -> str | None:
    """Preview file for a LoRA name ComfyUI lists, or None."""
    name = str(name or "")
    if not name or name not in folder_paths.get_filename_list("loras"):
        return None
    full = folder_paths.get_full_path("loras", name)
    if not full or not os.path.isfile(full):
        return None
    return _preview_for(full)


def list_lora_previews() -> dict[str, dict]:
    """``{name: {"kind": "image" | "video", "v": mtime}}`` for LoRAs with a preview."""
    out: dict[str, dict] = {}
    for name in folder_paths.get_filename_list("loras"):
        full = folder_paths.get_full_path("loras", name)
        if not full:
            continue
        path = _preview_for(full)
        if path:
            try:
                version = int(os.path.getmtime(path))
            except OSError:
                version = 0
            out[name] = {"kind": _kind(path), "v": version}
    return out


async def minimax_lora_previews(request):
    try:
        return web.json_response({"previews": list_lora_previews()})
    except Exception as exc:
        log.warning("MiniMax H3 Director LoRA preview list failed: %s", exc)
        return web.json_response({"previews": {}, "error": str(exc)}, status=500)


async def minimax_lora_preview(request):
    path = find_lora_preview(request.query.get("name", ""))
    if not path:
        return web.Response(status=404)
    ext = os.path.splitext(path)[1].lower()
    return web.FileResponse(
        path,
        headers={
            "Content-Type": _CONTENT_TYPES.get(ext, "application/octet-stream"),
            "Cache-Control": "max-age=86400",
        },
    )
