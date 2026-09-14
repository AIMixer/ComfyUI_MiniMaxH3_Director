"""Previews, trigger words and details for LoRAs, shown in the per-segment LoRA picker.

Previews are the sidecar files LoRA Manager and Civitai helpers keep next to a
LoRA: ``<name>.preview.<ext>`` or ``<name>.<ext>`` (png / jpg / jpeg / webp /
gif, or mp4 / webm), falling back to a local ``preview_url`` in LoRA Manager's
``<name>.metadata.json``. Trigger words come from that metadata (Civitai
``trainedWords``), the LoRA's own safetensors header
(``modelspec.trigger_phrase``) and kohya dataset folders in
``ss_tag_frequency`` (``"10_ohwx"`` -> ``ohwx``). Only names ComfyUI lists
under ``loras`` are looked up, and a preview is served only when it sits inside
a configured loras folder.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct

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
MAX_TRIGGERS = 12
MAX_TAGS = 6
_HEADER_LIMIT = 16 * 1024 * 1024
_KOHYA_DATASET = re.compile(r"^\d+_(.+)$")

# name -> ((lora mtime, metadata mtime, folder mtime), info)
_info_cache: dict[str, tuple[tuple, dict]] = {}


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


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
    url = str(_read_manager_metadata(base).get("preview_url") or "")
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


def _as_list(value) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [value] if value.strip() else []
    return value if isinstance(value, list) else []


def _read_manager_metadata(base: str) -> dict:
    """LoRA Manager's ``<name>.metadata.json``, or {}."""
    path = f"{base}.metadata.json"
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_header_metadata(path: str) -> dict:
    """``__metadata__`` of a safetensors header, without reading any tensors."""
    try:
        with open(path, "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            if size <= 0 or size > _HEADER_LIMIT:
                return {}
            meta = json.loads(fh.read(size)).get("__metadata__")
    except (OSError, ValueError, struct.error, AttributeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _trigger_words(manager: dict, header: dict) -> list[str]:
    civitai = manager.get("civitai") if isinstance(manager.get("civitai"), dict) else {}
    raw = _as_list(civitai.get("trainedWords")) + _as_list(manager.get("trainedWords"))
    phrase = header.get("modelspec.trigger_phrase")
    if phrase:
        raw.append(phrase)
    try:
        frequency = json.loads(header.get("ss_tag_frequency") or "{}")
    except (TypeError, ValueError):
        frequency = {}
    if isinstance(frequency, dict):
        for dataset in frequency:
            match = _KOHYA_DATASET.match(str(dataset))
            if match:
                raw.append(match.group(1))
    words: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = " ".join(str(item).split())[:200]
        if text and text.lower() not in seen:
            seen.add(text.lower())
            words.append(text)
        if len(words) >= MAX_TRIGGERS:
            break
    return words


def _lora_info(name: str, lora_path: str) -> dict:
    base = os.path.splitext(lora_path)[0]
    signature = (_mtime(lora_path), _mtime(f"{base}.metadata.json"), _mtime(os.path.dirname(lora_path)))
    cached = _info_cache.get(name)
    if cached and cached[0] == signature:
        return cached[1]

    manager = _read_manager_metadata(base)
    civitai = manager.get("civitai") if isinstance(manager.get("civitai"), dict) else {}
    model = civitai.get("model") if isinstance(civitai.get("model"), dict) else {}
    base_model = str(manager.get("base_model") or "").strip()
    if base_model.lower() in ("", "unknown"):
        base_model = str(civitai.get("baseModel") or "").strip()
    title = str(model.get("name") or manager.get("model_name") or "").strip()
    if title.lower() == os.path.basename(base).lower():
        title = ""
    preview = _preview_for(lora_path)
    info = {
        "preview": {"kind": _kind(preview), "v": int(_mtime(preview))} if preview else None,
        "triggers": _trigger_words(manager, _read_header_metadata(lora_path)),
        "base": base_model or None,
        "title": title or None,
        "tags": [str(t).strip() for t in _as_list(manager.get("tags")) if str(t).strip()][:MAX_TAGS],
    }
    _info_cache[name] = (signature, info)
    return info


def lora_catalog() -> dict[str, dict]:
    """``{name: {preview, triggers, base, title, tags}}`` for every listed LoRA."""
    out: dict[str, dict] = {}
    for name in folder_paths.get_filename_list("loras"):
        full = folder_paths.get_full_path("loras", name)
        if not full or not os.path.isfile(full):
            continue
        try:
            out[name] = _lora_info(name, full)
        except Exception as exc:
            log.debug("LoRA info failed for %s: %s", name, exc)
    return out


def list_lora_previews() -> dict[str, dict]:
    """``{name: {"kind": "image" | "video", "v": mtime}}`` for LoRAs with a preview."""
    return {name: info["preview"] for name, info in lora_catalog().items() if info.get("preview")}


def find_lora_preview(name: str) -> str | None:
    """Preview file for a LoRA name ComfyUI lists, or None."""
    name = str(name or "")
    if not name or name not in folder_paths.get_filename_list("loras"):
        return None
    full = folder_paths.get_full_path("loras", name)
    if not full or not os.path.isfile(full):
        return None
    return _preview_for(full)


async def minimax_lora_previews(request):
    try:
        catalog = lora_catalog()
        previews = {name: info["preview"] for name, info in catalog.items() if info.get("preview")}
        return web.json_response({"previews": previews, "catalog": catalog})
    except Exception as exc:
        log.warning("MiniMax H3 Director LoRA catalog failed: %s", exc)
        return web.json_response({"previews": {}, "catalog": {}, "error": str(exc)}, status=500)


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
