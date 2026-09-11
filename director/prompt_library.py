"""Prompt library and auto history for the MiniMax H3 Director.

Saved prompts keep their LoRA stack, the user's categories and a thumbnail;
every rendered group is also recorded in a history with a looping WebP cut from
its decoded frames. Files live in ``<user>/default/minimax_director/``:
``prompt_library.json``, ``prompt_history.json`` and ``thumbs/``. The Director
UI (web/js/minimax_prompt_library.js) reads and edits them through the routes
at the bottom of this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid

import folder_paths
from aiohttp import web

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.prompt_library")

HISTORY_LIMIT = 300
THUMB_MAX_SIDE = 320
THUMB_FRAMES = 16
THUMB_FPS = 8
MAX_TITLE = 120
MAX_CATEGORIES = 64
_THUMB_NAME = re.compile(r"^[hp]_[0-9a-f]{16}\.webp$")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_lock = threading.RLock()


def _root() -> str:
    path = os.path.join(folder_paths.get_user_directory(), "default", "minimax_director")
    os.makedirs(os.path.join(path, "thumbs"), exist_ok=True)
    return path


def _thumb_path(name: str) -> str:
    return os.path.join(_root(), "thumbs", name)


def _read_json(name: str, default):
    try:
        with open(os.path.join(_root(), name), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return default
    return data if isinstance(data, type(default)) else default


def _write_json(name: str, data) -> None:
    path = os.path.join(_root(), name)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _load() -> tuple[dict, list]:
    lib = _read_json("prompt_library.json", {})
    lib["version"] = 1
    for key in ("categories", "prompts"):
        if not isinstance(lib.get(key), list):
            lib[key] = []
    hist = [h for h in _read_json("prompt_history.json", []) if isinstance(h, dict)]
    return lib, hist


def _prune_thumbs(lib: dict, hist: list) -> None:
    keep = {e.get("thumb") for e in lib["prompts"]} | {e.get("thumb") for e in hist}
    folder = os.path.join(_root(), "thumbs")
    for name in os.listdir(folder):
        if _THUMB_NAME.match(name) and name not in keep:
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass


def _save(lib: dict, hist: list) -> None:
    _write_json("prompt_library.json", lib)
    _write_json("prompt_history.json", hist)
    _prune_thumbs(lib, hist)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _clean_loras(raw) -> list[dict]:
    from .segment_loras import normalize_lora_rows

    return [dict(row) for row in normalize_lora_rows(raw)]


def _auto_title(prompt: str) -> str:
    words = re.sub(r"[<>\[\]{}()\"“”]", " ", str(prompt or "")).split()
    title = " ".join(words[:7])
    if len(title) > 48:
        title = title[:47].rstrip() + "…"
    return title or "Untitled"


def _entry_key(prompt: str, loras: list[dict]) -> str:
    rows = [[r.get("name"), round(float(r.get("strength", 1)), 3), bool(r.get("active", True))] for r in loras]
    raw = json.dumps([prompt.strip(), rows], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _write_thumb(frames, prefix: str = "h") -> str | None:
    """Looping WebP (up to THUMB_FRAMES frames spread over the clip) from [F, H, W, C] 0..1 frames."""
    try:
        import torch
        from PIL import Image

        if frames is None or getattr(frames, "ndim", 0) != 4 or int(frames.shape[0]) <= 0:
            return None
        total = int(frames.shape[0])
        count = min(THUMB_FRAMES, total)
        picks = sorted({round(i * (total - 1) / max(1, count - 1)) for i in range(count)})
        images = []
        for i in picks:
            arr = (
                frames[i, :, :, :3].detach().float().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
            )
            img = Image.fromarray(arr)
            img.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
            images.append(img)
        name = f"{prefix}_{uuid.uuid4().hex[:16]}.webp"
        if len(images) == 1:
            images[0].save(_thumb_path(name), format="WEBP", quality=82)
        else:
            images[0].save(
                _thumb_path(name),
                format="WEBP",
                save_all=True,
                append_images=images[1:],
                duration=int(1000 / THUMB_FPS),
                loop=0,
                quality=75,
                method=4,
            )
        return name
    except Exception as exc:
        log.warning("Prompt history thumbnail failed: %s", exc)
        return None


def record_prompt_history(node_id, seg, plan, frames, *, seed=None, group=None, groups=None) -> None:
    """Add or refresh the history entry for one rendered group. Never raises.

    The same prompt + LoRA stack moves to the top with a fresh thumbnail and a
    run count instead of piling up duplicates.
    """
    try:
        prompt = str(getattr(seg, "prompt", "") or "").strip()
        loras = _clean_loras(getattr(seg, "loras", None))
        if not prompt and not loras:
            return
        try:
            seed_value = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed_value = None
        with _lock:
            lib, hist = _load()
            thumb = _write_thumb(frames)
            key = _entry_key(prompt, loras)
            entry = next((h for h in hist if h.get("key") == key), None)
            if entry is not None:
                hist.remove(entry)
                entry["runs"] = int(entry.get("runs") or 1) + 1
            else:
                entry = {"id": _new_id("h"), "key": key, "runs": 1}
            entry.update(
                prompt=prompt,
                negative=str(getattr(seg, "negative_prompt", "") or "").strip(),
                loras=loras,
                created=int(time.time()),
                node=str(node_id or ""),
                group=int(group) if group is not None else None,
                groups=int(groups) if groups is not None else None,
                task=str(getattr(seg, "task_key", "") or ""),
                seed=seed_value,
                frames=int(frames.shape[0]) if getattr(frames, "ndim", 0) == 4 else None,
            )
            if thumb:
                entry["thumb"] = thumb
            hist.insert(0, entry)
            del hist[HISTORY_LIMIT:]
            _save(lib, hist)
    except Exception as exc:
        log.warning("Prompt history not recorded: %s", exc)


def _pick(items: list, item_id) -> dict | None:
    return next((x for x in items if x.get("id") == item_id), None) if item_id else None


def _apply_prompt_fields(lib: dict, entry: dict, body: dict) -> None:
    if "prompt" in body:
        entry["prompt"] = str(body.get("prompt") or "").strip()
    if "negative" in body:
        entry["negative"] = str(body.get("negative") or "").strip()
    if "loras" in body:
        entry["loras"] = _clean_loras(body.get("loras"))
    if "categories" in body:
        valid = {c.get("id") for c in lib["categories"]}
        entry["categories"] = [c for c in dict.fromkeys(body.get("categories") or []) if c in valid]
    if "favorite" in body:
        entry["favorite"] = bool(body.get("favorite"))
    if "title" in body or not entry.get("title"):
        title = " ".join(str(body.get("title") or "").split())[:MAX_TITLE]
        entry["title"] = title or _auto_title(entry.get("prompt", ""))
    source = str(body.get("thumb_from") or "")
    if source and _THUMB_NAME.match(source) and os.path.isfile(_thumb_path(source)):
        name = f"p_{uuid.uuid4().hex[:16]}.webp"
        shutil.copyfile(_thumb_path(source), _thumb_path(name))
        entry["thumb"] = name
        entry["thumb_origin"] = str(body.get("thumb_origin") or source)
    for key, default in (("prompt", ""), ("negative", ""), ("loras", []), ("categories", []), ("favorite", False)):
        entry.setdefault(key, default)
    entry["updated"] = int(time.time())


def _op_save_prompt(lib, hist, body):
    entry = _pick(lib["prompts"], body.get("id"))
    if entry is None:
        entry = {"id": _new_id("p"), "created": int(time.time()), "uses": 0}
        lib["prompts"].insert(0, entry)
    _apply_prompt_fields(lib, entry, body)
    return {"prompt": entry}


def _op_save_prompts(lib, hist, body):
    items = [item for item in (body.get("items") or []) if isinstance(item, dict)]
    saved = [_op_save_prompt(lib, hist, item)["prompt"]["id"] for item in reversed(items)]
    return {"saved": list(reversed(saved))}


def _op_delete_prompt(lib, hist, body):
    lib["prompts"] = [p for p in lib["prompts"] if p.get("id") != body.get("id")]


def _op_use_prompt(lib, hist, body):
    entry = _pick(lib["prompts"], body.get("id"))
    if entry is not None:
        entry["uses"] = int(entry.get("uses") or 0) + 1
        entry["last_used"] = int(time.time())


def _category_name(body) -> str:
    return " ".join(str(body.get("name") or "").split())[:60]


def _op_add_category(lib, hist, body):
    name = _category_name(body)
    if not name:
        raise ValueError("Category name is empty.")
    if len(lib["categories"]) >= MAX_CATEGORIES:
        raise ValueError("Too many categories.")
    color = str(body.get("color") or "")
    category = {"id": _new_id("c"), "name": name, "color": color if _COLOR.match(color) else "#4fff8f"}
    lib["categories"].append(category)
    return {"category": category}


def _op_update_category(lib, hist, body):
    category = _pick(lib["categories"], body.get("id"))
    if category is None:
        raise ValueError("Unknown category.")
    name = _category_name(body)
    if name:
        category["name"] = name
    color = str(body.get("color") or "")
    if _COLOR.match(color):
        category["color"] = color
    return {"category": category}


def _op_delete_category(lib, hist, body):
    category_id = body.get("id")
    lib["categories"] = [c for c in lib["categories"] if c.get("id") != category_id]
    for prompt in lib["prompts"]:
        prompt["categories"] = [c for c in prompt.get("categories") or [] if c != category_id]


def _op_history_delete(lib, hist, body):
    hist[:] = [h for h in hist if h.get("id") != body.get("id")]


def _op_history_clear(lib, hist, body):
    hist.clear()


_OPS = {
    "save_prompt": _op_save_prompt,
    "save_prompts": _op_save_prompts,
    "delete_prompt": _op_delete_prompt,
    "use_prompt": _op_use_prompt,
    "add_category": _op_add_category,
    "update_category": _op_update_category,
    "delete_category": _op_delete_category,
    "history_delete": _op_history_delete,
    "history_clear": _op_history_clear,
}


def _state(lib: dict, hist: list) -> dict:
    return {"categories": lib["categories"], "prompts": lib["prompts"], "history": hist}


async def minimax_prompt_library(request):
    try:
        with _lock:
            lib, hist = _load()
        return web.json_response(_state(lib, hist))
    except Exception as exc:
        log.warning("MiniMax H3 Director prompt library read failed: %s", exc)
        return web.json_response({"categories": [], "prompts": [], "history": [], "error": str(exc)}, status=500)


async def minimax_prompt_library_op(request):
    try:
        body = await request.json()
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)
    if not isinstance(body, dict) or body.get("op") not in _OPS:
        return web.json_response({"ok": False, "error": "Unknown op."}, status=400)
    try:
        with _lock:
            lib, hist = _load()
            result = _OPS[body["op"]](lib, hist, body) or {}
            _save(lib, hist)
            return web.json_response({"ok": True, **result, **_state(lib, hist)})
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        log.warning("MiniMax H3 Director prompt library op %s failed: %s", body.get("op"), exc)
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def minimax_prompt_thumb(request):
    name = str(request.query.get("name") or "")
    if not _THUMB_NAME.match(name):
        return web.Response(status=404)
    path = _thumb_path(name)
    if not os.path.isfile(path):
        return web.Response(status=404)
    return web.FileResponse(
        path,
        headers={"Content-Type": "image/webp", "Cache-Control": "max-age=31536000, immutable"},
    )
