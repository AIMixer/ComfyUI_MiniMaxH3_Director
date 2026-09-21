"""Position-independent slot identity for the Director segment cache.

The disk cache used to be addressed by timeline position: segment *k* always
read and wrote ``seg_000k.*``. Inserting a segment anywhere but the tail
therefore shifted every later segment onto a different slot, so caches that had
not actually changed were reported as misses — and a merge/export could even
pick up a neighbour's frames off a stale slot.

A slot is now assigned per *stable identity*: the external group's graph node
label (``109:DNMediaToDirectorGroup``), which survives reordering, prompt edits
and duration tweaks, because the node keeps its graph id. That label is mapped
to a slot number in ``slots.json`` next to the cache files, so the payloads
stay named ``seg_0000.frames.mkv`` and deleting one segment's cache is still a
matter of knowing its slot number.

Design constraints:

* **Additive.** Panel-only timelines (no witness, or a witness whose group
  carries no node label) keep using ``seg.index`` as their slot — exactly the
  previous behaviour, no new files, no renames.
* **Migration-free for existing caches.** A directory without ``slots.json``
  is bootstrapped by reading the node label already stored in each
  ``*.meta.json``, so caches written before this change keep matching and do
  not have to be re-rendered.
* **Cheap.** The table is read once per change (keyed on the file's
  mtime+size) and written only when a new identity is added.

What is deliberately *not* stable: the witness record's ``slot`` field
(``groups.group_0``), which does move when the chain is reordered. It is
stripped from the fingerprint in :mod:`segment_cache` so that reordering alone
never invalidates a segment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache")

SLOT_TABLE_NAME = "slots.json"
SLOT_TABLE_VERSION = 1

#: Fingerprint key holding one segment's own external-group record.
EXTERNAL_SEGMENT_FP_KEY = "external_group"

#: ``seg_0000.meta.json`` and ``seg_0000.pre.meta.json``.
_ANY_META_NAME_RE = re.compile(r"^seg_(\d+)\.(?:pre\.)?meta\.json$")

#: path -> (file stamp, table). The stamp is ``None`` when the file is absent,
#: in which case the entry caches the bootstrapped table.
_TABLE_CACHE: dict[str, tuple[tuple[int, int] | None, dict[str, Any]]] = {}


# --------------------------------------------------------------------------- #
# Small file helpers (kept local so this module does not import segment_cache)
# --------------------------------------------------------------------------- #


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    mtime = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    return (mtime, int(st.st_size))


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a unique temp name; cloud mounts that block overwrite fall back."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        try:
            os.replace(tmp, path)
            return
        except OSError:
            pass
        _safe_unlink(path)
        try:
            os.replace(tmp, path)
            return
        except OSError:
            pass
        tmp.rename(path)
    finally:
        _safe_unlink(tmp)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Stable identity
# --------------------------------------------------------------------------- #


def node_key_for_index(index: Any, plan: Any) -> str | None:
    """Node label of the external group at ``index``, or ``None``.

    The witness is rebuilt from the live graph on every run, so ``groups[index]``
    is *this* segment's record; only the label it carries is position-free.
    """
    witness = getattr(plan, "external_groups_witness", None)
    if not isinstance(witness, dict):
        return None
    groups = witness.get("groups")
    if not isinstance(groups, list):
        return None
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if not 0 <= idx < len(groups):
        return None
    record = groups[idx]
    if not isinstance(record, dict):
        return None
    return str(record.get("node") or "").strip() or None


def node_key(seg: Any, plan: Any) -> str | None:
    """Stable slot identity of one segment (see :func:`node_key_for_index`)."""
    return node_key_for_index(getattr(seg, "index", 0), plan)


def strip_position(record: Any) -> Any:
    """Drop the position-bound ``slot`` from a witness record.

    ``groups.group_0`` is an ordinal, not an identity: reordering the chain
    renames every slot without changing what a group renders. Everything else in
    the record (node label, prompt digest, duration, sub-graph digest) is
    position-free and stays.
    """
    if not isinstance(record, dict):
        return record
    if "slot" not in record:
        return record
    return {key: value for key, value in record.items() if key != "slot"}


# --------------------------------------------------------------------------- #
# Slot table
# --------------------------------------------------------------------------- #


def _normalize(raw: Any) -> dict[str, Any]:
    slots: dict[str, int] = {}
    next_slot = 0
    if isinstance(raw, dict):
        raw_slots = raw.get("slots")
        if isinstance(raw_slots, dict):
            for key, value in raw_slots.items():
                name = str(key or "").strip()
                if not name or name in slots:
                    continue
                try:
                    slot = int(value)
                except (TypeError, ValueError):
                    continue
                if slot >= 0:
                    slots[name] = slot
        try:
            next_slot = int(raw.get("next"))
        except (TypeError, ValueError):
            next_slot = 0
    lowest_free = max(slots.values(), default=-1) + 1
    return {
        "version": SLOT_TABLE_VERSION,
        "next": max(next_slot, lowest_free),
        "slots": slots,
    }


def _record_node(meta: Any) -> str | None:
    """Node label stored in a segment's ``*.meta.json`` (fingerprint schema)."""
    if not isinstance(meta, dict):
        return None
    record = meta.get(EXTERNAL_SEGMENT_FP_KEY)
    if isinstance(record, dict):
        node = str(record.get("node") or "").strip()
        if node:
            return node
    return None


def _bootstrap(root: Path) -> dict[str, Any]:
    """Derive the table from node labels already stored in segment metas.

    This is what keeps pre-existing caches usable: the slot numbers on disk are
    taken as-is and only *labelled* with the identity they were written for.
    """
    slots: dict[str, int] = {}
    try:
        entries = sorted(root.iterdir())
    except OSError:
        entries = []
    for path in entries:
        match = _ANY_META_NAME_RE.match(path.name)
        if not match:
            continue
        try:
            slot = int(match.group(1))
        except ValueError:
            continue
        node = _record_node(_load_json(path))
        if not node or node in slots:
            continue
        slots[node] = slot
    if slots:
        log.info(
            "Segment cache slot table bootstrapped from %d existing cache entry(ies).",
            len(slots),
        )
    return _normalize({"slots": slots})


def _table_for(root: Path) -> dict[str, Any]:
    path = root / SLOT_TABLE_NAME
    stamp = _stamp(path)
    cached = _TABLE_CACHE.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    raw = _load_json(path) if stamp is not None else None
    table = _normalize(raw) if raw is not None else _bootstrap(root)
    _TABLE_CACHE[str(path)] = (stamp, table)
    return table


def _save_table(root: Path, table: dict[str, Any]) -> None:
    path = root / SLOT_TABLE_NAME
    text = json.dumps(table, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    try:
        _atomic_write_text(path, text)
    except OSError as exc:
        log.warning(
            "Segment cache slot table could not be written (%s); "
            "slot numbers stay valid for this run only.",
            exc,
        )
    _TABLE_CACHE.pop(str(path), None)


def resolve_slot_for_index(
    root: Path,
    index: Any,
    plan: Any,
    *,
    create: bool = True,
) -> int | None:
    """Slot number for the segment at ``index``, allocating one when needed.

    Returns ``seg.index`` (the legacy position) when the timeline has no stable
    identity to key on, and ``None`` when ``create`` is false and the identity
    is simply not cached yet — callers then report a miss instead of reading a
    neighbour's slot.
    """
    key = node_key_for_index(index, plan)
    try:
        fallback = int(index)
    except (TypeError, ValueError):
        fallback = 0
    if key is None:
        return fallback
    table = _table_for(root)
    slots = table["slots"]
    slot = slots.get(key)
    if slot is not None:
        return int(slot)
    if not create:
        return None
    slot = int(table["next"])
    slots[key] = slot
    table["next"] = slot + 1
    log.info(
        "Segment cache: new identity %s -> slot %d (was position %d).",
        key,
        slot,
        fallback,
    )
    _save_table(root, table)
    return slot


def resolve_slot(root: Path, seg: Any, plan: Any, *, create: bool = True) -> int:
    """Slot number for one :class:`SegmentPlan`.

    Same as :func:`resolve_slot_for_index` but always returns a usable slot:
    the read/write paths must never end up with ``None`` (that would either
    crash on a filename or silently skip a segment). Use the ``_for_index``
    variant with ``create=False`` when "not cached" has to stay distinguishable
    from "cached in slot N".
    """
    slot = resolve_slot_for_index(root, getattr(seg, "index", 0), plan, create=create)
    if slot is not None:
        return slot
    try:
        return int(getattr(seg, "index", 0) or 0)
    except (TypeError, ValueError):
        return 0


def valid_slots(root: Path, plan: Any, segments: Iterable[Any]) -> set[int]:
    """Slot numbers the current timeline still claims.

    A segment whose identity is not in the table yet falls back to its position,
    which is the conservative answer: that slot may still hold a cache the run
    is about to reuse, so pruning must not touch it.
    """
    out: set[int] = set()
    for seg in segments:
        slot = resolve_slot(root, seg, plan, create=False)
        if slot is None:
            try:
                slot = int(getattr(seg, "index", 0) or 0)
            except (TypeError, ValueError):
                continue
        out.add(slot)
    return out


def describe_slots(root: Path, plan: Any, segments: Iterable[Any]) -> list[dict[str, Any]]:
    """``[{slot, node, index}]`` for tooling / diagnostics; never raises."""
    rows: list[dict[str, Any]] = []
    table = _table_for(root)
    for seg in segments:
        try:
            index = int(getattr(seg, "index", 0) or 0)
        except (TypeError, ValueError):
            continue
        key = node_key_for_index(index, plan)
        slot = resolve_slot_for_index(root, index, plan, create=False)
        rows.append(
            {
                "index": index,
                "slot": slot if slot is not None else index,
                "node": key or "",
                "mapped": key is not None and key in table["slots"],
            }
        )
    return rows


def forget_cached_tables() -> None:
    """Drop the in-process table cache (tests, and after an external clear)."""
    _TABLE_CACHE.clear()
