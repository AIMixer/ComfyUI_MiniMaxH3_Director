/** Per-segment LoRA rows for the MiniMax H3 Director segment panel.
 *
 * A segment stores its stack as `segment.loras = [{name, strength, active}]`.
 * The backend (director/segment_loras.py) patches those onto that segment's
 * base MODEL right before sampling, so a single Director run can change LoRAs
 * shot by shot while the inter-segment guide still spans the whole timeline.
 *
 * LoRAs are chosen in a thumbnail picker. Previews are the image or video files
 * LoRA Manager / Civitai helpers keep next to each LoRA, served by
 * director/lora_previews.py; a LoRA without one gets a plain tile.
 */

import { api } from "../../scripts/api.js";
import { t } from "./minimax_i18n.js";

export const MAX_SEGMENT_LORAS = 8;
const MIN_STRENGTH = -10;
const MAX_STRENGTH = 10;

let _loraNamesPromise = null;
let _loraPreviewsPromise = null;
let _openPicker = null;

/** LoRA filenames, taken from the stock LoraLoader combo so no extra route is needed. */
export function loraNames() {
    if (!_loraNamesPromise) {
        _loraNamesPromise = api
            .fetchApi("/object_info/LoraLoader")
            .then((r) => r.json())
            .then((j) => {
                const opts = j?.LoraLoader?.input?.required?.lora_name?.[0];
                return Array.isArray(opts) ? opts : [];
            })
            .catch(() => []);
    }
    return _loraNamesPromise;
}

/** `{name: {kind: "image" | "video", v}}` for LoRAs that have a preview file. */
export function loraPreviews() {
    if (!_loraPreviewsPromise) {
        _loraPreviewsPromise = api
            .fetchApi("/minimax/director/lora_previews")
            .then((r) => (r.ok ? r.json() : null))
            .then((j) => (j && j.previews && typeof j.previews === "object" ? j.previews : {}))
            .catch(() => ({}));
    }
    return _loraPreviewsPromise;
}

/** Drop the cached lists so a newly copied LoRA or preview shows up without a reload. */
export function refreshLoraNames() {
    _loraNamesPromise = null;
    _loraPreviewsPromise = null;
    return loraNames();
}

function loraCatalog() {
    return Promise.all([loraNames(), loraPreviews()]).then(([names, previews]) => ({ names, previews }));
}

function clampStrength(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 1;
    return Math.max(MIN_STRENGTH, Math.min(MAX_STRENGTH, n));
}

export function normalizeLoraRows(raw) {
    if (!Array.isArray(raw)) return [];
    const out = [];
    for (const item of raw) {
        const row = typeof item === "string" ? { name: item } : item;
        const name = String(row?.name ?? "").trim();
        if (!name) continue;
        out.push({
            name,
            strength: clampStrength(row.strength ?? 1),
            active: row.active === undefined ? true : !!row.active,
        });
        if (out.length >= MAX_SEGMENT_LORAS) break;
    }
    return out;
}

/** "MiniMax H3\\character\\x.safetensors" -> {stem: "x", folder: "MiniMax H3/character"} */
function splitLoraName(name) {
    const parts = String(name || "").split(/[\\/]/);
    const file = parts.pop() || "";
    return { stem: file.replace(/\.[^.]+$/, ""), folder: parts.join("/") };
}

function previewURL(name, info) {
    const v = encodeURIComponent(String(info?.v ?? 0));
    return api.apiURL(`/minimax/director/lora_preview?name=${encodeURIComponent(name)}&v=${v}`);
}

function emptyThumb(box) {
    box.replaceChildren();
    box.classList.add("mmx-lora-thumb-empty");
    box.textContent = "LoRA";
}

function makeThumb(name, info, sizeClass) {
    const box = document.createElement("span");
    box.className = `mmx-lora-thumb ${sizeClass}`;
    if (!info) {
        emptyThumb(box);
        return box;
    }
    const src = previewURL(name, info);
    if (info.kind === "video") {
        const video = document.createElement("video");
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = "metadata";
        video.src = `${src}#t=0.1`;
        video.onerror = () => emptyThumb(box);
        box.appendChild(video);
    } else {
        const img = document.createElement("img");
        img.alt = "";
        img.loading = "lazy";
        img.decoding = "async";
        img.src = src;
        img.onerror = () => emptyThumb(box);
        box.appendChild(img);
    }
    return box;
}

/** Video previews inside `target` play while the pointer is over it. */
function playOnHover(target) {
    target.addEventListener("mouseenter", () => {
        target.querySelector("video")?.play?.()?.catch?.(() => {});
    });
    target.addEventListener("mouseleave", () => {
        target.querySelector("video")?.pause?.();
    });
}

export function closeLoraPicker() {
    _openPicker?.close();
}

/** Thumbnail grid for choosing a LoRA, anchored under `anchor`; calls onPick(name). */
export async function openLoraPicker(anchor, current, onPick) {
    if (_openPicker && _openPicker.anchor === anchor) {
        closeLoraPicker();
        return;
    }
    closeLoraPicker();
    ensureSegmentLoraStyles();
    const { names, previews } = await loraCatalog();
    if (!anchor?.isConnected) return;

    const state = { anchor, close: null };
    const panel = document.createElement("div");
    panel.className = "mmx-lora-picker";

    const head = document.createElement("div");
    head.className = "mmx-lora-picker-head";
    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = t("panel.searchLoras");
    const items = names.map((name) => ({ name, ...splitLoraName(name), hay: name.toLowerCase() }));
    const folders = [...new Set(items.map((it) => it.folder))].sort((a, b) => a.localeCompare(b));
    const folderSel = document.createElement("select");
    folderSel.append(new Option(t("panel.allFolders"), ""));
    for (const folder of folders) folderSel.append(new Option(folder || "/", folder));
    folderSel.hidden = folders.length < 2;
    const count = document.createElement("span");
    count.className = "mmx-lora-picker-count";
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "mmx-lora-picker-close";
    closeBtn.textContent = "×";
    head.append(search, folderSel, count, closeBtn);

    const grid = document.createElement("div");
    grid.className = "mmx-lora-picker-grid";
    const empty = document.createElement("div");
    empty.className = "mmx-lora-picker-empty";
    empty.textContent = names.length ? t("panel.noLoraMatch") : t("panel.noLorasFound");
    empty.hidden = true;
    panel.append(head, grid, empty);

    const close = () => {
        if (_openPicker !== state) return;
        _openPicker = null;
        document.removeEventListener("pointerdown", onDocDown, true);
        window.removeEventListener("resize", close);
        panel.querySelectorAll("video").forEach((v) => v.pause());
        panel.remove();
    };
    state.close = close;
    const finish = (name) => {
        close();
        onPick?.(name);
    };
    const onDocDown = (e) => {
        if (panel.contains(e.target) || anchor.contains(e.target)) return;
        close();
    };

    const tiles = items.map((it) => {
        const tile = document.createElement("button");
        tile.type = "button";
        tile.className = "mmx-lora-tile";
        if (it.name === current) tile.classList.add("is-current");
        tile.title = it.name;
        const label = document.createElement("span");
        label.className = "mmx-lora-tile-name";
        label.textContent = it.stem;
        const sub = document.createElement("span");
        sub.className = "mmx-lora-tile-folder";
        sub.textContent = it.folder;
        tile.append(makeThumb(it.name, previews[it.name], "mmx-lora-thumb-lg"), label, sub);
        playOnHover(tile);
        tile.onclick = () => finish(it.name);
        grid.appendChild(tile);
        return { it, tile };
    });

    const apply = () => {
        const words = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
        const folder = folderSel.value;
        let shown = 0;
        for (const { it, tile } of tiles) {
            const ok = (!folder || it.folder === folder) && words.every((w) => it.hay.includes(w));
            tile.hidden = !ok;
            if (ok) shown += 1;
        }
        count.textContent = `${shown}/${tiles.length}`;
        empty.hidden = shown > 0;
    };
    search.addEventListener("input", apply);
    folderSel.addEventListener("change", apply);
    closeBtn.onclick = close;

    // Keep the canvas and ComfyUI shortcuts from seeing picker input.
    for (const type of ["pointerdown", "mousedown", "click", "wheel", "contextmenu", "keyup", "keypress"]) {
        panel.addEventListener(type, (e) => e.stopPropagation());
    }
    panel.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Escape") {
            e.preventDefault();
            close();
        } else if (e.key === "Enter" && e.target === search) {
            e.preventDefault();
            const first = tiles.find(({ tile }) => !tile.hidden);
            if (first) finish(first.it.name);
        }
    });

    document.body.appendChild(panel);
    _openPicker = state;
    const width = Math.min(560, window.innerWidth - 16);
    panel.style.width = `${width}px`;
    panel.style.maxHeight = `${Math.min(480, window.innerHeight - 16)}px`;
    apply();
    const r = anchor.getBoundingClientRect();
    const h = panel.getBoundingClientRect().height;
    let top = r.bottom + 4;
    if (top + h > window.innerHeight - 8) top = Math.max(8, r.top - h - 4);
    panel.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - width - 8))}px`;
    panel.style.top = `${top}px`;
    document.addEventListener("pointerdown", onDocDown, true);
    window.addEventListener("resize", close);
    tiles.find(({ tile }) => tile.classList.contains("is-current"))?.tile.scrollIntoView({ block: "nearest" });
    setTimeout(() => search.focus(), 0);
}

/** Markup for the segment panel; mounted under the segment prompt row. */
export function segmentLoraTemplate() {
    return `
                <div class="bd-seg-loras-wrap" data-r="seg-loras-wrap">
                    <div class="bd-r2v-section-head">
                        <span class="bd-label bd-r2v-section-title" data-i18n="panel.segmentLoras">片段 LoRA</span>
                        <span class="bd-r2v-section-actions">
                            <button type="button" class="bd-r2v-pick-existing" data-r="seg-loras-add" data-i18n="panel.addLora" data-i18n-title="tooltip.segmentLoras">+ LoRA</button>
                            <span class="bd-r2v-section-count" data-r="seg-loras-count"></span>
                        </span>
                    </div>
                    <div class="bd-seg-loras" data-r="seg-loras"></div>
                </div>`;
}

/** Scoped styles, injected once. Keeps the rows readable inside the panel. */
export function ensureSegmentLoraStyles() {
    if (document.getElementById("mmx-seg-lora-css")) return;
    const style = document.createElement("style");
    style.id = "mmx-seg-lora-css";
    style.textContent = `
.bd-seg-loras { display:flex; flex-direction:column; gap:4px; margin-top:4px; }
.bd-seg-lora-row { display:flex; align-items:center; gap:6px; }
.bd-seg-lora-row .bd-seg-lora-strength { width:68px; flex:0 0 auto; }
.bd-seg-lora-row.bd-lora-off { opacity:0.45; }
.bd-seg-lora-pick { flex:1 1 auto; min-width:0; display:flex; align-items:center; gap:6px;
    background:#181818; border:1px solid #333; border-radius:4px; color:#eee;
    padding:2px 6px 2px 2px; font-size:11px; cursor:pointer; text-align:left; font-family:inherit; }
.bd-seg-lora-pick:hover { border-color:#4fff8f; }
.bd-seg-lora-label { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bd-seg-lora-pick.bd-seg-lora-missing .bd-seg-lora-label { color:#e0a06c; }
.bd-seg-lora-del { flex:0 0 auto; cursor:pointer; border:none; background:transparent;
    color:inherit; font-size:15px; line-height:1; padding:2px 5px; }
.bd-seg-lora-del:hover { color:#e06c6c; }
.mmx-lora-thumb { flex:0 0 auto; display:flex; align-items:center; justify-content:center;
    background:#0e0e0e; border-radius:3px; overflow:hidden; color:#555;
    font-size:8px; font-weight:700; letter-spacing:.06em; }
.mmx-lora-thumb img, .mmx-lora-thumb video { width:100%; height:100%; object-fit:cover; display:block; }
.mmx-lora-thumb-sm { width:30px; height:30px; }
.mmx-lora-thumb-lg { width:100%; aspect-ratio:1 / 1; border-radius:4px; font-size:12px; }
.mmx-lora-picker { position:fixed; z-index:10050; display:flex; flex-direction:column;
    background:#101010; border:1px solid #333; border-radius:10px; overflow:hidden;
    box-shadow:0 12px 32px rgba(0,0,0,.6); color:#eee; font-size:11px; }
.mmx-lora-picker-head { display:flex; align-items:center; gap:6px; padding:8px; border-bottom:1px solid #262626; }
.mmx-lora-picker-head input { flex:1 1 auto; min-width:0; background:#181818; border:1px solid #333;
    border-radius:4px; color:#eee; padding:5px 8px; font-size:12px; outline:none; }
.mmx-lora-picker-head input:focus { border-color:#4fff8f; }
.mmx-lora-picker-head select { background:#181818; border:1px solid #333; border-radius:4px;
    color:#eee; padding:4px 6px; font-size:11px; max-width:170px; }
.mmx-lora-picker-head select[hidden] { display:none; }
.mmx-lora-picker-count { color:#7d7d7d; font-variant-numeric:tabular-nums; white-space:nowrap; }
.mmx-lora-picker-close { background:transparent; border:none; color:#aaa; font-size:16px; cursor:pointer; padding:0 4px; }
.mmx-lora-picker-close:hover { color:#e06c6c; }
.mmx-lora-picker-grid { flex:1 1 auto; min-height:0; overflow-y:auto; display:grid;
    grid-template-columns:repeat(auto-fill, minmax(104px, 1fr)); gap:8px; padding:8px; }
.mmx-lora-tile { display:flex; flex-direction:column; gap:4px; padding:4px; background:#161616;
    border:1px solid #262626; border-radius:6px; color:#ddd; cursor:pointer; text-align:left; font:inherit; }
.mmx-lora-tile:hover { border-color:#4fff8f; }
.mmx-lora-tile.is-current { border-color:#4fff8f; box-shadow:0 0 0 1px #4fff8f inset; }
.mmx-lora-tile[hidden] { display:none; }
.mmx-lora-tile-name { font-size:11px; line-height:1.25; word-break:break-word; overflow:hidden;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
.mmx-lora-tile-folder { font-size:9px; color:#777; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mmx-lora-picker-empty { padding:16px; color:#777; text-align:center; }
.mmx-lora-picker-empty[hidden] { display:none; }
/* fl2v shot cards are only 220px wide */
.bd-fl2v-shot .bd-seg-lora-row { gap:4px; }
.bd-fl2v-shot .bd-seg-lora-strength { width:52px; }
.bd-fl2v-shot .mmx-lora-thumb-sm { width:24px; height:24px; }`;
    document.head.appendChild(style);
}

/** Self-contained section for one segment, used by the prompt-batch cards.
 *
 * The classic segment panel is only mounted for timeline-edited tasks; r2v and
 * the other prompt-batch modes edit each segment on its own card instead, so
 * the section has to be buildable standalone rather than bound to panel refs.
 */
export function createLoraSection(editor, seg, resolveSeg = null) {
    ensureSegmentLoraStyles();
    // fl2v rebuilds its shot objects on every sync, so a captured object can go
    // stale; resolveSeg returns the live one when the caller can provide it.
    const cur = () => (typeof resolveSeg === "function" && resolveSeg()) || seg;
    const wrap = document.createElement("div");
    wrap.className = "bd-seg-loras-wrap bd-r2v-section";

    const head = document.createElement("div");
    head.className = "bd-r2v-section-head";
    const title = document.createElement("span");
    title.className = "bd-label bd-r2v-section-title";
    title.textContent = t("panel.segmentLoras");
    const actions = document.createElement("span");
    actions.className = "bd-r2v-section-actions";
    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "bd-r2v-pick-existing";
    addBtn.textContent = t("panel.addLora");
    addBtn.title = t("tooltip.segmentLoras");
    const count = document.createElement("span");
    count.className = "bd-r2v-section-count";
    actions.append(addBtn, count);
    head.append(title, actions);

    const box = document.createElement("div");
    box.className = "bd-seg-loras";
    wrap.append(head, box);

    const paint = () => {
        const s = cur();
        const rows = normalizeLoraRows(s.loras);
        s.loras = rows;
        count.textContent = rows.length ? `${rows.length}/${MAX_SEGMENT_LORAS}` : "";
        box.innerHTML = "";
        if (!rows.length) return;
        loraCatalog().then((catalog) => {
            box.innerHTML = "";
            rows.forEach((row, idx) =>
                box.appendChild(buildLoraRow(editor, s, row, idx, catalog, paint)),
            );
        });
    };

    addBtn.onclick = async (e) => {
        e?.stopPropagation?.();
        const names = await loraNames();
        if (!names.length) {
            count.textContent = t("panel.noLorasFound");
            return;
        }
        if (normalizeLoraRows(cur().loras).length >= MAX_SEGMENT_LORAS) return;
        openLoraPicker(addBtn, null, (name) => {
            const s = cur();
            s.loras = normalizeLoraRows(s.loras);
            if (s.loras.length >= MAX_SEGMENT_LORAS) return;
            s.loras.push({ name, strength: 1, active: true });
            paint();
            editor.commit(true);
        });
    };

    paint();
    return wrap;
}

export function bindSegmentLoraRefs(ui) {
    ui.segLorasWrap = ui.root.querySelector('[data-r="seg-loras-wrap"]');
    ui.segLorasBox = ui.root.querySelector('[data-r="seg-loras"]');
    ui.segLorasAddBtn = ui.root.querySelector('[data-r="seg-loras-add"]');
    ui.segLorasCount = ui.root.querySelector('[data-r="seg-loras-count"]');
}

function currentSegment(ui) {
    return ui.timeline?.segments?.[ui.selectedIndex] || null;
}

export function bindSegmentLoraEvents(ui) {
    if (!ui.segLorasAddBtn) return;
    ui.segLorasAddBtn.onclick = async (e) => {
        e?.stopPropagation?.();
        const seg = currentSegment(ui);
        if (!seg) return;
        const names = await loraNames();
        if (!names.length) {
            if (ui.segLorasCount) ui.segLorasCount.textContent = t("panel.noLorasFound");
            return;
        }
        if (normalizeLoraRows(seg.loras).length >= MAX_SEGMENT_LORAS) return;
        openLoraPicker(ui.segLorasAddBtn, null, (name) => {
            seg.loras = normalizeLoraRows(seg.loras);
            if (seg.loras.length >= MAX_SEGMENT_LORAS) return;
            seg.loras.push({ name, strength: 1, active: true });
            renderSegmentLoras(ui, seg);
            ui.commit(true);
        });
    };
}

export function renderSegmentLoras(ui, seg) {
    const box = ui.segLorasBox;
    if (!box) return;
    box.innerHTML = "";
    if (!seg) return;

    const rows = normalizeLoraRows(seg.loras);
    seg.loras = rows;
    if (ui.segLorasCount) {
        ui.segLorasCount.textContent = rows.length ? `${rows.length}/${MAX_SEGMENT_LORAS}` : "";
    }
    if (!rows.length) return;

    loraCatalog().then((catalog) => {
        // The panel may have moved to another segment while the lists resolved.
        if (currentSegment(ui) !== seg) return;
        box.innerHTML = "";
        rows.forEach((row, idx) =>
            box.appendChild(buildLoraRow(ui, seg, row, idx, catalog, () => renderSegmentLoras(ui, seg))),
        );
    });
}

function buildLoraRow(ui, seg, row, idx, catalog, repaint) {
    const { names, previews } = catalog;
    const el = document.createElement("div");
    el.className = "bd-seg-lora-row";

    const on = document.createElement("input");
    on.type = "checkbox";
    on.checked = row.active !== false;
    on.title = t("tooltip.loraActive");
    on.onchange = () => {
        row.active = on.checked;
        el.classList.toggle("bd-lora-off", !on.checked);
        ui.commit(true);
    };

    // Thumbnail + name; opens the picker. An unknown name (moved/renamed file)
    // is kept and flagged so it is not lost.
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "bd-seg-lora-pick";
    const paintPick = () => {
        const missing = !names.includes(row.name);
        const label = document.createElement("span");
        label.className = "bd-seg-lora-label";
        label.textContent = splitLoraName(row.name).stem || row.name;
        pick.replaceChildren(makeThumb(row.name, previews[row.name], "mmx-lora-thumb-sm"), label);
        pick.classList.toggle("bd-seg-lora-missing", missing);
        pick.title = missing ? `${row.name}\n${t("panel.loraMissing")}` : `${row.name}\n${t("panel.pickLora")}`;
    };
    paintPick();
    playOnHover(pick);
    pick.onclick = (e) => {
        e?.stopPropagation?.();
        openLoraPicker(pick, row.name, (name) => {
            row.name = name;
            paintPick();
            ui.commit(true);
        });
    };
    pick.addEventListener("keydown", (e) => e.stopPropagation());

    const strength = document.createElement("input");
    strength.type = "number";
    strength.className = "bd-num bd-seg-lora-strength";
    strength.step = "0.05";
    strength.min = String(MIN_STRENGTH);
    strength.max = String(MAX_STRENGTH);
    strength.value = String(row.strength);
    strength.title = t("tooltip.loraStrength");
    const applyStrength = () => {
        row.strength = clampStrength(strength.value);
        strength.value = String(row.strength);
        ui.commit(true);
    };
    strength.onchange = applyStrength;
    strength.addEventListener("keydown", (e) => e.stopPropagation());
    strength.addEventListener("keyup", (e) => e.stopPropagation());

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "bd-seg-lora-del";
    remove.textContent = "×";
    remove.title = t("tooltip.removeLora");
    remove.onclick = (e) => {
        e?.stopPropagation?.();
        seg.loras.splice(idx, 1);
        if (repaint) repaint();
        else renderSegmentLoras(ui, seg);
        ui.commit(true);
    };

    if (row.active === false) el.classList.add("bd-lora-off");
    el.append(on, pick, strength, remove);
    return el;
}
