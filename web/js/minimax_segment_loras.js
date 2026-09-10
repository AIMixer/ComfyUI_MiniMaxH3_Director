/** Per-segment LoRA rows for the MiniMax H3 Director segment panel.
 *
 * A segment stores its stack as `segment.loras = [{name, strength, active}]`.
 * The backend (director/segment_loras.py) patches those onto that segment's
 * base MODEL right before sampling, so a single Director run can change LoRAs
 * shot by shot while the inter-segment guide still spans the whole timeline.
 *
 * LoRAs are chosen in a gallery with previews and trigger words, served by
 * director/lora_previews.py (sidecar image / video files, LoRA Manager
 * metadata, the LoRA's own safetensors header). Each row lists its trigger
 * words: a click adds one to that segment's prompt, Shift+click copies it.
 */

import { api } from "../../scripts/api.js";
import { t } from "./minimax_i18n.js";

export const MAX_SEGMENT_LORAS = 8;
const MIN_STRENGTH = -10;
const MAX_STRENGTH = 10;

let _loraNamesPromise = null;
let _loraInfoPromise = null;
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

/** `{name: {preview, triggers, base, title, tags}}` from director/lora_previews.py. */
export function loraInfo() {
    if (!_loraInfoPromise) {
        _loraInfoPromise = api
            .fetchApi("/minimax/director/lora_previews")
            .then((r) => (r.ok ? r.json() : null))
            .then((j) => {
                if (j?.catalog && typeof j.catalog === "object") return j.catalog;
                const out = {};
                for (const [name, preview] of Object.entries(j?.previews || {})) {
                    out[name] = { preview, triggers: [], tags: [] };
                }
                return out;
            })
            .catch(() => ({}));
    }
    return _loraInfoPromise;
}

/** `{name: {kind, v}}` for LoRAs that have a preview file. */
export function loraPreviews() {
    return loraInfo().then((info) => {
        const out = {};
        for (const [name, meta] of Object.entries(info)) if (meta?.preview) out[name] = meta.preview;
        return out;
    });
}

/** Drop the cached lists so a newly copied LoRA or preview shows up without a reload. */
export function refreshLoraNames() {
    _loraNamesPromise = null;
    _loraInfoPromise = null;
    return loraNames();
}

function loraCatalog() {
    return Promise.all([loraNames(), loraInfo()]).then(([names, info]) => ({ names, info }));
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

/** "MM-H3 - Passionate Kiss" -> "PK" for tiles without a preview. */
function loraInitials(stem) {
    const tail = stem.includes(" - ") ? stem.slice(stem.lastIndexOf(" - ") + 3) : stem;
    const words = tail.split(/[\s_\-.()]+/).filter((w) => /^[a-z]/i.test(w) && !/^v\d/i.test(w));
    return (words.slice(0, 2).map((w) => w[0]).join("") || stem.slice(0, 2)).toUpperCase();
}

function loraHue(text) {
    let h = 0;
    for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return h % 360;
}

/** Base model from metadata, else the top-level folder ("MiniMax H3", "Krea 2"). */
function baseLabel(name, meta) {
    return meta?.base || splitLoraName(name).folder.split("/")[0] || "";
}

function badge(text, cls = "") {
    const el = document.createElement("span");
    el.className = `mmx-lora-badge ${cls}`.trim();
    el.textContent = text;
    return el;
}

function previewURL(name, preview) {
    const v = encodeURIComponent(String(preview?.v ?? 0));
    return api.apiURL(`/minimax/director/lora_preview?name=${encodeURIComponent(name)}&v=${v}`);
}

function emptyThumb(box, name) {
    const { stem } = splitLoraName(name);
    box.replaceChildren();
    box.classList.add("mmx-lora-thumb-empty");
    box.style.setProperty("--mmx-hue", String(loraHue(stem)));
    box.textContent = loraInitials(stem);
}

function makeThumb(name, meta, sizeClass) {
    const box = document.createElement("span");
    box.className = `mmx-lora-thumb ${sizeClass}`;
    const preview = meta?.preview;
    if (!preview) {
        emptyThumb(box, name);
        return box;
    }
    const src = previewURL(name, preview);
    if (preview.kind === "video") {
        const video = document.createElement("video");
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = "metadata";
        video.src = `${src}#t=0.1`;
        video.onerror = () => emptyThumb(box, name);
        box.appendChild(video);
    } else {
        const img = document.createElement("img");
        img.alt = "";
        img.loading = "lazy";
        img.decoding = "async";
        img.src = src;
        img.onerror = () => emptyThumb(box, name);
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

/** Small confirmation bubble above `anchor`. */
function flash(anchor, text) {
    if (!anchor?.isConnected) return;
    const tip = document.createElement("div");
    tip.className = "mmx-lora-flash";
    tip.textContent = text;
    const r = anchor.getBoundingClientRect();
    tip.style.left = `${r.left + r.width / 2}px`;
    tip.style.top = `${r.top - 4}px`;
    document.body.appendChild(tip);
    setTimeout(() => tip.classList.add("out"), 650);
    setTimeout(() => tip.remove(), 950);
}

async function copyText(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        const area = document.createElement("textarea");
        area.value = text;
        area.style.cssText = "position:fixed;left:-9999px;opacity:0";
        document.body.appendChild(area);
        area.select();
        let ok = false;
        try {
            ok = document.execCommand("copy");
        } catch {
            ok = false;
        }
        area.remove();
        return ok;
    }
}

/** Positive-prompt textarea of the segment whose LoRA section contains `el`, if reachable. */
function promptFieldFor(el, ui) {
    const card = el?.closest?.(".bd-batch-card");
    if (card) return card.querySelector('textarea[data-f="prompt"]');
    if (ui?.segPrompt && ui.segLorasBox?.contains?.(el)) return ui.segPrompt;
    return null;
}

/** Append the words the prompt does not already contain; returns how many were added. */
function appendToPrompt(field, words) {
    if (!field) return 0;
    const current = field.value || "";
    const lower = current.toLowerCase();
    const missing = words.filter((w) => w && !lower.includes(w.toLowerCase()));
    if (!missing.length) return 0;
    const trimmed = current.replace(/\s+$/, "");
    const sep = !trimmed ? "" : /[,;.!?]$/.test(trimmed) ? " " : ", ";
    field.value = trimmed + sep + missing.join(", ");
    // r2v prompts render through the @-mention token editor; redraw it from the textarea.
    field.__bdTokenApi?.hydrateFromValue?.(field.value);
    field.dispatchEvent(new Event("input", { bubbles: true }));
    return missing.length;
}

async function addTriggersToPrompt(field, name) {
    if (!field) return 0;
    const words = (await loraInfo())[name]?.triggers || [];
    return words.length ? appendToPrompt(field, words) : 0;
}

export function closeLoraPicker() {
    _openPicker?.close();
}

/** LoRA gallery. Calls onPick(name, {addTriggers}) with the chosen LoRA.
 *
 * `opts.promptField()` returns the segment's prompt textarea when the caller has
 * one, which enables "Use + add trigger words".
 */
export async function openLoraPicker(anchor, current, onPick, opts = {}) {
    if (_openPicker && _openPicker.anchor === anchor) {
        closeLoraPicker();
        return;
    }
    closeLoraPicker();
    ensureSegmentLoraStyles();
    const { names, info } = await loraCatalog();
    if (!anchor?.isConnected) return;
    const promptField = typeof opts.promptField === "function" ? opts.promptField : () => null;

    const state = { anchor, close: null };
    const overlay = document.createElement("div");
    overlay.className = "mmx-lora-overlay";
    const dialog = document.createElement("div");
    dialog.className = "mmx-lora-picker";
    overlay.appendChild(dialog);

    const head = document.createElement("div");
    head.className = "mmx-lora-picker-head";
    const title = document.createElement("span");
    title.className = "mmx-lora-picker-title";
    title.textContent = t("panel.loraLibrary");
    const count = document.createElement("span");
    count.className = "mmx-lora-picker-count";
    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = t("panel.searchLoras");
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "mmx-lora-picker-close";
    closeBtn.textContent = "×";
    head.append(title, count, search, closeBtn);

    const items = names.map((name) => {
        const meta = info[name] || {};
        const hay = [name, meta.title, meta.base, ...(meta.triggers || []), ...(meta.tags || [])]
            .filter(Boolean)
            .join(" ")
            .toLowerCase();
        return { name, ...splitLoraName(name), meta, hay };
    });

    let folder = "";
    const folders = [...new Set(items.map((it) => it.folder))].sort((a, b) => a.localeCompare(b));
    const folderBar = document.createElement("div");
    folderBar.className = "mmx-lora-folders";
    folderBar.hidden = folders.length < 2;
    const folderChips = [];
    for (const [value, label] of [["", t("panel.allFolders")], ...folders.map((f) => [f, f || "/"])]) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "mmx-lora-folder";
        if (value === "") chip.classList.add("is-on");
        chip.textContent = label;
        chip.onclick = () => {
            folder = value;
            for (const c of folderChips) c.classList.toggle("is-on", c === chip);
            apply();
        };
        folderChips.push(chip);
        folderBar.appendChild(chip);
    }

    const body = document.createElement("div");
    body.className = "mmx-lora-picker-body";
    const gridWrap = document.createElement("div");
    gridWrap.className = "mmx-lora-picker-gridwrap";
    const grid = document.createElement("div");
    grid.className = "mmx-lora-picker-grid";
    const empty = document.createElement("div");
    empty.className = "mmx-lora-picker-empty";
    empty.textContent = names.length ? t("panel.noLoraMatch") : t("panel.noLorasFound");
    empty.hidden = true;
    gridWrap.append(grid, empty);
    const detail = document.createElement("div");
    detail.className = "mmx-lora-detail";
    body.append(gridWrap, detail);
    dialog.append(head, folderBar, body);

    const close = () => {
        if (_openPicker !== state) return;
        _openPicker = null;
        overlay.querySelectorAll("video").forEach((v) => v.pause());
        overlay.remove();
    };
    state.close = close;
    const finish = (name, how = {}) => {
        close();
        onPick?.(name, how);
    };

    let focused = null;
    let tiles = [];
    const showDetail = (it) => {
        if (!it || focused === it) return;
        focused = it;
        for (const { it: other, tile } of tiles) tile.classList.toggle("is-focus", other === it);
        detail.querySelectorAll("video").forEach((v) => v.pause());

        const media = makeThumb(it.name, it.meta, "mmx-lora-thumb-xl");
        const video = media.querySelector("video");
        if (video) {
            video.autoplay = true;
            video.play?.()?.catch?.(() => {});
        }
        const name = document.createElement("div");
        name.className = "mmx-lora-detail-name";
        name.textContent = it.stem;
        const sub = document.createElement("div");
        sub.className = "mmx-lora-detail-sub";
        sub.textContent = [it.meta.title, it.folder].filter(Boolean).join(" · ");
        const badges = document.createElement("div");
        badges.className = "mmx-lora-detail-badges";
        const base = baseLabel(it.name, it.meta);
        if (base) badges.appendChild(badge(base, "is-base"));
        for (const tag of it.meta.tags || []) badges.appendChild(badge(tag));

        const label = document.createElement("div");
        label.className = "mmx-lora-detail-label";
        label.textContent = t("panel.triggerWords");
        const trig = document.createElement("div");
        trig.className = "mmx-lora-detail-triggers";
        const words = it.meta.triggers || [];
        if (words.length) {
            for (const word of words) {
                const chip = document.createElement("button");
                chip.type = "button";
                chip.className = "mmx-trigger-chip";
                chip.textContent = word;
                chip.title = t("tooltip.triggerCopy");
                chip.onclick = async () => flash(chip, (await copyText(word)) ? t("panel.copied") : "✗");
                trig.appendChild(chip);
            }
            if (words.length > 1) {
                const all = document.createElement("button");
                all.type = "button";
                all.className = "mmx-trigger-chip is-ghost";
                all.textContent = t("panel.copyAll");
                all.onclick = async () => flash(all, (await copyText(words.join(", "))) ? t("panel.copied") : "✗");
                trig.appendChild(all);
            }
        } else {
            trig.classList.add("is-empty");
            trig.textContent = t("panel.noTriggerWords");
        }

        const actions = document.createElement("div");
        actions.className = "mmx-lora-detail-actions";
        const use = document.createElement("button");
        use.type = "button";
        use.className = "mmx-lora-use";
        use.textContent = it.name === current ? t("panel.inUse") : t("panel.useLora");
        use.onclick = () => finish(it.name);
        actions.appendChild(use);
        if (words.length && promptField()) {
            const useAdd = document.createElement("button");
            useAdd.type = "button";
            useAdd.className = "mmx-lora-use is-secondary";
            useAdd.textContent = t("panel.useLoraWithTriggers");
            useAdd.onclick = () => finish(it.name, { addTriggers: true });
            actions.appendChild(useAdd);
        }
        detail.replaceChildren(media, name, sub, badges, label, trig, actions);
    };

    tiles = items.map((it, i) => {
        const tile = document.createElement("button");
        tile.type = "button";
        tile.className = "mmx-lora-tile";
        tile.style.setProperty("--mmx-i", String(Math.min(i, 24)));
        tile.title = it.name;
        const top = document.createElement("span");
        top.className = "mmx-lora-tile-top";
        if (it.meta.preview?.kind === "video") top.appendChild(badge("▶", "is-video"));
        if (it.name === current) {
            tile.classList.add("is-current");
            top.appendChild(badge("✓", "is-tick"));
        }
        const base = baseLabel(it.name, it.meta);
        if (base) top.appendChild(badge(base, "is-base"));
        const shade = document.createElement("span");
        shade.className = "mmx-lora-tile-shade";
        const label = document.createElement("span");
        label.className = "mmx-lora-tile-name";
        label.textContent = it.stem;
        const meta = document.createElement("span");
        meta.className = "mmx-lora-tile-meta";
        const n = (it.meta.triggers || []).length;
        meta.textContent = [it.folder.split("/").pop(), n ? `✦ ${n}` : ""].filter(Boolean).join("  ·  ");
        shade.append(label, meta);
        tile.append(makeThumb(it.name, it.meta, "mmx-lora-thumb-lg"), top, shade);
        playOnHover(tile);
        let rest = 0;
        tile.addEventListener("mouseenter", () => {
            clearTimeout(rest);
            rest = setTimeout(() => showDetail(it), 90);
        });
        tile.addEventListener("mouseleave", () => clearTimeout(rest));
        tile.addEventListener("focus", () => showDetail(it));
        tile.onclick = () => finish(it.name);
        grid.appendChild(tile);
        return { it, tile };
    });

    const visible = () => tiles.filter(({ tile }) => !tile.hidden);
    const apply = () => {
        const words = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
        let shown = 0;
        let first = null;
        for (const { it, tile } of tiles) {
            const ok = (!folder || it.folder === folder) && words.every((w) => it.hay.includes(w));
            tile.hidden = !ok;
            if (ok) {
                shown += 1;
                first ||= it;
            }
        }
        count.textContent = `${shown}/${tiles.length}`;
        empty.hidden = shown > 0;
        const focusedTile = tiles.find((x) => x.it === focused)?.tile;
        if (!focusedTile || focusedTile.hidden) showDetail(first);
    };
    search.addEventListener("input", apply);
    closeBtn.onclick = close;

    const columns = () => {
        const vis = visible();
        if (!vis.length) return 1;
        const top0 = vis[0].tile.offsetTop;
        return Math.max(1, vis.filter(({ tile }) => tile.offsetTop === top0).length);
    };
    const move = (delta) => {
        const vis = visible();
        if (!vis.length) return;
        const at = vis.findIndex(({ it }) => it === focused);
        const next = vis[at < 0 ? 0 : Math.max(0, Math.min(vis.length - 1, at + delta))];
        next.tile.focus();
        next.tile.scrollIntoView({ block: "nearest" });
    };

    // Keep the canvas and ComfyUI shortcuts from seeing gallery input.
    for (const type of ["pointerdown", "mousedown", "click", "dblclick", "wheel", "contextmenu", "keyup", "keypress"]) {
        overlay.addEventListener(type, (e) => e.stopPropagation());
    }
    overlay.addEventListener("pointerdown", (e) => {
        if (e.target === overlay) close();
    });
    overlay.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Escape") {
            e.preventDefault();
            close();
        } else if (e.key === "Enter" && e.target === search) {
            e.preventDefault();
            const focusedTile = tiles.find((x) => x.it === focused)?.tile;
            const pick = focusedTile && !focusedTile.hidden ? focused : visible()[0]?.it;
            if (pick) finish(pick.name);
        } else if (e.key === "ArrowDown" && e.target === search) {
            e.preventDefault();
            move(0);
        } else if (e.target !== search) {
            const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: columns(), ArrowUp: -columns() }[e.key];
            if (step) {
                e.preventDefault();
                move(step);
            }
        }
    });

    document.body.appendChild(overlay);
    _openPicker = state;
    const currentItem = items.find((it) => it.name === current);
    showDetail(currentItem);
    apply();
    tiles.find(({ it }) => it === currentItem)?.tile.scrollIntoView({ block: "center" });
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

/** Scoped styles, injected once. */
export function ensureSegmentLoraStyles() {
    if (document.getElementById("mmx-seg-lora-css")) return;
    const style = document.createElement("style");
    style.id = "mmx-seg-lora-css";
    style.textContent = `
.bd-seg-loras { display:flex; flex-direction:column; gap:6px; margin-top:4px; }
.bd-seg-lora-item { display:flex; flex-direction:column; gap:4px; }
.bd-seg-lora-item.bd-lora-off { opacity:0.45; }
.bd-seg-lora-row { display:flex; align-items:center; gap:6px; }
.bd-seg-lora-row .bd-seg-lora-strength { width:68px; flex:0 0 auto; }
.bd-seg-lora-pick { flex:1 1 auto; min-width:0; display:flex; align-items:center; gap:6px;
    background:#181818; border:1px solid #333; border-radius:6px; color:#eee;
    padding:2px 8px 2px 2px; font-size:11px; cursor:pointer; text-align:left; font-family:inherit;
    transition:border-color .15s, box-shadow .15s; }
.bd-seg-lora-pick:hover { border-color:#4fff8f; box-shadow:0 0 0 2px rgba(79,255,143,.12); }
.bd-seg-lora-label { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bd-seg-lora-pick.bd-seg-lora-missing .bd-seg-lora-label { color:#e0a06c; }
.bd-seg-lora-del { flex:0 0 auto; cursor:pointer; border:none; background:transparent;
    color:inherit; font-size:15px; line-height:1; padding:2px 5px; }
.bd-seg-lora-del:hover { color:#e06c6c; }
.bd-seg-lora-triggers { display:flex; flex-wrap:wrap; gap:4px; padding-left:24px; }
.bd-seg-lora-triggers .mmx-trigger-chip { font-size:10px; padding:2px 6px; }

.mmx-lora-thumb { flex:0 0 auto; display:flex; align-items:center; justify-content:center; overflow:hidden;
    background:#0e0e0e; border-radius:4px; color:#555; font-weight:800; }
.mmx-lora-thumb img, .mmx-lora-thumb video { width:100%; height:100%; object-fit:cover; display:block; }
.mmx-lora-thumb-empty { letter-spacing:.04em;
    background:linear-gradient(135deg, hsl(var(--mmx-hue, 150) 45% 24%), hsl(calc(var(--mmx-hue, 150) + 50) 55% 9%));
    color:hsl(var(--mmx-hue, 150) 80% 82%); }
.mmx-lora-thumb-sm { width:30px; height:30px; font-size:10px; }
.mmx-lora-thumb-lg { font-size:30px; }
.mmx-lora-thumb-xl { width:100%; aspect-ratio:4 / 5; border-radius:10px; font-size:52px;
    box-shadow:0 10px 26px rgba(0,0,0,.55); flex:0 0 auto; }

.mmx-lora-overlay { position:fixed; inset:0; z-index:10050; display:flex; align-items:center; justify-content:center;
    background:rgba(4,6,5,.62); backdrop-filter:blur(4px); animation:mmx-fade .16s ease-out; }
.mmx-lora-picker { width:min(1000px, 96vw); height:min(660px, 90vh); display:grid; grid-template-rows:auto auto minmax(0, 1fr);
    background:linear-gradient(180deg, #151515, #0c0c0c); border:1px solid #2c2c2c; border-radius:14px; overflow:hidden;
    box-shadow:0 24px 60px rgba(0,0,0,.65), 0 0 0 1px rgba(79,255,143,.07); color:#eee; font-size:11px;
    animation:mmx-pop .2s cubic-bezier(.2,.9,.3,1.15); }
@keyframes mmx-fade { from { opacity:0; } }
@keyframes mmx-pop { from { opacity:0; transform:scale(.97) translateY(8px); } }
.mmx-lora-picker-head { display:flex; align-items:center; gap:10px; padding:12px 14px 9px; }
.mmx-lora-picker-title { font-size:13px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:#fff; }
.mmx-lora-picker-count { color:#6f6f6f; font-variant-numeric:tabular-nums; }
.mmx-lora-picker-head input { flex:1 1 auto; min-width:0; max-width:380px; margin-left:auto; background:#1a1a1a;
    border:1px solid #333; border-radius:8px; color:#eee; padding:7px 10px; font-size:12px; outline:none;
    transition:border-color .15s, box-shadow .15s; }
.mmx-lora-picker-head input:focus { border-color:#4fff8f; box-shadow:0 0 0 3px rgba(79,255,143,.15); }
.mmx-lora-picker-close { background:transparent; border:none; color:#999; font-size:20px; line-height:1; cursor:pointer; padding:0 4px; }
.mmx-lora-picker-close:hover { color:#e06c6c; }
.mmx-lora-folders { display:flex; flex-wrap:wrap; gap:6px; padding:0 14px 10px; }
.mmx-lora-folders[hidden] { display:none; }
.mmx-lora-folder { flex:0 0 auto; background:#191919; border:1px solid #2e2e2e; color:#bbb; border-radius:999px;
    padding:3px 10px; font-size:10px; cursor:pointer; white-space:nowrap; font-family:inherit; transition:all .15s; }
.mmx-lora-folder:hover { border-color:#4fff8f; color:#fff; }
.mmx-lora-folder.is-on { background:#4fff8f; border-color:#4fff8f; color:#062b14; font-weight:700; }
.mmx-lora-picker-body { display:grid; grid-template-columns:minmax(0, 1fr) 300px; min-height:0; border-top:1px solid #222; }
.mmx-lora-picker-gridwrap { min-height:0; overflow-y:auto; padding:12px 14px; }
.mmx-lora-picker-gridwrap, .mmx-lora-detail { scrollbar-width:thin; scrollbar-color:#3a3a3a transparent; }
.mmx-lora-picker-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(128px, 1fr)); gap:10px; }
.mmx-lora-tile { position:relative; display:block; padding:0; aspect-ratio:3 / 4; border-radius:10px; overflow:hidden;
    cursor:pointer; background:#151515; border:1px solid #262626; color:#fff; text-align:left; font:inherit;
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    animation:mmx-rise .3s ease-out both; animation-delay:calc(var(--mmx-i, 0) * 14ms); }
@keyframes mmx-rise { from { opacity:0; transform:translateY(8px); } }
.mmx-lora-tile:hover, .mmx-lora-tile.is-focus { transform:translateY(-3px); border-color:#4fff8f;
    box-shadow:0 10px 24px rgba(0,0,0,.5), 0 0 0 1px rgba(79,255,143,.35); }
.mmx-lora-tile:focus-visible { outline:2px solid #4fff8f; outline-offset:2px; }
.mmx-lora-tile.is-current { border-color:#4fff8f; box-shadow:0 0 0 2px #4fff8f inset; }
.mmx-lora-tile[hidden] { display:none; }
.mmx-lora-tile .mmx-lora-thumb-lg { position:absolute; inset:0; width:100%; height:100%; border-radius:0; }
.mmx-lora-tile .mmx-lora-thumb img, .mmx-lora-tile .mmx-lora-thumb video { transition:transform .35s ease; }
.mmx-lora-tile:hover .mmx-lora-thumb img, .mmx-lora-tile:hover .mmx-lora-thumb video { transform:scale(1.06); }
.mmx-lora-tile-top { position:absolute; top:6px; left:6px; right:6px; display:flex; gap:4px; pointer-events:none; }
.mmx-lora-tile-top .is-base { margin-left:auto; }
.mmx-lora-tile-shade { position:absolute; left:0; right:0; bottom:0; padding:24px 8px 7px; display:flex; flex-direction:column;
    gap:2px; pointer-events:none; background:linear-gradient(180deg, transparent, rgba(0,0,0,.55) 40%, rgba(0,0,0,.92)); }
.mmx-lora-tile-name { font-size:11px; font-weight:700; line-height:1.2; text-shadow:0 1px 2px #000; word-break:break-word;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.mmx-lora-tile-meta { font-size:9px; color:#9fdcb8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mmx-lora-badge { display:inline-flex; align-items:center; padding:1px 6px; border-radius:999px; font-size:9px; font-weight:700;
    background:rgba(0,0,0,.62); color:#ddd; border:1px solid rgba(255,255,255,.14); white-space:nowrap;
    max-width:100%; overflow:hidden; text-overflow:ellipsis; }
.mmx-lora-badge.is-base { color:#4fff8f; border-color:rgba(79,255,143,.35); }
.mmx-lora-badge.is-tick { background:#4fff8f; color:#062b14; border-color:#4fff8f; }
.mmx-lora-detail { min-height:0; overflow-y:auto; padding:14px; border-left:1px solid #222; background:#0b0b0b;
    display:flex; flex-direction:column; gap:8px; }
.mmx-lora-detail-name { font-size:14px; font-weight:800; line-height:1.25; word-break:break-word; }
.mmx-lora-detail-sub { font-size:10px; color:#888; word-break:break-word; }
.mmx-lora-detail-badges { display:flex; flex-wrap:wrap; gap:4px; }
.mmx-lora-detail-badges .mmx-lora-badge { background:#181818; }
.mmx-lora-detail-label { margin-top:4px; font-size:10px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:#8a8a8a; }
.mmx-lora-detail-triggers { display:flex; flex-wrap:wrap; gap:5px; }
.mmx-lora-detail-triggers.is-empty { color:#666; font-style:italic; }
.mmx-trigger-chip { background:rgba(79,255,143,.08); border:1px solid rgba(79,255,143,.35); color:#bfffd6; border-radius:6px;
    padding:3px 7px; font-size:10.5px; line-height:1.3; cursor:pointer; text-align:left; font-family:inherit;
    max-width:100%; word-break:break-word; transition:background .15s, color .15s, transform .1s; }
.mmx-trigger-chip:hover { background:rgba(79,255,143,.2); color:#fff; }
.mmx-trigger-chip:active { transform:scale(.97); }
.mmx-trigger-chip.is-ghost { background:transparent; border-style:dashed; color:#8fcfaa; }
.mmx-lora-detail-actions { margin-top:auto; display:flex; flex-direction:column; gap:6px; padding-top:8px; }
.mmx-lora-use { background:#4fff8f; color:#062b14; border:1px solid #4fff8f; border-radius:8px; padding:8px 10px; font-size:12px;
    font-weight:800; cursor:pointer; font-family:inherit; transition:filter .15s, transform .1s; }
.mmx-lora-use:hover { filter:brightness(1.08); }
.mmx-lora-use:active { transform:scale(.98); }
.mmx-lora-use.is-secondary { background:transparent; color:#4fff8f; }
.mmx-lora-picker-empty { padding:40px 16px; color:#777; text-align:center; font-size:12px; }
.mmx-lora-picker-empty[hidden] { display:none; }
@media (max-width: 760px) {
    .mmx-lora-picker-body { grid-template-columns:minmax(0, 1fr); }
    .mmx-lora-detail { display:none; }
}
.mmx-lora-flash { position:fixed; z-index:10060; transform:translate(-50%, -100%); background:#4fff8f; color:#062b14;
    font-size:10px; font-weight:800; padding:3px 8px; border-radius:6px; pointer-events:none; white-space:nowrap;
    box-shadow:0 4px 12px rgba(0,0,0,.4); animation:mmx-flash-in .15s ease-out; transition:opacity .25s, transform .25s; }
.mmx-lora-flash.out { opacity:0; transform:translate(-50%, -150%); }
@keyframes mmx-flash-in { from { opacity:0; transform:translate(-50%, -60%); } }
/* fl2v shot cards are only 220px wide */
.bd-fl2v-shot .bd-seg-lora-row { gap:4px; }
.bd-fl2v-shot .bd-seg-lora-strength { width:52px; }
.bd-fl2v-shot .mmx-lora-thumb-sm { width:24px; height:24px; font-size:9px; }
.bd-fl2v-shot .bd-seg-lora-triggers { padding-left:0; }`;
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
        const promptField = () => promptFieldFor(wrap, editor);
        openLoraPicker(
            addBtn,
            null,
            (name, how) => {
                const field = promptField();
                const s = cur();
                s.loras = normalizeLoraRows(s.loras);
                if (s.loras.length >= MAX_SEGMENT_LORAS) return;
                s.loras.push({ name, strength: 1, active: true });
                paint();
                editor.commit(true);
                if (how?.addTriggers) addTriggersToPrompt(field, name);
            },
            { promptField },
        );
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
        openLoraPicker(
            ui.segLorasAddBtn,
            null,
            (name, how) => {
                seg.loras = normalizeLoraRows(seg.loras);
                if (seg.loras.length >= MAX_SEGMENT_LORAS) return;
                seg.loras.push({ name, strength: 1, active: true });
                renderSegmentLoras(ui, seg);
                ui.commit(true);
                if (how?.addTriggers) addTriggersToPrompt(ui.segPrompt || null, name);
            },
            { promptField: () => ui.segPrompt || null },
        );
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
    const { names, info } = catalog;
    const meta = info[row.name] || {};
    const item = document.createElement("div");
    item.className = "bd-seg-lora-item";
    const el = document.createElement("div");
    el.className = "bd-seg-lora-row";
    const promptField = () => promptFieldFor(item, ui);
    const redraw = () => (repaint ? repaint() : renderSegmentLoras(ui, seg));

    const on = document.createElement("input");
    on.type = "checkbox";
    on.checked = row.active !== false;
    on.title = t("tooltip.loraActive");
    on.onchange = () => {
        row.active = on.checked;
        item.classList.toggle("bd-lora-off", !on.checked);
        ui.commit(true);
    };

    // Thumbnail + name; opens the gallery. An unknown name (moved/renamed file)
    // is kept and flagged so it is not lost.
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "bd-seg-lora-pick";
    const missing = !names.includes(row.name);
    const label = document.createElement("span");
    label.className = "bd-seg-lora-label";
    label.textContent = splitLoraName(row.name).stem || row.name;
    pick.append(makeThumb(row.name, meta, "mmx-lora-thumb-sm"), label);
    pick.classList.toggle("bd-seg-lora-missing", missing);
    pick.title = missing ? `${row.name}\n${t("panel.loraMissing")}` : `${row.name}\n${t("panel.pickLora")}`;
    playOnHover(pick);
    pick.onclick = (e) => {
        e?.stopPropagation?.();
        openLoraPicker(
            pick,
            row.name,
            (name, how) => {
                const field = promptField();
                row.name = name;
                ui.commit(true);
                if (how?.addTriggers) addTriggersToPrompt(field, name);
                redraw();
            },
            { promptField },
        );
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
        redraw();
        ui.commit(true);
    };

    el.append(on, pick, strength, remove);
    item.appendChild(el);

    // Trigger words: click adds to this segment's prompt, Shift+click (or no
    // reachable prompt, e.g. fl2v shot cards) copies.
    const words = meta.triggers || [];
    if (words.length) {
        const strip = document.createElement("div");
        strip.className = "bd-seg-lora-triggers";
        const use = async (chip, list, e) => {
            e.stopPropagation();
            const field = promptField();
            if (field && !e.shiftKey) {
                flash(chip, appendToPrompt(field, list) ? t("panel.addedToPrompt") : t("panel.alreadyInPrompt"));
            } else {
                flash(chip, (await copyText(list.join(", "))) ? t("panel.copied") : "✗");
            }
        };
        const hint = (chip, allWords) => () => {
            chip.title = promptField()
                ? t(allWords ? "tooltip.addAllTriggers" : "tooltip.triggerInsert")
                : t("tooltip.triggerCopy");
        };
        for (const word of words) {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "mmx-trigger-chip";
            chip.textContent = `✦ ${word}`;
            chip.onmouseenter = hint(chip, false);
            chip.onclick = (e) => use(chip, [word], e);
            strip.appendChild(chip);
        }
        if (words.length > 1) {
            const all = document.createElement("button");
            all.type = "button";
            all.className = "mmx-trigger-chip is-ghost";
            all.textContent = t("panel.addAllTriggers");
            all.onmouseenter = hint(all, true);
            all.onclick = (e) => use(all, words, e);
            strip.appendChild(all);
        }
        item.appendChild(strip);
    }

    if (row.active === false) item.classList.add("bd-lora-off");
    return item;
}
