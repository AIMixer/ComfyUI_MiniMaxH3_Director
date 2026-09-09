/** Per-segment LoRA rows for the MiniMax H3 Director segment panel.
 *
 * A segment stores its stack as `segment.loras = [{name, strength, active}]`.
 * The backend (director/segment_loras.py) patches those onto that segment's
 * base MODEL right before sampling, so a single Director run can change LoRAs
 * shot by shot while the inter-segment guide still spans the whole timeline.
 */

import { api } from "../../scripts/api.js";
import { t } from "./minimax_i18n.js";

export const MAX_SEGMENT_LORAS = 8;
const MIN_STRENGTH = -10;
const MAX_STRENGTH = 10;

let _loraNamesPromise = null;

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

/** Drop the cached list so a newly copied .safetensors shows up without a reload. */
export function refreshLoraNames() {
    _loraNamesPromise = null;
    return loraNames();
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
.bd-seg-lora-row .bd-seg-lora-name { flex:1 1 auto; min-width:0; }
.bd-seg-lora-row .bd-seg-lora-strength { width:68px; flex:0 0 auto; }
.bd-seg-lora-row.bd-lora-off { opacity:0.45; }
.bd-seg-lora-del { flex:0 0 auto; cursor:pointer; border:none; background:transparent;
    color:inherit; font-size:15px; line-height:1; padding:2px 5px; }
.bd-seg-lora-del:hover { color:#e06c6c; }`;
    document.head.appendChild(style);
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
        seg.loras = normalizeLoraRows(seg.loras);
        if (seg.loras.length >= MAX_SEGMENT_LORAS) return;
        seg.loras.push({ name: names[0], strength: 1, active: true });
        renderSegmentLoras(ui, seg);
        ui.commit(true);
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

    loraNames().then((names) => {
        // The panel may have moved to another segment while the list resolved.
        if (currentSegment(ui) !== seg) return;
        box.innerHTML = "";
        rows.forEach((row, idx) => box.appendChild(buildLoraRow(ui, seg, row, idx, names)));
    });
}

function buildLoraRow(ui, seg, row, idx, names) {
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

    const select = document.createElement("select");
    select.className = "bd-select bd-seg-lora-name";
    // An unknown name (moved/renamed file) stays selectable so it is not lost.
    const options = names.includes(row.name) ? names : [row.name, ...names];
    for (const name of options) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        if (name === row.name) opt.selected = true;
        select.appendChild(opt);
    }
    select.onchange = () => {
        row.name = select.value;
        ui.commit(true);
    };
    select.addEventListener("keydown", (e) => e.stopPropagation());

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
        renderSegmentLoras(ui, seg);
        ui.commit(true);
    };

    if (row.active === false) el.classList.add("bd-lora-off");
    el.append(on, select, strength, remove);
    return el;
}
