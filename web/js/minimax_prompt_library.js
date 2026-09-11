/** Prompt library for the MiniMax H3 Director.
 *
 * Saved prompts keep their LoRA stack, user categories and a looping thumbnail
 * cut from a real render; every rendered group is also auto-saved to a history
 * (director/prompt_library.py). A bar above the group toolbar opens the library
 * gallery, where a prompt is applied to any set of groups at once; prompts on
 * the bar can also be dragged onto a group card.
 */

import { api } from "../../scripts/api.js";
import { syncFl2vFromShots } from "./minimax_fl2v.js";
import { t } from "./minimax_i18n.js";
import {
    ensureSegmentLoraStyles,
    loraInfo,
    makeThumb as makeLoraThumb,
    normalizeLoraRows,
} from "./minimax_segment_loras.js";

const DRAG_TYPE = "application/x-mmx-prompt";
const QUICK_LIMIT = 8;
const CATEGORY_COLORS = ["#4fff8f", "#5cc8ff", "#ff7ab8", "#ffb35c", "#b18cff", "#ffe066", "#ff6b6b", "#6bffea"];
const ISOLATED_EVENTS = ["pointerdown", "mousedown", "click", "dblclick", "wheel", "contextmenu", "keyup", "keypress"];

let _state = { categories: [], prompts: [], history: [] };
let _loading = null;
let _runHooked = false;
let _dialog = null;
let _popover = null;
const _listeners = new Set();
const _applyPrefs = { prompt: true, loras: true, mode: "replace" };

// ── small helpers ────────────────────────────────────────────────────────

function tf(key, vars = {}) {
    let text = t(key);
    for (const [name, value] of Object.entries(vars)) text = text.split(`{${name}}`).join(String(value));
    return text;
}

function el(tag, cls = "", text = null) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
}

function btn(cls, text, onClick = null, title = "") {
    const node = el("button", cls, text);
    node.type = "button";
    if (title) node.title = title;
    if (onClick) node.onclick = onClick;
    return node;
}

function badge(text, cls = "") {
    return el("span", `mmx-pl-badge ${cls}`.trim(), text);
}

function excerpt(text, max = 140) {
    const flat = String(text || "").replace(/\s+/g, " ").trim();
    return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

function hueOf(text) {
    let h = 0;
    for (const ch of String(text || "")) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return h % 360;
}

function initialsOf(text) {
    const words = String(text || "").split(/[^\p{L}\p{N}]+/u).filter(Boolean);
    return (words.slice(0, 2).map((w) => w[0]).join("") || "✦").toUpperCase();
}

function timeAgo(ts) {
    const s = Math.max(0, Date.now() / 1000 - Number(ts || 0));
    if (s < 60) return t("pl.agoNow");
    if (s < 3600) return tf("pl.agoMin", { n: Math.floor(s / 60) });
    if (s < 86400) return tf("pl.agoHour", { n: Math.floor(s / 3600) });
    return tf("pl.agoDay", { n: Math.floor(s / 86400) });
}

function loraStem(name) {
    return String(name || "").split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
}

function hasPromptDrag(e) {
    return [...(e.dataTransfer?.types || [])].includes(DRAG_TYPE);
}

function flashRect(rect, text) {
    const tip = el("div", "mmx-pl-flash", text);
    tip.style.left = `${rect.left + (rect.width || 0) / 2}px`;
    tip.style.top = `${rect.top - 4}px`;
    document.body.appendChild(tip);
    setTimeout(() => tip.classList.add("out"), 800);
    setTimeout(() => tip.remove(), 1100);
}

function flash(anchor, text) {
    if (anchor?.isConnected) flashRect(anchor.getBoundingClientRect(), text);
}

function reportError(anchor, err) {
    console.warn("[MiniMax Director] prompt library:", err);
    flash(anchor, `${t("pl.error")}: ${err?.message || err}`);
}

function isolate(node) {
    for (const type of ISOLATED_EVENTS) node.addEventListener(type, (e) => e.stopPropagation());
}

// ── state ────────────────────────────────────────────────────────────────

function normalizeState(j) {
    return {
        categories: Array.isArray(j?.categories) ? j.categories : [],
        prompts: Array.isArray(j?.prompts) ? j.prompts : [],
        history: Array.isArray(j?.history) ? j.history : [],
    };
}

function emit() {
    for (const fn of [..._listeners]) {
        try {
            fn(_state);
        } catch (err) {
            console.warn("[MiniMax Director] prompt library listener:", err);
        }
    }
}

export function loadPromptLibrary(force = false) {
    if (!_loading || force) {
        _loading = api
            .fetchApi("/minimax/director/prompt_library")
            .then((r) => (r.ok ? r.json() : null))
            .then((j) => {
                if (j) {
                    _state = normalizeState(j);
                    emit();
                }
                return _state;
            })
            .catch(() => _state);
    }
    return _loading;
}

async function libraryOp(op, payload = {}) {
    const r = await api.fetchApi("/minimax/director/prompt_library", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ op, ...payload }),
    });
    const j = await r.json().catch(() => null);
    if (!r.ok || !j?.ok) throw new Error(j?.error || `HTTP ${r.status}`);
    _state = normalizeState(j);
    _loading = Promise.resolve(_state);
    emit();
    return j;
}

function catById(id) {
    return _state.categories.find((c) => c.id === id) || null;
}

function thumbURL(name) {
    return api.apiURL(`/minimax/director/prompt_thumb?name=${encodeURIComponent(name)}`);
}

/** Looping render thumbnail, or coloured initials when there is none yet. */
function promptThumb(entry, sizeClass) {
    const box = el("span", `mmx-pl-thumb ${sizeClass}`);
    const label = entry?.title || entry?.prompt || "";
    const fallback = () => {
        box.replaceChildren();
        box.classList.add("is-empty");
        box.style.setProperty("--mmx-hue", String(hueOf(label)));
        box.textContent = initialsOf(label);
    };
    if (!entry?.thumb) {
        fallback();
        return box;
    }
    const img = el("img");
    img.alt = "";
    img.loading = "lazy";
    img.decoding = "async";
    img.src = thumbURL(entry.thumb);
    img.onerror = fallback;
    box.appendChild(img);
    return box;
}

/** Newest history render of this prompt (same Director + group first, then any). */
function latestRenderFor(nodeId, group, prompt) {
    const text = String(prompt || "").trim();
    if (!text) return null;
    const same = (h) => h.thumb && String(h.prompt || "").trim() === text;
    return (
        (nodeId != null && _state.history.find((h) => same(h) && String(h.node) === String(nodeId) && Number(h.group) === group))
        || _state.history.find(same)
        || null
    );
}

// ── applying to groups ───────────────────────────────────────────────────

function editorGroups(editor) {
    if (editor?.isFl2vMode?.()) return { kind: "shots", items: editor.timeline?.shots || [] };
    return { kind: "segments", items: editor?.timeline?.segments || [] };
}

function joinPrompt(current, addition) {
    const trimmed = current.replace(/\s+$/, "");
    return trimmed + (/[,;.!?]$/.test(trimmed) ? " " : ", ") + addition;
}

function mergeLoras(current, incoming) {
    const rows = normalizeLoraRows(current);
    for (const row of incoming) {
        const hit = rows.find((r) => r.name === row.name);
        if (hit) Object.assign(hit, row);
        else rows.push({ ...row });
    }
    return normalizeLoraRows(rows);
}

/** Write a saved or history entry into groups `indices`; returns how many changed. */
export function applyPromptEntry(editor, entry, indices, opts = _applyPrefs) {
    const { kind, items } = editorGroups(editor);
    let applied = 0;
    for (const i of indices) {
        const group = items[i];
        if (!group) continue;
        if (opts.prompt && String(entry.prompt || "").trim()) {
            const current = String(group.prompt || "").trim();
            group.prompt = opts.mode === "append" && current ? joinPrompt(current, entry.prompt) : entry.prompt;
            if (kind === "segments") editor.writeExternalGroupPrompt?.(i, group.prompt);
        }
        if (opts.loras) {
            const incoming = normalizeLoraRows(entry.loras);
            group.loras = opts.mode === "append" ? mergeLoras(group.loras, incoming) : incoming;
        }
        applied += 1;
    }
    if (!applied) return 0;
    if (kind === "shots") syncFl2vFromShots(editor);
    editor.commit?.(false);
    if (String(entry.id || "").startsWith("p_")) libraryOp("use_prompt", { id: entry.id }).catch(() => {});
    return applied;
}

/** Group chips + what to apply + the Apply button. `source` is an entry or a getter. */
function buildApplySection(editor, source, { onDone } = {}) {
    const wrap = el("div", "mmx-pl-apply");
    const { items } = editorGroups(editor);
    const selected = new Set(items.length === 1 ? [0] : []);
    const chips = el("div", "mmx-pl-group-chips");
    const chipEls = items.map((g, i) => {
        const chip = btn(
            "mmx-pl-group-chip",
            String(i + 1),
            () => {
                if (selected.has(i)) selected.delete(i);
                else selected.add(i);
                sync();
            },
            `${tf("pl.groupN", { n: i + 1 })}\n${excerpt(g.prompt, 90) || "—"}`,
        );
        chips.appendChild(chip);
        return chip;
    });
    let allChip = null;
    if (items.length > 1) {
        allChip = btn("mmx-pl-group-chip is-all", t("pl.all"), () => {
            const every = selected.size === items.length;
            selected.clear();
            if (!every) items.forEach((_, i) => selected.add(i));
            sync();
        });
        chips.appendChild(allChip);
    }
    if (!items.length) chips.appendChild(el("span", "mmx-pl-muted", t("pl.noGroups")));

    const toggle = (key, label) => {
        const node = btn("mmx-pl-toggle", label, () => {
            _applyPrefs[key] = !_applyPrefs[key];
            node.classList.toggle("is-on", _applyPrefs[key]);
            sync();
        });
        node.classList.toggle("is-on", _applyPrefs[key]);
        return node;
    };
    const mode = btn("mmx-pl-toggle is-mode", "", () => {
        _applyPrefs.mode = _applyPrefs.mode === "append" ? "replace" : "append";
        paintMode();
    }, t("pl.modeHint"));
    const paintMode = () => {
        mode.textContent = _applyPrefs.mode === "append" ? `＋ ${t("pl.append")}` : `⇄ ${t("pl.replace")}`;
    };
    paintMode();
    const opts = el("div", "mmx-pl-apply-opts");
    opts.append(toggle("prompt", t("pl.optPrompt")), toggle("loras", t("pl.optLoras")), mode);

    const go = btn("mmx-pl-primary", "");
    function sync() {
        chipEls.forEach((chip, i) => chip.classList.toggle("is-on", selected.has(i)));
        allChip?.classList.toggle("is-on", items.length > 0 && selected.size === items.length);
        go.disabled = !selected.size || (!_applyPrefs.prompt && !_applyPrefs.loras);
        go.textContent = selected.size ? tf("pl.applyN", { n: selected.size }) : t("pl.selectGroups");
    }
    go.onclick = () => {
        const entry = typeof source === "function" ? source() : source;
        const n = applyPromptEntry(editor, entry, [...selected].sort((a, b) => a - b), { ..._applyPrefs });
        flash(go, tf("pl.applied", { n }));
        onDone?.(n);
    };
    sync();
    wrap.append(el("div", "mmx-pl-label", t("pl.applyTo")), chips, opts, go);
    return wrap;
}

function openApplyPopover(editor, anchor, entry) {
    _popover?.();
    ensurePromptLibraryStyles();
    const pop = el("div", "mmx-pl-popover");
    const head = el("div", "mmx-pl-pop-head");
    const titles = el("div", "mmx-pl-pop-titles");
    titles.append(
        el("span", "mmx-pl-pop-title", entry.title || excerpt(entry.prompt, 40)),
        el("span", "mmx-pl-pop-sub", excerpt(entry.prompt, 110) || t("pl.noPromptText")),
    );
    head.append(promptThumb(entry, "mmx-pl-thumb-sm"), titles);
    const onDoc = (e) => {
        if (!pop.contains(e.target) && !anchor.contains(e.target)) close();
    };
    const close = () => {
        if (_popover !== close) return;
        _popover = null;
        document.removeEventListener("pointerdown", onDoc, true);
        pop.remove();
    };
    pop.append(head, buildApplySection(editor, entry, { onDone: () => setTimeout(close, 650) }));
    isolate(pop);
    pop.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Escape") close();
    });
    document.body.appendChild(pop);
    const r = anchor.getBoundingClientRect();
    let top = r.bottom + 6;
    if (top + pop.offsetHeight > window.innerHeight - 8) top = Math.max(8, r.top - pop.offsetHeight - 6);
    pop.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - pop.offsetWidth - 8))}px`;
    pop.style.top = `${top}px`;
    _popover = close;
    setTimeout(() => document.addEventListener("pointerdown", onDoc, true), 0);
}

// ── the bar above the group toolbar ──────────────────────────────────────

function placeBar(editor, bar) {
    const batchOn = !!editor.batchPanel && !editor.batchPanel.classList.contains("hidden");
    const target = batchOn
        ? editor.batchPanel.querySelector(".bd-batch-toolbar")
        : editor.root?.querySelector(".bd-toolbar-wrap > .bd-toolbar");
    if (target?.parentNode && target.previousElementSibling !== bar) target.parentNode.insertBefore(bar, target);
}

/** Bar prompts dragged onto a batch group card apply to that group. */
function wireDropTargets(editor) {
    const list = editor.batchList;
    if (!list || list.__mmxPromptDrop) return;
    list.__mmxPromptDrop = true;
    let lit = null;
    const light = (card) => {
        if (lit === card) return;
        lit?.classList.remove("mmx-pl-drop");
        lit = card || null;
        lit?.classList.add("mmx-pl-drop");
    };
    const cardOf = (e) => e.target?.closest?.(".bd-batch-card") || null;
    list.addEventListener("dragover", (e) => {
        if (!hasPromptDrag(e)) return;
        const card = cardOf(e);
        light(card);
        if (!card) return;
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = "copy";
    }, true);
    list.addEventListener("dragleave", (e) => {
        if (hasPromptDrag(e) && !list.contains(e.relatedTarget)) light(null);
    }, true);
    list.addEventListener("drop", (e) => {
        if (!hasPromptDrag(e)) return;
        const card = cardOf(e);
        light(null);
        if (!card) return;
        e.preventDefault();
        e.stopPropagation();
        const entry = _state.prompts.find((p) => p.id === e.dataTransfer.getData(DRAG_TYPE));
        const index = Number.parseInt(card.dataset.batchIndex, 10);
        if (!entry || !Number.isFinite(index)) return;
        const n = applyPromptEntry(editor, entry, [index], { ..._applyPrefs });
        flashRect({ left: e.clientX, top: e.clientY }, tf("pl.applied", { n }));
    }, true);
    document.addEventListener("dragend", () => light(null));
}

export function mountPromptLibraryBar(editor) {
    if (!editor?.root) return null;
    if (editor.promptLibraryBar) return editor.promptLibraryBar;
    ensurePromptLibraryStyles();
    const bar = el("div", "mmx-pl-bar");
    const count = el("span", "mmx-pl-brand-count");
    const brand = btn("mmx-pl-brand", "", () => openPromptLibrary(editor), t("pl.openLibrary"));
    brand.append(el("span", "mmx-pl-brand-icon", "✦"), el("span", "mmx-pl-brand-text", t("pl.title")), count);
    const strip = el("div", "mmx-pl-quick");
    const histBtn = btn("mmx-pl-bar-btn", "", () => openPromptLibrary(editor, { view: "history" }), t("pl.historyHint"));
    const saveBtn = btn("mmx-pl-bar-btn", `💾 ${t("pl.saveGroups")}`, () => openPromptLibrary(editor, { view: "save" }));
    const libBtn = btn("mmx-pl-bar-btn is-accent", `${t("pl.library")} ↗`, () => openPromptLibrary(editor));
    const actions = el("div", "mmx-pl-bar-actions");
    actions.append(histBtn, saveBtn, libBtn);
    bar.append(brand, strip, actions);
    bar.addEventListener("pointerdown", (e) => e.stopPropagation());
    bar.addEventListener("mousedown", (e) => e.stopPropagation());

    const paint = (state) => {
        count.textContent = state.prompts.length ? String(state.prompts.length) : "";
        histBtn.textContent = `🕘 ${t("pl.history")}${state.history.length ? ` · ${state.history.length}` : ""}`;
        const quick = [...state.prompts]
            .sort(
                (a, b) => Number(!!b.favorite) - Number(!!a.favorite)
                    || (b.last_used || b.updated || 0) - (a.last_used || a.updated || 0),
            )
            .slice(0, QUICK_LIMIT);
        strip.replaceChildren();
        if (!quick.length) {
            strip.appendChild(el("span", "mmx-pl-quick-empty", t("pl.quickEmpty")));
            return;
        }
        for (const entry of quick) {
            const chip = btn(
                `mmx-pl-quick-chip${entry.favorite ? " is-fav" : ""}`,
                "",
                null,
                `${entry.title || ""}\n${t("pl.dragHint")}`,
            );
            chip.onclick = (e) => {
                e.stopPropagation();
                openApplyPopover(editor, chip, entry);
            };
            chip.draggable = true;
            chip.addEventListener("dragstart", (e) => {
                e.dataTransfer.setData(DRAG_TYPE, entry.id);
                e.dataTransfer.setData("text/plain", entry.prompt || "");
                e.dataTransfer.effectAllowed = "copy";
            });
            chip.append(promptThumb(entry, "mmx-pl-thumb-xs"), el("span", "mmx-pl-quick-title", entry.title || excerpt(entry.prompt, 28)));
            strip.appendChild(chip);
        }
    };
    _listeners.add(paint);
    const place = () => placeBar(editor, bar);
    place();
    if (editor.batchPanel) {
        new MutationObserver(place).observe(editor.batchPanel, { attributes: true, attributeFilter: ["class"] });
    }
    wireDropTargets(editor);
    editor.promptLibraryBar = bar;
    if (!_runHooked) {
        _runHooked = true;
        const refresh = () => setTimeout(() => loadPromptLibrary(true), 400);
        for (const type of ["execution_success", "execution_error", "execution_interrupted"]) {
            api.addEventListener(type, refresh);
        }
    }
    paint(_state);
    loadPromptLibrary();
    return bar;
}

// ── the library gallery ──────────────────────────────────────────────────

export function closePromptLibrary() {
    _dialog?.close();
}

export async function openPromptLibrary(editor, { view = "prompts", category = "all", selected = null } = {}) {
    closePromptLibrary();
    _popover?.();
    ensurePromptLibraryStyles();
    ensureSegmentLoraStyles();
    await loadPromptLibrary(true);

    const ui = { view, category, selected, query: "", editingCategory: null, newCategory: false };
    const nodeId = editor?.node?.id ?? null;
    const overlay = el("div", "mmx-pl-overlay");
    const dialog = el("div", "mmx-pl-dialog");
    const side = el("div", "mmx-pl-side");
    const main = el("div", "mmx-pl-main");
    const detail = el("div", "mmx-pl-detail");
    dialog.append(side, main, detail);
    overlay.appendChild(dialog);

    const head = el("div", "mmx-pl-head");
    const heading = el("span", "mmx-pl-heading");
    const search = el("input", "mmx-pl-search");
    search.type = "search";
    search.placeholder = t("pl.search");
    search.addEventListener("input", () => {
        ui.query = search.value;
        paintMain();
    });
    const closeBtn = btn("mmx-pl-close", "×", () => close());
    head.append(heading, search, closeBtn);
    const body = el("div", "mmx-pl-body");
    main.append(head, body);

    const handle = { close: null };
    const close = () => {
        if (_dialog !== handle) return;
        _dialog = null;
        _listeners.delete(repaint);
        overlay.remove();
    };
    handle.close = close;
    const fail = (err) => reportError(heading, err);
    const repaint = () => {
        paintSide();
        paintMain();
        paintDetail();
    };
    const go = (nextView, nextCategory = ui.category) => {
        ui.view = nextView;
        ui.category = nextCategory;
        ui.selected = null;
        repaint();
    };

    // sidebar
    const navItem = (active, icon, label, count, onClick) => {
        const item = btn(`mmx-pl-nav${active ? " is-on" : ""}`, "", onClick);
        item.append(
            el("span", "mmx-pl-nav-icon", icon),
            el("span", "mmx-pl-nav-label", label),
            el("span", "mmx-pl-nav-count", count ? String(count) : ""),
        );
        return item;
    };
    const categoryInput = (value, onCommit) => {
        const input = el("input", "mmx-pl-side-input");
        input.value = value;
        input.placeholder = t("pl.categoryName");
        let done = false;
        const finish = (name) => {
            if (done) return;
            done = true;
            onCommit(name);
        };
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                finish(input.value.trim());
            } else if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                finish(null);
            }
        });
        input.addEventListener("blur", () => finish(input.value.trim()));
        setTimeout(() => input.focus(), 0);
        return input;
    };

    function paintSide() {
        const s = _state;
        const onPrompts = ui.view === "prompts";
        side.replaceChildren(el("div", "mmx-pl-side-brand", `✦ ${t("pl.title")}`));
        side.append(
            navItem(onPrompts && ui.category === "all", "▦", t("pl.allPrompts"), s.prompts.length, () => go("prompts", "all")),
            navItem(onPrompts && ui.category === "fav", "★", t("pl.favorites"), s.prompts.filter((p) => p.favorite).length, () => go("prompts", "fav")),
            navItem(
                onPrompts && ui.category === "none",
                "◌",
                t("pl.uncategorized"),
                s.prompts.filter((p) => !(p.categories || []).length).length,
                () => go("prompts", "none"),
            ),
        );
        const catHead = el("div", "mmx-pl-side-head");
        catHead.append(
            el("span", "", t("pl.categories")),
            btn("mmx-pl-icon-btn", "+", () => {
                ui.newCategory = true;
                paintSide();
            }, t("pl.newCategory")),
        );
        side.appendChild(catHead);
        for (const c of s.categories) {
            if (ui.editingCategory === c.id) {
                side.appendChild(categoryInput(c.name, (name) => {
                    ui.editingCategory = null;
                    if (name && name !== c.name) libraryOp("update_category", { id: c.id, name }).catch(fail);
                    else paintSide();
                }));
                continue;
            }
            const count = s.prompts.filter((p) => (p.categories || []).includes(c.id)).length;
            const item = navItem(onPrompts && ui.category === c.id, "●", c.name, count, () => go("prompts", c.id));
            item.querySelector(".mmx-pl-nav-icon").style.color = c.color || CATEGORY_COLORS[0];
            const tools = el("span", "mmx-pl-nav-tools");
            tools.append(
                btn("mmx-pl-icon-btn", "✎", (e) => {
                    e.stopPropagation();
                    ui.editingCategory = c.id;
                    paintSide();
                }, t("pl.renameCategory")),
                btn("mmx-pl-icon-btn is-danger", "✕", (e) => {
                    e.stopPropagation();
                    if (!confirm(t("pl.confirmDeleteCategory"))) return;
                    if (ui.category === c.id) ui.category = "all";
                    libraryOp("delete_category", { id: c.id }).catch(fail);
                }, t("pl.deleteCategory")),
            );
            item.appendChild(tools);
            item.addEventListener("dragover", (e) => {
                if (!hasPromptDrag(e)) return;
                e.preventDefault();
                item.classList.add("is-drop");
            });
            item.addEventListener("dragleave", () => item.classList.remove("is-drop"));
            item.addEventListener("drop", (e) => {
                if (!hasPromptDrag(e)) return;
                e.preventDefault();
                item.classList.remove("is-drop");
                const entry = s.prompts.find((p) => p.id === e.dataTransfer.getData(DRAG_TYPE));
                if (!entry || (entry.categories || []).includes(c.id)) return;
                libraryOp("save_prompt", { id: entry.id, categories: [...(entry.categories || []), c.id] })
                    .then(() => flash(side.querySelector(".mmx-pl-side-brand"), t("pl.saved")))
                    .catch(fail);
            });
            side.appendChild(item);
        }
        if (ui.newCategory) {
            side.appendChild(categoryInput("", (name) => {
                ui.newCategory = false;
                if (name) {
                    libraryOp("add_category", {
                        name,
                        color: CATEGORY_COLORS[s.categories.length % CATEGORY_COLORS.length],
                    }).catch(fail);
                } else {
                    paintSide();
                }
            }));
        } else if (!s.categories.length) {
            side.appendChild(el("div", "mmx-pl-side-hint", t("pl.noCategoriesHint")));
        }
        side.append(
            el("div", "mmx-pl-side-sep"),
            navItem(ui.view === "history", "🕘", t("pl.history"), s.history.length, () => go("history")),
            navItem(ui.view === "save", "💾", t("pl.saveGroups"), editorGroups(editor).items.length, () => go("save")),
        );
    }

    // main grid
    function visibleEntries() {
        const words = ui.query.toLowerCase().split(/\s+/).filter(Boolean);
        const match = (e) => {
            if (!words.length) return true;
            const hay = [e.title, e.prompt, ...(e.loras || []).map((l) => l.name)].join(" ").toLowerCase();
            return words.every((w) => hay.includes(w));
        };
        if (ui.view === "history") return _state.history.filter(match);
        let list = _state.prompts;
        if (ui.category === "fav") list = list.filter((p) => p.favorite);
        else if (ui.category === "none") list = list.filter((p) => !(p.categories || []).length);
        else if (ui.category !== "all") list = list.filter((p) => (p.categories || []).includes(ui.category));
        return list.filter(match);
    }

    function headingText() {
        if (ui.view === "history") return t("pl.history");
        if (ui.view === "save") return t("pl.saveGroups");
        if (ui.category === "fav") return t("pl.favorites");
        if (ui.category === "none") return t("pl.uncategorized");
        if (ui.category === "all") return t("pl.allPrompts");
        return catById(ui.category)?.name || t("pl.allPrompts");
    }

    function card(entry, i) {
        const isHistory = ui.view === "history";
        const node = btn(`mmx-pl-card${ui.selected === entry.id ? " is-on" : ""}`, "");
        node.style.setProperty("--mmx-i", String(Math.min(i, 24)));
        const media = el("span", "mmx-pl-card-media");
        media.appendChild(promptThumb(entry, "mmx-pl-thumb-card"));
        const top = el("span", "mmx-pl-card-top");
        if (isHistory) {
            if (entry.group) top.appendChild(badge(`#${entry.group}`));
            if ((entry.runs || 1) > 1) top.appendChild(badge(`×${entry.runs}`));
        } else if (entry.favorite) {
            top.appendChild(badge("★", "is-fav"));
        }
        const loraCount = (entry.loras || []).length;
        if (loraCount) top.appendChild(badge(`🧩 ${loraCount}`, "is-lora"));
        media.appendChild(top);
        const text = el("span", "mmx-pl-card-text");
        text.append(
            el("span", "mmx-pl-card-title", isHistory ? timeAgo(entry.created) : entry.title || excerpt(entry.prompt, 60) || "—"),
            el("span", "mmx-pl-card-excerpt", excerpt(entry.prompt, 120) || t("pl.noPromptText")),
        );
        if (!isHistory) {
            const dots = el("span", "mmx-pl-card-dots");
            for (const id of entry.categories || []) {
                const c = catById(id);
                if (!c) continue;
                const dot = el("span", "mmx-pl-dot");
                dot.style.background = c.color;
                dot.style.color = c.color;
                dot.title = c.name;
                dots.appendChild(dot);
            }
            text.appendChild(dots);
            node.draggable = true;
            node.addEventListener("dragstart", (e) => {
                e.dataTransfer.setData(DRAG_TYPE, entry.id);
                e.dataTransfer.effectAllowed = "copy";
            });
        }
        node.append(media, text);
        node.onclick = () => {
            ui.selected = entry.id;
            for (const other of body.querySelectorAll(".mmx-pl-card")) other.classList.toggle("is-on", other === node);
            paintDetail();
        };
        return node;
    }

    function paintMain() {
        heading.textContent = headingText();
        search.hidden = ui.view === "save";
        body.replaceChildren();
        if (ui.view === "save") {
            body.appendChild(buildSaveView());
            return;
        }
        if (ui.view === "history" && _state.history.length) {
            const bar = el("div", "mmx-pl-histbar");
            bar.append(
                el("span", "mmx-pl-muted", t("pl.historyHint")),
                btn("mmx-pl-icon-text is-danger", t("pl.clearHistory"), () => {
                    if (confirm(t("pl.confirmClearHistory"))) libraryOp("history_clear").catch(fail);
                }),
            );
            body.appendChild(bar);
        }
        const list = visibleEntries();
        if (!list.length) {
            const msg = ui.view === "history" ? t("pl.emptyHistory") : ui.query ? t("pl.noMatch") : t("pl.empty");
            body.appendChild(el("div", "mmx-pl-empty", msg));
            return;
        }
        const grid = el("div", "mmx-pl-grid");
        list.forEach((entry, i) => grid.appendChild(card(entry, i)));
        body.appendChild(grid);
    }

    // detail pane
    function lorasBlock(loras) {
        const box = el("div", "mmx-pl-loras");
        const rows = normalizeLoraRows(loras);
        if (!rows.length) {
            box.appendChild(el("span", "mmx-pl-muted", t("pl.noLoras")));
            return box;
        }
        loraInfo().then((info) => {
            for (const row of rows) {
                const chip = el("span", `mmx-pl-lora${row.active === false ? " is-off" : ""}`);
                chip.title = row.name;
                chip.append(
                    makeLoraThumb(row.name, info[row.name], "mmx-lora-thumb-sm"),
                    el("span", "mmx-pl-lora-name", loraStem(row.name)),
                    el("span", "mmx-pl-lora-str", String(row.strength)),
                );
                box.appendChild(chip);
            }
        });
        return box;
    }

    function categorySelect(preferred) {
        const sel = el("select", "mmx-pl-select");
        sel.append(new Option(t("pl.noCategory"), ""));
        for (const c of _state.categories) sel.append(new Option(c.name, c.id));
        if (catById(preferred)) sel.value = preferred;
        return sel;
    }

    function promptDetail(entry) {
        const title = el("input", "mmx-pl-input");
        title.value = entry.title || "";
        title.placeholder = t("pl.titlePlaceholder");
        const text = el("textarea", "mmx-pl-textarea");
        text.value = entry.prompt || "";
        const chosen = new Set(entry.categories || []);
        const save = btn("mmx-pl-secondary", t("pl.save"));
        const dirty = () => save.classList.add("is-dirty");
        title.addEventListener("input", dirty);
        text.addEventListener("input", dirty);
        save.onclick = () => libraryOp("save_prompt", {
            id: entry.id,
            title: title.value,
            prompt: text.value,
            categories: [...chosen],
        }).then(() => flash(heading, t("pl.saved"))).catch(fail);
        const cats = el("div", "mmx-pl-cat-chips");
        for (const c of _state.categories) {
            const chip = btn(`mmx-pl-cat-chip${chosen.has(c.id) ? " is-on" : ""}`, c.name);
            chip.style.setProperty("--cat", c.color || CATEGORY_COLORS[0]);
            chip.onclick = () => {
                if (chosen.has(c.id)) chosen.delete(c.id);
                else chosen.add(c.id);
                chip.classList.toggle("is-on", chosen.has(c.id));
                dirty();
            };
            cats.appendChild(chip);
        }
        if (!_state.categories.length) cats.appendChild(el("span", "mmx-pl-muted", t("pl.noCategoriesHint")));
        const saveRow = el("div", "mmx-pl-row");
        saveRow.appendChild(save);

        const footer = el("div", "mmx-pl-footer");
        footer.append(
            btn("mmx-pl-icon-text", entry.favorite ? `★ ${t("pl.unfavorite")}` : `☆ ${t("pl.favorite")}`, () =>
                libraryOp("save_prompt", { id: entry.id, favorite: !entry.favorite }).catch(fail)),
            btn("mmx-pl-icon-text", `⧉ ${t("pl.duplicate")}`, () => libraryOp("save_prompt", {
                title: `${entry.title || t("pl.untitled")} 2`,
                prompt: entry.prompt,
                negative: entry.negative,
                loras: entry.loras,
                categories: entry.categories,
                thumb_from: entry.thumb || "",
            }).then((j) => {
                ui.selected = j.prompt?.id || null;
                repaint();
            }).catch(fail)),
        );
        const latest = latestRenderFor(null, null, entry.prompt);
        if (latest && latest.thumb !== entry.thumb_origin) {
            footer.appendChild(btn("mmx-pl-icon-text", `↻ ${t("pl.useLatestThumb")}`, () =>
                libraryOp("save_prompt", { id: entry.id, thumb_from: latest.thumb }).catch(fail)));
        }
        footer.appendChild(btn("mmx-pl-icon-text is-danger", `🗑 ${t("pl.delete")}`, () => {
            if (!confirm(t("pl.confirmDelete"))) return;
            libraryOp("delete_prompt", { id: entry.id }).then(() => {
                ui.selected = null;
                repaint();
            }).catch(fail);
        }));
        return [
            promptThumb(entry, "mmx-pl-thumb-xl"),
            title,
            el("div", "mmx-pl-label", t("pl.prompt")),
            text,
            el("div", "mmx-pl-label", t("pl.loras")),
            lorasBlock(entry.loras),
            el("div", "mmx-pl-label", t("pl.categories")),
            cats,
            saveRow,
            buildApplySection(editor, () => ({ ...entry, prompt: text.value })),
            footer,
        ];
    }

    function historyDetail(entry) {
        const meta = [
            entry.group ? tf("pl.groupOf", { n: entry.group, m: entry.groups || "?" }) : "",
            entry.task ? String(entry.task).toUpperCase() : "",
            timeAgo(entry.created),
            entry.seed != null ? `seed ${entry.seed}` : "",
            (entry.runs || 1) > 1 ? tf("pl.runs", { n: entry.runs }) : "",
        ].filter(Boolean).join(" · ");
        const text = el("textarea", "mmx-pl-textarea");
        text.value = entry.prompt || "";
        text.readOnly = true;
        const titleIn = el("input", "mmx-pl-input is-small");
        titleIn.placeholder = t("pl.titlePlaceholder");
        const catSel = categorySelect(ui.category);
        const keep = btn("mmx-pl-primary is-soft", `💾 ${t("pl.saveToLibrary")}`, () => libraryOp("save_prompt", {
            title: titleIn.value,
            prompt: entry.prompt,
            negative: entry.negative,
            loras: entry.loras,
            categories: catSel.value ? [catSel.value] : [],
            thumb_from: entry.thumb || "",
        }).then(() => flash(keep, t("pl.saved"))).catch(fail));
        const saveBox = el("div", "mmx-pl-savebox");
        const row = el("div", "mmx-pl-row");
        row.append(catSel, keep);
        saveBox.append(titleIn, row);
        const footer = el("div", "mmx-pl-footer");
        footer.appendChild(btn("mmx-pl-icon-text is-danger", `🗑 ${t("pl.delete")}`, () =>
            libraryOp("history_delete", { id: entry.id }).then(() => {
                ui.selected = null;
                repaint();
            }).catch(fail)));
        return [
            promptThumb(entry, "mmx-pl-thumb-xl"),
            el("div", "mmx-pl-meta", meta),
            el("div", "mmx-pl-label", t("pl.prompt")),
            text,
            el("div", "mmx-pl-label", t("pl.loras")),
            lorasBlock(entry.loras),
            el("div", "mmx-pl-label", t("pl.saveToLibrary")),
            saveBox,
            buildApplySection(editor, entry),
            footer,
        ];
    }

    function paintDetail() {
        detail.replaceChildren();
        if (ui.view === "save") {
            detail.appendChild(el("div", "mmx-pl-detail-empty", t("pl.saveHint")));
            return;
        }
        const pool = ui.view === "history" ? _state.history : _state.prompts;
        const entry = pool.find((e) => e.id === ui.selected);
        if (!entry) {
            detail.appendChild(el("div", "mmx-pl-detail-empty", t("pl.pickHint")));
            return;
        }
        detail.append(...(ui.view === "history" ? historyDetail(entry) : promptDetail(entry)));
    }

    // "save groups" view
    function buildSaveView() {
        const wrap = el("div", "mmx-pl-saveview");
        const { items } = editorGroups(editor);
        if (!items.length) {
            wrap.appendChild(el("div", "mmx-pl-empty", t("pl.noGroups")));
            return wrap;
        }
        const chosen = new Set();
        items.forEach((g, i) => {
            if (String(g.prompt || "").trim() || normalizeLoraRows(g.loras).length) chosen.add(i);
        });
        const saveGo = btn("mmx-pl-primary", "");
        const paintGo = () => {
            saveGo.textContent = tf("pl.saveSelected", { n: chosen.size });
            saveGo.disabled = !chosen.size;
        };
        const rows = el("div", "mmx-pl-saverows");
        items.forEach((g, i) => {
            const render = latestRenderFor(nodeId, i + 1, g.prompt);
            const row = el("label", "mmx-pl-saverow");
            const cb = el("input");
            cb.type = "checkbox";
            cb.checked = chosen.has(i);
            cb.onchange = () => {
                if (cb.checked) chosen.add(i);
                else chosen.delete(i);
                paintGo();
            };
            const textBox = el("span", "mmx-pl-saverow-text");
            textBox.append(
                el("span", "mmx-pl-saverow-title", tf("pl.groupN", { n: i + 1 })),
                el("span", "mmx-pl-saverow-excerpt", excerpt(g.prompt, 180) || t("pl.noPromptText")),
            );
            row.append(cb, promptThumb(render || { title: g.prompt || String(i + 1) }, "mmx-pl-thumb-sm"), textBox);
            const loraCount = normalizeLoraRows(g.loras).length;
            if (loraCount) row.appendChild(badge(`🧩 ${loraCount}`, "is-lora"));
            rows.appendChild(row);
        });
        const catSel = categorySelect(ui.category);
        saveGo.onclick = () => {
            const payload = [...chosen].sort((a, b) => a - b).map((i) => {
                const g = items[i];
                const render = latestRenderFor(nodeId, i + 1, g.prompt);
                return {
                    prompt: g.prompt || "",
                    negative: g.negativePrompt || "",
                    loras: normalizeLoraRows(g.loras),
                    categories: catSel.value ? [catSel.value] : [],
                    thumb_from: render?.thumb || "",
                };
            });
            libraryOp("save_prompts", { items: payload })
                .then(() => {
                    flash(heading, t("pl.saved"));
                    go("prompts", catSel.value || "all");
                })
                .catch(fail);
        };
        paintGo();
        const foot = el("div", "mmx-pl-savefoot");
        foot.append(el("span", "mmx-pl-label is-inline", t("pl.saveInto")), catSel, saveGo);
        wrap.append(rows, foot);
        return wrap;
    }

    isolate(overlay);
    overlay.addEventListener("pointerdown", (e) => {
        if (e.target === overlay) close();
    });
    overlay.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Escape" && !e.target.closest?.(".mmx-pl-side-input")) {
            e.preventDefault();
            close();
        }
    });
    _listeners.add(repaint);
    document.body.appendChild(overlay);
    _dialog = handle;
    repaint();
    if (ui.view !== "save") setTimeout(() => search.focus(), 0);
}

// ── styles ───────────────────────────────────────────────────────────────

export function ensurePromptLibraryStyles() {
    if (document.getElementById("mmx-prompt-library-css")) return;
    const style = document.createElement("style");
    style.id = "mmx-prompt-library-css";
    style.textContent = `
.mmx-pl-bar { display:flex; align-items:center; gap:8px; width:100%; box-sizing:border-box; flex-shrink:0;
    padding:6px 8px; margin:0 0 6px; min-width:0; border:1px solid #243428; border-radius:10px;
    background:linear-gradient(90deg, rgba(79,255,143,.11), rgba(79,255,143,.025) 45%, rgba(255,255,255,.015)); }
.mmx-pl-brand { display:flex; align-items:center; gap:6px; flex:0 0 auto; background:transparent; border:none;
    color:#eafff1; cursor:pointer; padding:2px 4px; font-family:inherit; }
.mmx-pl-brand-icon { color:#4fff8f; font-size:14px; text-shadow:0 0 8px rgba(79,255,143,.6); }
.mmx-pl-brand-text { font-size:11px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.mmx-pl-brand-count { font-size:10px; color:#7fbf98; font-variant-numeric:tabular-nums; }
.mmx-pl-brand:hover .mmx-pl-brand-text { color:#fff; }
.mmx-pl-quick { flex:1 1 auto; min-width:0; display:flex; gap:6px; overflow-x:auto; scrollbar-width:none; padding:1px; }
.mmx-pl-quick::-webkit-scrollbar { display:none; }
.mmx-pl-quick-chip { flex:0 0 auto; display:flex; align-items:center; gap:6px; max-width:190px; padding:2px 9px 2px 2px;
    background:#151515; border:1px solid #2c2c2c; border-radius:999px; color:#ddd; font-size:10.5px; cursor:grab;
    font-family:inherit; transition:border-color .15s, transform .12s, box-shadow .15s; }
.mmx-pl-quick-chip:hover { border-color:#4fff8f; transform:translateY(-1px); box-shadow:0 4px 12px rgba(0,0,0,.35); }
.mmx-pl-quick-chip.is-fav { border-color:rgba(255,211,92,.55); }
.mmx-pl-quick-title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mmx-pl-quick-empty { color:#6d7d74; font-size:10.5px; font-style:italic; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mmx-pl-bar-actions { display:flex; gap:6px; flex:0 0 auto; }
.mmx-pl-bar-btn { background:#161616; border:1px solid #333; color:#ccc; border-radius:7px; padding:4px 9px;
    font-size:10.5px; cursor:pointer; white-space:nowrap; font-family:inherit; }
.mmx-pl-bar-btn:hover { border-color:#4fff8f; color:#fff; }
.mmx-pl-bar-btn.is-accent { background:#4fff8f; border-color:#4fff8f; color:#062b14; font-weight:800; }
.mmx-pl-bar-btn.is-accent:hover { filter:brightness(1.08); color:#062b14; }
.bd-batch-card.mmx-pl-drop { outline:2px dashed #4fff8f; outline-offset:2px; box-shadow:0 0 0 5px rgba(79,255,143,.12); }

.mmx-pl-thumb { flex:0 0 auto; display:flex; align-items:center; justify-content:center; overflow:hidden;
    background:#0e0e0e; border-radius:6px; color:#555; font-weight:800; }
.mmx-pl-thumb img { width:100%; height:100%; object-fit:cover; display:block; }
.mmx-pl-thumb.is-empty { letter-spacing:.04em;
    background:linear-gradient(135deg, hsl(var(--mmx-hue, 150) 45% 24%), hsl(calc(var(--mmx-hue, 150) + 50) 55% 9%));
    color:hsl(var(--mmx-hue, 150) 80% 82%); }
.mmx-pl-thumb-xs { width:24px; height:24px; border-radius:999px; font-size:9px; }
.mmx-pl-thumb-sm { width:46px; height:46px; font-size:14px; border-radius:8px; }
.mmx-pl-thumb-card { position:absolute; inset:0; width:100%; height:100%; border-radius:0; font-size:34px; }
.mmx-pl-thumb-xl { width:100%; aspect-ratio:16 / 10; border-radius:10px; font-size:48px; box-shadow:0 10px 26px rgba(0,0,0,.55); }

.mmx-pl-overlay { position:fixed; inset:0; z-index:10040; display:flex; align-items:center; justify-content:center;
    background:rgba(4,6,5,.62); backdrop-filter:blur(4px); animation:mmx-pl-fade .16s ease-out; }
.mmx-pl-dialog { width:min(1200px, 97vw); height:min(740px, 92vh); display:grid;
    grid-template-columns:210px minmax(0, 1fr) 350px; overflow:hidden; color:#eee; font-size:11px;
    background:linear-gradient(180deg, #151515, #0c0c0c); border:1px solid #2c2c2c; border-radius:14px;
    box-shadow:0 24px 60px rgba(0,0,0,.65), 0 0 0 1px rgba(79,255,143,.07);
    animation:mmx-pl-pop .2s cubic-bezier(.2,.9,.3,1.15); }
@keyframes mmx-pl-fade { from { opacity:0; } }
@keyframes mmx-pl-pop { from { opacity:0; transform:scale(.97) translateY(8px); } }
@keyframes mmx-pl-rise { from { opacity:0; transform:translateY(8px); } }
.mmx-pl-side { display:flex; flex-direction:column; gap:3px; padding:14px 10px; overflow-y:auto; background:#0b0b0b;
    border-right:1px solid #222; scrollbar-width:thin; scrollbar-color:#333 transparent; }
.mmx-pl-side-brand { font-size:12px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:#fff; padding:0 6px 10px; }
.mmx-pl-nav { position:relative; display:flex; align-items:center; gap:8px; width:100%; padding:6px 8px; border:none;
    border-radius:8px; background:transparent; color:#bbb; cursor:pointer; text-align:left; font-family:inherit; font-size:11.5px; }
.mmx-pl-nav:hover { background:#161616; color:#fff; }
.mmx-pl-nav.is-on { background:rgba(79,255,143,.12); color:#fff; box-shadow:inset 2px 0 0 #4fff8f; }
.mmx-pl-nav.is-drop { background:rgba(79,255,143,.25); }
.mmx-pl-nav-icon { width:14px; flex:0 0 auto; text-align:center; color:#8a8a8a; }
.mmx-pl-nav-label { flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mmx-pl-nav-count { color:#666; font-variant-numeric:tabular-nums; font-size:10px; }
.mmx-pl-nav-tools { display:none; gap:2px; }
.mmx-pl-nav:hover .mmx-pl-nav-tools { display:flex; }
.mmx-pl-nav:hover .mmx-pl-nav-count:not(:empty) { display:none; }
.mmx-pl-icon-btn { background:transparent; border:none; color:#888; cursor:pointer; padding:0 3px; font-size:12px; font-family:inherit; }
.mmx-pl-icon-btn:hover { color:#4fff8f; }
.mmx-pl-icon-btn.is-danger:hover { color:#e06c6c; }
.mmx-pl-side-head { display:flex; align-items:center; justify-content:space-between; padding:12px 8px 4px;
    font-size:9.5px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:#666; }
.mmx-pl-side-head .mmx-pl-icon-btn { font-size:15px; }
.mmx-pl-side-hint { padding:4px 8px; color:#5d6a62; font-size:10px; line-height:1.4; font-style:italic; }
.mmx-pl-side-sep { height:1px; background:#1f1f1f; margin:10px 4px; }
.mmx-pl-side-input { width:100%; box-sizing:border-box; background:#181818; border:1px solid #4fff8f; border-radius:6px;
    color:#eee; padding:5px 8px; font-size:11px; outline:none; font-family:inherit; }
.mmx-pl-main { display:flex; flex-direction:column; min-width:0; min-height:0; }
.mmx-pl-head { display:flex; align-items:center; gap:10px; padding:12px 14px; border-bottom:1px solid #222; }
.mmx-pl-heading { flex:1 1 auto; min-width:0; font-size:13px; font-weight:800; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mmx-pl-search { flex:0 1 330px; min-width:0; background:#1a1a1a; border:1px solid #333; border-radius:8px; color:#eee;
    padding:7px 10px; font-size:12px; outline:none; transition:border-color .15s, box-shadow .15s; }
.mmx-pl-search:focus { border-color:#4fff8f; box-shadow:0 0 0 3px rgba(79,255,143,.15); }
.mmx-pl-search[hidden] { display:none; }
.mmx-pl-close { background:transparent; border:none; color:#999; font-size:20px; line-height:1; cursor:pointer; padding:0 4px; }
.mmx-pl-close:hover { color:#e06c6c; }
.mmx-pl-body { flex:1 1 auto; min-height:0; overflow-y:auto; padding:12px 14px; scrollbar-width:thin; scrollbar-color:#3a3a3a transparent; }
.mmx-pl-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(160px, 1fr)); gap:12px; }
.mmx-pl-card { display:flex; flex-direction:column; padding:0; border-radius:12px; overflow:hidden; cursor:pointer;
    background:#141414; border:1px solid #262626; color:#eee; text-align:left; font:inherit;
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    animation:mmx-pl-rise .3s ease-out both; animation-delay:calc(var(--mmx-i, 0) * 14ms); }
.mmx-pl-card:hover { transform:translateY(-3px); border-color:#4fff8f; box-shadow:0 10px 24px rgba(0,0,0,.5), 0 0 0 1px rgba(79,255,143,.3); }
.mmx-pl-card.is-on { border-color:#4fff8f; box-shadow:0 0 0 2px #4fff8f inset; }
.mmx-pl-card-media { position:relative; display:block; aspect-ratio:4 / 3; overflow:hidden; background:#0e0e0e; }
.mmx-pl-card-media img { transition:transform .35s ease; }
.mmx-pl-card:hover .mmx-pl-card-media img { transform:scale(1.05); }
.mmx-pl-card-top { position:absolute; top:6px; left:6px; right:6px; display:flex; gap:4px; flex-wrap:wrap; pointer-events:none; }
.mmx-pl-card-text { display:flex; flex-direction:column; gap:3px; padding:8px 9px 9px; }
.mmx-pl-card-title { font-size:11.5px; font-weight:700; line-height:1.25; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mmx-pl-card-excerpt { font-size:10px; color:#8d8d8d; line-height:1.35; overflow:hidden;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
.mmx-pl-card-dots { display:flex; gap:4px; min-height:7px; }
.mmx-pl-dot { width:7px; height:7px; border-radius:50%; box-shadow:0 0 6px currentColor; }
.mmx-pl-badge { display:inline-flex; align-items:center; padding:1px 6px; border-radius:999px; font-size:9px; font-weight:700;
    background:rgba(0,0,0,.62); color:#ddd; border:1px solid rgba(255,255,255,.14); white-space:nowrap; }
.mmx-pl-badge.is-fav { color:#ffd35c; border-color:rgba(255,211,92,.45); }
.mmx-pl-badge.is-lora { color:#4fff8f; border-color:rgba(79,255,143,.35); margin-left:auto; }
.mmx-pl-empty { padding:60px 20px; text-align:center; color:#777; font-size:12px; line-height:1.6; }
.mmx-pl-histbar { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:12px; }

.mmx-pl-detail { display:flex; flex-direction:column; gap:8px; min-height:0; padding:14px; overflow-y:auto;
    background:#0b0b0b; border-left:1px solid #222; scrollbar-width:thin; scrollbar-color:#3a3a3a transparent; }
.mmx-pl-detail-empty { margin:auto; padding:20px; color:#666; text-align:center; font-size:11.5px; line-height:1.6; }
.mmx-pl-label { margin-top:4px; font-size:9.5px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:#7a7a7a; }
.mmx-pl-label.is-inline { margin:0; white-space:nowrap; }
.mmx-pl-input, .mmx-pl-textarea, .mmx-pl-select { width:100%; box-sizing:border-box; background:#161616; border:1px solid #2e2e2e;
    border-radius:7px; color:#eee; padding:6px 8px; font-size:11.5px; font-family:inherit; outline:none; }
.mmx-pl-input:focus, .mmx-pl-textarea:focus, .mmx-pl-select:focus { border-color:#4fff8f; }
.mmx-pl-input { font-size:13px; font-weight:700; }
.mmx-pl-input.is-small { font-size:11.5px; font-weight:400; }
.mmx-pl-textarea { min-height:100px; resize:vertical; line-height:1.4; }
.mmx-pl-textarea[readonly] { color:#bbb; }
.mmx-pl-meta { font-size:10.5px; color:#8fcfaa; }
.mmx-pl-muted { color:#666; font-style:italic; }
.mmx-pl-row { display:flex; gap:6px; align-items:center; }
.mmx-pl-row .mmx-pl-select { flex:1 1 auto; width:auto; }
.mmx-pl-loras { display:flex; flex-direction:column; gap:4px; }
.mmx-pl-lora { display:flex; align-items:center; gap:6px; padding:2px 8px 2px 2px; background:#151515; border:1px solid #262626; border-radius:6px; }
.mmx-pl-lora.is-off { opacity:.45; }
.mmx-pl-lora-name { flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mmx-pl-lora-str { color:#4fff8f; font-variant-numeric:tabular-nums; font-size:10px; }
.mmx-pl-cat-chips { display:flex; flex-wrap:wrap; gap:5px; }
.mmx-pl-cat-chip { background:transparent; border:1px solid color-mix(in srgb, var(--cat) 55%, transparent); color:#ccc;
    border-radius:999px; padding:2px 9px; font-size:10.5px; cursor:pointer; font-family:inherit; }
.mmx-pl-cat-chip.is-on { background:color-mix(in srgb, var(--cat) 24%, transparent); color:#fff; border-color:var(--cat); }
.mmx-pl-savebox { display:flex; flex-direction:column; gap:6px; }
.mmx-pl-apply { display:flex; flex-direction:column; gap:7px; padding:10px; margin-top:6px; border:1px solid #243428; border-radius:10px;
    background:linear-gradient(180deg, rgba(79,255,143,.07), transparent); }
.mmx-pl-apply .mmx-pl-label { margin-top:0; }
.mmx-pl-group-chips { display:flex; flex-wrap:wrap; gap:5px; }
.mmx-pl-group-chip { min-width:28px; height:26px; padding:0 8px; border-radius:7px; background:#171717; border:1px solid #333;
    color:#ccc; font-size:11px; font-weight:700; cursor:pointer; font-family:inherit; transition:all .12s; }
.mmx-pl-group-chip:hover { border-color:#4fff8f; }
.mmx-pl-group-chip.is-on { background:#4fff8f; border-color:#4fff8f; color:#062b14; }
.mmx-pl-apply-opts { display:flex; gap:5px; flex-wrap:wrap; }
.mmx-pl-toggle { background:transparent; border:1px dashed #3a3a3a; color:#888; border-radius:6px; padding:3px 8px;
    font-size:10.5px; cursor:pointer; font-family:inherit; }
.mmx-pl-toggle.is-on { border-style:solid; border-color:rgba(79,255,143,.5); color:#bfffd6; background:rgba(79,255,143,.08); }
.mmx-pl-toggle.is-mode { border-style:solid; border-color:#3a3a3a; color:#ddd; margin-left:auto; }
.mmx-pl-primary { background:#4fff8f; color:#062b14; border:1px solid #4fff8f; border-radius:8px; padding:8px 10px;
    font-size:12px; font-weight:800; cursor:pointer; font-family:inherit; transition:filter .15s, transform .1s; }
.mmx-pl-primary:hover:not(:disabled) { filter:brightness(1.08); }
.mmx-pl-primary:active:not(:disabled) { transform:scale(.98); }
.mmx-pl-primary:disabled { opacity:.45; cursor:not-allowed; }
.mmx-pl-primary.is-soft { background:transparent; color:#4fff8f; white-space:nowrap; }
.mmx-pl-secondary { background:#1b1b1b; border:1px solid #333; color:#ddd; border-radius:7px; padding:5px 10px;
    font-size:11px; cursor:pointer; font-family:inherit; }
.mmx-pl-secondary.is-dirty { border-color:#4fff8f; color:#4fff8f; box-shadow:0 0 0 3px rgba(79,255,143,.12); }
.mmx-pl-footer { display:flex; flex-wrap:wrap; gap:6px; margin-top:auto; padding-top:10px; border-top:1px solid #1d1d1d; }
.mmx-pl-icon-text { background:transparent; border:1px solid #2e2e2e; color:#bbb; border-radius:7px; padding:4px 8px;
    font-size:10.5px; cursor:pointer; font-family:inherit; white-space:nowrap; }
.mmx-pl-icon-text:hover { border-color:#4fff8f; color:#fff; }
.mmx-pl-icon-text.is-danger:hover { border-color:#e06c6c; color:#e06c6c; }

.mmx-pl-saveview { display:flex; flex-direction:column; min-height:100%; }
.mmx-pl-saverows { display:flex; flex-direction:column; gap:6px; }
.mmx-pl-saverow { display:flex; align-items:center; gap:10px; padding:6px 10px; background:#131313; border:1px solid #242424;
    border-radius:10px; cursor:pointer; }
.mmx-pl-saverow:hover { border-color:#3a5a45; }
.mmx-pl-saverow input { accent-color:#4fff8f; width:15px; height:15px; flex:0 0 auto; }
.mmx-pl-saverow-text { flex:1 1 auto; min-width:0; display:flex; flex-direction:column; gap:2px; }
.mmx-pl-saverow-title { font-weight:800; font-size:11.5px; }
.mmx-pl-saverow-excerpt { color:#8d8d8d; font-size:10.5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mmx-pl-savefoot { position:sticky; bottom:0; display:flex; align-items:center; gap:8px; margin-top:auto; padding:10px;
    background:#0f0f0f; border:1px solid #243428; border-radius:10px; }
.mmx-pl-savefoot .mmx-pl-select { width:auto; flex:1 1 auto; }

.mmx-pl-popover { position:fixed; z-index:10045; width:310px; display:flex; flex-direction:column; gap:8px; padding:10px;
    background:#101010; border:1px solid #2c2c2c; border-radius:12px; box-shadow:0 16px 40px rgba(0,0,0,.6); color:#eee;
    font-size:11px; animation:mmx-pl-pop .16s ease-out; }
.mmx-pl-pop-head { display:flex; align-items:center; gap:8px; min-width:0; }
.mmx-pl-pop-titles { display:flex; flex-direction:column; gap:2px; min-width:0; }
.mmx-pl-pop-title { font-weight:800; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mmx-pl-pop-sub { color:#8d8d8d; font-size:10px; line-height:1.35; overflow:hidden;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
.mmx-pl-flash { position:fixed; z-index:10060; transform:translate(-50%, -100%); background:#4fff8f; color:#062b14;
    font-size:10px; font-weight:800; padding:3px 8px; border-radius:6px; pointer-events:none; white-space:nowrap;
    box-shadow:0 4px 12px rgba(0,0,0,.4); transition:opacity .25s, transform .25s; }
.mmx-pl-flash.out { opacity:0; transform:translate(-50%, -150%); }
@media (max-width: 900px) {
    .mmx-pl-dialog { grid-template-columns:170px minmax(0, 1fr); }
    .mmx-pl-detail { display:none; }
}`;
    document.head.appendChild(style);
}
