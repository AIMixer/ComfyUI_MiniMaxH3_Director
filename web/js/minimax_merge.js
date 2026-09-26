/** Manual "merge segments into one video" UI (segment-export / deferred runs). */

import { api } from "../../scripts/api.js";
import { t } from "./minimax_i18n.js";

async function postJson(path, body) {
    const resp = await api.fetchApi(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `HTTP ${resp.status}`);
    }
    return resp.json();
}

async function mergeAlert(editor, message, title) {
    if (typeof editor?.showBdMessage === "function") {
        await editor.showBdMessage(title || t("merge.failedTitle"), message);
        return;
    }
    window.alert(title ? `${title}\n${message}` : message);
}

async function mergeConfirm(editor, message, title) {
    if (typeof editor?.showBdDialog === "function") {
        return Boolean(await editor.showBdDialog({
            title: title || t("toolbar.mergeSegments"),
            message,
            confirmText: t("dialog.confirm"),
            cancelText: t("dialog.cancel"),
        }));
    }
    return window.confirm(message);
}

function getSelect(editor) {
    return editor?.root?.querySelector('[data-r="merge-run-select"]') || null;
}

function getModeSelect(editor) {
    return editor?.root?.querySelector('[data-r="merge-mode-select"]') || null;
}

function frameRate(editor) {
    try {
        return Number(editor?.getFrameRate?.() || 24) || 24;
    } catch {
        return 24;
    }
}

function runLabel(run) {
    const count = Number(run.count) || 0;
    let label = `${run.dir} · ${count}${t("merge.optionSuffix")}`;
    if (run.projectName) label += ` · ${run.projectName}`;
    return label;
}

function fmtTime(sec) {
    const n = Number(sec);
    if (!n) return "";
    try {
        return new Date(n * 1000).toLocaleString();
    } catch {
        return "";
    }
}

function kindLabel(kind) {
    if (kind === "pre") return t("merge.kindPre");
    if (kind === "facepre") return t("merge.kindFacepre");
    if (kind === "pass") return t("merge.kindPass");
    return t("merge.kindFinal");
}

/** Fetch segment-export runs from the backend; optionally scoped to a project. */
async function fetchRuns(projectId) {
    const query = projectId
        ? `/minimax/director/list_segment_runs?project_id=${encodeURIComponent(projectId)}`
        : "/minimax/director/list_segment_runs";
    const resp = await api.fetchApi(query);
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data?.runs) ? data.runs : [];
}

/** Reload the run dropdown from the backend; newest run first. */
export async function refreshMergeRuns(editor, projectId) {
    const select = getSelect(editor);
    if (!select) return [];
    let runs = [];
    try {
        runs = await fetchRuns(projectId);
    } catch (err) {
        console.error("[MiniMax H3 Director] list segment runs:", err);
    }
    const prev = select.value;
    select.innerHTML = "";
    if (!runs.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = t("merge.selectEmpty");
        select.appendChild(opt);
        return runs;
    }
    for (const run of runs) {
        const opt = document.createElement("option");
        opt.value = run.dir;
        opt.textContent = runLabel(run);
        select.appendChild(opt);
    }
    if (prev && runs.some((r) => r.dir === prev)) select.value = prev;
    return runs;
}

/* --------------------------- merge progress UI --------------------------- */

function genJobId() {
    try {
        if (window.crypto?.randomUUID) return window.crypto.randomUUID();
    } catch {
        /* ignore */
    }
    return "mj_" + Date.now() + "_" + Math.random().toString(36).slice(2);
}

/** Translate a backend phase token (e.g. "video 2/3") into a localized line. */
function progressMsg(code) {
    const c = String(code || "");
    if (c.startsWith("probe")) return t("merge.progressProbe", { detail: c });
    if (c.startsWith("decode")) return t("merge.progressDecode", { detail: c });
    if (c.startsWith("video")) return t("merge.progressVideo", { detail: c });
    if (c === "audio") return t("merge.progressAudio");
    if (c === "encode") return t("merge.progressEncode");
    if (c === "finalize") return t("merge.progressFinalize");
    if (c === "done") return t("merge.progressDone");
    if (c === "error") return t("merge.progressError");
    return c || t("merge.progressInit");
}

function closeProgress(editor) {
    const el = editor?.root?.querySelector('[data-r="merge-progress-overlay"]');
    if (el) el.remove();
}

function openProgress(editor) {
    const root = editor?.root;
    if (!root) return;
    closeProgress(editor);
    const overlay = document.createElement("div");
    overlay.className = "bd-modal-overlay bd-progress-overlay";
    overlay.setAttribute("data-r", "merge-progress-overlay");
    const panel = document.createElement("div");
    panel.className = "bd-modal bd-progress-panel";
    panel.innerHTML = `
        <div class="bd-modal-title"></div>
        <div class="bd-progress-track"><div class="bd-progress-fill" data-r="merge-progress-fill"></div></div>
        <div class="bd-progress-foot">
            <span class="bd-progress-msg" data-r="merge-progress-msg"></span>
            <span class="bd-progress-pct" data-r="merge-progress-pct">0%</span>
        </div>`;
    panel.querySelector(".bd-modal-title").textContent = t("merge.progressTitle");
    panel.querySelector('[data-r="merge-progress-msg"]').textContent = t("merge.progressInit");
    overlay.appendChild(panel);
    root.appendChild(overlay);
}

function updateProgress(editor, percent, message) {
    const el = editor?.root?.querySelector('[data-r="merge-progress-overlay"]');
    if (!el) return;
    const p = Math.max(0, Math.min(100, Number(percent) || 0));
    const fill = el.querySelector('[data-r="merge-progress-fill"]');
    const msg = el.querySelector('[data-r="merge-progress-msg"]');
    const pct = el.querySelector('[data-r="merge-progress-pct"]');
    if (fill) fill.style.width = p + "%";
    if (pct) pct.textContent = Math.round(p) + "%";
    if (msg) msg.textContent = progressMsg(message);
}

async function doMerge(editor, body) {
    const jobId = genJobId();
    const overlayBody = { ...body, job_id: jobId };
    openProgress(editor);
    let stopped = false;
    const pollPromise = (async () => {
        while (!stopped) {
            await new Promise((r) => setTimeout(r, 700));
            if (stopped) break;
            try {
                const resp = await api.fetchApi(
                    `/minimax/director/merge_progress?job_id=${encodeURIComponent(jobId)}`,
                );
                if (!resp.ok) continue;
                const st = await resp.json();
                updateProgress(editor, st.percent, st.message);
                if (st.done) break;
            } catch {
                /* transient poll error — keep trying */
            }
        }
    })();
    try {
        const data = await postJson("/minimax/director/merge_segments", overlayBody);
        stopped = true;
        updateProgress(editor, 100, "done");
        await new Promise((r) => setTimeout(r, 300));
        closeProgress(editor);
        await pollPromise;
        const duration = Math.round((Number(data.duration_s) || 0) * 100) / 100;
        let msg = t("merge.done", {
            count: data.segments,
            name: data.name,
            duration,
            frames: data.total_frames,
            path: data.path,
        });
        const usedRuns = Array.isArray(data.runs) ? data.runs : [];
        if (usedRuns.length) {
            msg += `\n${t("merge.doneSources", { runs: usedRuns.join(", ") })}`;
        }
        if (Number(data.preCount) > 0) {
            msg += `\n${t("merge.doneMix", { count: data.segments, pre: data.preCount })}`;
        }
        await mergeAlert(editor, msg, t("merge.doneTitle"));
    } catch (err) {
        stopped = true;
        closeProgress(editor);
        await pollPromise;
        console.error("[MiniMax H3 Director] merge:", err);
        await mergeAlert(editor, String(err?.message || err), t("merge.failedTitle"));
    }
}

/**
 * Merge segments into one video.
 *
 * Modes:
 *   「本项目最新」 — for every segment number, the newest version across the
 *   current project's runs is used（二采新就用二采，一采新就用一采）.
 *   「仅所选段」 — merge exactly the segments picked in the picker (本目录挑段).
 *
 * In 「仅所选段」 the merge is driven by the picker: no picker selection yet means
 * we open the picker so the user can pick (选目录仅为挑段，挑段后按段号合并成片).
 */
export async function mergeSelectedSegments(editor) {
    const mode = getModeSelect(editor)?.value || "project";
    const fps = frameRate(editor);
    const projectId = editor.activeProjectId;
    const runs = await refreshMergeRuns(editor, projectId);
    if (!runs.length) {
        await mergeAlert(
            editor,
            projectId ? t("merge.noneProject") : t("merge.none"),
            t("toolbar.mergeSegments"),
        );
        return;
    }

    if (mode === "single") {
        hydratePicks(editor);
        const picks = Array.isArray(editor?._mergePicks) ? editor._mergePicks : [];
        if (!picks.length) {
            await mergeAlert(editor, t("merge.pickFirst"), t("toolbar.mergeSegments"));
            await openMergePicker(editor);
            return;
        }
        if (!await mergeConfirm(
            editor,
            t("merge.confirmPicks", { count: picks.length }),
            t("toolbar.mergeSegments"),
        )) return;
        await doMerge(editor, {
            mode: "custom",
            picks,
            fps,
        });
        return;
    }

    const distinct = new Set();
    for (const run of runs) {
        for (const idx of (run.segs || [])) distinct.add(Number(idx));
    }
    if (!distinct.size) {
        await mergeAlert(editor, t("merge.noneProject"), t("toolbar.mergeSegments"));
        return;
    }

    if (!await mergeConfirm(
        editor,
        t("merge.confirmProject", { count: distinct.size, project: projectId || "" }),
        t("toolbar.mergeSegments"),
    )) return;
    await doMerge(editor, {
        mode: "latest",
        runs: runs.map((r) => r.dir),
        fps,
    });
}

/* ----------------------------- picker panel ------------------------------ */

function closePicker(editor) {
    const root = editor?.root;
    if (editor?._mergePickKey) {
        window.removeEventListener("keydown", editor._mergePickKey, true);
        editor._mergePickKey = null;
    }
    const el = root?.querySelector('[data-r="merge-pick-overlay"]');
    if (el) el.remove();
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
}

function versionRank(suffix) {
    const tag = String(suffix || "").toLowerCase();
    if (!tag) return 3;
    if (tag === "facepre") return 1;
    if (tag === "pre") return 2;
    if (/^p\d+$/.test(tag)) return 3;
    return 2;
}

/**
 * Manual per-segment picker: a scrollable grid of every segment mp4 in the
 * folder. Each card holds a checkbox + filename + a playable <video>. Card
 * grid flows 3–5 columns depending on panel width. This panel is selection
 * only — it does NOT merge. The checked files are persisted to
 * ``editor._mergePicks`` and the external「合并成片」button performs the merge.
 */
export async function openMergePicker(editor) {
    const root = editor?.root;
    if (!root) return;
    closePicker(editor);
    hydratePicks(editor);

    const overlay = document.createElement("div");
    overlay.className = "bd-modal-overlay bd-pick-overlay";
    overlay.setAttribute("data-r", "merge-pick-overlay");
    const panel = document.createElement("div");
    panel.className = "bd-modal";
    panel.style.maxWidth = "960px";
    panel.style.width = "94%";
    panel.innerHTML = `
        <div class="bd-modal-title"></div>
        <div class="bd-modal-body"></div>
        <div class="bd-pick-toolbar">
            <span class="bd-pick-count" data-r="pick-count">0</span>
            <div class="bd-pick-toolbar-actions">
                <button type="button" class="bd-btn" data-r="pick-select-all"></button>
                <button type="button" class="bd-btn" data-r="pick-clear"></button>
            </div>
        </div>
        <div class="bd-pick-grid" data-r="pick-grid" data-placeholder="1"></div>
        <div class="bd-modal-actions"></div>`;
    panel.querySelector(".bd-modal-title").textContent = t("merge.pickTitle");
    panel.querySelector(".bd-modal-body").textContent = t("merge.pickHint");
    panel.querySelector('[data-r="pick-select-all"]').textContent = t("merge.pickSelectAll");
    panel.querySelector('[data-r="pick-clear"]').textContent = t("merge.pickClear");
    overlay._toolbar = {
        countEl: panel.querySelector('[data-r="pick-count"]'),
        selectAllBtn: panel.querySelector('[data-r="pick-select-all"]'),
        clearBtn: panel.querySelector('[data-r="pick-clear"]'),
    };
    panel.querySelector('[data-r="pick-select-all"]').onclick = () => selectAllPicks(overlay, editor);
    panel.querySelector('[data-r="pick-clear"]').onclick = () => clearPicks(overlay, editor);

    const actionsEl = panel.querySelector(".bd-modal-actions");
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "bd-btn";
    closeBtn.textContent = t("merge.pickConfirm");
    closeBtn.onclick = () => {
        savePicks(editor, overlay);
        closePicker(editor);
    };
    actionsEl.appendChild(closeBtn);

    overlay.addEventListener("mousedown", (e) => {
        if (e.target === overlay) {
            savePicks(editor, overlay);
            closePicker(editor);
        }
    });
    editor._mergePickKey = (e) => {
        if (e.key === "Escape") {
            e.stopPropagation();
            savePicks(editor, overlay);
            closePicker(editor);
        }
    };
    window.addEventListener("keydown", editor._mergePickKey, true);

    overlay.appendChild(panel);
    root.appendChild(overlay);
    const mode = getModeSelect(editor)?.value || "project";
    overlay._scope = { mode };
    if (mode === "single") {
        overlay._scope.run = getSelect(editor)?.value || null;
    } else {
        overlay._scope.projectId = editor.activeProjectId;
    }
    await loadPicker(editor, overlay);
    closeBtn.focus();
}

/**
 * Persist the current checked files to ``editor._mergePicks`` for later merge.
 *
 * Only the runs covered by THIS panel's scope are replaced; picks that belong
 * to other runs (e.g. selections made after switching the run dropdown) are
 * preserved. Without this, switching the timestamp dropdown and switching back
 * would wipe the previous selection.
 */
function savePicks(editor, overlay) {
    const picks = buildPicks(overlay);
    if (!editor) return;
    const prev = Array.isArray(editor._mergePicks) ? editor._mergePicks.slice() : [];
    const scopeRuns = new Set();
    for (const seg of overlay._segments || []) {
        for (const v of seg.versions || []) scopeRuns.add(String(v.run || ""));
    }
    const kept = prev.filter((p) => !scopeRuns.has(String(p.run || "")));
    editor._mergePicks = kept.concat(picks);
    persistPicks(editor);
}

/** Mirror the picked segments into node properties so they survive a reload. */
function persistPicks(editor) {
    const node = editor?.node;
    if (!node) return;
    node.properties = node.properties || {};
    node.properties.mmh3_merge_picks_v1 =
        Array.isArray(editor._mergePicks) ? editor._mergePicks : [];
}

/** Restore picks from node properties when the in-memory list is empty. */
function hydratePicks(editor) {
    if (!editor) return;
    if (Array.isArray(editor._mergePicks) && editor._mergePicks.length) return;
    const saved = editor?.node?.properties?.mmh3_merge_picks_v1;
    if (Array.isArray(saved)) editor._mergePicks = saved.slice();
}

/** Version unique key across folders: same filename may exist in many runs. */
function pickKey(version) {
    return String(version?.run || "") + "::" + String(version?.name || "");
}

/** Total number of currently checked versions across all segments. */
function countChecked(overlay) {
    let n = 0;
    for (const seg of overlay._segments || []) n += (seg._checked?.size || 0);
    return n;
}

function updatePickCount(overlay) {
    if (overlay?._toolbar?.countEl) {
        overlay._toolbar.countEl.textContent = t("merge.pickCount", { count: countChecked(overlay) });
    }
}

function selectAllPicks(overlay, editor) {
    for (const seg of overlay._segments || []) {
        if (!seg._checked) seg._checked = new Set();
        for (const v of seg.versions || []) seg._checked.add(pickKey(v));
    }
    savePicks(editor, overlay);
    reapplyPicks(overlay);
}

function clearPicks(overlay, editor) {
    for (const seg of overlay._segments || []) {
        if (seg._checked) seg._checked.clear();
    }
    savePicks(editor, overlay);
    reapplyPicks(overlay);
}

/** Re-render just the checkbox states (and count) without rebuilding the grid. */
function reapplyPicks(overlay) {
    const checkedSet = new Set();
    for (const seg of overlay._segments || []) {
        for (const k of seg._checked || []) checkedSet.add(k);
    }
    const gridEl = overlay.querySelector('[data-r="pick-grid"]');
    const cards = gridEl?.querySelectorAll?.('.bd-pick-card[data-key]');
    if (cards) {
        for (const card of cards) {
            const checked = checkedSet.has(card.getAttribute("data-key"));
            card.classList.toggle("checked", !!checked);
            const cb = card.querySelector("input[type=checkbox]");
            if (cb) cb.checked = !!checked;
        }
    }
    updatePickCount(overlay);
}

/** All checked files across runs, in (segment, mtime) order. */
function buildPicks(overlay) {
    const picks = [];
    for (const seg of overlay._segments || []) {
        const checked = (seg.versions || []).filter((v) => seg._checked?.has(pickKey(v)));
        for (const v of checked) {
            picks.push({
                seg: seg.index,
                run: v.run,
                name: v.name,
                suffix: v.suffix,
                mtime: v.mtime || 0,
            });
        }
    }
    picks.sort((a, b) => a.seg - b.seg || (a.mtime || 0) - (b.mtime || 0));
    return picks;
}

async function loadPicker(editor, overlay) {
    const gridEl = overlay.querySelector('[data-r="pick-grid"]');
    gridEl.innerHTML = "";
    if (gridEl.dataset.placeholder) gridEl.textContent = t("merge.pickLoading");

    let data = null;
    const scope = overlay._scope || {};
    const params = new URLSearchParams();
    if (scope.run) params.set("run", scope.run);
    else if (scope.projectId) params.set("project_id", scope.projectId);
    const qs = params.toString();
    try {
        const query = qs
            ? `/minimax/director/list_segment_versions?${qs}`
            : "/minimax/director/list_segment_versions";
        const resp = await api.fetchApi(query);
        if (resp.ok) data = await resp.json();
    } catch (err) {
        console.error("[MiniMax H3 Director] list segment versions:", err);
    }
    const segments = Array.isArray(data?.segments) ? data.segments : [];
    if (!segments.length) {
        gridEl.textContent = t("merge.pickEmpty");
        return;
    }
    overlay._segments = segments;
    renderGrid(overlay, editor);
}

function renderGrid(overlay, editor) {
    const gridEl = overlay.querySelector('[data-r="pick-grid"]');
    gridEl.innerHTML = "";
    const savedPicks = new Map();
    if (editor && Array.isArray(editor._mergePicks)) {
        for (const p of editor._mergePicks) {
            savedPicks.set(String(p.run || "") + "::" + String(p.name || ""), true);
        }
    }
    for (const seg of overlay._segments) {
        const segNo = seg.index + 1;
        if (!seg._checked) seg._checked = new Set();
        if (!seg._seeded) {
            seg._seeded = true;
            for (const v of seg.versions || []) {
                if (savedPicks.has(pickKey(v))) seg._checked.add(pickKey(v));
            }
        }
        const label = document.createElement("div");
        label.className = "bd-pick-seg-label";
        label.textContent = t("merge.pickSeg", { no: segNo });
        gridEl.appendChild(label);

        const versions = (seg.versions || []).slice().sort((a, b) => {
            a._rank = a._rank || versionRank(a.suffix);
            b._rank = b._rank || versionRank(b.suffix);
            return b._rank - a._rank || b.mtime - a.mtime;
        });
        versions.forEach((version) => {
            const key = pickKey(version);
            const checked = seg._checked.has(key);
            const card = document.createElement("div");
            card.className = "bd-pick-card" + (checked ? " checked" : "");
            card.setAttribute("data-key", key);
            card.innerHTML = `
                <div class="bd-pick-card-head">
                    <label class="bd-pick-card-label">
                        <input type="checkbox"${checked ? " checked" : ""}>
                        <span class="bd-pick-card-name" title="${escapeHtml(version.name)}">${escapeHtml(version.name)}</span>
                    </label>
                </div>
                <video muted playsinline preload="metadata" controls src="${escapeHtml(version.url)}"></video>
                <div class="bd-pick-card-meta">${kindLabel(version.kind)} · ${escapeHtml(version.run)}${version.auto ? ` · ${t("merge.badgeAuto")}` : ""}</div>`;
            const cb = card.querySelector("input[type=checkbox]");
            cb.onchange = () => {
                if (cb.checked) seg._checked.add(key);
                else seg._checked.delete(key);
                card.classList.toggle("checked", cb.checked);
                savePicks(editor, overlay);
                updatePickCount(overlay);
            };
            const head = card.querySelector(".bd-pick-card-head");
            head.addEventListener("click", (e) => {
                if (e.target.tagName === "INPUT") return;
                cb.checked = !cb.checked;
                cb.dispatchEvent(new Event("change"));
            });
            const video = card.querySelector("video");
            video.addEventListener("loadedmetadata", () => {
                if (video.duration > 0.1) video.currentTime = 0.01;
            });
            gridEl.appendChild(card);
        });
    }
    updatePickCount(overlay);
}

export function bindMergeActions(editor) {
    const root = editor?.root;
    if (!root) return;
    const mergeBtn = root.querySelector('[data-a="merge-segments"]');
    if (mergeBtn) {
        mergeBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            void mergeSelectedSegments(editor);
        };
    }
    const pickBtn = root.querySelector('[data-a="merge-pick"]');
    if (pickBtn) {
        pickBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            openMergePicker(editor).catch((err) => {
                console.error("[MiniMax H3 Director] openMergePicker:", err);
                void mergeAlert(
                    editor,
                    String(err?.stack || err?.message || err),
                    "Picker error",
                );
            });
        };
    }
    const select = getSelect(editor);
    if (select) {
        // Refresh on open so newly finished runs show up without reloading the page.
        select.addEventListener("mousedown", () => {
            void refreshMergeRuns(editor, editor.activeProjectId);
        });
    }
    const modeSel = getModeSelect(editor);
    const updateModeUI = () => {
        const single = modeSel ? modeSel.value === "single" : false;
        if (select) select.classList.toggle("hidden", !single);
        if (pickBtn) pickBtn.classList.toggle("hidden", !single);
        // 「仅所选段」才需要目录下拉和挑段；切到该模式时刷新目录。
        if (single) void refreshMergeRuns(editor, editor.activeProjectId);
    };
    if (modeSel) {
        modeSel.addEventListener("change", updateModeUI);
    }
    updateModeUI();
}