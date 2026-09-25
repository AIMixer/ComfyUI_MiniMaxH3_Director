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

/** Reload the run dropdown from the backend; newest run first. */
export async function refreshMergeRuns(editor) {
    const select = getSelect(editor);
    if (!select) return [];
    let runs = [];
    try {
        const resp = await api.fetchApi("/minimax/director/list_segment_runs");
        if (resp.ok) {
            const data = await resp.json();
            runs = Array.isArray(data?.runs) ? data.runs : [];
        }
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
        select.disabled = true;
        return runs;
    }
    select.disabled = false;
    for (const run of runs) {
        const opt = document.createElement("option");
        opt.value = run.dir;
        const count = Number(run.count) || 0;
        opt.textContent = `${run.dir} · ${count}${t("merge.optionSuffix")}`;
        select.appendChild(opt);
    }
    if (prev && runs.some((r) => r.dir === prev)) select.value = prev;
    return runs;
}

/** Merge the selected (default: newest) segment-export run into one MP4. */
export async function mergeSelectedSegments(editor) {
    const select = getSelect(editor);
    const runs = await refreshMergeRuns(editor);
    if (!runs.length) {
        await mergeAlert(editor, t("merge.none"), t("toolbar.mergeSegments"));
        return;
    }
    const run = select && select.value ? select.value : runs[0].dir;
    const info = runs.find((r) => r.dir === run) || runs[0];
    if (!await mergeConfirm(editor, t("merge.confirm", { run, count: info.count }), t("toolbar.mergeSegments"))) {
        return;
    }
    let fps = 24;
    try {
        fps = Number(editor?.getFrameRate?.() || 24) || 24;
    } catch {
        fps = 24;
    }
    try {
        const data = await postJson("/minimax/director/merge_segments", { run, fps });
        const duration = Math.round((Number(data.duration_s) || 0) * 100) / 100;
        await mergeAlert(editor, t("merge.done", {
            count: data.segments,
            name: data.name,
            duration,
            frames: data.total_frames,
            path: data.path,
        }), t("merge.doneTitle"));
    } catch (err) {
        console.error("[MiniMax H3 Director] merge:", err);
        await mergeAlert(editor, String(err?.message || err), t("merge.failedTitle"));
    }
}

export function bindMergeActions(editor) {
    const btn = editor?.root?.querySelector('[data-a="merge-segments"]');
    if (btn) {
        btn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            void mergeSelectedSegments(editor);
        };
    }
    const select = getSelect(editor);
    if (select) {
        // Refresh on open so newly finished runs show up without reloading the page.
        select.addEventListener("mousedown", () => { void refreshMergeRuns(editor); });
    }
    void refreshMergeRuns(editor);
}