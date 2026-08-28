/** Director pack (script + assets) import/export UI. */

import { api } from "../../scripts/api.js";
import { t } from "./minimax_i18n.js";
import { safeUploadFilename } from "./minimax_gen_timeline.js";

const PACK_WIDGET_NAMES = ["steps", "sampler", "scheduler", "cfg", "shift_video", "shift_audio", "seed"];
const LARGE_PACK_BYTES = 500 * 1024 * 1024;
const CHUNK_SOFT_LIMIT = 95 * 1024 * 1024;
const CHUNK_SIZE = 8 * 1024 * 1024;

export function collectPackWidgets(editor) {
    const widgets = {};
    for (const name of PACK_WIDGET_NAMES) {
        const w = editor.widget?.(name);
        if (w && w.value != null && w.value !== "") widgets[name] = w.value;
    }
    const task = editor.taskTypeWidget?.value || editor.timeline?.global?.taskType || "";
    if (task) widgets.task_type = task;
    return widgets;
}

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

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || "MiniMaxH3Director.mmxpack.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function formatMissing(missing) {
    const list = (missing || []).filter(Boolean);
    if (!list.length) return "";
    const shown = list.slice(0, 12).join("\n");
    const extra = list.length > 12 ? `\n… +${list.length - 12}` : "";
    return shown + extra;
}

function pickZipFile() {
    return new Promise((resolve) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".zip,application/zip";
        input.style.display = "none";
        input.addEventListener("change", () => {
            const file = input.files?.[0] || null;
            input.remove();
            resolve(file);
        }, { once: true });
        document.body.appendChild(input);
        input.click();
    });
}

async function uploadZipChunked(file, onProgress) {
    const filename = safeUploadFilename(file?.name || "pack.mmxpack.zip", "application/zip");
    const zipName = filename.toLowerCase().endsWith(".zip") ? filename : `${filename}.zip`;
    const uploadId = crypto.randomUUID();
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const body = new FormData();
        body.append("upload_id", uploadId);
        body.append("chunk_index", String(i));
        body.append("total_chunks", String(totalChunks));
        body.append("filename", zipName);
        body.append("chunk", file.slice(start, end), `${zipName}.part`);
        const resp = await api.fetchApi("/minimax/director/upload_chunk", { method: "POST", body });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || t("upload.chunkFailed", { status: resp.status }));
        }
        onProgress?.((i + 1) / totalChunks);
        const data = await resp.json();
        if (data.name) return data;
    }
    throw new Error(t("upload.chunkIncomplete"));
}

export async function exportDirectorPack(editor) {
    editor.flushTimelineSync?.();
    const timeline = editor.buildTimelinePayload();
    const widgets = collectPackWidgets(editor);
    const preview = await postJson("/minimax/director/export_pack", {
        timeline,
        widgets,
        dryRun: true,
    });
    if (Number(preview.totalBytes) > LARGE_PACK_BYTES) {
        const mb = Math.max(1, Math.round(Number(preview.totalBytes) / (1024 * 1024)));
        if (!window.confirm(t("pack.largeConfirm", { mb }))) return;
    }
    const result = await postJson("/minimax/director/export_pack", { timeline, widgets });
    const filename = result.filename;
    if (!filename) throw new Error(t("pack.exportFailed"));
    const downloadName = result.downloadName || filename;
    const qs = `filename=${encodeURIComponent(filename)}&download=${encodeURIComponent(downloadName)}`;
    const resp = await api.fetchApi(`/minimax/director/download_pack?${qs}`);
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || t("pack.exportFailed"));
    }
    const blob = await resp.blob();
    downloadBlob(blob, downloadName);
    const miss = formatMissing(result.missing);
    if (miss) {
        window.alert(`${t("pack.exportDoneMissing")}\n${miss}`);
    }
}

export async function importDirectorPack(editor) {
    if (!window.confirm(t("pack.confirmReplace"))) return;
    const file = await pickZipFile();
    if (!file) return;
    let data;
    if (file.size <= CHUNK_SOFT_LIMIT) {
        const body = new FormData();
        body.append("pack", file, file.name || "pack.mmxpack.zip");
        const resp = await api.fetchApi("/minimax/director/import_pack", { method: "POST", body });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || t("pack.importFailed"));
        }
        data = await resp.json();
    } else {
        const uploaded = await uploadZipChunked(file);
        data = await postJson("/minimax/director/import_pack", {
            filename: uploaded.name || uploaded.filename,
            subfolder: uploaded.subfolder || "",
            type: uploaded.type || "input",
        });
    }
    if (!data?.timeline) throw new Error(t("pack.importFailed"));
    editor.applyImportedTimeline(data.timeline, data.widgets || {});
    const miss = formatMissing(data.missing);
    if (miss) {
        window.alert(`${t("pack.importDoneMissing")}\n${miss}`);
    }
}

export function bindPackActions(editor) {
    const bind = (sel, fn) => {
        const el = editor.root?.querySelector(sel);
        if (!el) return;
        el.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            try {
                await fn();
            } catch (err) {
                console.error("[MiniMax H3 Director] pack:", err);
                window.alert(t("pack.alertFailed", { err: err?.message || err }));
            }
        };
    };
    bind('[data-a="pack-export"]', () => exportDirectorPack(editor));
    bind('[data-a="pack-import"]', () => importDirectorPack(editor));
}
