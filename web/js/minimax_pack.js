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

async function packConfirm(editor, message, title) {
    if (typeof editor?.showBdDialog === "function") {
        return Boolean(await editor.showBdDialog({
            title: title || t("toolbar.importPack"),
            message,
            confirmText: t("dialog.confirm"),
            cancelText: t("dialog.cancel"),
        }));
    }
    return window.confirm(message);
}

async function packAlert(editor, message, title) {
    if (typeof editor?.showBdMessage === "function") {
        await editor.showBdMessage(title || t("pack.failedTitle"), message);
        return;
    }
    window.alert(title ? `${title}\n${message}` : message);
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

function ensurePackProgressStyles() {
    if (document.getElementById("bd-pack-progress-style")) return;
    const style = document.createElement("style");
    style.id = "bd-pack-progress-style";
    style.textContent = `
.bd-pack-progress{pointer-events:auto}
.bd-pack-bar{height:6px;background:#2a2a2a;border-radius:99px;overflow:hidden}
.bd-pack-bar-fill{height:100%;width:0;background:#3cb054;border-radius:99px}
.bd-pack-bar.is-indet .bd-pack-bar-fill{width:35%;animation:bd-pack-indet 1.1s ease-in-out infinite}
@keyframes bd-pack-indet{0%{transform:translateX(-120%)}100%{transform:translateX(320%)}}
`;
    document.head.appendChild(style);
}

function openPackProgress(editor, titleKey = "pack.importProgressTitle") {
    ensurePackProgressStyles();
    const host = editor?.root || document.body;
    const overlay = document.createElement("div");
    overlay.className = "bd-modal-overlay bd-pack-progress";
    overlay.innerHTML = `
        <div class="bd-modal">
            <div class="bd-modal-title"></div>
            <div class="bd-modal-body" data-pack-label></div>
            <div class="bd-pack-bar" data-pack-bar><div class="bd-pack-bar-fill" data-pack-fill></div></div>
        </div>`;
    overlay.querySelector(".bd-modal-title").textContent = t(titleKey);
    overlay.addEventListener("pointerdown", (e) => e.stopPropagation());
    overlay.addEventListener("click", (e) => e.stopPropagation());
    host.appendChild(overlay);
    const label = overlay.querySelector("[data-pack-label]");
    const bar = overlay.querySelector("[data-pack-bar]");
    const fill = overlay.querySelector("[data-pack-fill]");
    const paint = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
    return {
        async bar(labelKey, ratio) {
            const pct = Math.max(0, Math.min(100, Math.round((Number(ratio) || 0) * 100)));
            label.textContent = t(labelKey, { pct });
            bar.classList.remove("is-indet");
            fill.style.width = `${pct}%`;
            await paint();
        },
        async upload(ratio) {
            await this.bar("pack.importUploading", ratio);
        },
        async phase(key) {
            label.textContent = t(key);
            bar.classList.add("is-indet");
            fill.style.width = "";
            await paint();
        },
        hide() {
            overlay.style.display = "none";
        },
        show() {
            overlay.style.display = "";
        },
        close() {
            overlay.remove();
        },
    };
}

async function readResponseBytes(resp, onRatio) {
    const total = Number(resp.headers.get("Content-Length") || 0);
    if (!resp.body || typeof resp.body.getReader !== "function") {
        const bytes = new Uint8Array(await resp.arrayBuffer());
        if (bytes.byteLength) onRatio?.(1);
        return bytes;
    }
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        if (total > 0) onRatio?.(Math.min(1, loaded / total));
    }
    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
    }
    return bytes;
}

function postPackUpload(file, onUploadDone) {
    return new Promise((resolve, reject) => {
        const body = new FormData();
        body.append("pack", file, file.name || "pack.mmxpack.zip");
        const xhr = new XMLHttpRequest();
        const url = typeof api.apiURL === "function"
            ? api.apiURL("/minimax/director/import_pack")
            : "/minimax/director/import_pack";
        xhr.open("POST", url);
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && e.total > 0) onUploadDone?.("upload", e.loaded / e.total);
        };
        xhr.upload.onload = () => onUploadDone?.("processing");
        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    resolve(JSON.parse(xhr.responseText || "{}"));
                } catch (err) {
                    reject(err);
                }
                return;
            }
            reject(new Error(xhr.responseText || t("pack.importFailed")));
        };
        xhr.onerror = () => reject(new Error(t("pack.importFailed")));
        xhr.send(body);
    });
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
    const progress = openPackProgress(editor, "pack.exportProgressTitle");
    let missing = [];
    let downloadName = "MiniMaxH3Director.mmxpack.zip";
    let blob = null;
    try {
        await progress.phase("pack.exportScanning");
        const preview = await postJson("/minimax/director/export_pack", {
            timeline,
            widgets,
            dryRun: true,
        });
        if (Number(preview.totalBytes) > LARGE_PACK_BYTES) {
            progress.hide();
            const mb = Math.max(1, Math.round(Number(preview.totalBytes) / (1024 * 1024)));
            const ok = await packConfirm(editor, t("pack.largeConfirm", { mb }), t("toolbar.exportPack"));
            if (!ok) return;
            progress.show();
        }
        await progress.phase("pack.exportPacking");
        const resp = await api.fetchApi("/minimax/director/export_pack", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ timeline, widgets }),
        });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || t("pack.exportFailed"));
        }
        const ctype = (resp.headers.get("Content-Type") || "").toLowerCase();
        downloadName = resp.headers.get("X-Pack-Download-Name") || downloadName;
        const bytes = await readResponseBytes(resp, (ratio) => {
            progress.bar("pack.exportDownloading", ratio);
        });
        await progress.phase("pack.exportSaving");
        if (ctype.includes("application/json")) {
            const result = JSON.parse(new TextDecoder().decode(bytes));
            downloadName = result.downloadName || downloadName;
            missing = result.missing || [];
            const raw = result.zipB64;
            if (!raw) throw new Error(t("pack.exportFailed"));
            const bin = Uint8Array.from(atob(raw), (c) => c.charCodeAt(0));
            blob = new Blob([bin], { type: "application/zip" });
        } else {
            try {
                missing = JSON.parse(resp.headers.get("X-Pack-Missing") || "[]");
            } catch {
                missing = [];
            }
            blob = new Blob([bytes], { type: "application/zip" });
        }
        if (!blob.size) throw new Error(t("pack.exportFailed"));
        downloadBlob(blob, downloadName);
    } finally {
        progress.close();
    }
    const miss = formatMissing(missing);
    if (miss) {
        await packAlert(editor, `${t("pack.exportDoneMissing")}\n${miss}`, t("toolbar.exportPack"));
    }
}

export async function importDirectorPack(editor) {
    const file = await pickZipFile();
    if (!file) return;
    const progress = openPackProgress(editor);
    let data;
    try {
        await progress.upload(0);
        if (file.size <= CHUNK_SOFT_LIMIT) {
            data = await postPackUpload(file, (kind, ratio) => {
                if (kind === "processing") progress.phase("pack.importProcessing");
                else progress.upload(ratio);
            });
        } else {
            const uploaded = await uploadZipChunked(file, (ratio) => progress.upload(ratio));
            await progress.phase("pack.importProcessing");
            data = await postJson("/minimax/director/import_pack", {
                filename: uploaded.name || uploaded.filename,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
            });
        }
        if (!data?.timeline) throw new Error(t("pack.importFailed"));
        await progress.phase("pack.importApplying");
        editor.applyImportedTimeline(data.timeline, data.widgets || {});
    } finally {
        progress.close();
    }
    const miss = formatMissing(data.missing);
    if (miss) {
        await packAlert(editor, `${t("pack.importDoneMissing")}\n${miss}`, t("toolbar.importPack"));
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
                await packAlert(editor, String(err?.message || err), t("pack.failedTitle"));
            }
        };
    };
    bind('[data-a="pack-export"]', () => exportDirectorPack(editor));
    bind('[data-a="pack-import"]', () => importDirectorPack(editor));
}
