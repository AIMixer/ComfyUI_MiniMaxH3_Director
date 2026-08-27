import { api } from "../../scripts/api.js";
import { fileForComfyUpload } from "./minimax_gen_timeline.js";

const CHUNK_SIZE = 8 * 1024 * 1024;
const COMFY_UPLOAD_SOFT_LIMIT = 95 * 1024 * 1024;

export function isReferenceAudioFile(file) {
    return !!file && (
        String(file.type || "").startsWith("audio/")
        || /\.(wav|mp3|flac|ogg|m4a|aac|wma)$/i.test(file.name || "")
    );
}

export function isReferenceAudioVideoFile(file) {
    return !!file && (
        String(file.type || "").startsWith("video/")
        || /\.(mp4|mov|webm|mkv|avi|m4v|mpg|mpeg|mts|ts)$/i.test(file.name || "")
    );
}

export function isReferenceAudioSourceFile(file) {
    return isReferenceAudioFile(file) || isReferenceAudioVideoFile(file);
}

function normalizePreparedAudio(data, fallbackName = "") {
    const subfolder = String(data?.subfolder || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
    const name = data?.fileName || data?.name || fallbackName || "";
    const relPath = String(data?.relPath || (subfolder ? `${subfolder}/${name}` : name)).replace(/\\/g, "/");
    if (!relPath) throw new Error("Audio preparation returned an empty path.");
    return {
        relPath,
        fileName: data?.fileName || data?.name || fallbackName || relPath,
        subfolder,
        type: data?.type || "input",
        reused: !!data?.reused,
        sourceKind: data?.sourceKind || "",
    };
}

export async function prepareLocalReferenceAudio(file, onProgress) {
    if (!isReferenceAudioSourceFile(file)) {
        throw new Error("Please select an audio or video file.");
    }
    if (isReferenceAudioFile(file) && file.size <= COMFY_UPLOAD_SOFT_LIMIT) {
        const uploadFile = fileForComfyUpload(file);
        const body = new FormData();
        body.append("image", uploadFile, uploadFile.name);
        body.append("type", "input");
        body.append("overwrite", "false");
        const response = await api.fetchApi("/upload/image", { method: "POST", body });
        if (!response.ok) {
            const message = (await response.text()).trim();
            throw new Error(message || `Audio upload failed (${response.status}).`);
        }
        const prepared = normalizePreparedAudio(await response.json(), uploadFile.name);
        prepared.sourceKind = "audio";
        onProgress?.(1, 1, 1);
        return prepared;
    }

    const uploadId = crypto.randomUUID();
    const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
    for (let index = 0; index < totalChunks; index++) {
        const start = index * CHUNK_SIZE;
        const end = Math.min(file.size, start + CHUNK_SIZE);
        const body = new FormData();
        body.append("upload_id", uploadId);
        body.append("chunk_index", String(index));
        body.append("total_chunks", String(totalChunks));
        body.append("filename", file.name || "reference_audio.bin");
        body.append("chunk", file.slice(start, end), `${file.name || "reference_audio"}.part`);
        const response = await api.fetchApi("/minimax/director/prepare_reference_audio_chunk", {
            method: "POST",
            body,
        });
        if (!response.ok) {
            const message = (await response.text()).trim();
            throw new Error(message || `Audio preparation failed (${response.status}).`);
        }
        const data = await response.json();
        onProgress?.((index + 1) / totalChunks, index + 1, totalChunks);
        if (data?.relPath) return normalizePreparedAudio(data, file.name || "");
    }
    throw new Error("Audio preparation did not finish.");
}

export async function extractReferenceAudioFromExistingVideo(item) {
    const response = await api.fetchApi("/minimax/director/extract_reference_audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            videoFile: item?.relPath || item?.videoFile || "",
            fileName: item?.fileName || item?.name || "",
            subfolder: item?.subfolder || "",
            type: item?.type || "input",
        }),
    });
    if (!response.ok) {
        const message = (await response.text()).trim();
        throw new Error(message || `Audio extraction failed (${response.status}).`);
    }
    return normalizePreparedAudio(await response.json(), item?.fileName || item?.name || "");
}

export function hasDuplicateReferenceAudio(items, relPath, exceptIndex = null) {
    const normalized = String(relPath || "").replace(/\\/g, "/").toLowerCase();
    if (!normalized) return false;
    return (items || []).some((item) => {
        const index = Number(item?.index ?? item?.slot);
        if (exceptIndex != null && index === Number(exceptIndex)) return false;
        const path = String(item?.audioFile || item?.fileName || "")
            .replace(/\\/g, "/")
            .toLowerCase();
        return path === normalized;
    });
}
