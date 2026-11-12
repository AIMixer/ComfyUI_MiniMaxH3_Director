let uploadCounter = 0;

/** Upload session identity; works on LAN HTTP where randomUUID is unavailable. */
export function createUploadId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    if (globalThis.crypto?.getRandomValues) {
        return Array.from(globalThis.crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, "0")).join("");
    }
    return `${Date.now().toString(36)}-${++uploadCounter}-${Math.random().toString(36).slice(2)}`;
}
