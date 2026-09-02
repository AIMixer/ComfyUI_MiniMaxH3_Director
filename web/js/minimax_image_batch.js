/** Multi prompt-group UI for t2i / i2i / r2i / t2v / i2v / r2v (prompt batch mode). */

import { api } from "../../scripts/api.js";
import {
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MEGAPIXELS,
    defaultDurationSec,
    defaultFrameCount,
    durationToClampedMiniMaxFrames,
    durationToMiniMaxFrames,
    framesToDurationSec,
    imageBatchVariant,
    isContinuityMasterEnabled,
    isSegmentContinuityFromPrev,
    isVideoBatchTask,
    MAX_GEN_FRAMES,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    maxDurationSec,
    MINIMAX_CANVAS_MULTIPLE,
    minDurationSec,
    minFrameCount,
    newBatchSegment,
    preferredDurationSecFromFrames,
    refAudioLabel,
    refImageLabel,
    refVideoLabel,
    resolveSegmentRefImageSize,
    resolveTaskKey,
    roundDurationSec,
    sumFrameCounts,
    fileForComfyUpload,
    safeUploadFilename,
} from "./minimax_gen_timeline.js";
import { refreshPromptTokenEditors, teardownPromptImageMentions, wirePromptImageMentions } from "./minimax_prompt_mentions.js";
import { t } from "./minimax_i18n.js";
import {
    hasDuplicateReferenceAudio,
    isReferenceAudioSourceFile,
    prepareLocalReferenceAudio,
} from "./minimax_ref_audio.js";

const _players = new WeakMap();
/** r2v picture grid: 9 slots in 3×3; reveal 3 → 6 → 9. */
const R2V_PICTURE_SLOTS = MAX_REFERENCE_IMAGES;
const R2V_PICTURE_STEP = 3;
let _activeR2vMedia = null;

function clamp(n, lo, hi) {
    return Math.max(lo, Math.min(hi, n));
}

function _refHasImage(r) {
    return !!(r?.imageFile || r?.imageB64);
}

function _refHasAudio(r) {
    return !!(r?.audioFile || r?.fileName);
}

function _refHasVideo(r) {
    return !!(r?.videoFile || r?.fileName || r?.previewImageFile || r?.previewImageUrl || r?.linked);
}

/** Highest filled absolute index + 1 (0 when empty). */
export function nextRefIndexAfter(refs, hasFn) {
    let max = -1;
    for (const r of refs || []) {
        if (!hasFn(r)) continue;
        const idx = Number(r.index ?? r.slot);
        if (Number.isFinite(idx) && idx >= 0) max = Math.max(max, idx);
    }
    return max + 1;
}

/** When common params are on, group picture slots start after common's last filled index. */
export function r2vCommonPicOffset(editor) {
    if (!editor?.isR2vCommonEnabled?.()) return 0;
    return Math.min(
        R2V_PICTURE_SLOTS,
        nextRefIndexAfter(editor.timeline?.global?.refs, _refHasImage),
    );
}

export function r2vCommonAudioOffset(editor) {
    if (!editor?.isR2vCommonEnabled?.()) return 0;
    return Math.min(
        MAX_REFERENCE_AUDIOS,
        nextRefIndexAfter(editor.timeline?.global?.refAudios, _refHasAudio),
    );
}

export function r2vCommonVideoOffset(editor) {
    if (!editor?.isR2vCommonEnabled?.()) return 0;
    return Math.min(
        MAX_REFERENCE_VIDEOS,
        nextRefIndexAfter(editor.timeline?.global?.refVideos, _refHasVideo),
    );
}

export function listCommonImageRefs(editor) {
    if (!editor?.isR2vCommonEnabled?.()) return [];
    return [...(editor.timeline?.global?.refs || [])]
        .filter(_refHasImage)
        .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0));
}

export function listCommonVideoRefs(editor) {
    if (!editor?.isR2vCommonEnabled?.()) return [];
    return [...(editor.timeline?.global?.refVideos || [])]
        .filter(_refHasVideo)
        .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0));
}

/**
 * Keep group slots from colliding with common indices.
 * - Legacy (all group pics still 图片1…): shift the block up by picOff.
 * - Partial collisions (common grew into a group index): bump each colliding
 *   slot to the next free absolute index >= offset.
 */
function _rebaseIndexedMedia(list, hasFn, offset, maxSlots) {
    if (offset <= 0 || !Array.isArray(list) || !list.length) {
        return { list, changed: false };
    }
    const items = list.filter(hasFn);
    if (!items.length) return { list, changed: false };
    const idxs = items.map((r) => Number(r.index ?? r.slot));
    const hasLow = idxs.some((i) => Number.isFinite(i) && i < offset);
    if (!hasLow) return { list, changed: false };
    const hasHigh = idxs.some((i) => Number.isFinite(i) && i >= offset);

    // Legacy: entire group still uses 图片1…N → move as a block.
    if (!hasHigh) {
        return {
            changed: true,
            list: list.map((r) => {
                if (!hasFn(r)) return r;
                const i = Number(r.index ?? r.slot);
                if (!Number.isFinite(i)) return r;
                const next = i + offset;
                if (next >= maxSlots) return null;
                return { ...r, index: next, slot: undefined };
            }).filter(Boolean),
        };
    }

    // Mixed: only bump colliding (low) entries into free high slots.
    const used = new Set(
        idxs.filter((i) => Number.isFinite(i) && i >= offset),
    );
    let nextFree = offset;
    const takeFree = () => {
        while (nextFree < maxSlots && used.has(nextFree)) nextFree += 1;
        if (nextFree >= maxSlots) return null;
        const n = nextFree;
        used.add(n);
        nextFree += 1;
        return n;
    };
    let changed = false;
    const out = list.map((r) => {
        if (!hasFn(r)) return r;
        const i = Number(r.index ?? r.slot);
        if (!Number.isFinite(i) || i >= offset) return r;
        const n = takeFree();
        if (n == null) return null;
        changed = true;
        return { ...r, index: n, slot: undefined };
    }).filter(Boolean);
    return { list: out, changed };
}

export function rebaseR2vGroupSlotsForCommon(editor) {
    if (!editor?.isR2vCommonEnabled?.()) return false;
    const picOff = r2vCommonPicOffset(editor);
    const audOff = r2vCommonAudioOffset(editor);
    const vidOff = r2vCommonVideoOffset(editor);
    if (picOff <= 0 && audOff <= 0 && vidOff <= 0) return false;
    let changed = false;
    for (const seg of editor.timeline?.segments || []) {
        if (Array.isArray(seg.refs) && seg.refs.length) {
            const r = _rebaseIndexedMedia(seg.refs, _refHasImage, picOff, R2V_PICTURE_SLOTS);
            if (r.changed) {
                seg.refs = r.list;
                changed = true;
            }
        }
        if (Array.isArray(seg.refAudios) && seg.refAudios.length) {
            const r = _rebaseIndexedMedia(
                seg.refAudios, _refHasAudio, audOff, MAX_REFERENCE_AUDIOS,
            );
            if (r.changed) {
                seg.refAudios = r.list;
                changed = true;
            }
        }
        if (Array.isArray(seg.refVideos) && seg.refVideos.length) {
            const r = _rebaseIndexedMedia(
                seg.refVideos, _refHasVideo, vidOff, MAX_REFERENCE_VIDEOS,
            );
            if (r.changed) {
                seg.refVideos = r.list;
                changed = true;
            }
        }
    }
    return changed;
}

export function formatMediaDuration(sec) {
    if (!Number.isFinite(sec) || sec < 0) return "--:--";
    const total = Math.max(0, Math.round(sec));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
}

function pauseActiveR2vMedia(except = null) {
    if (_activeR2vMedia && _activeR2vMedia !== except) {
        try {
            _activeR2vMedia.pause();
        } catch (_) { /* ignore */ }
        const btn = _activeR2vMedia._r2vPlayBtn;
        if (btn) btn.textContent = "▶";
    }
    if (_activeR2vMedia !== except) _activeR2vMedia = null;
}

export function bindR2vMediaPlayback(mediaEl, playBtn, progressWrap = null) {
    mediaEl.classList.add("bd-r2v-media");
    mediaEl._r2vPlayBtn = playBtn;
    const fill = progressWrap?.querySelector?.(".bd-r2v-progress-fill");
    const syncBtn = () => {
        playBtn.textContent = mediaEl.paused ? "▶" : "⏸";
    };
    const syncProgress = () => {
        if (!progressWrap || !fill) return;
        const dur = mediaEl.duration;
        const pct = Number.isFinite(dur) && dur > 0
            ? Math.min(100, Math.max(0, (mediaEl.currentTime / dur) * 100))
            : 0;
        fill.style.width = `${pct}%`;
        progressWrap.classList.toggle("active", !mediaEl.paused);
        progressWrap.classList.toggle("playing", !mediaEl.paused);
    };
    playBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (mediaEl.paused) {
            pauseActiveR2vMedia(mediaEl);
            mediaEl.play().catch(() => {});
            _activeR2vMedia = mediaEl;
        } else {
            mediaEl.pause();
            if (_activeR2vMedia === mediaEl) _activeR2vMedia = null;
        }
        syncBtn();
        syncProgress();
    });
    mediaEl.addEventListener("play", () => {
        pauseActiveR2vMedia(mediaEl);
        _activeR2vMedia = mediaEl;
        syncBtn();
        syncProgress();
    });
    mediaEl.addEventListener("pause", () => {
        syncBtn();
        syncProgress();
    });
    mediaEl.addEventListener("timeupdate", syncProgress);
    mediaEl.addEventListener("ended", () => {
        if (_activeR2vMedia === mediaEl) _activeR2vMedia = null;
        mediaEl.currentTime = 0;
        syncBtn();
        syncProgress();
        progressWrap?.classList.remove("active", "playing");
        if (fill) fill.style.width = "0%";
    });
    if (progressWrap && fill) {
        progressWrap.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const dur = mediaEl.duration;
            if (!Number.isFinite(dur) || dur <= 0) return;
            const rect = progressWrap.getBoundingClientRect();
            const ratio = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
            mediaEl.currentTime = Math.min(dur, Math.max(0, ratio * dur));
            syncProgress();
        });
    }
}

export function wireMediaDuration(mediaEl, durEl, onReady) {
    const apply = () => {
        if (!Number.isFinite(mediaEl.duration) || mediaEl.duration === Infinity) return;
        durEl.textContent = formatMediaDuration(mediaEl.duration);
        onReady?.(mediaEl.duration);
    };
    mediaEl.addEventListener("loadedmetadata", apply);
    if (mediaEl.readyState >= 1) apply();
}

/**
 * User-facing seconds (1 decimal). durationSec is the source of truth when set;
 * only fall back to frames for legacy rows that never stored durationSec.
 */
function resolveSegmentDurationSec(seg, defFc) {
    if (seg.durationSec != null && Number.isFinite(Number(seg.durationSec))) {
        const { durationSec } = durationToClampedMiniMaxFrames(seg.durationSec, 24);
        return durationSec;
    }
    const fc = parseInt(seg.frameCount ?? seg.length ?? seg._videoFrameCount ?? defFc, 10) || defFc;
    return preferredDurationSecFromFrames(fc, 24);
}

/** Apply seconds to a segment by index (avoids stale closures after normalize). */
function applyBatchSegmentDuration(editor, index, rawSec) {
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const seg = editor.timeline.segments?.[index];
    if (!seg || !isVideoBatchTask(taskKey)) return null;
    const clamped = clamp(
        Number(rawSec) || defaultDurationSec(taskKey),
        minDurationSec(),
        maxDurationSec(),
    );
    const { frames, durationSec } = durationToClampedMiniMaxFrames(clamped, 24);
    seg.durationSec = durationSec;
    seg.frameCount = frames;
    seg.length = frames;
    seg._videoFrameCount = frames;
    // Stale drag preview must not override batch totals.
    if (editor._previewSegments) editor._previewSegments = null;
    normalizeImageBatchSegments(editor);
    return editor.timeline.segments[index] || null;
}

/**
 * Resolve a live segment from a batch card control.
 * Prefer segment id — after splice/reorder, DOM indices no longer match the array.
 */
function liveBatchSegmentFromEl(editor, el, indexAttr) {
    const segs = editor?.timeline?.segments;
    if (!el || !Array.isArray(segs) || !segs.length) return null;
    const id = el.getAttribute("data-batch-seg-id");
    if (id) {
        const index = segs.findIndex((s) => s?.id && s.id === id);
        if (index >= 0) return { seg: segs[index], index };
        return null;
    }
    const index = parseInt(el.getAttribute(indexAttr), 10);
    if (!Number.isFinite(index) || index < 0 || index >= segs.length) return null;
    const cardIdx = parseInt(el.closest?.(".bd-batch-card")?.dataset?.batchIndex, 10);
    if (Number.isFinite(cardIdx) && cardIdx >= 0 && cardIdx < segs.length) {
        return { seg: segs[cardIdx], index: cardIdx };
    }
    const nCards = editor.batchList?.querySelectorAll(".bd-batch-card")?.length ?? 0;
    if (nCards !== segs.length) return null;
    return { seg: segs[index], index };
}

/**
 * Pull prompt textareas into timeline.segments.
 * Must run before normalize / re-render / timeline sync — otherwise edits sit on
 * stale segment objects (or only in the DOM) and get wiped.
 */
export function flushBatchPromptInputs(editor) {
    const list = editor?.batchList;
    if (!list) return;
    const segs = editor?.timeline?.segments;
    if (!Array.isArray(segs) || !segs.length) return;
    list.querySelectorAll("textarea[data-batch-prompt-index]").forEach((el) => {
        el.__bdTokenApi?.sync?.();
        const live = liveBatchSegmentFromEl(editor, el, "data-batch-prompt-index");
        if (!live?.seg) return;
        live.seg.prompt = el.value || "";
        live.seg.negativePrompt = live.seg.negativePrompt ?? "";
    });
}

/** Flush visible 秒数 inputs into segments before a full card re-render. */
function flushBatchDurationInputs(editor) {
    const list = editor?.batchList;
    if (!list) return;
    // Persist prompts first — duration apply/normalize must not drop textarea drafts.
    flushBatchPromptInputs(editor);
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    if (!isVideoBatchTask(taskKey)) return;
    for (const input of list.querySelectorAll("input[data-batch-sec-index]")) {
        const live = liveBatchSegmentFromEl(editor, input, "data-batch-sec-index");
        if (!live?.seg) continue;
        clearTimeout(input._t);
        input._t = null;
        const displayed = parseFloat(input.value);
        if (!Number.isFinite(displayed)) continue;
        const current = Number(live.seg.durationSec);
        // Skip if already in sync (avoid churn while typing the same committed value).
        if (Number.isFinite(current) && roundDurationSec(displayed) === roundDurationSec(current)
            && input !== document.activeElement) {
            continue;
        }
        applyBatchSegmentDuration(editor, live.index, displayed);
    }
}

function formatPreviewFps(value) {
    const fps = Math.round(Number(value) * 100) / 100;
    if (Number.isInteger(fps)) return String(fps);
    return fps.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function stopPlayer(el) {
    const st = _players.get(el);
    if (!st) return;
    st.playing = false;
    if (st.timer) {
        clearInterval(st.timer);
        st.timer = null;
    }
}

function stopAllPlayers(root) {
    root?.querySelectorAll(".bd-batch-vpreview")?.forEach((wrap) => stopPlayer(wrap));
    pauseActiveR2vMedia(null);
    root?.querySelectorAll("video.bd-r2v-media, audio.bd-r2v-media")?.forEach((m) => {
        try { m.pause(); } catch (_) { /* ignore */ }
    });
}

export const IMAGE_BATCH_STYLES = `
.bd-btn.bd-disabled,.bd-btn:disabled{opacity:.38;cursor:not-allowed;pointer-events:none}
.bd-mode button.bd-disabled,.bd-mode button:disabled{opacity:.38;cursor:not-allowed;pointer-events:none}
.bd-batch{width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:8px}
.bd-batch-i2v-notice{display:none;color:#ffb74d;background:#3a2a12;border:1px solid #a67c00;border-radius:6px;padding:8px 10px;font-size:11px;line-height:1.5}
.bd-batch-i2v-notice.visible{display:block}
.bd-batch-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.bd-batch-picker{display:none;flex-wrap:wrap;gap:6px;width:100%;box-sizing:border-box;padding:2px 0 6px;flex-shrink:0}
.bd-batch-picker.visible{display:flex}
.bd-batch-pick{display:flex;flex-direction:column;gap:2px;min-width:92px;max-width:140px;padding:6px 8px;border:1px solid #333;border-radius:8px;background:#161616;cursor:pointer;color:#ccc;user-select:none}
.bd-batch-pick:hover{border-color:#4a7a5a}
.bd-batch-pick.selected{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35);color:#eafff0}
.bd-batch-pick.running{border-color:#4fff8f}
.bd-batch-pick.run-skipped{opacity:.45}
.bd-batch-pick-title{display:flex;align-items:center;gap:4px;font-size:11px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bd-batch-pick-meta{font-size:10px;color:#8aa}
.bd-batch-pick-thumb{width:100%;height:40px;object-fit:cover;border-radius:4px;background:#0d0d0d;margin-top:2px}
.bd-batch-run-select.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f}
.bd-batch-run-all{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#aaa;cursor:pointer;user-select:none}
.bd-batch-run-all.hidden{display:none!important}
.bd-batch-run-all input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
/* Default cap; batch-fill mode overrides via .bd-wrap.bd-batch-fill + JS max-height. */
.bd-batch-list{display:flex;flex-direction:column;gap:8px;width:100%;max-height:640px;overflow-y:auto;padding-right:2px;min-height:0}
.bd-batch-card{background:linear-gradient(165deg,#1a1a1a 0%,#141414 55%,#111 100%);border:1px solid #2c2c2c;border-radius:10px;padding:12px 14px;display:grid;gap:10px;align-items:stretch;box-shadow:inset 0 1px 0 rgba(255,255,255,.03);flex:0 0 auto}
/* t2v: 提示词为主，预览收成右侧窄栏 */
.bd-batch-card.bd-batch-plain{grid-template-columns:minmax(0,1fr) minmax(132px,168px)}
/* i2v / r2i: 源图或参考 | 提示词 | 窄预览 */
.bd-batch-card.bd-batch-source,.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){grid-template-columns:auto minmax(0,1fr) minmax(132px,168px)}
.bd-batch-plain .bd-batch-head,.bd-batch-source .bd-batch-head,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-head{padding-bottom:2px;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:2px}
.bd-batch-plain .bd-batch-head b,.bd-batch-source .bd-batch-head b,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-head b{color:#f0f0f0;font-size:12px;font-weight:650}
.bd-batch-plain .bd-batch-prompts,.bd-batch-source .bd-batch-prompts,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts{background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;gap:6px}
.bd-batch-plain .bd-batch-prompts .bd-label,.bd-batch-source .bd-batch-prompts .bd-label,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts .bd-label{color:#eaeaea;font-size:11px;font-weight:700;letter-spacing:.02em}
.bd-batch-plain .bd-batch-prompts textarea,.bd-batch-source .bd-batch-prompts textarea,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts textarea{background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45}
.bd-batch-plain .bd-batch-preview,.bd-batch-source .bd-batch-preview,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-preview{border-radius:10px;border-color:#262626;background:#0c0c0c}
/* ——— r2v asset stage (polished) ——— */
.bd-batch-card.bd-batch-r2v{display:flex;flex-direction:column;gap:12px;padding:14px 16px;background:linear-gradient(165deg,#1c1c1c 0%,#141414 52%,#111 100%);border:1px solid #2c2c2c;border-radius:12px;box-shadow:inset 0 1px 0 rgba(255,255,255,.035);align-items:stretch}
.bd-batch-card.running{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.25)}
.bd-batch-card.done{border-color:#3a5080}
.bd-batch-card.run-skipped{opacity:.42}
/* selected / run-on must win over .done so timeline ↔ card selection stays visible */
.bd-batch-card.selected,.bd-batch-card.selected.done{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-batch-card.run-on:not(.run-skipped){border-color:#3a7a55}
.bd-batch-card.selected.run-on,.bd-batch-card.selected.run-on.done{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.4)}
.bd-batch-head{grid-column:1/-1;display:flex;align-items:center;justify-content:flex-start;gap:8px;flex-wrap:wrap}
.bd-batch-r2v .bd-batch-head{padding-bottom:2px;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:2px;flex-shrink:0}
.bd-batch-head b{color:#ccc;font-size:11px}
.bd-batch-r2v .bd-batch-head b{color:#f0f0f0;font-size:13px;font-weight:650;letter-spacing:.02em}
.bd-batch-run-check{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f;flex-shrink:0}
.bd-batch-continuity{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#9ab;cursor:pointer;user-select:none;flex-shrink:0}
.bd-batch-continuity input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#6ab0ff;flex-shrink:0}
.bd-batch-continuity span{white-space:nowrap}
.bd-batch-head-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-left:auto}
.bd-batch-fc{display:flex;align-items:center;gap:6px;color:#aaa;font-size:12px}
.bd-batch-r2v .bd-batch-fc{color:#c8c8c8;font-size:12px;gap:8px;background:#0e0e0e;border:1px solid #2a2a2a;border-radius:8px;padding:5px 10px}
.bd-batch-fc input{width:72px;background:#181818;border:1px solid #444;border-radius:5px;color:#eee;padding:5px 8px;font-size:13px}
.bd-batch-r2v .bd-batch-fc input{width:76px;background:#161616;border-color:#3a3a3a;border-radius:6px;padding:5px 8px;font-size:13px}
.bd-batch-refsize{display:flex;align-items:center;gap:6px;color:#c8c8c8;font-size:12px;background:#0e0e0e;border:1px solid #2a2a2a;border-radius:8px;padding:5px 10px;white-space:nowrap}
.bd-batch-refsize select{background:#161616;border:1px solid #3a3a3a;border-radius:6px;color:#eee;padding:5px 6px;font-size:12px;max-width:88px}
.bd-batch-del{background:transparent;border:1px solid #553;color:#f88;border-radius:4px;padding:3px 8px;font-size:10px;cursor:pointer}
.bd-batch-r2v .bd-batch-del{border-radius:8px;padding:5px 10px;font-size:11px;border-color:#4a3030;color:#f0a0a0}
.bd-batch-del:hover{background:#3a1515}
.bd-batch-media{display:flex;flex-direction:column;gap:4px;min-width:88px;max-width:140px}
.bd-batch-media .bd-r2v-pick-existing{align-self:stretch;text-align:center}
/* Left = assets (narrower) · Right = prompt + preview (wider) */
.bd-batch-r2v-body{display:grid;grid-template-columns:minmax(260px,.85fr) minmax(0,1.4fr);gap:12px;width:100%;align-items:stretch;min-height:420px;flex:1 1 auto}
.bd-batch-r2v-assets{display:flex;flex-direction:column;gap:10px;min-width:0;min-height:0}
.bd-batch-r2v-main{display:flex;flex-direction:column;gap:10px;min-width:0;min-height:380px;flex:1 1 auto}
.bd-r2v-section{background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;display:flex;flex-direction:column;gap:8px;min-width:0;box-sizing:border-box}
.bd-r2v-section-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.bd-r2v-section-title{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#eaeaea;min-width:0}
.bd-r2v-section-actions{display:flex;align-items:center;gap:8px;flex-shrink:0}
.bd-r2v-section-count{font-size:11px;color:#7d7d7d;font-variant-numeric:tabular-nums;letter-spacing:.02em}
.bd-r2v-pick-existing{background:transparent;border:1px solid #3a3a3a;color:#c8c8c8;border-radius:6px;padding:2px 8px;font-size:10px;cursor:pointer;line-height:1.4;white-space:nowrap}
.bd-r2v-pick-existing:hover{border-color:#4fff8f;color:#4fff8f}
.bd-r2v-pick-existing:disabled{opacity:.4;cursor:not-allowed;border-color:#333;color:#666}
.bd-r2v-common-inherit{border-style:dashed;border-color:#3a4a5a;background:#0a1218}
.bd-r2v-common-inherit .bd-r2v-section-title{color:#9ab;text-transform:none;letter-spacing:.02em;font-size:11px}
.bd-r2v-common-inherit .bd-batch-ref{cursor:default;border-color:#2a3a4a}
.bd-r2v-common-inherit .bd-batch-ref:hover{border-color:#2a3a4a;background:#080808;transform:none}
.bd-r2v-common-inherit .bd-batch-ref .cap{color:#8af}
.bd-r2v-slot-hint{font-size:10px;color:#6a7a8a;line-height:1.35;margin:0}
.bd-batch-src{width:88px;height:88px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;color:#666;font-size:9px;text-align:center;padding:4px;box-sizing:border-box}
.bd-batch-src.has-img{border-style:solid;border-color:#444}
.bd-batch-src img{width:100%;height:100%;object-fit:contain;background:#000}
.bd-batch-refs{display:grid;grid-template-columns:repeat(3,1fr);gap:3px;width:108px}
.bd-batch-r2v .bd-batch-refs{grid-template-columns:repeat(3,minmax(0,1fr));width:100%;max-width:none;gap:6px}
.bd-batch-r2v .bd-batch-ref.bd-r2v-pic-hidden{display:none!important}
.bd-r2v-pics-toggle{align-self:stretch;margin-top:2px;background:transparent;border:1px dashed #333;border-radius:8px;color:#9a9a9a;font-size:11px;padding:6px 8px;cursor:pointer;transition:border-color .15s,color .15s,background .15s}
.bd-r2v-pics-toggle:hover{border-color:#555;color:#ddd;background:#121212}
.bd-batch-ref{position:relative;aspect-ratio:1;border:1px dashed #555;border-radius:3px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;font-size:8px;color:#666}
.bd-batch-r2v .bd-batch-ref{aspect-ratio:1;min-height:0;border-radius:8px;border:1px dashed #333;background:#080808;color:#555;font-size:10px;transition:border-color .15s,background .15s,transform .12s}
.bd-batch-r2v .bd-batch-ref:hover{border-color:#5a5a5a;background:#101010}
.bd-batch-ref.has-img{border-style:solid}
.bd-batch-r2v .bd-batch-ref.has-img{border-color:#3a3a3a;background:#000}
.bd-batch-ref img{width:100%;height:100%;object-fit:cover}
.bd-batch-r2v .bd-batch-ref img{width:100%;height:100%;object-fit:contain;object-position:center;background:#000}
.bd-batch-r2v .bd-batch-ref .dot{position:absolute;left:6px;top:6px;width:7px;height:7px;border-radius:50%;background:#4fff8f;box-shadow:0 0 0 2px rgba(0,0,0,.5);z-index:2}
.bd-batch-r2v .bd-batch-ref .cap{position:absolute;left:0;right:0;bottom:0;padding:14px 6px 5px;background:linear-gradient(180deg,transparent,rgba(0,0,0,.78));color:#ddd;font-size:10px;font-weight:600;text-align:center;pointer-events:none;z-index:2}
.bd-batch-r2v .bd-batch-ref:not(.has-img) .cap{position:static;padding:0;background:none;color:#666;font-weight:500}
.bd-batch-ref .x{position:absolute;top:0;right:2px;color:#f88;font-size:10px;display:none;line-height:1}
.bd-batch-r2v .bd-batch-ref .x{top:4px;right:4px;width:20px;height:20px;border-radius:6px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.72);color:#ff9a9a;font-size:14px;font-weight:700;z-index:3}
.bd-batch-ref:hover .x{display:block}
.bd-batch-r2v .bd-batch-ref:hover .x,.bd-batch-r2v .bd-batch-ref:focus-within .x{display:flex}
.bd-batch-media-block{display:flex;flex-direction:column;gap:4px;min-width:0}
.bd-batch-media-block .bd-label{color:#888;font-size:10px}
.bd-batch-audios,.bd-batch-videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;width:100%;max-width:420px}
.bd-batch-r2v .bd-batch-videos,.bd-batch-r2v .bd-batch-audios{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));max-width:none;gap:7px;width:100%}
.bd-batch-audio,.bd-batch-video{position:relative;min-height:44px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;cursor:pointer;padding:6px 4px;box-sizing:border-box;font-size:9px;color:#666;text-align:center;line-height:1.25}
.bd-batch-r2v .bd-batch-audio,.bd-batch-r2v .bd-batch-video{min-height:0;height:auto;flex-direction:column;align-items:stretch;justify-content:flex-start;gap:6px;padding:6px;border-radius:8px;border:1px dashed #333;background:#080808;text-align:left;font-size:11px;color:#777;transition:border-color .15s,background .15s}
.bd-batch-r2v .bd-batch-audio:hover,.bd-batch-r2v .bd-batch-video:hover{border-color:#555;background:#101010}
.bd-batch-audio.has-audio,.bd-batch-video.has-video{border-style:solid;border-color:#4a6a4a;color:#cfe;background:#152015}
.bd-batch-r2v .bd-batch-audio.has-audio,.bd-batch-r2v .bd-batch-video.has-video{border-color:#2f4a38;background:#101812;color:#d8ebe0}
.bd-batch-audio:hover,.bd-batch-video:hover{border-color:#7a9cff}
.bd-r2v-thumb{position:relative;width:38px;height:38px;border-radius:7px;background:#1a1a1a;border:1px solid #2e2e2e;flex-shrink:0;display:flex;align-items:center;justify-content:center;color:#666;font-size:14px;overflow:hidden}
.bd-batch-r2v .bd-batch-video .bd-r2v-thumb,.bd-r2v-thumb-video{width:100%;height:auto;aspect-ratio:16/9;border-radius:6px}
.bd-batch-r2v .bd-batch-audio .bd-r2v-thumb{width:100%;height:44px;border-radius:6px}
.bd-r2v-thumb-video video{width:100%;height:100%;object-fit:cover;display:block;background:#000;pointer-events:none}
.bd-r2v-play{position:absolute;inset:0;margin:auto;width:28px;height:28px;border:0;border-radius:50%;background:rgba(0,0,0,.62);color:#fff;font-size:12px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;z-index:2}
.bd-r2v-play:hover{background:rgba(20,20,20,.82);color:#4fff8f}
.bd-batch-r2v .has-audio .bd-r2v-thumb,.bd-batch-r2v .has-video .bd-r2v-thumb{border-color:#3a5a45;color:#8fdfb0;background:#152018}
.bd-r2v-meta{min-width:0;flex:1;display:flex;flex-direction:column;gap:2px}
.bd-batch-r2v .bd-batch-video .bd-r2v-meta,.bd-batch-r2v .bd-batch-audio .bd-r2v-meta{flex-direction:row;align-items:center;justify-content:space-between;gap:4px}
.bd-r2v-meta .tag{color:#cfcfcf;font-size:11px;font-weight:650}
.bd-r2v-dur{flex-shrink:0;min-width:2.6em;text-align:right;font-size:11px;color:#8a9;font-variant-numeric:tabular-nums}
.bd-r2v-paired-audio{flex-shrink:0;color:#7dbaff;font-size:12px;line-height:1;padding:0 2px}
.bd-batch-r2v .bd-batch-audio .name,.bd-batch-r2v .bd-batch-video .name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#8aa;font-size:10px;padding:0}
.bd-batch-r2v .bd-batch-video.has-video .name,.bd-batch-r2v .bd-batch-audio.has-audio .name{display:none}
.bd-batch-r2v .bd-batch-video:not(.has-video) .name,.bd-batch-r2v .bd-batch-audio:not(.has-audio) .name{display:block;color:#666}
.bd-batch-r2v .bd-batch-audio audio.bd-r2v-media{position:absolute;width:0;height:0;opacity:0;pointer-events:none}
.bd-r2v-progress{display:none;width:100%;height:3px;border-radius:99px;background:#222;overflow:hidden;cursor:pointer}
.bd-r2v-progress.active{display:block}
.bd-r2v-progress-fill{height:100%;width:0;background:linear-gradient(90deg,#2a6b4a,#4fff8f);border-radius:99px;transition:width .08s linear}
.bd-r2v-progress.playing .bd-r2v-progress-fill{transition:none}
.bd-batch-audio .name,.bd-batch-video .name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#9ad;font-size:9px;padding:0 2px}
.bd-batch-audio .x,.bd-batch-video .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;display:none;line-height:1}
.bd-batch-r2v .bd-batch-audio .x,.bd-batch-r2v .bd-batch-video .x{position:absolute;top:8px;right:8px;width:20px;height:20px;border-radius:6px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.72);color:#ff9a9a;font-size:14px;font-weight:700;z-index:3}
.bd-batch-audio:hover .x,.bd-batch-video:hover .x{display:block}
.bd-batch-r2v .bd-batch-audio:hover .x,.bd-batch-r2v .bd-batch-video:hover .x{display:flex}
.bd-batch-prompts{display:flex;flex-direction:column;gap:4px;min-width:0}
.bd-batch-prompts .bd-label{color:#888;font-size:10px}
.bd-batch-r2v .bd-batch-prompts{background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;gap:6px;flex:1 1 auto;min-height:380px;display:flex;flex-direction:column}
.bd-batch-r2v .bd-batch-prompts .bd-label{color:#eaeaea;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.bd-batch-prompts textarea,.bd-batch-prompts .bd-token-wrap{width:100%;min-height:88px;box-sizing:border-box}
.bd-batch-prompts textarea{background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:6px;resize:vertical;font-size:11px;font-family:inherit;line-height:1.35}
.bd-batch-plain .bd-batch-prompts textarea,.bd-batch-source .bd-batch-prompts textarea,
.bd-batch-plain .bd-batch-prompts .bd-token-wrap,.bd-batch-source .bd-batch-prompts .bd-token-wrap{min-height:120px;height:100%;resize:vertical;overflow:auto}
.bd-batch-r2v .bd-batch-prompts textarea,.bd-batch-r2v .bd-batch-prompts .bd-token-wrap{min-height:360px;height:100%;flex:1;resize:vertical;overflow:auto}
.bd-batch-r2v .bd-batch-prompts textarea{background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45}
.bd-batch-preview{background:#0d0d0d;border:1px solid #333;border-radius:4px;min-height:100px;display:flex;flex-direction:column;align-items:stretch;justify-content:center;overflow:hidden;color:#555;font-size:10px;text-align:center;padding:4px;box-sizing:border-box}
.bd-batch-plain .bd-batch-preview,.bd-batch-source .bd-batch-preview,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-preview{width:100%;max-width:220px;min-height:160px;justify-self:end}
.bd-batch-r2v .bd-batch-preview{min-height:220px;flex:0 0 auto;height:auto;border-radius:10px;border-color:#262626;background:#0c0c0c;padding:8px;font-size:11px;color:#666}
.bd-batch-preview img{width:100%;max-width:100%;max-height:200px;object-fit:contain;display:block;margin:0 auto}
.bd-batch-plain .bd-batch-preview img,.bd-batch-source .bd-batch-preview img{max-height:180px}
.bd-batch-r2v .bd-batch-preview img{width:100%;max-height:280px}
.bd-batch-vpreview{width:100%;height:100%;display:flex;flex-direction:column;align-items:stretch;gap:4px;min-height:0}
.bd-batch-vpreview canvas{width:100%;flex:1 1 auto;min-height:96px;max-height:200px;background:#000;border-radius:3px;display:block;object-fit:contain}
.bd-batch-r2v .bd-batch-vpreview canvas{border-radius:8px;max-height:280px;min-height:160px}
.bd-batch-plain .bd-batch-vpreview canvas,.bd-batch-source .bd-batch-vpreview canvas{max-height:180px;min-height:96px}
.bd-batch-vpreview-ctrl{display:flex;align-items:center;justify-content:center;gap:6px;flex-shrink:0}
.bd-batch-vpreview-ctrl button{font-size:10px;padding:2px 8px}
.bd-batch-vpreview-meta{color:#666;font-size:9px;text-align:center;flex-shrink:0}
.bd-batch-live-preview{position:relative;width:100%;min-height:160px;flex:1 1 auto;display:flex;align-items:center;justify-content:center;box-sizing:border-box}
.bd-batch-r2v .bd-batch-live-preview{min-height:200px}
.bd-batch-live-preview img{width:100%;height:auto;max-width:100%;max-height:280px;object-fit:contain;display:block;border-radius:6px}
.bd-batch-r2v .bd-batch-live-preview img{max-height:280px}
.bd-batch-live-badge{position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:999px;background:rgba(0,0,0,.72);color:#cfcfcf;font-size:10px;pointer-events:none}
@media(max-width:860px){
.bd-batch-r2v-body,.bd-batch-r2v-foot{grid-template-columns:1fr}
.bd-batch-r2v .bd-batch-preview{min-height:160px}
.bd-batch-card.bd-batch-plain{grid-template-columns:minmax(0,1fr) minmax(140px,180px)}
.bd-batch-card.bd-batch-source,.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){grid-template-columns:auto minmax(0,1fr) minmax(140px,180px)}
}
@media(max-width:720px){
.bd-batch-card,.bd-batch-card.bd-batch-plain,.bd-batch-card.bd-batch-source,.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){grid-template-columns:1fr}
.bd-batch-plain .bd-batch-preview,.bd-batch-source .bd-batch-preview,.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-preview{max-width:none;justify-self:stretch;min-height:140px}
.bd-batch-r2v .bd-batch-refs{grid-template-columns:repeat(3,minmax(0,1fr))}
}
`;

const BATCH_CHUNK_SIZE = 8 * 1024 * 1024;
const BATCH_UPLOAD_SOFT_LIMIT = 95 * 1024 * 1024;

async function uploadImage(file) {
    const uploadFile = fileForComfyUpload(file);
    const body = new FormData();
    body.append("image", uploadFile, uploadFile.name);
    body.append("type", "input");
    body.append("overwrite", "false");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) throw new Error(await resp.text() || `Upload failed (${resp.status})`);
    return resp.json();
}

async function uploadChunked(file) {
    const filename = safeUploadFilename(file?.name, file?.type);
    const uploadId = crypto.randomUUID();
    const totalChunks = Math.ceil(file.size / BATCH_CHUNK_SIZE);
    for (let i = 0; i < totalChunks; i++) {
        const start = i * BATCH_CHUNK_SIZE;
        const end = Math.min(start + BATCH_CHUNK_SIZE, file.size);
        const body = new FormData();
        body.append("upload_id", uploadId);
        body.append("chunk_index", String(i));
        body.append("total_chunks", String(totalChunks));
        body.append("filename", filename);
        body.append("chunk", file.slice(start, end), `${filename}.part`);
        const resp = await api.fetchApi("/minimax/director/upload_chunk", { method: "POST", body });
        if (!resp.ok) throw new Error(await resp.text() || t("upload.chunkFailed", { status: resp.status }));
        const data = await resp.json();
        if (data.name) return data;
    }
    throw new Error(t("upload.chunkIncomplete"));
}

async function uploadMedia(file) {
    if (file.size <= BATCH_UPLOAD_SOFT_LIMIT) {
        try {
            return await uploadImage(file);
        } catch (err) {
            const msg = String(err?.message || err || "");
            if (!/too large|size|413/i.test(msg)) throw err;
        }
    }
    return uploadChunked(file);
}

function relPath(upload) {
    const name = upload.name || upload.filename;
    const sub = (upload.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
    return sub ? `${sub}/${name}` : name;
}

function viewUrl(imageFile) {
    const norm = String(imageFile || "").replace(/\\/g, "/");
    const slash = norm.lastIndexOf("/");
    const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
    const subfolder = slash >= 0 ? norm.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, type: "input" });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

export function mountImageBatchPanel(root) {
    const panel = document.createElement("div");
    panel.className = "bd-batch hidden";
    panel.dataset.r = "batch-panel";
    panel.innerHTML = `
        <div class="bd-batch-toolbar">
            <button type="button" class="bd-btn bd-btn-primary" data-a="batch-add" data-i18n="batch.addPromptGroup">+ 添加提示词组</button>
            <button type="button" class="bd-btn bd-batch-run-select hidden" data-a="batch-run-select" data-i18n="toolbar.runSelect" data-i18n-title="tooltip.batchRunSelect">选择运行</button>
            <label class="bd-batch-run-all hidden" data-r="batch-run-all-wrap" data-i18n-title="tooltip.runSelectAll">
                <input type="checkbox" data-r="batch-run-all-cb">
                <span data-i18n="toolbar.selectAll">全选</span>
            </label>
            <button type="button" class="bd-btn" data-a="batch-detail-mode" data-i18n="toolbar.batchDetailSolo" data-i18n-title="tooltip.batchDetailSolo">单显模式</button>
            <span class="bd-meta" data-r="batch-hint" data-i18n="batch.hint.defaultImage">每组生成 1 张图片</span>
        </div>
        <div class="bd-batch-i2v-notice" data-r="batch-i2v-notice"></div>
        <div class="bd-batch-picker" data-r="batch-picker"></div>
        <div class="bd-batch-list" data-r="batch-list"></div>`;
    root.appendChild(panel);
    return {
        panel,
        list: panel.querySelector('[data-r="batch-list"]'),
        picker: panel.querySelector('[data-r="batch-picker"]'),
        hint: panel.querySelector('[data-r="batch-hint"]'),
        i2vNotice: panel.querySelector('[data-r="batch-i2v-notice"]'),
        addBtn: panel.querySelector('[data-a="batch-add"]'),
        runSelectBtn: panel.querySelector('[data-a="batch-run-select"]'),
        runSelectAllWrap: panel.querySelector('[data-r="batch-run-all-wrap"]'),
        runSelectAllCb: panel.querySelector('[data-r="batch-run-all-cb"]'),
        detailModeBtn: panel.querySelector('[data-a="batch-detail-mode"]'),
    };
}

export function wireBatchRunSelectControls(editor, batchUi) {
    editor.batchRunSelectBtn = batchUi.runSelectBtn;
    editor.batchRunSelectAllWrap = batchUi.runSelectAllWrap;
    editor.batchRunSelectAllCb = batchUi.runSelectAllCb;
    editor.batchDetailModeBtn = batchUi.detailModeBtn;
    editor.batchPicker = batchUi.picker;
    batchUi.runSelectBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        editor.toggleRunSelectMode?.();
    });
    batchUi.detailModeBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleBatchDetailMode(editor);
    });
    batchUi.runSelectAllCb?.addEventListener("change", (e) => {
        e.stopPropagation();
        if (!editor.isRunSelectEnabled?.()) return;
        editor.setRunSelectionAll?.(batchUi.runSelectAllCb.checked);
    });
}

function cloneRefs(refs) {
    if (!Array.isArray(refs) || !refs.length) return [];
    try {
        return JSON.parse(JSON.stringify(refs));
    } catch {
        return refs.map((r) => ({ ...r }));
    }
}

/** Copy global.refs into batch segments that have no refs (r2i only).
 *  r2v keeps refs on timeline.global as shared「公共参数」merged at plan time. */
export function migrateGlobalRefsIntoBatchSegments(editor, taskKey) {
    const key = resolveTaskKey(taskKey || editor.getTaskKey?.() || "");
    if (key !== "r2i") return false;
    const globalRefs = editor.timeline?.global?.refs;
    if (!Array.isArray(globalRefs) || !globalRefs.length) return false;
    let moved = false;
    for (const seg of editor.timeline.segments || []) {
        if ((seg.refs || []).length) continue;
        seg.refs = cloneRefs(globalRefs);
        moved = true;
    }
    return moved;
}

function mergeMediaByIndex(commonList, localList) {
    const byIdx = new Map();
    for (const item of commonList || []) {
        const idx = Number(item?.index ?? item?.slot);
        if (Number.isFinite(idx)) byIdx.set(idx, item);
    }
    for (const item of localList || []) {
        const idx = Number(item?.index ?? item?.slot);
        if (Number.isFinite(idx)) byIdx.set(idx, item);
    }
    return [...byIdx.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
}

export function ensureImageBatchTimeline(editor) {
    editor.timeline.editMode = "segment";
    editor.timeline.output = editor.timeline.output || {};
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    editor.timeline.output.mode = "fixed";
    if (!editor.timeline.output.aspectRatio) editor.timeline.output.aspectRatio = DEFAULT_ASPECT_RATIO;
    if (editor.timeline.output.megapixels == null) editor.timeline.output.megapixels = DEFAULT_MEGAPIXELS;
    if (editor.timeline.output.multiple == null) editor.timeline.output.multiple = MINIMAX_CANVAS_MULTIPLE;
    if (!isVideoBatchTask(taskKey)) {
        editor.timeline.output.exportMode = "all";
    }
    const defFc = defaultFrameCount(taskKey);
    if (taskKey === "i2v") {
        editor.timeline.video = {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
        };
        editor.timeline.videoClips = [];
    }
    if (!editor.timeline.segments?.length) {
        editor.timeline.segments = [newBatchSegment({ durationSec: defaultDurationSec(taskKey) })];
    }
    // r2i/r2v need per-group refs. If the user came from rv2v (global refs) or left
    // refs only on global, copy them into empty batch groups so generation actually
    // receives reference_image_* — otherwise it silently behaves like t2v/t2i.
    migrateGlobalRefsIntoBatchSegments(editor, taskKey);
    for (const seg of editor.timeline.segments) {
        if (isVideoBatchTask(taskKey)) {
            const { frames, durationSec } = durationToClampedMiniMaxFrames(
                resolveSegmentDurationSec(seg, defFc),
                24,
            );
            seg.durationSec = durationSec;
            seg.frameCount = frames;
            seg.length = frames;
            seg._videoFrameCount = frames;
        } else {
            const prevFc = parseInt(seg.frameCount ?? seg.length, 10) || 0;
            if (prevFc > 1) seg._videoFrameCount = prevFc;
            seg.frameCount = 1;
            seg.length = 1;
        }
        seg.negativePrompt = seg.negativePrompt ?? "";
        seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
        // Do NOT copy r2v refs into i2v/t2v here — each task keeps its own workspace.
        // Backend ignores refs on i2v/t2v; r2v snapshots restore them on switch-back.
        seg.refs = seg.refs || [];
        seg.refAudios = seg.refAudios || seg.ref_audios || [];
        seg.refVideos = seg.refVideos || seg.ref_videos || [];
        seg.previewB64 = seg.previewB64 || "";
        seg.previewFrames = seg.previewFrames || [];
        seg.previewFps = seg.previewFps || parseFloat(editor.frameRateWidget?.value || 24);
        if (!seg.id) seg.id = newBatchSegment().id;
    }
    normalizeImageBatchSegments(editor);
}

export function normalizeImageBatchSegments(editor) {
    // Keep textarea drafts on the live segment objects before touching them.
    flushBatchPromptInputs(editor);
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const isVideo = isVideoBatchTask(taskKey);
    const defFc = defaultFrameCount(taskKey);
    const defSec = defaultDurationSec(taskKey);
    let start = 0;
    const segs = editor.timeline.segments || [];
    // Mutate in place so card closures (prompt oninput) stay attached to the
    // same objects. Replacing with `{ ...seg }` orphans DOM writes and can
    // wipe group 5/6 prompts on the next sync/re-render.
    if (!segs.length) {
        editor.timeline.segments = [newBatchSegment({ durationSec: defSec })];
    }
    for (const seg of editor.timeline.segments) {
        let fc = 1;
        let durationSec;
        if (isVideo) {
            const resolved = durationToClampedMiniMaxFrames(
                clamp(resolveSegmentDurationSec(seg, defFc) || defSec, minDurationSec(), maxDurationSec()),
                24,
            );
            fc = resolved.frames;
            durationSec = resolved.durationSec;
            seg.durationSec = durationSec;
            seg._videoFrameCount = fc;
        }
        seg.start = start;
        seg.length = fc;
        seg.frameCount = fc;
        seg.negativePrompt = seg.negativePrompt ?? "";
        seg.genImage = seg.genImage || { imageFile: "" };
        seg.refs = seg.refs || [];
        seg.refAudios = seg.refAudios || [];
        seg.refVideos = seg.refVideos || [];
        seg.previewB64 = seg.previewB64 || "";
        seg.previewFrames = seg.previewFrames || [];
        seg.previewFps = seg.previewFps || parseFloat(editor.frameRateWidget?.value || 24);
        if (taskKey === "r2v") {
            seg.refImageSize = resolveSegmentRefImageSize(seg, editor.timeline?.output);
        }
        if (!seg.id) seg.id = newBatchSegment().id;
        start += fc;
    }
    const out = editor.timeline.segments;
    editor.timeline.totalFrames = start || out[0]?.frameCount || defFc;
}

export function addImageBatchGroup(editor) {
    if (editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.()) return;
    const taskKey = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    editor.timeline.segments.push(newBatchSegment({
        durationSec: defaultDurationSec(taskKey),
        negativePrompt: "",
    }));
    normalizeImageBatchSegments(editor);
    editor.selectedIndex = Math.max(0, editor.timeline.segments.length - 1);
    editor.renderImageBatchGroups();
    editor.commit();
    editor.updateVideoNameLabel?.();
    editor.updateDomWidgetHeight?.();
}

export function deleteImageBatchGroup(editor, index) {
    if (editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.()) return;
    if (editor.timeline.segments.length <= 1) return;
    // Persist drafts while DOM still matches the current array, then splice.
    flushBatchPromptInputs(editor);
    flushBatchDurationInputs(editor);
    editor.timeline.segments.splice(index, 1);
    normalizeImageBatchSegments(editor);
    editor.selectedIndex = clamp(
        editor.selectedIndex > index ? editor.selectedIndex - 1 : editor.selectedIndex,
        0,
        editor.timeline.segments.length - 1,
    );
    editor.renderImageBatchGroups();
    editor.commit();
    editor.updateVideoNameLabel?.();
    editor.updateDomWidgetHeight?.();
}

function pickFile(accept, onFile) {
    // Keep input in DOM until change/cancel — otherwise some Chromium builds
    // drop the dialog result when the element is GC'd.
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0;pointer-events:none";
    const cleanup = () => {
        input.remove();
    };
    input.onchange = () => {
        const file = input.files?.[0];
        cleanup();
        if (file) onFile(file);
    };
    input.addEventListener("cancel", cleanup);
    document.body.appendChild(input);
    input.click();
}

function isBatchImageFile(file) {
    return !!file && (
        String(file.type || "").startsWith("image/")
        || /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i.test(file.name || "")
    );
}

function isBatchVideoFile(file) {
    return !!file && (
        String(file.type || "").startsWith("video/")
        || /\.(mp4|mov|webm|mkv|avi|m4v|mpg|mpeg|mts|ts)$/i.test(file.name || "")
    );
}

function bindOsFileDrop(el, onFiles) {
    el.addEventListener("dragover", (e) => {
        const types = [...(e.dataTransfer?.types || [])];
        if (types.includes("application/x-minimax-ref-slot")) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = "copy";
    });
    el.addEventListener("drop", (e) => {
        const types = [...(e.dataTransfer?.types || [])];
        if (types.includes("application/x-minimax-ref-slot")) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        const files = [...(e.dataTransfer?.files || [])];
        if (!files.length) return;
        void onFiles(files, e);
    });
}

function applySegSourceImage(editor, index, imageFile, width = 0, height = 0) {
    const segId = editor.timeline.segments[index]?.id;
    const seg = (editor.timeline.segments || []).find((s) => s.id === segId)
        || editor.timeline.segments[index];
    if (!seg) return;
    seg.genImage = { imageFile, width: width || 0, height: height || 0 };
    seg.imageFile = imageFile;
    editor.renderImageBatchGroups();
    editor.updateOutputPreview?.();
    editor.commit(false, { syncTimeline: true });
    editor.scheduleRender?.();
    editor.scheduleTimelineSync?.();
}

async function assignSegSourceFromFile(editor, index, file) {
    const segId = editor.timeline.segments[index]?.id;
    try {
        if (!isBatchImageFile(file)) throw new Error("Not an image file");
        const uploaded = await uploadImage(file);
        const imageFile = relPath(uploaded);
        if (!imageFile) throw new Error("Upload returned empty filename");
        applySegSourceImage(editor, index, imageFile, 0, 0);
        try {
            const dims = await readImageDimensions(file);
            const live = (editor.timeline.segments || []).find((s) => s.id === segId) || editor.timeline.segments[index];
            if (live?.genImage?.imageFile === imageFile) {
                live.genImage = { imageFile, width: dims.width, height: dims.height };
                editor.updateOutputPreview?.();
                editor.scheduleTimelineSync?.();
            }
        } catch (dimErr) {
            console.warn("[MiniMax H3Director] batch source dims skipped:", dimErr);
        }
    } catch (err) {
        console.error("[MiniMax H3Director] batch source upload failed:", err);
        alert(t("upload.alertFailed", { err: err?.message || err }));
    }
}

async function uploadSegSource(editor, index) {
    pickFile("image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif", (file) => {
        void assignSegSourceFromFile(editor, index, file);
    });
}

async function pickExistingSegSource(editor, index) {
    try {
        const picked = await editor.chooseImageInput({
            title: t("mediaPicker.pickSourceImage"),
            currentValue: editor.timeline.segments[index]?.genImage?.imageFile || "",
        });
        if (!picked?.imageFile) return;
        applySegSourceImage(editor, index, picked.imageFile, picked.width, picked.height);
    } catch (err) {
        console.error("[MiniMax H3Director] batch source pick failed:", err);
        alert(t("upload.alertFailed", { err: err?.message || err }));
    }
}

function readImageDimensions(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        const done = (fn, arg) => {
            clearTimeout(timer);
            URL.revokeObjectURL(url);
            fn(arg);
        };
        const timer = setTimeout(() => done(reject, new Error("Image dimension timeout")), 8000);
        img.onload = () => done(resolve, { width: img.naturalWidth, height: img.naturalHeight });
        img.onerror = () => done(reject, new Error("Failed to read image dimensions"));
        img.src = url;
    });
}

async function assignSegRefFromFile(editor, index, slot, file) {
    if (!file?.type?.startsWith("image/")) return;
    try {
        const uploaded = await uploadImage(file);
        const seg = editor.timeline.segments[index];
        if (!seg) return;
        seg.refs = (seg.refs || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        seg.refs.push({ index: slot, imageFile: relPath(uploaded), imageB64: "" });
        editor.renderImageBatchGroups();
        editor.commit();
    } catch (err) {
        console.error("[MiniMax H3Director] batch ref upload failed:", err);
    }
}

async function uploadSegRef(editor, index, slot) {
    pickFile("image/*", (file) => assignSegRefFromFile(editor, index, slot, file));
}

async function assignSegRefFromPicked(editor, index, slot, picked) {
    if (!picked?.imageFile) return;
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.refs = (seg.refs || []).filter((r) => Number(r.index ?? r.slot) !== slot);
    seg.refs.push({ index: slot, imageFile: picked.imageFile, imageB64: "" });
    editor.renderImageBatchGroups();
    editor.commit();
}

async function pickExistingSegRef(editor, index, offset, slots) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    const slot = nextEmptyGroupSlot(seg.refs, offset, slots, (r) => r?.imageFile || r?.imageB64);
    if (slot < 0) {
        alert(t("mediaPicker.slotsFull"));
        return;
    }
    try {
        const picked = await editor.chooseImageInput({
            title: t("mediaPicker.pickReferenceImage"),
        });
        await assignSegRefFromPicked(editor, index, slot, picked);
    } catch (err) {
        console.error("[MiniMax H3Director] batch ref pick failed:", err);
        alert(t("upload.alertFailed", { err: err?.message || err }));
    }
}

function moveBatchRefSlot(editor, segIndex, fromSlot, toSlot) {
    if (fromSlot === toSlot) return;
    const seg = editor.timeline.segments[segIndex];
    if (!seg) return;
    const refs = [...(seg.refs || [])];
    const fromRef = refs.find((r) => Number(r.index ?? r.slot) === fromSlot);
    if (!fromRef) return;
    const toRef = refs.find((r) => Number(r.index ?? r.slot) === toSlot);
    seg.refs = refs.filter((r) => {
        const idx = Number(r.index ?? r.slot);
        return idx !== fromSlot && idx !== toSlot;
    });
    seg.refs.push({ ...fromRef, index: toSlot, slot: undefined });
    if (toRef) {
        seg.refs.push({ ...toRef, index: fromSlot, slot: undefined });
    }
    editor.renderImageBatchGroups();
    editor.commit();
}

function bindBatchRefDrop(slot, editor, index, slotIndex) {
    const hasImg = slot.classList.contains("has-img");
    slot.draggable = hasImg;
    slot.addEventListener("dragstart", (e) => {
        if (!hasImg) {
            e.preventDefault();
            return;
        }
        editor._batchRefDragMoved = false;
        const payload = JSON.stringify({ segIndex: index, from: slotIndex });
        e.dataTransfer.setData("application/x-minimax-ref-slot", payload);
        e.dataTransfer.setData("text/plain", payload);
        e.dataTransfer.effectAllowed = "move";
    });
    slot.addEventListener("dragend", () => {
        setTimeout(() => { editor._batchRefDragMoved = false; }, 0);
    });
    slot.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const types = [...(e.dataTransfer?.types || [])];
        e.dataTransfer.dropEffect = types.includes("application/x-minimax-ref-slot")
            ? "move"
            : "copy";
    });
    slot.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const raw = e.dataTransfer.getData("application/x-minimax-ref-slot")
            || e.dataTransfer.getData("text/plain");
        if (raw) {
            try {
                const data = JSON.parse(raw);
                if (Number(data.segIndex) !== index) return;
                editor._batchRefDragMoved = true;
                moveBatchRefSlot(editor, index, Number(data.from), slotIndex);
                return;
            } catch (_) { /* fall through */ }
        }
        const f = e.dataTransfer.files?.[0];
        if (f) assignSegRefFromFile(editor, index, slotIndex, f);
    });
}

function removeSegRef(editor, index, slot) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.refs = (seg.refs || []).filter((r) => Number(r.index ?? r.slot) !== slot);
    editor.renderImageBatchGroups();
    editor.commit();
}

async function assignSegAudioFromFile(editor, index, slot, file) {
    if (!isReferenceAudioSourceFile(file)) return false;
    try {
        const prepared = await prepareLocalReferenceAudio(file);
        const seg = editor.timeline.segments[index];
        if (!seg) return false;
        if (hasDuplicateReferenceAudio(seg.refAudios, prepared.relPath, slot)) {
            alert(t("ref.audioDuplicate"));
            return false;
        }
        seg.refAudios = (seg.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        seg.refAudios.push({
            index: slot,
            audioFile: prepared.relPath,
            fileName: prepared.fileName || file.name,
            type: prepared.type || "input",
            subfolder: prepared.subfolder || "",
        });
        editor.renderImageBatchGroups();
        editor.commit();
        return true;
    } catch (err) {
        console.error("[MiniMax H3Director] batch audio upload failed:", err);
        alert(t("upload.refAudioFailed", { err: err?.message || err }));
        return false;
    }
}

async function uploadSegAudio(editor, index, slot) {
    pickFile("audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts", (file) => {
        void assignSegAudioFromFile(editor, index, slot, file);
    });
}

function removeSegAudio(editor, index, slot) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.refAudios = (seg.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== slot);
    editor.renderImageBatchGroups();
    editor.commit();
}

async function assignSegVideoFromFile(editor, index, slot, file) {
    if (!isBatchVideoFile(file)) return false;
    try {
        const uploaded = await uploadMedia(file);
        const seg = editor.timeline.segments[index];
        if (!seg) return false;
        const videoFile = relPath(uploaded);
        seg.refVideos = (seg.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        seg.refVideos.push({
            index: slot,
            videoFile,
            fileName: uploaded?.name || file.name,
            type: "input",
            subfolder: uploaded?.subfolder || "",
        });
        editor.renderImageBatchGroups();
        editor.commit();
        return true;
    } catch (err) {
        console.error("[MiniMax H3Director] batch video upload failed:", err);
        alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        return false;
    }
}

async function uploadSegVideo(editor, index, slot) {
    pickFile("video/*,.mp4,.mov,.webm,.mkv", (file) => {
        void assignSegVideoFromFile(editor, index, slot, file);
    });
}

async function pickExistingSegVideo(editor, index, offset, slots) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    const slot = nextEmptyGroupSlot(seg.refVideos, offset, slots, (r) => r?.videoFile || r?.fileName);
    if (slot < 0) {
        alert(t("mediaPicker.slotsFull"));
        return;
    }
    try {
        const picked = await editor.chooseVideoInput({
            title: t("mediaPicker.pickReferenceVideo"),
        });
        if (!picked?.relPath) return;
        const live = editor.timeline.segments[index];
        if (!live) return;
        live.refVideos = (live.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        live.refVideos.push({
            index: slot,
            videoFile: picked.relPath,
            fileName: picked.fileName || picked.relPath,
            type: picked.type || "input",
            subfolder: picked.subfolder || "",
        });
        editor.renderImageBatchGroups();
        editor.commit();
    } catch (err) {
        console.error("[MiniMax H3Director] batch video pick failed:", err);
        alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
    }
}

async function pickExistingSegAudio(editor, index, offset, slots) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    const slot = nextEmptyGroupSlot(seg.refAudios, offset, slots, (r) => r?.audioFile || r?.fileName);
    if (slot < 0) {
        alert(t("mediaPicker.slotsFull"));
        return;
    }
    try {
        const picked = await editor.chooseAudioInput({
            title: t("mediaPicker.pickReferenceAudio"),
        });
        if (!picked?.relPath) return;
        const live = editor.timeline.segments[index];
        if (!live) return;
        if (hasDuplicateReferenceAudio(live.refAudios, picked.relPath, slot)) {
            alert(t("ref.audioDuplicate"));
            return;
        }
        live.refAudios = (live.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== slot);
        live.refAudios.push({
            index: slot,
            audioFile: picked.relPath,
            fileName: picked.fileName || picked.relPath,
            type: picked.type || "input",
            subfolder: picked.subfolder || "",
        });
        editor.renderImageBatchGroups();
        editor.commit();
    } catch (err) {
        console.error("[MiniMax H3Director] batch audio pick failed:", err);
        alert(t("upload.refAudioFailed", { err: err?.message || err }));
    }
}

function removeSegVideo(editor, index, slot) {
    const seg = editor.timeline.segments[index];
    if (!seg) return;
    seg.refVideos = (seg.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== slot);
    editor.renderImageBatchGroups();
    editor.commit();
}

function fileBaseName(path) {
    const s = String(path || "").replace(/\\/g, "/");
    return s.split("/").pop() || s;
}

function countFilledRefs(seg, { picOffset = 0, audOffset = 0, vidOffset = 0 } = {}) {
    let imgs = 0;
    let videos = 0;
    let audios = 0;
    for (const r of seg.refs || []) {
        const idx = Number(r.index ?? r.slot);
        if (
            r?.imageFile
            && Number.isFinite(idx)
            && idx >= picOffset
            && idx < R2V_PICTURE_SLOTS
        ) {
            imgs += 1;
        }
    }
    for (const r of seg.refVideos || []) {
        const idx = Number(r.index ?? r.slot);
        if (
            _refHasVideo(r)
            && Number.isFinite(idx)
            && idx >= vidOffset
            && idx < MAX_REFERENCE_VIDEOS
        ) {
            videos += 1;
        }
    }
    for (const r of seg.refAudios || []) {
        const idx = Number(r.index ?? r.slot);
        if (
            (r?.audioFile || r?.fileName)
            && Number.isFinite(idx)
            && idx >= audOffset
            && idx < MAX_REFERENCE_AUDIOS
        ) {
            audios += 1;
        }
    }
    return { imgs, videos, audios };
}

function nextEmptyGroupSlot(items, offset, slots, hasFn) {
    for (let local = 0; local < slots; local++) {
        const abs = offset + local;
        const hit = (items || []).find((r) => Number(r.index ?? r.slot) === abs);
        if (!hasFn(hit)) return abs;
    }
    return -1;
}

async function dropFilesIntoGroupSlots(editor, index, files, e, {
    isFile,
    slotSelector,
    offset,
    slots,
    itemsKey,
    hasFn,
    assignFile,
}) {
    const matching = files.filter(isFile);
    if (!matching.length) return;
    const hit = e.target.closest?.(slotSelector);
    const hitIndex = hit ? Number(hit.dataset.refIndex) : NaN;
    const replaceFirst = Number.isFinite(hitIndex) && hitIndex >= offset && hitIndex < offset + slots;
    for (let i = 0; i < matching.length; i++) {
        const seg = editor.timeline.segments[index];
        if (!seg) return;
        const target = (i === 0 && replaceFirst)
            ? hitIndex
            : nextEmptyGroupSlot(seg[itemsKey], offset, slots, hasFn);
        if (target < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        const ok = await assignFile(editor, index, target, matching[i]);
        if (!ok) return;
    }
}

function createR2vSection(title, countText, { onPickExisting, pickDisabled = false } = {}) {
    const section = document.createElement("div");
    section.className = "bd-r2v-section";
    const head = document.createElement("div");
    head.className = "bd-r2v-section-head";
    const titleEl = document.createElement("span");
    titleEl.className = "bd-r2v-section-title";
    titleEl.textContent = title;
    const actions = document.createElement("span");
    actions.className = "bd-r2v-section-actions";
    if (onPickExisting) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "bd-r2v-pick-existing";
        btn.textContent = t("mediaPicker.pickExisting");
        btn.title = pickDisabled ? t("mediaPicker.slotsFull") : t("mediaPicker.pickExistingHint");
        btn.disabled = !!pickDisabled;
        btn.onclick = (e) => {
            e.stopPropagation();
            if (btn.disabled) return;
            void onPickExisting();
        };
        actions.appendChild(btn);
    }
    const countEl = document.createElement("span");
    countEl.className = "bd-r2v-section-count";
    countEl.textContent = countText;
    actions.appendChild(countEl);
    head.appendChild(titleEl);
    head.appendChild(actions);
    section.appendChild(head);
    return section;
}

function renderAudioSlot(el, ref, slot, index, editor, { r2v = false } = {}) {
    const label = refAudioLabel(slot);
    const file = ref?.audioFile || ref?.fileName || "";
    el.className = `bd-batch-audio${file ? " has-audio" : ""}`;
    el.dataset.refKind = "audio";
    el.dataset.refIndex = String(slot);
    el.title = file
        ? t("ref.audioTitleFilled", { label, file })
        : t("ref.clickUpload", { label });
    el.innerHTML = "";
    if (r2v) {
        const thumb = document.createElement("div");
        thumb.className = "bd-r2v-thumb";
        const meta = document.createElement("div");
        meta.className = "bd-r2v-meta";
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = label;
        meta.appendChild(tag);
        el.appendChild(thumb);
        el.appendChild(meta);
        if (file) {
            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.className = "bd-r2v-play";
            playBtn.title = t("batch.r2v.play");
            playBtn.textContent = "▶";
            thumb.appendChild(playBtn);
            const dur = document.createElement("span");
            dur.className = "bd-r2v-dur";
            dur.textContent = ref?.durationSec != null
                ? formatMediaDuration(ref.durationSec)
                : "--:--";
            meta.appendChild(dur);
            const progress = document.createElement("div");
            progress.className = "bd-r2v-progress";
            progress.title = t("batch.r2v.seek");
            progress.innerHTML = `<div class="bd-r2v-progress-fill"></div>`;
            el.appendChild(progress);
            const audio = document.createElement("audio");
            audio.preload = "metadata";
            audio.src = viewUrl(file);
            audio.className = "bd-r2v-media";
            el.appendChild(audio);
            bindR2vMediaPlayback(audio, playBtn, progress);
            wireMediaDuration(audio, dur, (sec) => {
                if (ref) ref.durationSec = sec;
            });
            const x = document.createElement("span");
            x.className = "x";
            x.textContent = "×";
            x.onclick = (e) => { e.stopPropagation(); removeSegAudio(editor, index, slot); };
            el.appendChild(x);
        } else {
            thumb.textContent = "♪";
            const hint = document.createElement("span");
            hint.className = "name";
            hint.textContent = t("batch.r2v.uploadHint");
            meta.appendChild(hint);
        }
        return;
    }
    if (file) {
        const tag = document.createElement("span");
        tag.textContent = label;
        el.appendChild(tag);
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = fileBaseName(file);
        el.appendChild(name);
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegAudio(editor, index, slot); };
        el.appendChild(x);
    } else {
        el.textContent = t("ref.audioUpload", { label });
    }
}

function renderVideoSlot(el, ref, slot, index, editor, { r2v = false } = {}) {
    const label = refVideoLabel(slot);
    const file = ref?.videoFile || "";
    const posterSrc = ref?.previewImageUrl
        || (ref?.previewImageFile ? viewUrl(ref.previewImageFile) : "");
    const hasMedia = !!(file || posterSrc || ref?.linked);
    const titleFile = file || ref?.fileName || ref?.previewImageFile || "";
    el.className = `bd-batch-video${hasMedia ? " has-video" : ""}`;
    el.dataset.refKind = "video";
    el.dataset.refIndex = String(slot);
    el.title = hasMedia
        ? t("ref.videoTitleFilled", { label, file: titleFile || label })
        : t("ref.videoTitleEmpty", { label });
    el.innerHTML = "";
    if (r2v) {
        const thumb = document.createElement("div");
        thumb.className = "bd-r2v-thumb bd-r2v-thumb-video";
        const meta = document.createElement("div");
        meta.className = "bd-r2v-meta";
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = label;
        meta.appendChild(tag);
        el.appendChild(thumb);
        el.appendChild(meta);
        if (file) {
            const video = document.createElement("video");
            video.preload = "metadata";
            video.muted = true;
            video.playsInline = true;
            video.src = viewUrl(file);
            video.className = "bd-r2v-media";
            thumb.appendChild(video);
            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.className = "bd-r2v-play";
            playBtn.title = t("batch.r2v.play");
            playBtn.textContent = "▶";
            thumb.appendChild(playBtn);
            const dur = document.createElement("span");
            dur.className = "bd-r2v-dur";
            dur.textContent = ref?.durationSec != null
                ? formatMediaDuration(ref.durationSec)
                : "--:--";
            meta.appendChild(dur);
            bindR2vMediaPlayback(video, playBtn);
            playBtn.addEventListener("click", () => {
                video.muted = false;
            });
            wireMediaDuration(video, dur, (sec) => {
                if (ref) ref.durationSec = sec;
            });
            video.addEventListener("loadeddata", () => {
                if (video.readyState >= 2 && video.currentTime < 0.05) {
                    try { video.currentTime = Math.min(0.1, (video.duration || 1) * 0.05); } catch (_) { /* ignore */ }
                }
            }, { once: true });
            if (ref?.pairedAudioFile) {
                const audioBadge = document.createElement("span");
                audioBadge.className = "bd-r2v-paired-audio";
                audioBadge.title = ref.pairedAudioFile;
                audioBadge.textContent = "♪";
                meta.appendChild(audioBadge);
            }
            const x = document.createElement("span");
            x.className = "x";
            x.textContent = "×";
            x.onclick = (e) => { e.stopPropagation(); removeSegVideo(editor, index, slot); };
            el.appendChild(x);
        } else if (posterSrc) {
            // External IMAGE-batch video: show upstream first-frame poster (no file path).
            const img = document.createElement("img");
            img.className = "bd-r2v-media";
            img.src = posterSrc;
            img.alt = label;
            thumb.appendChild(img);
            const hint = document.createElement("span");
            hint.className = "name";
            hint.textContent = t("batch.r2v.externalPoster");
            meta.appendChild(hint);
            if (ref?.pairedAudioFile) {
                const audioBadge = document.createElement("span");
                audioBadge.className = "bd-r2v-paired-audio";
                audioBadge.title = ref.pairedAudioFile;
                audioBadge.textContent = "♪";
                meta.appendChild(audioBadge);
            }
        } else if (ref?.linked) {
            thumb.textContent = "▶";
            const hint = document.createElement("span");
            hint.className = "name";
            hint.textContent = t("batch.r2v.externalLinked");
            meta.appendChild(hint);
        } else {
            thumb.textContent = "▶";
            const hint = document.createElement("span");
            hint.className = "name";
            hint.textContent = t("batch.r2v.uploadHint");
            meta.appendChild(hint);
        }
        return;
    }
    if (file) {
        const tag = document.createElement("span");
        tag.textContent = label;
        el.appendChild(tag);
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = fileBaseName(file);
        el.appendChild(name);
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegVideo(editor, index, slot); };
        el.appendChild(x);
    } else {
        el.textContent = t("ref.videoUpload", { label });
    }
}

/**
 * r2v layout: left = pictures/videos/audio · right = prompt + preview (returned).
 * @returns {HTMLElement} main column for prompt/preview
 */
function appendR2vMediaSections(card, seg, index, editor) {
    const picOffset = r2vCommonPicOffset(editor);
    const audOffset = r2vCommonAudioOffset(editor);
    const vidOffset = r2vCommonVideoOffset(editor);
    const picSlots = Math.max(0, R2V_PICTURE_SLOTS - picOffset);
    const audSlots = Math.max(0, MAX_REFERENCE_AUDIOS - audOffset);
    const vidSlots = Math.max(0, MAX_REFERENCE_VIDEOS - vidOffset);
    const counts = countFilledRefs(seg, { picOffset, audOffset, vidOffset });
    const commonImgs = listCommonImageRefs(editor);
    const commonVids = listCommonVideoRefs(editor);
    const body = document.createElement("div");
    body.className = "bd-batch-r2v-body";

    const assets = document.createElement("div");
    assets.className = "bd-batch-r2v-assets";

    // Inherited common pictures (read-only preview) so groups still "see" shared cast.
    if (commonImgs.length) {
        const inherit = createR2vSection(
            t("batch.r2v.commonInheritPics"),
            `${commonImgs.length}`,
        );
        inherit.classList.add("bd-r2v-common-inherit");
        const inheritGrid = document.createElement("div");
        inheritGrid.className = "bd-batch-refs";
        for (const ref of commonImgs) {
            const abs = Number(ref.index ?? ref.slot ?? 0);
            const slot = document.createElement("div");
            slot.className = "bd-batch-ref has-img";
            slot.title = t("batch.r2v.commonInheritTip", { label: refImageLabel(abs) });
            const img = document.createElement("img");
            img.src = viewUrl(ref.imageFile);
            img.draggable = false;
            slot.appendChild(img);
            const cap = document.createElement("span");
            cap.className = "cap";
            cap.textContent = refImageLabel(abs);
            slot.appendChild(cap);
            inheritGrid.appendChild(slot);
        }
        inherit.appendChild(inheritGrid);
        assets.appendChild(inherit);
    }

    const groupPicTitle = picOffset > 0
        ? t("batch.r2v.sectionPicturesFrom", { n: picOffset + 1 })
        : t("batch.r2v.sectionPictures");
    const imgSection = createR2vSection(
        groupPicTitle,
        picSlots > 0 ? `${counts.imgs}/${picSlots}` : "0/0",
        picSlots > 0
            ? {
                onPickExisting: () => pickExistingSegRef(editor, index, picOffset, picSlots),
                pickDisabled: counts.imgs >= picSlots,
            }
            : {},
    );
    if (picOffset > 0) {
        const hint = document.createElement("p");
        hint.className = "bd-r2v-slot-hint";
        hint.textContent = t("batch.r2v.slotContinueHint", {
            from: picOffset + 1,
            common: picOffset,
        });
        imgSection.appendChild(hint);
    }
    const refs = document.createElement("div");
    refs.className = "bd-batch-refs";
    if (!editor._r2vPicsVisible) editor._r2vPicsVisible = {};
    const segKey = String(seg.id ?? index);
    let highestLocal = -1;
    for (const r of seg.refs || []) {
        const idx = Number(r.index ?? r.slot);
        if (r?.imageFile && Number.isFinite(idx) && idx >= picOffset) {
            highestLocal = Math.max(highestLocal, idx - picOffset);
        }
    }
    const minVisible = highestLocal >= 0
        ? Math.min(picSlots, Math.ceil((highestLocal + 1) / R2V_PICTURE_STEP) * R2V_PICTURE_STEP)
        : Math.min(picSlots, R2V_PICTURE_STEP);
    let visible = Number(editor._r2vPicsVisible[segKey]) || Math.min(picSlots, R2V_PICTURE_STEP);
    visible = Math.max(0, Math.min(picSlots, visible));
    if (picSlots > 0) {
        visible = Math.max(Math.min(picSlots, R2V_PICTURE_STEP), Math.min(picSlots, visible));
        if (visible < minVisible) visible = minVisible;
    }
    editor._r2vPicsVisible[segKey] = visible;

    const applyPicVisibility = () => {
        refs.querySelectorAll(".bd-batch-ref").forEach((el, i) => {
            el.classList.toggle("bd-r2v-pic-hidden", i >= visible);
        });
    };

    if (picSlots <= 0) {
        const empty = document.createElement("p");
        empty.className = "bd-r2v-slot-hint";
        empty.textContent = t("batch.r2v.noGroupPicSlots");
        imgSection.appendChild(empty);
    } else {
        for (let local = 0; local < picSlots; local++) {
            const abs = picOffset + local;
            const ref = (seg.refs || []).find((r) => Number(r.index ?? r.slot) === abs);
            const slot = document.createElement("div");
            slot.className = "bd-batch-ref";
            slot.dataset.refKind = "image";
            slot.dataset.refIndex = String(abs);
            if (local >= visible) slot.classList.add("bd-r2v-pic-hidden");
            renderR2vRefSlot(slot, ref, abs, index, editor);
            slot.onclick = () => {
                if (editor._batchRefDragMoved) {
                    editor._batchRefDragMoved = false;
                    return;
                }
                uploadSegRef(editor, index, abs);
            };
            bindBatchRefDrop(slot, editor, index, abs);
            refs.appendChild(slot);
        }
        imgSection.appendChild(refs);

        if (picSlots > R2V_PICTURE_STEP) {
            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "bd-r2v-pics-toggle";
            const syncToggleLabel = () => {
                if (visible < picSlots) {
                    const next = Math.min(R2V_PICTURE_STEP, picSlots - visible);
                    toggle.textContent = t("batch.r2v.expandPics", { n: next });
                } else {
                    toggle.textContent = t("batch.r2v.collapsePics");
                }
            };
            syncToggleLabel();
            toggle.onclick = (e) => {
                e.stopPropagation();
                if (visible < picSlots) {
                    visible = Math.min(picSlots, visible + R2V_PICTURE_STEP);
                } else {
                    visible = Math.max(Math.min(picSlots, R2V_PICTURE_STEP), minVisible);
                }
                editor._r2vPicsVisible[segKey] = visible;
                applyPicVisibility();
                syncToggleLabel();
                editor.updateDomWidgetHeight?.();
            };
            imgSection.appendChild(toggle);
        }
    }
    assets.appendChild(imgSection);

    if (commonVids.length) {
        const inheritVids = createR2vSection(
            t("batch.r2v.commonInheritVideos"),
            `${commonVids.length}`,
        );
        inheritVids.classList.add("bd-r2v-common-inherit");
        const inheritGrid = document.createElement("div");
        inheritGrid.className = "bd-batch-videos";
        for (const ref of commonVids) {
            const abs = Number(ref.index ?? ref.slot ?? 0);
            const slot = document.createElement("div");
            renderVideoSlot(slot, ref, abs, index, editor, { r2v: true });
            // Read-only inherit: strip remove control and upload click.
            slot.querySelector(".x")?.remove();
            slot.title = t("batch.r2v.commonInheritTip", { label: refVideoLabel(abs) });
            slot.onclick = (e) => {
                if (e.target.closest?.(".bd-r2v-play, .bd-r2v-dur, video")) return;
                if (e.target.closest?.(".bd-r2v-thumb")) {
                    slot.querySelector(".bd-r2v-play")?.click();
                }
            };
            inheritGrid.appendChild(slot);
        }
        inheritVids.appendChild(inheritGrid);
        assets.appendChild(inheritVids);
    }

    const groupVidTitle = vidOffset > 0
        ? t("batch.r2v.sectionVideosFrom", { n: vidOffset + 1 })
        : t("batch.r2v.sectionVideos");
    const videoSection = createR2vSection(
        groupVidTitle,
        vidSlots > 0 ? `${counts.videos}/${vidSlots}` : "0/0",
        vidSlots > 0
            ? {
                onPickExisting: () => pickExistingSegVideo(editor, index, vidOffset, vidSlots),
                pickDisabled: counts.videos >= vidSlots,
            }
            : {},
    );
    const videos = document.createElement("div");
    videos.className = "bd-batch-videos";
    if (vidSlots > 0) {
        bindOsFileDrop(videos, (files, e) => dropFilesIntoGroupSlots(editor, index, files, e, {
            isFile: isBatchVideoFile,
            slotSelector: ".bd-batch-video",
            offset: vidOffset,
            slots: vidSlots,
            itemsKey: "refVideos",
            hasFn: _refHasVideo,
            assignFile: assignSegVideoFromFile,
        }));
    }
    if (vidSlots <= 0 && vidOffset > 0) {
        const empty = document.createElement("p");
        empty.className = "bd-r2v-slot-hint";
        empty.textContent = t("batch.r2v.noGroupVideoSlots");
        videoSection.appendChild(empty);
    } else {
        for (let local = 0; local < vidSlots; local++) {
            const abs = vidOffset + local;
            const ref = (seg.refVideos || []).find((r) => Number(r.index ?? r.slot) === abs);
            const slot = document.createElement("div");
            renderVideoSlot(slot, ref, abs, index, editor, { r2v: true });
            slot.onclick = (e) => {
                if (e.target.closest?.(".bd-r2v-play, .bd-r2v-dur, .bd-r2v-progress, .x, video, audio")) return;
                if (ref && e.target.closest?.(".bd-r2v-thumb")) {
                    slot.querySelector(".bd-r2v-play")?.click();
                    return;
                }
                uploadSegVideo(editor, index, abs);
            };
            videos.appendChild(slot);
        }
        videoSection.appendChild(videos);
    }
    assets.appendChild(videoSection);

    const groupAudTitle = audOffset > 0
        ? t("batch.r2v.sectionAudiosFrom", { n: audOffset + 1 })
        : t("batch.r2v.sectionAudios");
    const audioSection = createR2vSection(
        groupAudTitle,
        audSlots > 0 ? `${counts.audios}/${audSlots}` : "0/0",
        audSlots > 0
            ? {
                onPickExisting: () => pickExistingSegAudio(editor, index, audOffset, audSlots),
                pickDisabled: counts.audios >= audSlots,
            }
            : {},
    );
    const audios = document.createElement("div");
    audios.className = "bd-batch-audios";
    if (audSlots > 0) {
        bindOsFileDrop(audios, (files, e) => dropFilesIntoGroupSlots(editor, index, files, e, {
            isFile: isReferenceAudioSourceFile,
            slotSelector: ".bd-batch-audio",
            offset: audOffset,
            slots: audSlots,
            itemsKey: "refAudios",
            hasFn: _refHasAudio,
            assignFile: assignSegAudioFromFile,
        }));
    }
    if (audSlots <= 0 && audOffset > 0) {
        const empty = document.createElement("p");
        empty.className = "bd-r2v-slot-hint";
        empty.textContent = t("batch.r2v.noGroupAudioSlots");
        audioSection.appendChild(empty);
    } else {
        for (let local = 0; local < audSlots; local++) {
            const abs = audOffset + local;
            const ref = (seg.refAudios || []).find((r) => Number(r.index ?? r.slot) === abs);
            const slot = document.createElement("div");
            renderAudioSlot(slot, ref, abs, index, editor, { r2v: true });
            slot.onclick = (e) => {
                if (e.target.closest?.(".bd-r2v-play, .bd-r2v-dur, .bd-r2v-progress, .x, video, audio")) return;
                if (ref && e.target.closest?.(".bd-r2v-thumb")) {
                    slot.querySelector(".bd-r2v-play")?.click();
                    return;
                }
                uploadSegAudio(editor, index, abs);
            };
            audios.appendChild(slot);
        }
        audioSection.appendChild(audios);
    }
    assets.appendChild(audioSection);

    const main = document.createElement("div");
    main.className = "bd-batch-r2v-main";

    body.appendChild(assets);
    body.appendChild(main);
    card.appendChild(body);
    return main;
}

function renderR2vRefSlot(el, ref, slot, index, editor) {
    const label = refImageLabel(slot);
    const has = !!ref?.imageFile;
    el.classList.toggle("has-img", has);
    el.innerHTML = "";
    el.title = t("ref.clickUploadMove", { label });
    if (has) {
        const img = document.createElement("img");
        img.src = viewUrl(ref.imageFile);
        img.draggable = false;
        el.appendChild(img);
        const dot = document.createElement("span");
        dot.className = "dot";
        el.appendChild(dot);
        const cap = document.createElement("span");
        cap.className = "cap";
        cap.textContent = label;
        el.appendChild(cap);
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegRef(editor, index, slot); };
        el.appendChild(x);
    } else {
        const cap = document.createElement("span");
        cap.className = "cap";
        cap.textContent = label;
        el.appendChild(cap);
    }
}

function renderSourceSlot(el, imageFile) {
    el.classList.toggle("has-img", !!imageFile);
    if (imageFile) {
        el.innerHTML = `<img src="${viewUrl(imageFile)}" alt="">`;
    } else {
        el.textContent = t("batch.uploadSource");
    }
}

function renderRefSlot(el, ref, slot, index, editor) {
    const label = refImageLabel(slot);
    el.classList.toggle("has-img", !!ref?.imageFile);
    el.innerHTML = "";
    el.title = t("ref.clickUploadMove", { label });
    if (ref?.imageFile) {
        const img = document.createElement("img");
        img.src = viewUrl(ref.imageFile);
        img.draggable = false;
        el.appendChild(img);
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "×";
        x.onclick = (e) => { e.stopPropagation(); removeSegRef(editor, index, slot); };
        el.appendChild(x);
    } else {
        el.textContent = label;
    }
}

function frameSrc(b64) {
    if (!b64) return "";
    return b64.startsWith("data:") ? b64 : `data:image/jpeg;base64,${b64}`;
}

function loadFrameImages(frames) {
    return Promise.all(frames.map((b64) => new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = frameSrc(b64);
    })));
}

function drawFrame(canvas, img) {
    const ctx = canvas.getContext("2d");
    if (!ctx || !img) return;
    const cw = canvas.clientWidth || 160;
    const ch = canvas.clientHeight || 90;
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cw, ch);
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
    const dw = img.naturalWidth * scale;
    const dh = img.naturalHeight * scale;
    ctx.drawImage(img, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
}

function mountLivePreview(el, seg, badgeText) {
    stopPlayer(el);
    el.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "bd-batch-live-preview";
    const img = document.createElement("img");
    img.className = "bd-live-preview";
    img.alt = "live preview";
    img.src = frameSrc(seg.previewB64);
    const badge = document.createElement("div");
    badge.className = "bd-batch-live-badge";
    badge.textContent = badgeText || t("batch.generating");
    wrap.appendChild(img);
    wrap.appendChild(badge);
    el.appendChild(wrap);
}

function mountVideoPreview(el, seg, running, fps, editor) {
    stopPlayer(el);
    el.innerHTML = "";
    if (running) {
        if (seg.previewB64) {
            const step = seg.previewStep;
            const total = seg.previewTotalSteps;
            const badge = (step && total)
                ? t("batch.generatingStep", { step, total })
                : t("batch.generating");
            mountLivePreview(el, seg, badge);
            return;
        }
        el.textContent = t("batch.generating");
        return;
    }
    const frames = (seg.previewFrames?.length ? seg.previewFrames : null)
        || (seg.previewB64 ? [seg.previewB64] : null);
    if (!frames?.length) {
        const refFile = firstRefPreviewFile(editor, seg);
        if (refFile) {
            mountRefStillPreview(el, refFile);
            return;
        }
        el.textContent = t("batch.previewVideoAfterRun");
        return;
    }
    const wrap = document.createElement("div");
    wrap.className = "bd-batch-vpreview";
    const canvas = document.createElement("canvas");
    canvas.height = 90;
    const ctrl = document.createElement("div");
    ctrl.className = "bd-batch-vpreview-ctrl";
    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "bd-btn";
    playBtn.textContent = t("batch.play");
    const meta = document.createElement("div");
    meta.className = "bd-batch-vpreview-meta";
    meta.textContent = t("batch.previewMeta", { n: frames.length, fps: formatPreviewFps(fps) });
    ctrl.appendChild(playBtn);
    wrap.appendChild(canvas);
    wrap.appendChild(ctrl);
    wrap.appendChild(meta);
    el.appendChild(wrap);

    const state = { playing: false, timer: null, idx: 0, images: null };
    _players.set(wrap, state);

    loadFrameImages(frames).then((images) => {
        state.images = images;
        drawFrame(canvas, images[0]);
    }).catch(() => {
        meta.textContent = t("batch.previewLoadFailed");
    });

    playBtn.onclick = (e) => {
        e.stopPropagation();
        if (!state.images?.length) return;
        if (state.playing) {
            state.playing = false;
            if (state.timer) clearInterval(state.timer);
            state.timer = null;
            playBtn.textContent = t("batch.play");
            return;
        }
        state.playing = true;
        playBtn.textContent = t("batch.pause");
        const interval = Math.max(20, 1000 / Math.max(1, fps));
        state.timer = setInterval(() => {
            if (!state.images?.length) return;
            state.idx = (state.idx + 1) % state.images.length;
            drawFrame(canvas, state.images[state.idx]);
        }, interval);
    };
}

function firstRefPreviewFile(editor, seg) {
    const groupFile = [...(seg?.refs || [])]
        .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0))
        .find((r) => r?.imageFile)?.imageFile;
    if (groupFile) return groupFile;
    return listCommonImageRefs(editor).find((r) => r?.imageFile)?.imageFile || "";
}

function mountRefStillPreview(el, imageFile) {
    el.innerHTML = "";
    const img = document.createElement("img");
    img.src = viewUrl(imageFile);
    img.alt = "ref preview";
    el.appendChild(img);
}

function renderImagePreview(el, seg, running, editor) {
    stopPlayer(el);
    el.innerHTML = "";
    if (running) {
        if (seg.previewB64) {
            const step = seg.previewStep;
            const total = seg.previewTotalSteps;
            const badge = (step && total)
                ? t("batch.generatingStep", { step, total })
                : t("batch.generating");
            mountLivePreview(el, seg, badge);
            return;
        }
        el.textContent = t("batch.generating");
        return;
    }
    if (seg.previewB64) {
        const img = document.createElement("img");
        img.src = frameSrc(seg.previewB64);
        img.alt = "preview";
        el.appendChild(img);
        return;
    }
    const refFile = firstRefPreviewFile(editor, seg);
    if (refFile) {
        mountRefStillPreview(el, refFile);
        return;
    }
    el.textContent = t("batch.previewAfterRun");
}

function renderPreview(el, seg, running, isVideo, fps, editor) {
    if (isVideo) mountVideoPreview(el, seg, running, fps, editor);
    else renderImagePreview(el, seg, running, editor);
}

export function isBatchDetailSolo(editor) {
    return (editor?.timeline?.batchDetailMode || "solo") !== "all";
}

function syncBatchDetailModeButton(editor) {
    const btn = editor?.batchDetailModeBtn;
    if (!btn) return;
    const solo = isBatchDetailSolo(editor);
    btn.textContent = t(solo ? "toolbar.batchDetailSolo" : "toolbar.batchDetailAll");
    btn.title = t(solo ? "tooltip.batchDetailSolo" : "tooltip.batchDetailAll");
    btn.setAttribute("data-i18n", solo ? "toolbar.batchDetailSolo" : "toolbar.batchDetailAll");
    btn.setAttribute("data-i18n-title", solo ? "tooltip.batchDetailSolo" : "tooltip.batchDetailAll");
}

export function toggleBatchDetailMode(editor) {
    if (!editor?.timeline) return;
    flushBatchPromptInputs(editor);
    flushBatchDurationInputs(editor);
    editor.timeline.batchDetailMode = isBatchDetailSolo(editor) ? "all" : "solo";
    syncBatchDetailModeButton(editor);
    editor.commit?.(false, { syncTimeline: true });
    editor.updateDomWidgetHeight?.();
}

export function selectBatchGroup(editor, index) {
    const segs = editor?.timeline?.segments || [];
    if (!segs.length) return;
    const next = Math.max(0, Math.min(segs.length - 1, Number(index) || 0));
    if (next === editor.selectedIndex) {
        editor._syncR2vCardSelection?.();
        return;
    }
    flushBatchPromptInputs(editor);
    flushBatchDurationInputs(editor);
    editor.selectedIndex = next;
    if (isBatchDetailSolo(editor)) {
        editor.renderImageBatchGroups?.();
    } else {
        editor._syncR2vCardSelection?.();
        editor.scheduleRender?.();
    }
    editor.updateVideoNameLabel?.();
}

function batchCardEl(editor, segmentIndex) {
    return editor?.batchList?.querySelector?.(`.bd-batch-card[data-batch-index="${segmentIndex}"]`);
}

function renderBatchGroupPicker(editor, ctx) {
    const picker = editor.batchPicker;
    if (!picker) return;
    const segs = editor.timeline?.segments || [];
    const solo = isBatchDetailSolo(editor);
    const usePicker = solo && segs.length > 1 && !editor.usesBatchTimeline?.();
    picker.innerHTML = "";
    picker.classList.toggle("visible", usePicker);
    if (!usePicker) return;
    const runSelectOn = !!(editor.isRunSelectEnabled?.() && editor.supportsRunSelect?.());
    const { key, isVideo, runningIdx, externalLocked } = ctx;
    segs.forEach((seg, index) => {
        const chip = document.createElement("div");
        chip.setAttribute("role", "button");
        chip.tabIndex = 0;
        chip.className = "bd-batch-pick";
        chip.dataset.batchIndex = String(index);
        const runEnabled = !runSelectOn || !!editor.isSegmentRunEnabled?.(index);
        if (index === editor.selectedIndex) chip.classList.add("selected");
        if (index === runningIdx) chip.classList.add("running");
        if (runSelectOn && !runEnabled) chip.classList.add("run-skipped");
        const head = document.createElement("div");
        head.className = "bd-batch-pick-title";
        if (runSelectOn) {
            const runCb = document.createElement("input");
            runCb.type = "checkbox";
            runCb.className = "bd-batch-run-check";
            runCb.checked = runEnabled;
            runCb.title = t("tooltip.batchRunCheck");
            runCb.onclick = (e) => {
                e.stopPropagation();
                editor.toggleSegmentRun(index);
            };
            head.appendChild(runCb);
        }
        const title = document.createElement("span");
        title.textContent = t(key === "r2v" ? "batch.groupTitle.asset" : "batch.groupTitle.prompt", { n: index + 1 });
        head.appendChild(title);
        chip.appendChild(head);
        const meta = document.createElement("span");
        meta.className = "bd-batch-pick-meta";
        if (isVideo) {
            const sec = resolveSegmentDurationSec(seg, defaultFrameCount(key));
            meta.textContent = `${Number(sec).toFixed(1)}s`;
        } else {
            meta.textContent = `#${index + 1}`;
        }
        chip.appendChild(meta);
        const thumbSrc = seg.previewB64 || (Array.isArray(seg.previewFrames) ? seg.previewFrames[0] : "");
        if (thumbSrc) {
            const img = document.createElement("img");
            img.className = "bd-batch-pick-thumb";
            img.alt = "";
            img.src = frameSrc(thumbSrc);
            chip.appendChild(img);
        }
        chip.onclick = (e) => {
            if (e.target.closest?.("input")) return;
            selectBatchGroup(editor, index);
        };
        if (externalLocked) chip.title = t("external.durationLocked");
        picker.appendChild(chip);
    });
}

export function renderImageBatchGroups(editor) {
    const list = editor.batchList;
    if (!list) return;
    // Recover drafts before wiping the DOM (duration flush also flushes prompts).
    flushBatchPromptInputs(editor);
    flushBatchDurationInputs(editor);
    stopAllPlayers(list);
    const key = resolveTaskKey(editor.getTaskKey?.() || editor.taskTypeWidget?.value);
    const variant = imageBatchVariant(key);
    const isVideo = isVideoBatchTask(key);
    const runningIdx = editor._runHighlightSeg;
    const fps = parseFloat(editor.frameRateWidget?.value || editor.timeline?.frameRate || 24);

    if (editor.batchHint) {
        const hintKey = `batch.hint.${key}`;
        editor.batchHint.textContent = t(hintKey) !== hintKey
            ? t(hintKey)
            : t(isVideo ? "batch.hint.defaultVideo" : "batch.hint.defaultImage");
    }
    const externalLocked = !!(editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.());
    if (editor.batchI2vNotice) {
        const needsRefs = key === "r2i" || key === "r2v";
        const global = editor.timeline.global || {};
        const commonOn = key === "r2v" && !!(global.commonEnabled ?? global.common_enabled);
        const hasCommonMedia = commonOn && (
            (global.refs || []).some((r) => r?.imageFile)
            || (global.refAudios || []).some((r) => r?.audioFile || r?.fileName)
            || (global.refVideos || []).some((r) => _refHasVideo(r))
        );
        const hasAnyMedia = hasCommonMedia || (editor.timeline.segments || []).some((s) => (
            (s.refs || []).some((r) => r?.imageFile)
            || (s.refAudios || []).some((r) => r?.audioFile || r?.fileName)
            || (s.refVideos || []).some((r) => (
                r?.videoFile || r?.fileName || r?.previewImageFile || r?.previewImageUrl || r?.linked
            ))
        ));
        // External graph media may exist as tensors even when UI path sync failed —
        // don't scare users with a false "will degrade to t2v" notice.
        if (needsRefs && !hasAnyMedia && !externalLocked) {
            editor.batchI2vNotice.textContent = t(key === "r2v" ? "batch.notice.r2vNoRefs" : "batch.notice.r2iNoRefs");
            editor.batchI2vNotice.classList.add("visible");
        } else {
            editor.batchI2vNotice.classList.remove("visible");
            editor.batchI2vNotice.textContent = "";
        }
    }
    const addBtn = editor.batchPanel?.querySelector('[data-a="batch-add"]');
    if (addBtn) {
        addBtn.textContent = t(key === "r2v" ? "batch.addRefGroup" : "batch.addPromptGroup");
        addBtn.setAttribute("data-i18n", key === "r2v" ? "batch.addRefGroup" : "batch.addPromptGroup");
        // r2v: add from toolbar (left of task select), like fl2v.
        // External groups: never add UI cards (graph is source of truth).
        addBtn.classList.toggle("hidden", key === "r2v" || externalLocked);
        addBtn.disabled = externalLocked;
    }

    teardownPromptImageMentions(list);
    list.innerHTML = "";
    const ctx = { key, variant, isVideo, runningIdx, fps, externalLocked };
    const segs = editor.timeline.segments || [];
    if (editor.selectedIndex == null || editor.selectedIndex < 0 || editor.selectedIndex >= segs.length) {
        editor.selectedIndex = 0;
    }
    syncBatchDetailModeButton(editor);
    renderBatchGroupPicker(editor, ctx);
    const solo = isBatchDetailSolo(editor);
    const indices = solo ? [editor.selectedIndex] : segs.map((_, i) => i);
    for (const index of indices) {
        const seg = segs[index];
        if (!seg) continue;
        appendBatchCard(list, editor, seg, index, ctx);
    }
    refreshPromptTokenEditors(list);
    editor.updateDomWidgetHeight?.();
}

function appendBatchCard(list, editor, seg, index, ctx) {
        const { key, variant, isVideo, runningIdx, fps, externalLocked } = ctx;
        const isR2v = key === "r2v";
        const card = document.createElement("div");
        const layoutClass = isR2v
            ? "bd-batch-r2v"
            : (variant === "source" ? "bd-batch-source"
                : (variant === "refs" ? "bd-batch-refs" : "bd-batch-plain"));
        card.className = `bd-batch-card ${layoutClass}`;
        card.dataset.batchIndex = String(index);
        const runSelectOn = !!(editor.isRunSelectEnabled?.() && editor.supportsRunSelect?.());
        const runEnabled = !runSelectOn || !!editor.isSegmentRunEnabled?.(index);
        // r2v: always show focus selected. t2v/i2v: only run-select participation chrome.
        if (isR2v && index === editor.selectedIndex) card.classList.add("selected");
        if (index === runningIdx) card.classList.add("running");
        if (runSelectOn && runEnabled) card.classList.add("run-on");
        if (runSelectOn && !runEnabled) card.classList.add("run-skipped");
        card.onclick = (e) => {
            if (e.target.closest?.("button, input, textarea, select, .bd-batch-ref, .bd-batch-audio, .bd-batch-video, .bd-batch-src, .bd-r2v-section, .bd-r2v-play, .x, video, audio")) {
                return;
            }
            selectBatchGroup(editor, index);
        };
        const hasPreview = isVideo
            ? (seg.previewFrames?.length > 0 || seg.previewB64)
            : !!seg.previewB64;
        if (hasPreview && index !== runningIdx) card.classList.add("done");

        const head = document.createElement("div");
        head.className = "bd-batch-head";
        // Timeline + cards stay in sync for run-select (incl. r2v).
        if (runSelectOn) {
            const runCb = document.createElement("input");
            runCb.type = "checkbox";
            runCb.className = "bd-batch-run-check";
            runCb.checked = runEnabled;
            runCb.title = t("tooltip.batchRunCheck");
            runCb.onclick = (e) => {
                e.stopPropagation();
                editor.toggleSegmentRun(index);
            };
            head.appendChild(runCb);
        }
        const title = document.createElement("b");
        title.textContent = t(isR2v ? "batch.groupTitle.asset" : "batch.groupTitle.prompt", { n: index + 1 });
        head.appendChild(title);
        // Per-segment continuity (master「段间引导」must be on; skip segment 1).
        const masterCont = isContinuityMasterEnabled(editor.timeline?.output);
        if (masterCont && index > 0 && isVideo) {
            const contLabel = document.createElement("label");
            contLabel.className = "bd-batch-continuity";
            contLabel.title = t("tooltip.segmentContinuityFromPrev");
            const contCb = document.createElement("input");
            contCb.type = "checkbox";
            contCb.className = "bd-batch-continuity-check";
            contCb.checked = isSegmentContinuityFromPrev(seg, index);
            contCb.onchange = (e) => {
                e.stopPropagation();
                seg.continuityFromPrev = !!contCb.checked;
                // Flush timeline_data immediately so Queue Prompt cannot race the debounce.
                editor.commit?.(false, { syncTimeline: true });
                editor.flushTimelineSync?.();
            };
            contCb.onclick = (e) => e.stopPropagation();
            const contText = document.createElement("span");
            contText.setAttribute("data-i18n", "batch.continuityFromPrev");
            contText.textContent = t("batch.continuityFromPrev");
            contLabel.appendChild(contCb);
            contLabel.appendChild(contText);
            head.appendChild(contLabel);
        }
        const meta = document.createElement("div");
        meta.className = "bd-batch-head-meta";
        if (isR2v) {
            const sizeRow = document.createElement("label");
            sizeRow.className = "bd-batch-refsize";
            sizeRow.title = t("tooltip.refImageSize");
            const sizeLabel = document.createElement("span");
            sizeLabel.setAttribute("data-i18n", "output.refImageSize.label");
            sizeLabel.textContent = t("output.refImageSize.label");
            const sizeSel = document.createElement("select");
            sizeSel.className = "bd-select";
            const curSize = resolveSegmentRefImageSize(seg, editor.timeline?.output);
            seg.refImageSize = curSize;
            for (const opt of ["match", "max"]) {
                const o = document.createElement("option");
                o.value = opt;
                o.setAttribute("data-i18n", `output.refImageSize.${opt}`);
                o.textContent = t(`output.refImageSize.${opt}`);
                if (opt === curSize) o.selected = true;
                sizeSel.appendChild(o);
            }
            sizeSel.onchange = (e) => {
                e.stopPropagation();
                const liveIdx = (editor.timeline.segments || []).findIndex((s) => s?.id && s.id === seg.id);
                const live = editor.timeline.segments?.[liveIdx >= 0 ? liveIdx : index];
                if (!live) return;
                live.refImageSize = resolveSegmentRefImageSize({ refImageSize: sizeSel.value });
                editor.commit?.(false, { syncTimeline: true });
                editor.flushTimelineSync?.();
            };
            sizeSel.onclick = (e) => e.stopPropagation();
            sizeRow.appendChild(sizeLabel);
            sizeRow.appendChild(sizeSel);
            meta.appendChild(sizeRow);
        }
        if (isVideo) {
            const secRow = document.createElement("label");
            secRow.className = "bd-batch-fc";
            const curSec = resolveSegmentDurationSec(seg, defaultFrameCount(key));
            const { frames, durationSec: syncedSec } = durationToClampedMiniMaxFrames(curSec, 24);
            const playSec = framesToDurationSec(frames, 24);
            seg.durationSec = syncedSec;
            seg.frameCount = frames;
            seg.length = frames;
            seg._videoFrameCount = frames;
            secRow.innerHTML = `${t("batch.seconds")} <input type="number" data-batch-sec-index="${index}" data-batch-seg-id="${seg.id || ""}" min="${minDurationSec()}" max="${maxDurationSec()}" step="0.1" value="${seg.durationSec}" title="${t("batch.durationTooltip", { frames, play: playSec })}">`;
            const secInput = secRow.querySelector("input");
            const applySec = () => {
                const updated = applyBatchSegmentDuration(editor, index, secInput.value);
                if (!updated) return;
                const play = framesToDurationSec(updated.frameCount, 24);
                secInput.value = String(updated.durationSec);
                secInput.title = t("batch.durationTooltip", {
                    frames: updated.frameCount,
                    play,
                });
                editor.scheduleTimelineSync();
                editor.scheduleRender?.();
                editor.updateVideoNameLabel?.();
                editor.updateOutputPreview?.();
                // Keep total_frames widget in sync with sum of group frames.
                if (editor.totalFramesWidget) {
                    editor.totalFramesWidget.value = sumFrameCounts(editor.timeline.segments);
                }
            };
            if (externalLocked) {
                secInput.readOnly = true;
                secInput.disabled = true;
                secInput.title = t("external.durationLocked");
            } else {
                secInput.onchange = applySec;
                secInput.oninput = () => {
                    clearTimeout(secInput._t);
                    secInput._t = setTimeout(applySec, 200);
                };
                secInput.onblur = () => {
                    clearTimeout(secInput._t);
                    secInput._t = null;
                    applySec();
                };
            }
            meta.appendChild(secRow);
        }
        if (!externalLocked) {
            const del = document.createElement("button");
            del.type = "button";
            del.className = "bd-batch-del";
            del.textContent = t("batch.delete");
            del.disabled = editor.timeline.segments.length <= 1;
            del.onclick = (e) => {
                e.stopPropagation();
                const liveIdx = (editor.timeline.segments || []).findIndex((s) => s?.id && s.id === seg.id);
                deleteImageBatchGroup(editor, liveIdx >= 0 ? liveIdx : index);
            };
            meta.appendChild(del);
        }
        head.appendChild(meta);
        card.appendChild(head);

        if (variant === "source") {
            const media = document.createElement("div");
            media.className = "bd-batch-media";
            const src = document.createElement("div");
            src.className = "bd-batch-src";
            renderSourceSlot(src, seg.genImage?.imageFile);
            src.onclick = () => uploadSegSource(editor, index);
            bindOsFileDrop(src, (files) => {
                const file = files.find(isBatchImageFile);
                if (file) void assignSegSourceFromFile(editor, index, file);
            });
            media.appendChild(src);
            const pickSrc = document.createElement("button");
            pickSrc.type = "button";
            pickSrc.className = "bd-r2v-pick-existing";
            pickSrc.textContent = t("mediaPicker.pickExisting");
            pickSrc.title = t("mediaPicker.pickExistingHint");
            pickSrc.onclick = (e) => {
                e.stopPropagation();
                void pickExistingSegSource(editor, index);
            };
            media.appendChild(pickSrc);
            card.appendChild(media);
        }
        let r2vMain = null;
        if (variant === "refs" && isR2v) {
            r2vMain = appendR2vMediaSections(card, seg, index, editor);
        } else if (variant === "refs") {
            const media = document.createElement("div");
            media.className = "bd-batch-media";
            const refs = document.createElement("div");
            refs.className = "bd-batch-refs";
            for (let i = 0; i < MAX_REFERENCE_IMAGES; i++) {
                const ref = (seg.refs || []).find((r) => Number(r.index ?? r.slot) === i);
                const slot = document.createElement("div");
                slot.className = "bd-batch-ref";
                slot.dataset.refKind = "image";
                slot.dataset.refIndex = String(i);
                renderRefSlot(slot, ref, i, index, editor);
                slot.onclick = () => {
                    if (editor._batchRefDragMoved) {
                        editor._batchRefDragMoved = false;
                        return;
                    }
                    uploadSegRef(editor, index, i);
                };
                bindBatchRefDrop(slot, editor, index, i);
                refs.appendChild(slot);
            }
            media.appendChild(refs);
            card.appendChild(media);
        }

        const prompts = document.createElement("div");
        prompts.className = "bd-batch-prompts";
        const ph = t(isR2v ? "placeholder.batchR2v" : "placeholder.batchDefault");
        prompts.innerHTML = `
            <span class="bd-label">${t("batch.prompt")}</span>
            <textarea data-f="prompt" data-batch-prompt-index="${index}" data-batch-seg-id="${seg.id || ""}" placeholder=""></textarea>`;
        prompts.querySelector("textarea").placeholder = ph;
        prompts.querySelector("textarea").value = seg.prompt || "";
        const promptEl = prompts.querySelector('[data-f="prompt"]');
        const segIndex = index;
        const segId = seg.id;
        promptEl.oninput = (e) => {
            // Write by index/id — never capture a stale `seg` after normalize.
            const live = (editor.timeline.segments || []).find((s) => s?.id && s.id === segId)
                || editor.timeline.segments?.[segIndex];
            if (!live) return;
            live.prompt = e.target.value;
            live.negativePrompt = live.negativePrompt ?? "";
            editor.scheduleTimelineSync();
            // External groups execute from Group-node widgets — keep them aligned.
            editor.writeExternalGroupPrompt?.(segIndex, live.prompt);
        };
        if (isR2v) {
            wirePromptImageMentions(editor, promptEl, () => {
                const g = editor.timeline?.global || {};
                const on = !!(g.commonEnabled ?? g.common_enabled);
                const live = (editor.timeline.segments || []).find((s) => s?.id && s.id === segId)
                    || editor.timeline.segments?.[segIndex]
                    || seg;
                // Absolute indices: common 图片1…N + group 图片N+1… (no renumber clash).
                return {
                    refs: on ? mergeMediaByIndex(g.refs || [], live.refs || []) : (live.refs || []),
                    audios: on
                        ? mergeMediaByIndex(g.refAudios || [], live.refAudios || [])
                        : (live.refAudios || []),
                    videos: on
                        ? mergeMediaByIndex(g.refVideos || [], live.refVideos || [])
                        : (live.refVideos || []),
                };
            });
        }

        const preview = document.createElement("div");
        preview.className = "bd-batch-preview";
        renderPreview(preview, seg, index === runningIdx, isVideo, seg.previewFps || fps, editor);

        if (isR2v && r2vMain) {
            r2vMain.appendChild(prompts);
            r2vMain.appendChild(preview);
        } else {
            card.appendChild(prompts);
            card.appendChild(preview);
        }

        list.appendChild(card);
}

export function setImageBatchPreview(editor, segmentIndex, imageB64, extra = {}) {
    const seg = editor.timeline.segments[segmentIndex];
    if (!seg) return;
    seg.previewB64 = imageB64 || "";
    if (extra.step != null) seg.previewStep = extra.step;
    if (extra.total_steps != null) seg.previewTotalSteps = extra.total_steps;
    if (Array.isArray(extra.frames) && extra.frames.length) {
        seg.previewFrames = extra.frames;
        seg.previewFps = extra.fps || seg.previewFps || 24;
        seg.previewLive = false;
    } else if (imageB64) {
        if (extra.live) {
            // Keep final multi-frame playback until a real final payload arrives.
            if (!Array.isArray(seg.previewFrames) || seg.previewFrames.length <= 1) {
                seg.previewFrames = [imageB64];
            }
            seg.previewLive = true;
        } else {
            seg.previewFrames = [imageB64];
            seg.previewLive = false;
        }
    }

    // Live sampling updates: patch the card preview in-place (avoid full re-render thrash).
    if (extra.live && imageB64) {
        const card = batchCardEl(editor, segmentIndex);
        const preview = card?.querySelector?.(".bd-batch-preview");
        if (preview) {
            const step = seg.previewStep;
            const total = seg.previewTotalSteps;
            const badgeText = (step && total)
                ? t("batch.generatingStep", { step, total })
                : t("batch.generating");
            let img = preview.querySelector("img.bd-live-preview");
            let badge = preview.querySelector(".bd-batch-live-badge");
            if (!img) {
                mountLivePreview(preview, seg, badgeText);
            } else {
                img.src = frameSrc(imageB64);
                if (badge) badge.textContent = badgeText;
            }
            const pickThumb = editor.batchPicker?.querySelector?.(`.bd-batch-pick[data-batch-index="${segmentIndex}"] img.bd-batch-pick-thumb`);
            if (pickThumb) pickThumb.src = frameSrc(imageB64);
            return;
        }
        const pick = editor.batchPicker?.querySelector?.(`.bd-batch-pick[data-batch-index="${segmentIndex}"]`);
        if (pick) {
            pick.classList.add("running");
            let img = pick.querySelector("img.bd-batch-pick-thumb");
            if (!img) {
                img = document.createElement("img");
                img.className = "bd-batch-pick-thumb";
                img.alt = "";
                pick.appendChild(img);
            }
            img.src = frameSrc(imageB64);
        }
        return;
    }
    editor.renderImageBatchGroups();
}

export function bindImageBatchEvents(editor) {
    editor.batchAddBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        addImageBatchGroup(editor);
    });
}

/** Default list viewport when the node is at content-sized height (not user-stretched). */
export const BATCH_LIST_MAX_H = 640;
const BATCH_LIST_MIN_H = 160;
const BATCH_LIST_GAP = 8;
const BATCH_TOOLBAR_H = 48;
const BATCH_PANEL_CHROME = 28;

export function getImageBatchUiHeight(editor) {
    const solo = isBatchDetailSolo(editor);
    const n = solo ? 1 : Math.max(1, editor?.timeline?.segments?.length || 1);
    const key = resolveTaskKey(editor?.getTaskKey?.() || editor?.taskTypeWidget?.value);
    // r2v cards are tall; list scrolls inside BATCH_LIST_MAX_H — do NOT sum full card
    // heights into node size or the DOM widget grows a huge empty region below.
    const rowH = key === "r2v" ? 420 : (isVideoBatchTask(key) ? 155 : 130);
    const showPicker = solo && (editor?.timeline?.segments?.length || 0) > 1 && !editor?.usesBatchTimeline?.();
    const pickerH = showPicker ? 56 : 0;
    const listContentH = n * rowH + Math.max(0, n - 1) * BATCH_LIST_GAP + pickerH;
    const listH = Math.min(listContentH, BATCH_LIST_MAX_H);
    return BATCH_TOOLBAR_H + BATCH_PANEL_CHROME + listH;
}

function _clearBatchListFillStyles(list, host, wrap, panel) {
    if (list) {
        list.style.height = "";
        list.style.maxHeight = "";
        list.style.minHeight = "";
        list.style.flex = "";
        list.classList?.remove("bd-batch-solo");
        for (const card of list.querySelectorAll?.(".bd-batch-card") || []) {
            card.style.flex = "";
            card.style.minHeight = "";
            card.style.height = "";
        }
    }
    if (panel) {
        panel.style.flex = "";
        panel.style.minHeight = "";
        panel.style.height = "";
        panel.style.maxHeight = "";
        panel.style.overflow = "";
    }
    if (wrap) {
        wrap.style.height = "";
        wrap.style.minHeight = "";
        wrap.style.maxHeight = "";
        wrap.style.overflow = "";
    }
    if (host) {
        host.style.height = "";
        host.style.maxHeight = "";
        host.style.overflow = "";
    }
    const main = host?.querySelector?.(".bd-main");
    if (main) {
        main.style.height = "";
        main.style.minHeight = "";
        main.style.maxHeight = "";
        main.style.overflow = "";
    }
}

/** Soft cap above content min — blocks Vue ResizeObserver / stretch feedback runaway. */
export const DIRECTOR_UI_MAX_EXTRA_H = 1200;

function _mmxHeightDebug(...args) {
    try {
        if (typeof localStorage !== "undefined" && localStorage.getItem("mmxHeightDebug") === "1") {
            console.debug("[mmx-height]", ...args);
        }
    } catch {
        /* ignore */
    }
}

/**
 * Trusted pixel height LiteGraph/ComfyUI allocates to the DOM widget.
 * Never derive from node.size (other-widgets underestimate → write-back loop) or from
 * host.clientHeight after we stamped style.height (self-referential ratchet).
 *
 * Do NOT clamp to DIRECTOR_UI_MAX_EXTRA_H here — that cap is only for heal/runaway.
 * Clamping fill caused "drag taller → content stops → huge blank below".
 */
function measureWidgetSlotHeight(editor) {
    const widget = editor?.domWidget;
    const minH = contentDomWidgetMinHeight(editor);
    const computed = Number(widget?.computedHeight);
    if (Number.isFinite(computed) && computed > 0) {
        return computed;
    }
    // Only read element box when we have not forced an inline height (avoids self-inflate).
    const el = widget?.element;
    if (el && !el.style.height) {
        const elH = Number(el.clientHeight || el.offsetHeight);
        if (Number.isFinite(elH) && elH > 0) return Math.max(elH, minH);
    }
    const host = editor?.container;
    const parent = host?.parentElement;
    if (parent && host && !host.style.height) {
        const pH = Number(parent.clientHeight);
        if (Number.isFinite(pH) && pH > minH) return pH;
    }
    // No trusted slot → content min only. Callers must not invent height from node.size.
    return minH;
}

/** True when measure came from LiteGraph computedHeight (safe to allocate px inside). */
function hasTrustedComputedSlot(editor) {
    const computed = Number(editor?.domWidget?.computedHeight);
    return Number.isFinite(computed) && computed > 0;
}

export function contentDomWidgetMinHeight(editor) {
    const minH = typeof editor?.getDirectorUiMinHeight === "function"
        ? editor.getDirectorUiMinHeight()
        : 0;
    return Math.max(0, minH || 0);
}

/**
 * Make the Director DOM widget *growable* in LiteGraph._arrangeWidgets.
 *
 * IMPORTANT: never assign widget.computeSize. If computeSize exists, LiteGraph
 * treats the widget as fixed-height and never distributes free space when the
 * user drags the node taller — 素材组 stays short with a void below.
 * Use computeLayoutSize + getMinHeight/getMaxHeight only.
 */
export function bindDomWidgetContentComputeSize(editor) {
    const widget = editor?.domWidget;
    const minH = contentDomWidgetMinHeight(editor);
    if (!widget) return minH;
    // Remove any fixed-size hook (ours or leftover) so free space can flow in.
    try {
        delete widget.computeSize;
    } catch {
        widget.computeSize = undefined;
    }
    // No maxHeight → LiteGraph gives this widget all leftover node height on drag.
    widget.computeLayoutSize = () => ({
        minHeight: minH,
        maxHeight: undefined,
        minWidth: 0,
    });
    if (widget.options) {
        widget.options.getMinHeight = () => contentDomWidgetMinHeight(editor);
        delete widget.options.getMaxHeight;
        delete widget.options.getHeight;
    }
    return minH;
}

/**
 * Grow the 素材组 list into leftover node height when the user drags the Director taller.
 * Uses the widget slot height for *layout only* — never writes it into getMinHeight
 * (that was the infinite-growth / 公共参数挤压 bug).
 *
 * @param {{ settle?: boolean }} [opts] settle=false skips rAF (progress path); default one rAF.
 */
export function syncBatchPanelFillHeight(editor, opts = {}) {
    const list = editor?.batchList;
    const wrap = editor?.root;
    const host = editor?.container;
    const panel = editor?.batchPanel;
    const main = editor?.mainBody;
    if (!list || !wrap || !host) return;

    const batchOn = !!editor.isImageBatch?.()
        && !panel?.classList?.contains("hidden");
    wrap.classList.toggle("bd-batch-fill", batchOn);
    const minH = contentDomWidgetMinHeight(editor);
    bindDomWidgetContentComputeSize(editor);

    host.style.minHeight = `${minH || 0}px`;
    host.style.setProperty("--comfy-widget-min-height", `${minH || 0}px`);

    if (!batchOn) {
        _clearBatchListFillStyles(list, host, wrap, panel);
        return;
    }

    const applyFill = () => {
        if (!editor.batchList || editor.batchPanel?.classList?.contains("hidden")) return;

        // LiteGraph DOM widgets keep a vertical margin inside computedHeight; if we
        // size content to the full slot, the run-status bar paints past the node edge.
        const widget = editor.domWidget;
        const margin = Number(widget?.margin ?? widget?.options?.margin ?? 10);
        const inset = Math.max(12, margin * 2 + 4);
        const trusted = hasTrustedComputedSlot(editor);
        const rawSlot = measureWidgetSlotHeight(editor);
        // Full LiteGraph slot (minus widget margin). Never EXTRA-cap — user drag must fill.
        const slotH = Math.max(0, rawSlot - inset);

        // Fill the allocated widget box. Prefer % so we never paint shorter than parent
        // (pixel maxHeight < computed was the "blank below 素材组" bug after EXTRA clamp).
        host.style.height = "";
        wrap.style.height = "";
        wrap.style.minHeight = "0";
        host.style.maxHeight = "100%";
        wrap.style.maxHeight = "100%";
        host.style.overflow = "hidden";
        wrap.style.overflow = "hidden";

        const status = wrap.querySelector(".bd-run-status");
        const statusH = status ? (status.offsetHeight + 6) : 0;
        let topChrome = 0;
        for (const child of wrap.children) {
            if (child === main || child === status) continue;
            if (child.classList?.contains("hidden")) continue;
            if (getComputedStyle(child).display === "none") continue;
            topChrome += child.offsetHeight + 6;
        }

        const budget = slotH > 0
            ? slotH
            : Math.max(minH, Number(wrap.clientHeight || host.clientHeight) || minH);
        const mainH = Math.max(0, budget - statusH - topChrome);

        if (main) {
            main.style.flex = "1 1 0";
            main.style.minHeight = "0";
            main.style.overflow = "hidden";
            if (trusted && slotH > 0) {
                main.style.height = `${mainH}px`;
                main.style.maxHeight = `${mainH}px`;
            } else {
                main.style.height = "";
                main.style.maxHeight = "";
            }
        }

        let used = 0;
        let visible = 0;
        if (main) {
            for (const child of main.children) {
                if (child === panel) continue;
                if (child.classList?.contains("hidden")) continue;
                if (getComputedStyle(child).display === "none") continue;
                used += child.offsetHeight;
                visible += 1;
            }
        }
        const gaps = 6 * Math.max(0, visible);
        const batchH = Math.max(0, Math.floor(mainH - used - gaps));
        panel.style.flex = "1 1 0";
        panel.style.minHeight = "0";
        panel.style.overflow = "hidden";
        if (trusted && slotH > 0) {
            panel.style.height = `${batchH}px`;
            panel.style.maxHeight = `${batchH}px`;
        } else {
            panel.style.height = "";
            panel.style.maxHeight = "";
        }

        const batchToolbar = panel.querySelector?.(".bd-batch-toolbar");
        const notice = panel.querySelector?.(".bd-batch-i2v-notice");
        const picker = editor.batchPicker;
        const noticeH = notice?.classList?.contains("visible") ? (notice.offsetHeight + 8) : 0;
        const pickerH = picker?.classList?.contains("visible") ? (picker.offsetHeight + 8) : 0;
        const listH = Math.max(
            0,
            batchH - (batchToolbar?.offsetHeight || 0) - noticeH - pickerH - 10,
        );
        list.style.flex = "1 1 0";
        list.style.minHeight = "0";
        if (trusted && slotH > 0) {
            list.style.height = `${listH}px`;
            list.style.maxHeight = `${listH}px`;
        } else {
            list.style.height = "";
            list.style.maxHeight = "";
        }

        const solo = (editor.timeline?.segments?.length || 0) <= 1 || isBatchDetailSolo(editor);
        list.classList.toggle("bd-batch-solo", solo);
        for (const card of list.querySelectorAll(".bd-batch-card")) {
            if (solo && (listH > 0 || !trusted)) {
                card.style.flex = "1 1 auto";
                card.style.minHeight = "0";
                card.style.height = "100%";
            } else {
                card.style.flex = "";
                card.style.minHeight = "";
                card.style.height = "";
            }
        }

        _mmxHeightDebug({
            task: editor.getTaskKey?.(),
            trusted,
            minH,
            rawSlot,
            slotH,
            nodeH: editor.node?.size?.[1],
            computed: editor.domWidget?.computedHeight,
        });
    };

    applyFill();
    // One settle frame after layout (resize / mode switch). Never triple-rAF on progress.
    if (opts.settle !== false) {
        requestAnimationFrame(applyFill);
    }
}

export function setToolbarDisabledForBatch(editor, disabled) {
    const btns = [
        editor.btnVideo,
        editor.btnVideoExisting,
        editor.btnVideoAppend,
        editor.root?.querySelector('[data-a="split"]'),
        editor.root?.querySelector('[data-a="smart-split"]'),
        editor.root?.querySelector('[data-a="equal"]'),
        editor.root?.querySelector('[data-a="del"]'),
        editor.root?.querySelector('[data-a="mode-global"]'),
        editor.root?.querySelector('[data-a="mode-segment"]'),
    ];
    for (const btn of btns) {
        if (!btn) continue;
        // Batch / t2v / i2v: fully hide video-editing controls (not just disable).
        btn.classList.toggle("hidden", disabled);
        btn.disabled = disabled;
        btn.classList.toggle("bd-disabled", disabled);
    }
    if (editor.equalCountInput) {
        editor.equalCountInput.classList.toggle("hidden", disabled);
        editor.equalCountInput.disabled = disabled;
        editor.equalCountInput.classList.toggle("bd-disabled", disabled);
    }
    editor.root?.querySelector('[data-r="equal-n"]')?.classList.toggle("hidden", disabled);
    editor.root?.querySelector(".bd-mode")?.classList.toggle("hidden", disabled);
}

/** r2v: fl2v-like toolbar — timeline visible; add group sits left of task select. */
export function setR2vToolbar(editor, enabled) {
    const hide = [
        editor.btnVideo,
        editor.btnVideoExisting,
        editor.btnVideoAppend,
        editor.root?.querySelector('[data-a="split"]'),
        editor.root?.querySelector('[data-a="smart-split"]'),
        editor.root?.querySelector('[data-a="equal"]'),
        editor.root?.querySelector('[data-a="mode-global"]'),
        editor.root?.querySelector('[data-a="mode-segment"]'),
    ];
    for (const btn of hide) {
        if (!btn) continue;
        btn.classList.toggle("hidden", enabled);
        btn.disabled = enabled;
        btn.classList.toggle("bd-disabled", enabled);
    }
    if (editor.equalCountInput) {
        editor.equalCountInput.classList.toggle("hidden", enabled);
        editor.equalCountInput.disabled = enabled;
        editor.equalCountInput.classList.toggle("bd-disabled", enabled);
    }
    editor.root?.querySelector('[data-r="equal-n"]')?.classList.toggle("hidden", enabled);
    editor.root?.querySelector(".bd-mode")?.classList.toggle("hidden", enabled);

    const externalLocked = !!(editor.hasExternalI2vGroups?.() || editor.hasExternalR2vGroups?.());
    const del = editor.root?.querySelector('[data-a="del"]');
    if (del) {
        if (externalLocked) {
            del.classList.add("hidden");
            del.disabled = true;
        } else {
            del.disabled = false;
            del.classList.remove("bd-disabled", "hidden");
            del.textContent = enabled ? t("toolbar.deleteSelectedGroup") : t("toolbar.deleteSegment");
            del.setAttribute("data-i18n", enabled ? "toolbar.deleteSelectedGroup" : "toolbar.deleteSegment");
            del.setAttribute("data-i18n-title", enabled ? "tooltip.deleteSelectedFl2vGroup" : "tooltip.deleteSegment");
            del.title = enabled
                ? t("tooltip.deleteSelectedFl2vGroup")
                : t("tooltip.deleteSegment");
        }
    }
    const addBtn = editor.root?.querySelector('[data-a="r2v-add-group"]');
    if (addBtn) {
        addBtn.classList.toggle("hidden", !enabled || externalLocked);
        addBtn.disabled = !enabled || externalLocked;
    }
    const batchAdd = editor.batchPanel?.querySelector('[data-a="batch-add"]');
    if (batchAdd) batchAdd.classList.toggle("hidden", enabled || externalLocked);
    updateR2vToolbarBtns(editor);
}

export function updateR2vToolbarBtns(editor) {
    const addBtn = editor?.root?.querySelector?.('[data-a="r2v-add-group"]');
    if (!addBtn) return;
    const externalLocked = !!(editor?.hasExternalI2vGroups?.() || editor?.hasExternalR2vGroups?.());
    const show = !!editor?.isR2vBatch?.() && !externalLocked;
    addBtn.classList.toggle("hidden", !show);
    addBtn.disabled = !show;
}
