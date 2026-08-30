import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    CUSTOM_ASPECT_RATIO,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MEGAPIXELS,
    defaultDurationSec,
    defaultFrameCount,
    durationToClampedMiniMaxFrames,
    framesToDurationSec,
    preferredDurationSecFromFrames,
    roundDurationSec,
    genLayoutHint,
    getDirectorMode,
    imageBatchRequiresFixedOutput,
    isContinuityMasterEnabled,
    isCustomAspectRatio,
    isPromptBatchTask,
    isSegmentContinuityFromPrev,
    isVideoBatchTask,
    MAX_GEN_FRAMES,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MINIMAX_CANVAS_MULTIPLE,
    minFrameCount,
    newBatchSegment,
    NO_VIDEO_UPLOAD_TASKS,
    normalizeAspectRatioLabel,
    parseMegapixelsInput,
    clampMegapixels,
    refAudioLabel,
    refImageLabel,
    refVideoLabel,
    RESOLUTION_ASPECTS,
    resolutionFromSelector,
    resolveTaskKey,
    resolveSegmentRefImageSize,
    normalizeRefImageSize,
    snapResolutionDim,
    sumFrameCounts,
    taskUsesReferenceAudios,
    taskUsesReferenceImages,
    taskUsesReferenceVideo,
    fileForComfyUpload,
    safeUploadFilename,
} from "./minimax_gen_timeline.js";
import {
    IMAGE_BATCH_STYLES,
    addImageBatchGroup,
    bindImageBatchEvents,
    bindR2vMediaPlayback,
    deleteImageBatchGroup,
    ensureImageBatchTimeline,
    formatMediaDuration,
    getImageBatchUiHeight,
    isBatchDetailSolo,
    listCommonImageRefs,
    mountImageBatchPanel,
    flushBatchPromptInputs,
    normalizeImageBatchSegments,
    rebaseR2vGroupSlotsForCommon,
    renderImageBatchGroups,
    setImageBatchPreview,
    setR2vToolbar,
    setToolbarDisabledForBatch,
    bindDomWidgetContentComputeSize,
    contentDomWidgetMinHeight,
    DIRECTOR_UI_MAX_EXTRA_H,
    syncBatchPanelFillHeight,
    updateR2vToolbarBtns,
    wireBatchRunSelectControls,
    wireMediaDuration,
} from "./minimax_image_batch.js";
import {
    extractReferenceAudioFromExistingVideo,
    hasDuplicateReferenceAudio,
    prepareLocalReferenceAudio,
} from "./minimax_ref_audio.js";
import {
    FL2V_STYLES,
    bindFl2vEvents,
    buildFl2vPayloadFields,
    drawFl2vSegmentThumbnails,
    ensureFl2vTimeline,
    fl2vStartIndices,
    getFl2vTotalDurationSec,
    getFl2vSampleFrames,
    getFl2vVisualFrames,
    getFl2vUiHeight,
    removeFl2vShot,
    rippleFl2vRightEdge,
    mountFl2vPanel,
    newFl2vShot,
    normalizeFl2vSegments,
    openFl2vUpload,
    setFl2vToolbar,
    flushFl2vPromptDraft,
    syncFl2vDurationSecAfterDrag,
    syncFl2vFromShots,
    updateFl2vDetailUI,
    updateFl2vToolbarBtns,
} from "./minimax_fl2v.js";
import { mountPromptImageMentions, refreshPromptTokenEditors, teardownPromptImageMentions } from "./minimax_prompt_mentions.js";
import {
    applyI18nDom,
    aspectDisplayLabel,
    getLocale,
    onLocaleChange,
    t,
    taskDisplayLabel,
    toggleLocale,
} from "./minimax_i18n.js";

const RULER_H = 24;
const SEG_LABEL_H = 20;
const TRACK_H = 160;
const TRACK_Y = RULER_H + SEG_LABEL_H;
/** Nice major steps (seconds) for CapCut-style ruler labels. */
const RULER_MAJOR_SEC = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const RULER_MIN_MAJOR_PX = 64;
const RULER_MIN_MINOR_PX = 7;
const STAGE_PREVIEW_H = 220;
const LIVE_SAMPLE_PREVIEW_H = 320;
const MIN_SEG = 4;
const HANDLE_PX = 14;
/** Canvas-drawn run-select checkbox (not a DOM control). */
const RUN_CHECK_SIZE = 14;
const RUN_CHECK_HIT_PAD_X = 8;
const RUN_CHECK_HIT_PAD_Y = 4;
/** Canvas-drawn 段间引导 marker at a clip joint (only when master switch is on). */
const CONT_JOINT_W = 22;
const CONT_JOINT_H = 16;
const CONT_JOINT_Y = TRACK_Y + 4;
const CONT_JOINT_HIT_PAD = 5;
const THUMB_MAX_W = 168;
const THUMB_JPEG_Q = 0.55;
const TIMELINE_SYNC_DEBOUNCE_MS = 500;
const MAX_THUMBS_PER_SEGMENT = 20;
const THUMB_PREFETCH_BATCH = 6;
const DIRECTOR_MIN_WIDTH = 900;
const COMFY_UPLOAD_SOFT_LIMIT = 95 * 1024 * 1024;
const MINIMAX_CHUNK_SIZE = 8 * 1024 * 1024;

/** Segment continuity is opt-in; default off unless explicitly enabled in output. */
function isContinuityEnabled(output) {
    if (!output) return false;
    const raw = output.continuityEnabled ?? output.continuity_enabled;
    if (raw === true || raw === 1) return true;
    if (typeof raw === "string") {
        const s = raw.trim().toLowerCase();
        return s === "true" || s === "1" || s === "yes" || s === "on";
    }
    return false;
}

/** Whether段间引导 controls apply for the current task + segment count. */
function isContinuityEligible(editor) {
    if (!editor) return false;
    const taskKey = resolveTaskKey(
        editor.getTaskKey?.() || editor.taskTypeWidget?.value || editor.globalTask?.value || "",
    );
    if (!CONTINUITY_TASKS.has(taskKey)) return false;
    const segCount = Array.isArray(editor.timeline?.segments) ? editor.timeline.segments.length : 0;
    const fl2vCount = Array.isArray(editor.timeline?.fl2vGroups) ? editor.timeline.fl2vGroups.length : 0;
    const r2vCount = Array.isArray(editor.timeline?.r2vGroups) ? editor.timeline.r2vGroups.length : 0;
    return Math.max(segCount, fl2vCount, r2vCount) >= 2;
}

function normalizeAudioMode(value) {
    const raw = String(value || "generate").trim().toLowerCase();
    if (raw === "source" || raw === "original" || raw === "passthrough") return "source";
    if (raw === "mute" || raw === "silent" || raw === "silence") return "mute";
    return "generate";
}

const CONTINUITY_FRAME_CHOICES = [5, 22, 39, 56];
/** Official Motion Context baseline recommendation. */
const DEFAULT_CONTINUITY_FRAMES = 22;
const CONTINUITY_TASKS = new Set(["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"]);

function isVideoEditTaskKey(taskKey) {
    return taskKey === "v2v" || taskKey === "rv2v";
}

function snapContinuityFrames(raw) {
    const n = parseInt(raw, 10);
    const value = Number.isFinite(n) ? n : DEFAULT_CONTINUITY_FRAMES;
    let best = DEFAULT_CONTINUITY_FRAMES;
    let bestDist = Infinity;
    for (const choice of CONTINUITY_FRAME_CHOICES) {
        const dist = Math.abs(choice - value);
        if (dist < bestDist || (dist === bestDist && choice > best)) {
            best = choice;
            bestDist = dist;
        }
    }
    return best;
}

function normalizeOutputContinuity(output = {}) {
    const rawOverlap = output.continuityOverlapFrames ?? output.continuity_overlap_frames ?? DEFAULT_CONTINUITY_FRAMES;
    return {
        ...output,
        continuityEnabled: isContinuityEnabled(output),
        continuityOverlapFrames: snapContinuityFrames(rawOverlap),
        audioMode: normalizeAudioMode(output.audioMode ?? output.audio_mode),
        refImageSize: normalizeRefImageSize(output.refImageSize ?? output.ref_image_size),
    };
}

function stripTimelineContinuityRootFields(timeline) {
    if (!timeline || typeof timeline !== "object") return;
    delete timeline.continuityEnabled;
    delete timeline.continuity_enabled;
    delete timeline.continuityOverlapFrames;
    delete timeline.continuity_overlap_frames;
}

/** Drop ephemeral UI-only fields so they never persist in timeline_data. */
function sanitizeRefImage(ref) {
    if (!ref || typeof ref !== "object") return ref;
    const index = Number(ref.index ?? ref.slot);
    return {
        index: Number.isFinite(index) ? index : 0,
        imageFile: ref.imageFile || "",
        fileName: ref.fileName || "",
        type: ref.type || "input",
        subfolder: ref.subfolder || "",
    };
}

function sanitizeRefAudio(ref) {
    if (!ref || typeof ref !== "object") return ref;
    const index = Number(ref.index ?? ref.slot);
    return {
        index: Number.isFinite(index) ? index : 0,
        audioFile: ref.audioFile || "",
        fileName: ref.fileName || "",
        type: ref.type || "input",
        subfolder: ref.subfolder || "",
        durationSec: ref.durationSec,
    };
}

function sanitizeRefVideo(ref) {
    if (!ref || typeof ref !== "object") return ref;
    const index = Number(ref.index ?? ref.slot);
    return {
        index: Number.isFinite(index) ? index : 0,
        videoFile: ref.videoFile || "",
        fileName: ref.fileName || "",
        type: ref.type || "input",
        subfolder: ref.subfolder || "",
        durationSec: ref.durationSec,
        pairedAudioFile: ref.pairedAudioFile || "",
        previewImageFile: ref.previewImageFile || "",
        previewImageUrl: ref.previewImageUrl || "",
        linked: !!ref.linked || !!(ref.videoFile || ref.previewImageFile || ref.previewImageUrl),
    };
}

function sanitizeSegmentForPayload(seg) {
    if (!seg || typeof seg !== "object") return seg;
    const {
        previewB64,
        previewFrames,
        imageB64,
        ...rest
    } = seg;
    return {
        ...rest,
        refs: Array.isArray(rest.refs) ? rest.refs.map(sanitizeRefImage) : [],
        refAudios: Array.isArray(rest.refAudios) ? rest.refAudios.map(sanitizeRefAudio) : [],
        refVideos: Array.isArray(rest.refVideos) ? rest.refVideos.map(sanitizeRefVideo) : [],
        genImage: rest.genImage
            ? { imageFile: rest.genImage.imageFile || "", fileName: rest.genImage.fileName || "" }
            : undefined,
        referenceVideo: rest.referenceVideo
            ? {
                videoFile: rest.referenceVideo.videoFile || "",
                fileName: rest.referenceVideo.fileName || "",
                type: rest.referenceVideo.type || "input",
                subfolder: rest.referenceVideo.subfolder || "",
            }
            : undefined,
    };
}

function cloneJson(value, fallback) {
    try {
        if (value == null) return fallback;
        return JSON.parse(JSON.stringify(value));
    } catch {
        return fallback;
    }
}

function sanitizeBatchGlobalCommon(gc) {
    const src = gc && typeof gc === "object" ? gc : {};
    return {
        commonEnabled: !!src.commonEnabled,
        commonCollapsed: !!src.commonCollapsed,
        prompt: src.prompt || "",
        refs: Array.isArray(src.refs) ? src.refs.map(sanitizeRefImage) : [],
        refAudios: Array.isArray(src.refAudios) ? src.refAudios.map(sanitizeRefAudio) : [],
        refVideos: Array.isArray(src.refVideos)
            ? src.refVideos.map(sanitizeRefVideo)
            : (Array.isArray(src.ref_videos) ? src.ref_videos.map(sanitizeRefVideo) : []),
    };
}

/** Persistable t2v/i2v/r2v snapshot (no preview frames). */
function sanitizeBatchWorkspace(ws) {
    if (!ws || typeof ws !== "object" || !Array.isArray(ws.segments) || !ws.segments.length) {
        return null;
    }
    return {
        selectedIndex: Number.isFinite(Number(ws.selectedIndex)) ? Number(ws.selectedIndex) : 0,
        editMode: ws.editMode || "segment",
        runSelectEnabled: !!ws.runSelectEnabled,
        runSelection: Array.isArray(ws.runSelection) ? [...ws.runSelection] : [],
        segments: ws.segments.map(sanitizeSegmentForPayload),
        globalCommon: sanitizeBatchGlobalCommon(ws.globalCommon),
    };
}

function sanitizeVideoMedia(video) {
    if (!video || typeof video !== "object") return video;
    const hasFile = !!(video.videoFile || video.fileName);
    const dropFrames = hasFile
        || (Array.isArray(video.frames) && video.frames.length > 8);
    return { ...video, frames: dropFrames ? [] : (video.frames || []) };
}

/** Persistable v2v/rv2v snapshot (no decoded frame blobs). */
function sanitizeVideoWorkspace(ws) {
    if (!ws || typeof ws !== "object") return null;
    return {
        selectedIndex: Number.isFinite(Number(ws.selectedIndex)) ? Number(ws.selectedIndex) : 0,
        currentFrame: Math.max(0, Number(ws.currentFrame) || 0),
        editMode: ws.editMode || "global",
        runSelectEnabled: !!ws.runSelectEnabled,
        runSelection: Array.isArray(ws.runSelection) ? [...ws.runSelection] : [],
        totalFrames: ws.totalFrames,
        frameRate: ws.frameRate,
        storageWidth: ws.storageWidth || 0,
        storageHeight: ws.storageHeight || 0,
        segments: Array.isArray(ws.segments) ? ws.segments.map(sanitizeSegmentForPayload) : [],
        video: sanitizeVideoMedia(ws.video || {}),
        videoClips: Array.isArray(ws.videoClips) ? ws.videoClips.map(sanitizeVideoMedia) : [],
        globalCommon: sanitizeBatchGlobalCommon(ws.globalCommon),
    };
}

function stripTimelineEphemeralFields(timeline) {
    if (!timeline || typeof timeline !== "object") return;
    delete timeline.videoWorkspace;
    delete timeline.batchWorkspace;
    delete timeline.fl2vWorkspace;
    if (timeline.batchWorkspaces && typeof timeline.batchWorkspaces === "object") {
        const cleaned = {};
        for (const [key, ws] of Object.entries(timeline.batchWorkspaces)) {
            const safe = sanitizeBatchWorkspace(ws);
            if (safe) cleaned[key] = safe;
        }
        timeline.batchWorkspaces = cleaned;
    }
    if (timeline.videoWorkspaces && typeof timeline.videoWorkspaces === "object") {
        const cleaned = {};
        for (const [key, ws] of Object.entries(timeline.videoWorkspaces)) {
            const safe = sanitizeVideoWorkspace(ws);
            if (safe) cleaned[key] = safe;
        }
        timeline.videoWorkspaces = cleaned;
    }
    // Shallow-cloned payloads still share nested refs with live state — reassign, don't mutate.
    if (Array.isArray(timeline.segments)) {
        timeline.segments = timeline.segments.map(sanitizeSegmentForPayload);
    }
    if (timeline.global && typeof timeline.global === "object") {
        timeline.global = {
            ...timeline.global,
            refs: Array.isArray(timeline.global.refs)
                ? timeline.global.refs.map(sanitizeRefImage)
                : [],
            refAudios: Array.isArray(timeline.global.refAudios)
                ? timeline.global.refAudios.map(sanitizeRefAudio)
                : [],
            refVideos: Array.isArray(timeline.global.refVideos)
                ? timeline.global.refVideos.map(sanitizeRefVideo)
                : (Array.isArray(timeline.global.ref_videos)
                    ? timeline.global.ref_videos.map(sanitizeRefVideo)
                    : []),
            referenceVideo: timeline.global.referenceVideo
                ? {
                    videoFile: timeline.global.referenceVideo.videoFile || "",
                    fileName: timeline.global.referenceVideo.fileName || "",
                    type: timeline.global.referenceVideo.type || "input",
                    subfolder: timeline.global.referenceVideo.subfolder || "",
                }
                : timeline.global.referenceVideo,
        };
    }
    if (timeline.video && typeof timeline.video === "object") {
        const hasFile = !!(timeline.video.videoFile || timeline.video.fileName);
        const dropFrames = hasFile
            || (Array.isArray(timeline.video.frames) && timeline.video.frames.length > 8);
        timeline.video = {
            ...timeline.video,
            frames: dropFrames ? [] : (timeline.video.frames || []),
        };
    }
    if (Array.isArray(timeline.videoClips)) {
        timeline.videoClips = timeline.videoClips.map((clip) => {
            if (!clip || typeof clip !== "object") return clip;
            const dropFrames = !!(clip.videoFile || clip.fileName)
                || (Array.isArray(clip.frames) && clip.frames.length > 8);
            return dropFrames ? { ...clip, frames: [] } : { ...clip };
        });
    }
}

const HIDDEN_WIDGETS = [
    "timeline_data", "total_frames", "width", "height", "ref_max_size",
    "task_type", "global_prompt", "frame_rate", "cfg",
    "export_source_images",
    // seed stays visible under 采样设置 (with control_after_generate)
];

const DIRECTOR_WIDGET_LABEL_KEYS = {
    seed: "widget.seed",
    clear_vram_between_segments: "widget.clearVram",
    export_source_images: "widget.exportSourceImages",
    control_after_generate: "widget.controlAfterGenerate",
    "control after generate": "widget.controlAfterGenerate",
};

const DIRECTOR_WIDGET_TOOLTIP_KEYS = {
    clear_vram_between_segments: "widget.tooltip.clearVram",
    export_source_images: "widget.tooltip.exportSourceImages",
};

const DIRECTOR_GROUP_LABEL_KEYS = {
    bd_grp_sample: "widget.grpSample",
    bd_grp_advanced: "widget.grpAdvanced",
    bd_grp_perf: "widget.grpPerf",
};

function widgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

const DIRECTOR_SAMPLE_VALUE_WIDGETS = [
    "steps",
    "sampler",
    "scheduler",
    "shift_video",
    "shift_audio",
    "cfg",
    "seed",
];

/** Scheduler names that are never valid KSampler sampler ids. */
const SCHEDULER_ONLY_NAMES = new Set([
    "simple",
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "kl_optimal",
]);

function hiddenWidgetSize() {
    return [0, -4];
}

function hiddenWidgetDraw() {}

function isValidSampleWidgetValue(name, widget, value) {
    if (value === undefined || value === null || value === "") return false;
    if (name === "sampler" || name === "scheduler") {
        if (typeof value !== "string") return false;
        const opts = widget?.options?.values;
        if (Array.isArray(opts) && opts.length && !opts.includes(value)) return false;
        if (name === "sampler" && SCHEDULER_ONLY_NAMES.has(value)) return false;
        return true;
    }
    const num = typeof value === "number" ? value : Number(value);
    return Number.isFinite(num);
}

function snapshotDirectorSampleWidgets(node, { force = false, includeHidden = false } = {}) {
    if (!node || node._mmxRestoringSampleWidgets || node._mmxLockSampleSnap) return;
    const snap = { ...(node._mmxSampleWidgetSnap || {}) };
    for (const name of DIRECTOR_SAMPLE_VALUE_WIDGETS) {
        const w = widgetByName(node, name);
        if (!w || w.value === undefined) continue;
        if (!includeHidden && w._mmxForceHidden) continue;
        if (!force && !isValidSampleWidgetValue(name, w, w.value)) continue;
        snap[name] = w.value;
    }
    node._mmxSampleWidgetSnap = snap;
}

function restoreDirectorSampleWidgets(node) {
    const snap = node?._mmxSampleWidgetSnap;
    if (!snap || !node) return;
    node._mmxRestoringSampleWidgets = true;
    try {
        for (const name of DIRECTOR_SAMPLE_VALUE_WIDGETS) {
            const w = widgetByName(node, name);
            if (!w || snap[name] === undefined) continue;
            if (w.value !== snap[name]) w.value = snap[name];
        }
    } finally {
        node._mmxRestoringSampleWidgets = false;
    }
}

function lockDirectorSampleSnap(node, ms = 250) {
    if (!node) return;
    node._mmxLockSampleSnap = true;
    clearTimeout(node._mmxUnlockSampleSnapTimer);
    node._mmxUnlockSampleSnapTimer = setTimeout(() => {
        node._mmxLockSampleSnap = false;
    }, ms);
}

function hookDirectorSampleWidgetSnapshots(node) {
    if (!node) return;
    for (const name of DIRECTOR_SAMPLE_VALUE_WIDGETS) {
        const w = widgetByName(node, name);
        if (!w || w._mmxSnapPatched) continue;
        w._mmxSnapPatched = true;
        const prev = w.callback;
        w.callback = function (...args) {
            const r = prev?.apply(this, args);
            snapshotDirectorSampleWidgets(node);
            return r;
        };
    }
}

function lockDirectorSigmasInput(node) {
    const inp = (node?.inputs || []).find((i) => String(i.name) === "sigmas");
    if (!inp) return;
    inp.tooltip = t("widget.tooltip.sigmas");
    if (inp._mmxSigmasLocked) return;
    inp._mmxSigmasLocked = true;
    try {
        Object.defineProperty(inp, "widget", {
            get() { return undefined; },
            set() { /* socket-only: never convert SIGMAS to a widget */ },
            configurable: true,
            enumerable: true,
        });
    } catch {
        inp.widget = undefined;
    }
}

function hideDirectorSigmasWidget(node) {
    const w = widgetByName(node, "sigmas");
    if (!w) return;
    if (typeof w.computeSize === "function"
        && w.computeSize !== hiddenWidgetSize
        && !w._mmxOrigComputeSize) {
        w._mmxOrigComputeSize = w.computeSize.bind(w);
    }
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    w.computeSize = hiddenWidgetSize;
    w.draw = hiddenWidgetDraw;
    if (w.element) w.element.style.display = "none";
}

function setDirectorWidgetVisible(node, name, visible) {
    const w = widgetByName(node, name);
    if (!w) return;
    if (typeof w.computeSize === "function"
        && w.computeSize !== hiddenWidgetSize
        && !w._mmxOrigComputeSize) {
        w._mmxOrigComputeSize = w.computeSize.bind(w);
    }
    const hide = !visible;
    if (!w.options) w.options = {};
    w.hidden = hide;
    w.options.hidden = hide;
    w._mmxForceHidden = hide;
    if (hide) {
        if (!w._mmxOrigDraw && typeof w.draw === "function" && w.draw !== hiddenWidgetDraw) {
            w._mmxOrigDraw = w.draw;
        }
        w.computeSize = hiddenWidgetSize;
        w.draw = hiddenWidgetDraw;
        if (w.element) w.element.style.display = "none";
    } else {
        if (w._mmxOrigComputeSize) w.computeSize = w._mmxOrigComputeSize;
        else if (w.computeSize === hiddenWidgetSize) delete w.computeSize;
        if (w._mmxOrigDraw) w.draw = w._mmxOrigDraw;
        else if (w.draw === hiddenWidgetDraw) delete w.draw;
        if (w.element) w.element.style.display = "";
    }
}

function directorHasSigmasLink(node) {
    const inp = (node?.inputs || []).find((i) => String(i.name) === "sigmas");
    if (!inp) return false;
    if (inp.link != null) return true;
    return Array.isArray(inp.links) && inp.links.length > 0;
}

function syncDirectorSchedulerWidgets(node, { restore = false } = {}) {
    if (!node) return;
    hookDirectorSampleWidgetSnapshots(node);
    lockDirectorSigmasInput(node);
    hideDirectorSigmasWidget(node);
    if (restore) restoreDirectorSampleWidgets(node);
    const wired = directorHasSigmasLink(node);
    setDirectorWidgetVisible(node, "steps", !wired);
    setDirectorWidgetVisible(node, "scheduler", !wired);
    if (restore) restoreDirectorSampleWidgets(node);
    node.setDirtyCanvas?.(true, true);
}

function settleDirectorSampleWidgets(node) {
    syncDirectorSchedulerWidgets(node, { restore: true });
    queueMicrotask(() => {
        restoreDirectorSampleWidgets(node);
        syncDirectorSchedulerWidgets(node, { restore: true });
    });
    setTimeout(() => {
        restoreDirectorSampleWidgets(node);
        syncDirectorSchedulerWidgets(node, { restore: true });
    }, 0);
}

function applyDirectorWidgetLabels(node) {
    for (const w of node.widgets || []) {
        const name = String(w.name || "");
        const key = DIRECTOR_WIDGET_LABEL_KEYS[name]
            || (/(control[_\s]?after[_\s]?generate|生成前后)/i.test(name) || /生成前后/.test(String(w.label || ""))
                ? "widget.controlAfterGenerate"
                : null);
        if (key) {
            w.label = t(key);
            if (w.options) w.options.label = w.label;
        }
        const tipKey = DIRECTOR_WIDGET_TOOLTIP_KEYS[name];
        if (tipKey && w.options) w.options.tooltip = t(tipKey);
        const gKey = DIRECTOR_GROUP_LABEL_KEYS[name] || w._mmxGroupI18nKey;
        if (gKey) {
            const label = t(gKey);
            w._mmxGroupI18nKey = gKey;
            w._bdGroupLabel = label;
            w.value = label;
            if (w.element) w.element.textContent = label;
        }
        // Linked seed → control_after_generate combo (ComfyUI core).
        for (const linked of w.linkedWidgets || []) {
            const ln = String(linked?.name || linked?.label || "");
            if (/(control[_\s]?after[_\s]?generate|生成前后)/i.test(ln)) {
                linked.label = t("widget.controlAfterGenerate");
                if (linked.options) linked.options.label = linked.label;
            }
        }
    }
    syncDirectorSchedulerWidgets(node);
}

function drawGroupHeader(ctx, node, widget_width, y, H, label) {
    const margin = 10;
    const barH = Math.max(18, H - 4);
    ctx.fillStyle = "#2e2e2e";
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) {
        ctx.roundRect(margin, y + 2, widget_width - margin * 2, barH, 4);
    } else {
        ctx.rect(margin, y + 2, widget_width - margin * 2, barH);
    }
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#d8dce8";
    ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, margin + 10, y + 2 + barH / 2);
}

function makeGroupHeaderWidget(inputName, inputData) {
    const opts = inputData?.[1] || {};
    const i18nKey = DIRECTOR_GROUP_LABEL_KEYS[inputName];
    const label = i18nKey ? t(i18nKey) : (opts.default || opts.label || inputName);
    const el = document.createElement("div");
    el.className = "bd-widget-group";
    el.textContent = label;
    el.style.cssText = [
        "width:100%;box-sizing:border-box;margin:8px 0 4px;padding:6px 10px",
        "border:1px solid #555;border-left:3px solid #7a9cff;border-radius:4px",
        "color:#d8dce8;font-size:11px;font-weight:600;letter-spacing:.02em",
        "background:linear-gradient(180deg,#2e2e2e 0%,#242424 100%)",
        "pointer-events:none;user-select:none",
    ].join(";");
    return {
        name: inputName,
        type: "BDGROUP",
        value: label,
        label: "",
        element: el,
        options: opts,
        _bdGroupHeader: true,
        _mmxGroupI18nKey: i18nKey || null,
        _bdGroupLabel: label,
        draw(ctx, node, widget_width, y, H) {
            const text = this._mmxGroupI18nKey ? t(this._mmxGroupI18nKey) : (this._bdGroupLabel || label);
            drawGroupHeader(ctx, node, widget_width, y, H, text);
        },
        computeSize(width) {
            return [width, 26];
        },
        mouse() {
            return false;
        },
    };
}

const STYLES = `
/* min-height = content only; height:100% fills LiteGraph free space without raising
   getMinHeight (avoids Vue-node ResizeObserver feedback growth).
   overflow:hidden keeps run-status from painting past the node bottom edge. */
.mmx-host{width:100%;box-sizing:border-box;display:flex;flex-direction:column;min-height:var(--comfy-widget-min-height,0px);height:100%;max-height:100%;overflow:hidden}
/* Default: fill allocated box. Batch-fill mode stretches list into leftover space. */
.bd-wrap{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;color:#e0e0e0;font-size:11px;display:flex;flex-direction:column;gap:6px;width:100%;box-sizing:border-box;position:relative;min-height:0;height:100%;flex:1 1 auto;overflow:hidden}
.bd-wrap.bd-batch-fill{height:100%!important;min-height:0!important;max-height:100%;flex:1 1 0;overflow:hidden}
.bd-main{flex:0 1 auto;min-height:0;display:flex;flex-direction:column;gap:6px;width:100%}
/*
 * Batch is inside .bd-main (sibling of .bd-split which holds 公共参数).
 * Main grows with the node; .bd-split may shrink/scroll so .bd-batch always keeps space.
 */
.bd-wrap.bd-batch-fill .bd-main{flex:1 1 0;min-height:0;overflow:hidden}
.bd-wrap.bd-batch-fill .bd-main>:not(.bd-batch):not(.bd-split){flex:0 0 auto}
/* 公共参数区：可收缩+内部滚动，避免展开后把素材组挤出视口 */
.bd-wrap.bd-batch-fill .bd-main>.bd-split{
  flex:0 1 auto;min-height:0;max-height:42%;overflow:auto;width:100%
}
.bd-wrap.bd-batch-fill .bd-main>.bd-batch:not(.hidden){
  flex:1 1 0;min-height:0;overflow:hidden;display:flex;flex-direction:column
}
.bd-wrap.bd-batch-fill .bd-batch-toolbar,.bd-wrap.bd-batch-fill .bd-batch-i2v-notice,.bd-wrap.bd-batch-fill .bd-batch-picker{flex-shrink:0}
.bd-wrap.bd-batch-fill .bd-batch-list{
  flex:1 1 0;min-height:0;max-height:none!important;overflow-y:auto;height:auto;
  display:flex;flex-direction:column
}
.bd-wrap.bd-batch-fill .bd-run-status{flex:0 0 auto;margin-top:0;flex-shrink:0}
/* Fixed min so progress text wrap does not change node chrome height every tick. */
.bd-run-status{min-height:52px;box-sizing:border-box}
/* Solo material group fills the viewport by default, but may grow beyond it when
   the user drags the rich prompt editor. The list then scrolls instead of
   clipping the editor or forcing its height back to auto. */
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card{flex:1 1 auto;min-height:0;align-self:stretch}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-r2v{
  display:flex;flex-direction:column;flex:0 0 auto!important;min-height:100%;height:auto!important
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-r2v-body{
  flex:0 0 auto;min-height:280px;max-height:none;align-self:stretch
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-r2v-main{
  flex:0 0 auto;min-height:0;height:auto;max-height:none
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-prompts{
  flex:0 0 auto;min-height:140px;max-height:none;overflow:visible
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-token-wrap{
  flex:0 0 auto;min-height:120px;max-height:none;height:auto;overflow:visible
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-token-editor{
  flex:0 0 auto;min-height:120px;max-height:none;height:360px;overflow:auto;resize:vertical
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-plain,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-source,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){
  /* Header stays compact; leftover height goes to the prompt row (not a blank gap). */
  grid-template-rows:auto minmax(0,1fr);
  align-content:stretch
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-plain .bd-batch-head,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-source .bd-batch-head,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-head{align-self:start}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-plain .bd-batch-prompts,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-source .bd-batch-prompts,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts{
  height:100%;min-height:0;max-height:100%;overflow:hidden;align-self:stretch
}
.bd-modal-overlay{position:absolute;inset:0;z-index:200;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:10px;box-sizing:border-box;border-radius:6px}
.bd-modal{background:#1e1e1e;border:1px solid #333;border-radius:6px;padding:12px;width:100%;max-width:460px;max-height:calc(100% - 8px);display:flex;flex-direction:column;gap:10px;box-shadow:0 10px 28px rgba(0,0,0,.5)}
.bd-modal-title{color:#e0e0e0;font-size:12px;font-weight:600;line-height:1.35}
.bd-modal-body{color:#aaa;font-size:11px;line-height:1.5;white-space:pre-wrap}
.bd-modal-body.hidden{display:none}
.bd-modal-list{flex:1;min-height:140px;max-height:240px;overflow:auto;background:#181818;border:1px solid #333;border-radius:6px;padding:4px;display:flex;flex-direction:column;gap:2px}
.bd-modal-list.hidden{display:none}
.bd-modal-item{padding:7px 8px;border-radius:4px;cursor:pointer;color:#ccc;font-size:11px;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border:1px solid transparent}
.bd-modal-item:hover{background:#252525;color:#eee}
.bd-modal-item.selected{background:#2a2a2a;border-color:#4fff8f;color:#fff}
.bd-modal-actions{display:flex;gap:8px;justify-content:flex-end;flex-shrink:0}
.bd-media-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
.bd-media-head .bd-modal-title{flex:1;min-width:0;padding-top:4px}
.bd-media-head-actions{display:flex;gap:8px;flex-shrink:0;flex-wrap:wrap;justify-content:flex-end}
.bd-media-status{color:#999;font-size:11px;line-height:1.4;min-height:15px}
.bd-media-modal{max-width:860px}
.bd-media-modal .bd-btn-primary{background:#2f9e44;border-color:#3cb054;color:#fff}
.bd-media-modal .bd-btn-primary:hover{background:#38b04a;border-color:#4bc45c;color:#fff}
.bd-media-body{display:grid;grid-template-columns:minmax(320px,1.2fr) minmax(240px,.8fr);gap:10px;min-height:280px}
.bd-media-left,.bd-media-right{display:flex;flex-direction:column;gap:8px;min-width:0}
.bd-media-table{width:100%;min-height:220px;max-height:320px;background:#141414;border:1px solid #333;border-radius:6px;color:#eee;box-sizing:border-box;flex:1;display:flex;flex-direction:column;overflow:hidden;outline:none}
.bd-media-thead{display:grid;grid-template-columns:minmax(0,1fr) 86px 128px;flex-shrink:0;border-bottom:1px solid #333;background:#1a1a1a}
.bd-media-th{appearance:none;background:transparent;border:none;color:#8e8e8e;font-size:11px;text-align:left;padding:8px 10px;cursor:pointer;display:flex;align-items:center;gap:5px;min-width:0;font-family:inherit;line-height:1.35}
.bd-media-th:hover{color:#ddd}
.bd-media-th.is-active{color:#d8d8d8}
.bd-media-sort{display:inline-block;width:0;height:0;border-left:4px solid transparent;border-right:4px solid transparent;opacity:0;flex-shrink:0}
.bd-media-th.is-active .bd-media-sort{opacity:1;border-top:5px solid #6ea8ff;border-bottom:0}
.bd-media-th.is-active.is-asc .bd-media-sort{border-top:0;border-bottom:5px solid #6ea8ff}
.bd-media-tbody{flex:1;overflow:auto;min-height:0}
.bd-media-tr{display:grid;grid-template-columns:minmax(0,1fr) 86px 128px;align-items:center;cursor:pointer;border-bottom:1px solid #262626;color:#ddd;font-size:11px;line-height:1.35}
.bd-media-table.bd-media-nodims .bd-media-thead,
.bd-media-table.bd-media-nodims .bd-media-tr{grid-template-columns:minmax(0,1fr) 128px}
.bd-media-tr:hover{background:#222}
.bd-media-tr.selected{background:#2c2c2c}
.bd-media-td{padding:8px 10px;min-width:0}
.bd-media-td-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#eee}
.bd-media-td-dims,.bd-media-td-time{color:#9a9a9a;white-space:nowrap;font-variant-numeric:tabular-nums}
.bd-media-empty-row{padding:18px 10px;color:#666;font-size:11px;text-align:center}
.bd-media-preview{flex:1;min-height:220px;background:#111;border:1px solid #333;border-radius:6px;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
.bd-media-preview img,.bd-media-preview video{display:block;width:100%;height:100%;object-fit:contain;background:#000}
.bd-media-preview-empty{padding:18px;color:#666;font-size:11px;line-height:1.45;text-align:center}
.bd-media-meta{display:flex;flex-direction:column;gap:4px;color:#9a9a9a;font-size:10px;line-height:1.45;word-break:break-all}
.bd-toolbar-wrap{display:flex;flex-direction:column;gap:4px;width:100%}
.bd-toolbar{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;width:100%}
.bd-actions{display:flex;gap:6px;flex-wrap:wrap;align-items:center;flex:1;min-width:0}
.bd-smart-split-msg{width:100%;box-sizing:border-box;font-size:11px;line-height:1.4;color:#f66;padding:0 2px;min-height:0}
.bd-smart-split-msg.hidden{display:none!important}
.bd-smart-split-msg.ok{color:#8c8}
.bd-external-groups-msg{width:100%;box-sizing:border-box;font-size:11px;line-height:1.45;color:#9ad;padding:8px 10px;margin:0 0 4px;background:#152018;border:1px solid #2f4a38;border-radius:6px}
.bd-external-groups-msg.hidden{display:none!important}
.bd-wrap.bd-external-groups .bd-batch-card,.bd-wrap.bd-external-groups .bd-fl2v-shot{opacity:.48;pointer-events:none}
.bd-wrap.bd-external-groups .bd-run-select-bar,.bd-wrap.bd-external-groups .bd-batch-run-check,.bd-wrap.bd-external-groups .bd-run-select-all-wrap{pointer-events:auto;opacity:1}
.bd-wrap.bd-external-groups .bd-batch-card .bd-batch-run-check{pointer-events:auto;opacity:1}
/* External mode: duration/delete stay non-interactive; allow media preview playback. */
.bd-wrap.bd-external-groups .bd-batch-del,.bd-wrap.bd-external-groups .bd-batch-fc input{pointer-events:none!important;opacity:.55}
.bd-wrap.bd-external-groups .bd-r2v-play,.bd-wrap.bd-external-groups .bd-batch-video video,.bd-wrap.bd-external-groups .bd-batch-audio audio,.bd-wrap.bd-external-groups .bd-r2v-thumb{pointer-events:auto;opacity:1}
.bd-stage{width:100%;box-sizing:border-box;background:#0c0c0c;border:1px solid #222;border-bottom:none;border-radius:6px 6px 0 0;overflow:hidden;position:relative;min-height:120px;max-height:280px;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center}
.bd-stage.hidden{display:none!important}
.bd-stage-video,.bd-stage-img{width:100%;height:100%;max-height:280px;object-fit:contain;background:#000;display:block}
.bd-stage-img.hidden,.bd-stage-video.hidden{display:none!important}
.bd-stage-empty{color:#555;font-size:11px;pointer-events:none}
.bd-stage-badge{position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:3px;background:rgba(0,0,0,.65);color:#ccc;font-size:10px;line-height:1.4;cursor:pointer;user-select:none}
.bd-stage-badge:hover{color:#fff;background:rgba(0,0,0,.8)}
.bd-frame-jump{display:inline-flex;align-items:center;gap:4px;color:#ddd;font-size:11px;white-space:nowrap;font-variant-numeric:tabular-nums}
.bd-frame-jump .bd-frame-input{width:64px;background:#181818;border:1px solid #444;border-radius:4px;color:#eee;padding:4px 4px;font-size:11px;text-align:center;-moz-appearance:textfield}
.bd-frame-jump .bd-frame-input:focus{border-color:#4fff8f;outline:none}
.bd-frame-jump .bd-frame-input::-webkit-outer-spin-button,.bd-frame-jump .bd-frame-input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.bd-frame-jump .bd-frame-total{color:#888;min-width:2.5em}
.bd-controls{width:100%;box-sizing:border-box;background:#151515;border:1px solid #222;border-radius:0 0 6px 6px;padding:8px 10px;margin-top:0;flex-shrink:0}
.bd-stage.hidden+.bd-controls{border-radius:6px;border-color:#333;background:#1e1e1e}
.bd-viewport{width:100%;max-width:100%;min-width:0;overflow-x:hidden;overflow-y:hidden;border-radius:6px;border:1px solid #111;background:#2a2a2a;box-sizing:border-box;flex-shrink:0}
.bd-viewport.bd-zoomed{overflow-x:auto;scrollbar-width:thin;scrollbar-color:#555 #1a1a1a}
.bd-viewport.bd-zoomed::-webkit-scrollbar{height:10px}
.bd-viewport.bd-zoomed::-webkit-scrollbar-track{background:#1a1a1a;border-radius:5px}
.bd-viewport.bd-zoomed::-webkit-scrollbar-thumb{background:#555;border-radius:5px}
.bd-viewport.bd-zoomed::-webkit-scrollbar-thumb:hover{background:#777}
/* object-fit:fill + mismatched CSS/bitmap aspect stretches thumbs (esp. under graph zoom). */
.bd-canvas{display:block;width:100%;min-width:100%;height:auto;cursor:pointer;box-sizing:border-box;flex-shrink:0;object-fit:fill}
.bd-canvas.bd-grab{cursor:grab}
.bd-canvas.bd-grabbing{cursor:grabbing}
.bd-output{width:100%;box-sizing:border-box;display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:6px 8px;background:#1e1e1e;border:1px solid #333;border-radius:6px}
.bd-out-audio-wrap,.bd-out-source-wrap{display:inline-flex;align-items:center;gap:6px}
.bd-out-source-wrap.hidden{display:none}
.bd-output .bd-out-source-wrap label{display:inline-flex;align-items:center;gap:4px;cursor:pointer}
.bd-split{display:block;width:100%;box-sizing:border-box;min-width:0}
.bd-r2v-common-hint{margin:0 0 8px;font-size:11px;line-height:1.4;color:#9ab;opacity:.95}
.bd-panel.bd-r2v-common-panel{border:1px solid #3a4a5a;background:linear-gradient(180deg,#1a222c 0%,#151a20 100%)}
.bd-r2v-common-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 0 8px}
.bd-r2v-common-titles{display:flex;flex-direction:column;gap:2px;min-width:0;flex:1}
.bd-r2v-common-titles b{margin:0}
.bd-r2v-common-status{display:none;font-size:11px;color:#8a9}
.bd-r2v-common-status.on{color:#8fdfb0}
.bd-panel.bd-r2v-common-panel .bd-r2v-common-status{display:inline}
.bd-r2v-common-actions{display:none;align-items:center;gap:8px;flex:0 0 auto}
.bd-panel.bd-r2v-common-panel .bd-r2v-common-actions{display:flex}
.bd-btn.bd-r2v-common-toggle,.bd-btn.bd-r2v-common-fold{display:inline-block;flex:0 0 auto;padding:5px 10px;font-size:12px;border-radius:6px;border:1px solid #4a6a8a;background:#243040;color:#d8e6f5;cursor:pointer}
.bd-btn.bd-r2v-common-fold{border-color:#3a4a5a;background:#1c2430}
.bd-btn.bd-r2v-common-toggle:hover,.bd-btn.bd-r2v-common-fold:hover{border-color:#6a9aca;background:#2c3c50}
.bd-btn.bd-r2v-common-toggle.on{border-color:#7a3a3a;background:#301a1a;color:#f0c0c0}
.bd-btn.bd-r2v-common-toggle.on:hover{border-color:#a05050;background:#3a2020}
.bd-panel.bd-r2v-common-panel.bd-r2v-common-collapsed{padding-bottom:10px}
.bd-panel.bd-r2v-common-panel.bd-r2v-common-collapsed .bd-r2v-common-body{display:none!important}
.bd-panel.bd-r2v-common-panel .bd-r2v-common-body{min-width:0}
.bd-panel.bd-r2v-common-panel .bd-refs-col{height:auto;min-height:0}
.bd-panel.bd-r2v-common-panel .bd-rv2v-layout .bd-ref{min-height:72px}
.bd-panel.bd-r2v-common-panel .bd-rv2v-layout .bd-ref-audio{min-height:44px}
.bd-panel.bd-r2v-common-panel .bd-rv2v-layout .bd-ref-video{min-height:0}
.bd-player{display:flex;align-items:center;gap:10px;flex-wrap:wrap;width:100%}
.bd-btn{background:#222;color:#e0e0e0;border:1px solid #111;border-radius:4px;padding:6px 12px;font-size:11px;line-height:1.35;box-sizing:border-box;cursor:pointer;display:inline-flex;align-items:center}
.bd-actions>.bd-btn{height:29px;min-height:29px}
.bd-btn:hover{background:#333;border-color:#555}
.bd-btn-danger:hover{background:#4a1515;border-color:#c44;color:#faa}
.bd-split-edit-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;width:100%;box-sizing:border-box;padding:6px 10px;margin:0 0 4px;background:#241818;border:1px solid #633;border-radius:6px}
.bd-split-edit-bar.hidden{display:none!important}
.bd-split-edit-bar .bd-split-edit-hint{flex:1;min-width:140px;font-size:11px;line-height:1.35;color:#f88}
.bd-btn-del-split{background:#3a2020;border-color:#e66;color:#f88}
.bd-btn-del-split:hover{background:#4a1515;border-color:#f88;color:#fcc}
.bd-btn-sm{padding:3px 8px;font-size:10px}
.bd-btn-run-select.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f}
.bd-output .bd-btn-live-preview{margin-left:auto;background:#222;border-color:#333;color:#aaa;white-space:nowrap;height:29px;min-height:29px;padding:4px 12px}
.bd-output .bd-btn-live-preview:hover{background:#2a2a2a;border-color:#555;color:#ddd}
.bd-output .bd-btn-live-preview.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-live-sample{width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:8px;padding:10px 12px;background:linear-gradient(165deg,#1a1a1a 0%,#121212 100%);border:1px solid #333;border-radius:10px;flex-shrink:0}
.bd-live-sample.hidden{display:none!important}
.bd-live-sample.receiving{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-live-sample-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap}
.bd-live-sample-head b{color:#f0f0f0;font-size:12px;font-weight:650;letter-spacing:.02em}
.bd-live-sample-head .bd-meta{color:#888;font-size:11px}
.bd-live-sample-body{position:relative;width:100%;min-height:220px;max-height:360px;flex:1 1 auto;background:#0a0a0a;border:1px solid #262626;border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center}
.bd-live-sample-body img{width:100%;height:100%;max-width:100%;max-height:360px;object-fit:contain;display:block}
.bd-live-sample-body img.hidden{display:none!important}
.bd-live-sample-empty{color:#666;font-size:12px;text-align:center;padding:16px;line-height:1.45}
.bd-live-sample-empty.hidden{display:none!important}
.bd-live-sample-badge{position:absolute;left:10px;bottom:10px;padding:3px 8px;border-radius:999px;background:rgba(0,0,0,.75);color:#cfcfcf;font-size:11px;pointer-events:none}
.bd-live-sample-badge.hidden{display:none!important}
.bd-main>.bd-live-sample{margin:0 0 4px}
.bd-run-select-bar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:10px;color:#aaa}
.bd-run-select-all-wrap{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#aaa;cursor:pointer;user-select:none;margin-left:2px}
.bd-run-select-all-wrap.hidden{display:none!important}
.bd-run-select-all-wrap input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
.bd-run-select-bar.hidden{display:none!important}
.bd-batch-run-check{margin-right:6px;width:14px;height:14px;cursor:pointer;accent-color:#4fff8f;flex-shrink:0}
.bd-btn-primary{background:#1a3a2a;border-color:#4fff8f;color:#4fff8f}
.bd-mode{display:flex;border:1px solid #333;border-radius:4px;overflow:hidden}
.bd-mode button{border:none;background:#222;color:#aaa;padding:6px 12px;font-size:11px;cursor:pointer}
.bd-mode button.active{background:#333;color:#fff}
.bd-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap;flex-shrink:0}
.bd-tl-zoom{display:inline-flex;align-items:center;gap:6px;flex-shrink:0}
.bd-tl-zoom.hidden{display:none!important}
.bd-btn.bd-btn-zoom.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-tl-zoom-slider{width:128px;height:18px;margin:0;accent-color:#4fff8f;cursor:pointer;flex-shrink:0;touch-action:none}
.bd-bounds,.bd-timecode{color:#aaa;font-size:11px}
.bd-timecode{color:#fff;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}
.bd-player .bd-timecode{min-width:88px;font-size:11px;color:#ddd}
.bd-icon-btn{background:#2a2a2a;border:1px solid #444;color:#eee;cursor:pointer;padding:6px 10px;border-radius:4px}
.bd-icon-btn.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-seek{flex:1;min-width:120px;height:6px}
.bd-panel{width:100%;box-sizing:border-box;background:#222;border:1px solid #111;border-radius:6px;padding:8px;display:flex;flex-direction:column;gap:6px}
.bd-panel.bd-rv2v-panel,.bd-panel.bd-v2v-panel{background:linear-gradient(165deg,#1c1c1c 0%,#141414 52%,#111 100%);border:1px solid #2c2c2c;border-radius:12px;padding:12px 14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.035);gap:10px}
.bd-panel.bd-rv2v-panel>b,.bd-panel.bd-v2v-panel>b,.bd-seg-head>b{color:#f0f0f0;font-size:13px;font-weight:650;letter-spacing:.02em}
.bd-seg-head{display:flex;align-items:center;justify-content:flex-start;gap:10px;flex-wrap:wrap;min-width:0}
.bd-seg-head>b{flex-shrink:0;margin:0}
.bd-seg-refsize{display:inline-flex;align-items:center;gap:6px;color:#c8c8c8;font-size:11px;white-space:nowrap;margin-left:auto;flex-shrink:0}
.bd-seg-refsize select{max-width:88px}
.bd-seg-continuity{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#9ab;cursor:pointer;user-select:none;flex-shrink:0}
.bd-seg-continuity input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#6ab0ff}
.bd-seg-head .bd-meta,.bd-panel.bd-v2v-panel .bd-seg-head .bd-meta,.bd-panel.bd-rv2v-panel .bd-seg-head .bd-meta{color:#8a8a8a;font-size:11px;line-height:1.45;padding:0;min-width:0}
.bd-prompt-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(110px,38%);gap:8px;align-items:stretch}
.bd-prompt-layout>.bd-prompt-col{order:1}
.bd-prompt-layout>.bd-refs-col{order:2}
.bd-prompt-layout.bd-rv2v-layout{grid-template-columns:minmax(240px,.85fr) minmax(0,1.4fr);gap:12px}
.bd-prompt-layout.bd-rv2v-layout>.bd-refs-col{order:1}
.bd-prompt-layout.bd-rv2v-layout>.bd-prompt-col{order:2}
/* rv2v live preview: under prompt (same stack as r2v right column) */
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live>.bd-prompt-col{gap:10px}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live>.bd-prompt-col .bd-prompt{min-height:160px;flex:1 1 auto}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live>.bd-prompt-col>.bd-live-sample{margin:0;padding:8px 10px;min-height:0;flex:0 0 auto;border-radius:10px}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live .bd-live-sample-body{min-height:180px;max-height:280px}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live .bd-live-sample-body img{width:100%;max-height:280px;object-fit:contain}
.bd-prompt-layout.bd-v2v-layout{grid-template-columns:1fr;gap:0}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live{grid-template-columns:minmax(0,1.25fr) minmax(240px,.9fr);gap:12px;align-items:stretch}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-prompt-col{order:1;min-height:220px}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-prompt-col .bd-prompt{flex:1 1 auto;min-height:180px}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-refs-col{display:none!important}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-live-sample{order:2;margin:0;height:100%;min-height:220px;align-self:stretch}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live .bd-live-sample-body{flex:1 1 auto;min-height:180px;max-height:none}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live .bd-live-sample-body img{max-height:100%}
.bd-prompt-col{display:flex;flex-direction:column;gap:5px;min-width:0}
.bd-rv2v-layout .bd-prompt-col,.bd-v2v-layout .bd-prompt-col{background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;gap:6px;min-height:220px}
.bd-v2v-layout .bd-prompt-col{min-height:200px}
.bd-prompt-col .bd-label,.bd-refs-col .bd-label{color:#888;font-size:10px;line-height:1.2;flex-shrink:0}
.bd-rv2v-layout .bd-prompt-col .bd-label,.bd-v2v-layout .bd-prompt-col .bd-label{color:#eaeaea;font-size:11px;font-weight:700;letter-spacing:.02em}
.bd-wrap.locale-en .bd-rv2v-layout .bd-prompt-col .bd-label,.bd-wrap.locale-en .bd-v2v-layout .bd-prompt-col .bd-label{text-transform:uppercase;letter-spacing:.08em}
.bd-prompt{width:100%;min-height:96px;background:#181818;border:1px solid #333;border-radius:6px;color:#eee;padding:8px;resize:vertical;font-size:12px;box-sizing:border-box;font-family:inherit;line-height:1.35;flex:1}
.bd-prompt-col .bd-token-wrap{flex:1 1 auto;min-height:96px;width:100%}
.bd-ref.bd-ref-flash,.bd-batch-ref.bd-ref-flash,.bd-ref-audio.bd-ref-flash,.bd-batch-audio.bd-ref-flash,.bd-batch-video.bd-ref-flash{outline:2px solid #4fff8f;outline-offset:1px;border-color:#4fff8f!important}
.bd-rv2v-layout .bd-prompt,.bd-v2v-layout .bd-prompt{min-height:220px;background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45}
.bd-v2v-layout .bd-prompt{min-height:180px}
.bd-prompt-negative{display:none!important}
.bd-refs-col{display:flex;flex-direction:column;gap:4px;min-width:0;height:100%}
.bd-rv2v-layout .bd-refs-col{gap:10px}
.bd-refs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px;width:100%;flex:1;align-content:start}
.bd-rv2v-layout .bd-refs{grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;flex:0}
.bd-ref{position:relative;width:100%;aspect-ratio:1;min-width:0;max-height:64px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;font-size:9px;color:#666;transition:border-color .15s,background .15s}
.bd-rv2v-layout .bd-ref{max-height:none;min-height:0;border-radius:8px;border:1px dashed #333;background:#080808;color:#555;font-size:10px}
.bd-ref.has-img{cursor:grab;border-style:solid}
.bd-rv2v-layout .bd-ref.has-img{border-color:#3a3a3a;background:#000}
.bd-ref.has-img:active{cursor:grabbing}
.bd-ref:hover{border-color:#7a9cff;background:#1a1a1a}
.bd-rv2v-layout .bd-ref:hover{border-color:#5a5a5a;background:#101010}
.bd-ref .bd-ref-tag{position:absolute;inset:auto 0 3px 0;text-align:center;font-size:9px;color:#777;pointer-events:none;line-height:1}
.bd-ref.has-img .bd-ref-tag{display:none}
.bd-rv2v-layout .bd-ref .bd-ref-tag,.bd-rv2v-layout .bd-ref .cap{position:absolute;left:0;right:0;bottom:0;padding:14px 6px 5px;background:linear-gradient(180deg,transparent,rgba(0,0,0,.78));color:#ddd;font-size:10px;font-weight:600;text-align:center;pointer-events:none;z-index:2}
.bd-rv2v-layout .bd-ref:not(.has-img) .bd-ref-tag,.bd-rv2v-layout .bd-ref:not(.has-img) .cap{position:static;padding:0;background:none;color:#666;font-weight:500}
.bd-rv2v-layout .bd-ref.has-img .bd-ref-tag{display:block}
.bd-rv2v-layout .bd-ref img{object-fit:contain;object-position:center;background:#000}
.bd-rv2v-layout .bd-ref .dot{position:absolute;left:6px;top:6px;width:7px;height:7px;border-radius:50%;background:#4fff8f;box-shadow:0 0 0 2px rgba(0,0,0,.5);z-index:2}
.bd-rv2v-layout .bd-ref .x{top:4px;right:4px;width:20px;height:20px;border-radius:6px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.72);color:#ff9a9a;font-size:14px;font-weight:700;z-index:3}
.bd-rv2v-layout .bd-ref:hover .x,.bd-rv2v-layout .bd-ref:focus-within .x{display:flex}
.bd-rv2v-layout .bd-ref.bd-r2v-pic-hidden{display:none!important}
.bd-rv2v-layout .bd-refs-images-wrap,.bd-rv2v-layout .bd-ref-audios-wrap,.bd-rv2v-layout .bd-ref-videos-wrap{margin-top:0}
.bd-select{background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:4px 6px;font-size:11px;max-width:240px;box-sizing:border-box}
.bd-actions>.bd-select{padding:6px 10px;font-size:11px;line-height:1.35;height:29px;min-height:29px;max-width:min(480px,55vw)}
.bd-ref img{width:100%;height:100%;object-fit:cover}
.bd-ref .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;line-height:1;display:none}
.bd-ref:hover .x{display:block}
.bd-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.bd-meta{color:#888;font-size:10px}
.bd-video-tag{color:#4fff8f;font-size:10px}
.bd-num{width:42px;background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:5px 4px;font-size:11px;text-align:center;-moz-appearance:textfield}
.bd-num::-webkit-outer-spin-button,.bd-num::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.bd-output label{color:#888;font-size:10px;white-space:nowrap}
.bd-output .bd-out-fixed{display:flex;gap:4px;align-items:center}
.bd-output .bd-out-fixed.hidden{display:none}
/* Do not use margin-top:auto — with an oversized min-height it creates a huge empty gap above the status bar. */
.bd-run-status{width:100%;box-sizing:border-box;padding:8px 10px;background:#151515;border:1px solid #333;border-radius:6px;display:flex;flex-direction:column;gap:5px;margin-top:6px;margin-bottom:0;flex-shrink:0}
.bd-run-status.idle .bd-run-title{color:#888}
.bd-run-status.active .bd-run-title{color:#4fff8f}
.bd-run-status.done .bd-run-title{color:#7a9cff}
.bd-run-status.error .bd-run-title{color:#f88}
.bd-run-title{font-size:11px;font-weight:600;line-height:1.35}
.bd-run-detail{color:#999;font-size:10px;line-height:1.4}
.bd-run-bars{display:flex;flex-direction:column;gap:3px}
.bd-run-bar{height:5px;background:#2a2a2a;border-radius:3px;overflow:hidden}
.bd-run-bar-fill{height:100%;background:linear-gradient(90deg,#2a6b4a,#4fff8f);border-radius:3px;transition:width .15s ease}
.bd-run-bar-sub .bd-run-bar-fill{background:linear-gradient(90deg,#3a5080,#7a9cff)}
.hidden{display:none!important}
.bd-controls.hidden{display:none!important}
.bd-gen-src{width:100%;min-height:72px;max-height:100px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;color:#666;font-size:10px;margin-top:4px;position:relative;box-sizing:border-box}
.bd-gen-src.has-img{border-style:solid;border-color:#444}
.bd-gen-src img{width:100%;height:100%;object-fit:contain;background:#000}
.bd-gen-src .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;line-height:1;display:none;cursor:pointer;z-index:2}
.bd-gen-src.has-img:hover .x{display:block}
.bd-gen-src.has-video{padding:0;cursor:default;align-items:stretch;justify-content:flex-start;flex-direction:column}
.bd-gen-src.has-video .bd-ref-video-preview{width:100%;flex:1;min-height:100px;max-height:220px;object-fit:contain;background:#000;display:block;border-radius:3px}
.bd-gen-src .bd-ref-replace{position:absolute;bottom:4px;left:4px;z-index:3;background:rgba(0,0,0,.72);color:#ccc;border:1px solid #555;border-radius:3px;padding:2px 7px;font-size:9px;cursor:pointer;line-height:1.4}
.bd-gen-src .bd-ref-replace:hover{color:#fff;border-color:#888}
.bd-gen-src.has-video .x{display:block;z-index:3}
.bd-ref-video-col{display:flex;flex-direction:column;gap:4px;min-width:0;width:100%;flex:1}
.bd-ref-video-col .bd-gen-src{min-height:140px;max-height:none;flex:1}
.bd-ref-video-name{word-break:break-all;line-height:1.3}
.bd-ref-audios-wrap,.bd-ref-videos-wrap{display:flex;flex-direction:column;gap:4px;margin-top:6px;width:100%}
.bd-ref-audios,.bd-ref-videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;width:100%}
.bd-rv2v-layout .bd-ref-audios,.bd-rv2v-layout .bd-ref-videos{gap:7px}
.bd-ref-audio,.bd-ref-video{position:relative;min-height:52px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;cursor:pointer;padding:6px 4px;box-sizing:border-box;font-size:9px;color:#666;text-align:center;line-height:1.25}
.bd-rv2v-layout .bd-ref-audio,.bd-rv2v-layout .bd-ref-video{min-height:0;height:auto;align-items:stretch;justify-content:flex-start;gap:6px;padding:6px;border-radius:8px;border:1px dashed #333;background:#080808;text-align:left;font-size:11px;color:#777}
.bd-ref-audio.has-audio,.bd-ref-video.has-video{border-style:solid;border-color:#4a6a4a;color:#cfe;background:#152015}
.bd-rv2v-layout .bd-ref-audio.has-audio,.bd-rv2v-layout .bd-ref-video.has-video{border-color:#2f4a38;background:#101812;color:#d8ebe0}
.bd-ref-audio:hover,.bd-ref-video:hover{border-color:#7a9cff;background:#1a1a1a}
.bd-rv2v-layout .bd-ref-audio:hover,.bd-rv2v-layout .bd-ref-video:hover{border-color:#555;background:#101010}
.bd-ref-audio.has-audio:hover,.bd-ref-video.has-video:hover{background:#1a2a1a}
.bd-rv2v-layout .bd-ref-audio .bd-r2v-thumb{width:100%;height:44px;border-radius:6px}
.bd-rv2v-layout .bd-ref-video .bd-r2v-thumb,.bd-rv2v-layout .bd-ref-video .bd-r2v-thumb-video{width:100%;height:auto;aspect-ratio:16/9;border-radius:6px;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#0c1014;border:1px solid #222;color:#6a7a8a;position:relative}
.bd-rv2v-layout .bd-ref-audio.has-audio .bd-r2v-thumb,.bd-rv2v-layout .bd-ref-video.has-video .bd-r2v-thumb{border-color:#3a5a45;color:#8fdfb0;background:#152018}
.bd-rv2v-layout .bd-ref-audio .bd-r2v-meta,.bd-rv2v-layout .bd-ref-video .bd-r2v-meta{flex-direction:row;align-items:center;justify-content:space-between;gap:4px}
.bd-rv2v-layout .bd-ref-audio audio.bd-r2v-media{position:absolute;width:0;height:0;opacity:0;pointer-events:none}
.bd-rv2v-layout .bd-ref-video video.bd-r2v-media{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;pointer-events:none}
.bd-ref-audio .bd-ref-audio-name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#9ad;font-size:9px;padding:0 2px}
.bd-rv2v-layout .bd-ref-audio .bd-ref-audio-name,.bd-rv2v-layout .bd-ref-audio .name,.bd-rv2v-layout .bd-ref-video .name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#666;font-size:10px;padding:0}
.bd-ref-audio .x,.bd-ref-video .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;line-height:1;display:none}
.bd-rv2v-layout .bd-ref-audio .x,.bd-rv2v-layout .bd-ref-video .x{top:8px;right:8px;width:20px;height:20px;border-radius:6px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.72);color:#ff9a9a;font-size:14px;font-weight:700;z-index:3}
.bd-ref-audio:hover .x,.bd-ref-video:hover .x{display:block}
.bd-rv2v-layout .bd-ref-audio:hover .x,.bd-rv2v-layout .bd-ref-video:hover .x{display:flex}
.bd-rv2v-layout .bd-refs-images-wrap.bd-r2v-section,.bd-rv2v-layout .bd-ref-audios-wrap.bd-r2v-section,.bd-rv2v-layout .bd-ref-videos-wrap.bd-r2v-section{display:flex;flex-direction:column;gap:8px}
.bd-r2v-section-count:empty{display:none}
.bd-r2v-section-actions{display:flex;align-items:center;gap:8px;flex-shrink:0}
.bd-r2v-pick-existing{background:transparent;border:1px solid #3a3a3a;color:#c8c8c8;border-radius:6px;padding:2px 8px;font-size:10px;cursor:pointer;line-height:1.4;white-space:nowrap}
.bd-r2v-pick-existing:hover{border-color:#4fff8f;color:#4fff8f}
.bd-r2v-pick-existing:disabled{opacity:.4;cursor:not-allowed;border-color:#333;color:#666}
.bd-prompt-layout:not(.bd-rv2v-layout) .bd-r2v-section-head{display:contents}
.bd-prompt-layout:not(.bd-rv2v-layout) .bd-r2v-section-count,
.bd-prompt-layout:not(.bd-rv2v-layout) .bd-r2v-pick-existing{display:none}
.bd-continuous-ref{display:flex;align-items:center;gap:6px;font-size:10px;color:#aaa;user-select:none;margin-left:8px}
.bd-continuous-ref label{display:flex;align-items:center;gap:4px;cursor:pointer}
.bd-continuous-ref input[type="checkbox"]{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
.bd-gen-fc-row{display:flex;align-items:center;gap:6px;margin-top:6px}
${IMAGE_BATCH_STYLES}
${FL2V_STYLES}
@media(max-width:768px){
.bd-prompt-layout,.bd-prompt-layout.bd-rv2v-layout,.bd-prompt-layout.bd-v2v-layout,.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live{grid-template-columns:1fr}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-live-sample{order:3;min-height:160px}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live .bd-live-sample-body{min-height:96px;max-height:140px}
.bd-ref{max-height:64px}
.bd-rv2v-layout .bd-ref{max-height:none}
.bd-v2v-layout .bd-prompt,.bd-rv2v-layout .bd-prompt{min-height:140px}
.bd-media-body{grid-template-columns:1fr}
.bd-media-preview{min-height:180px}
.bd-media-thead,.bd-media-tr{grid-template-columns:minmax(0,1fr) 72px 108px}
.bd-media-table.bd-media-nodims .bd-media-thead,
.bd-media-table.bd-media-nodims .bd-media-tr{grid-template-columns:minmax(0,1fr) 108px}
.bd-media-td,.bd-media-th{padding:7px 8px}
}
`;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

function snapDim(v, stride = 32) {
    return Math.max(stride, Math.round(v / stride) * stride);
}

/**
 * Match Python ``lib.image_prep.fit_long_edge``:
 * round(dim * scale / stride) * stride — keeps aspect, long side ≤ budget.
 * Stride must be 32 for MiniMax H3 (VAE 16× then 2×2 patch). Stride 16 can
 * yield 496×864 → odd latent width → patchify_video crash under continuity/v2v.
 */
function snapScaledDim(dim, scale, stride = 32) {
    return Math.max(stride, Math.round((dim * scale) / stride) * stride);
}

function resolveOutputDimensions(sourceW, sourceH, output, fallback = {}) {
    const mode = String(output?.mode || "long_edge").toLowerCase();
    const canvasStride = 32;
    if (mode === "fixed") {
        const w = snapDim(+(output?.width ?? fallback.width ?? 864), canvasStride);
        const h = snapDim(+(output?.height ?? fallback.height ?? 480), canvasStride);
        return { mode: "fixed", width: w, height: h, refMaxSize: Math.max(w, h) };
    }
    const longEdge = Math.max(canvasStride, +(output?.longEdge ?? output?.long_edge ?? fallback.refMaxSize ?? 848));
    const sw = sourceW || 0;
    const sh = sourceH || 0;
    if (!sw || !sh) {
        // Missing source: keep long-edge budget only — do not invent a 16:9 canvas
        // (that would center-crop ultrawide footage later via fit_canvas).
        return { mode: "long_edge", width: longEdge, height: canvasStride, refMaxSize: longEdge };
    }
    // Always recompute from source (even when already ≤ longEdge) so snapped
    // dims stay aspect-correct; never reuse a stale fixed W×H.
    const scale = Math.min(1, longEdge / Math.max(sw, sh));
    return {
        mode: "long_edge",
        width: snapScaledDim(sw, scale, canvasStride),
        height: snapScaledDim(sh, scale, canvasStride),
        refMaxSize: longEdge,
    };
}

/** Upload a file to ComfyUI input/ (videos use the same endpoint as images). */
function isUploadSizeError(err) {
    const msg = String(err?.message || err);
    return /body size|413|max_upload|too large|104857600/i.test(msg);
}

function formatUploadError(err) {
    const msg = String(err?.message || err);
    if (isUploadSizeError(err)) return t("upload.sizeLimitDetail");
    return msg;
}

function pickRulerMajorStepSec(pxPerSec) {
    const pps = Math.max(0.001, Number(pxPerSec) || 0.001);
    for (const step of RULER_MAJOR_SEC) {
        if (step * pps >= RULER_MIN_MAJOR_PX) return step;
    }
    return RULER_MAJOR_SEC[RULER_MAJOR_SEC.length - 1];
}

function pickRulerMinorStepSec(majorSec, pxPerSec) {
    const pps = Math.max(0.001, Number(pxPerSec) || 0.001);
    for (const div of [10, 5, 4, 2]) {
        const minor = majorSec / div;
        if (minor >= 1 && Number.isInteger(minor) && minor * pps >= RULER_MIN_MINOR_PX) {
            return minor;
        }
    }
    return majorSec;
}

/** 0, 5, 10 … below one minute; 1:00, 1:30 … at/after 60s. */
function formatRulerTime(sec) {
    const s = Math.max(0, Math.round(Number(sec) || 0));
    if (s < 60) return String(s);
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${String(r).padStart(2, "0")}`;
}

function formatProbeFps(value) {
    const fps = Math.round(Number(value) * 100) / 100;
    if (Number.isInteger(fps)) return String(fps);
    return fps.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function coerceTimelineFps(value, fallback = 24) {
    const fps = Number(value);
    if (!Number.isFinite(fps) || fps <= 0) return coerceTimelineFps(fallback, 24);
    return Math.round(clamp(fps, 1, 240) * 100) / 100;
}

async function uploadToInput(file) {
    const uploadFile = fileForComfyUpload(file);
    const body = new FormData();
    body.append("image", uploadFile, uploadFile.name);
    body.append("type", "input");
    body.append("overwrite", "false");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `Upload failed (${resp.status})`);
    }
    return resp.json();
}

async function uploadVideoChunked(file, onProgress) {
    const filename = safeUploadFilename(file?.name, file?.type);
    const uploadId = crypto.randomUUID();
    const totalChunks = Math.ceil(file.size / MINIMAX_CHUNK_SIZE);
    for (let i = 0; i < totalChunks; i++) {
        const start = i * MINIMAX_CHUNK_SIZE;
        const end = Math.min(start + MINIMAX_CHUNK_SIZE, file.size);
        const body = new FormData();
        body.append("upload_id", uploadId);
        body.append("chunk_index", String(i));
        body.append("total_chunks", String(totalChunks));
        body.append("filename", filename);
        body.append("chunk", file.slice(start, end), `${filename}.part`);
        const resp = await api.fetchApi("/minimax/director/upload_chunk", { method: "POST", body });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || t("upload.chunkFailed", { status: resp.status }));
        }
        onProgress?.((i + 1) / totalChunks, i + 1, totalChunks);
        const data = await resp.json();
        if (data.name) return data;
    }
    throw new Error(t("upload.chunkIncomplete"));
}

async function uploadToInputSmart(file, onProgress) {
    if (file.size <= COMFY_UPLOAD_SOFT_LIMIT) {
        try {
            return await uploadToInput(file);
        } catch (err) {
            if (!isUploadSizeError(err)) throw err;
        }
    }
    return uploadVideoChunked(file, onProgress);
}

function videoRelativePath(upload) {
    const name = upload.name || upload.filename;
    const sub = (upload.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
    return sub ? `${sub}/${name}` : name;
}

function inputViewUrl(relativePath, type = "input") {
    const norm = String(relativePath || "").replace(/\\/g, "/");
    const slash = norm.lastIndexOf("/");
    const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
    const subfolder = slash >= 0 ? norm.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, type });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

function refViewUrl(imageFile) {
    return inputViewUrl(imageFile, "input");
}

function deletedSourceRanges(video) {
    return video?.deletedSourceRanges || video?.deleted_source_ranges || [];
}

function logicalToSourceFrame(logical, video) {
    const map = video?.frameMap;
    if (map?.length) {
        return normalizeFrameMapEntry(map[clamp(logical, 0, map.length - 1)]).frame;
    }
    let src = logical;
    for (const [start, end] of [...deletedSourceRanges(video)].sort((a, b) => a[0] - b[0])) {
        if (src >= start) src += end - start;
        else break;
    }
    return src;
}

/** Inverse of logicalToSourceFrame for sparse deletes; -1 if source is in a deleted gap. */
function sourceToLogicalFrame(srcFrame, video) {
    const map = video?.frameMap;
    if (map?.length) {
        let best = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.frame === srcFrame) return i;
            if (e.frame < srcFrame) best = i;
            else if (best < 0) return -1; // before first kept
        }
        return best;
    }
    let logical = srcFrame;
    for (const [start, end] of [...deletedSourceRanges(video)].sort((a, b) => a[0] - b[0])) {
        if (srcFrame >= end) logical -= (end - start);
        else if (srcFrame >= start) return -1;
        else break;
    }
    return Math.max(0, logical);
}

function buildIdentityFrameMap(count) {
    return Array.from({ length: count }, (_, i) => i);
}

function normalizeFrameMapEntry(entry, defaultClip = 0) {
    if (entry == null) return { clip: defaultClip, frame: 0 };
    if (typeof entry === "number") return { clip: defaultClip, frame: entry };
    return {
        clip: entry.clip ?? entry.videoClip ?? defaultClip,
        frame: entry.frame ?? 0,
    };
}

function buildClipFrameMap(clipIndex, count) {
    return Array.from({ length: count }, (_, i) => ({ clip: clipIndex, frame: i }));
}

const CLIP_SEGMENT_COLORS = ["rgba(255,200,50,0.9)", "rgba(102,170,255,0.9)", "rgba(79,255,143,0.9)", "rgba(255,102,170,0.9)"];

function getDirectorUiHeight(editor) {
    if (editor?.getDirectorMode?.() === "prompt_batch") {
        const batchH = getImageBatchUiHeight(editor);
        // t2v / i2v / r2v show the main timeline track above batch cards.
        if (editor?.usesBatchTimeline?.()) {
            const track = editor?.canvasHeight || RULER_H + SEG_LABEL_H + TRACK_H;
            // toolbar + track + batch panel (batchH already includes list max-height cap)
            return batchH + track + 100;
        }
        return batchH + 100;
    }
    if (editor?.getDirectorMode?.() === "fl2v") {
        let h = getFl2vUiHeight(editor) + 110;
        if (editor?.needsLiveSamplePanel?.()) h += LIVE_SAMPLE_PREVIEW_H + 12;
        return h;
    }
    let h = (editor?.canvasHeight || RULER_H + SEG_LABEL_H + TRACK_H) + 370 + 52;
    if (
        editor?.hasVideo?.()
        && !editor?.isImageBatch?.()
        && !editor?.isGenMode?.()
        && !editor?.isFl2vMode?.()
    ) {
        h += STAGE_PREVIEW_H + 10;
    }
    // v2v live preview sits beside the prompt (no extra vertical stack).
    if (editor?.needsLiveSamplePanel?.() && !editor?.usesV2vPromptStyle?.()) {
        h += LIVE_SAMPLE_PREVIEW_H + 12;
    }
    return h;
}

function hookTaskTypeWidget(node) {
    const tw = node.widgets?.find((w) => w.name === "task_type");
    if (!tw || tw._berniniTaskHooked) return;
    tw._berniniTaskHooked = true;
    const orig = tw.callback;
    tw.callback = function (...args) {
        const r = orig?.apply(this, args);
        const ed = node._minimaxEditor;
        if (ed?.globalTask) ed.globalTask.value = tw.value;
        ed?.onTaskTypeChanged?.(tw.value);
        return r;
    };
}

/**
 * Only snap *runaway* heights (old infinite-growth corruption).
 * ideal+1200 was far too aggressive: r2v users routinely drag taller, and init heal
 * wiped the workflow-saved size on every Comfy restart (#7 regression).
 */
const DIRECTOR_UI_RUNAWAY_ABS_H = 12000;
const DIRECTOR_UI_RUNAWAY_EXTRA_H = 8000;

function healOversizedDirectorNode(node, editor) {
    if (!node?.size || !node.computeSize) return false;
    bindDomWidgetContentComputeSize(editor);
    const ideal = node.computeSize()?.[1];
    if (ideal == null) return false;
    const curH = node.size[1] || 0;
    const runaway = curH > DIRECTOR_UI_RUNAWAY_ABS_H
        || curH > ideal + DIRECTOR_UI_RUNAWAY_EXTRA_H;
    if (!runaway) return false;
    // Keep a modest stretch so heal does not feel like a hard snap to content min.
    const safeH = Math.max(ideal, Math.min(curH, ideal + DIRECTOR_UI_MAX_EXTRA_H));
    node.setSize([node.size[0], safeH]);
    node.setDirtyCanvas?.(true, true);
    return true;
}

/** After graph load / init: re-fill once LiteGraph assigns computedHeight from saved size. */
function scheduleDirectorLayoutSettle(editor) {
    if (!editor) return;
    const run = () => {
        if (editor.isPlaying || editor._pauseSettling) return;
        bindDomWidgetContentComputeSize(editor);
        // Do not ensure/heal here — preserve workflow size; only re-fill batch panel.
        syncBatchPanelFillHeight(editor, { settle: true });
    };
    requestAnimationFrame(() => {
        run();
        requestAnimationFrame(run);
        setTimeout(run, 80);
        setTimeout(run, 250);
    });
}

/** Grow once when content min increases (mode switch); never shrink; never use stretch. */
function ensureDirectorNodeFitsContent(node, editor) {
    if (!node?.size || !node.computeSize) return false;
    // Progress ticks must not grow the node — status text / rebuild noise used to ratchet.
    if (editor?.runStatusEl?.classList?.contains("active")) return false;
    bindDomWidgetContentComputeSize(editor);
    const ideal = node.computeSize()?.[1];
    if (ideal == null) return false;
    if ((node.size[1] || 0) >= ideal - 2) return false;
    const maxOk = ideal + DIRECTOR_UI_MAX_EXTRA_H;
    node.setSize([node.size[0], Math.min(ideal, maxOk)]);
    node.setDirtyCanvas?.(true, true);
    return true;
}

function syncDirectorNodeSize(node, editor) {
    if (editor?.isPlaying) return;
    // Update CSS/content min. Avoid stretch bookkeeping + Vue RO feedback loops.
    // User-dragged height is preserved by never shrinking (#7).
    editor?.updateDomWidgetHeight?.();
}

function ensureDirectorDomWidgetWidth(node) {
    const widget = node?._minimaxDomWidget;
    const fullW = node?.size?.[0];
    if (!widget || !fullW) return false;
    if (widget.width === fullW) return false;
    widget.width = fullW;
    return true;
}

function moveDirectorDomWidgetToEnd(node) {
    const widget = node?._minimaxDomWidget;
    if (!widget || !node?.widgets?.length) return;
    const idx = node.widgets.indexOf(widget);
    if (idx === -1 || idx === node.widgets.length - 1) return;
    node.widgets.splice(idx, 1);
    node.widgets.push(widget);
}

const PERF_WIDGET_ORDER = ["bd_grp_perf", "clear_vram_between_segments"];

function moveDirectorPerfWidgetsBeforeTimeline(node) {
    const dom = node?._minimaxDomWidget;
    if (!node?.widgets?.length) return;

    const perfWidgets = PERF_WIDGET_ORDER
        .map((name) => node.widgets.find((w) => w.name === name))
        .filter(Boolean);
    if (!perfWidgets.length) return;

    for (const w of perfWidgets) {
        const idx = node.widgets.indexOf(w);
        if (idx !== -1) node.widgets.splice(idx, 1);
    }

    const insertAt = dom ? node.widgets.indexOf(dom) : -1;
    const at = insertAt === -1 ? node.widgets.length : insertAt;
    node.widgets.splice(at, 0, ...perfWidgets);
}

function finalizeDirectorWidgetOrder(node) {
    moveDirectorPerfWidgetsBeforeTimeline(node);
    moveDirectorDomWidgetToEnd(node);
}

const DIRECTOR_DOM_WIDGET_NAME = "minimax_director_ui";

function listDirectorDomWidgets(node) {
    return (node?.widgets || []).filter((w) => w?.name === DIRECTOR_DOM_WIDGET_NAME);
}

/** Keep one Director DOM widget; drop extras left by double-wrapped onNodeCreated. */
function pruneDirectorDomWidgets(node) {
    if (!node) return null;
    const dups = listDirectorDomWidgets(node);
    const keep = dups.find((w) => w.element?.querySelector?.(":scope > .bd-wrap"))
        || dups.find((w) => w === node._minimaxDomWidget && w?.element)
        || dups.find((w) => w?.element)
        || node._minimaxDomWidget
        || null;
    if (dups.length > 1) {
        for (const w of dups) {
            if (w === keep) continue;
            const idx = node.widgets.indexOf(w);
            if (idx !== -1) node.widgets.splice(idx, 1);
            try { w.onRemove?.(); } catch { /* ignore */ }
            w.element?.remove?.();
        }
    }
    if (keep) node._minimaxDomWidget = keep;
    return keep || null;
}

function destroyDirectorEditor(node) {
    const ed = node?._minimaxEditor;
    if (!ed) {
        if (node) node._minimaxEditor = null;
        return;
    }
    try { ed.destroy(); } catch { /* ignore */ }
    if (node._minimaxEditor === ed) node._minimaxEditor = null;
    if (ed.domWidget?._minimaxEditor === ed) ed.domWidget._minimaxEditor = null;
}

function bindDirectorDomWidgetSizing(node, widget, getEditor) {
    const editor = getEditor?.();
    const minHeight = () => getDirectorUiHeight(getEditor?.());
    // Do not set computeSize — fixed-size widgets never receive resize free space.
    try {
        delete widget.computeSize;
    } catch {
        widget.computeSize = undefined;
    }
    widget.computeLayoutSize = () => ({
        minHeight: minHeight(),
        maxHeight: undefined,
        minWidth: DIRECTOR_MIN_WIDTH,
    });
    if (widget.options) {
        widget.options.getMinHeight = minHeight;
        delete widget.options.getMaxHeight;
    }
    const el = widget.element;
    if (el) {
        el.style.minHeight = `${minHeight()}px`;
        el.style.setProperty("--comfy-widget-min-height", `${minHeight()}px`);
    }
    if (editor) bindDomWidgetContentComputeSize(editor);
}

function initDirectorEditor(node) {
    // Must not share Bernini's `_directorDomWidget` — their loadedGraphNode mounts on that key.
    if (!isMiniMaxH3DirectorNode(node)) return null;
    const widget = pruneDirectorDomWidgets(node);
    const container = widget?.element;
    if (!container) return null;

    const existing = node._minimaxEditor || widget._minimaxEditor;
    if (existing?.root && existing.container === container) {
        node._minimaxEditor = existing;
        widget._minimaxEditor = existing;
        for (const wrap of [...container.querySelectorAll(":scope > .bd-wrap")]) {
            if (wrap !== existing.root) wrap.remove();
        }
        return existing;
    }
    // Constructor runs before `_minimaxEditor` is assigned; block re-entrant mounts
    // from onConfigure / loadedGraphNode / layout callbacks during buildDOM.
    if (node._minimaxEditorMounting) return existing || null;

    if (existing) destroyDirectorEditor(node);

    node._minimaxEditorMounting = true;
    try {
        for (const wrap of [...container.querySelectorAll(":scope > .bd-wrap")]) wrap.remove();
        hookTaskTypeWidget(node);
        const editor = new MiniMaxH3DirectorEditor(node, container, widget);
        node._minimaxEditor = editor;
        widget._minimaxEditor = editor;
        ensureDirectorDomWidgetWidth(node);
        bindDirectorDomWidgetSizing(node, widget, () => node._minimaxEditor);
        // Only clamp true runaway; never steal normal user/workflow height (#7).
        healOversizedDirectorNode(node, editor);
        syncDirectorNodeSize(node, editor);
        scheduleDirectorLayoutSettle(editor);
        return editor;
    } catch (err) {
        console.error("[MiniMax H3Director] UI init failed:", err);
        return node._minimaxEditor || null;
    } finally {
        node._minimaxEditorMounting = false;
    }
}

function patchDirectorDomWidgetLayout() {
    const canvas = app.canvas;
    if (!canvas || canvas._minimaxDirectorLayoutPatch) return;
    canvas._minimaxDirectorLayoutPatch = true;
    const prev = canvas.onDrawForeground;
    canvas.onDrawForeground = function (ctx) {
        const graph = app.graph ?? canvas.graph;
        for (const node of graph?._nodes ?? graph?.nodes ?? []) {
            if (node._minimaxEditor?.isPlaying) continue;
            ensureDirectorDomWidgetWidth(node);
        }
        return prev?.apply(this, arguments);
    };
}

function stopDomEvent(e) {
    e.stopPropagation();
}

/** True when focus/target is a Director text field (incl. contenteditable token editor). */
function directorEditableFromEventTarget(target) {
    let node = target;
    if (node?.nodeType === Node.TEXT_NODE) node = node.parentElement;
    if (!node?.closest) return null;
    if (!node.closest(".mmx-host")) return null;
    return node.closest("input, textarea, select, [contenteditable='true'], .bd-token-editor");
}

/**
 * Stop Comfy graph copy/paste from firing while typing in Director prompts.
 * contenteditable chips are invisible to Comfy's INPUT/TEXTAREA checks, so Ctrl+V
 * otherwise pastes the last copied nodes beside the Director.
 */
function installDirectorClipboardGuard() {
    if (typeof document === "undefined" || document.__mmxDirectorClipboardGuard) return;
    document.__mmxDirectorClipboardGuard = true;

    const blockBubbleToCanvas = (e) => {
        if (!directorEditableFromEventTarget(e.target)
            && !directorEditableFromEventTarget(document.activeElement)) {
            return;
        }
        e.stopImmediatePropagation();
    };

    for (const type of ["paste", "copy", "cut"]) {
        document.addEventListener(type, blockBubbleToCanvas, true);
    }
    document.addEventListener("keydown", (e) => {
        if (!(e.ctrlKey || e.metaKey)) return;
        const k = e.key?.toLowerCase?.();
        if (k !== "v" && k !== "c" && k !== "x") return;
        blockBubbleToCanvas(e);
    }, true);

    const patchPaste = () => {
        const canvas = app.canvas;
        if (!canvas) return;
        const wrap = (obj, key) => {
            if (!obj || typeof obj[key] !== "function" || obj[key].__mmxDirectorPatched) return;
            const orig = obj[key];
            const patched = function (...args) {
                if (directorEditableFromEventTarget(document.activeElement)) return null;
                return orig.apply(this, args);
            };
            patched.__mmxDirectorPatched = true;
            obj[key] = patched;
        };
        wrap(canvas, "pasteFromClipboard");
        wrap(canvas.constructor?.prototype, "pasteFromClipboard");
        // Some frontend builds expose paste on the LiteGraph canvas proto only.
        try {
            const LG = globalThis.LiteGraph?.LGraphCanvas?.prototype;
            wrap(LG, "pasteFromClipboard");
        } catch {
            /* ignore */
        }
    };
    patchPaste();
    queueMicrotask(patchPaste);
    setTimeout(patchPaste, 0);
    setTimeout(patchPaste, 500);
}

function hideWidget(w) {
    if (!w) return;
    // Group headers in HIDDEN_WIDGETS duplicate timeline panel sections — hide them too.
    if (w._bdGroupHeader && !HIDDEN_WIDGETS.includes(w.name)) return;
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    w.computeSize = () => [0, 0];
    if (w.element) w.element.style.display = "none";
}

function parseTimeline(raw, totalFrames, fps) {
    const total = totalFrames || 124;
    const base = {
        version: 4,
        editMode: "global",
        totalFrames: total,
        frameRate: coerceTimelineFps(fps || 24),
        video: {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
        },
        videoClips: [],
        global: {
            taskType: "", prompt: "", refs: [], refAudios: [], referenceVideo: {},
            continuousReference: false, commonEnabled: false, commonCollapsed: false,
        },
        output: {
            // v2v/rv2v default: scale by long edge (preserve aspect). Fixed = center-crop.
            mode: "long_edge",
            aspectRatio: DEFAULT_ASPECT_RATIO,
            megapixels: DEFAULT_MEGAPIXELS,
            multiple: MINIMAX_CANVAS_MULTIPLE,
            longEdge: 848, width: 848, height: 480,
            maxExportFrames: 0, exportMode: "all",
            audioMode: "generate",
            exportSourceImages: false,
            refImageSize: "match",
            continuityEnabled: false, continuityOverlapFrames: DEFAULT_CONTINUITY_FRAMES,
        },
        runSelectEnabled: false,
        runSelection: [],
        liveTaePreview: false,
        batchDetailMode: "solo",
        segments: [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], refAudios: [], referenceVideo: {} }],
    };
    if (!raw?.trim()) return base;
    try {
        const data = JSON.parse(raw);
        data.version = data.version || 4;
        data.editMode = data.editMode || "global";
        data.frameRate = coerceTimelineFps(data.frameRate ?? fps ?? 24);
        data.video = data.video || { fileName: "", frames: [] };
        if (!data.video.videoFile && data.video.fileName) {
            data.video.videoFile = data.video.fileName;
        }
        data.video.type = data.video.type || "input";
        data.video.subfolder = data.video.subfolder || "";
        data.video.frames = data.video.frames || [];
        data.global = data.global || {
            refs: [], refAudios: [], referenceVideo: {},
            continuousReference: false, commonEnabled: false, commonCollapsed: false,
        };
        data.global.refs = data.global.refs || [];
        data.global.refAudios = data.global.refAudios || data.global.ref_audios || [];
        data.global.refVideos = data.global.refVideos || data.global.ref_videos || [];
        if (Array.isArray(data.global.refVideos)) {
            data.global.refVideos = data.global.refVideos.map(sanitizeRefVideo);
        }
        data.global.referenceVideo = data.global.referenceVideo || data.global.reference_video || {};
        data.global.continuousReference = !!data.global.continuousReference || !!data.global.continuous_reference;
        // r2v shared params: default OFF unless explicitly enabled.
        data.global.commonEnabled = !!(
            data.global.commonEnabled ?? data.global.common_enabled
        );
        // UI fold only — does not affect runtime merge when commonEnabled is true.
        data.global.commonCollapsed = !!(
            data.global.commonCollapsed ?? data.global.common_collapsed
        );
        const legacyRef = data.referenceVideo || data.reference_video;
        if (legacyRef && (legacyRef.videoFile || legacyRef.fileName)
            && !(data.global.referenceVideo.videoFile || data.global.referenceVideo.fileName)) {
            data.global.referenceVideo = { ...legacyRef };
        }
        delete data.referenceVideo;
        delete data.reference_video;
        data.output = normalizeOutputContinuity({
            mode: data.output?.mode || "long_edge",
            // Keep ResolutionSelector fields across reload (were previously dropped → always 16:9).
            aspectRatio: data.output?.aspectRatio != null
                ? normalizeAspectRatioLabel(data.output.aspectRatio)
                : undefined,
            megapixels: data.output?.megapixels ?? data.output?.megaPixels ?? undefined,
            multiple: data.output?.multiple ?? MINIMAX_CANVAS_MULTIPLE,
            longEdge: data.output?.longEdge ?? data.output?.long_edge ?? data.refMaxSize ?? 848,
            width: data.output?.width ?? data.width ?? 864,
            height: data.output?.height ?? data.height ?? 480,
            maxExportFrames: data.output?.maxExportFrames ?? data.output?.max_export_frames ?? 0,
            exportMode: data.output?.exportMode ?? data.output?.export_mode ?? "all",
            audioMode: normalizeAudioMode(data.output?.audioMode ?? data.output?.audio_mode),
            exportSourceImages: data.output?.exportSourceImages === true
                || data.output?.export_source_images === true,
            refImageSize: normalizeRefImageSize(data.output?.refImageSize ?? data.output?.ref_image_size),
            continuityEnabled: data.output?.continuityEnabled ?? data.output?.continuity_enabled,
            continuityOverlapFrames: data.output?.continuityOverlapFrames ?? data.output?.continuity_overlap_frames,
        });
        // Infer aspectRatio from saved width/height when older payloads omitted the label.
        if (!data.output.aspectRatio && data.output.width > 0 && data.output.height > 0) {
            const rw = data.output.width;
            const rh = data.output.height;
            const match = RESOLUTION_ASPECTS.find(([, aw, ah]) => Math.abs(rw / rh - aw / ah) < 0.02);
            data.output.aspectRatio = match ? match[0] : CUSTOM_ASPECT_RATIO;
        }
        if (!data.output.aspectRatio) data.output.aspectRatio = DEFAULT_ASPECT_RATIO;
        if (data.output.megapixels == null) data.output.megapixels = DEFAULT_MEGAPIXELS;
        stripTimelineContinuityRootFields(data);
        stripTimelineEphemeralFields(data);
        const legacyFrames = data.video.frames?.length || 0;
        if (!data.video.frameMap?.length) {
            const n = data.totalFrames || data.video.sourceFrameCount || legacyFrames || total;
            data.totalFrames = n;
            data.video.sourceFrameCount = data.video.sourceFrameCount || n;
            data.video.deletedSourceRanges = data.video.deletedSourceRanges || [];
            data.video.frameMap = [];
        }
        if (!data.segments?.length) {
            const n = data.totalFrames || data.video.sourceFrameCount || legacyFrames || total;
            data.segments = [{ id: uid(), start: 0, length: Math.max(MIN_SEG, n), prompt: "", taskType: "", refs: [], refAudios: [], referenceVideo: {} }];
        }
        for (const seg of data.segments) {
            if (!seg.id) seg.id = uid();
            if (seg.length == null && seg.end != null) seg.length = seg.end - seg.start;
            if (seg.frameCount == null && seg.length != null) seg.frameCount = seg.length;
            seg.refs = seg.refs || [];
            seg.refAudios = seg.refAudios || seg.ref_audios || [];
            seg.referenceVideo = seg.referenceVideo || seg.reference_video || {};
            seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
            seg.negativePrompt = seg.negativePrompt ?? "";
        }
        data.gen = data.gen || { defaultFrameCount: 124 };
        if (data.global) {
            data.global.genImage = data.global.genImage || { imageFile: data.global.imageFile || "" };
        }
        data.runSelectEnabled = !!data.runSelectEnabled;
        data.runSelection = Array.isArray(data.runSelection) ? data.runSelection.map((i) => parseInt(i, 10)).filter((i) => i >= 0) : [];
        // Default off when missing. Explicit true keeps in-node TAE + segment playback.
        data.liveTaePreview = data.liveTaePreview === true || data.live_tae_preview === true;
        const detailMode = data.batchDetailMode ?? data.batch_detail_mode;
        data.batchDetailMode = detailMode === "all" ? "all" : "solo";
        if (data.timelineMode === "fl2v" || resolveTaskKey(data.global?.taskType || "") === "fl2v") {
            data.timelineMode = "fl2v";
            data.editMode = "segment";
            data.keyframes = Array.isArray(data.keyframes) ? data.keyframes : [];
            data.shots = Array.isArray(data.shots) ? data.shots : [];
            const stored = parseInt(data.totalFrames, 10);
            const farthest = Math.max(
                0,
                ...(data.segments || []).map((s) => (parseInt(s.start, 10) || 0) + (parseInt(s.length ?? s.frameCount, 10) || 0)),
                ...(data.keyframes || []).map((k) => (parseInt(k.start, 10) || 0) + (parseInt(k.frameCount ?? k.length, 10) || 0)),
            );
            data.totalFrames = (Number.isFinite(stored) && stored > 0)
                ? stored
                : Math.max(farthest, total, 240);
            return data;
        }
        if (data.timelineMode === "image_batch" || data.timelineMode === "prompt_batch") {
            data.timelineMode = "prompt_batch";
            data.editMode = "segment";
            data.totalFrames = sumFrameCounts(data.segments) || data.totalFrames || total;
            return data;
        }
        if (data.timelineMode === "gen_blank" || data.timelineMode === "gen_image") {
            const gkey = resolveTaskKey(data.global?.taskType || "");
            if (isPromptBatchTask(gkey)) {
                data.timelineMode = "prompt_batch";
                data.editMode = "segment";
            }
            data.totalFrames = sumFrameCounts(data.segments) || data.totalFrames || total;
            return data;
        }
        if (!data.videoClips?.length && data.video?.videoFile) {
            data.videoClips = [{
                id: data.video.id || uid(),
                fileName: data.video.fileName || "",
                videoFile: data.video.videoFile || data.video.fileName || "",
                subfolder: data.video.subfolder || "",
                type: data.video.type || "input",
                width: data.video.width || 0,
                height: data.video.height || 0,
                duration: data.video.duration || 0,
                nativeFps: data.video.nativeFps || data.video.native_fps || 0,
                nativeFrameCount: data.video.nativeFrameCount || data.video.native_frame_count || 0,
                sourceFrameCount: data.video.sourceFrameCount || data.video.frameMap?.length || 0,
                storageWidth: data.video.storageWidth,
                storageHeight: data.video.storageHeight,
            }];
        }
        data.videoClips = data.videoClips || [];
        data.totalFrames = data.totalFrames || data.video.sourceFrameCount || data.video.frameMap?.length || total;
        return data;
    } catch {
        return base;
    }
}

class MiniMaxH3DirectorEditor {
    constructor(node, container, domWidget) {
        this.node = node;
        this.container = container;
        this.domWidget = domWidget;
        this.zoom = 1;
        this.zoomEnabled = false;
        this.selectedIndex = 0;
        /** @type {number|null} Selected editable split-point frame (logical). */
        this.selectedSplitFrame = null;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.isLooping = false;
        this._playRaf = null;
        this._drag = null;
        this._previewSegments = null;
        this._edgeSnapshot = null;
        this._isHovering = false;
        this._thumbCache = new Map();
        this._thumbPending = new Set();
        this._seekChain = Promise.resolve();
        this._legacyFrames = [];
        this._storageWidth = 0;
        this._storageHeight = 0;
        this._previewVideo = null;
        this._previewVideos = new Map();
        this._thumbCanvas = null;
        this._syncTimer = null;
        this._resizeRaf = null;
        this._renderPending = false;
        this._settleRenderTimer = null;
        this._settleRenderLateTimer = null;
        this._promptRenderTimer = null;
        this._lastSeekUiMs = 0;
        this._playCanvasWidth = 0;
        this._pauseSettling = false;
        this._runHighlightSeg = -1;
        this._modalEl = null;
        this._modalKeyHandler = null;
        this._drawWidth = 0;
        this._reorderDropRank = -1;
        this._reorderFromRank = -1;
        this.canvasHeight = RULER_H + SEG_LABEL_H + TRACK_H;
        this._stageClipIndex = -1;
        this._stageSyncMs = 0;
        this._playHandoff = false;

        for (const w of node.widgets || []) {
            if (HIDDEN_WIDGETS.includes(w.name)) hideWidget(w);
        }

        this.timelineWidget = this.widget("timeline_data");
        this.totalFramesWidget = this.widget("total_frames");
        this.frameRateWidget = this.widget("frame_rate");
        this.taskTypeWidget = this.widget("task_type");
        this.globalPromptWidget = this.widget("global_prompt");
        this.negativePromptWidget = null;
        this.widthWidget = this.widget("width");
        this.heightWidget = this.widget("height");
        this.refMaxWidget = this.widget("ref_max_size");

        const initTotal = Math.max(0, parseInt(this.totalFramesWidget?.value || 124, 10));
        const initFps = coerceTimelineFps(this.frameRateWidget?.value || 24);
        this.timeline = parseTimeline(this.timelineWidget?.value, initTotal, initFps);
        this.buildDOM();
        this.bindEvents();
        this._unsubLocale = onLocaleChange(() => this.applyLocale());
        this.applyLocale();
        this._directorMode = getDirectorMode(this.taskTypeWidget?.value);
        this._taskKey = resolveTaskKey(this.taskTypeWidget?.value);
        if (this._directorMode === "video") {
            this.restoreVideoFromTimeline();
        } else if (this._directorMode === "prompt_batch" || this._directorMode === "image_batch") {
            ensureImageBatchTimeline(this);
        } else {
            this.ensureGenTimeline();
        }
        this.applyTaskLayout(this._directorMode);

        this.updateDomWidgetHeight();
        this.applyZoomWidth();
        this.syncFromWidgets();
        this.updateModeUI();
        this.updateSelectionUI();
        this.commit(true, { syncTimeline: false });
        this._observeViewportResize();
        this.syncExternalGroupsTimeline();
        this.scheduleSettleRender();
    }

    _inputLinkConnected(name) {
        const inp = this.node?.inputs?.find((i) => i?.name === name);
        return inp != null && inp.link != null;
    }

    hasExternalI2vGroups() {
        return this._inputLinkConnected("i2v_groups");
    }

    hasExternalR2vGroups() {
        return this._inputLinkConnected("r2v_groups");
    }

    updateExternalGroupsBanner() {
        const el = this.externalGroupsMsgEl || this.root?.querySelector('[data-r="external-groups-msg"]');
        if (!el) return;
        const i2v = this.hasExternalI2vGroups();
        const r2v = this.hasExternalR2vGroups();
        const active = i2v || r2v;
        el.classList.toggle("hidden", !active);
        this.root?.classList.toggle("bd-external-groups", active);
        // Refresh add/delete visibility when external wiring toggles.
        if (this.isR2vBatch?.()) setR2vToolbar(this, true);
        else if (this.isFl2vMode?.()) setFl2vToolbar(this, true);
        else {
            updateR2vToolbarBtns(this);
            updateFl2vToolbarBtns(this);
        }
        if (!active) {
            el.textContent = "";
            return;
        }
        const specs = collectExternalGroupSpecs(this);
        const n = specs?.length || 0;
        const base = i2v ? t("external.i2vActive") : t("external.r2vActive");
        const count = n > 0 ? ` (${t("external.groupCount", { n })})` : "";
        el.textContent = `${base}${count} ${t("external.durationHint")}`;
    }

    /**
     * Push a Director-card prompt edit into the matching external Group node
     * widget so execution (and the next sync) don't revive stale graph text.
     */
    writeExternalGroupPrompt(segIndex, prompt) {
        if (!this.hasExternalI2vGroups?.() && !this.hasExternalR2vGroups?.()) return;
        const nodes = collectExternalGroupNodes(this);
        const node = nodes?.[segIndex];
        if (!node) return;
        const w = (node.widgets || []).find((x) => x?.name === "prompt");
        if (!w) return;
        const next = String(prompt ?? "");
        if (String(w.value ?? "") === next) return;
        // Avoid feedback loop: our widget callback triggers syncExternalGroupsTimeline.
        w._mmxSkipExternalSync = true;
        try {
            w.value = next;
            // ComfyUI V3 / custom widgets may need callback for persistence.
            w.callback?.(next);
        } finally {
            queueMicrotask(() => { w._mmxSkipExternalSync = false; });
        }
    }

    /** Mirror graph-wired Group count/duration into the Director timeline UI. */
    syncExternalGroupsTimeline() {
        this.updateExternalGroupsBanner();
        // Keep any in-progress Director textarea edits before rebuilding from graph.
        if (this.isImageBatch?.()) flushBatchPromptInputs(this);
        if (this.isFl2vMode?.()) flushFl2vPromptDraft(this);
        const specs = collectExternalGroupSpecs(this);
        if (!specs?.length) {
            this._externalGroupsSyncSig = null;
            return;
        }

        const mode = this.getDirectorMode?.() || this._directorMode;
        const taskKey = resolveTaskKey(this.getTaskKey?.() || this.taskTypeWidget?.value);
        const sig = JSON.stringify(specs.map((s) => [
            s.nodeId ?? "",
            Number(s.durationSec) || 0,
            s.prompt || "",
            s.firstImageFile || "",
            s.lastImageFile || "",
            (s.refImages || []).map((r) => `${r.index}:${r.imageFile || ""}`).join(","),
            (s.refVideos || []).map((r) => [
                r.index,
                r.videoFile || "",
                r.previewImageFile || "",
                r.previewImageUrl || "",
                r.pairedAudioFile || "",
                r.linked ? 1 : 0,
            ].join(":")).join(","),
            (s.refAudios || []).map((r) => `${r.index}:${r.audioFile || ""}`).join(","),
        ]));
        if (this._externalGroupsSyncSig === sig) return;
        this._externalGroupsSyncSig = sig;

        if (mode === "fl2v") {
            const prev = this.timeline.shots || [];
            const prevByNode = new Map(
                prev.filter((s) => s?.externalNodeId != null)
                    .map((s) => [String(s.externalNodeId), s]),
            );
            const allowIndexFallback = !prev.some((s) => s?.externalNodeId != null);
            this.timeline.shots = specs.map((spec, i) => {
                const matched = (spec.nodeId != null && prevByNode.get(String(spec.nodeId)))
                    || (allowIndexFallback ? (prev[i] || null) : null);
                // Same Group node → keep Director draft if widget briefly empty.
                // Different/new node at this index → never inherit another shot's prompt.
                const specPrompt = String(spec.prompt ?? "").trim();
                const prompt = specPrompt
                    || (matched ? String(matched.prompt || "").trim() : "");
                return newFl2vShot({
                    id: matched?.id,
                    durationSec: spec.durationSec ?? defaultDurationSec("fl2v"),
                    prompt,
                    externalNodeId: spec.nodeId ?? null,
                    // External graph is source of truth for media previews.
                    startImage: imageRefFromPath(spec.firstImageFile),
                    endImage: imageRefFromPath(spec.lastImageFile),
                });
            });
            syncFl2vFromShots(this);
            this.selectedIndex = Math.min(this.selectedIndex ?? 0, Math.max(0, this.timeline.shots.length - 1));
            updateFl2vDetailUI?.(this);
            this.scheduleRender?.();
            this.commit?.(false, { syncTimeline: true });
            this.updateVideoNameLabel?.();
            this.updateDomWidgetHeight?.();
            this.updateRunSelectUI?.();
            return;
        }

        if (mode === "prompt_batch" || mode === "image_batch" || isPromptBatchTask(taskKey)) {
            const prev = this.timeline.segments || [];
            const prevByNode = new Map(
                prev.filter((s) => s?.externalNodeId != null)
                    .map((s) => [String(s.externalNodeId), s]),
            );
            // First wire / pre-nodeId eras: allow index align once. After segments are
            // tagged, never inherit prompt from a different Group at the same index.
            const allowIndexFallback = !prev.some((s) => s?.externalNodeId != null);
            const isR2v = taskKey === "r2v" || this.hasExternalR2vGroups?.();
            const promptWriteBack = [];
            const activePromptIndex = (() => {
                const el = typeof document !== "undefined" ? document.activeElement : null;
                if (!el?.getAttribute) return -1;
                const n = parseInt(el.getAttribute("data-batch-prompt-index"), 10);
                return Number.isFinite(n) ? n : -1;
            })();
            this.timeline.segments = specs.map((spec, i) => {
                const matched = (spec.nodeId != null && prevByNode.get(String(spec.nodeId)))
                    || (allowIndexFallback ? (prev[i] || null) : null);
                const firstRef = imageRefFromPath(spec.firstImageFile);
                const genImage = firstRef
                    || (isR2v ? (matched?.genImage || { imageFile: "" }) : { imageFile: "" });
                // External graph is source of truth for r2v media (do not keep stale UI uploads).
                const refs = isR2v
                    ? (spec.refImages || []).map((r) => ({
                        index: r.index,
                        imageFile: r.imageFile || "",
                        imageB64: "",
                    }))
                    : (matched?.refs || []);
                const refVideos = isR2v
                    ? (spec.refVideos || []).map((r) => ({
                        index: r.index,
                        videoFile: r.videoFile || "",
                        fileName: r.fileName || "",
                        type: r.type || "input",
                        subfolder: r.subfolder || "",
                        pairedAudioFile: r.pairedAudioFile || "",
                        previewImageFile: r.previewImageFile || "",
                        previewImageUrl: r.previewImageUrl || "",
                        linked: !!r.linked || !!(r.videoFile || r.previewImageFile || r.previewImageUrl),
                    }))
                    : (matched?.refVideos || []);
                const refAudios = isR2v
                    ? (spec.refAudios || []).map((r) => ({
                        index: r.index,
                        audioFile: r.audioFile || "",
                        fileName: r.fileName || "",
                        type: r.type || "input",
                        subfolder: r.subfolder || "",
                    }))
                    : (matched?.refAudios || []);
                const specPrompt = String(spec.prompt ?? "").trim();
                const draftPrompt = matched ? String(matched.prompt || "").trim() : "";
                // Priority: focused Director textarea > Group widget > same-node draft.
                // Prevents a just-pasted Group-3 prompt from being replaced by stale
                // widget text from a previous short film during an incidental sync.
                let prompt = specPrompt;
                if (activePromptIndex === i && draftPrompt) {
                    prompt = draftPrompt;
                } else if (!specPrompt) {
                    prompt = draftPrompt;
                }
                if (prompt && prompt !== specPrompt) {
                    promptWriteBack.push({ index: i, prompt });
                }
                return newBatchSegment({
                    ...(matched?.id ? { id: matched.id } : {}),
                    durationSec: spec.durationSec ?? defaultDurationSec(taskKey),
                    prompt,
                    negativePrompt: matched?.negativePrompt ?? "",
                    externalNodeId: spec.nodeId ?? null,
                    refs,
                    refAudios,
                    refVideos,
                    genImage: genImage?.imageFile ? genImage : { imageFile: "" },
                    imageFile: genImage?.imageFile || "",
                    // Preserve preview frames for the same Group node across syncs.
                    previewB64: matched?.previewB64 || "",
                    previewFrames: matched?.previewFrames || [],
                    previewFps: matched?.previewFps,
                    refImageSize: matched?.refImageSize ?? matched?.ref_image_size,
                    ...(matched?.runEnabled != null ? { runEnabled: matched.runEnabled } : {}),
                });
            });
            for (const item of promptWriteBack) {
                this.writeExternalGroupPrompt(item.index, item.prompt);
            }
            normalizeImageBatchSegments(this);
            this.selectedIndex = Math.min(this.selectedIndex ?? 0, Math.max(0, this.timeline.segments.length - 1));
            this.renderImageBatchGroups?.();
            this.scheduleRender?.();
            this.commit?.(false, { syncTimeline: true });
            this.updateVideoNameLabel?.();
            this.updateDomWidgetHeight?.();
            this.updateRunSelectUI?.();
            this.updateSelectionUI?.();
        }
    }

    _observeViewportResize() {
        if (!this.viewport || typeof ResizeObserver === "undefined") return;
        this._resizeObserver?.disconnect();
        this._resizeObserver = new ResizeObserver(() => {
            if (this.isPlaying || this._pauseSettling) return;
            this.scheduleRender();
        });
        this._resizeObserver.observe(this.viewport);
        if (this.container && this.container !== this.viewport) {
            this._resizeObserver.observe(this.container);
        }
    }

    /**
     * CSS layout width for the timeline bitmap.
     * Must NOT use getBoundingClientRect — ComfyUI graph zoom transforms inflate/deflate
     * that value while width:100% still follows clientWidth, and object-fit:fill then
     * stretches segment thumbnails.
     */
    _measureDrawWidth() {
        if (this.isPlaying && this._playCanvasWidth > 0) return this._playCanvasWidth;
        if (this.getTimelineZoom() > 1) {
            const zoomed = this.canvas?.clientWidth || this.canvas?.offsetWidth || 0;
            if (zoomed > 0) return zoomed;
        }
        return this.viewport?.clientWidth
            || this.canvas?.clientWidth
            || this.canvas?.offsetWidth
            || this.container?.clientWidth
            || this.root?.clientWidth
            || 0;
    }

    /** Redraw after layout/zoom settles (first mount often measures before the node finishes sizing). */
    scheduleSettleRender() {
        this.scheduleRender();
        if (this._settleRenderTimer != null) return;
        this._settleRenderTimer = setTimeout(() => {
            this._settleRenderTimer = null;
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    if (!this.isPlaying) this.scheduleRender();
                });
            });
        }, 0);
        // Extra pass after ComfyUI node size / graph zoom finishes applying.
        clearTimeout(this._settleRenderLateTimer);
        this._settleRenderLateTimer = setTimeout(() => {
            this._settleRenderLateTimer = null;
            if (!this.isPlaying) this.scheduleRender();
        }, 100);
    }

    _capturePlayCanvasWidth() {
        const w = this.viewport?.clientWidth
            || this.container?.offsetWidth
            || this.node?.size?.[0]
            || DIRECTOR_MIN_WIDTH;
        if (w > 0) this._playCanvasWidth = w;
        return this._playCanvasWidth;
    }

    _lockPlayLayout() {
        this._capturePlayCanvasWidth();
    }

    _resetLayoutStyles() {
        if (this.isPlaying) return;
        for (const el of [this.container, this.root, this.viewport]) {
            if (!el) continue;
            el.style.removeProperty("width");
            el.style.removeProperty("min-width");
            el.style.removeProperty("max-width");
        }
        this._playCanvasWidth = 0;
        this.applyZoomWidth();
    }

    _releasePlayLayoutLock() {
        this._resetLayoutStyles();
    }

    getDirectorUiMinHeight() {
        return getDirectorUiHeight(this);
    }

    updateDomWidgetHeight(opts = {}) {
        const h = contentDomWidgetMinHeight(this) || getDirectorUiHeight(this);
        this.container?.style.setProperty("--comfy-widget-min-height", `${h}px`);
        if (this.container) this.container.style.minHeight = `${h}px`;
        // Content min only — never bake node.size / stretch into computeSize.
        bindDomWidgetContentComputeSize(this);
        const runActive = !!this.runStatusEl?.classList?.contains("active");
        // Grow only when content needs more room (e.g. mode switch). Never shrink
        // a user-enlarged node (#7). During live progress: never grow; heal runaway.
        if (!this.isPlaying) {
            if (runActive) healOversizedDirectorNode(this.node, this);
            else ensureDirectorNodeFitsContent(this.node, this);
        }
        syncBatchPanelFillHeight(this, {
            settle: opts.settle !== false && !runActive,
        });
    }

    /** Patch batch card `.running` without tearing down the list (progress path). */
    _syncBatchRunHighlight() {
        if (!this.isImageBatch?.() || !this.batchList) return;
        const runningIdx = this._runHighlightSeg;
        this.batchList.querySelectorAll(".bd-batch-card").forEach((card) => {
            const i = parseInt(card.dataset.batchIndex, 10);
            card.classList.toggle("running", Number.isFinite(i) && i === runningIdx);
        });
        this.batchPicker?.querySelectorAll?.(".bd-batch-pick").forEach((chip) => {
            const i = parseInt(chip.dataset.batchIndex, 10);
            chip.classList.toggle("running", Number.isFinite(i) && i === runningIdx);
        });
        this._syncR2vCardSelection?.();
    }

    scheduleRender() {
        if (this._renderPending) return;
        this._renderPending = true;
        this._resizeRaf = requestAnimationFrame(() => {
            this._renderPending = false;
            if (this.isPlaying) this.renderTimelineOnly();
            else this.render();
        });
    }

    buildTimelinePayload() {
        if (this.isFl2vMode()) {
            const fl = buildFl2vPayloadFields(this);
            const outMode = this.timeline.output?.mode || "long_edge";
            const output = normalizeOutputContinuity({
                ...(this.timeline.output || {}),
                mode: outMode,
            });
            const body = { ...this.timeline };
            stripTimelineContinuityRootFields(body);
            stripTimelineEphemeralFields(body);
            return {
                ...body,
                version: 5,
                ...fl,
                frameRate: this.getFrameRate(),
                global: {
                    ...(this.timeline.global || {}),
                    taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                    prompt: this.timeline.global?.prompt || "",
                },
                output,
                ...this._runSelectionPayload(),
            };
        }
        if (this.isImageBatch()) {
            this._persistCurrentBatchWorkspace();
            const taskKey = this.getTaskKey();
            const i2iSrc = (taskKey === "i2i" || taskKey === "i2v") ? this.getI2iSourceDimensions() : null;
            const outMode = imageBatchRequiresFixedOutput(taskKey)
                ? "fixed"
                : (this.timeline.output?.mode || "long_edge");
            const output = normalizeOutputContinuity({
                ...this.timeline.output,
                mode: outMode,
            });
            if (!isVideoBatchTask(taskKey)) {
                output.exportMode = "all";
            }
            if (i2iSrc?.width > 0 && i2iSrc?.height > 0) {
                output.sourceWidth = i2iSrc.width;
                output.sourceHeight = i2iSrc.height;
            }
            const batchBody = { ...this.timeline };
            stripTimelineContinuityRootFields(batchBody);
            stripTimelineEphemeralFields(batchBody);
            return {
                ...batchBody,
                version: 5,
                timelineMode: "prompt_batch",
                editMode: "segment",
                totalFrames: sumFrameCounts(this.timeline.segments),
                frameRate: this.getFrameRate(),
                width: this.timeline.output?.width,
                height: this.timeline.output?.height,
                global: {
                    ...this.timeline.global,
                    taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                    prompt: this.timeline.global?.prompt || "",
                    ...(i2iSrc?.width > 0 ? { sourceWidth: i2iSrc.width, sourceHeight: i2iSrc.height } : {}),
                },
                output,
                segments: this.timeline.segments.map((s, i) => {
                    const clean = sanitizeSegmentForPayload(s);
                    return {
                        id: clean.id,
                        start: clean.start,
                        length: clean.frameCount ?? clean.length ?? 1,
                        frameCount: clean.frameCount ?? clean.length ?? 1,
                        durationSec: clean.durationSec,
                        prompt: clean.prompt || "",
                        negativePrompt: clean.negativePrompt || "",
                        taskType: clean.taskType || "",
                        refs: clean.refs || [],
                        refAudios: clean.refAudios || [],
                        refVideos: clean.refVideos || [],
                        genImage: clean.genImage || { imageFile: "" },
                        // Persist per-segment「引用上段」(default true when unset).
                        continuityFromPrev: isSegmentContinuityFromPrev(clean, i),
                        refImageSize: resolveSegmentRefImageSize(clean, this.timeline.output),
                    };
                }),
                ...this._runSelectionPayload(),
            };
        }
        if (this.isGenMode()) {
            const mode = this.getDirectorMode();
            const genBody = { ...this.timeline };
            stripTimelineContinuityRootFields(genBody);
            stripTimelineEphemeralFields(genBody);
            return {
                ...genBody,
                version: 5,
                timelineMode: mode,
                totalFrames: sumFrameCounts(this.timeline.segments),
                frameRate: this.getFrameRate(),
                width: this.timeline.output?.width,
                height: this.timeline.output?.height,
                refMaxSize: this.timeline.output?.longEdge,
                global: {
                    ...this.timeline.global,
                    taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                    prompt: this.timeline.global?.prompt || "",
                },
                output: normalizeOutputContinuity({ ...this.timeline.output }),
                segments: this.timeline.segments.map((s) => {
                const clean = sanitizeSegmentForPayload(s);
                return {
                    ...clean,
                    frameCount: clean.frameCount ?? clean.length,
                };
            }),
            ...this._runSelectionPayload(),
            };
        }
        this._persistCurrentVideoWorkspace();
        const video = { ...(this.timeline.video || {}) };
        const frameMap = video.frameMap?.length ? video.frameMap : [];
        const src = this.getSourceDimensions();
        const resolved = resolveOutputDimensions(src.width, src.height, this.timeline.output || {}, {
            refMaxSize: this.refMaxWidget?.value,
        });
        const storageW = resolved.width || video.storageWidth || this._storageWidth;
        const storageH = resolved.height || video.storageHeight || this._storageHeight;
        const clips = this.getVideoClips().map((c) => ({
            ...c,
            storageWidth: storageW,
            storageHeight: storageH,
        }));
        const { referenceVideo: _legacyRefVideo, reference_video: _legacyRefVideo2, ...timelineBody } = this.timeline;
        stripTimelineContinuityRootFields(timelineBody);
        stripTimelineEphemeralFields(timelineBody);
        const clipSourceTotal = clips.reduce(
            (s, c) => s + (parseInt(c.sourceFrameCount, 10) || 0),
            0,
        );
        const sourceFrameCount = parseInt(video.sourceFrameCount, 10)
            || clipSourceTotal
            || (frameMap.length ? 0 : this.getTotalFrames());
        return {
            ...timelineBody,
            version: 4,
            timelineMode: "video",
            totalFrames: this.getTotalFrames(),
            frameRate: this.getFrameRate(),
            videoClips: clips,
            global: {
                ...(this.timeline.global || {}),
                taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                prompt: this.timeline.global?.prompt || "",
                referenceVideo: this.timeline.global?.referenceVideo || {},
                continuousReference: !!this.timeline.global?.continuousReference,
            },
            segments: (this.timeline.segments || []).map((s) => {
                const clean = sanitizeSegmentForPayload(s);
                return {
                    ...clean,
                    referenceVideo: clean.referenceVideo || {},
                };
            }),
            video: {
                ...video,
                frameMap,
                sourceFrameCount,
                deletedSourceRanges: frameMap.length ? [] : (video.deletedSourceRanges || []),
                frames: (video.videoFile || video.fileName)
                    ? []
                    : (this._legacyFrames.length ? this._legacyFrames : []),
                storageWidth: storageW,
                storageHeight: storageH,
            },
            output: normalizeOutputContinuity({ ...this.timeline.output }),
            ...this._runSelectionPayload(),
        };
    }

    flushTimelineSync() {
        clearTimeout(this._syncTimer);
        this._syncTimer = null;
        // Refresh visibility first so queue flush can pull checkbox state reliably.
        this.updateSegmentContinuityUI();
        this._writeTimelineWidget();
    }

    scheduleTimelineSync() {
        clearTimeout(this._syncTimer);
        this._syncTimer = setTimeout(() => this._writeTimelineWidget(), TIMELINE_SYNC_DEBOUNCE_MS);
    }

    _flushPromptTokenEditors() {
        this.globalPrompt?.__bdTokenApi?.sync?.();
        this.segPrompt?.__bdTokenApi?.sync?.();
        refreshPromptTokenEditors(this.root || document);
    }

    _writeTimelineWidget() {
        if (!this.timelineWidget) return;
        // Token editors keep textarea.value in sync on blur/input; force-flush
        // before serialize so a focused editor cannot drop the latest draft.
        this._flushPromptTokenEditors();
        // Batch prompt textareas can lag behind segment objects after duration
        // normalize — always harvest DOM drafts before serializing timeline_data.
        if (this.isImageBatch?.()) flushBatchPromptInputs(this);
        if (this.isFl2vMode?.()) flushFl2vPromptDraft(this);
        this.syncFromWidgets();
        this.timelineWidget.value = JSON.stringify(this.buildTimelinePayload());
        this.node.setDirtyCanvas(true, false);
    }

    _markNodeDirtyLight() {
        this.node.setDirtyCanvas(true, false);
    }

    buildDOM() {
        this.root = document.createElement("div");
        this.root.className = "bd-wrap";
        this.root.innerHTML = `<style>${STYLES}</style>`;

        const toolbarWrap = document.createElement("div");
        toolbarWrap.className = "bd-toolbar-wrap";
        toolbarWrap.innerHTML = `
            <div class="bd-toolbar">
                <div class="bd-actions">
                    <button type="button" class="bd-btn bd-btn-primary hidden" data-a="r2v-add-group" data-i18n="toolbar.addRefGroup" data-i18n-title="tooltip.addRefGroup">添加素材组</button>
                    <button type="button" class="bd-btn bd-btn-primary" data-a="video" data-i18n="toolbar.uploadVideo">上传视频</button>
                    <button type="button" class="bd-btn" data-a="video-existing" data-i18n="mediaPicker.pickExistingVideo" data-i18n-title="mediaPicker.pickExistingHint">选已有视频</button>
                    <button type="button" class="bd-btn bd-btn-primary hidden" data-a="fl2v-add-shot" data-i18n="toolbar.addShot" data-i18n-title="tooltip.addShot">添加一组</button>
                    <button type="button" class="bd-btn" data-a="video-append" data-i18n="toolbar.appendVideo" data-i18n-title="tooltip.appendVideo">追加视频</button>
                    <button type="button" class="bd-btn" data-a="split" data-i18n="toolbar.split">+ 分割</button>
                    <input type="number" class="bd-num" data-r="equal-n" min="2" max="64" value="2" data-i18n-title="tooltip.equalSplitN">
                    <button type="button" class="bd-btn" data-a="equal" data-i18n="toolbar.equalSplit">均分</button>
                    <button type="button" class="bd-btn" data-a="smart-split" data-i18n="toolbar.smartSplit" data-i18n-title="tooltip.smartSplit">智能分割</button>
                    <button type="button" class="bd-btn" data-a="run-select-toggle" data-i18n="toolbar.runSelect" data-i18n-title="tooltip.runSelect">选择运行</button>
                    <label class="bd-run-select-all-wrap hidden" data-r="run-select-all-wrap" data-i18n-title="tooltip.runSelectAll">
                        <input type="checkbox" data-r="run-select-all-cb">
                        <span data-i18n="toolbar.selectAll">全选</span>
                    </label>
                    <button type="button" class="bd-btn bd-btn-danger" data-a="del" data-i18n="toolbar.deleteSegment" data-i18n-title="tooltip.deleteSegment">删除片段</button>
                    <div class="bd-mode">
                        <button type="button" data-a="mode-global" class="active" data-i18n="toolbar.modeGlobal">全局模式</button>
                        <button type="button" data-a="mode-segment" data-i18n="toolbar.modeSegment">分段模式</button>
                    </div>
                    <select class="bd-select" data-r="global-task" title="task_type"></select>
                    <span class="bd-video-tag" data-r="video-name" data-i18n="toolbar.noVideo">未上传视频</span>
                </div>
                <div class="bd-right">
                    <div class="bd-tl-zoom" data-r="tl-zoom">
                        <button type="button" class="bd-btn bd-btn-zoom" data-a="zoom-toggle" data-i18n="toolbar.timelineZoom" data-i18n-title="toolbar.timelineZoomTitle">放大</button>
                        <input type="range" class="bd-tl-zoom-slider hidden" data-r="zoom" min="1" max="10" step="any" value="1" data-i18n-title="tooltip.timelineZoom">
                    </div>
                    <button type="button" class="bd-btn" data-a="lang-toggle" data-i18n="toolbar.langToggle" data-i18n-title="toolbar.langToggleTitle">EN</button>
                    <div class="bd-bounds" data-r="bounds">起点: 0.00 | 终点: -</div>
                    <div class="bd-timecode" data-r="timecode">0.00s</div>
                </div>
            </div>
            <div class="bd-smart-split-msg hidden" data-r="smart-split-msg" role="status"></div>
            <div class="bd-external-groups-msg hidden" data-r="external-groups-msg" role="status"></div>`;
        this.root.appendChild(toolbarWrap);
        this.smartSplitMsgEl = toolbarWrap.querySelector('[data-r="smart-split-msg"]');
        this.externalGroupsMsgEl = toolbarWrap.querySelector('[data-r="external-groups-msg"]');
        this.langToggleBtn = toolbarWrap.querySelector('[data-a="lang-toggle"]');

        this.mainBody = document.createElement("div");
        this.mainBody.className = "bd-main";
        this.root.appendChild(this.mainBody);

        const stage = document.createElement("div");
        stage.className = "bd-stage hidden";
        stage.setAttribute("data-r", "video-stage");
        stage.innerHTML = `
            <video class="bd-stage-video hidden" data-r="stage-video" muted playsinline preload="auto"></video>
            <img class="bd-stage-img hidden" data-r="stage-img" alt="">
            <div class="bd-stage-empty" data-r="stage-empty" data-i18n="stage.empty">上传视频后可在此预览播放</div>
            <div class="bd-stage-badge hidden" data-r="stage-badge"></div>`;
        this.mainBody.appendChild(stage);

        // Playback bar sits between video stage and timeline edit area.
        const controls = document.createElement("div");
        controls.className = "bd-controls";
        controls.innerHTML = `
            <div class="bd-player">
                <button type="button" class="bd-icon-btn" data-a="play" data-i18n-title="player.playPause">▶</button>
                <button type="button" class="bd-icon-btn" data-a="loop" data-i18n-title="player.loopOn">⟳</button>
                <button type="button" class="bd-icon-btn" data-a="frame-prev" data-i18n-title="player.framePrev">‹</button>
                <button type="button" class="bd-icon-btn" data-a="frame-next" data-i18n-title="player.frameNext">›</button>
                <span class="bd-frame-jump" data-i18n-title="player.frameJump">
                    <span data-i18n="player.frame">帧</span>
                    <input type="number" class="bd-frame-input" data-r="frame-input" min="1" step="1" value="1">
                    <span>/</span>
                    <span class="bd-frame-total" data-r="frame-total">0</span>
                </span>
                <div class="bd-timecode" data-r="player-timecode">0.00 / 0.00</div>
                <input type="range" class="bd-seek" data-r="seek" min="0" value="0" step="1">
            </div>`;
        this.mainBody.appendChild(controls);

        // Appears above the timeline when a split point is selected.
        const splitEditBar = document.createElement("div");
        splitEditBar.className = "bd-split-edit-bar hidden";
        splitEditBar.setAttribute("data-r", "split-edit-bar");
        splitEditBar.innerHTML = `
            <span class="bd-split-edit-hint" data-r="split-edit-hint" data-i18n="split.selectedHint">已选中分割点</span>
            <button type="button" class="bd-btn bd-btn-del-split" data-a="del-split" data-i18n="toolbar.deleteSplitPoint" data-i18n-title="tooltip.deleteSplitPoint">删除分割点</button>`;
        this.mainBody.appendChild(splitEditBar);
        this.splitEditBarEl = splitEditBar;
        this.splitEditHintEl = splitEditBar.querySelector('[data-r="split-edit-hint"]');

        this.viewport = document.createElement("div");
        this.viewport.className = "bd-viewport";
        this.canvas = document.createElement("canvas");
        this.canvas.className = "bd-canvas";
        this.viewport.appendChild(this.canvas);
        this.mainBody.appendChild(this.viewport);
        this.ctx = this.canvas.getContext("2d");

        const outputBar = document.createElement("div");
        outputBar.className = "bd-output";
        outputBar.innerHTML = `
            <span class="bd-fl2v-total-wrap hidden" data-r="fl2v-total-wrap" data-i18n-title="tooltip.fl2vTotalDuration">
                <label data-i18n="output.totalDurationSec">总时长（秒）</label>
                <input type="number" class="bd-num" data-r="fl2v-total" min="1" max="99999" step="0.1" value="5" style="width:64px" disabled data-i18n-title="tooltip.fl2vTotalInput">
            </span>
            <label data-i18n="output.resolution">输出分辨率</label>
            <select class="bd-select" data-r="out-aspect" data-i18n-title="tooltip.aspectRatio" style="max-width:200px">
                ${RESOLUTION_ASPECTS.map(([label]) => `<option value="${label}"${label === DEFAULT_ASPECT_RATIO ? " selected" : ""}>${aspectDisplayLabel(label)}</option>`).join("")}
                <option value="${CUSTOM_ASPECT_RATIO}">${aspectDisplayLabel(CUSTOM_ASPECT_RATIO)}</option>
            </select>
            <span class="bd-out-mp-wrap" data-r="out-mp-wrap" data-i18n-title="tooltip.megapixels">
                <label data-i18n="output.megapixels">百万像素</label>
                <input type="number" class="bd-num" data-r="out-mp" min="0.1" max="16" step="0.1" value="${DEFAULT_MEGAPIXELS}" style="width:56px">
            </span>
            <span class="bd-out-long hidden" data-r="out-long-wrap">
                <label data-i18n="output.longEdge">最长边</label>
                <input type="number" class="bd-num" data-r="out-long" min="32" max="8192" step="1" value="864" style="width:56px" data-i18n-title="tooltip.longEdge">
            </span>
            <span class="bd-out-fixed hidden" data-r="out-fixed-wrap" data-i18n-title="tooltip.customWH">
                <label data-i18n="output.width">宽</label>
                <input type="number" class="bd-num" data-r="out-w" min="32" max="8192" step="32" value="864" style="width:56px">
                <label data-i18n="output.height">高</label>
                <input type="number" class="bd-num" data-r="out-h" min="32" max="8192" step="32" value="480" style="width:56px">
            </span>
            <select class="bd-select hidden" data-r="out-mode" data-i18n-title="tooltip.outputMode">
                <option value="long_edge" data-i18n="output.mode.longEdge">最长边缩放</option>
                <option value="fixed" data-i18n="output.mode.fixed">固定宽高</option>
            </select>
            <label data-i18n="output.fpsLabel" data-i18n-title="tooltip.fps">帧率</label>
            <input type="number" class="bd-num" data-r="timeline-fps" min="1" max="240" step="0.01" value="24" style="width:64px" data-i18n-title="tooltip.timelineFps">
            <span class="bd-out-audio-wrap hidden" data-r="out-audio-wrap" data-i18n-title="tooltip.audioMode">
                <label data-i18n="output.audio.label">声音</label>
                <select class="bd-select" data-r="out-audio-mode" style="max-width:120px">
                    <option value="generate" data-i18n="output.audio.generate">生成声音</option>
                    <option value="source" data-i18n="output.audio.source">使用原声</option>
                    <option value="mute" data-i18n="output.audio.mute">静音</option>
                </select>
            </span>
            <span class="bd-out-source-wrap hidden" data-r="out-source-wrap" data-i18n-title="widget.tooltip.exportSourceImages">
                <label>
                    <input type="checkbox" data-r="out-export-source">
                    <span data-i18n="output.exportSourceImages">输出原片</span>
                </label>
            </span>
            <span class="bd-meta" data-r="out-preview">—</span>
            <span class="bd-meta hidden" data-r="out-hint"></span>
            <label data-i18n="output.exportMode.label" data-i18n-title="tooltip.exportMode">导出方式</label>
            <select class="bd-select" data-r="out-export-mode" data-i18n-title="tooltip.exportMode">
                <option value="all" data-i18n="output.exportMode.all">全部导出</option>
                <option value="segments" data-i18n="output.exportMode.segments">分段导出</option>
            </select>
            <span class="hidden" data-r="out-max-frames-wrap" hidden aria-hidden="true">
                <label data-i18n="output.maxFrames">最大帧数</label>
                <input type="number" class="bd-num" data-r="out-max-frames" min="0" max="999999" step="1" value="0" style="width:64px">
            </span>
            <span class="bd-continuous-ref hidden" data-r="segment-continuity-wrap" hidden aria-hidden="true" title="">
                <label><input type="checkbox" data-r="segment-continuity-cb"><span data-i18n="output.segmentContinuity">段间引导</span></label>
                <span class="bd-meta" data-i18n="output.continuityOverlap">上下文帧数</span>
                <select class="bd-num" data-r="segment-continuity-overlap" style="width:64px">
                    <option value="5">5</option>
                    <option value="22" selected>22</option>
                    <option value="39">39</option>
                    <option value="56">56</option>
                </select>
            </span>
            <button type="button" class="bd-btn bd-btn-live-preview" data-a="live-tae-preview" data-i18n="toolbar.liveTaePreview" data-i18n-title="tooltip.liveTaePreview">实时预览</button>`;
        this.mainBody.appendChild(outputBar);
        this.outputBarEl = outputBar;

        const liveSample = document.createElement("div");
        liveSample.className = "bd-live-sample hidden";
        liveSample.setAttribute("data-r", "live-sample");
        liveSample.innerHTML = `
            <div class="bd-live-sample-head">
                <b data-i18n="liveSample.title">采样预览</b>
                <span class="bd-meta" data-r="live-sample-meta" data-i18n="liveSample.idleHint">开启后，采样过程中显示实时画面</span>
            </div>
            <div class="bd-live-sample-body">
                <img class="hidden" data-r="live-sample-img" alt="live preview">
                <div class="bd-live-sample-empty" data-r="live-sample-empty" data-i18n="liveSample.waiting">等待采样…</div>
                <div class="bd-live-sample-badge hidden" data-r="live-sample-badge"></div>
            </div>`;
        this.mainBody.appendChild(liveSample);
        this.liveSampleEl = liveSample;
        this.liveSampleImg = liveSample.querySelector('[data-r="live-sample-img"]');
        this.liveSampleEmpty = liveSample.querySelector('[data-r="live-sample-empty"]');
        this.liveSampleBadge = liveSample.querySelector('[data-r="live-sample-badge"]');
        this.liveSampleMeta = liveSample.querySelector('[data-r="live-sample-meta"]');
        this._liveSampleHost = "main";

        const bottom = document.createElement("div");
        bottom.className = "bd-split";
        bottom.innerHTML = `
            <div class="bd-panel" data-r="global-panel">
                <div class="bd-r2v-common-head" data-r="r2v-common-head">
                    <div class="bd-r2v-common-titles">
                        <b data-r="global-panel-title" data-i18n="panel.globalPromptAndRefs">全局提示词 & 参考图 (图片1–9)</b>
                        <span class="bd-r2v-common-status" data-r="r2v-common-status" data-i18n="panel.r2vCommonOff">未启用 · 各组独立素材与提示词</span>
                    </div>
                    <div class="bd-r2v-common-actions">
                        <button type="button" class="bd-btn bd-r2v-common-fold hidden" data-r="r2v-common-fold" data-i18n="panel.r2vCommonCollapse">收起公共参数</button>
                        <button type="button" class="bd-btn bd-r2v-common-toggle" data-r="r2v-common-toggle" data-i18n="panel.r2vCommonEnable">启用公共参数</button>
                    </div>
                </div>
                <div class="bd-r2v-common-body" data-r="r2v-common-body">
                    <div class="bd-meta bd-r2v-common-hint hidden" data-r="r2v-common-hint" data-i18n="panel.r2vCommonHint">公共参考图/视频/音频供各组读取；公共提示词会与每组提示词拼接成完整提示词。同槽位以组内素材优先。</div>
                    <div class="bd-prompt-layout" data-r="global-prompt-layout">
                        <div class="bd-refs-col" data-r="global-refs-col">
                            <div class="bd-refs-images-wrap" data-r="global-refs-images-wrap">
                                <div class="bd-r2v-section-head" data-r="global-refs-head">
                                    <span class="bd-label bd-r2v-section-title" data-r="global-refs-label" data-i18n="panel.refImages">参考图 (图片1–9)</span>
                                    <span class="bd-r2v-section-actions">
                                        <button type="button" class="bd-r2v-pick-existing" data-r="global-refs-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                        <span class="bd-r2v-section-count" data-r="global-refs-count"></span>
                                    </span>
                                </div>
                                <div class="bd-refs" data-r="global-refs"></div>
                            </div>
                            <div class="bd-ref-videos-wrap hidden" data-r="global-ref-videos-wrap">
                                <div class="bd-r2v-section-head" data-r="global-videos-head">
                                    <span class="bd-label bd-r2v-section-title" data-i18n="batch.r2v.sectionVideos">参考视频</span>
                                    <span class="bd-r2v-section-actions">
                                        <button type="button" class="bd-r2v-pick-existing" data-r="global-videos-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                        <span class="bd-r2v-section-count" data-r="global-videos-count"></span>
                                    </span>
                                </div>
                                <div class="bd-ref-videos" data-r="global-ref-videos"></div>
                            </div>
                            <div class="bd-ref-audios-wrap hidden" data-r="global-ref-audios-wrap">
                                <div class="bd-r2v-section-head" data-r="global-audios-head">
                                    <span class="bd-label bd-r2v-section-title" data-i18n="batch.r2v.sectionAudios">参考音频</span>
                                    <span class="bd-r2v-section-actions">
                                        <button type="button" class="bd-r2v-pick-existing" data-r="global-audios-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                        <span class="bd-r2v-section-count" data-r="global-audios-count"></span>
                                    </span>
                                </div>
                                <div class="bd-ref-audios" data-r="global-ref-audios"></div>
                            </div>
                            <div class="bd-ref-video-col hidden" data-r="global-ref-video-col">
                                <span class="bd-label" data-i18n="panel.refVideo">参考视频（植入内容）</span>
                                <div class="bd-gen-src" data-r="global-ref-video" data-i18n="panel.uploadRefVideo" data-i18n-title="tooltip.uploadRefVideo">点击上传参考视频</div>
                                <span class="bd-meta bd-ref-video-name" data-r="global-ref-video-name"></span>
                                <label class="bd-continuous-ref hidden" data-r="continuous-ref-wrap" data-i18n-title="tooltip.continuousRef">
                                    <input type="checkbox" data-r="continuous-ref-cb">
                                    <span data-i18n="panel.continuousRef">连续参考</span>
                                </label>
                            </div>
                            <div class="bd-gen-src hidden" data-r="gen-global-img" data-i18n="panel.uploadSourceImage" data-i18n-title="tooltip.uploadSourceImage">点击上传源图片</div>
                        </div>
                        <div class="bd-prompt-col">
                            <span class="bd-label" data-i18n="panel.prompt">提示词</span>
                            <textarea class="bd-prompt" data-r="global-prompt" data-i18n-placeholder="placeholder.globalPrompt" placeholder=""></textarea>
                            <textarea class="bd-prompt bd-prompt-negative hidden" data-r="global-negative" hidden aria-hidden="true"></textarea>
                        </div>
                    </div>
                    <div class="bd-gen-fc-row hidden" data-r="gen-global-fc-row">
                        <span class="bd-label" data-i18n="panel.defaultSegmentFrames">默认片段帧数</span>
                        <input type="number" class="bd-num" data-r="gen-default-fc" min="1" max="${MAX_GEN_FRAMES}" value="124" style="width:72px">
                    </div>
                </div>
            </div>
            <div class="bd-panel" data-r="segment-panel" style="display:none">
                <div class="bd-seg-head">
                    <b data-r="seg-label">片段 1</b>
                    <label class="bd-seg-continuity hidden" data-r="seg-continuity-from-prev-wrap" hidden>
                        <input type="checkbox" data-r="seg-continuity-from-prev">
                        <span data-i18n="batch.continuityFromPrev">引用上段</span>
                    </label>
                    <div class="bd-meta" data-r="seg-info"></div>
                    <label class="bd-seg-refsize hidden" data-r="seg-ref-image-size-wrap" hidden data-i18n-title="tooltip.refImageSize">
                        <span data-i18n="output.refImageSize.label">参考图尺寸</span>
                        <select class="bd-select" data-r="seg-ref-image-size">
                            <option value="match" data-i18n="output.refImageSize.match">match</option>
                            <option value="max" data-i18n="output.refImageSize.max">max</option>
                        </select>
                    </label>
                </div>
                <div class="bd-prompt-layout" data-r="seg-prompt-layout">
                    <div class="bd-refs-col" data-r="seg-refs-col">
                        <div class="bd-refs-images-wrap" data-r="seg-refs-images-wrap">
                            <div class="bd-r2v-section-head" data-r="seg-refs-head">
                                <span class="bd-label bd-r2v-section-title" data-r="seg-refs-label" data-i18n="panel.segmentRefImages">片段参考图 (图片1–9)</span>
                                <span class="bd-r2v-section-actions">
                                    <button type="button" class="bd-r2v-pick-existing" data-r="seg-refs-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                    <span class="bd-r2v-section-count" data-r="seg-refs-count"></span>
                                </span>
                            </div>
                            <div class="bd-refs" data-r="seg-refs"></div>
                        </div>
                        <div class="bd-ref-audios-wrap hidden" data-r="seg-ref-audios-wrap">
                            <div class="bd-r2v-section-head" data-r="seg-audios-head">
                                <span class="bd-label bd-r2v-section-title" data-i18n="batch.r2v.sectionAudios">参考音频</span>
                                <span class="bd-r2v-section-actions">
                                    <button type="button" class="bd-r2v-pick-existing" data-r="seg-audios-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                    <span class="bd-r2v-section-count" data-r="seg-audios-count"></span>
                                </span>
                            </div>
                            <div class="bd-ref-audios" data-r="seg-ref-audios"></div>
                        </div>
                        <div class="bd-ref-video-col hidden" data-r="seg-ref-video-col">
                            <span class="bd-label" data-i18n="panel.segmentRefVideo">片段参考视频（植入内容）</span>
                            <div class="bd-gen-src" data-r="seg-ref-video" data-i18n="panel.uploadRefVideo" data-i18n-title="tooltip.uploadRefVideo">点击上传参考视频</div>
                            <span class="bd-meta bd-ref-video-name" data-r="seg-ref-video-name"></span>
                        </div>
                        <div class="bd-gen-src hidden" data-r="gen-seg-img" data-i18n="panel.uploadSegmentSourceImage" data-i18n-title="tooltip.uploadSourceImage">点击上传源图片</div>
                    </div>
                    <div class="bd-prompt-col">
                        <span class="bd-label" data-i18n="panel.prompt">提示词</span>
                        <textarea class="bd-prompt" data-r="seg-prompt" data-i18n-placeholder="placeholder.segmentPrompt" placeholder=""></textarea>
                        <textarea class="bd-prompt bd-prompt-negative hidden" data-r="seg-negative" hidden aria-hidden="true"></textarea>
                    </div>
                </div>
                <div class="bd-gen-fc-row hidden" data-r="gen-seg-fc-row">
                    <span class="bd-label" data-i18n="panel.segmentFrames">片段帧数</span>
                    <input type="number" class="bd-num" data-r="gen-seg-fc" min="1" max="${MAX_GEN_FRAMES}" value="124" style="width:72px">
                </div>
            </div>`;
        this.mainBody.appendChild(bottom);

        const batchUi = mountImageBatchPanel(this.mainBody);
        this.batchPanel = batchUi.panel;
        this.batchList = batchUi.list;
        this.batchHint = batchUi.hint;
        this.batchI2vNotice = batchUi.i2vNotice;
        this.batchAddBtn = batchUi.addBtn;
        this.batchPicker = batchUi.picker;
        this.batchDetailModeBtn = batchUi.detailModeBtn;
        wireBatchRunSelectControls(this, batchUi);

        this.fl2vUi = mountFl2vPanel(this.mainBody);
        this.fl2vTotalWrap = this.root.querySelector('[data-r="fl2v-total-wrap"]');
        if (this.fl2vUi) {
            this.fl2vUi.totalInput = this.root.querySelector('[data-r="fl2v-total"]');
        }
        bindFl2vEvents(this);

        const runStatus = document.createElement("div");
        runStatus.className = "bd-run-status idle";
        runStatus.dataset.r = "run-status";
        runStatus.innerHTML = `
            <div class="bd-run-title" data-r="run-title" data-i18n="run.titleIdle">运行状态：待命</div>
            <div class="bd-run-detail" data-r="run-detail" data-i18n="run.detailIdle">队列执行时将显示当前片段与阶段进度</div>
            <div class="bd-run-select-bar hidden" data-r="run-select-bar">
                <span data-r="run-select-summary" data-i18n="run.summaryAllSegments">将运行全部片段</span>
            </div>
            <div class="bd-run-bars">
                <div class="bd-run-bar" data-i18n-title="run.bar.overall"><div class="bd-run-bar-fill" data-r="run-overall" style="width:0%"></div></div>
                <div class="bd-run-bar bd-run-bar-sub" data-i18n-title="run.bar.phase"><div class="bd-run-bar-fill" data-r="run-phase" style="width:0%"></div></div>
            </div>`;
        this.root.appendChild(runStatus);

        if (this.container) {
            for (const wrap of [...this.container.querySelectorAll(":scope > .bd-wrap")]) {
                wrap.remove();
            }
            this.container.appendChild(this.root);
        }

        this._previewVideo = document.createElement("video");
        this._previewVideo.crossOrigin = "anonymous";
        this._previewVideo.muted = true;
        this._previewVideo.playsInline = true;
        this._previewVideo.preload = "auto";
        this._previewVideo.style.cssText = "position:fixed;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none";
        document.body.appendChild(this._previewVideo);

        this._thumbCanvas = document.createElement("canvas");
        this._thumbCtx = this._thumbCanvas.getContext("2d", { alpha: false });

        this.videoNameEl = this.root.querySelector('[data-r="video-name"]');
        this.equalCountInput = this.root.querySelector('[data-r="equal-n"]');
        this.boundsEl = this.root.querySelector('[data-r="bounds"]');
        this.timecodeEl = this.root.querySelector('[data-r="timecode"]');
        this.playerTimecodeEl = this.root.querySelector('[data-r="player-timecode"]');
        this.frameInputEl = this.root.querySelector('[data-r="frame-input"]');
        this.frameTotalEl = this.root.querySelector('[data-r="frame-total"]');
        this.seekBar = this.root.querySelector('[data-r="seek"]');
        this.tlZoomWrap = this.root.querySelector('[data-r="tl-zoom"]');
        this.zoomToggleBtn = this.root.querySelector('[data-a="zoom-toggle"]');
        this.zoomSlider = this.root.querySelector('[data-r="zoom"]');
        this.stageEl = this.root.querySelector('[data-r="video-stage"]');
        this.stageVideo = this.root.querySelector('[data-r="stage-video"]');
        this.stageImg = this.root.querySelector('[data-r="stage-img"]');
        this.stageEmpty = this.root.querySelector('[data-r="stage-empty"]');
        this.stageBadge = this.root.querySelector('[data-r="stage-badge"]');
        if (this.stageVideo) {
            this.stageVideo.crossOrigin = "anonymous";
            this.stageVideo.muted = true;
            this.stageVideo.playsInline = true;
        }
        this.globalTask = this.root.querySelector('[data-r="global-task"]');
        this.globalPanel = this.root.querySelector('[data-r="global-panel"]');
        this.globalPanelTitle = this.globalPanel?.querySelector('[data-r="global-panel-title"]')
            || this.globalPanel?.querySelector("b");
        this.r2vCommonHead = this.root.querySelector('[data-r="r2v-common-head"]');
        this.r2vCommonBody = this.root.querySelector('[data-r="r2v-common-body"]');
        this.r2vCommonHint = this.root.querySelector('[data-r="r2v-common-hint"]');
        this.r2vCommonStatus = this.root.querySelector('[data-r="r2v-common-status"]');
        this.r2vCommonFold = this.root.querySelector('[data-r="r2v-common-fold"]');
        this.r2vCommonToggle = this.root.querySelector('[data-r="r2v-common-toggle"]');
        this.segmentPanel = this.root.querySelector('[data-r="segment-panel"]');
        this.globalPrompt = this.root.querySelector('[data-r="global-prompt"]');
        this.globalNegative = this.root.querySelector('[data-r="global-negative"]');
        this.globalPromptLayout = this.root.querySelector('[data-r="global-prompt-layout"]');
        this.segPromptLayout = this.root.querySelector('[data-r="seg-prompt-layout"]');
        this.globalRefsBox = this.root.querySelector('[data-r="global-refs"]');
        this.globalRefsImagesWrap = this.root.querySelector('[data-r="global-refs-images-wrap"]');
        this.globalRefsCount = this.root.querySelector('[data-r="global-refs-count"]');
        this.globalAudiosCount = this.root.querySelector('[data-r="global-audios-count"]');
        this.segRefsImagesWrap = this.root.querySelector('[data-r="seg-refs-images-wrap"]');
        this.segRefsCount = this.root.querySelector('[data-r="seg-refs-count"]');
        this.segAudiosCount = this.root.querySelector('[data-r="seg-audios-count"]');
        this.globalRefAudiosWrap = this.root.querySelector('[data-r="global-ref-audios-wrap"]');
        this.globalRefAudiosBox = this.root.querySelector('[data-r="global-ref-audios"]');
        this.globalRefVideosWrap = this.root.querySelector('[data-r="global-ref-videos-wrap"]');
        this.globalRefVideosBox = this.root.querySelector('[data-r="global-ref-videos"]');
        this.globalVideosCount = this.root.querySelector('[data-r="global-videos-count"]');
        this.segRefAudiosWrap = this.root.querySelector('[data-r="seg-ref-audios-wrap"]');
        this.segRefAudiosBox = this.root.querySelector('[data-r="seg-ref-audios"]');
        this.segLabel = this.root.querySelector('[data-r="seg-label"]');
        this.segContinuityFromPrevWrap = this.root.querySelector('[data-r="seg-continuity-from-prev-wrap"]');
        this.segContinuityFromPrevCb = this.root.querySelector('[data-r="seg-continuity-from-prev"]');
        this.segRefImageSizeWrap = this.root.querySelector('[data-r="seg-ref-image-size-wrap"]');
        this.segRefImageSize = this.root.querySelector('[data-r="seg-ref-image-size"]');
        this.segInfo = this.root.querySelector('[data-r="seg-info"]');
        this.segPrompt = this.root.querySelector('[data-r="seg-prompt"]');
        this.segNegative = this.root.querySelector('[data-r="seg-negative"]');
        this.segRefsBox = this.root.querySelector('[data-r="seg-refs"]');
        this.globalRefsCol = this.root.querySelector('[data-r="global-refs-col"]');
        this.segRefsCol = this.root.querySelector('[data-r="seg-refs-col"]');
        this.globalRefVideoCol = this.root.querySelector('[data-r="global-ref-video-col"]');
        this.globalRefVideo = this.root.querySelector('[data-r="global-ref-video"]');
        this.globalRefVideoNameEl = this.root.querySelector('[data-r="global-ref-video-name"]');
        this.segRefVideoCol = this.root.querySelector('[data-r="seg-ref-video-col"]');
        this.segRefVideo = this.root.querySelector('[data-r="seg-ref-video"]');
        this.segRefVideoNameEl = this.root.querySelector('[data-r="seg-ref-video-name"]');
        this.continuousRefWrap = this.root.querySelector('[data-r="continuous-ref-wrap"]');
        this.continuousRefCb = this.root.querySelector('[data-r="continuous-ref-cb"]');
        this.genGlobalImg = this.root.querySelector('[data-r="gen-global-img"]');
        this.genSegImg = this.root.querySelector('[data-r="gen-seg-img"]');
        this.genGlobalFcRow = this.root.querySelector('[data-r="gen-global-fc-row"]');
        this.genSegFcRow = this.root.querySelector('[data-r="gen-seg-fc-row"]');
        this.genDefaultFc = this.root.querySelector('[data-r="gen-default-fc"]');
        this.genSegFc = this.root.querySelector('[data-r="gen-seg-fc"]');
        this.controlsBar = this.root.querySelector(".bd-controls");
        this.btnVideo = this.root.querySelector('[data-a="video"]');
        this.btnVideoExisting = this.root.querySelector('[data-a="video-existing"]');
        this.btnFl2vAddShot = this.root.querySelector('[data-a="fl2v-add-shot"]');
        this.btnVideoAppend = this.root.querySelector('[data-a="video-append"]');
        this.outHint = this.root.querySelector('[data-r="out-hint"]');
        this.outMode = this.root.querySelector('[data-r="out-mode"]');
        this.outAspect = this.root.querySelector('[data-r="out-aspect"]');
        this.outMpWrap = this.root.querySelector('[data-r="out-mp-wrap"]');
        this.outMp = this.root.querySelector('[data-r="out-mp"]');
        this.outLongWrap = this.root.querySelector('[data-r="out-long-wrap"]');
        this.outFixedWrap = this.root.querySelector('[data-r="out-fixed-wrap"]');
        this.outLong = this.root.querySelector('[data-r="out-long"]');
        this.outW = this.root.querySelector('[data-r="out-w"]');
        this.outH = this.root.querySelector('[data-r="out-h"]');
        this.fpsInput = this.root.querySelector('[data-r="timeline-fps"]');
        this.outAudioWrap = this.root.querySelector('[data-r="out-audio-wrap"]');
        this.outAudioMode = this.root.querySelector('[data-r="out-audio-mode"]');
        this.exportSourceImagesWrap = this.root.querySelector('[data-r="out-source-wrap"]');
        this.exportSourceImagesCb = this.root.querySelector('[data-r="out-export-source"]');
        this.outMaxFrames = this.root.querySelector('[data-r="out-max-frames"]');
        this.outExportMode = this.root.querySelector('[data-r="out-export-mode"]');
        this.segmentContinuityWrap = this.root.querySelector('[data-r="segment-continuity-wrap"]');
        this.segmentContinuityCb = this.root.querySelector('[data-r="segment-continuity-cb"]');
        this.segmentContinuityOverlap = this.root.querySelector('[data-r="segment-continuity-overlap"]');
        this.outPreview = this.root.querySelector('[data-r="out-preview"]');
        this.runStatusEl = this.root.querySelector('[data-r="run-status"]');
        this.runTitleEl = this.root.querySelector('[data-r="run-title"]');
        this.runDetailEl = this.root.querySelector('[data-r="run-detail"]');
        this.runOverallEl = this.root.querySelector('[data-r="run-overall"]');
        this.runPhaseEl = this.root.querySelector('[data-r="run-phase"]');
        this.runSelectBar = this.root.querySelector('[data-r="run-select-bar"]');
        this.runSelectSummary = this.root.querySelector('[data-r="run-select-summary"]');
        this.btnRunSelectToggle = this.root.querySelector('[data-a="run-select-toggle"]');
        this.runSelectAllWrap = this.root.querySelector('[data-r="run-select-all-wrap"]');
        this.runSelectAllCb = this.root.querySelector('[data-r="run-select-all-cb"]');

        this.populateTaskSelect(this.globalTask, this.taskTypeWidget?.value);
        this.syncNegativeFromWidget();
        this.syncOutputUIFromTimeline();
        bindImageBatchEvents(this);
    }

    renderImageBatchGroups() {
        renderImageBatchGroups(this);
    }

    normalizeImageBatchSegments() {
        normalizeImageBatchSegments(this);
    }

    syncNegativeFromWidget() {
        const v = this.negativePromptWidget?.value ?? "";
        if (this.globalNegative) this.globalNegative.value = v;
        if (this.segNegative) this.segNegative.value = v;
    }

    bindEvents() {
        const bind = (sel, fn) => {
            const el = this.root.querySelector(sel);
            if (!el) return;
            el.onclick = (e) => { stopDomEvent(e); fn(); };
        };
        bind('[data-a="video"]', () => this.pickVideoFile());
        bind('[data-a="video-existing"]', () => { void this.pickExistingVideoFile(); });
        bind('[data-a="fl2v-add-shot"]', () => openFl2vUpload(this));
        bind('[data-a="r2v-add-group"]', () => addImageBatchGroup(this));
        bind('[data-a="video-append"]', () => this.pickAppendVideoFile());
        bind('[data-a="split"]', () => this.splitAtFrame(this.currentFrame));
        bind('[data-a="equal"]', () => this.equalSplit());
        bind('[data-a="smart-split"]', () => { void this.smartSplit(); });
        bind('[data-a="del-split"]', () => this.deleteSelectedSplitPoint());
        bind('[data-a="run-select-toggle"]', () => this.toggleRunSelectMode());
        bind('[data-a="del"]', () => this.deleteSelectedSegment());
        bind('[data-a="mode-global"]', () => this.setEditMode("global"));
        bind('[data-a="mode-segment"]', () => this.setEditMode("segment"));
        bind('[data-a="lang-toggle"]', () => toggleLocale());
        bind('[data-a="zoom-toggle"]', () => this.toggleTimelineZoom());
        bind('[data-a="play"]', () => this.togglePlay());
        bind('[data-a="loop"]', () => this.toggleLoop());
        bind('[data-a="live-tae-preview"]', () => this.toggleLiveTaePreview());
        bind('[data-a="frame-prev"]', () => this.stepFrame(-1));
        bind('[data-a="frame-next"]', () => this.stepFrame(1));
        this.refreshLiveTaePreviewButton();
        this.updateLiveSamplePanel();

        this.seekBar.oninput = () => {
            this.seekToFrame(+this.seekBar.value, { fromUi: true });
        };
        if (this.frameInputEl) {
            const applyFrameInput = () => {
                const total = this.getTotalFrames();
                if (total < 1) return;
                const raw = parseInt(this.frameInputEl.value, 10);
                if (!Number.isFinite(raw)) {
                    this.frameInputEl.value = String(this.currentFrame + 1);
                    return;
                }
                // UI is 1-based; internal currentFrame is 0-based.
                this.seekToFrame(raw - 1, { fromUi: true });
            };
            this.frameInputEl.addEventListener("keydown", (e) => {
                e.stopPropagation();
                if (e.key === "Enter") {
                    e.preventDefault();
                    applyFrameInput();
                    this.frameInputEl.blur();
                } else if (e.key === "Escape") {
                    e.preventDefault();
                    this.frameInputEl.value = String(this.currentFrame + 1);
                    this.frameInputEl.blur();
                }
            });
            this.frameInputEl.addEventListener("change", applyFrameInput);
            this.frameInputEl.addEventListener("focus", () => {
                if (this.isPlaying) this._stopPlay();
                this.frameInputEl.select();
            });
        }
        if (this.stageBadge) {
            this.stageBadge.title = t("player.badgeJump");
            this.stageBadge.addEventListener("click", (e) => {
                e.stopPropagation();
                if (this.isPlaying) this._stopPlay();
                this.frameInputEl?.focus();
                this.frameInputEl?.select();
            });
        }
        if (this.zoomSlider) {
            const zoomMin = () => Number(this.zoomSlider.min) || 1;
            const zoomMax = () => Number(this.zoomSlider.max) || 10;
            const applySliderZoom = () => {
                if (!this.zoomEnabled) return;
                this.zoom = clamp(+this.zoomSlider.value, zoomMin(), zoomMax());
                this.applyZoomWidth();
                this.scheduleRender();
            };
            const zoomFromClientX = (clientX) => {
                const rect = this.zoomSlider.getBoundingClientRect();
                const min = zoomMin();
                const max = zoomMax();
                const t = rect.width > 1 ? clamp((clientX - rect.left) / rect.width, 0, 1) : 0;
                return min + t * (max - min);
            };
            const scrubZoom = (e) => {
                this.zoomSlider.value = String(zoomFromClientX(e.clientX));
                applySliderZoom();
            };
            this.zoomSlider.oninput = applySliderZoom;
            const onZoomPointerDown = (e) => {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation?.();
                if (!this.zoomEnabled) return;
                if (e.button != null && e.button !== 0) return;
                this._zoomPointerId = e.pointerId;
                try { this.zoomSlider.setPointerCapture(e.pointerId); } catch { /* ignore */ }
                this.zoomSlider.focus({ preventScroll: true });
                scrubZoom(e);
            };
            const onZoomPointerMove = (e) => {
                if (this._zoomPointerId == null || e.pointerId !== this._zoomPointerId) return;
                e.preventDefault();
                e.stopPropagation();
                scrubZoom(e);
            };
            const onZoomPointerUp = (e) => {
                if (this._zoomPointerId == null || e.pointerId !== this._zoomPointerId) return;
                e.stopPropagation();
                this._zoomPointerId = null;
                try { this.zoomSlider.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
            };
            this.zoomSlider.addEventListener("pointerdown", onZoomPointerDown, true);
            this.zoomSlider.addEventListener("pointermove", onZoomPointerMove);
            this.zoomSlider.addEventListener("pointerup", onZoomPointerUp, true);
            this.zoomSlider.addEventListener("pointercancel", onZoomPointerUp, true);
            this.zoomSlider.addEventListener("mousedown", (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, true);
            this.zoomSlider.addEventListener("wheel", (e) => e.stopPropagation(), true);
        }
        this.viewport?.addEventListener("wheel", (e) => {
            if (this.getTimelineZoom() <= 1) return;
            e.stopPropagation();
            if (e.deltaX === 0 && e.deltaY !== 0) {
                e.preventDefault();
                this.viewport.scrollLeft += e.deltaY;
            }
        }, { passive: false });
        if (this.runSelectAllCb) {
            this.runSelectAllCb.onchange = (e) => {
                stopDomEvent(e);
                if (!this.isRunSelectEnabled()) return;
                this.setRunSelectionAll(this.runSelectAllCb.checked);
            };
        }
        this.globalTask.onchange = () => this.onGlobalField("taskType", this.globalTask.value);
        this.globalPrompt.oninput = () => this.onGlobalField("prompt", this.globalPrompt.value);
        if (this.r2vCommonFold) {
            this.r2vCommonFold.onclick = (e) => {
                stopDomEvent(e);
                if (!this.usesR2vCommonPanel() || !this.isR2vCommonEnabled()) return;
                this.timeline.global = this.timeline.global || {
                    refs: [], refAudios: [], prompt: "",
                    commonEnabled: true, commonCollapsed: false,
                };
                this.timeline.global.commonCollapsed = !this.isR2vCommonCollapsed();
                this.syncR2vCommonCollapse();
                this.scheduleTimelineSync();
                this.updateDomWidgetHeight?.();
            };
        }
        if (this.r2vCommonToggle) {
            this.r2vCommonToggle.onclick = (e) => {
                stopDomEvent(e);
                if (!this.usesR2vCommonPanel()) return;
                this.timeline.global = this.timeline.global || {
                    refs: [], refAudios: [], refVideos: [], prompt: "",
                    commonEnabled: false, commonCollapsed: false,
                };
                this.timeline.global.refs = this.timeline.global.refs || [];
                this.timeline.global.refAudios = this.timeline.global.refAudios || [];
                this.timeline.global.refVideos = this.timeline.global.refVideos || [];
                const nextOn = !this.isR2vCommonEnabled();
                this.timeline.global.commonEnabled = nextOn;
                // Enable → expand; disable → collapse and stop runtime merge.
                this.timeline.global.commonCollapsed = !nextOn;
                if (nextOn) {
                    rebaseR2vGroupSlotsForCommon(this);
                }
                // Must refresh visibility + render ref/audio slots (they stay empty until first paint).
                this.updateModeUI();
                this.renderImageBatchGroups?.();
                this.scheduleTimelineSync();
                this.updateDomWidgetHeight?.();
            };
        }
        if (this.continuousRefCb) {
            this.continuousRefCb.onchange = () => {
                this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
                this.timeline.global.continuousReference = !!this.continuousRefCb.checked;
                this.scheduleTimelineSync();
            };
        }
        this.segPrompt.oninput = () => this.onSegField("prompt", this.segPrompt.value);
        this.globalNegative.oninput = () => this.onNegativePrompt(this.globalNegative.value);
        this.segNegative.oninput = () => this.onNegativePrompt(this.segNegative.value);

        mountPromptImageMentions(this);

        this.outMode.onchange = () => this.onOutputField("mode", this.outMode.value);
        if (this.outAspect) {
            this.outAspect.onchange = () => this.onOutputField("aspectRatio", this.outAspect.value);
        }
        if (this.outMp) {
            // Do not coerce incomplete drafts ("0", "0.") — that snaps back to 0.4 mid-typing.
            const applyMp = ({ force = false } = {}) => {
                const parsed = parseMegapixelsInput(this.outMp.value);
                if (parsed == null) {
                    if (!force) return;
                    const restored = clampMegapixels(
                        this.timeline.output?.megapixels ?? DEFAULT_MEGAPIXELS,
                    );
                    this.outMp.value = String(restored);
                    this.onOutputField("megapixels", restored);
                    return;
                }
                this.onOutputField("megapixels", parsed);
            };
            this.outMp.onchange = () => applyMp({ force: true });
            this.outMp.onblur = () => applyMp({ force: true });
            this.outMp.oninput = () => {
                clearTimeout(this._mpInputTimer);
                this._mpInputTimer = setTimeout(() => applyMp({ force: false }), 280);
            };
            this.outMp.addEventListener("keydown", (e) => e.stopPropagation());
        }
        this.outLong.onchange = () => this.onOutputField("longEdge", +this.outLong.value);
        this.outW.onchange = () => this.onOutputField("width", +this.outW.value);
        this.outH.onchange = () => this.onOutputField("height", +this.outH.value);
        this.fpsInput.onchange = () => this.onFrameRateChanged(this.fpsInput.value);
        this.fpsInput.oninput = () => {
            clearTimeout(this._fpsInputTimer);
            this._fpsInputTimer = setTimeout(() => this.onFrameRateChanged(this.fpsInput.value), 350);
        };
        this.outMaxFrames.onchange = () => this.onOutputField("maxExportFrames", +this.outMaxFrames.value);
        this.outExportMode.onchange = () => this.onOutputField("exportMode", this.outExportMode.value);
        if (this.outAudioMode) {
            this.outAudioMode.onchange = () => this.onOutputField("audioMode", this.outAudioMode.value);
        }
        if (this.exportSourceImagesCb) {
            this.exportSourceImagesCb.onchange = () => {
                this.timeline.output = this.timeline.output || {};
                this.timeline.output.exportSourceImages = !!this.exportSourceImagesCb.checked;
                const w = this.widget("export_source_images");
                if (w) w.value = this.timeline.output.exportSourceImages;
                this.commit(true);
            };
        }
        if (this.segRefImageSize) {
            this.segRefImageSize.onchange = () => {
                const seg = this.timeline.segments?.[this.selectedIndex];
                if (!seg) return;
                seg.refImageSize = normalizeRefImageSize(this.segRefImageSize.value);
                this.commit(true);
            };
        }
        if (this.segmentContinuityCb) {
            this.segmentContinuityCb.onchange = () => {
                this.onOutputField("continuityEnabled", this.segmentContinuityCb.checked);
                this.updateSegmentContinuityUI();
            };
        }
        if (this.segmentContinuityOverlap) {
            const applyOverlap = () => this.onOutputField("continuityOverlapFrames", +this.segmentContinuityOverlap.value);
            this.segmentContinuityOverlap.onchange = applyOverlap;
            this.segmentContinuityOverlap.oninput = applyOverlap;
            this.segmentContinuityOverlap.addEventListener("keydown", (e) => e.stopPropagation());
            this.segmentContinuityOverlap.addEventListener("keyup", (e) => e.stopPropagation());
        }
        if (this.segContinuityFromPrevCb) {
            this.segContinuityFromPrevCb.onchange = () => {
                const seg = this.timeline.segments?.[this.selectedIndex];
                if (!seg || this.selectedIndex <= 0) return;
                seg.continuityFromPrev = !!this.segContinuityFromPrevCb.checked;
                this.commit(true);
            };
            this.segContinuityFromPrevWrap?.setAttribute(
                "title",
                t("tooltip.segmentContinuityFromPrev"),
            );
        }

        this.genGlobalImg?.addEventListener("click", (e) => { stopDomEvent(e); this.pickGenSrcImage(true); });
        this.genSegImg?.addEventListener("click", (e) => { stopDomEvent(e); this.pickGenSrcImage(false); });
        this.root.querySelector('[data-r="global-refs-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingRef(true);
        });
        this.root.querySelector('[data-r="seg-refs-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingRef(false);
        });
        this.root.querySelector('[data-r="global-videos-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingR2vCommonVideo();
        });
        this.root.querySelector('[data-r="global-audios-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingRefAudio(true);
        });
        this.root.querySelector('[data-r="seg-audios-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingRefAudio(false);
        });
        this.genDefaultFc?.addEventListener("change", () => this.onGenDefaultFcChange());
        this.genSegFc?.addEventListener("change", () => this.onGenSegFcChange());

        this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
        this.canvas.addEventListener("dblclick", (e) => {
            stopDomEvent(e);
            e.preventDefault();
            // fl2v: double-click still replaces the start frame. Other modes do not split.
            if (!this.isFl2vMode()) return;
            const { x, y } = this.getMousePos(e);
            const hit = this.hitTest(x, y);
            if (hit?.type === "segment" || hit?.type === "edge") {
                const idx = hit.index ?? this.selectedIndex;
                if (idx !== this.selectedIndex) flushFl2vPromptDraft(this);
                this.selectedIndex = idx;
                this.updateSelectionUI();
                updateFl2vDetailUI(this);
                this._fl2vUploadMode = "slot";
                this._fl2vSlotKind = "start";
                this._fl2vSlotShotIndex = idx;
                this.fl2vUi?.fileInput?.click();
            }
        });
        this.canvas.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            if (this.isFl2vMode()) return;
            this.addSplitAtMouse(e);
        });
        this._onMouseMove = (e) => this.onMouseMove(e);
        this._onMouseUp = () => this.onMouseUp();
        this._onCanvasHover = (e) => {
            if (this._drag || this.isPlaying) return;
            const { x, y } = this.getMousePos(e);
            const hit = this.hitTest(x, y);
            this.canvas.classList.remove("bd-grab");
            if (hit?.type === "run-check" || hit?.type === "split" || hit?.type === "continuity-joint") {
                this.canvas.style.cursor = "pointer";
                if (hit.type === "continuity-joint") {
                    this.canvas.title = t(
                        hit.on ? "tooltip.continuityJointOn" : "tooltip.continuityJointOff",
                        { a: hit.a, b: hit.b },
                    );
                } else {
                    this.canvas.title = "";
                }
            } else if (hit?.type === "edge") {
                // Edge drag is always horizontal (change start/length); keep ↔ cursor.
                this.canvas.style.cursor = "ew-resize";
                this.canvas.title = this.isFl2vMode()
                    ? t("tooltip.dragFl2vDuration")
                    : "";
            } else if (this.needsSourceVideoUpload?.() && (hit?.type === "segment" || hit?.type === "edge" || !hit)) {
                this.canvas.style.cursor = "pointer";
                this.canvas.title = t("canvas.clickUploadVideo");
            } else if (hit?.type === "segment" && (this.isFl2vMode() || this.usesBatchTimeline() || this.timeline.segments.length >= 2)) {
                this.canvas.classList.add("bd-grab");
                this.canvas.style.cursor = "";
                this.canvas.title = this.isFl2vMode()
                    ? t("tooltip.dragFl2vSwap")
                    : (this.isR2vBatch()
                        ? t("tooltip.dragR2vOrder")
                        : (this.usesBatchTimeline()
                            ? t("tooltip.dragPromptGroupOrder")
                            : t("tooltip.dragSegmentOrder")));
            } else {
                this.canvas.style.cursor = "";
                this.canvas.title = "";
            }
        };
        window.addEventListener("mousemove", this._onMouseMove);
        window.addEventListener("mouseup", this._onMouseUp);
        this.canvas.addEventListener("mousemove", this._onCanvasHover);
        this.canvas.addEventListener("mouseleave", () => {
            this.canvas.classList.remove("bd-grab");
            this.canvas.style.cursor = "";
            this.canvas.title = "";
        });

        this.root.addEventListener("mouseenter", () => { this._isHovering = true; });
        this.root.addEventListener("mouseleave", () => { this._isHovering = false; });
        this._onKeyDown = (e) => {
            if (!this._isHovering) return;
            const el = document.activeElement;
            const tag = el?.tagName;
            // Prompt editors are contenteditable DIVs (bd-token-editor), not TEXTAREA —
            // must ignore typing shortcuts there or Backspace deletes the asset group.
            if (
                tag === "INPUT"
                || tag === "TEXTAREA"
                || tag === "SELECT"
                || el?.isContentEditable
                || el?.closest?.('[contenteditable="true"], .bd-token-editor')
            ) {
                return;
            }
            if ((e.key === "Delete" || e.key === "Backspace") && this.timeline.segments.length >= 1) {
                // Split points only delete via the toolbar button; Delete removes segments.
                if (this.selectedSplitFrame != null) {
                    e.preventDefault();
                    return;
                }
                this.deleteSelectedSegment();
                e.preventDefault();
            } else if (e.code === "Space") {
                this.togglePlay(); e.preventDefault();
            } else if (e.key === "ArrowLeft") {
                this.stepFrame(e.shiftKey ? -10 : -1);
                e.preventDefault();
            } else if (e.key === "ArrowRight") {
                this.stepFrame(e.shiftKey ? 10 : 1);
                e.preventDefault();
            }
        };
        window.addEventListener("keydown", this._onKeyDown, true);

        this.root.addEventListener("dragover", (e) => e.preventDefault());
        this.root.addEventListener("drop", (e) => {
            e.preventDefault();
            // Slot-to-slot moves are handled on .bd-ref; don't also treat as new upload.
            const types = [...(e.dataTransfer?.types || [])];
            if (types.includes("application/x-minimax-ref-slot")) return;
            if (types.includes("application/x-minimax-fl2v-slot")) return;
            if (types.includes("application/x-minimax-fl2v-shot")) return;
            if (e.target.closest?.(".bd-ref, .bd-batch-ref, .bd-batch-src, .bd-batch-video, .bd-batch-audio, .bd-batch-videos, .bd-batch-audios, .bd-fl2v-slot, .bd-fl2v-shot")) return;
            const f = e.dataTransfer.files?.[0];
            if (f?.type.startsWith("video/")) this.loadVideoFile(f);
            else if (f?.type.startsWith("image/")) {
                if (this.isImageBatch?.() && e.target.closest?.(".bd-batch-ref")) return;
                if (this.isImageBatch?.()) return;
                this.addRefFromFile(f, this.getRefTarget());
            }
        });
    }

    destroy() {
        clearTimeout(this._syncTimer);
        clearTimeout(this._settleRenderTimer);
        clearTimeout(this._settleRenderLateTimer);
        clearTimeout(this._promptRenderTimer);
        this._promptRenderTimer = null;
        this._settleRenderTimer = null;
        this._settleRenderLateTimer = null;
        cancelAnimationFrame(this._resizeRaf);
        cancelAnimationFrame(this._playRaf);
        this._resizeObserver?.disconnect();
        this._unsubLocale?.();
        this._unsubLocale = null;
        this._closeBdModal();
        teardownPromptImageMentions(this.root);
        this._clearPreviewVideos(true);
        this._previewVideos?.clear();
        try {
            this._previewVideo?.pause();
            this._previewVideo?.removeAttribute("src");
            this._previewVideo?.load();
        } catch { /* ignore */ }
        this._previewVideo?.remove();
        this._previewVideo = null;
        window.removeEventListener("mousemove", this._onMouseMove);
        window.removeEventListener("mouseup", this._onMouseUp);
        this.canvas?.removeEventListener("mousemove", this._onCanvasHover);
        this.canvas?.classList.remove("bd-grab", "bd-grabbing");
        window.removeEventListener("keydown", this._onKeyDown, true);
        this.root?.remove();
        this.root = null;
        if (this.node?._minimaxEditor === this) this.node._minimaxEditor = null;
        if (this.domWidget?._minimaxEditor === this) this.domWidget._minimaxEditor = null;
    }

    widget(name) { return this.node.widgets?.find((w) => w.name === name); }

    _videoIdentityFromParts(video, clips) {
        const list = Array.isArray(clips) && clips.length ? clips : [];
        if (list.length) {
            return list
                .map((c) => `${c?.type || "input"}:${c?.videoFile || c?.fileName || ""}`)
                .filter((id) => id && id !== "input:");
        }
        const v = video || {};
        const id = `${v.type || "input"}:${v.videoFile || v.fileName || ""}`;
        return id && id !== "input:" ? [id] : [];
    }

    _clipThumbIdentity(clipIndex = 0) {
        const clips = this.getVideoClips();
        const c = clips[clipIndex] || clips[0] || this.timeline?.video || {};
        const id = `${c.type || "input"}:${c.videoFile || c.fileName || ""}`;
        return id === "input:" ? "" : id;
    }

    _videoThumbIdentity() {
        return this._videoIdentityFromParts(this.timeline?.video, this.timeline?.videoClips).join("|");
    }

    _liveVideoFileIdentities() {
        return this._videoIdentityFromParts(this.timeline?.video, this.timeline?.videoClips);
    }

    _knownVideoFileIdentities() {
        const ids = new Set(this._liveVideoFileIdentities());
        for (const ws of Object.values(this._videoWsMem || {})) {
            for (const id of this._videoIdentityFromParts(ws?.video, ws?.videoClips)) ids.add(id);
        }
        for (const ws of Object.values(this.timeline?.videoWorkspaces || {})) {
            for (const id of this._videoIdentityFromParts(ws?.video, ws?.videoClips)) ids.add(id);
        }
        return ids;
    }

    _frameThumbKey(logicalFrame) {
        const entry = this.getFrameMapEntry(logicalFrame);
        const id = this._clipThumbIdentity(entry.clip) || this._videoThumbIdentity() || "none";
        if (this._legacyFrames.length) return `${id}#legacy:${logicalFrame}`;
        return `${id}#${entry.clip}:${entry.frame}`;
    }

    _dropThumbsForIdentity(identity) {
        if (!identity) return;
        const prefix = `${identity}#`;
        for (const key of [...this._thumbCache.keys()]) {
            if (key === identity || String(key).startsWith(prefix)) this._thumbCache.delete(key);
        }
        for (const key of [...this._thumbPending]) {
            if (key === identity || String(key).startsWith(prefix)) this._thumbPending.delete(key);
        }
    }

    _dropThumbsIfUnused(identities) {
        const list = Array.isArray(identities) ? identities : [identities];
        const used = this._knownVideoFileIdentities();
        for (const id of list) {
            if (!id || used.has(id)) continue;
            this._dropThumbsForIdentity(id);
        }
    }

    _flushPendingThumbDrops() {
        this._dropThumbsIfUnused(this._thumbIdsPendingDrop);
        this._thumbIdsPendingDrop = [];
    }

    _invalidateVideoThumbs() {
        this._thumbCache.clear();
        this._thumbPending.clear();
    }

    _usesSourceVideoThumbs() {
        return this.getDirectorMode() === "video";
    }

    hasVideo() {
        const v = this.timeline?.video || {};
        return !!(this.getVideoClips().length || v.videoFile || this._legacyFrames.length || v.frames?.length);
    }

    /** v2v / rv2v empty canvas: placeholder says click to upload. */
    needsSourceVideoUpload() {
        return this.getDirectorMode() === "video" && !this.hasVideo();
    }

    getVideoClips() {
        if (this.timeline.videoClips?.length) return this.timeline.videoClips;
        const v = this.timeline?.video || {};
        if (v.videoFile || v.fileName) {
            return [{
                id: v.id || "c0",
                fileName: v.fileName || "",
                videoFile: v.videoFile || v.fileName || "",
                subfolder: v.subfolder || "",
                type: v.type || "input",
                width: v.width || 0,
                height: v.height || 0,
                duration: v.duration || 0,
                nativeFps: v.nativeFps || v.native_fps || 0,
                nativeFrameCount: v.nativeFrameCount || v.native_frame_count || 0,
                sourceFrameCount: v.sourceFrameCount || this.getFrameMap().length,
                storageWidth: v.storageWidth,
                storageHeight: v.storageHeight,
            }];
        }
        return [];
    }

    _ensureVideoClipsArray() {
        if (!this.timeline.videoClips?.length) {
            const v = this.timeline?.video || {};
            if (v.videoFile || v.fileName) {
                this.timeline.videoClips = [{
                    id: v.id || uid(),
                    fileName: v.fileName || "",
                    videoFile: v.videoFile || v.fileName || "",
                    subfolder: v.subfolder || "",
                    type: v.type || "input",
                    width: v.width || 0,
                    height: v.height || 0,
                    duration: v.duration || 0,
                    nativeFps: v.nativeFps || v.native_fps || 0,
                    nativeFrameCount: v.nativeFrameCount || v.native_frame_count || 0,
                    sourceFrameCount: v.sourceFrameCount || this.getFrameMap().length,
                    storageWidth: v.storageWidth,
                    storageHeight: v.storageHeight,
                }];
            } else {
                this.timeline.videoClips = [];
            }
        }
    }

    getClipViewUrl(clipIndex) {
        const clip = this.getVideoClips()[clipIndex];
        if (!clip?.videoFile) return "";
        return inputViewUrl(clip.videoFile, clip.type || "input");
    }

    getRefVideoTarget() {
        if (this.isGlobalMode()) {
            this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
            if (!this.timeline.global.referenceVideo) this.timeline.global.referenceVideo = {};
            return this.timeline.global;
        }
        const seg = this.timeline.segments[this.selectedIndex];
        if (seg) {
            if (!seg.referenceVideo) seg.referenceVideo = {};
            return seg;
        }
        this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
        return this.timeline.global;
    }

    getReferenceVideoViewUrl(ref) {
        const block = ref || {};
        const file = block.videoFile || block.fileName;
        if (!file) return "";
        return inputViewUrl(file, block.type || "input");
    }

    _stopRefVideoPreviews(onlyEls = null) {
        const targets = onlyEls || [this.globalRefVideo, this.segRefVideo];
        for (const el of targets) {
            const v = el?.querySelector("video");
            if (v) {
                v.pause();
                v.removeAttribute("src");
                v.load();
            }
        }
    }

    getTaskKey() {
        return resolveTaskKey(
            this.globalTask?.value
            || this.timeline.global?.taskType
            || this.taskTypeWidget?.value,
        );
    }

    getRunnableSegmentCount() {
        if (this.isFl2vMode()) return fl2vStartIndices(this).length;
        return this.timeline.segments?.length || 0;
    }

    supportsRunSelect() {
        const n = this.getRunnableSegmentCount();
        if (n < 2) return false;
        const mode = this.getDirectorMode();
        if (mode === "video") return true;
        if (mode === "fl2v") return true;
        if (this.isImageBatch()) return isPromptBatchTask(this.getTaskKey());
        return false;
    }

    getRunProgressSegmentTotal() {
        const n = this.getRunnableSegmentCount();
        if (!this.isRunSelectEnabled() || n < 2) return Math.max(n, 1);
        const count = (this.timeline.runSelection || []).length;
        return count > 0 ? count : Math.max(n, 1);
    }

    isRunSelectEnabled() {
        return !!this.timeline.runSelectEnabled;
    }

    normalizeRunSelection() {
        if (!this.isRunSelectEnabled()) return;
        if (this.isFl2vMode()) {
            const valid = new Set(fl2vStartIndices(this));
            this.timeline.runSelection = [...new Set(
                (this.timeline.runSelection || []).filter((i) => valid.has(i)),
            )].sort((a, b) => a - b);
            return;
        }
        const n = this.getRunnableSegmentCount();
        if (n < 1) return;
        this.timeline.runSelection = [...new Set(
            (this.timeline.runSelection || []).filter((i) => i >= 0 && i < n),
        )].sort((a, b) => a - b);
    }

    isSegmentRunEnabled(index) {
        if (!this.isRunSelectEnabled()) return true;
        return (this.timeline.runSelection || []).includes(index);
    }

    toggleSegmentRun(index) {
        if (!this.isRunSelectEnabled()) return;
        if (this.isFl2vMode()) {
            if (!this.timeline.segments?.[index]?.isStartFrame) return;
        } else {
            const n = this.getRunnableSegmentCount();
            if (index < 0 || index >= n) return;
        }
        const sel = new Set(this.timeline.runSelection || []);
        if (sel.has(index)) sel.delete(index);
        else sel.add(index);
        this.timeline.runSelection = [...sel].sort((a, b) => a - b);
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    toggleRunSelectMode() {
        if (!this.supportsRunSelect()) return;
        this.timeline.runSelectEnabled = !this.timeline.runSelectEnabled;
        if (this.timeline.runSelectEnabled) {
            if (!(this.timeline.runSelection || []).length) {
                if (this.isFl2vMode()) {
                    this.timeline.runSelection = fl2vStartIndices(this);
                } else {
                    const n = this.getRunnableSegmentCount();
                    this.timeline.runSelection = Array.from({ length: n }, (_, i) => i);
                }
            } else {
                this.normalizeRunSelection();
            }
        }
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    setRunSelectionAll(on) {
        if (!this.isRunSelectEnabled()) return;
        if (this.isFl2vMode()) {
            this.timeline.runSelection = on ? fl2vStartIndices(this) : [];
            this.updateRunSelectUI();
            this.commit(false, { syncTimeline: true });
            this.scheduleRender();
            return;
        }
        const n = this.getRunnableSegmentCount();
        this.timeline.runSelection = on ? Array.from({ length: n }, (_, i) => i) : [];
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    updateRunSelectUI() {
        const n = this.getRunnableSegmentCount();
        const canRunSelect = this.supportsRunSelect();
        const enabled = this.isRunSelectEnabled() && canRunSelect;
        // r2v uses timeline checkboxes (fl2v-style); other batch tasks use the card bar.
        const useBatchBar = this.isImageBatch() && canRunSelect && !this.isR2vBatch();
        this.btnRunSelectToggle?.classList.toggle("active", enabled);
        this.btnRunSelectToggle?.classList.toggle("bd-btn-run-select", true);
        this.btnRunSelectToggle?.classList.toggle("hidden", !canRunSelect || useBatchBar);
        this.batchRunSelectBtn?.classList.toggle("active", enabled);
        this.batchRunSelectBtn?.classList.toggle("hidden", !useBatchBar);
        this.runSelectAllWrap?.classList.toggle("hidden", !enabled || useBatchBar);
        this.batchRunSelectAllWrap?.classList.toggle("hidden", !enabled || !useBatchBar);
        // Keep the chip hidden while a run is active — otherwise commit/sync
        // re-shows it on top of the green progress title.
        const running = !!this.runStatusEl?.classList.contains("active");
        this.runSelectBar?.classList.toggle("hidden", !enabled || running);
        if (!canRunSelect) return;
        this.normalizeRunSelection();
        const count = (this.timeline.runSelection || []).length;
        const syncAllCb = (cb) => {
            if (!cb) return;
            cb.checked = count >= n && n > 0;
            cb.indeterminate = count > 0 && count < n;
        };
        syncAllCb(this.runSelectAllCb);
        syncAllCb(this.batchRunSelectAllCb);
        const label = t(this.isImageBatch() ? "unit.group" : "unit.segment");
        if (!this.runSelectSummary) return;
        if (!count) {
            this.runSelectSummary.textContent = t("runSelect.noneChecked", { unit: label });
            this.runSelectSummary.style.color = "#f88";
        } else if (count >= n) {
            this.runSelectSummary.textContent = t("runSelect.all", { n, unit: label });
            this.runSelectSummary.style.color = "#aaa";
        } else {
            const nums = (this.timeline.runSelection || []).map((i) => i + 1).join(", ");
            const exportHint = this.timeline.output?.exportMode === "segments"
                ? t("runSelect.exportOnlyChecked")
                : t("runSelect.fillUnchecked");
            this.runSelectSummary.textContent = count === 1
                ? t("runSelect.sampleOne", { unit: label, nums, hint: exportHint })
                : t("runSelect.sampleMany", { count, unit: label, nums, hint: exportHint });
            this.runSelectSummary.style.color = "#4fff8f";
        }
    }

    /** Drop live run-select flags (mode switch). Stashed workspaces keep their own copy. */
    _clearLiveRunSelection() {
        this.timeline.runSelectEnabled = false;
        this.timeline.runSelection = [];
    }

    _runSelectionPayload() {
        // Never leak video-mode「选择运行」into i2v/batch (or vice versa).
        if (!this.supportsRunSelect() || !this.timeline.runSelectEnabled) {
            return { runSelectEnabled: false, runSelection: [] };
        }
        this.normalizeRunSelection();
        return {
            runSelectEnabled: true,
            runSelection: [...(this.timeline.runSelection || [])],
        };
    }

    getDirectorMode() {
        return getDirectorMode(this.globalTask?.value || this.taskTypeWidget?.value);
    }

    isGenMode() {
        const mode = this.getDirectorMode();
        return mode !== "video" && mode !== "prompt_batch" && mode !== "fl2v";
    }

    isImageBatch() {
        const mode = this.getDirectorMode();
        return mode === "prompt_batch" || mode === "image_batch";
    }

    isGenBlank() {
        return this.getDirectorMode() === "gen_blank";
    }

    isGenImage() {
        return this.getDirectorMode() === "gen_image";
    }

    isFl2vMode() {
        return this.getDirectorMode() === "fl2v";
    }

    isR2vBatch() {
        return this.isImageBatch() && this.getTaskKey() === "r2v";
    }

    /** t2v / i2v / r2v: duration groups on the main timeline track. */
    usesBatchTimeline() {
        return this.isImageBatch() && isVideoBatchTask(this.getTaskKey());
    }

    _syncR2vCardSelection() {
        if (!this.isImageBatch() || !this.batchList) return;
        const runSelectOn = this.isRunSelectEnabled() && this.supportsRunSelect();
        const focusSel = this.isR2vBatch();
        const cards = this.batchList.querySelectorAll(".bd-batch-card");
        cards.forEach((el) => {
            const i = parseInt(el.dataset.batchIndex, 10);
            if (!Number.isFinite(i)) return;
            const runOn = !runSelectOn || this.isSegmentRunEnabled(i);
            el.classList.toggle("selected", focusSel && i === this.selectedIndex);
            el.classList.toggle("run-on", runSelectOn && runOn);
            el.classList.toggle("run-skipped", runSelectOn && !runOn);
            const cb = el.querySelector(".bd-batch-run-check");
            if (cb) cb.checked = runOn;
        });
        this.batchPicker?.querySelectorAll?.(".bd-batch-pick").forEach((el) => {
            const i = parseInt(el.dataset.batchIndex, 10);
            if (!Number.isFinite(i)) return;
            const runOn = !runSelectOn || this.isSegmentRunEnabled(i);
            el.classList.toggle("selected", i === this.selectedIndex);
            el.classList.toggle("run-on", runSelectOn && runOn);
            el.classList.toggle("run-skipped", runSelectOn && !runOn);
            const cb = el.querySelector(".bd-batch-run-check");
            if (cb) cb.checked = runOn;
        });
    }

    onTaskTypeChanged(value) {
        this.onGlobalField("taskType", value);
    }

    /** Snapshot v2v/rv2v workspace for a specific task key (session + persist). */
    _captureVideoWorkspace() {
        const video = this.timeline.video || {};
        const clips = this.timeline.videoClips || [];
        const g = this.timeline.global || {};
        return {
            segments: cloneJson(this.timeline.segments || [], []),
            selectedIndex: this.selectedIndex,
            currentFrame: this.currentFrame,
            editMode: this.timeline.editMode || "global",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: Array.isArray(this.timeline.runSelection)
                ? [...this.timeline.runSelection]
                : [],
            video: cloneJson(video, {}),
            videoClips: cloneJson(clips, []),
            totalFrames: this.timeline.totalFrames ?? this.getTotalFrames(),
            frameRate: this.timeline.frameRate ?? this.getFrameRate(),
            legacyFrames: this._legacyFrames?.length ? [...this._legacyFrames] : [],
            storageWidth: this._storageWidth || 0,
            storageHeight: this._storageHeight || 0,
            globalCommon: {
                commonEnabled: !!g.commonEnabled,
                commonCollapsed: !!g.commonCollapsed,
                prompt: g.prompt || "",
                refs: cloneJson(g.refs, []),
                refAudios: cloneJson(g.refAudios || g.ref_audios, []),
                refVideos: cloneJson(g.refVideos || g.ref_videos, []),
            },
        };
    }

    _applyVideoWorkspace(ws) {
        if (!ws || typeof ws !== "object") return false;
        if (ws.video && typeof ws.video === "object") {
            this.timeline.video = cloneJson(ws.video, {});
        } else {
            this.timeline.video = {
                fileName: "", videoFile: "", subfolder: "", type: "input", frames: [], frameMap: [],
            };
        }
        this.timeline.videoClips = Array.isArray(ws.videoClips) ? cloneJson(ws.videoClips, []) : [];
        this.timeline.segments = Array.isArray(ws.segments) ? cloneJson(ws.segments, []) : [];
        if (ws.totalFrames != null) this.timeline.totalFrames = ws.totalFrames;
        if (ws.frameRate != null) this.timeline.frameRate = ws.frameRate;
        this.timeline.editMode = ws.editMode || "global";
        this.timeline.runSelectEnabled = !!ws.runSelectEnabled;
        this.timeline.runSelection = Array.isArray(ws.runSelection) ? [...ws.runSelection] : [];
        const gc = ws.globalCommon || {};
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.commonEnabled = !!gc.commonEnabled;
        this.timeline.global.commonCollapsed = !!gc.commonCollapsed;
        this.timeline.global.prompt = gc.prompt || "";
        this.timeline.global.refs = cloneJson(gc.refs, []);
        this.timeline.global.refAudios = cloneJson(gc.refAudios, []);
        this.timeline.global.refVideos = cloneJson(gc.refVideos, []);
        if (this.globalPrompt) this.globalPrompt.value = this.timeline.global.prompt || "";
        if (this.globalPromptWidget) this.globalPromptWidget.value = this.timeline.global.prompt || "";
        this.selectedIndex = clamp(
            ws.selectedIndex ?? 0,
            0,
            Math.max(0, (this.timeline.segments?.length || 1) - 1),
        );
        this.currentFrame = Math.max(0, ws.currentFrame ?? 0);
        if (Array.isArray(ws.legacyFrames) && ws.legacyFrames.length) {
            this._legacyFrames = [...ws.legacyFrames];
        } else {
            this._legacyFrames = [];
        }
        if (ws.storageWidth) this._storageWidth = ws.storageWidth;
        if (ws.storageHeight) this._storageHeight = ws.storageHeight;

        this.normalizeSegments();
        this.restoreVideoFromTimeline();
        const total = this.getTotalFrames();
        this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, total - 1));
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = this.currentFrame;
        }
        if (this.totalFramesWidget) this.totalFramesWidget.value = total;
        this.updateVideoNameLabel();
        this.updateStageVisibility();
        return true;
    }

    _resetVideoWorkspaceLive() {
        this._clearVideoState();
        this.timeline.segments = [];
        this.timeline.editMode = "global";
        this.selectedIndex = 0;
        this.currentFrame = 0;
        this._clearLiveRunSelection();
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.prompt = "";
        this.timeline.global.refs = [];
        this.timeline.global.refAudios = [];
        this.timeline.global.refVideos = [];
        this.timeline.global.referenceVideo = {};
        this.timeline.global.continuousReference = false;
        this.timeline.global.commonEnabled = false;
        this.timeline.global.commonCollapsed = false;
        if (this.globalPrompt) this.globalPrompt.value = "";
        if (this.globalPromptWidget) this.globalPromptWidget.value = "";
        this.updateVideoNameLabel();
    }

    _stashVideoWorkspace(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        if (getDirectorMode(key) !== "video") return;
        this._videoWsMem = this._videoWsMem || {};
        const full = this._captureVideoWorkspace();
        this._videoWsMem[key] = full;
        this.timeline.videoWorkspaces = this.timeline.videoWorkspaces || {};
        const safe = sanitizeVideoWorkspace(full);
        if (safe) this.timeline.videoWorkspaces[key] = safe;
    }

    _persistCurrentVideoWorkspace() {
        if (this.getDirectorMode() !== "video") return;
        const key = this.getTaskKey();
        if (getDirectorMode(key) !== "video") return;
        const g = this.timeline.global || {};
        const safe = sanitizeVideoWorkspace({
            segments: this.timeline.segments || [],
            selectedIndex: this.selectedIndex,
            currentFrame: this.currentFrame,
            editMode: this.timeline.editMode || "global",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: this.timeline.runSelection,
            video: this.timeline.video || {},
            videoClips: this.timeline.videoClips || [],
            totalFrames: this.timeline.totalFrames ?? this.getTotalFrames(),
            frameRate: this.timeline.frameRate ?? this.getFrameRate(),
            storageWidth: this._storageWidth || 0,
            storageHeight: this._storageHeight || 0,
            globalCommon: {
                commonEnabled: !!g.commonEnabled,
                commonCollapsed: !!g.commonCollapsed,
                prompt: g.prompt || "",
                refs: g.refs,
                refAudios: g.refAudios || g.ref_audios,
                refVideos: g.refVideos || g.ref_videos,
            },
        });
        if (!safe) return;
        this.timeline.videoWorkspaces = this.timeline.videoWorkspaces || {};
        this.timeline.videoWorkspaces[key] = safe;
    }

    _restoreVideoWorkspace(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        const mem = this._videoWsMem?.[key];
        const persisted = this.timeline.videoWorkspaces?.[key];
        const ws = mem || persisted;
        return this._applyVideoWorkspace(ws);
    }

    _switchToVideoTaskWorkspace(prevTaskKey, currentKey) {
        const prevIds = new Set(this._liveVideoFileIdentities());
        if (prevTaskKey && getDirectorMode(prevTaskKey) === "video" && prevTaskKey !== currentKey) {
            this._stashVideoWorkspace(prevTaskKey);
            this._clearLiveRunSelection();
        }
        const nextWs = this._videoWsMem?.[currentKey] || this.timeline.videoWorkspaces?.[currentKey];
        const nextIds = new Set(this._videoIdentityFromParts(nextWs?.video, nextWs?.videoClips));
        const sameFiles = prevIds.size === nextIds.size && [...prevIds].every((id) => nextIds.has(id));
        if (!sameFiles) this._clearPreviewVideos?.(true);
        if (this._restoreVideoWorkspace(currentKey)) return;
        this._resetVideoWorkspaceLive();
    }

    _captureBatchWorkspace() {
        const g = this.timeline.global || {};
        return {
            segments: cloneJson(this.timeline.segments || [], []),
            selectedIndex: this.selectedIndex,
            editMode: this.timeline.editMode || "segment",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: Array.isArray(this.timeline.runSelection)
                ? [...this.timeline.runSelection]
                : [],
            globalCommon: {
                commonEnabled: !!g.commonEnabled,
                commonCollapsed: !!g.commonCollapsed,
                prompt: g.prompt || "",
                refs: cloneJson(g.refs, []),
                refAudios: cloneJson(g.refAudios || g.ref_audios, []),
                refVideos: cloneJson(g.refVideos || g.ref_videos, []),
            },
        };
    }

    _applyBatchWorkspace(ws) {
        if (!ws || !Array.isArray(ws.segments) || !ws.segments.length) return false;
        this.timeline.segments = cloneJson(ws.segments, []);
        this.timeline.editMode = ws.editMode || "segment";
        this.timeline.runSelectEnabled = !!ws.runSelectEnabled;
        this.timeline.runSelection = Array.isArray(ws.runSelection) ? [...ws.runSelection] : [];
        const gc = ws.globalCommon || {};
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.commonEnabled = !!gc.commonEnabled;
        this.timeline.global.commonCollapsed = !!gc.commonCollapsed;
        this.timeline.global.prompt = gc.prompt || "";
        this.timeline.global.refs = cloneJson(gc.refs, []);
        this.timeline.global.refAudios = cloneJson(gc.refAudios, []);
        this.timeline.global.refVideos = cloneJson(gc.refVideos, []);
        this.selectedIndex = clamp(
            ws.selectedIndex ?? 0,
            0,
            Math.max(0, this.timeline.segments.length - 1),
        );
        if (this.globalPrompt) this.globalPrompt.value = this.timeline.global.prompt || "";
        if (this.globalPromptWidget) this.globalPromptWidget.value = this.timeline.global.prompt || "";
        return true;
    }

    _resetBatchWorkspaceLive(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        this.timeline.segments = [newBatchSegment({ durationSec: defaultDurationSec(key) })];
        this.timeline.editMode = "segment";
        this.selectedIndex = 0;
        this._clearLiveRunSelection();
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.commonEnabled = false;
        this.timeline.global.commonCollapsed = false;
        this.timeline.global.prompt = "";
        this.timeline.global.refs = [];
        this.timeline.global.refAudios = [];
        this.timeline.global.refVideos = [];
        if (this.globalPrompt) this.globalPrompt.value = "";
        if (this.globalPromptWidget) this.globalPromptWidget.value = "";
    }

    /** Snapshot t2v / i2v / r2v groups for a specific task key (session + persist). */
    _stashBatchWorkspace(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        if (!isVideoBatchTask(key)) return;
        if (this.isImageBatch?.()) flushBatchPromptInputs(this);
        const segs = this.timeline.segments || [];
        if (!segs.length) return;
        this._batchWsMem = this._batchWsMem || {};
        const full = this._captureBatchWorkspace();
        this._batchWsMem[key] = full;
        this.timeline.batchWorkspaces = this.timeline.batchWorkspaces || {};
        const safe = sanitizeBatchWorkspace(full);
        if (safe) this.timeline.batchWorkspaces[key] = safe;
    }

    _persistCurrentBatchWorkspace() {
        if (!this.isImageBatch?.()) return;
        const key = this.getTaskKey();
        if (!isVideoBatchTask(key)) return;
        const g = this.timeline.global || {};
        const safe = sanitizeBatchWorkspace({
            segments: this.timeline.segments || [],
            selectedIndex: this.selectedIndex,
            editMode: this.timeline.editMode || "segment",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: this.timeline.runSelection,
            globalCommon: {
                commonEnabled: !!g.commonEnabled,
                commonCollapsed: !!g.commonCollapsed,
                prompt: g.prompt || "",
                refs: g.refs,
                refAudios: g.refAudios || g.ref_audios,
                refVideos: g.refVideos || g.ref_videos,
            },
        });
        if (!safe) return;
        this.timeline.batchWorkspaces = this.timeline.batchWorkspaces || {};
        this.timeline.batchWorkspaces[key] = safe;
    }

    _restoreBatchWorkspace(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        const mem = this._batchWsMem?.[key];
        const persisted = this.timeline.batchWorkspaces?.[key];
        const ws = (mem?.segments?.length ? mem : null) || persisted;
        return this._applyBatchWorkspace(ws);
    }

    _switchToBatchTaskWorkspace(prevTaskKey, currentKey) {
        if (prevTaskKey && isVideoBatchTask(prevTaskKey) && prevTaskKey !== currentKey) {
            this._stashBatchWorkspace(prevTaskKey);
            this._clearLiveRunSelection();
        }
        if (this._restoreBatchWorkspace(currentKey)) return;
        const externalLocked = (currentKey === "i2v" && this.hasExternalI2vGroups?.())
            || (currentKey === "r2v" && this.hasExternalR2vGroups?.());
        if (!externalLocked) this._resetBatchWorkspaceLive(currentKey);
    }

    ensureGenTimeline() {
        const key = this.getTaskKey();
        this.timeline.gen = this.timeline.gen || {};
        const defFc = defaultFrameCount(key);
        if (!this.timeline.segments?.length || !sumFrameCounts(this.timeline.segments)) {
            this.timeline.segments = [{
                id: uid(), start: 0, length: defFc, frameCount: defFc,
                prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
            }];
        }
        for (const seg of this.timeline.segments) {
            if (seg.frameCount == null) seg.frameCount = seg.length ?? defFc;
            seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
        }
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.genImage = this.timeline.global.genImage || { imageFile: "" };
        if (this.isGenBlank()) {
            this.timeline.output = this.timeline.output || {};
            this.timeline.output.mode = "fixed";
        }
        this.normalizeGenSegments();
    }

    normalizeGenSegments() {
        const key = this.getTaskKey();
        const minFc = minFrameCount(key);
        let start = 0;
        const fixed = [];
        for (const seg of [...this.timeline.segments]) {
            let fc = clamp(parseInt(seg.frameCount ?? seg.length, 10) || defaultFrameCount(key), minFc, MAX_GEN_FRAMES);
            fixed.push({
                ...seg,
                start,
                length: fc,
                frameCount: fc,
                refs: seg.refs || [],
                genImage: seg.genImage || { imageFile: "" },
            });
            start += fc;
        }
        if (!fixed.length) {
            const fc = defaultFrameCount(key);
            fixed.push({
                id: uid(), start: 0, length: fc, frameCount: fc,
                prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
            });
        }
        this.timeline.segments = fixed;
        this.timeline.totalFrames = start || fixed[0].frameCount;
        this.selectedIndex = clamp(this.selectedIndex, 0, fixed.length - 1);
    }

    /** rv2v (and video-timeline tasks with refs) use the polished r2v-like asset stage. */
    usesRv2vRefStyle(taskKey = this.getTaskKey()) {
        const key = resolveTaskKey(taskKey);
        // r2v shared panel reuses the polished image/audio slot chrome.
        return key === "rv2v" || key === "vrc2v" || key === "vi2v" || key === "r2v";
    }

    /** v2v prompt-only video edit — full-width polished prompt stage. */
    usesV2vPromptStyle(taskKey = this.getTaskKey()) {
        const key = resolveTaskKey(taskKey);
        return key === "v2v" || key === "mv2v";
    }

    syncRv2vRefLayoutClasses({ hideTimeline = false, seg = null } = {}) {
        const globalKey = this.getTaskKey();
        const segKey = resolveTaskKey(
            seg?.taskType || this.timeline.global?.taskType || this.globalTask?.value || globalKey,
        );
        const globalRefStyle = !hideTimeline && this.usesRv2vRefStyle(globalKey);
        const segRefStyle = !hideTimeline && this.usesRv2vRefStyle(segKey);
        const globalV2vStyle = !hideTimeline && this.usesV2vPromptStyle(globalKey);
        const segV2vStyle = !hideTimeline && this.usesV2vPromptStyle(segKey);

        this.globalPanel?.classList.toggle("bd-rv2v-panel", globalRefStyle);
        this.segmentPanel?.classList.toggle("bd-rv2v-panel", segRefStyle);
        this.globalPanel?.classList.toggle("bd-v2v-panel", globalV2vStyle);
        this.segmentPanel?.classList.toggle("bd-v2v-panel", segV2vStyle);

        this.globalPromptLayout?.classList.toggle("bd-rv2v-layout", globalRefStyle);
        this.segPromptLayout?.classList.toggle("bd-rv2v-layout", segRefStyle);
        this.globalPromptLayout?.classList.toggle("bd-v2v-layout", globalV2vStyle);
        this.segPromptLayout?.classList.toggle("bd-v2v-layout", segV2vStyle);

        for (const wrap of [this.globalRefsImagesWrap, this.globalRefAudiosWrap, this.globalRefVideosWrap]) {
            wrap?.classList.toggle("bd-r2v-section", globalRefStyle);
        }
        for (const wrap of [this.segRefsImagesWrap, this.segRefAudiosWrap]) {
            wrap?.classList.toggle("bd-r2v-section", segRefStyle);
        }
        const gLabel = this.root.querySelector('[data-r="global-refs-label"]');
        if (gLabel) {
            const key = globalRefStyle ? "batch.r2v.sectionPictures" : "panel.refImages";
            gLabel.textContent = t(key);
            gLabel.setAttribute("data-i18n", key);
        }
        const sLabel = this.root.querySelector('[data-r="seg-refs-label"]');
        if (sLabel) {
            const key = segRefStyle ? "batch.r2v.sectionPictures" : "panel.segmentRefImages";
            sLabel.textContent = t(key);
            sLabel.setAttribute("data-i18n", key);
        }
    }

    updateReferenceImageVisibility({ hideTimeline = false, seg = null } = {}) {
        const globalKey = this.getTaskKey();
        const showGlobalRefs = !hideTimeline && taskUsesReferenceImages(globalKey);
        const showGlobalRefAudios = !hideTimeline && taskUsesReferenceAudios(globalKey);
        // r2v common panel: multi-slot ref videos (distinct from ads2v single referenceVideo).
        const showGlobalR2vVideos = !hideTimeline && this.usesR2vCommonPanel();
        const showGlobalRefVideo = !hideTimeline && taskUsesReferenceVideo(globalKey);

        this.globalRefsCol?.classList.toggle(
            "hidden",
            !showGlobalRefs && !showGlobalRefVideo && !showGlobalRefAudios && !showGlobalR2vVideos,
        );
        this.globalRefsImagesWrap?.classList.toggle("hidden", !showGlobalRefs);
        this.globalRefVideosWrap?.classList.toggle("hidden", !showGlobalR2vVideos);
        this.globalRefAudiosWrap?.classList.toggle("hidden", !showGlobalRefAudios);
        this.globalRefVideoCol?.classList.toggle("hidden", !showGlobalRefVideo);
        if (this.globalPanelTitle) {
            let titleKey = "panel.globalPromptOnly";
            if (this.usesR2vCommonPanel()) {
                titleKey = "panel.r2vCommonParams";
            } else if (showGlobalRefVideo) {
                titleKey = "panel.globalPromptAndRefVideo";
            } else if (showGlobalRefs || showGlobalRefAudios) {
                titleKey = showGlobalRefAudios
                    ? "panel.globalPromptAndRefsMedia"
                    : "panel.globalPromptAndRefs";
            }
            this.globalPanelTitle.textContent = t(titleKey);
            this.globalPanelTitle.setAttribute("data-i18n", titleKey);
        }
        this.syncR2vCommonCollapse();

        const segKey = resolveTaskKey(
            seg?.taskType || this.timeline.global?.taskType || this.globalTask?.value || globalKey,
        );
        const showSegRefs = !hideTimeline && taskUsesReferenceImages(segKey);
        const showSegRefAudios = !hideTimeline && taskUsesReferenceAudios(segKey);
        const showSegRefVideo = !hideTimeline && taskUsesReferenceVideo(segKey);
        this.segRefsCol?.classList.toggle(
            "hidden",
            !showSegRefs && !showSegRefVideo && !showSegRefAudios,
        );
        this.segRefsImagesWrap?.classList.toggle("hidden", !showSegRefs);
        this.segRefAudiosWrap?.classList.toggle("hidden", !showSegRefAudios);
        this.segRefVideoCol?.classList.toggle("hidden", !showSegRefVideo);
        const showContinuousRef = !hideTimeline
            && this.isGlobalMode()
            && showGlobalRefVideo
            && globalKey === "ads2v";
        this.continuousRefWrap?.classList.toggle("hidden", !showContinuousRef);
        if (this.continuousRefCb) {
            this.continuousRefCb.checked = !!this.timeline.global?.continuousReference;
        }
        this.syncRv2vRefLayoutClasses({ hideTimeline, seg });
        if (showGlobalRefVideo || showSegRefVideo) this.renderRefVideoSlot();
        if (showGlobalR2vVideos) this.renderR2vCommonVideoSlots();
        if (showGlobalRefAudios || showSegRefAudios) this.renderRefAudioSlots();
    }

    _stashFl2vWorkspace() {
        const shots = this.timeline.shots || [];
        const segs = this.timeline.segments || [];
        const keys = this.timeline.keyframes || [];
        if (!shots.length && !segs.length && !keys.length) return;
        this.timeline.fl2vWorkspace = {
            shots: JSON.parse(JSON.stringify(shots)),
            segments: JSON.parse(JSON.stringify(segs)),
            keyframes: JSON.parse(JSON.stringify(keys)),
            durationSec: this.timeline.durationSec,
            totalFrames: this.timeline.totalFrames,
            selectedIndex: this.selectedIndex,
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: Array.isArray(this.timeline.runSelection)
                ? [...this.timeline.runSelection]
                : [],
            output: this.timeline.output
                ? JSON.parse(JSON.stringify(this.timeline.output))
                : undefined,
        };
    }

    _restoreFl2vWorkspace() {
        const ws = this.timeline.fl2vWorkspace;
        if (!ws) return false;
        const hasShots = Array.isArray(ws.shots) && ws.shots.length;
        const hasSegs = Array.isArray(ws.segments) && ws.segments.length;
        const hasKeys = Array.isArray(ws.keyframes) && ws.keyframes.length;
        if (!hasShots && !hasSegs && !hasKeys) return false;
        if (hasShots) this.timeline.shots = JSON.parse(JSON.stringify(ws.shots));
        if (hasSegs) this.timeline.segments = JSON.parse(JSON.stringify(ws.segments));
        if (hasKeys) this.timeline.keyframes = JSON.parse(JSON.stringify(ws.keyframes));
        if (ws.durationSec != null) this.timeline.durationSec = ws.durationSec;
        if (ws.totalFrames != null) this.timeline.totalFrames = ws.totalFrames;
        if (ws.selectedIndex != null) this.selectedIndex = ws.selectedIndex;
        if (ws.runSelectEnabled != null) this.timeline.runSelectEnabled = !!ws.runSelectEnabled;
        if (Array.isArray(ws.runSelection)) this.timeline.runSelection = [...ws.runSelection];
        if (ws.output && typeof ws.output === "object") {
            this.timeline.output = { ...(this.timeline.output || {}), ...JSON.parse(JSON.stringify(ws.output)) };
        }
        this.timeline.fl2vWorkspace = null;
        return true;
    }

    applyTaskLayout(prevMode, prevTaskKey) {
        const mode = this.getDirectorMode();
        const prev = prevMode || "video";
        const wasBatch = prev === "prompt_batch" || prev === "image_batch";
        const isBatch = mode === "prompt_batch";
        const wasFl2v = prev === "fl2v";
        const isFl2v = mode === "fl2v";
        const wasGen = prev !== "video" && prev !== "prompt_batch" && prev !== "image_batch" && prev !== "fl2v";
        const isGen = mode !== "video" && mode !== "prompt_batch" && mode !== "fl2v";
        const currentKey = this.getTaskKey();
        const stashBatchKey = prevTaskKey && isVideoBatchTask(prevTaskKey) ? prevTaskKey : null;
        const stashVideoKey = prevTaskKey && getDirectorMode(prevTaskKey) === "video" ? prevTaskKey : null;

        if (this.isPlaying) this._stopPlay();

        if (isFl2v) {
            if (prev === "video") {
                this._stashVideoWorkspace(stashVideoKey);
                this._clearLiveRunSelection();
            } else if (wasBatch) {
                this._stashBatchWorkspace(stashBatchKey);
                this._clearLiveRunSelection();
            }
            if (!this._restoreFl2vWorkspace()) {
                ensureFl2vTimeline(this);
                this._clearLiveRunSelection();
            } else {
                ensureFl2vTimeline(this);
            }
        } else if (isBatch) {
            if (!wasBatch) {
                if (wasFl2v) {
                    this._stashFl2vWorkspace();
                    this._clearLiveRunSelection();
                }
                // Keep v2v/rv2v video + segments so switching back can restore them.
                // Run-select is per workspace: stash video's, then clear live so i2v/batch
                // does not inherit「选择运行」from rv2v.
                if (prev === "video") {
                    this._stashVideoWorkspace(stashVideoKey);
                    this._clearLiveRunSelection();
                    this._clearVideoState();
                }
                this._switchToBatchTaskWorkspace(null, currentKey);
            } else if (prevTaskKey && prevTaskKey !== currentKey) {
                // t2v / i2v / r2v used to share one segment list — isolate per task.
                this._switchToBatchTaskWorkspace(prevTaskKey, currentKey);
            }
            ensureImageBatchTimeline(this);
        } else if (isGen) {
            if (wasBatch) {
                this._stashBatchWorkspace(stashBatchKey);
                this._clearLiveRunSelection();
            }
            if (wasFl2v) {
                this._stashFl2vWorkspace();
                this._clearLiveRunSelection();
            }
            if (!wasGen && !wasBatch && !wasFl2v) {
                if (prev === "video") {
                    this._stashVideoWorkspace(stashVideoKey);
                    this._clearLiveRunSelection();
                }
                const key = this.getTaskKey();
                const defFc = defaultFrameCount(key);
                const keepPrompt = this.timeline.global?.prompt || "";
                this.timeline.segments = [{
                    id: uid(),
                    start: 0,
                    length: defFc,
                    frameCount: defFc,
                    prompt: keepPrompt,
                    taskType: "",
                    refs: [],
                    genImage: { imageFile: "" },
                }];
            }
            this.ensureGenTimeline();
        } else if (prev !== "video") {
            // Leaving batch/gen/fl2v for video — stash before video restore.
            if (wasBatch) {
                this._stashBatchWorkspace(stashBatchKey);
                this._clearLiveRunSelection();
            }
            if (wasFl2v) {
                this._stashFl2vWorkspace();
                this._clearLiveRunSelection();
            }
            this.timeline.timelineMode = "video";
            this._switchToVideoTaskWorkspace(stashVideoKey, currentKey);
        } else if (prevTaskKey && prevTaskKey !== currentKey) {
            // v2v ↔ rv2v share director mode "video" but keep separate workspaces.
            this._switchToVideoTaskWorkspace(prevTaskKey, currentKey);
        }
        this.timeline.timelineMode = mode;
        this._directorMode = mode;
        const taskKey = currentKey;
        this._taskKey = taskKey;

        const isR2v = isBatch && taskKey === "r2v";
        const showBatchTrack = isBatch && isVideoBatchTask(taskKey);
        // fl2v / t2v / i2v / r2v use the main timeline track; image batch + gen hide it.
        const hideTimeline = (isBatch && !showBatchTrack) || isGen;
        const hideVideoUpload = hideTimeline || NO_VIDEO_UPLOAD_TASKS.has(taskKey) || isR2v;
        const showBatchExport = (isBatch && isVideoBatchTask(taskKey)) || isFl2v;
        // t2v / i2v / r2v: never show source-video upload (fl2v keeps "上传图片").
        this.btnVideo?.classList.toggle("hidden", (hideVideoUpload && !isFl2v) || isR2v);
        this.btnVideoExisting?.classList.toggle("hidden", hideVideoUpload || isFl2v || isR2v);
        this.btnVideoAppend?.classList.toggle("hidden", hideVideoUpload || isFl2v || isR2v);
        // Playback / seek / zoom are for source-video (v2v). fl2v / t2v / batch have no source clip.
        this.controlsBar?.classList.toggle("hidden", hideTimeline || isBatch || isFl2v);
        this.boundsEl?.classList.toggle("hidden", hideTimeline || isBatch || isFl2v);
        this.timecodeEl?.classList.toggle("hidden", hideTimeline || isBatch || isFl2v);
        this.viewport?.classList.toggle("hidden", isBatch && !showBatchTrack);
        this.tlZoomWrap?.classList.toggle("hidden", isBatch && !showBatchTrack);
        this.updateStageVisibility();
        this.updateLiveSamplePanel();
        this.syncExternalGroupsTimeline();
        // r2v keeps bd-split visible so the shared「公共参数」panel can sit above batch cards.
        this.root.querySelector(".bd-split")?.classList.toggle("hidden", (isBatch && !isR2v) || isFl2v);
        this.batchPanel?.classList.toggle("hidden", !isBatch);
        this.root?.classList.toggle("bd-batch-fill", !!isBatch);
        this.fl2vUi?.root?.classList.toggle("hidden", !isFl2v);
        this.fl2vTotalWrap?.classList.toggle("hidden", !isFl2v);
        if (isFl2v) {
            setR2vToolbar(this, false);
            setFl2vToolbar(this, true);
            setToolbarDisabledForBatch(this, false);
            // Re-apply fl2v-specific disables after clearing batch disables.
            setFl2vToolbar(this, true);
        } else if (isR2v) {
            setFl2vToolbar(this, false);
            setToolbarDisabledForBatch(this, false);
            setR2vToolbar(this, true);
            if (this.btnVideo) {
                this.btnVideo.textContent = t("toolbar.uploadVideo");
                this.btnVideo.setAttribute("data-i18n", "toolbar.uploadVideo");
            }
            updateFl2vToolbarBtns(this);
        } else {
            setFl2vToolbar(this, false);
            setR2vToolbar(this, false);
            setToolbarDisabledForBatch(this, isBatch);
            if (this.btnVideo) {
                this.btnVideo.textContent = t("toolbar.uploadVideo");
                this.btnVideo.setAttribute("data-i18n", "toolbar.uploadVideo");
            }
            const del = this.root?.querySelector('[data-a="del"]');
            if (del) {
                if (showBatchTrack) {
                    const externalLocked = !!(this.hasExternalI2vGroups?.() || this.hasExternalR2vGroups?.());
                    if (externalLocked) {
                        del.classList.add("hidden");
                        del.disabled = true;
                    } else {
                        del.disabled = false;
                        del.classList.remove("bd-disabled", "hidden");
                        del.textContent = t("toolbar.deleteSelectedGroup");
                        del.setAttribute("data-i18n", "toolbar.deleteSelectedGroup");
                        del.setAttribute("data-i18n-title", "tooltip.deleteSelectedPromptGroup");
                        del.title = t("tooltip.deleteSelectedPromptGroup");
                    }
                } else {
                    del.textContent = t("toolbar.deleteSegment");
                    del.setAttribute("data-i18n", "toolbar.deleteSegment");
                    del.setAttribute("data-i18n-title", "tooltip.deleteSegment");
                }
            }
            updateFl2vToolbarBtns(this);
            updateR2vToolbarBtns(this);
        }

        // Side ref panels stay hidden for most batch modes (refs live in cards).
        // r2v shows a collapsible「公共参数」bar; refs only when enabled/expanded.
        this.syncR2vCommonCollapse();
        this.updateReferenceImageVisibility({
            hideTimeline: (isBatch && !this.isR2vCommonEnabled()) || isGen,
        });

        const showGenImg = mode === "gen_image";
        this.genGlobalImg?.classList.toggle("hidden", !showGenImg || !this.isGlobalMode());
        this.genSegImg?.classList.toggle("hidden", !showGenImg || this.isGlobalMode());
        this.genGlobalFcRow?.classList.toggle("hidden", !isGen || !this.isGlobalMode());
        this.genSegFcRow?.classList.toggle("hidden", !isGen || this.isGlobalMode());

        if (isBatch || isGen || isFl2v || NO_VIDEO_UPLOAD_TASKS.has(taskKey)) {
            this.timeline.output = this.timeline.output || {};
            this.timeline.output.mode = "fixed";
            if (isBatch && !isVideoBatchTask(taskKey)) this.timeline.output.exportMode = "all";
            if (!this.timeline.output.aspectRatio) this.timeline.output.aspectRatio = DEFAULT_ASPECT_RATIO;
            else this.timeline.output.aspectRatio = normalizeAspectRatioLabel(this.timeline.output.aspectRatio);
            if (this.timeline.output.megapixels == null) this.timeline.output.megapixels = DEFAULT_MEGAPIXELS;
            if (this.timeline.output.multiple == null) this.timeline.output.multiple = MINIMAX_CANVAS_MULTIPLE;
            if (isCustomAspectRatio(this.timeline.output.aspectRatio)) {
                this.applyCustomResolution();
            } else {
                this.applyResolutionSelector();
            }
            this.updateOutputModeUI();
        } else if (this.outMode) {
            this.outMode.disabled = false;
            // Video edit (v2v/rv2v): prefer long-edge so ultrawide sources are not
            // center-cropped into a leftover 16:9 fixed canvas from batch modes.
            this.timeline.output = this.timeline.output || {};
            if (!this.timeline.output.mode || this.timeline.output.mode === "fixed") {
                const fromBatchFixed = this._lastOutputWasBatchFixed;
                if (fromBatchFixed || !this.timeline.output.mode) {
                    this.timeline.output.mode = "long_edge";
                    if (!this.timeline.output.longEdge) this.timeline.output.longEdge = 848;
                }
            }
            this._lastOutputWasBatchFixed = false;
            this.updateOutputModeUI();
        }
        if (isBatch || isGen || isFl2v) this._lastOutputWasBatchFixed = true;

        if (this.outHint) {
            const isVideoEdit = taskKey === "v2v" || taskKey === "rv2v";
            const showHint = isGen || isBatch || isFl2v || isVideoEdit;
            this.outHint.classList.toggle("hidden", !showHint);
            this.outHint.textContent = showHint ? genLayoutHint(this.getTaskKey()) : "";
        }
        const isVideoEditTask = isVideoEditTaskKey(taskKey);
        this.outAudioWrap?.classList.toggle("hidden", !isVideoEditTask);
        this.syncExportSourceImagesUI();
        if (this.outExportMode) {
            this.outExportMode.disabled = (isBatch || isFl2v) && !showBatchExport;
            this.outExportMode.classList.toggle("hidden", (isBatch || isFl2v) && !showBatchExport);
            this.outExportMode.previousElementSibling?.classList.toggle("hidden", (isBatch || isFl2v) && !showBatchExport);
        }
        if (this.outMaxFrames) {
            this.outMaxFrames.disabled = (isBatch || isFl2v) && !showBatchExport;
            this.outMaxFrames.classList.toggle("hidden", (isBatch || isFl2v) && !showBatchExport);
            this.outMaxFrames.previousElementSibling?.classList.toggle("hidden", (isBatch || isFl2v) && !showBatchExport);
        }

        if ((isGen || isBatch || isFl2v) && prev === "video") {
            this.currentFrame = 0;
        }
        this.updateVideoNameLabel();
        if (isFl2v) {
            this.timeline.editMode = "segment";
            ensureFl2vTimeline(this);
            this.updateSelectionUI();
            updateFl2vDetailUI(this);
            this.updateVideoNameLabel();
        } else if (isBatch) {
            this.timeline.editMode = "segment";
            this.renderImageBatchGroups();
            // Must refresh globalPanel display — r2v common params stay display:none
            // if we only ran updateModeUI in the non-batch branch (segment → r2v).
            this.updateModeUI();
            if (showBatchTrack) {
                this.updateSelectionUI();
                this._syncR2vCardSelection();
            }
        } else {
            this.updateModeUI();
            this.updateSelectionUI();
        }
        this.updateDomWidgetHeight();
        this.syncOutputUIFromTimeline();
        this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
        if (!isBatch || showBatchTrack) this.scheduleRender();
        this.scheduleTimelineSync();
        this.updateRunSelectUI();
    }

    renderGenSrcSlot(el, imageFile, label) {
        if (!el) return;
        el.classList.toggle("has-img", !!imageFile);
        if (imageFile) {
            el.innerHTML = `<img src="${refViewUrl(imageFile)}" alt="">`;
        } else {
            el.textContent = label;
        }
    }

    _paintRefVideoSlot(el, nameEl, refBlock) {
        if (!el) return;
        const ref = refBlock || {};
        const has = !!(ref.videoFile || ref.fileName);
        el.classList.toggle("has-img", false);
        el.classList.toggle("has-video", has);
        if (nameEl) {
            if (has) {
                const dur = ref.duration > 0 ? ` · ${ref.duration.toFixed(2)}s` : "";
                const fps = ref.nativeFps > 0 ? ` · ${Math.round(ref.nativeFps)}fps` : "";
                const dim = ref.width && ref.height ? ` · ${ref.width}×${ref.height}` : "";
                nameEl.textContent = `${ref.fileName || ref.videoFile || ""}${dim}${dur}${fps}`;
            } else {
                nameEl.textContent = "";
            }
        }
        if (!has) {
            el.innerHTML = "";
            el.textContent = t("panel.uploadRefVideo");
            el.onclick = () => this.pickReferenceVideoFile();
            return;
        }
        const viewUrl = this.getReferenceVideoViewUrl(ref);
        el.innerHTML = `
            <video class="bd-ref-video-preview" muted playsinline preload="metadata" controls></video>
            <button type="button" class="bd-ref-replace" title="${t("ref.replace")}">${t("ref.replace")}</button>
            <span class="x" title="${t("ref.removeVideo")}">×</span>`;
        el.onclick = null;
        const video = el.querySelector("video");
        if (video && viewUrl) {
            video.src = viewUrl;
            video.addEventListener("click", (e) => e.stopPropagation());
            video.addEventListener("dblclick", (e) => {
                e.stopPropagation();
                if (video.paused) video.play().catch(() => {});
                else video.pause();
            });
        }
        const replaceBtn = el.querySelector(".bd-ref-replace");
        if (replaceBtn) {
            replaceBtn.onclick = (e) => {
                e.stopPropagation();
                this.pickReferenceVideoFile();
            };
        }
        const removeBtn = el.querySelector(".x");
        if (removeBtn) {
            removeBtn.onclick = (e) => {
                e.stopPropagation();
                this.clearReferenceVideo();
            };
        }
    }

    renderRefVideoSlot() {
        if (this.isGlobalMode()) {
            this._stopRefVideoPreviews([this.segRefVideo]);
            this._paintRefVideoSlot(
                this.globalRefVideo,
                this.globalRefVideoNameEl,
                this.timeline.global?.referenceVideo || {},
            );
        } else {
            this._stopRefVideoPreviews([this.globalRefVideo]);
            const seg = this.timeline.segments[this.selectedIndex];
            this._paintRefVideoSlot(this.segRefVideo, this.segRefVideoNameEl, seg?.referenceVideo || {});
        }
    }

    _activeRefVideoTaskKey() {
        if (this.isGlobalMode()) return this.getTaskKey();
        const seg = this.timeline.segments[this.selectedIndex];
        return resolveTaskKey(seg?.taskType || this.timeline.global?.taskType || this.getTaskKey());
    }

    pickReferenceVideoFile() {
        if (!taskUsesReferenceVideo(this._activeRefVideoTaskKey())) return;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*";
        input.onchange = () => {
            if (input.files?.[0]) this.loadReferenceVideoFile(input.files[0]);
        };
        input.click();
    }

    async pickExistingReferenceVideo() {
        if (!taskUsesReferenceVideo(this._activeRefVideoTaskKey())) return;
        const currentValue = this.getRefVideoTarget()?.referenceVideo?.videoFile || "";
        const picked = await this.chooseVideoInput({
            title: t("mediaPicker.pickReferenceVideo"),
            currentValue,
        });
        if (!picked?.relPath) return;
        const slotEl = this.isGlobalMode() ? this.globalRefVideo : this.segRefVideo;
        const nameEl = this.isGlobalMode() ? this.globalRefVideoNameEl : this.segRefVideoNameEl;
        const status = t("upload.inProgress", { name: picked.fileName || picked.relPath });
        if (slotEl) {
            slotEl.classList.remove("has-img", "has-video");
            slotEl.textContent = status;
        }
        if (nameEl) nameEl.textContent = status;
        try {
            const prep = await this._prepareVideoFrames({
                fileName: picked.fileName || picked.relPath,
                relPath: picked.relPath,
                subfolder: picked.subfolder || "",
                type: picked.type || "input",
                statusPrefix: t("parse.refVideo"),
                syncNativeFps: false,
            });
            this.getRefVideoTarget().referenceVideo = this._buildClipRecord(prep);
            this.renderRefVideoSlot();
            this.commit(false, { syncTimeline: true });
        } catch (err) {
            console.error("[MiniMax H3Director] reference video load failed:", err);
            if (nameEl) nameEl.textContent = t("upload.refVideoFailed", { err: formatUploadError(err) });
            this.renderRefVideoSlot();
        }
    }

    clearReferenceVideo() {
        const target = this.getRefVideoTarget();
        this._stopRefVideoPreviews();
        target.referenceVideo = {};
        this.renderRefVideoSlot();
        this.commit();
    }

    async loadReferenceVideoFile(file) {
        const slotEl = this.isGlobalMode() ? this.globalRefVideo : this.segRefVideo;
        const nameEl = this.isGlobalMode() ? this.globalRefVideoNameEl : this.segRefVideoNameEl;
        const status = t("upload.inProgress", { name: file.name });
        if (slotEl) {
            slotEl.classList.remove("has-img", "has-video");
            slotEl.textContent = status;
        }
        if (nameEl) nameEl.textContent = status;
        try {
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > COMFY_UPLOAD_SOFT_LIMIT ? t("upload.chunkMode") : t("upload.mode");
                if (nameEl) {
                    nameEl.textContent = t("upload.refVideoProgress", {
                        mode, name: file.name, cur, total, pct,
                    });
                }
            });
            const relPath = videoRelativePath(uploaded);
            const prep = await this._prepareVideoFrames({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: t("parse.refVideo"),
                syncNativeFps: false,
            });
            this.getRefVideoTarget().referenceVideo = this._buildClipRecord(prep);
            this.renderRefVideoSlot();
            this.commit(false, { syncTimeline: true });
        } catch (err) {
            console.error("[MiniMax H3Director] reference video load failed:", err);
            if (nameEl) nameEl.textContent = t("upload.refVideoFailed", { err: formatUploadError(err) });
            this.renderRefVideoSlot();
        }
    }

    pickGenSrcImage(isGlobal) {
        if (!this.isGenImage()) return;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.onchange = async () => {
            const file = input.files?.[0];
            if (!file) return;
            try {
                const uploaded = await uploadToInput(file);
                const relPath = videoRelativePath(uploaded);
                if (isGlobal) {
                    this.timeline.global = this.timeline.global || { refs: [] };
                    this.timeline.global.genImage = { imageFile: relPath };
                } else {
                    const seg = this.timeline.segments[this.selectedIndex];
                    if (seg) {
                        seg.genImage = { imageFile: relPath };
                        seg.imageFile = relPath;
                    }
                }
                this.commit();
            } catch (err) {
                console.error("[MiniMax H3Director] gen image upload failed:", err);
            }
        };
        input.click();
    }

    onGenDefaultFcChange() {
        const fc = clamp(parseInt(this.genDefaultFc?.value, 10) || 1, minFrameCount(this.getTaskKey()), MAX_GEN_FRAMES);
        if (this.genDefaultFc) this.genDefaultFc.value = fc;
        this.timeline.gen = this.timeline.gen || {};
        this.timeline.gen.defaultFrameCount = fc;
        if (this.timeline.segments.length === 1) {
            this.timeline.segments[0].frameCount = fc;
            this.timeline.segments[0].length = fc;
        }
        this.commit();
    }

    onGenSegFcChange() {
        const seg = this.timeline.segments[this.selectedIndex];
        if (!seg) return;
        const minFc = minFrameCount(this.getTaskKey());
        seg.frameCount = clamp(parseInt(this.genSegFc?.value, 10) || minFc, minFc, MAX_GEN_FRAMES);
        if (this.genSegFc) this.genSegFc.value = seg.frameCount;
        this.commit();
    }

    genSplitAtFrame(frame) {
        const total = this.getTotalFrames();
        const minFc = minFrameCount(this.getTaskKey());
        if (frame <= minFc || frame >= total - minFc) return;
        const newSegs = [];
        let cursor = 0;
        for (const seg of this.timeline.segments) {
            const fc = seg.frameCount ?? seg.length;
            const end = cursor + fc;
            if (frame > cursor && frame < end) {
                const left = frame - cursor;
                const right = end - frame;
                newSegs.push({ ...seg, frameCount: left, length: left });
                newSegs.push({
                    id: uid(), start: frame, frameCount: right, length: right,
                    prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
                });
            } else {
                newSegs.push({ ...seg });
            }
            cursor = end;
        }
        this.timeline.segments = newSegs;
        this.commit();
    }

    genEqualSplit() {
        const n = parseInt(this.equalCountInput?.value || "2", 10);
        if (!n || n < 2) return;
        const total = this.getTotalFrames();
        const minFc = minFrameCount(this.getTaskKey());
        const count = clamp(n, 2, Math.max(2, Math.floor(total / minFc)));
        const base = Math.floor(total / count);
        let rem = total - base * count;
        this.timeline.segments = Array.from({ length: count }, () => {
            const fc = base + (rem > 0 ? 1 : 0);
            if (rem > 0) rem -= 1;
            return {
                id: uid(), frameCount: fc, length: fc, prompt: "", taskType: "", refs: [],
                genImage: { imageFile: "" },
            };
        });
        this.commit();
    }

    genDeleteSelectedSegment() {
        if (this.timeline.segments.length <= 1) return;
        this.timeline.segments.splice(this.selectedIndex, 1);
        this.selectedIndex = clamp(this.selectedIndex, 0, this.timeline.segments.length - 1);
        this.commit();
    }

    updateVideoNameLabel() {
        if (this.isFl2vMode()) {
            const shots = this.timeline.shots || [];
            const n = shots.length;
            const total = this.getTotalFrames();
            const withEnd = shots.filter((s) => s.endImage?.imageFile).length;
            const withStart = shots.filter((s) => s.startImage?.imageFile).length;
            const sec = getFl2vTotalDurationSec(this);
            if (!n) {
                this.videoNameEl.textContent = t("videoName.fl2vEmpty", { sec, frames: total });
            } else {
                this.videoNameEl.textContent = t("videoName.fl2vSummary", {
                    n, start: withStart, end: withEnd, sec, frames: total,
                });
            }
            return;
        }
        if (this.isImageBatch()) {
            // Prefer live drag preview so toolbar totals track the divider.
            const segs = this._previewSegments || this.timeline.segments || [];
            const n = segs.length || 0;
            const key = this.getTaskKey();
            if (isVideoBatchTask(key)) {
                let sec = 0;
                let total = 0;
                for (const seg of segs) {
                    const fc = Math.max(1, parseInt(seg.frameCount ?? seg.length, 10) || 1);
                    const raw = Number(seg.durationSec);
                    // During edge drag, frames are authoritative; durationSec may be stale.
                    const resolved = this._previewSegments
                        ? {
                            frames: fc,
                            durationSec: preferredDurationSecFromFrames(fc, 24),
                        }
                        : durationToClampedMiniMaxFrames(
                            Number.isFinite(raw)
                                ? raw
                                : preferredDurationSecFromFrames(fc || defaultFrameCount(key), 24),
                            24,
                        );
                    sec += resolved.durationSec;
                    total += resolved.frames;
                }
                sec = roundDurationSec(sec);
                const play = framesToDurationSec(total, 24);
                this.videoNameEl.textContent = total
                    ? t("videoName.batchVideo", {
                        key,
                        n,
                        sec: sec || play,
                        frames: total,
                        play,
                    })
                    : t("videoName.batchVideoEmpty", { key, n });
            } else {
                this.videoNameEl.textContent = t("videoName.batchImage", { key, n });
            }
            return;
        }
        if (this.isGenMode()) {
            const total = this.getTotalFrames();
            const key = this.getTaskKey();
            if (this.isGenBlank()) {
                this.videoNameEl.textContent = total
                    ? t("videoName.blankCanvas", { frames: total })
                    : t("videoName.blankCanvasNeedFrames");
            } else {
                this.videoNameEl.textContent = total
                    ? `${key} · ${total}f`
                    : t("videoName.genNeedSource", { key });
            }
            return;
        }
        const clips = this.getVideoClips();
        const total = this.getTotalFrames();
        if (!clips.length || !total) {
            this.videoNameEl.textContent = t("toolbar.noVideo");
            return;
        }
        if (clips.length === 1) {
            const c = clips[0];
            const nativeWh = c.width && c.height ? `${c.width}×${c.height}` : "";
            const storeW = c.storageWidth || this._storageWidth;
            const storeH = c.storageHeight || this._storageHeight;
            const storeWh = storeW && storeH ? `${storeW}×${storeH}` : "";
            let dim = "";
            if (nativeWh && storeWh && nativeWh !== storeWh) dim = ` · ${nativeWh} → ${storeWh}`;
            else if (nativeWh) dim = ` · ${nativeWh}`;
            else if (storeWh) dim = ` · ${storeWh}`;
            const nativeHint = c.nativeFps > 0 ? t("canvas.nativeFps", { fps: formatProbeFps(c.nativeFps) }) : "";
            const tlFps = this.getFrameRate();
            const dur = this.getTimelineDurationSec().toFixed(2);
            const name = c.fileName || c.videoFile;
            this.videoNameEl.textContent = t("videoName.singleClip", {
                name, total, fps: formatProbeFps(tlFps), dur, native: nativeHint, dim,
            });
            return;
        }
        const tlFps = this.getFrameRate();
        const dur = this.getTimelineDurationSec().toFixed(2);
        this.videoNameEl.textContent = t("videoName.multiClip", {
            n: clips.length, total, fps: formatProbeFps(tlFps), dur,
        });
    }

    getFrameMapEntry(logicalFrame) {
        const map = this.getFrameMap();
        if (map.length) return normalizeFrameMapEntry(map[clamp(logicalFrame, 0, map.length - 1)]);
        return { clip: 0, frame: logicalToSourceFrame(logicalFrame, this.timeline.video || {}) };
    }

    getSegmentClipIndex(seg) {
        return this.getFrameMapEntry(seg.start).clip;
    }

    getClipBoundaries() {
        const map = this.getFrameMap();
        const boundaries = [];
        for (let i = 1; i < map.length; i++) {
            const a = normalizeFrameMapEntry(map[i - 1]);
            const b = normalizeFrameMapEntry(map[i]);
            if (b.clip !== a.clip) boundaries.push(i);
        }
        return boundaries;
    }

    _segmentMetaAtFrame(frame) {
        const segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        for (const seg of segs) {
            if (frame >= seg.start && frame < seg.start + seg.length) {
                return {
                    prompt: seg.prompt || "",
                    taskType: seg.taskType || "",
                    refs: seg.refs ? JSON.parse(JSON.stringify(seg.refs)) : [],
                };
            }
        }
        const last = segs[segs.length - 1];
        if (last) {
            return {
                prompt: last.prompt || "",
                taskType: last.taskType || "",
                refs: last.refs ? JSON.parse(JSON.stringify(last.refs)) : [],
            };
        }
        return { prompt: "", taskType: "", refs: [] };
    }

    _buildSegmentsFromSplitPoints(points, forcedPoints = null) {
        const forced = new Set(forcedPoints || []);
        forced.add(0);
        const sorted = [...new Set(points)].sort((a, b) => a - b);
        forced.add(sorted[sorted.length - 1]);
        const newSegs = [];
        for (let i = 0; i < sorted.length - 1; i++) {
            const start = sorted[i];
            const length = sorted[i + 1] - start;
            const endsForced = forced.has(sorted[i + 1]);
            const startsForced = forced.has(start);
            if (length < MIN_SEG && !endsForced && !startsForced) continue;
            if (length < 1) continue;
            const meta = this._segmentMetaAtFrame(start);
            newSegs.push({
                id: uid(),
                start,
                length,
                prompt: meta.prompt,
                taskType: meta.taskType,
                refs: meta.refs,
            });
        }
        if (!newSegs.length) return null;
        let cursor = 0;
        return newSegs.map((seg) => {
            const s = { ...seg, start: cursor, length: seg.length };
            cursor += s.length;
            return s;
        });
    }

    _getReorderInsertFrame(dropRank, fromRank) {
        const ordered = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        const lengths = ordered.map((s) => s.length);
        const without = lengths.filter((_, i) => i !== fromRank);
        let frame = 0;
        for (let i = 0; i < dropRank && i < without.length; i++) frame += without[i];
        return frame;
    }

    _orderedSegmentsWithRank() {
        return [...this.timeline.segments]
            .map((seg, arrayIndex) => ({ seg, arrayIndex }))
            .sort((a, b) => a.seg.start - b.seg.start)
            .map((item, visualRank) => ({ ...item, visualRank }));
    }

    _visualRankFromArrayIndex(arrayIndex) {
        const ordered = this._orderedSegmentsWithRank();
        return ordered.find((o) => o.arrayIndex === arrayIndex)?.visualRank ?? arrayIndex;
    }

    _computeReorderDropRank(frame, fromRank) {
        const ordered = this._orderedSegmentsWithRank();
        if (!ordered.length) return fromRank;

        // fl2v: swap slots — drop target = the clip currently under the pointer.
        if (this.isFl2vMode()) {
            for (const item of ordered) {
                const lo = item.seg.start;
                const hi = item.seg.start + item.seg.length;
                if (frame >= lo && frame < hi) return item.visualRank;
            }
            // In a gap / past the end: snap to nearest clip by center distance.
            let best = fromRank;
            let bestDist = Infinity;
            for (const item of ordered) {
                const mid = item.seg.start + item.seg.length / 2;
                const d = Math.abs(frame - mid);
                if (d < bestDist) {
                    bestDist = d;
                    best = item.visualRank;
                }
            }
            return best;
        }

        // Video / gen / batch: return the final insertion index after removing
        // the dragged item. This keeps forward moves from collapsing to no-op.
        const remaining = ordered.filter((item) => item.visualRank !== fromRank);
        for (let index = 0; index < remaining.length; index++) {
            const item = remaining[index];
            const mid = item.seg.start + item.seg.length / 2;
            if (frame < mid) return index;
        }
        return remaining.length;
    }

    reorderSegmentsByRank(fromRank, toRank) {
        const ordered = [...this.timeline.segments]
            .map((seg) => ({ seg }))
            .sort((a, b) => a.seg.start - b.seg.start);
        if (fromRank < 0 || fromRank >= ordered.length) return;
        if (toRank < 0 || toRank >= ordered.length) return;
        if (fromRank === toRank) return;

        // fl2v: reorder shots[] (source of truth), then rebuild segments.
        if (this.isFl2vMode()) {
            const shots = [...(this.timeline.shots || [])];
            if (fromRank < 0 || fromRank >= shots.length) return;
            if (toRank < 0 || toRank >= shots.length) return;
            const [moved] = shots.splice(fromRank, 1);
            const insertRank = toRank;
            shots.splice(insertRank, 0, moved);
            this.timeline.shots = shots;
            syncFl2vFromShots(this);
            this.selectedIndex = insertRank;
            updateFl2vDetailUI(this);
            this.updateVideoNameLabel();
            return;
        }
        // r2v / t2v / i2v: move whole groups then renumber starts.
        if (this.usesBatchTimeline()) {
            const metas = ordered.map((o) => ({
                ...o.seg,
                refs: o.seg.refs ? JSON.parse(JSON.stringify(o.seg.refs)) : [],
                refAudios: o.seg.refAudios ? JSON.parse(JSON.stringify(o.seg.refAudios)) : [],
                refVideos: o.seg.refVideos ? JSON.parse(JSON.stringify(o.seg.refVideos)) : [],
            }));
            const [mMeta] = metas.splice(fromRank, 1);
            const insertRank = toRank;
            metas.splice(insertRank, 0, mMeta);
            this.timeline.segments = metas;
            normalizeImageBatchSegments(this);
            this.selectedIndex = insertRank;
            this.updateVideoNameLabel();
            return;
        }
        // gen: no video frameMap — reorder by segment metadata only.
        if (this.isGenMode()) {
            const metas = ordered.map((o) => ({
                ...o.seg,
                refs: o.seg.refs ? JSON.parse(JSON.stringify(o.seg.refs)) : [],
            }));
            const slots = ordered.map((o) => ({
                start: o.seg.start,
                length: o.seg.length || o.seg.frameCount || minFrameCount(this.getTaskKey()),
            }));
            const [mMeta] = metas.splice(fromRank, 1);
            const insertRank = toRank;
            metas.splice(insertRank, 0, mMeta);
            for (let i = 0; i < metas.length; i++) {
                const slot = slots[i] || slots[slots.length - 1];
                metas[i].start = slot.start;
                metas[i].length = slot.length;
                metas[i].frameCount = slot.length;
            }
            this.timeline.segments = metas;
            this.normalizeGenSegments();
            this.selectedIndex = insertRank;
            this.updateVideoNameLabel();
            return;
        }

        if (!this.getFrameMap().length && this.getTotalFrames() > 0) {
            this.materializeFrameMap();
        }
        const map = [...this.getFrameMap()];
        const slices = ordered.map((o) => map.slice(o.seg.start, o.seg.start + o.seg.length));
        const metas = ordered.map((o) => ({
            ...o.seg,
            refs: o.seg.refs ? JSON.parse(JSON.stringify(o.seg.refs)) : [],
        }));

        const [mSlice] = slices.splice(fromRank, 1);
        const [mMeta] = metas.splice(fromRank, 1);
        const insertRank = toRank;
        slices.splice(insertRank, 0, mSlice);
        metas.splice(insertRank, 0, mMeta);

        const newMap = slices.flat();
        let start = 0;
        const newSegs = metas.map((seg, idx) => {
            const s = { ...seg, start, length: slices[idx].length };
            start += s.length;
            return s;
        });

        this.setFrameMap(newMap);
        this.timeline.segments = newSegs;
        this._syncPrimaryVideoFromClips(newMap);
        this.selectedIndex = insertRank;
        this._prefetchSegmentThumbs(0, Math.min(newMap.length, THUMB_PREFETCH_BATCH * 4));
    }

    materializeFrameMap() {
        const total = this.getTotalFrames();
        const video = this.timeline.video || {};
        if (video.frameMap?.length === total) return;
        const map = [];
        for (let i = 0; i < total; i++) map.push(this.getFrameMapEntry(i));
        video.frameMap = map;
        video.deletedSourceRanges = [];
        this.timeline.video = video;
        this.timeline.totalFrames = total;
    }

    getFrameMap() {
        const v = this.timeline?.video || {};
        if (v.frameMap?.length) return v.frameMap;
        if (this._legacyFrames.length) return buildIdentityFrameMap(this._legacyFrames.length);
        if (v.frames?.length) return buildIdentityFrameMap(v.frames.length);
        return [];
    }

    setFrameMap(map) {
        this.timeline.video = this.timeline.video || {};
        this.timeline.video.frameMap = map;
        if (map.length) {
            this.timeline.totalFrames = map.length;
            this.timeline.video.deletedSourceRanges = [];
        }
    }

    setSparseVideoFrames(totalFrames) {
        this.timeline.video = this.timeline.video || {};
        this.timeline.video.frameMap = [];
        this.timeline.video.sourceFrameCount = totalFrames;
        this.timeline.video.deletedSourceRanges = [];
        this.timeline.totalFrames = totalFrames;
    }

    logicalToSourceFrame(logical) {
        return logicalToSourceFrame(logical, this.timeline.video || {});
    }

    getTotalFrames() {
        // fl2v: visual canvas may be longer than the sampling window (overflow = dashed).
        if (this.isFl2vMode()) return getFl2vVisualFrames(this);
        if (this.isImageBatch() || this.isGenMode()) {
            // t2v/i2v: never use drag preview for totals (inputs are the source of truth).
            // r2v may temporarily use _previewSegments while resizing on the timeline.
            if (this.usesBatchTimeline() && this._previewSegments) {
                return sumFrameCounts(this._previewSegments);
            }
            return sumFrameCounts(this.timeline.segments);
        }
        const mapLen = this.timeline?.video?.frameMap?.length || 0;
        if (mapLen > 0) return mapLen;
        // Sparse deletes: sourceFrameCount − ranges beats a stale totalFrames.
        const src = parseInt(this.timeline?.video?.sourceFrameCount || 0, 10);
        if (src > 0) {
            const removed = deletedSourceRanges(this.timeline.video).reduce((s, [a, b]) => s + (b - a), 0);
            return Math.max(0, src - removed);
        }
        const total = Math.max(0, parseInt(this.timeline?.totalFrames || this.totalFramesWidget?.value || 0, 10));
        if (total > 0) return total;
        if (!this.hasVideo()) return 0;
        return 0;
    }

    getMaxExportFrames() {
        const n = parseInt(this.timeline.output?.maxExportFrames ?? 0, 10);
        return Number.isFinite(n) && n > 0 ? n : 0;
    }

    getExportFrameTotal() {
        const total = this.getTotalFrames();
        const cap = this.getMaxExportFrames();
        return cap > 0 ? Math.min(total, cap) : total;
    }

    getFrameRate() {
        return coerceTimelineFps(this.fpsInput?.value ?? this.frameRateWidget?.value ?? this.timeline.frameRate ?? 24);
    }

    syncFrameRateUI(value = null) {
        const fps = coerceTimelineFps(value ?? this.fpsInput?.value ?? this.frameRateWidget?.value ?? this.timeline.frameRate ?? 24);
        this.timeline.frameRate = fps;
        if (this.frameRateWidget) this.frameRateWidget.value = fps;
        if (this.fpsInput) this.fpsInput.value = fps;
        return fps;
    }

    _clipFrameCountAtFps(clip, fps, fallback = 0) {
        const nativeFps = Number(clip?.nativeFps || 0);
        const nativeCount = Number(clip?.nativeFrameCount || 0);
        if (nativeFps > 0 && nativeCount > 0) {
            return Math.max(1, Math.round((nativeCount / nativeFps) * fps));
        }
        const duration = Number(clip?.duration || 0);
        if (duration > 0) return Math.max(1, Math.round(duration * fps));
        return Math.max(1, Math.round(fallback || Number(clip?.sourceFrameCount || 0) || 1));
    }

    _timelineFrameCountAtFps(fps, oldFps = null, oldTotal = null) {
        const nextFps = coerceTimelineFps(fps);
        const prevTotal = Number(oldTotal ?? this.getTotalFrames() ?? 0);
        const prevFps = coerceTimelineFps(oldFps ?? this.timeline.frameRate ?? this.frameRateWidget?.value ?? 24);
        // When user changes timeline FPS, preserve wall-clock duration: T = N/fps → N' = T * fps'.
        if (prevTotal > 0 && oldFps != null && Math.abs(prevFps - nextFps) >= 0.001) {
            return Math.max(1, Math.round(prevTotal * nextFps / prevFps));
        }
        const clips = this.getVideoClips();
        if (clips.length && clips.some((c) => Number(c.duration || 0) > 0 || Number(c.nativeFrameCount || 0) > 0)) {
            return clips.reduce((sum, clip) => sum + this._clipFrameCountAtFps(clip, nextFps), 0);
        }
        if (prevTotal > 0) {
            return Math.max(1, Math.round(prevTotal * nextFps / Math.max(prevFps, 0.001)));
        }
        return 1;
    }

    _rescaleSegmentsForTotal(oldTotal, newTotal) {
        if (!oldTotal || !newTotal || !this.timeline.segments?.length) {
            this._setSingleSegment(newTotal);
            return;
        }
        const ordered = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        let cursor = 0;
        this.timeline.segments = ordered.map((seg, idx) => {
            const rawStart = idx === 0 ? 0 : Math.round((seg.start / oldTotal) * newTotal);
            const rawEnd = idx === ordered.length - 1
                ? newTotal
                : Math.round(((seg.start + seg.length) / oldTotal) * newTotal);
            const start = clamp(rawStart, cursor, newTotal);
            const end = clamp(rawEnd, start + 1, newTotal);
            cursor = end;
            return {
                ...seg,
                start,
                length: Math.max(1, end - start),
                frameCount: Math.max(1, end - start),
            };
        });
    }

    _syncClipFrameCountsForFps(fps, oldFps = null) {
        const clips = this.getVideoClips();
        if (!clips.length) return;
        const prevFps = coerceTimelineFps(oldFps ?? this.timeline.frameRate ?? 24);
        this.timeline.videoClips = clips.map((clip) => {
            const fallback = Number(clip.sourceFrameCount || 0) * fps / Math.max(prevFps, 0.001);
            return { ...clip, sourceFrameCount: this._clipFrameCountAtFps(clip, fps, fallback) };
        });
    }

    _resampleFrameMapForFps(oldFps, newFps, newTotal) {
        const oldTotal = this.getTotalFrames();
        if (!oldTotal || !newTotal) return [];
        const oldEntries = Array.from({ length: oldTotal }, (_, i) => this.getFrameMapEntry(i));
        const clips = this.getVideoClips();
        const map = [];
        for (let i = 0; i < newTotal; i++) {
            const oldLogical = clamp(Math.round((i / newFps) * oldFps), 0, oldTotal - 1);
            const entry = normalizeFrameMapEntry(oldEntries[oldLogical]);
            const clip = clips[entry.clip] || clips[0] || {};
            const maxFrame = this._clipFrameCountAtFps(clip, newFps) - 1;
            const sourceTime = Number(entry.frame || 0) / Math.max(oldFps, 0.001);
            map.push({
                clip: entry.clip,
                frame: clamp(Math.round(sourceTime * newFps), 0, Math.max(0, maxFrame)),
            });
        }
        return map;
    }

    _resampleTimelineForFrameRate(oldFps, newFps) {
        if (this.isImageBatch() || this.isGenMode() || !this.hasVideo()) return;
        const oldTotal = this.getTotalFrames();
        const newTotal = this._timelineFrameCountAtFps(newFps, oldFps, oldTotal);
        const hasExplicitMap = this.getFrameMap().length > 0;
        const hasSparseDeletes = deletedSourceRanges(this.timeline.video || {}).length > 0;

        if (hasExplicitMap || hasSparseDeletes || this.getVideoClips().length > 1) {
            const newMap = this._resampleFrameMapForFps(oldFps, newFps, newTotal);
            this.setFrameMap(newMap);
            this._syncClipFrameCountsForFps(newFps, oldFps);
            this._syncPrimaryVideoFromClips(newMap);
        } else {
            this._syncClipFrameCountsForFps(newFps, oldFps);
            this.setSparseVideoFrames(newTotal);
            this._syncPrimaryVideoFromClips([]);
        }

        this._rescaleSegmentsForTotal(oldTotal, newTotal);
        this.currentFrame = clamp(Math.round((this.currentFrame / Math.max(oldTotal, 1)) * newTotal), 0, Math.max(0, newTotal - 1));
        if (this.totalFramesWidget) this.totalFramesWidget.value = newTotal;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, newTotal - 1);
            this.seekBar.value = this.currentFrame;
        }
        this._prefetchSegmentThumbs(0, Math.min(newTotal, THUMB_PREFETCH_BATCH * 4));
    }

    onFrameRateChanged(value) {
        const oldFps = coerceTimelineFps(this.timeline.frameRate ?? this.frameRateWidget?.value ?? 24);
        const newFps = this.syncFrameRateUI(value);
        if (Math.abs(oldFps - newFps) < 0.001) {
            this.commit(false, { syncTimeline: true });
            return;
        }
        this._resampleTimelineForFrameRate(oldFps, newFps);
        this.updateVideoNameLabel();
        this.updateOutputPreview();
        this.scheduleRender();
        this.commit(false, { syncTimeline: true });
    }

    getTimelineDurationSec() {
        if (this.isFl2vMode()) return getFl2vTotalDurationSec(this);
        const total = this.getTotalFrames();
        const fps = this.getFrameRate();
        return total / Math.max(fps, 0.001);
    }

    /** User-facing seconds for ruler ticks (batch 秒数, not MiniMax-aligned play length). */
    getRulerDurationSec() {
        if (this.isFl2vMode()) return Math.max(0.001, getFl2vTotalDurationSec(this));
        if (this.usesBatchTimeline()) {
            const segs = this._previewSegments || this.timeline.segments || [];
            let sec = 0;
            const dragging = !!this._previewSegments;
            for (const seg of segs) {
                const fc = Math.max(0, parseInt(seg.frameCount ?? seg.length, 10) || 0);
                const raw = Number(seg.durationSec);
                if (dragging) {
                    sec += preferredDurationSecFromFrames(fc, 24);
                } else if (Number.isFinite(raw) && raw > 0) {
                    sec += raw;
                } else if (fc > 0) {
                    sec += preferredDurationSecFromFrames(fc, 24);
                }
            }
            return Math.max(0.001, roundDurationSec(sec));
        }
        return Math.max(0.001, this.getTimelineDurationSec());
    }

    isGlobalMode() { return (this.timeline.editMode || "global") === "global"; }

    /** r2v batch: show timeline.global as shared params for all asset groups. */
    usesR2vCommonPanel() {
        return !!this.isR2vBatch?.();
    }

    /** Whether shared common params are enabled at run time. Default off. */
    isR2vCommonEnabled() {
        if (!this.usesR2vCommonPanel()) return false;
        return !!(this.timeline?.global?.commonEnabled ?? this.timeline?.global?.common_enabled);
    }

    /** UI-only fold; when enabled+collapsed, runtime still merges common params. */
    isR2vCommonCollapsed() {
        if (!this.isR2vCommonEnabled()) return true;
        return !!(this.timeline?.global?.commonCollapsed ?? this.timeline?.global?.common_collapsed);
    }

    /** Global / shared-ref panel owns timeline.global refs + prompt when enabled. */
    usesGlobalRefPanel() {
        return this.isGlobalMode() || this.isR2vCommonEnabled();
    }

    syncR2vCommonCollapse() {
        const r2v = this.usesR2vCommonPanel();
        const on = this.isR2vCommonEnabled();
        const folded = this.isR2vCommonCollapsed();
        const bodyHidden = !on || folded;
        this.globalPanel?.classList.toggle("bd-r2v-common-panel", r2v);
        this.globalPanel?.classList.toggle("bd-r2v-common-collapsed", r2v && bodyHidden);
        this.r2vCommonHint?.classList.toggle("hidden", !r2v || bodyHidden);
        if (this.r2vCommonFold) {
            this.r2vCommonFold.classList.toggle("hidden", !r2v || !on);
            if (r2v && on) {
                const fkey = folded ? "panel.r2vCommonExpand" : "panel.r2vCommonCollapse";
                this.r2vCommonFold.textContent = t(fkey);
                this.r2vCommonFold.setAttribute("data-i18n", fkey);
                this.r2vCommonFold.title = t(
                    folded ? "tooltip.r2vCommonExpand" : "tooltip.r2vCommonCollapse",
                );
            }
        }
        if (this.r2vCommonToggle) {
            this.r2vCommonToggle.classList.toggle("on", on);
            const key = on ? "panel.r2vCommonDisable" : "panel.r2vCommonEnable";
            this.r2vCommonToggle.textContent = t(key);
            this.r2vCommonToggle.setAttribute("data-i18n", key);
            this.r2vCommonToggle.title = t(on ? "tooltip.r2vCommonDisable" : "tooltip.r2vCommonEnable");
        }
        if (this.r2vCommonStatus) {
            this.r2vCommonStatus.classList.toggle("on", on);
            const skey = !on
                ? "panel.r2vCommonOff"
                : (folded ? "panel.r2vCommonOnCollapsed" : "panel.r2vCommonOn");
            this.r2vCommonStatus.textContent = t(skey);
            this.r2vCommonStatus.setAttribute("data-i18n", skey);
        }
        if (r2v && this.globalPrompt) {
            this.globalPrompt.placeholder = t("placeholder.r2vCommonPrompt");
            this.globalPrompt.setAttribute("data-i18n-placeholder", "placeholder.r2vCommonPrompt");
        }
        // Keep shared layout class in sync so image/audio slot chrome paints correctly.
        // Layout chrome follows enablement (not UI fold) so group inherit previews stay correct.
        if (r2v) {
            this.globalPromptLayout?.classList.toggle("bd-rv2v-layout", on);
            this.globalPanel?.classList.toggle("bd-rv2v-panel", on);
        }
    }

    setEditMode(mode) {
        this.timeline.editMode = mode;
        this.root.querySelector('[data-a="mode-global"]').classList.toggle("active", mode === "global");
        this.root.querySelector('[data-a="mode-segment"]').classList.toggle("active", mode === "segment");
        this.updateModeUI();
        this.commit();
    }

    updateModeUI() {
        const global = this.isGlobalMode();
        const r2vCommon = this.usesR2vCommonPanel();
        const r2vOn = this.isR2vCommonEnabled();
        this.globalPanel.style.display = (global || r2vCommon) ? "flex" : "none";
        this.segmentPanel.style.display = (global || r2vCommon) ? "none" : "flex";
        this.syncR2vCommonCollapse();
        this.updateReferenceImageVisibility({
            // Show shared ref chrome only when r2v common is enabled (expanded).
            hideTimeline: (this.isImageBatch() && !r2vOn) || this.isGenMode(),
            seg: (global || r2vOn) ? null : this.timeline.segments[this.selectedIndex],
        });
        if (!global && !r2vCommon) this.updateSelectionUI();
        else {
            this.updateSelectionUI();
            if (taskUsesReferenceVideo(this.getTaskKey())) this.renderRefVideoSlot();
        }
        this.updateLiveSamplePanel();
    }

    getRefTarget() {
        if (this.usesGlobalRefPanel()) return this.timeline.global;
        const seg = this.timeline.segments[this.selectedIndex];
        return seg || this.timeline.global;
    }

    getDisplayPrompt(seg) {
        if (this.isGlobalMode()) return this.timeline.global?.prompt || "";
        return seg?.prompt || "";
    }

    populateTaskSelect(el, selected) {
        if (!el) return;
        const opts = this.taskTypeWidget?.options?.values || [];
        const prev = selected || el.value;
        el.innerHTML = "";
        for (const v of opts) {
            const o = document.createElement("option");
            o.value = v;
            const key = resolveTaskKey(v);
            o.textContent = taskDisplayLabel(key) || v;
            el.appendChild(o);
        }
        if (prev) el.value = prev;
    }

    refreshAspectSelectLabels() {
        if (!this.outAspect) return;
        const cur = this.outAspect.value;
        for (const opt of this.outAspect.options || []) {
            opt.textContent = aspectDisplayLabel(opt.value);
        }
        if (cur) this.outAspect.value = cur;
    }

    applyLocale() {
        this.root?.classList.toggle("locale-en", getLocale() === "en");
        this.root?.classList.toggle("locale-zh", getLocale() !== "en");
        applyI18nDom(this.root);
        applyDirectorWidgetLabels(this.node);
        this.populateTaskSelect(this.globalTask, this.taskTypeWidget?.value || this.globalTask?.value);
        this.refreshAspectSelectLabels();
        // Re-apply dynamic UI strings that overwrite data-i18n nodes.
        this.updateVideoNameLabel?.();
        this.updateRunSelectUI?.();
        this.updateOutputPreview?.();
        this.updateSelectionUI?.();
        this.refreshLoopButtonTitle?.();
        this.refreshLiveTaePreviewButton?.();
        this.updateLiveSamplePanel?.();
        this.syncTimelineZoomUI?.();
        this.syncExternalGroupsTimeline?.();
        updateFl2vDetailUI?.(this);
        updateFl2vToolbarBtns?.(this);
        updateR2vToolbarBtns?.(this);
        this.renderImageBatchGroups?.();
        const r2vOn = this.isR2vCommonEnabled?.();
        this.syncR2vCommonCollapse?.();
        this.syncRv2vRefLayoutClasses?.({
            hideTimeline: (this.isImageBatch?.() && !r2vOn) || this.isGenMode?.(),
            seg: this.usesGlobalRefPanel?.() ? null : this.timeline?.segments?.[this.selectedIndex],
        });
        if (this.usesGlobalRefPanel?.() && taskUsesReferenceImages(this.getTaskKey())) {
            if (this.timeline?.global) this.timeline.global.refs = this.timeline.global.refs || [];
            this.renderRefSlots?.(this.timeline.global?.refs, this.globalRefsBox, true);
        } else if (!this.usesGlobalRefPanel?.()) {
            const seg = this.timeline?.segments?.[this.selectedIndex];
            if (seg && taskUsesReferenceImages(resolveTaskKey(seg.taskType || this.getTaskKey()))) {
                this.renderRefSlots?.(seg.refs, this.segRefsBox, false);
            }
        }
        if (taskUsesReferenceAudios(this.getTaskKey())) this.renderRefAudioSlots?.();
        if (this.usesR2vCommonPanel?.()) this.renderR2vCommonVideoSlots?.();
        this.scheduleRender?.();
        this.node?.setDirtyCanvas?.(true, true);
    }

    getI2iSourceDimensions() {
        for (const seg of this.timeline.segments || []) {
            const gi = seg.genImage || {};
            const w = +(gi.width || 0);
            const h = +(gi.height || 0);
            if (w > 0 && h > 0) return { width: w, height: h };
        }
        const out = this.timeline.output || {};
        if (+(out.sourceWidth || 0) > 0 && +(out.sourceHeight || 0) > 0) {
            return { width: +out.sourceWidth, height: +out.sourceHeight };
        }
        return { width: 0, height: 0 };
    }

    getSourceDimensions() {
        const clips = this.getVideoClips?.() || [];
        const video = clips[0] || this.timeline.video || {};
        // Prefer native clip/source size — never fall back to output canvas W×H
        // (that makes long_edge look like a no-op and keeps a cropped 16:9).
        if (+(video.width || 0) > 0 && +(video.height || 0) > 0) {
            return { width: +video.width, height: +video.height };
        }
        for (const clip of clips) {
            if (+(clip?.width || 0) > 0 && +(clip?.height || 0) > 0) {
                return { width: +clip.width, height: +clip.height };
            }
        }
        return { width: 0, height: 0 };
    }

    _refreshVideoStorageDimensions(resolved) {
        if (!resolved?.width || !resolved?.height) return;
        this._storageWidth = resolved.width;
        this._storageHeight = resolved.height;
        if (this.timeline.video) {
            this.timeline.video.storageWidth = resolved.width;
            this.timeline.video.storageHeight = resolved.height;
        }
        for (const clip of this.getVideoClips()) {
            clip.storageWidth = resolved.width;
            clip.storageHeight = resolved.height;
        }
    }

    syncOutputUIFromTimeline() {
        const out = this.timeline.output || {
            mode: "long_edge",
            aspectRatio: DEFAULT_ASPECT_RATIO,
            megapixels: DEFAULT_MEGAPIXELS,
            multiple: MINIMAX_CANVAS_MULTIPLE,
            longEdge: 848, width: 848, height: 480,
            maxExportFrames: 0, exportMode: "all",
            audioMode: "generate",
            refImageSize: "match",
            continuityEnabled: false, continuityOverlapFrames: DEFAULT_CONTINUITY_FRAMES,
        };
        // Prefer ResolutionSelector fields; backfill from width/height when missing.
        // Custom keeps explicit width/height and does not recompute from megapixels.
        if (!isCustomAspectRatio(out.aspectRatio) && (out.aspectRatio == null || out.megapixels == null)) {
            const resolved = resolutionFromSelector(
                out.aspectRatio || DEFAULT_ASPECT_RATIO,
                out.megapixels ?? DEFAULT_MEGAPIXELS,
                out.multiple ?? MINIMAX_CANVAS_MULTIPLE,
            );
            if (resolved) {
                out.aspectRatio = resolved.aspectRatio;
                out.megapixels = resolved.megapixels;
                out.multiple = resolved.multiple;
                if (out.width == null) out.width = resolved.width;
                if (out.height == null) out.height = resolved.height;
                this.timeline.output = { ...out };
            }
        }
        if (this.outMode) this.outMode.value = out.mode || "long_edge";
        if (this.outAspect) {
            const ar = isCustomAspectRatio(out.aspectRatio)
                ? CUSTOM_ASPECT_RATIO
                : normalizeAspectRatioLabel(out.aspectRatio || DEFAULT_ASPECT_RATIO);
            this.outAspect.value = ar;
            if (out.aspectRatio !== ar) {
                out.aspectRatio = ar;
                this.timeline.output = { ...out };
            }
        }
        if (this.outMp) this.outMp.value = String(out.megapixels ?? DEFAULT_MEGAPIXELS);
        if (this.outLong) this.outLong.value = String(out.longEdge ?? 864);
        if (this.outW) this.outW.value = String(out.width ?? 864);
        if (this.outH) this.outH.value = String(out.height ?? 480);
        if (this.outMaxFrames) this.outMaxFrames.value = String(out.maxExportFrames ?? 0);
        if (this.outExportMode) this.outExportMode.value = out.exportMode === "segments" ? "segments" : "all";
        if (this.outAudioMode) {
            const am = normalizeAudioMode(out.audioMode);
            this.outAudioMode.value = am;
            if (out.audioMode !== am) {
                out.audioMode = am;
                this.timeline.output = { ...out };
            }
        }
        if (this.segmentContinuityCb) this.segmentContinuityCb.checked = isContinuityEnabled(out);
        if (this.segmentContinuityOverlap) {
            this.segmentContinuityOverlap.value = String(
                snapContinuityFrames(out.continuityOverlapFrames ?? DEFAULT_CONTINUITY_FRAMES),
            );
        }
        this.syncFrameRateUI(this.timeline.frameRate);
        this.updateOutputModeUI();
        this.updateSegmentContinuityUI();
        this.syncExportSourceImagesUI();
        this.updateOutputPreview();
    }

    syncExportSourceImagesUI() {
        const show = isVideoEditTaskKey(this.getTaskKey());
        this.exportSourceImagesWrap?.classList.toggle("hidden", !show);
        this.timeline.output = this.timeline.output || {};
        const w = this.widget("export_source_images");
        if (this.timeline.output.exportSourceImages == null && w?.value) {
            this.timeline.output.exportSourceImages = true;
        }
        const on = show && !!this.timeline.output.exportSourceImages;
        if (this.exportSourceImagesCb) this.exportSourceImagesCb.checked = on;
        if (w) w.value = on;
    }

    updateSegmentContinuityUI() {
        const show = isContinuityEligible(this);
        if (this.segmentContinuityWrap) {
            this.segmentContinuityWrap.classList.toggle("hidden", !show);
            this.segmentContinuityWrap.hidden = !show;
            this.segmentContinuityWrap.setAttribute("aria-hidden", show ? "false" : "true");
            this.segmentContinuityWrap.title = show
                ? t("tooltip.segmentContinuity")
                : "";
        }
        if (!show && this.timeline?.output) {
            // Hide only — keep saved preference so it returns when multi-segment again.
        }
        if (this.segmentContinuityOverlap && this.timeline?.output) {
            const frames = snapContinuityFrames(
                this.timeline.output.continuityOverlapFrames ?? DEFAULT_CONTINUITY_FRAMES,
            );
            this.segmentContinuityOverlap.value = String(frames);
            this.timeline.output.continuityOverlapFrames = frames;
        }
        if (this.segmentContinuityCb && this.timeline?.output) {
            // Keep DOM aligned with timeline; eligibility only gates visibility.
            this.segmentContinuityCb.checked = isContinuityEnabled(this.timeline.output);
        }
        this.syncSegmentContinuityFromPrevUI();
        this.syncSegmentRefImageSizeUI();
    }

    /** Per-segment「引用上段」on v2v/rv2v segment panel (index>0 + master on). */
    syncSegmentContinuityFromPrevUI() {
        const wrap = this.segContinuityFromPrevWrap;
        const cb = this.segContinuityFromPrevCb;
        if (!wrap || !cb) return;
        const idx = this.selectedIndex ?? 0;
        const masterOn = isContinuityEligible(this)
            && isContinuityMasterEnabled(this.timeline?.output);
        const show = masterOn && idx > 0 && !this.isImageBatch() && !this.isFl2vMode();
        wrap.classList.toggle("hidden", !show);
        wrap.hidden = !show;
        if (!show) return;
        const seg = this.timeline.segments?.[idx];
        cb.checked = isSegmentContinuityFromPrev(seg, idx);
        wrap.title = t("tooltip.segmentContinuityFromPrev");
    }

    /** Per-segment ref_image_size for rv2v (r2v uses the group card control). */
    syncSegmentRefImageSizeUI() {
        const wrap = this.segRefImageSizeWrap;
        const sel = this.segRefImageSize;
        if (!wrap || !sel) return;
        const show = this.getTaskKey() === "rv2v" && !this.isImageBatch() && !this.isFl2vMode();
        wrap.classList.toggle("hidden", !show);
        wrap.hidden = !show;
        if (!show) return;
        const seg = this.timeline.segments?.[this.selectedIndex ?? 0];
        const value = resolveSegmentRefImageSize(seg, this.timeline.output);
        sel.value = value;
        if (seg && seg.refImageSize !== value) seg.refImageSize = value;
        wrap.title = t("tooltip.refImageSize");
    }

    /** Apply ResolutionSelector → fixed width/height on timeline + node widgets. */
    applyResolutionSelector(aspectRatio = null, megapixels = null) {
        const out = this.timeline.output || {};
        const ar = aspectRatio ?? out.aspectRatio ?? this.outAspect?.value ?? DEFAULT_ASPECT_RATIO;
        if (isCustomAspectRatio(ar)) {
            return this.applyCustomResolution(out.width, out.height);
        }
        const resolved = resolutionFromSelector(
            ar,
            megapixels ?? out.megapixels ?? this.outMp?.value ?? DEFAULT_MEGAPIXELS,
            out.multiple ?? MINIMAX_CANVAS_MULTIPLE,
        );
        if (!resolved) {
            return this.applyCustomResolution(out.width, out.height);
        }
        this.timeline.output = {
            ...out,
            mode: "fixed",
            aspectRatio: resolved.aspectRatio,
            megapixels: resolved.megapixels,
            multiple: resolved.multiple,
            width: resolved.width,
            height: resolved.height,
            longEdge: Math.max(resolved.width, resolved.height),
        };
        if (this.widthWidget) this.widthWidget.value = resolved.width;
        if (this.heightWidget) this.heightWidget.value = resolved.height;
        if (this.refMaxWidget) this.refMaxWidget.value = Math.max(resolved.width, resolved.height);
        if (this.outW) this.outW.value = String(resolved.width);
        if (this.outH) this.outH.value = String(resolved.height);
        if (this.outAspect) this.outAspect.value = resolved.aspectRatio;
        // Keep the in-progress typed text while the field is focused.
        if (this.outMp && document.activeElement !== this.outMp) {
            this.outMp.value = String(resolved.megapixels);
        }
        return resolved;
    }

    /** Apply explicit custom width × height (snapped to canvas multiple). */
    applyCustomResolution(width = null, height = null) {
        const out = this.timeline.output || {};
        const mult = out.multiple ?? MINIMAX_CANVAS_MULTIPLE;
        const w = snapResolutionDim(width ?? out.width ?? this.outW?.value ?? this.widthWidget?.value ?? 864, mult);
        const h = snapResolutionDim(height ?? out.height ?? this.outH?.value ?? this.heightWidget?.value ?? 480, mult);
        this.timeline.output = {
            ...out,
            mode: "fixed",
            aspectRatio: CUSTOM_ASPECT_RATIO,
            megapixels: out.megapixels ?? DEFAULT_MEGAPIXELS,
            multiple: mult,
            width: w,
            height: h,
            longEdge: Math.max(w, h),
        };
        if (this.widthWidget) this.widthWidget.value = w;
        if (this.heightWidget) this.heightWidget.value = h;
        if (this.refMaxWidget) this.refMaxWidget.value = Math.max(w, h);
        if (this.outW) this.outW.value = String(w);
        if (this.outH) this.outH.value = String(h);
        if (this.outAspect) this.outAspect.value = CUSTOM_ASPECT_RATIO;
        if (this.outMp) this.outMp.value = String(this.timeline.output.megapixels);
        return {
            width: w,
            height: h,
            megapixels: this.timeline.output.megapixels,
            aspectRatio: CUSTOM_ASPECT_RATIO,
            multiple: mult,
        };
    }

    updateOutputModeUI() {
        const taskKey = this.getTaskKey();
        const useSelector = this.isImageBatch() || this.isGenMode() || this.isFl2vMode()
            || NO_VIDEO_UPLOAD_TASKS.has(taskKey);
        // Gen / batch / fl2v: aspect + megapixels, or Custom width/height.
        // Video edit (v2v): long_edge / fixed — must toggle .hidden (CSS uses !important).
        if (this.outAspect) this.outAspect.classList.toggle("hidden", !useSelector);
        if (this.outMode) this.outMode.classList.toggle("hidden", useSelector);
        if (this.outLongWrap) this.outLongWrap.style.display = "";
        if (useSelector) {
            const custom = isCustomAspectRatio(this.timeline.output?.aspectRatio ?? this.outAspect?.value);
            if (this.outMpWrap) this.outMpWrap.classList.toggle("hidden", custom);
            if (this.outLongWrap) this.outLongWrap.classList.add("hidden");
            if (this.outFixedWrap) this.outFixedWrap.classList.toggle("hidden", !custom);
            if (custom) this.applyCustomResolution();
            else this.applyResolutionSelector();
            return;
        }
        if (this.outMpWrap) this.outMpWrap.classList.add("hidden");
        const mode = this.timeline.output?.mode || "long_edge";
        const isFixed = mode === "fixed";
        if (this.outLongWrap) this.outLongWrap.classList.toggle("hidden", isFixed);
        if (this.outFixedWrap) this.outFixedWrap.classList.toggle("hidden", !isFixed);
    }

    updateOutputPreview() {
        if (!this.outPreview) return;
        if (this.isImageBatch() && (this.getTaskKey() === "i2i" || this.getTaskKey() === "i2v")) {
            const out = this.timeline.output || {};
            if ((out.mode || "long_edge") === "long_edge") {
                const src = this.getI2iSourceDimensions();
                const resolved = resolveOutputDimensions(src.width, src.height, out, {
                    refMaxSize: this.refMaxWidget?.value,
                });
                const note = src.width > 0 ? "" : t("output.preview.needSourceForLongEdge");
                this.outPreview.textContent = `→ ${resolved.width}×${resolved.height}${note}${this._exportPreviewSuffix()}`;
            } else {
                const w = snapDim(+(out.width ?? this.outW?.value ?? 864));
                const h = snapDim(+(out.height ?? this.outH?.value ?? 480));
                this.outPreview.textContent = `→ ${w}×${h}${this._exportPreviewSuffix()}`;
            }
            return;
        }
        if (this.isGenBlank() || this.isImageBatch() || this.isFl2vMode()) {
            const out = this.timeline.output || {};
            if (isCustomAspectRatio(out.aspectRatio)) {
                const w = snapResolutionDim(out.width ?? this.outW?.value ?? 864, out.multiple ?? MINIMAX_CANVAS_MULTIPLE);
                const h = snapResolutionDim(out.height ?? this.outH?.value ?? 480, out.multiple ?? MINIMAX_CANVAS_MULTIPLE);
                this.outPreview.textContent = t("output.preview.custom", { w, h }) + this._exportPreviewSuffix();
                return;
            }
            const resolved = resolutionFromSelector(
                out.aspectRatio || DEFAULT_ASPECT_RATIO,
                out.megapixels ?? DEFAULT_MEGAPIXELS,
                out.multiple ?? MINIMAX_CANVAS_MULTIPLE,
            );
            if (!resolved) {
                const w = snapResolutionDim(out.width ?? 864);
                const h = snapResolutionDim(out.height ?? 480);
                this.outPreview.textContent = `→ ${w}×${h}${this._exportPreviewSuffix()}`;
                return;
            }
            const w = resolved.width;
            const h = resolved.height;
            const ar = resolved.aspectRatio.split(" ")[0];
            this.outPreview.textContent = `→ ${w}×${h} · ${ar} · ${resolved.megapixels}MP${this._exportPreviewSuffix()}`;
            return;
        }
        const src = this.getSourceDimensions();
        const out = this.timeline.output || {};
        const resolved = resolveOutputDimensions(src.width, src.height, out, {
            width: this.widthWidget?.value,
            height: this.heightWidget?.value,
            refMaxSize: this.refMaxWidget?.value,
        });
        if (src.width > 0 && src.height > 0) {
            const mode = (out.mode || "long_edge").toLowerCase();
            const note = mode === "long_edge"
                ? t("output.preview.scaleKeepAspect")
                : t("output.preview.fixedCrop");
            this.outPreview.textContent = `${src.width}×${src.height} → ${resolved.width}×${resolved.height}${note}${this._exportPreviewSuffix()}`;
        } else {
            this.outPreview.textContent = `→ ${resolved.width}×${resolved.height}${t("output.preview.needSourceForLongEdge")}${this._exportPreviewSuffix()}`;
        }
    }

    _exportPreviewSuffix() {
        const cap = this.getMaxExportFrames();
        const exportMode = this.timeline.output?.exportMode === "segments"
            ? t("output.preview.segmentExport")
            : "";
        const dur = this.getTimelineDurationSec().toFixed(2);
        const fps = formatProbeFps(this.getFrameRate());
        const timeHint = t("output.preview.timeFps", { dur, fps });
        if (cap <= 0) return `${timeHint}${exportMode}`;
        const total = this.getTotalFrames();
        const exportTotal = this.getExportFrameTotal();
        if (exportTotal >= total) {
            return `${timeHint}${t("output.preview.exportFrames", { n: exportTotal })}${exportMode}`;
        }
        return `${timeHint}${t("output.preview.exportFramesPartial", { n: exportTotal, total })}${exportMode}`;
    }

    onOutputField(key, value) {
        this.timeline.output = this.timeline.output || {
            mode: "long_edge",
            aspectRatio: DEFAULT_ASPECT_RATIO,
            megapixels: DEFAULT_MEGAPIXELS,
            multiple: MINIMAX_CANVAS_MULTIPLE,
            longEdge: 848, width: 848, height: 480,
            maxExportFrames: 0, exportMode: "all",
            audioMode: "generate",
            refImageSize: "match",
            continuityEnabled: false, continuityOverlapFrames: DEFAULT_CONTINUITY_FRAMES,
        };
        if (key === "aspectRatio") {
            if (isCustomAspectRatio(value)) {
                // Keep current computed size when entering custom mode.
                this.applyCustomResolution(
                    this.timeline.output.width ?? this.outW?.value,
                    this.timeline.output.height ?? this.outH?.value,
                );
            } else {
                this.applyResolutionSelector(value, null);
            }
        } else if (key === "megapixels") {
            const mp = clampMegapixels(value);
            if (!isCustomAspectRatio(this.timeline.output.aspectRatio)) {
                this.applyResolutionSelector(null, mp);
            } else {
                this.timeline.output.megapixels = mp;
            }
        } else if (key === "mode") {
            this.timeline.output.mode = value;
        } else if (key === "longEdge") {
            // Long-edge is a size budget, not a canvas dim — do not snap to 32
            // (848 would become 864). Final W/H still snap via resolveOutputDimensions.
            const n = Math.round(Number(value) || 864);
            this.timeline.output.longEdge = Math.max(32, n);
        } else if (key === "width") {
            const useSelector = this.isImageBatch() || this.isGenMode() || this.isFl2vMode()
                || NO_VIDEO_UPLOAD_TASKS.has(this.getTaskKey());
            if (useSelector) {
                this.applyCustomResolution(value, this.timeline.output.height ?? this.outH?.value);
            } else {
                this.timeline.output.width = snapDim(value || 864);
            }
        } else if (key === "height") {
            const useSelector = this.isImageBatch() || this.isGenMode() || this.isFl2vMode()
                || NO_VIDEO_UPLOAD_TASKS.has(this.getTaskKey());
            if (useSelector) {
                this.applyCustomResolution(this.timeline.output.width ?? this.outW?.value, value);
            } else {
                this.timeline.output.height = snapDim(value || 480);
            }
        } else if (key === "maxExportFrames") {
            const n = parseInt(value, 10);
            this.timeline.output.maxExportFrames = Number.isFinite(n) && n > 0 ? n : 0;
        } else if (key === "exportMode") {
            this.timeline.output.exportMode = value === "segments" ? "segments" : "all";
        } else if (key === "audioMode") {
            this.timeline.output.audioMode = normalizeAudioMode(value);
        } else if (key === "continuityEnabled") {
            this.timeline.output.continuityEnabled = !!value;
        } else if (key === "continuityOverlapFrames") {
            this.timeline.output.continuityOverlapFrames = snapContinuityFrames(value);
        }
        this.syncOutputUIFromTimeline();
        if (this.isFl2vMode()) updateFl2vDetailUI(this);
        // Refresh per-segment「引用上段」checkboxes when master toggle changes.
        if (key === "continuityEnabled") {
            if (this.isImageBatch()) this.renderImageBatchGroups?.();
            this.syncSegmentContinuityFromPrevUI?.();
        }
        this.commit();
        this.flushTimelineSync();
    }

    syncOutputToWidgets() {
        if (this.isImageBatch() && (this.getTaskKey() === "i2i" || this.getTaskKey() === "i2v")) {
            const out = this.timeline.output || {};
            const mode = (out.mode || "long_edge").toLowerCase();
            if (mode === "long_edge") {
                const src = this.getI2iSourceDimensions();
                const resolved = resolveOutputDimensions(src.width, src.height, out, {
                    width: this.widthWidget?.value,
                    height: this.heightWidget?.value,
                    refMaxSize: this.refMaxWidget?.value,
                });
                this.timeline.output = {
                    ...out,
                    mode: "long_edge",
                    longEdge: out.longEdge ?? resolved.refMaxSize,
                    width: resolved.width,
                    height: resolved.height,
                };
                if (this.widthWidget) this.widthWidget.value = resolved.width;
                if (this.heightWidget) this.heightWidget.value = resolved.height;
                if (this.refMaxWidget) this.refMaxWidget.value = resolved.refMaxSize;
                this.timeline.width = resolved.width;
                this.timeline.height = resolved.height;
                this.timeline.refMaxSize = resolved.refMaxSize;
            } else {
                const w = snapDim(+(out.width ?? this.widthWidget?.value ?? 864));
                const h = snapDim(+(out.height ?? this.heightWidget?.value ?? 480));
                this.timeline.output = { ...out, mode: "fixed", width: w, height: h };
                if (this.widthWidget) this.widthWidget.value = w;
                if (this.heightWidget) this.heightWidget.value = h;
                this.timeline.width = w;
                this.timeline.height = h;
            }
            this.updateOutputPreview();
            return;
        }
        if (this.isGenBlank() || this.isImageBatch() || this.isFl2vMode()) {
            const out = this.timeline.output || {};
            const resolved = isCustomAspectRatio(out.aspectRatio)
                ? this.applyCustomResolution(out.width, out.height)
                : this.applyResolutionSelector();
            this.timeline.width = resolved.width;
            this.timeline.height = resolved.height;
            this.timeline.refMaxSize = Math.max(resolved.width, resolved.height);
            this.updateOutputPreview();
            return;
        }
        const src = this.getSourceDimensions();
        const prevOut = this.timeline.output || {};
        const resolved = resolveOutputDimensions(src.width, src.height, prevOut, {
            width: this.timeline.width,
            height: this.timeline.height,
            refMaxSize: this.timeline.refMaxSize,
        });
        // Preserve audioMode / aspect / megapixels etc. — do not rebuild a bare object.
        this.timeline.output = {
            ...prevOut,
            mode: resolved.mode,
            longEdge: prevOut.longEdge ?? resolved.refMaxSize,
            width: resolved.width,
            height: resolved.height,
            maxExportFrames: prevOut.maxExportFrames ?? 0,
            exportMode: prevOut.exportMode ?? "all",
            audioMode: normalizeAudioMode(prevOut.audioMode),
            refImageSize: normalizeRefImageSize(prevOut.refImageSize ?? prevOut.ref_image_size),
            continuityEnabled: isContinuityEnabled(prevOut),
            continuityOverlapFrames: snapContinuityFrames(
                prevOut.continuityOverlapFrames ?? DEFAULT_CONTINUITY_FRAMES,
            ),
        };
        if (this.widthWidget) this.widthWidget.value = resolved.width;
        if (this.heightWidget) this.heightWidget.value = resolved.height;
        if (this.refMaxWidget) this.refMaxWidget.value = resolved.refMaxSize;
        this.timeline.width = resolved.width;
        this.timeline.height = resolved.height;
        this.timeline.refMaxSize = resolved.refMaxSize;
        this._refreshVideoStorageDimensions(resolved);
        this.updateOutputPreview();
    }

    syncFromWidgets() {
        this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {}, continuousReference: false };
        this.timeline.global.taskType = this.globalTask?.value || this.taskTypeWidget?.value || "";
        this.timeline.global.prompt = this.globalPrompt?.value ?? this.globalPromptWidget?.value ?? "";
        if (this.continuousRefCb) {
            this.timeline.global.continuousReference = !!this.continuousRefCb.checked;
        }
        // fl2v: totalFrames stores the sampling window (总时长), not visual overflow length.
        this.timeline.totalFrames = this.isFl2vMode()
            ? getFl2vSampleFrames(this)
            : this.getTotalFrames();
        this.timeline.frameRate = this.getFrameRate();
        this.timeline.output = this.timeline.output || {
            mode: "long_edge", longEdge: 864, width: 864, height: 480,
            maxExportFrames: 0, exportMode: "all",
            audioMode: "generate",
            refImageSize: "match",
            continuityEnabled: false, continuityOverlapFrames: DEFAULT_CONTINUITY_FRAMES,
        };
        if (this.timeline.output.audioMode == null) {
            this.timeline.output.audioMode = "generate";
        }
        if (isVideoEditTaskKey(this.getTaskKey()) && this.exportSourceImagesCb) {
            this.timeline.output.exportSourceImages = !!this.exportSourceImagesCb.checked;
        }
        const exportSrcW = this.widget("export_source_images");
        if (exportSrcW) {
            exportSrcW.value = isVideoEditTaskKey(this.getTaskKey())
                && !!this.timeline.output.exportSourceImages;
        }
        // Sync from DOM when task+segments are eligible — do not rely on CSS
        // "hidden" class (can lag behind equal-split / task changes at queue time).
        const continuityEligible = isContinuityEligible(this);
        if (continuityEligible && this.segmentContinuityCb) {
            this.timeline.output.continuityEnabled = !!this.segmentContinuityCb.checked;
        } else {
            // Normalize stored flag without clearing preference while ineligible.
            this.timeline.output.continuityEnabled = isContinuityEnabled(this.timeline.output);
        }
        if (continuityEligible && this.segmentContinuityOverlap) {
            this.timeline.output.continuityOverlapFrames = snapContinuityFrames(
                this.segmentContinuityOverlap.value
                    ?? this.timeline.output.continuityOverlapFrames
                    ?? DEFAULT_CONTINUITY_FRAMES,
            );
        } else if (this.timeline.output.continuityOverlapFrames != null) {
            this.timeline.output.continuityOverlapFrames = snapContinuityFrames(
                this.timeline.output.continuityOverlapFrames,
            );
        }
        this.syncOutputToWidgets();
    }

    commit(skipRender = false, { syncTimeline = true } = {}) {
        this.syncFromWidgets();
        this.normalizeSegments();
        if (this.isRunSelectEnabled()) this.normalizeRunSelection();
        this.updateRunSelectUI();
        this.updateSegmentContinuityUI();
        if (this.taskTypeWidget) this.taskTypeWidget.value = this.timeline.global.taskType;
        if (this.globalPromptWidget) this.globalPromptWidget.value = this.timeline.global.prompt;
        if (this.negativePromptWidget) {
            const neg = this.globalNegative?.value ?? this.segNegative?.value ?? this.negativePromptWidget.value ?? "";
            this.negativePromptWidget.value = neg;
        }
        if (this.totalFramesWidget) {
            this.totalFramesWidget.value = Math.max(
                0,
                this.isFl2vMode() ? getFl2vSampleFrames(this) : this.getTotalFrames(),
            );
        }
        this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
        if (syncTimeline) this.scheduleTimelineSync();
        if (!skipRender) this.scheduleRender();
        if (this.usesGlobalRefPanel() && taskUsesReferenceImages(this.getTaskKey())) {
            this.renderRefSlots(this.timeline.global.refs, this.globalRefsBox, true);
        }
        if (this.usesGlobalRefPanel() && taskUsesReferenceAudios(this.getTaskKey())) {
            this.renderRefAudioSlots();
        }
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.updateSelectionUI();
    }

    normalizeSegments() {
        if (this.isImageBatch()) {
            this.normalizeImageBatchSegments();
            return;
        }
        if (this.isFl2vMode()) {
            normalizeFl2vSegments(this);
            const n = this.timeline.segments?.length || 0;
            this.selectedIndex = clamp(this.selectedIndex, 0, Math.max(0, n - 1));
            return;
        }
        if (this.isGenMode()) {
            this.normalizeGenSegments();
            return;
        }
        const total = this.getTotalFrames();
        let segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        if (!total) {
            this.timeline.segments = [];
            this.timeline.totalFrames = 0;
            return;
        }
        if (!segs.length) segs = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
        const fixed = [];
        let cursor = 0;
        for (const seg of segs) {
            const start = clamp(seg.start, cursor, total);
            let length = Math.max(MIN_SEG, seg.length ?? (total - start));
            if (start + length > total) length = total - start;
            if (length < MIN_SEG) continue;
            fixed.push({ ...seg, start, length, refs: seg.refs || [] });
            cursor = start + length;
        }
        if (fixed.length && cursor < total) fixed[fixed.length - 1].length += total - cursor;
        this.timeline.segments = fixed;
        this.timeline.totalFrames = total;
        this.selectedIndex = clamp(this.selectedIndex, 0, Math.max(0, fixed.length - 1));
        this.updateSegmentContinuityUI();
    }

    getVideoViewUrl() {
        return this.getClipViewUrl(0);
    }

    getSourceFrameIndex(logicalFrame) {
        return this.getFrameMapEntry(logicalFrame).frame;
    }

    _previewUrlEquals(videoEl, url) {
        if (!videoEl || !url) return false;
        const src = videoEl.currentSrc || videoEl.getAttribute("src") || videoEl.src || "";
        if (!src) return false;
        try {
            return new URL(src, location.href).href === new URL(url, location.href).href;
        } catch {
            return src === url;
        }
    }

    _assignPreviewSrc(videoEl, url) {
        if (!videoEl || !url) return;
        if (this._previewUrlEquals(videoEl, url)) return;
        videoEl.pause();
        videoEl.src = url;
        videoEl.load();
    }

    _getPreviewVideoForClip(clipIndex) {
        const url = this.getClipViewUrl(clipIndex);
        if (!this._previewVideos) this._previewVideos = new Map();
        if (clipIndex === 0 && this._previewVideo && !this._previewVideos.has(0)) {
            this._previewVideos.set(0, this._previewVideo);
        }
        if (!url) return this._previewVideos.get(clipIndex) || (clipIndex === 0 ? this._previewVideo : null);
        let v = this._previewVideos.get(clipIndex);
        if (!v) {
            v = document.createElement("video");
            v.crossOrigin = "anonymous";
            v.muted = true;
            v.playsInline = true;
            v.preload = "auto";
            v.style.cssText = "position:fixed;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none";
            document.body.appendChild(v);
            this._previewVideos.set(clipIndex, v);
        }
        this._assignPreviewSrc(v, url);
        return v;
    }

    async _ensurePreviewReady(clipIndex, timeoutMs = 8000) {
        const v = this._getPreviewVideoForClip(clipIndex);
        if (!v) return null;
        if (v.videoWidth && v.readyState >= 2) return v;
        await new Promise((resolve) => {
            let done = false;
            const finish = () => {
                if (done) return;
                done = true;
                v.removeEventListener("loadeddata", finish);
                v.removeEventListener("canplay", finish);
                resolve();
            };
            v.addEventListener("loadeddata", finish);
            v.addEventListener("canplay", finish);
            if (v.videoWidth && v.readyState >= 2) finish();
            else setTimeout(finish, timeoutMs);
        });
        return v.videoWidth ? v : null;
    }

    _restorePreviewVideos() {
        const clips = this.getVideoClips();
        if (!clips.length) return;
        for (let i = 0; i < clips.length; i++) this._getPreviewVideoForClip(i);
        this._previewVideo = this._previewVideos.get(0) || this._previewVideo;
    }

    _clearPreviewVideos(removeExtra = true) {
        if (!this._previewVideos) return;
        for (const [idx, v] of this._previewVideos.entries()) {
            v.pause();
            if (idx === 0 && v === this._previewVideo) {
                v.removeAttribute("src");
                v.load();
                continue;
            }
            if (removeExtra) {
                v.removeAttribute("src");
                v.load();
                v.remove();
            }
        }
        const keep = this._previewVideo;
        this._previewVideos.clear();
        if (keep) this._previewVideos.set(0, keep);
    }

    async _seekPreviewVideo(timeSec, clipIndex = 0) {
        this._seekChain = this._seekChain.then(() => new Promise((resolve) => {
            const v = this._getPreviewVideoForClip(clipIndex);
            if (!v || !(v.currentSrc || v.getAttribute("src"))) { resolve(); return; }
            const target = Math.max(0, timeSec);
            let settled = false;
            const finish = () => {
                if (settled) return;
                settled = true;
                v.removeEventListener("seeked", finish);
                resolve();
            };
            v.addEventListener("seeked", finish);
            try {
                v.currentTime = target;
            } catch {
                finish();
                return;
            }
            if (Math.abs(v.currentTime - target) < 0.02 && v.readyState >= 2) {
                finish();
            } else {
                setTimeout(finish, 1500);
            }
        }));
        return this._seekChain;
    }

    updateStageVisibility() {
        if (!this.stageEl) return;
        const show = this.hasVideo()
            && !this.isImageBatch()
            && !this.isGenMode()
            && !this.isFl2vMode();
        this.stageEl.classList.toggle("hidden", !show);
        if (!show) {
            if (this.stageVideo) {
                this.stageVideo.pause();
                this.stageVideo.classList.add("hidden");
            }
            this.stageImg?.classList.add("hidden");
            this.stageEmpty?.classList.remove("hidden");
            this.stageBadge?.classList.add("hidden");
            this._stageClipIndex = -1;
        } else {
            this._syncStagePreview(this.currentFrame, { force: true });
        }
        this.updateDomWidgetHeight();
        syncDirectorNodeSize(this.node, this);
    }

    _updateStageBadge(logicalFrame) {
        if (!this.stageBadge) return;
        const total = this.getTotalFrames();
        const frame = clamp(logicalFrame | 0, 0, Math.max(0, total - 1));
        const clips = this.getVideoClips();
        const entry = this.getFrameMapEntry(frame);
        const clipHint = clips.length > 1 ? t("canvas.clipHint", { n: entry.clip + 1 }) : "";
        this.stageBadge.textContent = t("player.frameOf", { cur: frame + 1, total, clip: clipHint });
        this.stageBadge.classList.remove("hidden");
    }

    _logicalRangeForClip(clipIndex) {
        const map = this.getFrameMap();
        let start = -1;
        let end = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip !== clipIndex) {
                if (start >= 0) break;
                continue;
            }
            if (start < 0) start = i;
            end = i + 1;
        }
        if (start < 0) return { start: 0, end: this.getTotalFrames() };
        return { start, end };
    }

    _logicalFromStageTime(clipIndex, timeSec) {
        const fps = Math.max(0.001, this.getFrameRate());
        const srcFrame = Math.max(0, Math.round(Number(timeSec) * fps));
        const map = this.getFrameMap();
        if (!map.length) {
            const logical = sourceToLogicalFrame(srcFrame, this.timeline.video || {});
            if (logical < 0) return -1; // source lands in a deleted gap
            return clamp(logical, 0, Math.max(0, this.getTotalFrames() - 1));
        }
        let first = -1;
        let best = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip !== clipIndex) continue;
            if (first < 0) first = i;
            if (e.frame === srcFrame) return i;
            if (e.frame <= srcFrame) best = i;
        }
        if (best >= 0) return best;
        if (first >= 0) return first;
        return 0;
    }

    /** Next logical index whose source frame is strictly after srcFrame (same clip). */
    _nextLogicalAfterSourceFrame(clipIndex, srcFrame) {
        const map = this.getFrameMap();
        if (!map.length) {
            // Sparse: walk forward until source maps to a kept logical frame.
            const total = this.getTotalFrames();
            const startLogical = sourceToLogicalFrame(srcFrame, this.timeline.video || {});
            const from = startLogical < 0 ? 0 : startLogical;
            for (let i = from; i < total; i++) {
                if (this.logicalToSourceFrame(i) > srcFrame) return i;
            }
            return -1;
        }
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip === clipIndex && e.frame > srcFrame) return i;
        }
        return -1;
    }

    _syncStagePreview(logicalFrame, { force = false } = {}) {
        if (!this.stageEl || this.stageEl.classList.contains("hidden")) return;
        if (!this.hasVideo()) {
            this.stageEmpty?.classList.remove("hidden");
            this.stageVideo?.classList.add("hidden");
            this.stageImg?.classList.add("hidden");
            return;
        }

        // During native playback, do not seek every tick (that causes stutter).
        // Only refresh the badge; playhead is driven from video.currentTime.
        if (this.isPlaying && !force && !this._legacyFrames.length) {
            this._updateStageBadge(logicalFrame);
            return;
        }

        const frame = clamp(logicalFrame | 0, 0, Math.max(0, this.getTotalFrames() - 1));
        const fps = Math.max(0.001, this.getFrameRate());

        if (this._legacyFrames.length) {
            const dataUrl = this._legacyFrames[frame];
            if (this.stageVideo) {
                this.stageVideo.pause();
                this.stageVideo.classList.add("hidden");
            }
            if (this.stageImg && dataUrl) {
                this.stageImg.src = dataUrl;
                this.stageImg.classList.remove("hidden");
                this.stageEmpty?.classList.add("hidden");
            }
            this._updateStageBadge(frame);
            return;
        }

        const entry = this.getFrameMapEntry(frame);
        const url = this.getClipViewUrl(entry.clip);
        const v = this.stageVideo;
        if (!v || !url) {
            this.stageEmpty?.classList.remove("hidden");
            return;
        }

        this.stageImg?.classList.add("hidden");
        this.stageEmpty?.classList.add("hidden");
        v.classList.remove("hidden");

        let sameSrc = false;
        if (v.src && url) {
            try {
                sameSrc = new URL(v.src, location.href).href === new URL(url, location.href).href;
            } catch {
                sameSrc = v.src === url;
            }
        }
        // Must reload when the file changes even if clip index stays 0 (replace upload).
        if (this._stageClipIndex !== entry.clip || !sameSrc) {
            this._stageClipIndex = entry.clip;
            if (!sameSrc) {
                v.pause();
                v.src = url;
                v.load();
            }
        }

        const target = Math.max(0, entry.frame / fps);
        if (force || Math.abs(v.currentTime - target) > 0.035) {
            try {
                v.currentTime = target;
            } catch {
                /* ignore seek races while loading */
            }
        }
        if (this.isPlaying && force) {
            v.play().catch(() => {});
        }
        this._updateStageBadge(frame);
    }

    async _ensureStageReadyForFrame(logicalFrame) {
        this._syncStagePreview(logicalFrame, { force: true });
        const v = this.stageVideo;
        if (!v || this._legacyFrames.length) return false;
        if (v.readyState >= 2) return true;
        await new Promise((resolve) => {
            const done = () => {
                v.removeEventListener("loadeddata", done);
                v.removeEventListener("canplay", done);
                resolve();
            };
            v.addEventListener("loadeddata", done);
            v.addEventListener("canplay", done);
            setTimeout(done, 800);
        });
        return true;
    }

    _queueThumbPrefetch(logicalFrame) {
        if (this.isPlaying) return;
        if (!this._usesSourceVideoThumbs()) return;
        const cacheKey = this._frameThumbKey(logicalFrame);
        if (this._thumbCache.has(cacheKey) || this._thumbPending.has(cacheKey)) return;
        if (!this.hasVideo() && !this._legacyFrames.length) return;
        this._thumbPending.add(cacheKey);
        this._fetchThumb(logicalFrame).then((img) => {
            this._thumbPending.delete(cacheKey);
            if (!img) return;
            if (this._frameThumbKey(logicalFrame) !== cacheKey) return;
            this._thumbCache.set(cacheKey, img);
            this.scheduleRender();
        }).catch(() => {
            this._thumbPending.delete(cacheKey);
        });
    }

    /** Capture a still from an r2v reference video for the timeline strip. */
    _queueR2vVideoThumb(cacheKey, videoFile, type = "input") {
        if (!cacheKey || !videoFile) return;
        if (this._thumbCache.has(cacheKey) || this._thumbPending.has(cacheKey)) return;
        this._thumbPending.add(cacheKey);
        const url = inputViewUrl(videoFile, type || "input");
        const v = document.createElement("video");
        v.muted = true;
        v.playsInline = true;
        v.preload = "auto";
        v.crossOrigin = "anonymous";
        let done = false;
        const finish = (img) => {
            if (done) return;
            done = true;
            this._thumbPending.delete(cacheKey);
            try {
                v.removeAttribute("src");
                v.load();
            } catch (_) { /* ignore */ }
            if (img) this._thumbCache.set(cacheKey, img);
            this.scheduleRender();
        };
        const capture = () => {
            try {
                if (!v.videoWidth) {
                    finish(null);
                    return;
                }
                if (!this._thumbCanvas) {
                    this._thumbCanvas = document.createElement("canvas");
                    this._thumbCtx = this._thumbCanvas.getContext("2d", { alpha: false });
                }
                const ratio = v.videoWidth > THUMB_MAX_W ? THUMB_MAX_W / v.videoWidth : 1;
                const tw = Math.max(1, Math.round(v.videoWidth * ratio));
                const th = Math.max(1, Math.round(v.videoHeight * ratio));
                this._thumbCanvas.width = tw;
                this._thumbCanvas.height = th;
                this._thumbCtx.drawImage(v, 0, 0, tw, th);
                const img = new Image();
                img.onload = () => finish(img);
                img.onerror = () => finish(null);
                img.src = this._thumbCanvas.toDataURL("image/jpeg", THUMB_JPEG_Q);
            } catch (_) {
                finish(null);
            }
        };
        v.addEventListener("loadeddata", () => {
            const seekTo = Math.min(0.15, Math.max(0, (v.duration || 1) * 0.05));
            const onSeeked = () => {
                v.removeEventListener("seeked", onSeeked);
                capture();
            };
            v.addEventListener("seeked", onSeeked);
            try {
                v.currentTime = seekTo;
            } catch (_) {
                capture();
            }
            setTimeout(() => {
                if (!done) capture();
            }, 700);
        }, { once: true });
        v.onerror = () => finish(null);
        v.src = url;
    }

    async _fetchThumb(logicalFrame) {
        if (this._legacyFrames.length) {
            const dataUrl = this._legacyFrames[logicalFrame];
            if (!dataUrl) return null;
            return this._decodeThumb(dataUrl);
        }
        const entry = this.getFrameMapEntry(logicalFrame);
        const v = await this._ensurePreviewReady(entry.clip);
        if (!v?.videoWidth) return null;
        const t = Math.max(0, entry.frame / this.getFrameRate());
        await this._seekPreviewVideo(t, entry.clip);
        if (!v.videoWidth) return null;
        try {
            const ratio = v.videoWidth > THUMB_MAX_W ? THUMB_MAX_W / v.videoWidth : 1;
            const tw = Math.max(1, Math.round(v.videoWidth * ratio));
            const th = Math.max(1, Math.round(v.videoHeight * ratio));
            if (!this._thumbCanvas) {
                this._thumbCanvas = document.createElement("canvas");
                this._thumbCtx = this._thumbCanvas.getContext("2d", { alpha: false });
            }
            this._thumbCanvas.width = tw;
            this._thumbCanvas.height = th;
            this._thumbCtx.drawImage(v, 0, 0, tw, th);
            const dataUrl = this._thumbCanvas.toDataURL("image/jpeg", THUMB_JPEG_Q);
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => resolve(null);
                img.src = dataUrl;
            });
        } catch {
            return null;
        }
    }

    _clearVideoState({ dropUnusedThumbs = false } = {}) {
        const oldIds = dropUnusedThumbs ? this._liveVideoFileIdentities() : [];
        this._legacyFrames = [];
        this.timeline.videoClips = [];
        this.timeline.videoWorkspace = null;
        // Wipe video identity BEFORE visibility sync — otherwise hasVideo() stays
        // true via the old videoFile and stage reloads the previous clip.
        this.timeline.video = {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
            deletedSourceRanges: [],
            sourceFrameCount: 0,
            width: 0,
            height: 0,
        };
        this.timeline.totalFrames = 0;
        this._storageWidth = 0;
        this._storageHeight = 0;
        this._clearPreviewVideos(true);
        if (this._previewVideo) {
            this._previewVideo.pause();
            this._previewVideo.removeAttribute("src");
            this._previewVideo.load();
        }
        if (this.stageVideo) {
            this.stageVideo.pause();
            this.stageVideo.removeAttribute("src");
            this.stageVideo.load();
            this.stageVideo.classList.add("hidden");
        }
        this.stageImg?.classList.add("hidden");
        if (this.stageImg) this.stageImg.removeAttribute("src");
        this.stageEmpty?.classList.remove("hidden");
        this.stageBadge?.classList.add("hidden");
        this._stageClipIndex = -1;
        if (dropUnusedThumbs) this._dropThumbsIfUnused(oldIds);
        this.updateStageVisibility();
    }

    _resetTimelineForReplaceUpload() {
        const ids = this._liveVideoFileIdentities();
        if (ids.length) this._thumbIdsPendingDrop = ids;
        this._clearVideoState();
        this.timeline.segments = [];
        this.selectedIndex = 0;
        this.currentFrame = 0;
        if (this.seekBar) {
            this.seekBar.value = 0;
            this.seekBar.max = 0;
        }
    }

    _setSingleSegment(totalFrames) {
        const total = Math.max(0, totalFrames);
        this.timeline.segments = total > 0
            ? [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }]
            : [];
        this.selectedIndex = 0;
        this.currentFrame = 0;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = 0;
        }
    }

    restoreVideoFromTimeline() {
        const video = this.timeline.video || {};
        this._storageWidth = video.storageWidth || 0;
        this._storageHeight = video.storageHeight || 0;

        const legacy = video.frames || [];
        if (legacy.length && !video.videoFile) {
            this._legacyFrames = legacy;
            this.setFrameMap(buildIdentityFrameMap(legacy.length));
            this.videoNameEl.textContent = t("videoName.legacy", {
                name: video.fileName || t("videoName.defaultVideo"),
                frames: legacy.length,
            });
            this._prefetchSegmentThumbs(0, legacy.length);
            this.updateStageVisibility();
            return;
        }

        if (!video.videoFile) {
            this._clearVideoState();
            return;
        }

        this._restorePreviewVideos();
        const n = this.getTotalFrames();
        this._prefetchSegmentThumbs(0, Math.min(n, THUMB_PREFETCH_BATCH * 4));
        this.updateVideoNameLabel();
        if (taskUsesReferenceVideo(this.getTaskKey()) && this.getReferenceVideoViewUrl(this.timeline.global?.referenceVideo)) {
            this.renderRefVideoSlot();
        }
        this.updateStageVisibility();
    }

    _prefetchSegmentThumbs(from, to) {
        if (!this._usesSourceVideoThumbs()) return;
        for (let f = from; f < to; f++) this._queueThumbPrefetch(f);
    }

    _decodeThumb(dataUrl) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                if (!img.naturalWidth || img.naturalWidth <= THUMB_MAX_W) {
                    resolve(img);
                    return;
                }
                const ratio = THUMB_MAX_W / img.naturalWidth;
                const w = THUMB_MAX_W;
                const h = Math.max(1, Math.round(img.naturalHeight * ratio));
                const c = document.createElement("canvas");
                c.width = w;
                c.height = h;
                c.getContext("2d").drawImage(img, 0, 0, w, h);
                const thumb = new Image();
                thumb.onload = () => resolve(thumb);
                thumb.onerror = () => resolve(img);
                thumb.src = c.toDataURL("image/jpeg", THUMB_JPEG_Q);
            };
            img.onerror = () => resolve(null);
            img.src = dataUrl.startsWith("data:") ? dataUrl : `data:image/jpeg;base64,${dataUrl}`;
        });
    }

    pickVideoFile() {
        if (this.isFl2vMode()) {
            openFl2vUpload(this);
            return;
        }
        const input = document.createElement("input");
        input.type = "file"; input.accept = "video/*";
        input.onchange = () => { if (input.files?.[0]) this.loadVideoFile(input.files[0]); };
        input.click();
    }

    async pickExistingVideoFile() {
        if (this.isFl2vMode()) return;
        try {
            const picked = await this.chooseVideoInput({
                title: t("mediaPicker.pickVideo"),
                currentValue: this.timeline.video?.videoFile || "",
            });
            if (!picked?.relPath) return;
            const btn = this.root.querySelector('[data-a="video-existing"]');
            if (btn) { btn.disabled = true; btn.textContent = t("common.analyzing"); }
            this.videoNameEl.textContent = t("upload.inProgress", { name: picked.fileName || picked.relPath });
            try {
                await this._applyLoadedVideo({
                    fileName: picked.fileName || picked.relPath,
                    relPath: picked.relPath,
                    subfolder: picked.subfolder || "",
                    type: picked.type || "input",
                    statusPrefix: t("parse.prefix"),
                });
            } catch (err) {
                console.error("[MiniMax H3Director] video load failed:", err);
                this.videoNameEl.textContent = t("upload.loadFailed", { err: formatUploadError(err) });
                this.updateVideoNameLabel();
                this._flushPendingThumbDrops();
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = t("mediaPicker.pickExistingVideo");
                }
            }
        } catch (err) {
            console.error("[MiniMax H3Director] video pick failed:", err);
        }
    }

    pickAppendVideoFile() {
        if (!this.hasVideo()) {
            this.showBdMessage(
                t("dialog.appendVideoTitle"),
                t("dialog.appendVideoNeedFirst")
            );
            return;
        }
        const input = document.createElement("input");
        input.type = "file"; input.accept = "video/*";
        input.onchange = () => { if (input.files?.[0]) this.appendVideoFile(input.files[0]); };
        input.click();
    }

    async appendVideoFile(file) {
        const btn = this.root.querySelector('[data-a="video-append"]');
        if (btn) { btn.disabled = true; btn.textContent = t("common.uploading"); }
        this.videoNameEl.textContent = t("upload.appendProgress", { name: file.name });
        try {
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > COMFY_UPLOAD_SOFT_LIMIT ? t("upload.chunkMode") : t("upload.mode");
                this.videoNameEl.textContent = t("upload.appendChunk", {
                    mode, name: file.name, cur, total, pct,
                });
            });
            const relPath = videoRelativePath(uploaded);
            await this._applyAppendedVideo({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: t("parse.prefix"),
            });
        } catch (err) {
            console.error("[MiniMax H3Director] append video failed:", err);
            this.videoNameEl.textContent = t("upload.appendFailed", { err: formatUploadError(err) });
            this.updateVideoNameLabel();
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = t("toolbar.appendVideo");
            }
        }
    }

    async loadVideoFile(file) {
        const btn = this.root.querySelector('[data-a="video"]');
        if (btn) { btn.disabled = true; btn.textContent = t("common.uploading"); }
        this.videoNameEl.textContent = t("upload.inProgress", { name: file.name });
        try {
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > COMFY_UPLOAD_SOFT_LIMIT ? t("upload.chunkMode") : t("upload.mode");
                this.videoNameEl.textContent = t("upload.loadChunk", {
                    mode, name: file.name, cur, total, pct,
                });
            });
            const relPath = videoRelativePath(uploaded);
            await this._applyLoadedVideo({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: t("parse.prefix"),
            });
        } catch (err) {
            console.error("[MiniMax H3Director] video load failed:", err);
            this.videoNameEl.textContent = t("upload.loadFailed", { err: formatUploadError(err) });
            this.updateVideoNameLabel();
            this._flushPendingThumbDrops();
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = t("toolbar.uploadVideo");
            }
        }
    }

    _closeBdModal() {
        if (this._modalKeyHandler) {
            window.removeEventListener("keydown", this._modalKeyHandler, true);
            this._modalKeyHandler = null;
        }
        if (this._modalEl) {
            this._modalEl.remove();
            this._modalEl = null;
        }
    }

    showBdMessage(title, message) {
        return this.showBdDialog({ title, message, confirmText: t("dialog.confirm"), cancelText: null });
    }

    showBdDialog(opts = {}) {
        const { title, message, items } = opts;
        const confirmText = opts.confirmText ?? t("dialog.confirm");
        const cancelText = Object.prototype.hasOwnProperty.call(opts, "cancelText")
            ? opts.cancelText
            : t("dialog.cancel");
        return new Promise((resolve) => {
            this._closeBdModal();

            const overlay = document.createElement("div");
            overlay.className = "bd-modal-overlay";
            const panel = document.createElement("div");
            panel.className = "bd-modal";
            panel.innerHTML = `
                <div class="bd-modal-title"></div>
                <div class="bd-modal-body hidden"></div>
                <div class="bd-modal-list hidden"></div>
                <div class="bd-modal-actions"></div>`;

            panel.querySelector(".bd-modal-title").textContent = title || "";

            const bodyEl = panel.querySelector(".bd-modal-body");
            const listEl = panel.querySelector(".bd-modal-list");
            const actionsEl = panel.querySelector(".bd-modal-actions");

            let selectedValue = items?.length ? items[0].value : null;

            const finish = (val) => {
                this._closeBdModal();
                resolve(val);
            };

            if (message) {
                bodyEl.textContent = message;
                bodyEl.classList.remove("hidden");
            }

            if (items?.length) {
                listEl.classList.remove("hidden");
                for (const item of items) {
                    const row = document.createElement("div");
                    row.className = "bd-modal-item";
                    row.textContent = item.label ?? item.value;
                    row.title = item.label ?? item.value;
                    row.dataset.value = item.value;
                    if (item.value === selectedValue) row.classList.add("selected");
                    row.onclick = () => {
                        selectedValue = item.value;
                        for (const el of listEl.querySelectorAll(".bd-modal-item")) {
                            el.classList.toggle("selected", el === row);
                        }
                    };
                    row.ondblclick = () => finish(item.value);
                    listEl.appendChild(row);
                }
            }

            if (cancelText) {
                const cancelBtn = document.createElement("button");
                cancelBtn.type = "button";
                cancelBtn.className = "bd-btn";
                cancelBtn.textContent = cancelText;
                cancelBtn.onclick = () => finish(null);
                actionsEl.appendChild(cancelBtn);
            }

            const okBtn = document.createElement("button");
            okBtn.type = "button";
            okBtn.className = "bd-btn bd-btn-primary";
            okBtn.textContent = confirmText;
            okBtn.onclick = () => finish(items?.length ? selectedValue : true);
            actionsEl.appendChild(okBtn);

            overlay.onclick = (e) => {
                if (e.target === overlay && cancelText) finish(null);
            };
            panel.onclick = (e) => e.stopPropagation();

            this._modalKeyHandler = (e) => {
                if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    finish(cancelText ? null : true);
                } else if (e.key === "Enter" && items?.length) {
                    e.preventDefault();
                    finish(selectedValue);
                }
            };
            window.addEventListener("keydown", this._modalKeyHandler, true);

            overlay.appendChild(panel);
            this.root.appendChild(overlay);
            this._modalEl = overlay;
            okBtn.focus();
        });
    }

    pickLocalFile(accept = "") {
        return new Promise((resolve) => {
            const input = document.createElement("input");
            input.type = "file";
            if (accept) input.accept = accept;
            input.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0;pointer-events:none";
            const cleanup = () => input.remove();
            input.onchange = () => {
                const file = input.files?.[0] || null;
                cleanup();
                resolve(file);
            };
            input.addEventListener("cancel", () => {
                cleanup();
                resolve(null);
            }, { once: true });
            document.body.appendChild(input);
            input.click();
        });
    }

    async listInputMedia(kind) {
        const resp = await api.fetchApi(`/minimax/director/list_input_media?kind=${encodeURIComponent(kind)}`);
        if (!resp.ok) {
            const text = (await resp.text()).trim();
            if (resp.status === 404) throw new Error(t("mediaPicker.needRestart"));
            throw new Error(text || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        return Array.isArray(data?.items) ? data.items : [];
    }

    probeInputImageDimensions(relPath, type = "input") {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve({
                width: img.naturalWidth || img.width || 0,
                height: img.naturalHeight || img.height || 0,
            });
            img.onerror = () => resolve({ width: 0, height: 0 });
            img.src = inputViewUrl(relPath, type || "input");
        });
    }

    showInputMediaPicker({ kind, title, accept, currentValue = "" } = {}) {
        return new Promise((resolve) => {
            this._closeBdModal();

            const overlay = document.createElement("div");
            overlay.className = "bd-modal-overlay";
            const panel = document.createElement("div");
            panel.className = "bd-modal bd-media-modal";
            panel.innerHTML = `
                <div class="bd-media-head">
                    <div class="bd-modal-title"></div>
                    <div class="bd-modal-actions bd-media-head-actions"></div>
                </div>
                <div class="bd-media-status"></div>
                <div class="bd-media-body">
                    <div class="bd-media-left">
                        <div class="bd-media-table" tabindex="0">
                            <div class="bd-media-thead">
                                <button type="button" class="bd-media-th" data-sort="name">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                                <button type="button" class="bd-media-th" data-sort="dims">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                                <button type="button" class="bd-media-th" data-sort="time">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                            </div>
                            <div class="bd-media-tbody"></div>
                        </div>
                    </div>
                    <div class="bd-media-right">
                        <div class="bd-media-preview">
                            <div class="bd-media-preview-empty"></div>
                        </div>
                        <div class="bd-media-meta"></div>
                    </div>
                </div>`;

            panel.querySelector(".bd-modal-title").textContent = title || "";
            const statusEl = panel.querySelector(".bd-media-status");
            const actionsTop = panel.querySelector(".bd-media-head-actions");
            const tableEl = panel.querySelector(".bd-media-table");
            const tbodyEl = panel.querySelector(".bd-media-tbody");
            const previewEl = panel.querySelector(".bd-media-preview");
            const previewEmptyEl = panel.querySelector(".bd-media-preview-empty");
            const metaEl = panel.querySelector(".bd-media-meta");
            const showDims = kind !== "audio" && kind !== "reference_audio";
            if (!showDims) {
                tableEl.classList.add("bd-media-nodims");
                tableEl.querySelector('.bd-media-th[data-sort="dims"]')?.remove();
            }
            const thEls = [...panel.querySelectorAll(".bd-media-th")];
            thEls.forEach((th) => {
                const key = th.dataset.sort;
                const label = key === "dims" ? t("mediaPicker.dims")
                    : key === "time" ? t("mediaPicker.time")
                    : t("mediaPicker.file");
                th.querySelector("span").textContent = label;
            });

            let selectedValue = currentValue || "";
            let itemsByPath = new Map();
            let listedItems = [];
            let sortKey = "time";
            let sortDir = "desc";

            const finish = (val) => {
                this._closeBdModal();
                resolve(val);
            };

            const selectedChoice = () => {
                const item = itemsByPath.get(selectedValue || "");
                if (!item) return null;
                return {
                    source: "existing",
                    relPath: item.relPath,
                    fileName: item.fileName || item.name || item.relPath,
                    subfolder: item.subfolder || "",
                    type: item.type || "input",
                    mediaKind: item.mediaKind || kind,
                };
            };

            const formatMediaTime = (unixSec) => {
                const n = Number(unixSec);
                if (!Number.isFinite(n) || n <= 0) return "—";
                const d = new Date(n * 1000);
                if (Number.isNaN(d.getTime())) return "—";
                const pad = (v) => String(v).padStart(2, "0");
                return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
            };

            const dimsText = (item) => {
                const w = Number(item?.width) || 0;
                const h = Number(item?.height) || 0;
                return w > 0 && h > 0 ? `${w}x${h}` : "—";
            };

            const dimScore = (item) => {
                const w = Number(item?.width) || 0;
                const h = Number(item?.height) || 0;
                return w > 0 && h > 0 ? w * 100000 + h : -1;
            };

            const sortedItems = () => {
                const copy = listedItems.slice();
                copy.sort((a, b) => {
                    if (sortKey === "name") {
                        const av = (a.fileName || a.name || a.relPath || "").toLowerCase();
                        const bv = (b.fileName || b.name || b.relPath || "").toLowerCase();
                        const c = av.localeCompare(bv, undefined, { numeric: true, sensitivity: "base" });
                        if (c) return sortDir === "asc" ? c : -c;
                    } else if (sortKey === "dims") {
                        const as = dimScore(a);
                        const bs = dimScore(b);
                        const aMiss = as < 0;
                        const bMiss = bs < 0;
                        if (aMiss !== bMiss) return aMiss ? 1 : -1;
                        if (as !== bs) return sortDir === "asc" ? as - bs : bs - as;
                    } else {
                        const av = Number(a.modified) || 0;
                        const bv = Number(b.modified) || 0;
                        if (av !== bv) return sortDir === "asc" ? av - bv : bv - av;
                    }
                    return (a.relPath || "").localeCompare(b.relPath || "");
                });
                return copy;
            };

            const syncHeaderState = () => {
                thEls.forEach((th) => {
                    const active = th.dataset.sort === sortKey;
                    th.classList.toggle("is-active", active);
                    th.classList.toggle("is-asc", active && sortDir === "asc");
                });
            };

            const renderPreview = (item) => {
                previewEl.innerHTML = "";
                metaEl.innerHTML = "";
                if (!item?.relPath) {
                    previewEmptyEl.textContent = t("mediaPicker.previewEmpty");
                    previewEl.appendChild(previewEmptyEl);
                    return;
                }
                const relPath = item.relPath;
                const type = item.type || "input";
                const previewKind = kind === "reference_audio" ? item.mediaKind : kind;
                if (previewKind === "image") {
                    const img = document.createElement("img");
                    img.src = inputViewUrl(relPath, type);
                    img.alt = item.fileName || item.name || relPath;
                    previewEl.appendChild(img);
                } else if (previewKind === "audio") {
                    const audio = document.createElement("audio");
                    audio.src = inputViewUrl(relPath, type);
                    audio.controls = true;
                    audio.preload = "metadata";
                    previewEl.appendChild(audio);
                } else {
                    const video = document.createElement("video");
                    video.src = inputViewUrl(relPath, type);
                    video.controls = true;
                    video.preload = "metadata";
                    video.muted = true;
                    video.playsInline = true;
                    previewEl.appendChild(video);
                }
                const fileEl = document.createElement("div");
                fileEl.textContent = `${t("mediaPicker.file")}: ${item.fileName || item.name || relPath}`;
                metaEl.appendChild(fileEl);
                const pathEl = document.createElement("div");
                pathEl.textContent = `${t("mediaPicker.path")}: ${relPath}`;
                metaEl.appendChild(pathEl);
            };

            const selectRow = (relPath, { scroll = false } = {}) => {
                selectedValue = relPath || "";
                tbodyEl.querySelectorAll(".bd-media-tr").forEach((row) => {
                    const on = row.dataset.path === selectedValue;
                    row.classList.toggle("selected", on);
                    if (on && scroll) row.scrollIntoView({ block: "nearest" });
                });
                renderPreview(itemsByPath.get(selectedValue));
            };

            const renderRows = () => {
                tbodyEl.innerHTML = "";
                const rows = sortedItems();
                if (!rows.length) {
                    const empty = document.createElement("div");
                    empty.className = "bd-media-empty-row";
                    empty.textContent = t("mediaPicker.empty");
                    tbodyEl.appendChild(empty);
                    selectedValue = "";
                    renderPreview(null);
                    syncHeaderState();
                    return;
                }
                if (!selectedValue || !itemsByPath.has(selectedValue)) {
                    selectedValue = rows[0].relPath || "";
                }
                for (const item of rows) {
                    const row = document.createElement("div");
                    row.className = "bd-media-tr";
                    row.dataset.path = item.relPath;
                    if (item.relPath === selectedValue) row.classList.add("selected");
                    const nameTd = document.createElement("div");
                    nameTd.className = "bd-media-td bd-media-td-name";
                    const mediaPrefix = kind === "reference_audio"
                        ? (item.mediaKind === "video" ? "🎞 " : "♪ ")
                        : "";
                    nameTd.textContent = `${mediaPrefix}${item.relPath || item.fileName || item.name || ""}`;
                    nameTd.title = nameTd.textContent;
                    const timeTd = document.createElement("div");
                    timeTd.className = "bd-media-td bd-media-td-time";
                    timeTd.textContent = formatMediaTime(item.modified);
                    if (showDims) {
                        const dimsTd = document.createElement("div");
                        dimsTd.className = "bd-media-td bd-media-td-dims";
                        dimsTd.textContent = dimsText(item);
                        row.append(nameTd, dimsTd, timeTd);
                    } else {
                        row.append(nameTd, timeTd);
                    }
                    row.addEventListener("click", () => selectRow(item.relPath));
                    row.addEventListener("dblclick", () => {
                        const choice = selectedChoice();
                        if (choice) finish(choice);
                    });
                    tbodyEl.appendChild(row);
                }
                syncHeaderState();
                renderPreview(itemsByPath.get(selectedValue));
                const selectedRow = tbodyEl.querySelector(".bd-media-tr.selected");
                selectedRow?.scrollIntoView({ block: "nearest" });
            };

            const moveSelection = (delta) => {
                const rows = sortedItems();
                if (!rows.length) return;
                const idx = rows.findIndex((item) => item.relPath === selectedValue);
                const next = rows[Math.max(0, Math.min(rows.length - 1, (idx < 0 ? 0 : idx) + delta))];
                if (next) selectRow(next.relPath, { scroll: true });
            };

            const loadItems = async () => {
                statusEl.textContent = t("mediaPicker.loading");
                tbodyEl.innerHTML = "";
                renderPreview(null);
                try {
                    listedItems = await this.listInputMedia(kind);
                    itemsByPath = new Map(listedItems.map((item) => [item.relPath, item]));
                    renderRows();
                    statusEl.textContent = listedItems.length
                        ? t("mediaPicker.count", { n: listedItems.length })
                        : t("mediaPicker.empty");
                } catch (err) {
                    listedItems = [];
                    itemsByPath = new Map();
                    tbodyEl.innerHTML = "";
                    statusEl.textContent = err?.message || String(err);
                }
            };

            thEls.forEach((th) => {
                th.addEventListener("click", () => {
                    const key = th.dataset.sort || "time";
                    if (sortKey === key) {
                        sortDir = sortDir === "asc" ? "desc" : "asc";
                    } else {
                        sortKey = key;
                        sortDir = key === "name" ? "asc" : "desc";
                    }
                    renderRows();
                });
            });

            const refreshBtn = document.createElement("button");
            refreshBtn.type = "button";
            refreshBtn.className = "bd-btn";
            refreshBtn.textContent = t("mediaPicker.refresh");
            refreshBtn.onclick = () => { void loadItems(); };
            actionsTop.appendChild(refreshBtn);

            const uploadBtn = document.createElement("button");
            uploadBtn.type = "button";
            uploadBtn.className = "bd-btn";
            uploadBtn.textContent = t("mediaPicker.upload");
            uploadBtn.onclick = async () => {
                const file = await this.pickLocalFile(accept || "");
                if (file) finish({ source: "file", file });
            };
            actionsTop.appendChild(uploadBtn);

            const actionsBottom = document.createElement("div");
            actionsBottom.className = "bd-modal-actions";
            const cancelBtn = document.createElement("button");
            cancelBtn.type = "button";
            cancelBtn.className = "bd-btn";
            cancelBtn.textContent = t("dialog.cancel");
            cancelBtn.onclick = () => finish(null);
            actionsBottom.appendChild(cancelBtn);

            const okBtn = document.createElement("button");
            okBtn.type = "button";
            okBtn.className = "bd-btn bd-btn-primary";
            okBtn.textContent = t("mediaPicker.useSelected");
            okBtn.onclick = () => {
                const choice = selectedChoice();
                if (choice) finish(choice);
            };
            actionsBottom.appendChild(okBtn);
            panel.appendChild(actionsBottom);

            overlay.onclick = (e) => {
                if (e.target === overlay) finish(null);
            };
            panel.onclick = (e) => e.stopPropagation();

            this._modalKeyHandler = (e) => {
                if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    finish(null);
                    return;
                }
                if (e.key === "ArrowDown" || e.key === "ArrowUp") {
                    if (!panel.contains(e.target) && e.target !== document.body) return;
                    e.preventDefault();
                    moveSelection(e.key === "ArrowDown" ? 1 : -1);
                    return;
                }
                if (e.key !== "Enter") return;
                if (e.target?.closest?.(".bd-media-th, .bd-media-head-actions, button.bd-btn:not(.bd-btn-primary)")) return;
                e.preventDefault();
                okBtn.click();
            };
            window.addEventListener("keydown", this._modalKeyHandler, true);

            overlay.appendChild(panel);
            this.root.appendChild(overlay);
            this._modalEl = overlay;
            void loadItems();
            tableEl.focus();
        });
    }

    async chooseImageInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "image",
            title: opts.title || t("mediaPicker.pickImage"),
            accept: "image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff",
            currentValue: opts.currentValue || "",
        });
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            const uploaded = await uploadToInput(choice.file);
            const relPath = videoRelativePath(uploaded);
            const dims = await this.probeInputImageDimensions(relPath, uploaded.type || "input");
            return {
                imageFile: relPath,
                fileName: uploaded?.name || choice.file.name || relPath,
                subfolder: uploaded?.subfolder || "",
                type: uploaded?.type || "input",
                width: dims.width || 0,
                height: dims.height || 0,
            };
        }
        const dims = await this.probeInputImageDimensions(choice.relPath, choice.type || "input");
        return {
            imageFile: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
            width: dims.width || 0,
            height: dims.height || 0,
        };
    }

    async chooseVideoInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "video",
            title: opts.title || t("mediaPicker.pickVideo"),
            accept: "video/*,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
        });
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            const uploaded = await uploadToInputSmart(choice.file);
            return {
                relPath: videoRelativePath(uploaded),
                fileName: uploaded?.name || choice.file.name || "",
                subfolder: uploaded?.subfolder || "",
                type: uploaded?.type || "input",
            };
        }
        return {
            relPath: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
        };
    }

    async chooseAudioInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "reference_audio",
            title: opts.title || t("mediaPicker.pickAudio"),
            accept: "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
        });
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            return prepareLocalReferenceAudio(choice.file);
        }
        if (choice.mediaKind === "video") {
            return extractReferenceAudioFromExistingVideo(choice);
        }
        return {
            relPath: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
        };
    }

    async _prepareVideoFrames({ fileName, relPath, subfolder, type, statusPrefix, syncNativeFps = true }) {
        this.videoNameEl.textContent = `${statusPrefix}: ${fileName}…`;
        const viewUrl = inputViewUrl(relPath, type || "input");

        let serverProbe = null;
        try {
            serverProbe = await this.probeVideoFile(relPath, subfolder, type);
        } catch (err) {
            console.warn("[MiniMax H3Director] video probe failed, using browser estimate:", err);
        }
        const browserMeta = await this.probeVideoMetadata(viewUrl);
        const nativeFps = Number(serverProbe?.native_fps || 0);
        const nativeFrameCount = Number(serverProbe?.frame_count || 0);
        const meta = {
            width: Number(serverProbe?.width || browserMeta.width || 0),
            height: Number(serverProbe?.height || browserMeta.height || 0),
            duration: Number(serverProbe?.duration ?? browserMeta.duration ?? 0),
            nativeFps,
            nativeFrameCount,
            probeMethod: serverProbe?.probe_method || "browser_estimate",
        };

        if (syncNativeFps && nativeFps > 0) {
            this.syncFrameRateUI(nativeFps);
        }

        const fps = this.getFrameRate();
        const totalFrames = Math.max(
            1,
            Math.round(meta.duration * fps) || nativeFrameCount,
        );

        const store = resolveOutputDimensions(meta.width, meta.height, this.timeline.output || { mode: "long_edge", longEdge: 864 }, {
            refMaxSize: this.refMaxWidget?.value,
        });

        return { fileName, relPath, subfolder, type, meta, totalFrames, store, viewUrl };
    }

    async probeVideoFile(relPath, subfolder = "", type = "input") {
        const resp = await api.fetchApi("/minimax/director/probe_video", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ videoFile: relPath, subfolder, type: type || "input" }),
        });
        if (!resp.ok) {
            throw new Error(await resp.text());
        }
        return resp.json();
    }

    _buildClipRecord({ fileName, relPath, subfolder, type, meta, totalFrames, store }) {
        return {
            id: uid(),
            fileName,
            videoFile: relPath,
            subfolder: subfolder || "",
            type: type || "input",
            width: meta.width,
            height: meta.height,
            duration: meta.duration,
            nativeFps: meta.nativeFps || null,
            nativeFrameCount: meta.nativeFrameCount || null,
            sourceFrameCount: totalFrames,
            storageWidth: store.width,
            storageHeight: store.height,
        };
    }

    _syncPrimaryVideoFromClips(frameMap) {
        const clips = this.getVideoClips();
        const primary = clips[0] || {};
        const prev = this.timeline.video || {};
        const map = Array.isArray(frameMap) ? frameMap : (prev.frameMap || []);
        this.timeline.video = {
            ...prev,
            ...primary,
            // Keep path/type from the clip record, but never drop timeline edits.
            fileName: primary.fileName || prev.fileName || "",
            videoFile: primary.videoFile || prev.videoFile || "",
            subfolder: primary.subfolder ?? prev.subfolder ?? "",
            type: primary.type || prev.type || "input",
            frames: prev.frames || [],
            frameMap: map,
            // Explicit map already encodes deletes; sparse mode keeps ranges.
            deletedSourceRanges: map.length ? [] : (prev.deletedSourceRanges || []),
            sourceFrameCount: prev.sourceFrameCount || primary.sourceFrameCount || map.length || 0,
        };
        if (map.length) this.timeline.totalFrames = map.length;
    }

    async _applyLoadedVideo({ fileName, relPath, subfolder, type, statusPrefix }) {
        const prep = await this._prepareVideoFrames({ fileName, relPath, subfolder, type, statusPrefix });
        const { totalFrames, store, viewUrl } = prep;

        this._storageWidth = store.width;
        this._storageHeight = store.height;
        const clip = this._buildClipRecord(prep);

        this.timeline.videoClips = [clip];
        this.setSparseVideoFrames(totalFrames);
        this._syncPrimaryVideoFromClips([]);
        this._setSingleSegment(totalFrames);

        this._clearPreviewVideos(true);
        this._previewVideo = this._getPreviewVideoForClip(0);
        if (this._previewVideo && viewUrl) this._previewVideo.src = viewUrl;

        // Force stage to drop any previous media before binding the new clip.
        this._stageClipIndex = -1;
        if (this.stageVideo) {
            this.stageVideo.pause();
            this.stageVideo.removeAttribute("src");
            this.stageVideo.load();
        }
        this.currentFrame = 0;

        if (this.totalFramesWidget) this.totalFramesWidget.value = totalFrames;
        this.syncOutputUIFromTimeline();
        this.updateVideoNameLabel();
        this._flushPendingThumbDrops();
        this._prefetchSegmentThumbs(0, Math.min(totalFrames, THUMB_PREFETCH_BATCH * 4));
        this.updateStageVisibility();
        this._syncStagePreview(this.currentFrame, { force: true });
        this.commit(false, { syncTimeline: true });
    }

    async _applyAppendedVideo({ fileName, relPath, subfolder, type, statusPrefix }) {
        const prep = await this._prepareVideoFrames({
            fileName, relPath, subfolder, type, statusPrefix,
            syncNativeFps: false,
        });
        const { totalFrames, store } = prep;

        this._ensureVideoClipsArray();
        const clipIndex = this.timeline.videoClips.length;
        const clip = this._buildClipRecord(prep);
        this.timeline.videoClips.push(clip);

        const prevTotal = this.getTotalFrames();
        if (!this.getFrameMap().length && prevTotal > 0) {
            this.materializeFrameMap();
        }
        const newEntries = buildClipFrameMap(clipIndex, totalFrames);
        const map = [...this.getFrameMap(), ...newEntries];
        this.setFrameMap(map);
        this.timeline.totalFrames = map.length;
        this._syncPrimaryVideoFromClips(map);

        this._getPreviewVideoForClip(clipIndex);

        this.timeline.segments.push({
            id: uid(),
            start: prevTotal,
            length: totalFrames,
            prompt: "",
            taskType: "",
            refs: [],
            referenceVideo: {},
            videoClipId: clip.id,
        });

        if (this.totalFramesWidget) this.totalFramesWidget.value = map.length;
        this.selectedIndex = this.timeline.segments.length - 1;
        this.currentFrame = prevTotal;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, map.length - 1);
            this.seekBar.value = this.currentFrame;
        }

        this.normalizeSegments();
        this.syncOutputUIFromTimeline();
        this.updateVideoNameLabel();
        this._prefetchSegmentThumbs(prevTotal, Math.min(prevTotal + totalFrames, prevTotal + THUMB_PREFETCH_BATCH * 4));
        this.updateStageVisibility();
        this.commit(false, { syncTimeline: true });
    }

    async probeVideoMetadata(url) {
        const video = document.createElement("video");
        video.src = url;
        video.muted = true;
        video.playsInline = true;
        video.preload = "metadata";
        await new Promise((res, rej) => {
            video.onloadedmetadata = () => res();
            video.onerror = () => rej(new Error(t("upload.metaReadFailed")));
        });
        return {
            width: video.videoWidth || 0,
            height: video.videoHeight || 0,
            duration: video.duration || 0,
        };
    }

    onNodeResize() {
        if (this.isPlaying || this._pauseSettling) return;
        // Growable layout (no computeSize) → LiteGraph puts free space into computedHeight.
        bindDomWidgetContentComputeSize(this);
        this._resetLayoutStyles();
        this.applyZoomWidth();
        syncBatchPanelFillHeight(this);
        // Re-fill after LiteGraph finishes arranging widgets for the new node size.
        requestAnimationFrame(() => syncBatchPanelFillHeight(this));
        this.scheduleSettleRender();
    }

    getTimelineZoom() {
        return this.zoomEnabled ? Math.max(1, Number(this.zoom) || 1) : 1;
    }

    syncTimelineZoomUI() {
        this.zoomToggleBtn?.classList.toggle("active", !!this.zoomEnabled);
        this.zoomSlider?.classList.toggle("hidden", !this.zoomEnabled);
        if (this.zoomSlider && this.zoomEnabled) this.zoomSlider.value = String(this.zoom);
    }

    toggleTimelineZoom() {
        this.zoomEnabled = !this.zoomEnabled;
        this.syncTimelineZoomUI();
        this.applyZoomWidth();
        this.scheduleRender();
    }

    applyZoomWidth() {
        if (!this.canvas) return;
        const z = this.getTimelineZoom();
        const vp = this.viewport;
        if (z <= 1) {
            this.canvas.style.width = "100%";
            vp?.classList.remove("bd-zoomed");
            if (vp) vp.scrollLeft = 0;
            return;
        }
        const base = vp?.clientWidth || 960;
        const nextW = Math.max(base, base * z);
        const prevW = this.canvas.clientWidth || base;
        const midRatio = prevW > 0 ? ((vp?.scrollLeft || 0) + (vp?.clientWidth || base) / 2) / prevW : 0.5;
        this.canvas.style.width = `${nextW}px`;
        vp?.classList.add("bd-zoomed");
        if (vp) {
            requestAnimationFrame(() => {
                vp.scrollLeft = Math.max(0, midRatio * nextW - vp.clientWidth / 2);
            });
        }
    }

    adjustZoom(delta) {
        if (!this.zoomEnabled) return;
        this.zoom = clamp(this.zoom + delta, 1, 10);
        this.syncTimelineZoomUI();
        this.applyZoomWidth();
        this.scheduleRender();
    }

    frameToX(frame, width) { return (frame / Math.max(1, this.getTotalFrames())) * width; }
    xToFrame(x, width) { return clamp(Math.round((x / width) * this.getTotalFrames()), 0, this.getTotalFrames()); }

    getLayoutWidth() {
        return this._drawWidth || this._measureDrawWidth();
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const layoutW = this.getLayoutWidth();
        const layoutH = this.canvasHeight || (RULER_H + SEG_LABEL_H + TRACK_H);
        const scaleX = rect.width > 0 ? layoutW / rect.width : 1;
        const scaleY = rect.height > 0 ? layoutH / rect.height : 1;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    /** Shared draw + hit geometry for per-segment run checkboxes (segment top-left). */
    _runCheckGeometry(seg, width) {
        const x0 = this.frameToX(seg.start, width);
        const size = RUN_CHECK_SIZE;
        const boxX = x0 + 5;
        const boxY = TRACK_Y + 5;
        return {
            boxX,
            boxY,
            size,
            hitX0: boxX - RUN_CHECK_HIT_PAD_X,
            hitY0: boxY - RUN_CHECK_HIT_PAD_Y,
            hitX1: boxX + size + RUN_CHECK_HIT_PAD_X,
            hitY1: boxY + size + RUN_CHECK_HIT_PAD_Y,
        };
    }

    /** Master「段间引导」on + eligible task with ≥2 clips. */
    _showsContinuityJoints() {
        return isContinuityEligible(this) && isContinuityMasterEnabled(this.timeline?.output);
    }

    /**
     * Touching clip pairs in visual (time) order. `rightIndex` is the array index
     * whose `continuityFromPrev` flag owns this joint.
     */
    _continuityJointList(segs) {
        const ordered = (segs || [])
            .map((seg, arrayIndex) => ({ seg, arrayIndex }))
            .sort((a, b) => a.seg.start - b.seg.start || a.arrayIndex - b.arrayIndex);
        const joints = [];
        for (let r = 1; r < ordered.length; r++) {
            const left = ordered[r - 1];
            const right = ordered[r];
            const leftEnd = (left.seg.start || 0) + (left.seg.length || 0);
            if (Math.abs(leftEnd - (right.seg.start || 0)) > 2) continue;
            joints.push({
                leftIndex: left.arrayIndex,
                rightIndex: right.arrayIndex,
                a: r,
                b: r + 1,
                frame: right.seg.start,
                on: isSegmentContinuityFromPrev(right.seg, right.arrayIndex),
            });
        }
        return joints;
    }

    _continuityJointGeometry(frame, width) {
        const x = this.frameToX(frame, width);
        const w = CONT_JOINT_W;
        const h = CONT_JOINT_H;
        const y = CONT_JOINT_Y;
        // fl2v 首帧/尾帧 badges sit on the seam; give them a wider click target.
        const padX = this.isFl2vMode() ? 36 : CONT_JOINT_HIT_PAD;
        const padY = this.isFl2vMode() ? 12 : CONT_JOINT_HIT_PAD;
        return {
            x,
            y,
            w,
            h,
            hitX0: x - Math.max(w / 2, 16) - padX,
            hitX1: x + Math.max(w / 2, 16) + padX,
            // Include the S{n}→S{n+1} chip in the label band.
            hitY0: RULER_H + 1,
            hitY1: y + h + padY,
        };
    }

    _roundRectPath(ctx, x, y, w, h, r) {
        const rr = Math.min(r, w / 2, h / 2);
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
            ctx.roundRect(x, y, w, h, rr);
            return;
        }
        ctx.moveTo(x + rr, y);
        ctx.arcTo(x + w, y, x + w, y + h, rr);
        ctx.arcTo(x + w, y + h, x, y + h, rr);
        ctx.arcTo(x, y + h, x, y, rr);
        ctx.arcTo(x, y, x + w, y, rr);
        ctx.closePath();
    }

    /** Simple ↔ link glyph centered in the joint pill. */
    _drawContinuityLinkArrow(ctx, cx, cy, color) {
        const half = 6.5;
        const head = 3.2;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.6;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(cx - half, cy);
        ctx.lineTo(cx + half, cy);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - half + head, cy - 3);
        ctx.lineTo(cx - half, cy);
        ctx.lineTo(cx - half + head, cy + 3);
        ctx.moveTo(cx + half - head, cy - 3);
        ctx.lineTo(cx + half, cy);
        ctx.lineTo(cx + half - head, cy + 3);
        ctx.stroke();
        ctx.restore();
    }

    _drawContinuityJoints(width, segs) {
        if (!this._showsContinuityJoints() || this._drag?.kind === "reorder") return;
        const ctx = this.ctx;
        for (const joint of this._continuityJointList(segs)) {
            const g = this._continuityJointGeometry(joint.frame, width);
            const on = joint.on;
            const accent = on ? "#4fff8f" : "#7a7a7a";
            const fill = on ? "rgba(18, 48, 32, 0.96)" : "rgba(38, 38, 38, 0.94)";
            const rx = g.x - g.w / 2;
            ctx.save();
            this._roundRectPath(ctx, rx, g.y, g.w, g.h, 8);
            ctx.fillStyle = fill;
            ctx.fill();
            ctx.strokeStyle = accent;
            ctx.lineWidth = on ? 1.5 : 1.15;
            ctx.stroke();
            this._drawContinuityLinkArrow(ctx, g.x, g.y + g.h / 2, accent);
            const label = `S${joint.a}→S${joint.b}`;
            ctx.font = "800 8px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const tw = Math.max(g.w + 4, ctx.measureText(label).width + 8);
            const ly = RULER_H + 3;
            const lh = SEG_LABEL_H - 6;
            this._roundRectPath(ctx, g.x - tw / 2, ly, tw, lh, 4);
            ctx.fillStyle = fill;
            ctx.fill();
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = on ? "#b8ffd0" : "#9a9a9a";
            ctx.fillText(label, g.x, ly + lh / 2);
            ctx.restore();
        }
    }

    toggleContinuityJoint(rightIndex) {
        if (!(rightIndex > 0)) return;
        const seg = this.timeline.segments?.[rightIndex];
        if (!seg) return;
        const next = !isSegmentContinuityFromPrev(seg, rightIndex);
        seg.continuityFromPrev = next;
        if (this.isFl2vMode()) {
            const shot = this.timeline.shots?.[rightIndex];
            if (shot) shot.continuityFromPrev = next;
        }
        this.commit(false, { syncTimeline: true });
        this.flushTimelineSync?.();
        if (this.isFl2vMode()) updateFl2vDetailUI(this);
    }

    /** Draw fl2v edge grips; joints are split (top=prev yellow, bottom=next cyan). */
    _drawFl2vEdgeHandles(segs, index, x0, x1, width) {
        const ordered = (segs || [])
            .map((seg, i) => ({ seg, i }))
            .sort((a, b) => a.seg.start - b.seg.start || a.i - b.i);
        const rank = ordered.findIndex((o) => o.i === index);
        if (rank < 0) return;
        const prev = rank > 0 ? ordered[rank - 1] : null;
        const next = rank < ordered.length - 1 ? ordered[rank + 1] : null;
        const prevX1 = prev
            ? this.frameToX(prev.seg.start + prev.seg.length, width)
            : null;
        const nextX0 = next ? this.frameToX(next.seg.start, width) : null;
        const jointLeft = prev != null && Math.abs(prevX1 - x0) <= 2;
        const jointRight = next != null && Math.abs(nextX0 - x1) <= 2;
        const mid = TRACK_Y + TRACK_H / 2;
        const half = Math.max(10, TRACK_H / 2 - 6);

        if (!jointLeft) {
            this.ctx.fillStyle = "#ffcc00";
            this.ctx.fillRect(x0 - 2, mid - 12, 4, 24);
        }
        if (jointRight) {
            // Draw once on the left segment of the joint.
            this.ctx.fillStyle = "#ffcc00";
            this.ctx.fillRect(x1 - 2, TRACK_Y + 4, 4, half);
            this.ctx.fillStyle = "#5ec8ff";
            this.ctx.fillRect(x1 - 2, mid + 2, 4, half);
            this.ctx.fillStyle = "rgba(255,255,255,0.85)";
            this.ctx.fillRect(x1 - 3, mid - 1, 6, 2);
        } else {
            this.ctx.fillStyle = "#ffcc00";
            this.ctx.fillRect(x1 - 2, mid - 12, 4, 24);
        }
    }

    /**
     * fl2v edge handles: top half → previous clip's right edge;
     * bottom half → next clip's left edge.
     */
    _hitTestFl2vEdge(x, y, width, segs) {
        const ordered = (segs || [])
            .map((seg, index) => ({ seg, index }))
            .sort((a, b) => a.seg.start - b.seg.start || a.index - b.index);
        if (!ordered.length) return null;
        const trackMid = TRACK_Y + TRACK_H / 2;
        const preferNext = y >= trackMid;
        let best = null;
        let bestDist = HANDLE_PX + 1;

        for (let r = 0; r < ordered.length; r++) {
            const { seg, index } = ordered[r];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const prev = r > 0 ? ordered[r - 1] : null;
            const next = r < ordered.length - 1 ? ordered[r + 1] : null;
            const prevX1 = prev
                ? this.frameToX(prev.seg.start + prev.seg.length, width)
                : null;
            const nextX0 = next ? this.frameToX(next.seg.start, width) : null;
            const jointLeft = prev != null && Math.abs(prevX1 - x0) <= 2;
            const jointRight = next != null && Math.abs(nextX0 - x1) <= 2;

            const d0 = Math.abs(x - x0);
            if (d0 <= HANDLE_PX && d0 < bestDist) {
                if (jointLeft) {
                    best = preferNext
                        ? { type: "edge", index, edge: "left" }
                        : { type: "edge", index: prev.index, edge: "right" };
                } else {
                    best = { type: "edge", index, edge: "left" };
                }
                bestDist = d0;
            }
            const d1 = Math.abs(x - x1);
            if (d1 <= HANDLE_PX && d1 < bestDist) {
                if (jointRight) {
                    best = preferNext
                        ? { type: "edge", index: next.index, edge: "left" }
                        : { type: "edge", index, edge: "right" };
                } else {
                    best = { type: "edge", index, edge: "right" };
                }
                bestDist = d1;
            }
        }
        return best;
    }

    hitTest(x, y) {
        const width = this.getLayoutWidth();
        if (!width) return null;
        const segs = this._previewSegments || this.timeline.segments;
        const phx = this.frameToX(this.currentFrame, width);
        const trackBottom = TRACK_Y + TRACK_H;

        if (y <= RULER_H) {
            if (Math.abs(x - phx) <= HANDLE_PX) return { type: "playhead" };
            return { type: "ruler" };
        }

        // Checkbox corner wins over generic segment hit (same toggle action either way
        // in run-select mode; keeps hit type accurate for cursor / future hooks).
        if (this.isRunSelectEnabled() && this.getRunnableSegmentCount() >= 2 && y >= TRACK_Y && y <= trackBottom) {
            for (let i = segs.length - 1; i >= 0; i--) {
                if (this.isFl2vMode() && !segs[i]?.isStartFrame) continue;
                const g = this._runCheckGeometry(segs[i], width);
                if (x >= g.hitX0 && x <= g.hitX1 && y >= g.hitY0 && y <= g.hitY1) {
                    return { type: "run-check", index: i };
                }
            }
        }

        // Continuity pills sit on the seam (top of clip); win over split/edge in that pad.
        if (this._showsContinuityJoints() && y >= RULER_H && y <= TRACK_Y + CONT_JOINT_H + 24) {
            for (const joint of this._continuityJointList(segs)) {
                const g = this._continuityJointGeometry(joint.frame, width);
                if (x >= g.hitX0 && x <= g.hitX1 && y >= g.hitY0 && y <= g.hitY1) {
                    return {
                        type: "continuity-joint",
                        rightIndex: joint.rightIndex,
                        leftIndex: joint.leftIndex,
                        a: joint.a,
                        b: joint.b,
                        on: joint.on,
                    };
                }
            }
        }

        // Split markers: label band + full track height, before segment/edge hits.
        // (Previously label band returned null, so diamond clicks never registered.)
        if (y >= RULER_H && y <= trackBottom) {
            const hitPad = Math.max(HANDLE_PX, 12);
            let best = null;
            let bestDist = hitPad + 1;
            for (const frame of this.getEditableSplitFrames()) {
                const sx = this.frameToX(frame, width);
                const dist = Math.abs(x - sx);
                if (dist <= hitPad && dist < bestDist) {
                    bestDist = dist;
                    best = { type: "split", frame };
                }
            }
            if (best) return best;
        }

        if (y < TRACK_Y) return null;

        // Edge handles first so fl2v/gen can drag-extend duration (repeat thumbs).
        if (y >= TRACK_Y && y <= trackBottom) {
            if (this.isFl2vMode()) {
                const flHit = this._hitTestFl2vEdge(x, y, width, segs);
                if (flHit) return flHit;
            } else {
                for (let i = 0; i < segs.length; i++) {
                    const seg = segs[i];
                    const x0 = this.frameToX(seg.start, width);
                    const x1 = this.frameToX(seg.start + seg.length, width);
                    if (Math.abs(x - x0) <= HANDLE_PX) return { type: "edge", index: i, edge: "left" };
                    if (Math.abs(x - x1) <= HANDLE_PX) return { type: "edge", index: i, edge: "right" };
                }
            }
        }

        for (let i = segs.length - 1; i >= 0; i--) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const isLast = i === segs.length - 1;
            const insideX = isLast ? (x >= x0 && x <= x1) : (x >= x0 && x < x1);
            if (insideX && y >= TRACK_Y && y <= trackBottom) {
                return { type: "segment", index: i };
            }
        }

        if (Math.abs(x - phx) <= HANDLE_PX) return { type: "playhead" };
        return null;
    }

    onMouseDown(e) {
        if (e.button !== 0) return;
        // Keep LiteGraph / node drag from eating timeline clicks.
        stopDomEvent(e);
        e.preventDefault();
        const { x, y } = this.getMousePos(e);
        const hit = this.hitTest(x, y);
        if (!hit) {
            if (
                this.isFl2vMode()
                && !(this.timeline.segments || []).length
                && y >= TRACK_Y
                && y <= TRACK_Y + TRACK_H
            ) {
                openFl2vUpload(this);
            } else if (
                this.usesBatchTimeline()
                && !(this.timeline.segments || []).length
                && y >= TRACK_Y
                && y <= TRACK_Y + TRACK_H
            ) {
                addImageBatchGroup(this);
            } else if (
                this.needsSourceVideoUpload()
                && y >= TRACK_Y
                && y <= TRACK_Y + TRACK_H
            ) {
                this.pickVideoFile();
            }
            return;
        }
        if (
            this.needsSourceVideoUpload()
            && (hit.type === "segment" || hit.type === "edge")
        ) {
            this.pickVideoFile();
            return;
        }
        const width = this.getLayoutWidth();
        if (hit.type === "playhead" || hit.type === "ruler") {
            this.currentFrame = this.xToFrame(x, width);
            this._drag = { kind: "playhead" };
            this.clearSplitSelection();
        } else if (hit.type === "run-check") {
            this.toggleSegmentRun(hit.index);
            this._drag = null;
        } else if (hit.type === "continuity-joint") {
            this.toggleContinuityJoint(hit.rightIndex);
            this._drag = null;
        } else if (hit.type === "split") {
            this.selectSplitFrame(hit.frame);
            this._drag = null;
        } else if (hit.type === "segment") {
            if (this.isFl2vMode() && hit.index !== this.selectedIndex) {
                flushFl2vPromptDraft(this);
            }
            this.selectedIndex = hit.index;
            this.clearSplitSelection();
            this.updateSelectionUI();
            if (this.isFl2vMode() || this.usesBatchTimeline() || this.timeline.segments.length >= 2) {
                // Drag body to reorder / swap clip positions; edges still resize.
                this._drag = {
                    kind: "segment-pending",
                    index: hit.index,
                    x0: x,
                    y0: y,
                    fromRank: this._visualRankFromArrayIndex(hit.index),
                };
            } else {
                this._drag = { kind: "segment" };
            }
        } else if (hit.type === "edge") {
            if (this.isFl2vMode() && hit.index !== this.selectedIndex) {
                flushFl2vPromptDraft(this);
            }
            this.selectedIndex = hit.index;
            this.clearSplitSelection();
            this.updateSelectionUI();
            this._drag = { kind: "edge", index: hit.index, edge: hit.edge };
            this._edgeSnapshot = JSON.parse(JSON.stringify(this.timeline.segments));
        }
        this.scheduleRender();
    }

    onMouseMove(e) {
        if (!this._drag) return;
        const { x, y } = this.getMousePos(e);
        const width = this.getLayoutWidth();
        const frame = this.xToFrame(x, width);

        if (this._drag.kind === "segment-pending") {
            if (Math.hypot(x - this._drag.x0, y - this._drag.y0) > 6) {
                this._drag = {
                    kind: "reorder",
                    fromRank: this._drag.fromRank,
                    index: this._drag.index,
                    pointerX: x,
                    pointerY: y,
                    originX: this._drag.x0,
                    originY: this._drag.y0,
                };
                this._reorderFromRank = this._drag.fromRank;
                this._reorderDropRank = this._drag.fromRank;
                this.canvas.classList.add("bd-grabbing");
                this.canvas.style.cursor = "grabbing";
            }
            return;
        }

        if (this._drag.kind === "playhead") {
            this.currentFrame = frame;
        } else if (this._drag.kind === "reorder") {
            this._drag.pointerX = x;
            this._drag.pointerY = y;
            this._reorderDropRank = this._computeReorderDropRank(frame, this._drag.fromRank);
            this.scheduleRender();
            return;
        } else if (this._drag.kind === "fl2v-move") {
            // Block-move: this clip + all later clips shift together (LTX ripple).
            const snap = this._edgeSnapshot || this.timeline.segments;
            const segs = snap.map((s) => ({ ...s }));
            const i = this._drag.index;
            const seg = segs[i];
            if (!seg) return;
            const width = this.getLayoutWidth();
            const frame0 = this.xToFrame(this._drag.x0, width);
            let delta = frame - frame0;
            const ordered = segs
                .map((s, idx) => ({ s, idx }))
                .sort((a, b) => a.s.start - b.s.start || a.idx - b.idx);
            const rank = ordered.findIndex((o) => o.s.id === seg.id);
            if (rank < 0) return;
            const prev = rank > 0 ? ordered[rank - 1].s : null;
            const minStart = prev ? prev.start + prev.length : 0;
            const desired = this._drag.start0 + delta;
            const clampedStart = Math.max(minStart, desired);
            delta = clampedStart - this._drag.start0;
            for (let r = rank; r < ordered.length; r++) {
                const orig = snap.find((x) => x.id === ordered[r].s.id) || ordered[r].s;
                ordered[r].s.start = Math.max(0, (parseInt(orig.start, 10) || 0) + delta);
                ordered[r].s.length = Math.max(minFrameCount("fl2v"), parseInt(orig.length, 10) || minFrameCount("fl2v"));
                ordered[r].s.frameCount = ordered[r].s.length;
            }
            this._previewSegments = segs;
        } else if (this._drag.kind === "edge") {
            const segs = this._edgeSnapshot.map((s) => ({ ...s }));
            const i = this._drag.index;
            const seg = segs[i];
            const isFl2v = this.isFl2vMode();
            const isGen = this.isGenMode();
            const isBatchTrack = this.usesBatchTimeline();
            const minLen = (isFl2v || isGen || isBatchTrack) ? minFrameCount(this.getTaskKey()) : MIN_SEG;
            if (isFl2v) {
                // LTX-style ripple: resize this clip's right edge and shift ALL later clips.
                // Left edge of a non-first clip = ripple the previous clip's right edge.
                // May extend past the sampling window (dashed overflow, not sampled).
                const ordered = [...segs]
                    .map((s, idx) => ({ s, idx }))
                    .sort((a, b) => a.s.start - b.s.start || a.idx - b.idx);
                const rank = ordered.findIndex((o) => o.s.id === seg.id);
                if (this._drag.edge === "right") {
                    const newEnd = Math.max(seg.start + minLen, frame);
                    rippleFl2vRightEdge(segs, i, newEnd, minLen, this);
                } else if (this._drag.edge === "left") {
                    if (rank > 0) {
                        const prevIdx = ordered[rank - 1].idx;
                        const prev = ordered[rank - 1].s;
                        const newEnd = Math.max(prev.start + minLen, frame);
                        rippleFl2vRightEdge(segs, prevIdx, newEnd, minLen, this);
                    }
                    // First clip's left edge stays at 0 (no negative timeline).
                }
            } else if (this._drag.edge === "left") {
                const prev = segs[i - 1];
                const minStart = prev ? prev.start + minLen : 0;
                const maxStart = seg.start + seg.length - minLen;
                seg.start = clamp(frame, minStart, maxStart);
                seg.length = (this._edgeSnapshot[i].start + this._edgeSnapshot[i].length) - seg.start;
                if (isGen || isBatchTrack) seg.frameCount = seg.length;
                if (prev) {
                    prev.length = seg.start - prev.start;
                    if (isGen || isBatchTrack) prev.frameCount = prev.length;
                }
            } else {
                const next = segs[i + 1];
                const minEnd = seg.start + minLen;
                let maxEnd;
                if (next) {
                    maxEnd = this._edgeSnapshot[i + 1].start + this._edgeSnapshot[i + 1].length;
                    if (isGen || isBatchTrack) maxEnd -= minLen;
                } else if (isGen || isBatchTrack) {
                    maxEnd = seg.start + MAX_GEN_FRAMES;
                } else {
                    maxEnd = this.getTotalFrames();
                }
                const end = clamp(frame, minEnd, maxEnd);
                seg.length = end - seg.start;
                if (isGen || isBatchTrack) seg.frameCount = seg.length;
                if (next) {
                    next.start = end;
                    next.length = (this._edgeSnapshot[i + 1].start + this._edgeSnapshot[i + 1].length) - end;
                    if (isGen || isBatchTrack) next.frameCount = next.length;
                }
            }
            this._previewSegments = segs;
            this._syncLiveDurationUiFromPreview();
        }
        this.scheduleRender();
    }

    /**
     * While dragging a segment edge, keep group seconds / toolbar / output preview in sync
     * without rebuilding the whole batch DOM (that would break the drag).
     */
    _syncLiveDurationUiFromPreview() {
        const segs = this._previewSegments;
        if (!segs?.length) return;

        if (this.isR2vBatch() || (this.isImageBatch() && isVideoBatchTask(this.getTaskKey()))) {
            for (const input of this.batchList?.querySelectorAll("input[data-batch-sec-index]") || []) {
                if (input === document.activeElement) continue;
                const index = parseInt(input.getAttribute("data-batch-sec-index"), 10);
                if (!Number.isFinite(index)) continue;
                const seg = segs[index];
                if (!seg) continue;
                const fc = Math.max(1, parseInt(seg.frameCount ?? seg.length, 10) || 1);
                const sec = preferredDurationSecFromFrames(fc, 24);
                const play = framesToDurationSec(fc, 24);
                if (input.value !== String(sec)) input.value = String(sec);
                input.title = t("batch.durationTooltip", { frames: fc, play });
            }
            this.updateVideoNameLabel();
            this.updateOutputPreview();
            return;
        }

        if (this.isFl2vMode()) {
            const shots = this.timeline.shots || [];
            for (const card of this.fl2vUi?.shotsEl?.querySelectorAll(".bd-fl2v-shot") || []) {
                const index = parseInt(card.dataset.shotIndex, 10);
                const input = card.querySelector('[data-r="shot-sec"]');
                const shot = shots[index];
                if (!input || !shot || input === document.activeElement) continue;
                const sec = roundDurationSec(Number(shot.durationSec) || 0);
                if (input.value !== String(sec)) input.value = String(sec);
            }
            if (this.fl2vUi?.totalInput && this.fl2vUi.totalInput !== document.activeElement) {
                this.fl2vUi.totalInput.value = String(getFl2vTotalDurationSec(this));
            }
            this.updateVideoNameLabel();
            this.updateOutputPreview();
            return;
        }

        // v2v / rv2v video timeline: live-update segment header + bounds while dragging.
        this._updateSegInfoFromSegment?.(segs[this.selectedIndex]);
        this._updateTimelineDom?.({ skipSeek: true });
        this.updateOutputPreview();
    }

    /** Build / refresh the segment panel meta line (frames, duration, ref counts). */
    _updateSegInfoFromSegment(seg) {
        if (!this.segInfo || !seg || this.isGlobalMode()) return;
        const fps = this.getFrameRate();
        const segKey = resolveTaskKey(
            seg.taskType || this.timeline.global?.taskType || this.getTaskKey(),
        );
        let info;
        if (this.isGenMode()) {
            const fc = seg.frameCount ?? seg.length;
            info = t("segment.infoFrames", { n: fc });
            if (this.isGenImage()) {
                info += seg.genImage?.imageFile ? t("segment.uploadedImage") : t("segment.noImage");
            }
        } else {
            info = t("segment.infoRange", {
                start: seg.start,
                end: seg.start + seg.length,
                length: seg.length,
                sec: (seg.length / fps).toFixed(2),
            });
            const clips = this.getVideoClips();
            if (clips.length > 1) {
                const clip = clips[this.getSegmentClipIndex(seg)];
                const clipName = clip?.fileName || clip?.videoFile
                    || t("slot.video", { n: this.getSegmentClipIndex(seg) + 1 });
                info += ` · ${clipName}`;
            }
            if (taskUsesReferenceVideo(segKey)) {
                info += seg.referenceVideo?.videoFile || seg.referenceVideo?.fileName
                    ? t("segment.refVideoUploaded")
                    : t("segment.refVideoMissing");
            }
            if (taskUsesReferenceImages(segKey) || taskUsesReferenceAudios(segKey)) {
                let imgs = 0;
                let audios = 0;
                for (const r of seg.refs || []) {
                    if (r?.imageFile || r?.imageB64) imgs += 1;
                }
                for (const r of seg.refAudios || []) {
                    if (r?.audioFile || r?.fileName) audios += 1;
                }
                info += ` · ${t("segment.refSummary", {
                    imgs,
                    maxImgs: MAX_REFERENCE_IMAGES,
                    audios,
                    maxAudios: MAX_REFERENCE_AUDIOS,
                })}`;
            }
        }
        this.segInfo.textContent = info;
    }

    onMouseUp() {
        if (
            (this._drag?.kind === "edge" || this._drag?.kind === "fl2v-move")
            && this._previewSegments
        ) {
            const preview = this._previewSegments;
            this._previewSegments = null;
            if (this.isFl2vMode()) {
                // Shot durations already updated during drag; rebuild layout from shots.
                syncFl2vDurationSecAfterDrag(this);
                updateFl2vDetailUI(this);
                this.updateVideoNameLabel();
            } else if (this.usesBatchTimeline()) {
                this.timeline.segments = preview;
                for (const seg of this.timeline.segments) {
                    const fc = Math.max(1, parseInt(seg.frameCount ?? seg.length, 10) || 1);
                    seg.frameCount = fc;
                    seg.length = fc;
                    seg.durationSec = preferredDurationSecFromFrames(fc, 24);
                }
                normalizeImageBatchSegments(this);
                this.renderImageBatchGroups();
                this.updateVideoNameLabel();
            } else {
                this._applyOuterVideoCrop(preview);
            }
            this.commit();
        } else if (this._drag?.kind === "reorder") {
            const toRank = this._reorderDropRank;
            if (toRank >= 0 && toRank !== this._drag.fromRank) {
                this.reorderSegmentsByRank(this._drag.fromRank, toRank);
                this.commit(false, { syncTimeline: true });
                if (this.isFl2vMode()) {
                    updateFl2vDetailUI(this);
                    this.updateVideoNameLabel();
                } else if (this.usesBatchTimeline()) {
                    this.renderImageBatchGroups();
                    this.updateVideoNameLabel();
                }
            }
            this._reorderDropRank = -1;
            this._reorderFromRank = -1;
            this.canvas.classList.remove("bd-grabbing");
            this.canvas.style.cursor = "";
        } else if (this._drag) {
            this.seekBar.value = this.currentFrame;
            this.scheduleRender();
        }
        this._drag = null;
        this._edgeSnapshot = null;
    }

    _applyOuterVideoCrop(preview) {
        const total = this.getTotalFrames();
        const ordered = [...(preview || [])].sort((a, b) => a.start - b.start);
        if (!ordered.length || total <= 0) {
            this.timeline.segments = preview || [];
            return false;
        }
        const cropStart = clamp(Math.round(Number(ordered[0].start) || 0), 0, total);
        const last = ordered[ordered.length - 1];
        const cropEnd = clamp(
            Math.round((Number(last.start) || 0) + (Number(last.length) || 0)),
            cropStart,
            total,
        );
        if (cropStart <= 0 && cropEnd >= total) {
            this.timeline.segments = preview;
            return false;
        }

        if (!this.getFrameMap().length) this.materializeFrameMap();
        const croppedMap = this.getFrameMap().slice(cropStart, cropEnd);
        const selectedId = this.timeline.segments?.[this.selectedIndex]?.id;
        this.setFrameMap(croppedMap);
        this.timeline.totalFrames = croppedMap.length;
        this._syncPrimaryVideoFromClips(croppedMap);
        this.timeline.videoWorkspace = null;
        this.timeline.segments = ordered.flatMap((seg) => {
            const start = Math.max(cropStart, Number(seg.start) || 0);
            const end = Math.min(cropEnd, (Number(seg.start) || 0) + (Number(seg.length) || 0));
            if (end - start < MIN_SEG) return [];
            return [{ ...seg, start: start - cropStart, length: end - start }];
        });
        this.selectedIndex = Math.max(
            0,
            this.timeline.segments.findIndex((seg) => seg.id === selectedId),
        );
        this.currentFrame = clamp(this.currentFrame - cropStart, 0, Math.max(0, croppedMap.length - 1));
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, croppedMap.length - 1);
            this.seekBar.value = this.currentFrame;
        }
        if (this.totalFramesWidget) this.totalFramesWidget.value = croppedMap.length;
        this._thumbCache.clear();
        this._thumbPending.clear();
        this._prefetchSegmentThumbs(0, Math.min(croppedMap.length, THUMB_PREFETCH_BATCH * 4));
        this._syncStagePreview(this.currentFrame, { force: true });
        this.updateVideoNameLabel();
        return true;
    }

    addSplitAtMouse(e) {
        const { x } = this.getMousePos(e);
        this.splitAtFrame(this.xToFrame(x, this.getLayoutWidth()));
    }

    splitAtFrame(frame) {
        if (this.isGenMode()) {
            this.genSplitAtFrame(frame);
            return;
        }
        const total = this.getTotalFrames();
        if (frame <= MIN_SEG || frame >= total - MIN_SEG) return;
        const newSegs = [];
        for (const seg of [...this.timeline.segments].sort((a, b) => a.start - b.start)) {
            const end = seg.start + seg.length;
            if (frame > seg.start && frame < end) {
                newSegs.push({ ...seg, length: frame - seg.start });
                newSegs.push({ id: uid(), start: frame, length: end - frame, prompt: "", taskType: "", refs: [], referenceVideo: {} });
            } else newSegs.push({ ...seg });
        }
        this.timeline.segments = newSegs;
        this.selectedSplitFrame = null;
        this.commit();
        this.updateSplitPointUI();
    }

    equalSplit() {
        if (this.isGenMode()) {
            this.genEqualSplit();
            return;
        }
        const n = parseInt(this.equalCountInput?.value || "2", 10);
        if (!n || n < 2) return;
        const total = this.getTotalFrames();
        if (total < MIN_SEG * 2) return;
        const maxSeg = Math.floor(total / MIN_SEG);
        const count = clamp(n, 2, Math.max(2, maxSeg || 2));
        if (this.equalCountInput) this.equalCountInput.value = String(count);

        const points = new Set([0, total]);
        const clipBounds = this.getClipBoundaries();
        for (const b of clipBounds) {
            if (b > 0 && b < total) points.add(b);
        }
        for (let i = 1; i < count; i++) {
            const p = Math.round((i * total) / count);
            if (p > 0 && p < total) points.add(p);
        }

        const forced = new Set([0, total, ...clipBounds]);
        const newSegs = this._buildSegmentsFromSplitPoints([...points], forced);
        if (!newSegs?.length) return;
        this.timeline.segments = newSegs;
        this.selectedSplitFrame = null;
        this.commit();
        this.updateSplitPointUI();
    }

    /** Logical ranges for each video clip on the timeline. */
    getClipLogicalRanges() {
        const clips = this.getVideoClips();
        const total = this.getTotalFrames();
        if (!clips.length) return [];
        const map = this.getFrameMap();
        if (map.length) {
            const ranges = clips.map((clip, clipIndex) => ({
                clip,
                clipIndex,
                start: total,
                end: 0,
            }));
            for (let i = 0; i < map.length; i++) {
                const entry = normalizeFrameMapEntry(map[i]);
                const r = ranges[entry.clip];
                if (!r) continue;
                if (i < r.start) r.start = i;
                if (i + 1 > r.end) r.end = i + 1;
            }
            return ranges.filter((r) => r.end > r.start);
        }
        if (clips.length === 1) {
            return [{ clip: clips[0], clipIndex: 0, start: 0, end: total }];
        }
        let cursor = 0;
        return clips.map((clip, clipIndex) => {
            const len = Math.max(0, parseInt(clip.sourceFrameCount, 10) || 0);
            const start = cursor;
            const end = Math.min(total, cursor + len);
            cursor = end;
            return { clip, clipIndex, start, end };
        }).filter((r) => r.end > r.start);
    }

    /** Interior segment boundaries that can be selected/deleted (not clip seams). */
    getEditableSplitFrames() {
        if (this.isFl2vMode() || this.isGenMode() || this.isImageBatch()) return [];
        const total = this.getTotalFrames();
        if (total < MIN_SEG * 2) return [];
        const forced = new Set([0, total, ...this.getClipBoundaries()]);
        const segs = this._previewSegments || this.timeline.segments || [];
        const points = [];
        for (const seg of segs) {
            const start = Math.max(0, parseInt(seg.start, 10) || 0);
            if (start > 0 && start < total && !forced.has(start)) points.push(start);
        }
        return [...new Set(points)].sort((a, b) => a - b);
    }

    selectSplitFrame(frame) {
        const editable = this.getEditableSplitFrames();
        const n = Number(frame);
        if (!Number.isFinite(n) || !editable.includes(n)) {
            this.selectedSplitFrame = null;
        } else {
            // Toggle off if clicking the same selected split again.
            this.selectedSplitFrame = this.selectedSplitFrame === n ? null : n;
            if (this.selectedSplitFrame != null) {
                const segs = this.timeline.segments || [];
                const idx = segs.findIndex((s) => (parseInt(s.start, 10) || 0) === n);
                if (idx >= 0) this.selectedIndex = idx;
            }
        }
        this.updateSplitPointUI();
        this.updateSelectionUI();
        this.scheduleRender();
    }

    clearSplitSelection() {
        if (this.selectedSplitFrame == null) return;
        this.selectedSplitFrame = null;
        this.updateSplitPointUI();
        this.scheduleRender();
    }

    updateSplitPointUI() {
        const bar = this.splitEditBarEl || this.root?.querySelector('[data-r="split-edit-bar"]');
        const hint = this.splitEditHintEl || this.root?.querySelector('[data-r="split-edit-hint"]');
        const btn = this.root?.querySelector('[data-a="del-split"]');
        if (this.isImageBatch() || this.isGenMode()) {
            bar?.classList.add("hidden");
            return;
        }
        const has = this.selectedSplitFrame != null
            && this.getEditableSplitFrames().includes(this.selectedSplitFrame);
        if (bar) bar.classList.toggle("hidden", !has);
        if (hint && has) {
            hint.textContent = t("split.hintSelected", { f: this.selectedSplitFrame });
        }
        if (btn) {
            btn.disabled = !has;
            btn.title = has
                ? t("split.tooltipDelete", { f: this.selectedSplitFrame })
                : t("split.tooltipSelectFirst");
        }
        if (has && this.boundsEl) {
            this.boundsEl.textContent = t("split.boundsSelected", { f: this.selectedSplitFrame });
        }
    }

    deleteSelectedSplitPoint() {
        if (this.isGenMode() || this.isImageBatch()) return;
        const frame = this.selectedSplitFrame;
        if (frame == null) return;
        if (!this.getEditableSplitFrames().includes(frame)) {
            this.clearSplitSelection();
            return;
        }
        const segs = [...(this.timeline.segments || [])].sort((a, b) => a.start - b.start);
        const rightIdx = segs.findIndex((s) => (parseInt(s.start, 10) || 0) === frame);
        if (rightIdx <= 0) {
            this.clearSplitSelection();
            return;
        }
        const left = segs[rightIdx - 1];
        const right = segs[rightIdx];
        left.length = (parseInt(left.length, 10) || 0) + (parseInt(right.length, 10) || 0);
        segs.splice(rightIdx, 1);
        this.timeline.segments = segs;
        this.selectedSplitFrame = null;
        this.selectedIndex = Math.max(0, rightIdx - 1);
        this.commit();
        this.updateSelectionUI();
        this.updateSplitPointUI();
        this.setSmartSplitMessage("");
        this.scheduleRender();
    }

    setSmartSplitMessage(text, { ok = false } = {}) {
        const el = this.smartSplitMsgEl || this.root?.querySelector('[data-r="smart-split-msg"]');
        if (!el) return;
        const msg = String(text || "").trim();
        if (!msg) {
            el.textContent = "";
            el.classList.add("hidden");
            el.classList.remove("ok");
            return;
        }
        el.textContent = msg;
        el.classList.toggle("ok", !!ok);
        el.classList.remove("hidden");
    }

    async smartSplit() {
        if (this.isGenMode() || this.isImageBatch()) return;
        if (!this.hasVideo()) {
            this.setSmartSplitMessage(t("smartSplit.needVideo"));
            return;
        }
        const total = this.getTotalFrames();
        if (total < MIN_SEG * 2) {
            this.setSmartSplitMessage(t("smartSplit.tooShort"));
            return;
        }
        const ranges = this.getClipLogicalRanges();
        if (!ranges.length) {
            this.setSmartSplitMessage(t("smartSplit.noMaterial"));
            return;
        }
        const btn = this.root?.querySelector('[data-a="smart-split"]');
        const prevLabel = btn?.textContent;
        if (btn) {
            btn.disabled = true;
            btn.textContent = t("common.analyzing");
        }
        this.setSmartSplitMessage(t("smartSplit.analyzing"));
        try {
            const clips = ranges.map((r) => ({
                videoFile: r.clip.videoFile || r.clip.fileName,
                subfolder: r.clip.subfolder || "",
                type: r.clip.type || "input",
                logicalStart: r.start,
                logicalEnd: r.end,
                nativeFps: r.clip.nativeFps || r.clip.native_fps || null,
            }));
            const resp = await api.fetchApi("/minimax/director/detect_shots", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    clips,
                    frameRate: this.getFrameRate(),
                    totalFrames: total,
                    sensitivity: "medium",
                    minShotFrames: Math.max(MIN_SEG, 12),
                }),
            });
            if (!resp.ok) {
                throw new Error((await resp.text()) || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const cutFrames = Array.isArray(data.cutFrames) ? data.cutFrames.map((n) => parseInt(n, 10) || 0) : [];
            const points = new Set([0, total, ...cutFrames.filter((f) => f > 0 && f < total)]);
            const clipBounds = this.getClipBoundaries();
            for (const b of clipBounds) {
                if (b > 0 && b < total) points.add(b);
            }
            const forced = new Set([0, total, ...clipBounds]);
            const newSegs = this._buildSegmentsFromSplitPoints([...points], forced);
            if (!newSegs?.length) {
                this.setSmartSplitMessage(t("smartSplit.noSegments"));
                return;
            }
            this.timeline.segments = newSegs;
            this.selectedIndex = 0;
            this.selectedSplitFrame = null;
            this.commit();
            this.updateSelectionUI();
            this.updateSplitPointUI();
            const shotCount = data.shotCount ?? Math.max(0, newSegs.length);
            const warn = Array.isArray(data.warnings) && data.warnings.length
                ? ` ${data.warnings[0]}`
                : "";
            this.setSmartSplitMessage(
                t("smartSplit.done", { shots: shotCount, segs: newSegs.length }) + (warn || ""),
                { ok: !warn },
            );
        } catch (err) {
            console.error("[MiniMax H3 Director] smartSplit failed", err);
            this.setSmartSplitMessage(t("smartSplit.failed", { err: err?.message || err }));
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = prevLabel || t("toolbar.smartSplit");
            }
        }
    }

    deleteSelectedSegment() {
        if (this.isGenMode()) {
            this.genDeleteSelectedSegment();
            return;
        }
        if (this.usesBatchTimeline()) {
            deleteImageBatchGroup(this, this.selectedIndex);
            this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, this.getTotalFrames() - 1));
            if (this.seekBar) {
                this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
                this.seekBar.value = this.currentFrame;
            }
            this.updateVideoNameLabel();
            this.updateDomWidgetHeight();
            this.scheduleRender();
            return;
        }
        if (this.isImageBatch()) {
            this.genDeleteSelectedSegment();
            return;
        }
        if (this.isFl2vMode()) {
            const idx = this.selectedIndex;
            const shots = this.timeline.shots || [];
            if (!shots[idx] && !(this.timeline.segments || [])[idx]) return;
            removeFl2vShot(this, idx);
            this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, this.getTotalFrames() - 1));
            if (this.seekBar) {
                this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
                this.seekBar.value = this.currentFrame;
            }
            this.commit(false, { syncTimeline: true });
            updateFl2vDetailUI(this);
            this.updateVideoNameLabel();
            this.updateDomWidgetHeight();
            return;
        }
        const idx = this.selectedIndex;
        const seg = this.timeline.segments[idx];
        if (!seg) return;

        const start = Math.max(0, parseInt(seg.start, 10) || 0);
        const len = Math.max(0, parseInt(seg.length, 10) || 0);
        this.selectedSplitFrame = null;

        // Remove segment UI entry first, then cut matching frames from the
        // logical timeline so preview / export no longer include that range.
        this.timeline.segments.splice(idx, 1);

        let total = this.getTotalFrames();
        let map = [];
        if (len > 0 && total > 0) {
            // Sparse uploads start with an empty frameMap; materialize so we can
            // splice out the deleted range from the source-frame mapping.
            if (!this.getFrameMap().length) this.materializeFrameMap();
            map = [...this.getFrameMap()];
            if (map.length) {
                const from = clamp(start, 0, map.length);
                const count = clamp(len, 0, map.length - from);
                if (count > 0) map.splice(from, count);
                this.setFrameMap(map);
                this._syncPrimaryVideoFromClips(map);
                total = map.length;
            } else {
                // Fallback: record deleted source ranges (kept across sync).
                const video = this.timeline.video || {};
                video.deletedSourceRanges = video.deletedSourceRanges || [];
                const srcStart = this.logicalToSourceFrame(start);
                video.deletedSourceRanges.push([srcStart, srcStart + len]);
                video.deletedSourceRanges.sort((a, b) => a[0] - b[0]);
                this.timeline.video = video;
                total = this.getTotalFrames();
                this.timeline.totalFrames = total;
                this._syncPrimaryVideoFromClips([]);
            }
        }

        // Keep thumbs for remaining source frames; drop only if the clip is gone unused.
        this.timeline.videoWorkspace = null;

        if (this.totalFramesWidget) this.totalFramesWidget.value = total;

        this.compactSegmentsAfterDelete();

        this.selectedIndex = clamp(idx, 0, Math.max(0, this.timeline.segments.length - 1));
        this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, total - 1));
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = this.currentFrame;
        }

        if (!total) {
            this.videoNameEl.textContent = t("toolbar.noVideo");
            this._clearVideoState({ dropUnusedThumbs: true });
        } else {
            this.updateVideoNameLabel();
            this._prefetchSegmentThumbs(0, Math.min(total, THUMB_PREFETCH_BATCH * 4));
            this._syncStagePreview(this.currentFrame, { force: true });
            this.updateStageVisibility();
        }

        this.commit(false, { syncTimeline: true });
    }

    compactSegmentsAfterDelete() {
        const total = this.getTotalFrames();
        if (total <= 0) {
            this.timeline.segments = [];
            return;
        }
        const segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        if (!segs.length) {
            this.timeline.segments = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
            return;
        }
        let cursor = 0;
        const fixed = [];
        for (const seg of segs) {
            let length = seg.length ?? MIN_SEG;
            if (cursor + length > total) length = total - cursor;
            if (length < MIN_SEG) {
                if (fixed.length) fixed[fixed.length - 1].length += length;
                cursor += length;
                continue;
            }
            fixed.push({ ...seg, start: cursor, length, refs: seg.refs || [] });
            cursor += length;
        }
        if (!fixed.length) {
            this.timeline.segments = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
        } else if (cursor < total) {
            fixed[fixed.length - 1].length += total - cursor;
        }
        this.timeline.segments = fixed;
    }

    getFrameImage(frameIndex) {
        return this._thumbCache.get(this._frameThumbKey(frameIndex)) || null;
    }

    drawSegmentThumbnails(ctx, seg, startX, pxWidth, y0, h, index = -1) {
        if (this.isFl2vMode()) {
            drawFl2vSegmentThumbnails(this, ctx, seg, startX, pxWidth, y0, h);
            return;
        }
        ctx.save();
        ctx.beginPath();
        ctx.rect(startX, y0 + 1, pxWidth, h - 2);
        ctx.clip();

        if (this.isR2vBatch()) {
            ctx.fillStyle = "#0d0d0d";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            const refs = [...(seg.refs || [])].sort(
                (a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0),
            );
            const commonRefs = listCommonImageRefs(this);
            const imgFile = refs.find((r) => r?.imageFile)?.imageFile
                || commonRefs.find((r) => r?.imageFile)?.imageFile
                || "";
            const previewB64 = seg.previewB64 || (Array.isArray(seg.previewFrames) ? seg.previewFrames[0] : "");
            const vidRef = [...(seg.refVideos || [])]
                .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0))
                .find((r) => r?.videoFile || r?.previewImageFile || r?.previewImageUrl || r?.linked);
            const vidPath = vidRef?.videoFile || "";
            const vidType = vidRef?.type || "input";
            const posterFile = vidRef?.previewImageFile || "";
            const posterUrl = vidRef?.previewImageUrl || "";
            let cacheKey = "";
            let srcKind = "";
            if (imgFile) {
                cacheKey = `r2v:${imgFile}`;
                srcKind = "image";
            } else if (previewB64) {
                cacheKey = `r2v-prev:${seg.id || startX}`;
                srcKind = "preview";
            } else if (vidPath) {
                cacheKey = `r2v-vid:${vidType}:${vidPath}`;
                srcKind = "video";
            } else if (posterFile || posterUrl) {
                cacheKey = `r2v-vid-poster:${posterFile || posterUrl}`;
                srcKind = "poster";
            }
            const drawCached = (img) => {
                if (!img?.naturalWidth && !img?.width) return false;
                const natW = img.naturalWidth || img.width;
                const natH = Math.max(1, img.naturalHeight || img.height);
                const ratio = natW / natH;
                let dw = pxWidth - 4;
                let dh = dw / ratio;
                if (dh > h - 4) {
                    dh = h - 4;
                    dw = dh * ratio;
                }
                ctx.drawImage(img, startX + (pxWidth - dw) / 2, y0 + (h - dh) / 2, dw, dh);
                return true;
            };
            if (cacheKey) {
                const img = this._thumbCache.get(cacheKey);
                if (!drawCached(img)) {
                    if (srcKind === "video") {
                        this._queueR2vVideoThumb(cacheKey, vidPath, vidType);
                    } else if (!this._thumbPending.has(cacheKey)) {
                        this._thumbPending.add(cacheKey);
                        const el = new Image();
                        el.crossOrigin = "anonymous";
                        el.onload = () => {
                            this._thumbCache.set(cacheKey, el);
                            this._thumbPending.delete(cacheKey);
                            this.scheduleRender();
                        };
                        el.onerror = () => this._thumbPending.delete(cacheKey);
                        if (srcKind === "image") el.src = refViewUrl(imgFile);
                        else if (srcKind === "poster") {
                            el.src = posterUrl || refViewUrl(posterFile);
                        } else {
                            el.src = String(previewB64).startsWith("data:")
                                ? previewB64
                                : `data:image/png;base64,${previewB64}`;
                        }
                    }
                }
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(t("canvas.uploadR2vMedia"), startX + pxWidth / 2, y0 + h / 2);
            }
            ctx.restore();
            return;
        }

        if (this.getTaskKey() === "i2v") {
            ctx.fillStyle = "#0d0d0d";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            const imgFile = seg.genImage?.imageFile || seg.imageFile || "";
            const previewB64 = seg.previewB64 || (Array.isArray(seg.previewFrames) ? seg.previewFrames[0] : "");
            const cacheKey = imgFile
                ? `i2v:${imgFile}`
                : (previewB64 ? `i2v-prev:${seg.id || startX}` : "");
            const drawCached = (img) => {
                if (!img?.naturalWidth && !img?.width) return false;
                const natW = img.naturalWidth || img.width;
                const natH = Math.max(1, img.naturalHeight || img.height);
                const ratio = natW / natH;
                let dw = pxWidth - 4;
                let dh = dw / ratio;
                if (dh > h - 4) {
                    dh = h - 4;
                    dw = dh * ratio;
                }
                ctx.drawImage(img, startX + (pxWidth - dw) / 2, y0 + (h - dh) / 2, dw, dh);
                return true;
            };
            if (cacheKey) {
                const img = this._thumbCache.get(cacheKey);
                if (!drawCached(img) && !this._thumbPending.has(cacheKey)) {
                    this._thumbPending.add(cacheKey);
                    const el = new Image();
                    el.crossOrigin = "anonymous";
                    el.onload = () => {
                        this._thumbCache.set(cacheKey, el);
                        this._thumbPending.delete(cacheKey);
                        this.scheduleRender();
                    };
                    el.onerror = () => this._thumbPending.delete(cacheKey);
                    if (imgFile) el.src = refViewUrl(imgFile);
                    else {
                        el.src = String(previewB64).startsWith("data:")
                            ? previewB64
                            : `data:image/png;base64,${previewB64}`;
                    }
                }
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(t("batch.uploadSource"), startX + pxWidth / 2, y0 + h / 2);
            }
            ctx.restore();
            return;
        }

        if (this.getTaskKey() === "t2v") {
            const segs = this._previewSegments || this.timeline.segments || [];
            const idx = (Number.isFinite(index) && index >= 0)
                ? index
                : Math.max(0, segs.indexOf(seg));
            const fills = ["#2a2618", "#182028", "#182818", "#281820"];
            const accents = ["#d4a017", "#66aaff", "#4fff8f", "#ff66aa"];
            ctx.fillStyle = fills[idx % fills.length];
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            ctx.fillStyle = accents[idx % accents.length];
            ctx.fillRect(startX, y0 + 1, Math.min(4, Math.max(2, pxWidth * 0.04)), h - 2);
            if (pxWidth > 40) {
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                const title = t("batch.groupTitle.prompt", { n: idx + 1 });
                const sec = Number(seg.durationSec);
                const sub = Number.isFinite(sec) && sec > 0
                    ? `${sec.toFixed(1)}s`
                    : `${seg.length || 0}f`;
                ctx.fillStyle = "#e8e8e8";
                ctx.font = "bold 12px sans-serif";
                ctx.fillText(title, startX + pxWidth / 2, y0 + h * 0.38);
                ctx.fillStyle = "#9aa";
                ctx.font = "11px sans-serif";
                ctx.fillText(sub, startX + pxWidth / 2, y0 + h * 0.52);
            }
            ctx.restore();
            return;
        }

        if (this.isGenBlank()) {
            ctx.fillStyle = "#0d0d0d";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(startX + 2, y0 + 4, pxWidth - 4, h - 8);
            ctx.setLineDash([]);
            ctx.fillStyle = "#888";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const fc = seg.frameCount ?? seg.length;
            ctx.fillText(`${fc}f`, startX + pxWidth / 2, y0 + h / 2 - 6);
            ctx.fillStyle = "#555";
            ctx.font = "10px sans-serif";
            ctx.fillText(t("canvas.blankCanvas"), startX + pxWidth / 2, y0 + h / 2 + 8);
            ctx.restore();
            return;
        }

        if (this.isGenImage()) {
            const imgFile = this.isGlobalMode()
                ? this.timeline.global?.genImage?.imageFile
                : (seg.genImage?.imageFile || "");
            ctx.fillStyle = "#111";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            if (imgFile) {
                const cacheKey = `gen:${imgFile}`;
                let img = this._thumbCache.get(cacheKey);
                if (img?.naturalWidth) {
                    const ratio = img.naturalWidth / img.naturalHeight;
                    let dw = pxWidth - 4, dh = dw / ratio;
                    if (dh > h - 4) { dh = h - 4; dw = dh * ratio; }
                    ctx.drawImage(img, startX + (pxWidth - dw) / 2, y0 + (h - dh) / 2, dw, dh);
                } else if (!this._thumbPending.has(cacheKey)) {
                    this._thumbPending.add(cacheKey);
                    const el = new Image();
                    el.crossOrigin = "anonymous";
                    el.onload = () => {
                        this._thumbCache.set(cacheKey, el);
                        this._thumbPending.delete(cacheKey);
                        this.scheduleRender();
                    };
                    el.onerror = () => this._thumbPending.delete(cacheKey);
                    el.src = refViewUrl(imgFile);
                }
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(t("canvas.uploadSourceImage"), startX + pxWidth / 2, y0 + h / 2);
            }
            ctx.restore();
            return;
        }

        ctx.fillStyle = "#000";
        ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
        if (!this.hasVideo()) {
            ctx.fillStyle = "#666";
            ctx.font = "12px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
                this.isFl2vMode() ? t("canvas.clickAddShot") : t("canvas.clickUploadVideo"),
                startX + pxWidth / 2,
                y0 + h / 2,
            );
            ctx.restore();
            return;
        }
        const thumbW = Math.max(32, pxWidth / Math.max(1, Math.min(MAX_THUMBS_PER_SEGMENT, Math.ceil(seg.length / 4))));
        const step = Math.max(1, Math.floor(seg.length / Math.max(1, Math.ceil(pxWidth / thumbW))));
        let drawn = 0;
        for (let f = seg.start; f < seg.start + seg.length && drawn < MAX_THUMBS_PER_SEGMENT; f += step, drawn++) {
            this._queueThumbPrefetch(f);
            const img = this.getFrameImage(f);
            const tx = startX + ((f - seg.start) / seg.length) * pxWidth;
            if (img?.naturalWidth) {
                const ratio = img.naturalWidth / img.naturalHeight;
                let dw = thumbW, dh = thumbW / ratio;
                if (dh > h - 2) { dh = h - 2; dw = dh * ratio; }
                ctx.drawImage(img, tx, y0 + (h - dh) / 2, dw, dh);
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(tx, y0 + 2, Math.max(8, thumbW * 0.6), h - 4);
            }
        }
        ctx.restore();
    }

    _drawSegmentRunCheck(x, y, enabled) {
        const ctx = this.ctx;
        const s = RUN_CHECK_SIZE;
        ctx.save();
        // Opaque plate so the control never blends into timeline chrome.
        ctx.fillStyle = "#0e0e0e";
        ctx.fillRect(x - 1, y - 1, s + 2, s + 2);
        ctx.fillStyle = enabled ? "#1a3a2a" : "#1c1c1c";
        ctx.strokeStyle = enabled ? "#4fff8f" : "#888";
        ctx.lineWidth = 1;
        ctx.fillRect(x, y, s, s);
        ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
        if (enabled) {
            ctx.fillStyle = "#4fff8f";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "alphabetic";
            ctx.fillText("✓", x + 2, y + 11);
        }
        ctx.restore();
    }

    _drawReorderInsertMarker(ix) {
        const ctx = this.ctx;
        const y0 = TRACK_Y;
        const y1 = TRACK_Y + TRACK_H;
        ctx.save();
        ctx.strokeStyle = "#4fff8f";
        ctx.fillStyle = "#4fff8f";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(ix, y0);
        ctx.lineTo(ix, y1);
        ctx.stroke();
        // Triangles at top/bottom
        const t = 7;
        ctx.beginPath();
        ctx.moveTo(ix, y0);
        ctx.lineTo(ix - t, y0 - t);
        ctx.lineTo(ix + t, y0 - t);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(ix, y1);
        ctx.lineTo(ix - t, y1 + t);
        ctx.lineTo(ix + t, y1 + t);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    /** Floating ghost card that follows the pointer while reordering clips. */
    _drawReorderGhost(width, segs, fromRank) {
        if (fromRank < 0 || this._drag?.pointerX == null) return;
        const ordered = this._orderedSegmentsWithRank();
        const item = ordered.find((o) => o.visualRank === fromRank);
        if (!item?.seg) return;
        const seg = item.seg;
        const srcW = Math.max(48, this.frameToX(seg.start + seg.length, width) - this.frameToX(seg.start, width));
        const gw = Math.min(140, Math.max(72, srcW * 0.55));
        const gh = TRACK_H * 0.78;
        const gx = this._drag.pointerX - gw / 2;
        const gy = clamp(this._drag.pointerY - gh / 2, TRACK_Y - 8, TRACK_Y + TRACK_H - gh + 8);
        const ctx = this.ctx;
        ctx.save();
        // Drop shadow
        ctx.fillStyle = "rgba(0,0,0,0.45)";
        ctx.fillRect(gx + 4, gy + 5, gw, gh);
        ctx.globalAlpha = 0.95;
        this.drawSegmentThumbnails(ctx, seg, gx, gw, gy, gh, item.arrayIndex);
        ctx.strokeStyle = "#4fff8f";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(gx + 0.5, gy + 0.5, gw - 1, gh - 1);
        ctx.fillStyle = "rgba(20,40,28,0.9)";
        ctx.fillRect(gx + 4, gy + 4, 44, 16);
        ctx.fillStyle = "#4fff8f";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(t("canvas.dragging"), gx + 8, gy + 12);
        ctx.restore();
    }

    drawPromptOverlay(ctx, seg, startX, pxWidth, y0, h) {
        const prompt = this.getDisplayPrompt(seg);
        if (!prompt || pxWidth < 24) return;
        const overlayH = Math.round(h * 0.22);
        const overlayY = y0 + h - overlayH;
        ctx.save();
        ctx.beginPath();
        ctx.rect(startX, overlayY, pxWidth, overlayH);
        ctx.clip();
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.fillRect(startX, overlayY, pxWidth, overlayH);
        ctx.font = `${Math.min(11, overlayH * 0.55)}px sans-serif`;
        ctx.fillStyle = "#e0e3ed";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        let label = prompt;
        const maxW = pxWidth - 10;
        if (ctx.measureText(label).width > maxW) {
            // Estimate truncation first — avoids O(n) measureText on long prompts.
            const fullW = ctx.measureText(label).width;
            const approx = Math.floor(maxW * label.length / Math.max(1, fullW));
            label = label.slice(0, Math.max(0, approx - 2));
            while (label.length > 0 && ctx.measureText(label + "…").width > maxW) label = label.slice(0, -1);
            while (label.length < prompt.length && ctx.measureText(label + prompt[label.length] + "…").width <= maxW) {
                label += prompt[label.length];
            }
            label += "…";
        }
        ctx.fillText(label, startX + pxWidth / 2, overlayY + overlayH / 2);
        ctx.restore();
    }

    render() {
        if (this.isPlaying) {
            this.renderTimelineOnly();
            return;
        }
        const width = this._measureDrawWidth();
        if (!width) {
            // Host not laid out yet — retry after the next layout pass.
            this.scheduleSettleRender();
            return;
        }
        this._drawWidth = width;
        this._drawTimelineCanvas(width);
        this._updateTimelineDom();
        this._syncStagePreview(this.currentFrame);
    }

    renderTimelineOnly() {
        const width = this._measureDrawWidth()
            || this.node?.size?.[0]
            || 0;
        if (!width) return;
        this._drawWidth = width;
        this._drawTimelineCanvas(width);
        this._syncStagePreview(this.currentFrame);
    }

    _drawTimelineCanvas(width) {
        const height = this.canvasHeight;
        const dpr = window.devicePixelRatio || 1;
        const bw = Math.round(width * dpr);
        const bh = Math.round(height * dpr);
        // Keep bitmap ↔ CSS aspect in lockstep. Mixing getBoundingClientRect (graph-zoom
        // transformed) with width:100% clientWidth used to squash/stretch thumbs.
        if (this.canvas.width !== bw || this.canvas.height !== bh) {
            this.canvas.width = bw;
            this.canvas.height = bh;
        }
        if (this.getTimelineZoom() > 1) {
            this.canvas.style.width = `${Math.round(width)}px`;
        } else if (this.canvas.style.width !== "100%") {
            this.canvas.style.width = "100%";
        }
        this.canvas.style.height = `${height}px`;
        this.canvas.style.maxHeight = `${height}px`;
        this.canvas.style.minHeight = `${height}px`;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.ctx.clearRect(0, 0, width, height);

        const total = this.getTotalFrames();
        const segs = this._previewSegments || this.timeline.segments;

        this.ctx.fillStyle = "#252525";
        this.ctx.fillRect(0, 0, width, RULER_H);
        this.ctx.font = "10px sans-serif";
        this.ctx.textAlign = "left";
        this.ctx.textBaseline = "alphabetic";
        const fl2vSampleN = this.isFl2vMode() ? getFl2vSampleFrames(this) : total;
        // Batch/fl2v: ticks follow user 秒数 so 5.0s clips land on "5".
        // v2v: ticks follow play length (frames / fps).
        const durationSec = this.getRulerDurationSec();
        const spanX = this.isFl2vMode() ? this.frameToX(fl2vSampleN, width) : width;
        const pxPerSec = spanX / Math.max(durationSec, 0.001);
        const majorSec = pickRulerMajorStepSec(pxPerSec);
        const minorSec = pickRulerMinorStepSec(majorSec, pxPerSec);
        const secToX = (s) => (s / Math.max(durationSec, 0.001)) * spanX;
        if (minorSec < majorSec) {
            this.ctx.fillStyle = "#5a5a5a";
            const nMinor = Math.floor(durationSec / minorSec + 1e-9);
            for (let i = 0; i <= nMinor; i++) {
                const s = i * minorSec;
                if (s % majorSec === 0) continue;
                this.ctx.fillRect(secToX(s), RULER_H - 4, 1, 4);
            }
        }
        this.ctx.fillStyle = "#aaa";
        const nMajor = Math.floor(durationSec / majorSec + 1e-9);
        for (let i = 0; i <= nMajor; i++) {
            const s = i * majorSec;
            const x = secToX(s);
            this.ctx.fillRect(x, RULER_H - 7, 1, 7);
            const label = formatRulerTime(s);
            const tw = this.ctx.measureText(label).width;
            if (x + 3 + tw > width - 2 && s > 0) {
                if (x - tw - 2 >= 2) this.ctx.fillText(label, x - tw - 2, 11);
            } else {
                this.ctx.fillText(label, x + 3, 11);
            }
        }
        // Sample-window end marker on ruler (overflow hatch drawn after segments).
        if (this.isFl2vMode() && total > fl2vSampleN) {
            const ox = this.frameToX(fl2vSampleN, width);
            this.ctx.save();
            this.ctx.strokeStyle = "rgba(180,180,180,0.75)";
            this.ctx.lineWidth = 1.5;
            this.ctx.setLineDash([6, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(ox + 0.5, 0);
            this.ctx.lineTo(ox + 0.5, RULER_H + SEG_LABEL_H);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            this.ctx.fillStyle = "#999";
            this.ctx.font = "10px sans-serif";
            this.ctx.textAlign = "left";
            if (width - ox > 64) {
                this.ctx.fillText(t("canvas.beyondSampling"), ox + 6, RULER_H - 3);
            }
            this.ctx.restore();
        }

        // Frame-range labels above each segment (1-based inclusive, e.g. 1-10).
        this.ctx.fillStyle = "#1a1a1a";
        this.ctx.fillRect(0, RULER_H, width, SEG_LABEL_H);
        this.ctx.font = "10px sans-serif";
        this.ctx.textBaseline = "middle";
        for (let i = 0; i < segs.length; i++) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const pxW = Math.max(0, x1 - x0);
            if (pxW < 8 || seg.length <= 0) continue;
            const a = seg.start + 1;
            const b = seg.start + seg.length;
            const rangeText = `${a}-${b}`;
            // v2v / r2v / fl2v: emphasize selected segment label (matches card selection).
            const showSegSel = this.usesBatchTimeline() || this.isFl2vMode()
                || !(this.isImageBatch() || this.isGenMode());
            this.ctx.fillStyle = (showSegSel && i === this.selectedIndex) ? "#eee" : "#9a9a9a";
            let draw = rangeText;
            if (this.ctx.measureText(draw).width > pxW - 6) {
                while (draw.length > 1 && this.ctx.measureText(`${draw}…`).width > pxW - 6) {
                    draw = draw.slice(0, -1);
                }
                draw = draw.length < rangeText.length ? `${draw}…` : draw;
            }
            this.ctx.fillText(draw, x0 + 4, RULER_H + SEG_LABEL_H / 2);
        }

        this.ctx.fillStyle = "#111";
        this.ctx.fillRect(0, TRACK_Y, width, TRACK_H);

        if (!segs.length && (this.isFl2vMode() || this.usesBatchTimeline())) {
            this.ctx.fillStyle = "#666";
            this.ctx.font = "12px sans-serif";
            this.ctx.textAlign = "center";
            this.ctx.textBaseline = "middle";
            this.ctx.fillText(
                this.isR2vBatch()
                    ? t("canvas.clickAddRefGroup")
                    : (this.usesBatchTimeline() ? t("canvas.clickAddPromptGroup") : t("canvas.clickAddShot")),
                width / 2,
                TRACK_Y + TRACK_H / 2,
            );
        }

        const clipBounds = this.usesBatchTimeline() ? [] : this.getClipBoundaries();
        if (clipBounds.length) {
            this.ctx.strokeStyle = "rgba(102,170,255,0.55)";
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 4]);
            for (const b of clipBounds) {
                const bx = this.frameToX(b, width);
                this.ctx.beginPath();
                this.ctx.moveTo(bx, TRACK_Y);
                this.ctx.lineTo(bx, TRACK_Y + TRACK_H);
                this.ctx.stroke();
            }
            this.ctx.setLineDash([]);
        }

        const reordering = this._drag?.kind === "reorder";
        const dragFromRank = reordering ? this._drag.fromRank : -1;
        const dropRank = reordering ? this._reorderDropRank : -1;
        // v2v / r2v / fl2v: selection chrome matches card selected border.
        const showSegSel = this.usesBatchTimeline() || this.isFl2vMode()
            || !(this.isImageBatch() || this.isGenMode());

        for (let i = 0; i < segs.length; i++) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const pxW = x1 - x0;
            const sel = showSegSel && i === this.selectedIndex;
            const running = i === this._runHighlightSeg;
            const runOn = this.isSegmentRunEnabled(i);
            const fl2vStart = !this.isFl2vMode() || !!seg.isStartFrame;
            const visualRank = this._visualRankFromArrayIndex(i);
            const isDragSource = reordering && visualRank === dragFromRank;
            const isDropTarget = reordering && dropRank >= 0 && visualRank === dropRank && dropRank !== dragFromRank;
            if (this.isRunSelectEnabled() && this.getRunnableSegmentCount() >= 2 && fl2vStart && !runOn) {
                this.ctx.globalAlpha = 0.32;
            } else if (isDragSource) {
                this.ctx.globalAlpha = 0.28;
            } else if (reordering && !isDropTarget) {
                this.ctx.globalAlpha = 0.55;
            } else if (this.isFl2vMode() && !seg.isStartFrame) {
                this.ctx.globalAlpha = 0.72;
            }
            this.drawSegmentThumbnails(this.ctx, seg, x0, pxW, TRACK_Y, TRACK_H, i);
            if (!this.isFl2vMode() || seg.isStartFrame) {
                this.drawPromptOverlay(this.ctx, seg, x0, pxW, TRACK_Y, TRACK_H);
            }
            const clipIdx = this.usesBatchTimeline() ? i : this.getSegmentClipIndex(seg);
            const clipColor = CLIP_SEGMENT_COLORS[clipIdx % CLIP_SEGMENT_COLORS.length];
            if (isDropTarget) {
                this.ctx.fillStyle = "rgba(79,255,143,0.14)";
                this.ctx.fillRect(x0, TRACK_Y, pxW, TRACK_H);
                this.ctx.strokeStyle = "#4fff8f";
                this.ctx.lineWidth = 3;
                this.ctx.setLineDash([7, 4]);
                this.ctx.strokeRect(x0 + 1, TRACK_Y + 1, pxW - 2, TRACK_H - 2);
                this.ctx.setLineDash([]);
                this.ctx.fillStyle = "rgba(20,40,28,0.92)";
                const label = this.isFl2vMode() ? t("canvas.swapHere") : t("canvas.insertHere");
                this.ctx.font = "bold 11px sans-serif";
                const tw = this.ctx.measureText(label).width + 12;
                this.ctx.fillRect(x0 + (pxW - tw) / 2, TRACK_Y + 8, tw, 18);
                this.ctx.fillStyle = "#4fff8f";
                this.ctx.textAlign = "center";
                this.ctx.textBaseline = "middle";
                this.ctx.fillText(label, x0 + pxW / 2, TRACK_Y + 17);
            } else {
                this.ctx.strokeStyle = running || sel ? "#4fff8f" : clipColor;
                this.ctx.lineWidth = running ? 3 : sel ? 2.5 : 1.5;
                this.ctx.strokeRect(x0 + 0.5, TRACK_Y + 0.5, pxW - 1, TRACK_H - 1);
                if (sel && !running) {
                    this.ctx.fillStyle = "rgba(79,255,143,0.08)";
                    this.ctx.fillRect(x0 + 1, TRACK_Y + 1, Math.max(0, pxW - 2), TRACK_H - 2);
                }
            }
            if (this.isFl2vMode()) {
                // Hatch the portion past the sampling window (不计入采样).
                const sampleN = getFl2vSampleFrames(this);
                const segEnd = seg.start + seg.length;
                if (segEnd > sampleN && seg.start < segEnd) {
                    const ox0 = this.frameToX(Math.max(seg.start, sampleN), width);
                    const ox1 = this.frameToX(segEnd, width);
                    if (ox1 > ox0 + 1) {
                        this.ctx.save();
                        this.ctx.beginPath();
                        this.ctx.rect(ox0, TRACK_Y + 1, ox1 - ox0, TRACK_H - 2);
                        this.ctx.clip();
                        this.ctx.fillStyle = "rgba(0,0,0,0.45)";
                        this.ctx.fillRect(ox0, TRACK_Y + 1, ox1 - ox0, TRACK_H - 2);
                        this.ctx.strokeStyle = "rgba(200,200,200,0.55)";
                        this.ctx.lineWidth = 1;
                        this.ctx.setLineDash([5, 4]);
                        this.ctx.strokeRect(ox0 + 0.5, TRACK_Y + 1.5, Math.max(0, ox1 - ox0 - 1), TRACK_H - 3);
                        this.ctx.setLineDash([]);
                        this.ctx.restore();
                    }
                }
                this._drawFl2vEdgeHandles(segs, i, x0, x1, width);
            } else {
                this.ctx.fillStyle = "#ffcc00";
                this.ctx.fillRect(x0 - 2, TRACK_Y + TRACK_H / 2 - 12, 4, 24);
                this.ctx.fillRect(x1 - 2, TRACK_Y + TRACK_H / 2 - 12, 4, 24);
            }
            this.ctx.globalAlpha = 1;
            // Checkbox on top-left; drawn last so it stays clear on dimmed segments.
            if (
                this.isRunSelectEnabled()
                && this.getRunnableSegmentCount() >= 2
                && pxW >= RUN_CHECK_SIZE + 8
                && (!this.isFl2vMode() || seg.isStartFrame)
            ) {
                const g = this._runCheckGeometry(seg, width);
                this._drawSegmentRunCheck(g.boxX, g.boxY, runOn);
            }
        }

        // fl2v: dashed overlay for the region past the sampling window.
        if (this.isFl2vMode()) {
            const sampleN = getFl2vSampleFrames(this);
            if (total > sampleN) {
                const ox = this.frameToX(sampleN, width);
                this.ctx.save();
                this.ctx.strokeStyle = "rgba(180,180,180,0.7)";
                this.ctx.lineWidth = 1.5;
                this.ctx.setLineDash([6, 5]);
                this.ctx.beginPath();
                this.ctx.moveTo(ox + 0.5, TRACK_Y);
                this.ctx.lineTo(ox + 0.5, TRACK_Y + TRACK_H);
                this.ctx.stroke();
                this.ctx.strokeRect(ox + 1, TRACK_Y + 1, Math.max(0, width - ox - 2), TRACK_H - 2);
                this.ctx.setLineDash([]);
                this.ctx.restore();
            }
        }

        if (reordering) {
            this._drawReorderGhost(width, segs, dragFromRank);
            if (dropRank >= 0 && dropRank !== dragFromRank && !this.isFl2vMode()) {
                const insertFrame = this._getReorderInsertFrame(dropRank, dragFromRank);
                const ix = this.frameToX(insertFrame, width);
                this._drawReorderInsertMarker(ix);
            }
        }

        // Editable split-point markers: click = select only; delete via toolbar button.
        const splitFrames = this.getEditableSplitFrames();
        if (splitFrames.length) {
            for (const frame of splitFrames) {
                const sx = this.frameToX(frame, width);
                const selected = this.selectedSplitFrame === frame;
                this.ctx.strokeStyle = selected ? "#ffe066" : "rgba(80, 220, 255, 0.95)";
                this.ctx.fillStyle = selected ? "#ffe066" : "rgba(80, 220, 255, 0.9)";
                this.ctx.lineWidth = selected ? 3.5 : 2;
                this.ctx.beginPath();
                this.ctx.moveTo(sx, RULER_H + 2);
                this.ctx.lineTo(sx, TRACK_Y + TRACK_H - 2);
                this.ctx.stroke();
                const cy = RULER_H + SEG_LABEL_H / 2;
                const r = selected ? 8 : 6;
                this.ctx.beginPath();
                this.ctx.moveTo(sx, cy - r);
                this.ctx.lineTo(sx + r, cy);
                this.ctx.lineTo(sx, cy + r);
                this.ctx.lineTo(sx - r, cy);
                this.ctx.closePath();
                this.ctx.fill();
                if (selected) {
                    this.ctx.strokeStyle = "#fff";
                    this.ctx.lineWidth = 1.5;
                    this.ctx.stroke();
                    // Halo so selection is obvious on dense timelines.
                    this.ctx.strokeStyle = "rgba(255, 224, 102, 0.55)";
                    this.ctx.lineWidth = 6;
                    this.ctx.beginPath();
                    this.ctx.moveTo(sx, TRACK_Y);
                    this.ctx.lineTo(sx, TRACK_Y + TRACK_H);
                    this.ctx.stroke();
                }
            }
        }

        this._drawContinuityJoints(width, segs);

        const phx = this.frameToX(this.currentFrame, width);
        this.ctx.strokeStyle = "#ff4444";
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(phx, 0);
        this.ctx.lineTo(phx, height);
        this.ctx.stroke();

        const exportCap = this.getMaxExportFrames();
        const exportTotal = this.getExportFrameTotal();
        if (exportCap > 0 && exportTotal < total) {
            const capX = this.frameToX(exportTotal, width);
            this.ctx.fillStyle = "rgba(0,0,0,0.35)";
            this.ctx.fillRect(capX, TRACK_Y, width - capX, TRACK_H);
            this.ctx.strokeStyle = "#66aaff";
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([4, 3]);
            this.ctx.beginPath();
            this.ctx.moveTo(capX, 0);
            this.ctx.lineTo(capX, height);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            this.ctx.fillStyle = "#66aaff";
            this.ctx.font = "10px sans-serif";
            this.ctx.fillText(t("canvas.exportCap", { n: exportTotal }), capX + 4, TRACK_Y + 12);
        }
    }

    _updateTimelineDom({ skipSeek = false } = {}) {
        const segs = this._previewSegments || this.timeline.segments;
        const totalFrames = Math.max(0, this.getTotalFrames());
        const cur = this.formatTime(this.currentFrame);
        const total = this.formatTime(totalFrames);
        if (this.timecodeEl) this.timecodeEl.textContent = `${cur}s`;
        if (this.playerTimecodeEl) this.playerTimecodeEl.textContent = `${cur} / ${total}`;
        if (this.frameTotalEl) this.frameTotalEl.textContent = String(totalFrames);
        if (this.frameInputEl) {
            this.frameInputEl.max = String(Math.max(1, totalFrames));
            // Don't overwrite while the user is typing a target frame.
            if (document.activeElement !== this.frameInputEl) {
                this.frameInputEl.value = String(totalFrames > 0 ? this.currentFrame + 1 : 1);
            }
        }
        if (!skipSeek && this.seekBar && +this.seekBar.value !== this.currentFrame) {
            this.seekBar.value = this.currentFrame;
        }
        if (this.seekBar) this.seekBar.max = Math.max(0, totalFrames - 1);
        if (this.selectedSplitFrame != null && this.getEditableSplitFrames().includes(this.selectedSplitFrame)) {
            if (this.boundsEl) {
                this.boundsEl.textContent = t("split.boundsEditable", { f: this.selectedSplitFrame });
            }
        } else {
            const seg = segs[this.selectedIndex];
            if (seg && this.boundsEl) {
                this.boundsEl.textContent = t("bounds.range", {
                    start: this.formatTime(seg.start),
                    end: this.formatTime(seg.start + seg.length),
                });
            }
        }
        this.updateSplitPointUI();
    }

    /** Jump to an exact 0-based logical frame; syncs seek bar, preview, playhead. */
    seekToFrame(frame, { fromUi = false } = {}) {
        const total = this.getTotalFrames();
        if (total < 1) return;
        if (this.isPlaying) this._stopPlay();
        const next = clamp(Math.round(Number(frame) || 0), 0, total - 1);
        this.currentFrame = next;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = next;
        }
        this._syncStagePreview(next, { force: true });
        this._updateTimelineDom({ skipSeek: true });
        // Select the segment that contains this frame for editing context.
        const segs = this.timeline.segments || [];
        for (let i = 0; i < segs.length; i++) {
            const s = segs[i];
            if (next >= s.start && next < s.start + s.length) {
                if (this.selectedIndex !== i) {
                    this.selectedIndex = i;
                    this.updateSelectionUI();
                }
                break;
            }
        }
        this.scheduleRender();
        if (fromUi) this._queueThumbPrefetch?.(next);
    }

    stepFrame(delta) {
        const total = this.getTotalFrames();
        if (total < 1) return;
        this.seekToFrame(this.currentFrame + (Number(delta) || 0), { fromUi: true });
    }

    formatTime(frames) { return (frames / this.getFrameRate()).toFixed(2); }

    updateSelectionUI() {
        this.timeline.global = this.timeline.global || { taskType: "", prompt: "", refs: [] };
        if (this.globalTask) this.globalTask.value = this.timeline.global.taskType || "";
        if (this.globalPrompt) this.globalPrompt.value = this.timeline.global.prompt || "";
        this.syncNegativeFromWidget();
        updateFl2vToolbarBtns(this);
        updateR2vToolbarBtns(this);
        if (this.isFl2vMode()) updateFl2vDetailUI(this);
        if (this.isImageBatch()) {
            const shown = this.batchList?.querySelector(".bd-batch-card");
            const shownIdx = shown ? parseInt(shown.dataset.batchIndex, 10) : NaN;
            if (isBatchDetailSolo(this) && shownIdx !== this.selectedIndex) {
                this.renderImageBatchGroups();
            } else {
                this._syncR2vCardSelection();
            }
        }

        const r2vOn = this.isR2vCommonEnabled();
        const hideTimeline = (this.isImageBatch() && !r2vOn) || this.isGenMode();
        const seg = this.usesGlobalRefPanel() ? null : this.timeline.segments[this.selectedIndex];
        this.updateReferenceImageVisibility({ hideTimeline, seg: seg || null });

        if (this.usesGlobalRefPanel() && taskUsesReferenceImages(this.getTaskKey())) {
            this.timeline.global.refs = this.timeline.global.refs || [];
            this.renderRefSlots(this.timeline.global.refs, this.globalRefsBox, true);
        }
        if (this.usesGlobalRefPanel() && taskUsesReferenceAudios(this.getTaskKey())) {
            this.timeline.global.refAudios = this.timeline.global.refAudios || [];
            this.renderRefAudioSlots();
        }
        if (this.usesGlobalRefPanel() && this.usesR2vCommonPanel()) {
            this.timeline.global.refVideos = this.timeline.global.refVideos || [];
            this.renderR2vCommonVideoSlots();
        }
        const refVideoKey = this.usesGlobalRefPanel()
            ? this.getTaskKey()
            : resolveTaskKey(seg?.taskType || this.timeline.global?.taskType || this.getTaskKey());
        if (taskUsesReferenceVideo(refVideoKey)) {
            this.renderRefVideoSlot();
        }
        if (this.isGenImage() && this.isGlobalMode()) {
            this.renderGenSrcSlot(
                this.genGlobalImg,
                this.timeline.global?.genImage?.imageFile,
                t("panel.uploadSourceImage"),
            );
        }
        if (this.isGenMode() && this.isGlobalMode()) {
            const defFc = this.timeline.gen?.defaultFrameCount ?? defaultFrameCount(this.getTaskKey());
            if (this.genDefaultFc) this.genDefaultFc.value = defFc;
        }

        if (this.usesGlobalRefPanel()) {
            this.syncSegmentRefImageSizeUI();
            return;
        }

        if (!seg) return;
        const liveSeg = (this._previewSegments || this.timeline.segments)?.[this.selectedIndex] || seg;
        const segKey = resolveTaskKey(liveSeg.taskType || this.timeline.global?.taskType || this.getTaskKey());
        this.segLabel.textContent = t("panel.segmentN", { n: this.selectedIndex + 1 });
        this.syncSegmentContinuityFromPrevUI();
        this.syncSegmentRefImageSizeUI();
        this._updateSegInfoFromSegment(liveSeg);
        this.segPrompt.value = liveSeg.prompt || "";
        if (taskUsesReferenceImages(segKey)) {
            this.renderRefSlots(liveSeg.refs, this.segRefsBox, false);
        }
        if (taskUsesReferenceAudios(segKey)) {
            this.renderRefAudioSlots();
        }
        if (this.isGenImage() && !this.isGlobalMode()) {
            this.renderGenSrcSlot(this.genSegImg, liveSeg.genImage?.imageFile, t("panel.uploadSegmentSourceImage"));
        }
        if (this.isGenMode() && !this.isGlobalMode()) {
            const fc = liveSeg.frameCount ?? liveSeg.length ?? defaultFrameCount(this.getTaskKey());
            if (this.genSegFc) this.genSegFc.value = fc;
        }
        if (this.isFl2vMode()) updateFl2vDetailUI(this);
    }

    renderRefSlots(refs, box, isGlobal) {
        if (!box) return;
        box.innerHTML = "";
        const target = isGlobal
            ? this.timeline.global
            : this.timeline.segments[this.selectedIndex];
        const taskKey = isGlobal
            ? this.getTaskKey()
            : resolveTaskKey(
                target?.taskType || this.timeline.global?.taskType || this.globalTask?.value || this.getTaskKey(),
            );
        const polished = this.usesRv2vRefStyle(taskKey);
        const wrap = isGlobal ? this.globalRefsImagesWrap : this.segRefsImagesWrap;
        const countEl = isGlobal ? this.globalRefsCount : this.segRefsCount;
        const PIC_STEP = 3;
        const PIC_SLOTS = MAX_REFERENCE_IMAGES;

        let filled = 0;
        let highestFilled = -1;
        for (const r of refs || []) {
            const idx = Number(r.index ?? r.slot);
            const has = !!(r?.imageFile || r?.imageB64);
            if (!has || !Number.isFinite(idx) || idx < 0 || idx >= PIC_SLOTS) continue;
            filled += 1;
            highestFilled = Math.max(highestFilled, idx);
        }
        if (countEl) countEl.textContent = polished ? `${filled}/${PIC_SLOTS}` : "";
        this._syncPickExistingDisabled(
            isGlobal ? '[data-r="global-refs-pick"]' : '[data-r="seg-refs-pick"]',
            filled >= PIC_SLOTS,
        );

        if (!this._rv2vPicsVisible) this._rv2vPicsVisible = {};
        const visKey = isGlobal ? "global" : `seg:${target?.id ?? this.selectedIndex}`;
        const minVisible = highestFilled >= 0
            ? Math.min(PIC_SLOTS, Math.ceil((highestFilled + 1) / PIC_STEP) * PIC_STEP)
            : PIC_STEP;
        let visible = polished
            ? (Number(this._rv2vPicsVisible[visKey]) || PIC_STEP)
            : PIC_SLOTS;
        if (polished) {
            visible = Math.max(PIC_STEP, Math.min(PIC_SLOTS, visible));
            if (visible < minVisible) visible = minVisible;
            this._rv2vPicsVisible[visKey] = visible;
        }

        for (let i = 0; i < PIC_SLOTS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref";
            if (polished && i >= visible) el.classList.add("bd-r2v-pic-hidden");
            el.dataset.refSlot = String(i);
            el.dataset.refKind = "image";
            el.dataset.refIndex = String(i);
            el.dataset.refScope = isGlobal ? "global" : "seg";
            const label = refImageLabel(i);
            el.title = t("ref.slotTitle", { label });
            const ref = (refs || []).find((r) => Number(r.index ?? r.slot) === i);
            const tag = document.createElement("span");
            tag.className = polished ? "cap" : "bd-ref-tag";
            tag.textContent = label;
            el.appendChild(tag);
            if (ref?.imageFile) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = refViewUrl(ref.imageFile);
                img.draggable = false;
                el.appendChild(img);
                if (polished) {
                    const dot = document.createElement("span");
                    dot.className = "dot";
                    el.appendChild(dot);
                }
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(target, i);
                };
                el.appendChild(x);
            } else if (ref?.imageB64) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = ref.imageB64.startsWith("data:") ? ref.imageB64 : `data:image/png;base64,${ref.imageB64}`;
                img.draggable = false;
                el.appendChild(img);
                if (polished) {
                    const dot = document.createElement("span");
                    dot.className = "dot";
                    el.appendChild(dot);
                }
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(target, i);
                };
                el.appendChild(x);
            }
            this._bindRefSlotDnD(el, target, i, isGlobal);
            el.onclick = () => {
                if (this._refDragMoved) {
                    this._refDragMoved = false;
                    return;
                }
                this.pickRef(target, i, isGlobal);
            };
            box.appendChild(el);
        }

        wrap?.querySelectorAll(".bd-r2v-pics-toggle").forEach((btn) => btn.remove());
        if (polished && wrap) {
            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "bd-r2v-pics-toggle";
            const syncToggleLabel = () => {
                if (visible < PIC_SLOTS) {
                    const next = Math.min(PIC_STEP, PIC_SLOTS - visible);
                    toggle.textContent = t("batch.r2v.expandPics", { n: next });
                } else {
                    toggle.textContent = t("batch.r2v.collapsePics");
                }
            };
            syncToggleLabel();
            toggle.onclick = (e) => {
                e.stopPropagation();
                if (visible < PIC_SLOTS) {
                    visible = Math.min(PIC_SLOTS, visible + PIC_STEP);
                } else {
                    visible = Math.max(PIC_STEP, minVisible);
                }
                this._rv2vPicsVisible[visKey] = visible;
                box.querySelectorAll(".bd-ref").forEach((el, i) => {
                    el.classList.toggle("bd-r2v-pic-hidden", i >= visible);
                });
                syncToggleLabel();
                this.updateDomWidgetHeight?.();
            };
            wrap.appendChild(toggle);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    _bindRefSlotDnD(el, target, slotIndex, isGlobal) {
        const hasImg = el.classList.contains("has-img");
        el.draggable = hasImg;
        el.addEventListener("dragstart", (e) => {
            if (!hasImg) {
                e.preventDefault();
                return;
            }
            this._refDragMoved = false;
            const payload = JSON.stringify({
                scope: isGlobal ? "global" : "seg",
                segIndex: isGlobal ? -1 : this.selectedIndex,
                from: slotIndex,
            });
            e.dataTransfer.setData("application/x-minimax-ref-slot", payload);
            e.dataTransfer.setData("text/plain", payload);
            e.dataTransfer.effectAllowed = "move";
        });
        el.addEventListener("dragend", () => {
            // click may fire after dragend; keep suppress for one tick
            setTimeout(() => { this._refDragMoved = false; }, 0);
        });
        el.addEventListener("dragover", (e) => {
            const types = e.dataTransfer?.types || [];
            if (![...types].includes("application/x-minimax-ref-slot") && ![...types].includes("Files")) {
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            e.dataTransfer.dropEffect = [...types].includes("application/x-minimax-ref-slot")
                ? "move"
                : "copy";
        });
        el.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const raw = e.dataTransfer.getData("application/x-minimax-ref-slot")
                || e.dataTransfer.getData("text/plain");
            if (raw) {
                try {
                    const data = JSON.parse(raw);
                    const scope = isGlobal ? "global" : "seg";
                    if (data.scope !== scope) return;
                    if (!isGlobal && data.segIndex !== this.selectedIndex) return;
                    this._refDragMoved = true;
                    this.moveRefSlot(target, Number(data.from), slotIndex, isGlobal);
                    return;
                } catch (_) { /* fall through to file drop */ }
            }
            const f = e.dataTransfer.files?.[0];
            if (f?.type?.startsWith("image/")) {
                this.addRefFromFile(f, target, slotIndex, isGlobal);
            }
        });
    }

    moveRefSlot(target, fromIndex, toIndex, isGlobal) {
        if (!target || fromIndex === toIndex) return;
        const refs = [...(target.refs || [])];
        const fromRef = refs.find((r) => Number(r.index ?? r.slot) === fromIndex);
        if (!fromRef) return;
        const toRef = refs.find((r) => Number(r.index ?? r.slot) === toIndex);
        target.refs = refs.filter((r) => {
            const idx = Number(r.index ?? r.slot);
            return idx !== fromIndex && idx !== toIndex;
        });
        target.refs.push({ ...fromRef, index: toIndex, slot: undefined });
        if (toRef) {
            target.refs.push({ ...toRef, index: fromIndex, slot: undefined });
        }
        if (isGlobal) {
            this.timeline.global = target;
            if (this.isR2vCommonEnabled()) rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
    }

    removeRef(target, index) {
        target.refs = (target.refs || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled() && target === this.timeline.global) {
            rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
    }

    renderRefAudioSlots() {
        const isGlobal = this.usesGlobalRefPanel();
        const box = isGlobal ? this.globalRefAudiosBox : this.segRefAudiosBox;
        if (!box) return;
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [], refAudios: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refAudios = target.refAudios || [];
        const taskKey = isGlobal
            ? this.getTaskKey()
            : resolveTaskKey(
                target?.taskType || this.timeline.global?.taskType || this.globalTask?.value || this.getTaskKey(),
            );
        const polished = this.usesRv2vRefStyle(taskKey);
        const countEl = isGlobal ? this.globalAudiosCount : this.segAudiosCount;
        let filled = 0;
        for (const r of target.refAudios) {
            if (r?.audioFile || r?.fileName) filled += 1;
        }
        if (countEl) countEl.textContent = polished ? `${filled}/${MAX_REFERENCE_AUDIOS}` : "";
        this._syncPickExistingDisabled(
            isGlobal ? '[data-r="global-audios-pick"]' : '[data-r="seg-audios-pick"]',
            filled >= MAX_REFERENCE_AUDIOS,
        );

        box.innerHTML = "";
        for (let i = 0; i < MAX_REFERENCE_AUDIOS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref-audio";
            el.dataset.audioSlot = String(i);
            el.dataset.refKind = "audio";
            el.dataset.refIndex = String(i);
            const label = refAudioLabel(i);
            const ref = (target.refAudios || []).find((r) => Number(r.index ?? r.slot) === i);
            const file = ref?.audioFile || ref?.fileName || "";
            el.title = file
                ? t("ref.audioTitleFilled", { label, file })
                : t("ref.audioTitleEmpty", { label });
            if (polished) {
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
                    el.classList.add("has-audio");
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
                    audio.src = refViewUrl(file);
                    audio.className = "bd-r2v-media";
                    el.appendChild(audio);
                    bindR2vMediaPlayback(audio, playBtn, progress);
                    wireMediaDuration(audio, dur, (sec) => {
                        if (ref) ref.durationSec = sec;
                    });
                    const x = document.createElement("span");
                    x.className = "x";
                    x.textContent = "×";
                    x.onclick = (e) => {
                        e.stopPropagation();
                        this.removeRefAudio(target, i);
                    };
                    el.appendChild(x);
                } else {
                    thumb.textContent = "♪";
                    const hint = document.createElement("span");
                    hint.className = "name";
                    hint.textContent = t("batch.r2v.uploadHint");
                    meta.appendChild(hint);
                }
            } else if (file) {
                el.classList.add("has-audio");
                const tag = document.createElement("span");
                tag.textContent = label;
                el.appendChild(tag);
                const name = document.createElement("span");
                name.className = "bd-ref-audio-name";
                name.textContent = file.split("/").pop() || file;
                el.appendChild(name);
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRefAudio(target, i);
                };
                el.appendChild(x);
            } else {
                el.textContent = t("ref.audioUpload", { label });
            }
            el.onclick = (e) => {
                if (e.target?.closest?.(".bd-r2v-play, .bd-r2v-progress, .x")) return;
                this.pickRefAudio(target, i);
            };
            box.appendChild(el);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    removeRefAudio(target, index) {
        if (!target) return;
        target.refAudios = (target.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled() && target === this.timeline.global) {
            rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
        this.renderRefAudioSlots();
    }

    pickRefAudio(target, index) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefAudioFromFile(file, target, index);
        };
        input.click();
    }

    async addRefAudioFromFile(file, target, slotIndex = null) {
        if (!target || !file) return;
        target.refAudios = target.refAudios || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_AUDIOS }, (_, i) => i)
                .find((i) => !target.refAudios.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const prepared = await prepareLocalReferenceAudio(file);
            const relPath = prepared.relPath;
            if (hasDuplicateReferenceAudio(target.refAudios, relPath, index)) {
                alert(t("ref.audioDuplicate"));
                return;
            }
            target.refAudios = target.refAudios.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refAudios.push({
                index,
                audioFile: relPath,
                fileName: prepared.fileName || file.name,
                type: prepared.type || "input",
                subfolder: prepared.subfolder || "",
            });
            if (this.isR2vCommonEnabled() && target === this.timeline.global) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderRefAudioSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] ref audio upload failed:", err);
            alert(t("upload.refAudioFailed", { err: err?.message || err }));
        }
    }

    /** r2v common panel: multi-slot global.refVideos (1–3), merged into groups at run time. */
    renderR2vCommonVideoSlots() {
        const box = this.globalRefVideosBox;
        if (!box || !this.usesR2vCommonPanel()) return;
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        let filled = 0;
        for (const r of target.refVideos) {
            if (r?.videoFile || r?.fileName || r?.previewImageFile || r?.previewImageUrl || r?.linked) {
                filled += 1;
            }
        }
        if (this.globalVideosCount) {
            this.globalVideosCount.textContent = `${filled}/${MAX_REFERENCE_VIDEOS}`;
        }
        this._syncPickExistingDisabled('[data-r="global-videos-pick"]', filled >= MAX_REFERENCE_VIDEOS);
        box.innerHTML = "";
        for (let i = 0; i < MAX_REFERENCE_VIDEOS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref-video";
            el.dataset.videoSlot = String(i);
            el.dataset.refKind = "video";
            el.dataset.refIndex = String(i);
            const label = refVideoLabel(i);
            const ref = (target.refVideos || []).find((r) => Number(r.index ?? r.slot) === i);
            const file = ref?.videoFile || "";
            const posterSrc = ref?.previewImageUrl
                || (ref?.previewImageFile ? refViewUrl(ref.previewImageFile) : "");
            const hasMedia = !!(file || posterSrc || ref?.linked);
            const titleFile = file || ref?.fileName || ref?.previewImageFile || "";
            el.title = hasMedia
                ? t("ref.videoTitleFilled", { label, file: titleFile || label })
                : t("ref.videoTitleEmpty", { label });
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
                el.classList.add("has-video");
                const video = document.createElement("video");
                video.preload = "metadata";
                video.muted = true;
                video.playsInline = true;
                video.src = refViewUrl(file);
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
                playBtn.addEventListener("click", () => { video.muted = false; });
                wireMediaDuration(video, dur, (sec) => {
                    if (ref) ref.durationSec = sec;
                });
                video.addEventListener("loadeddata", () => {
                    if (video.readyState >= 2 && video.currentTime < 0.05) {
                        try {
                            video.currentTime = Math.min(0.1, (video.duration || 1) * 0.05);
                        } catch (_) { /* ignore */ }
                    }
                }, { once: true });
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeR2vCommonVideo(i);
                };
                el.appendChild(x);
            } else if (posterSrc) {
                el.classList.add("has-video");
                const img = document.createElement("img");
                img.className = "bd-r2v-media";
                img.src = posterSrc;
                img.alt = label;
                thumb.appendChild(img);
                const hint = document.createElement("span");
                hint.className = "name";
                hint.textContent = t("batch.r2v.externalPoster");
                meta.appendChild(hint);
            } else {
                thumb.textContent = "▶";
                const hint = document.createElement("span");
                hint.className = "name";
                hint.textContent = t("batch.r2v.uploadHint");
                meta.appendChild(hint);
            }
            el.onclick = (e) => {
                if (e.target?.closest?.(".bd-r2v-play, .bd-r2v-dur, .x, video")) return;
                if (file && e.target?.closest?.(".bd-r2v-thumb")) {
                    el.querySelector(".bd-r2v-play")?.click();
                    return;
                }
                this.pickR2vCommonVideo(i);
            };
            box.appendChild(el);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    removeR2vCommonVideo(index) {
        const target = this.timeline.global;
        if (!target) return;
        target.refVideos = (target.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled()) {
            rebaseR2vGroupSlotsForCommon(this);
            this.renderImageBatchGroups?.();
        }
        this.commit();
        this.renderR2vCommonVideoSlots();
    }

    pickR2vCommonVideo(index) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*,.mp4,.mov,.webm,.mkv";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addR2vCommonVideoFromFile(file, index);
        };
        input.click();
    }

    async pickExistingR2vCommonVideo() {
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        const index = Array.from({ length: MAX_REFERENCE_VIDEOS }, (_, i) => i)
            .find((i) => !target.refVideos.some((r) => Number(r.index ?? r.slot) === i && (r.videoFile || r.fileName)));
        if (index == null) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseVideoInput({
                title: t("mediaPicker.pickReferenceVideo"),
            });
            if (!picked?.relPath) return;
            target.refVideos = target.refVideos.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refVideos.push({
                index,
                videoFile: picked.relPath,
                fileName: picked.fileName || picked.relPath,
                type: picked.type || "input",
                subfolder: picked.subfolder || "",
            });
            if (this.isR2vCommonEnabled()) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderR2vCommonVideoSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] common ref video pick failed:", err);
            alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        }
    }

    async addR2vCommonVideoFromFile(file, slotIndex = null) {
        if (!file) return;
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_VIDEOS }, (_, i) => i)
                .find((i) => !target.refVideos.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInputSmart(file);
            const relPath = videoRelativePath(uploaded);
            target.refVideos = target.refVideos.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refVideos.push({
                index,
                videoFile: relPath,
                fileName: uploaded?.name || file.name,
                type: "input",
                subfolder: uploaded?.subfolder || "",
            });
            if (this.isR2vCommonEnabled()) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderR2vCommonVideoSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] common ref video upload failed:", err);
            alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        }
    }

    pickRef(target, index, isGlobal) {
        const input = document.createElement("input");
        input.type = "file"; input.accept = "image/*";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefFromFile(file, target, index, isGlobal);
        };
        input.click();
    }

    _nextEmptyMediaSlot(items, max, hasFn) {
        for (let i = 0; i < max; i++) {
            const hit = (items || []).find((r) => Number(r.index ?? r.slot) === i);
            if (!hasFn(hit)) return i;
        }
        return -1;
    }

    _syncPickExistingDisabled(selector, disabled) {
        const btn = this.root?.querySelector(selector);
        if (!btn) return;
        btn.disabled = !!disabled;
        btn.title = disabled ? t("mediaPicker.slotsFull") : t("mediaPicker.pickExistingHint");
    }

    async pickExistingRef(isGlobal) {
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refs = target.refs || [];
        const index = this._nextEmptyMediaSlot(
            target.refs,
            MAX_REFERENCE_IMAGES,
            (r) => !!(r?.imageFile || r?.imageB64),
        );
        if (index < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseImageInput({
                title: t("mediaPicker.pickReferenceImage"),
            });
            if (!picked?.imageFile) return;
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: picked.imageFile, imageB64: "" });
            if (isGlobal) {
                this.timeline.global = target;
                if (this.isR2vCommonEnabled()) {
                    rebaseR2vGroupSlotsForCommon(this);
                    this.renderImageBatchGroups?.();
                }
            }
            this.commit();
            this.renderRefSlots(
                target.refs,
                isGlobal ? this.globalRefsBox : this.segRefsBox,
                isGlobal,
            );
        } catch (err) {
            console.error("[MiniMax H3Director] ref pick failed:", err);
        }
    }

    async pickExistingRefAudio(isGlobal) {
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [], refAudios: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refAudios = target.refAudios || [];
        const index = this._nextEmptyMediaSlot(
            target.refAudios,
            MAX_REFERENCE_AUDIOS,
            (r) => !!(r?.audioFile || r?.fileName),
        );
        if (index < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseAudioInput({
                title: t("mediaPicker.pickReferenceAudio"),
            });
            if (!picked?.relPath) return;
            if (hasDuplicateReferenceAudio(target.refAudios, picked.relPath, index)) {
                alert(t("ref.audioDuplicate"));
                return;
            }
            target.refAudios = target.refAudios.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refAudios.push({
                index,
                audioFile: picked.relPath,
                fileName: picked.fileName || picked.relPath,
                type: picked.type || "input",
                subfolder: picked.subfolder || "",
            });
            if (this.isR2vCommonEnabled() && isGlobal) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderRefAudioSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] ref audio pick failed:", err);
            alert(t("upload.refAudioFailed", { err: err?.message || err }));
        }
    }

    async addRefFromFile(file, target, slotIndex = null, isGlobal = null) {
        target.refs = target.refs || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_IMAGES }, (_, i) => i)
                .find((i) => !target.refs.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInput(file);
            const relPath = videoRelativePath(uploaded);
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: relPath, imageB64: "" });
            if (isGlobal) {
                this.timeline.global = target;
                if (this.isR2vCommonEnabled()) {
                    rebaseR2vGroupSlotsForCommon(this);
                    this.renderImageBatchGroups?.();
                }
            }
            this.commit();
        } catch (err) {
            console.error("[MiniMax H3Director] ref upload failed:", err);
        }
    }

    onGlobalField(field, value) {
        this.timeline.global = this.timeline.global || { refs: [] };
        const taskTypeChanged = field === "taskType";
        if (taskTypeChanged) {
            const prevTaskKey = this._taskKey || resolveTaskKey(this.timeline.global?.taskType || "");
            this.timeline.global[field] = value;
            const prevMode = this._directorMode || "video";
            if (this.globalTask && this.globalTask.value !== value) this.globalTask.value = value;
            if (this.taskTypeWidget) this.taskTypeWidget.value = value;
            if (prevTaskKey === "ads2v" && resolveTaskKey(value) !== "ads2v") {
                this._stopRefVideoPreviews();
            }
            this.applyTaskLayout(prevMode, prevTaskKey);
            this.updateSegmentContinuityUI();
        } else {
            this.timeline.global[field] = value;
        }
        if (field === "prompt" && this.globalPromptWidget) this.globalPromptWidget.value = value;
        this.scheduleTimelineSync();
        if (taskTypeChanged) {
            // The Director mode selector is a custom DOM control, so assigning the
            // hidden task_type widget does not fire LiteGraph onWidgetChanged.
            // Notify the linked Refine after the task workspace has been restored;
            // its status request flushes timeline_data before comparing cache meta.
            this.node?._mmxRefreshFirstPassCache?.(0);
        }
        if (field === "prompt") this._schedulePromptRender();
        else this.scheduleRender();
    }

    /** Debounced render for prompt typing — avoids full canvas redraw on every keystroke. */
    _schedulePromptRender() {
        if (this._promptRenderTimer != null) return;
        this._promptRenderTimer = setTimeout(() => {
            this._promptRenderTimer = null;
            this.scheduleRender();
        }, 160);
    }

    onSegField(field, value) {
        const seg = this.timeline.segments[this.selectedIndex];
        if (!seg) return;
        seg[field] = value;
        this.scheduleTimelineSync();
        this._schedulePromptRender();
    }

    onNegativePrompt(value) {
        if (this.negativePromptWidget) this.negativePromptWidget.value = value;
        if (this.globalNegative && this.globalNegative.value !== value) this.globalNegative.value = value;
        if (this.segNegative && this.segNegative.value !== value) this.segNegative.value = value;
        this._markNodeDirtyLight();
    }

    toggleLoop() {
        this.isLooping = !this.isLooping;
        const btn = this.root.querySelector('[data-a="loop"]');
        btn?.classList.toggle("active", this.isLooping);
        this.refreshLoopButtonTitle();
    }

    refreshLoopButtonTitle() {
        const btn = this.root?.querySelector('[data-a="loop"]');
        if (!btn) return;
        btn.title = this.isLooping ? t("player.loopEnabled") : t("player.loopOff");
        btn.removeAttribute("data-i18n-title");
    }

    isLiveTaePreviewEnabled() {
        return this.timeline?.liveTaePreview === true;
    }

    /** fl2v / v2v / rv2v (and aliases): show dedicated live-sample panel when toggle is on. */
    needsLiveSamplePanel() {
        if (!this.isLiveTaePreviewEnabled()) return false;
        if (this.isImageBatch?.()) return false;
        if (this.isFl2vMode?.()) return true;
        const key = this.getTaskKey?.() || "";
        return key === "v2v" || key === "mv2v" || key === "ads2v"
            || key === "rv2v" || key === "vrc2v" || key === "vi2v";
    }

    toggleLiveTaePreview() {
        this.timeline.liveTaePreview = !this.isLiveTaePreviewEnabled();
        this.refreshLiveTaePreviewButton();
        this.updateLiveSamplePanel();
        this.scheduleTimelineSync();
        this.updateDomWidgetHeight?.();
        syncDirectorNodeSize(this.node, this);
    }

    refreshLiveTaePreviewButton() {
        const btn = this.root?.querySelector('[data-a="live-tae-preview"]');
        if (!btn) return;
        const on = this.isLiveTaePreviewEnabled();
        btn.classList.toggle("active", on);
        btn.textContent = t("toolbar.liveTaePreview");
        btn.title = on ? t("tooltip.liveTaePreviewOn") : t("tooltip.liveTaePreviewOff");
        btn.setAttribute("data-i18n", "toolbar.liveTaePreview");
        btn.removeAttribute("data-i18n-title");
    }

    _clearEmbeddedLiveLayoutClasses() {
        this.globalPromptLayout?.classList.remove("bd-v2v-with-live", "bd-rv2v-with-live");
        this.segPromptLayout?.classList.remove("bd-v2v-with-live", "bd-rv2v-with-live");
    }

    _activePromptLayout() {
        return this.isGlobalMode?.() ? this.globalPromptLayout : this.segPromptLayout;
    }

    _placeLiveSamplePanel() {
        const panel = this.liveSampleEl;
        if (!panel) return;

        if (this.isFl2vMode?.() && this.fl2vUi?.workbench && this.fl2vUi?.shotsEl) {
            this._clearEmbeddedLiveLayoutClasses();
            if (this._liveSampleHost !== "fl2v" || panel.parentElement !== this.fl2vUi.workbench) {
                this.fl2vUi.workbench.insertBefore(panel, this.fl2vUi.shotsEl);
                this._liveSampleHost = "fl2v";
            }
            return;
        }

        const layout = this._activePromptLayout();

        // v2v: preview sits to the right of the prompt column.
        if (this.usesV2vPromptStyle?.() && this.isLiveTaePreviewEnabled() && layout) {
            if (panel.parentElement !== layout) layout.appendChild(panel);
            this.globalPromptLayout?.classList.toggle("bd-v2v-with-live", layout === this.globalPromptLayout);
            this.segPromptLayout?.classList.toggle("bd-v2v-with-live", layout === this.segPromptLayout);
            this.globalPromptLayout?.classList.remove("bd-rv2v-with-live");
            this.segPromptLayout?.classList.remove("bd-rv2v-with-live");
            this._liveSampleHost = "v2v";
            return;
        }

        // rv2v: preview under the prompt (same stack as r2v right column).
        if (this.usesRv2vRefStyle?.() && this.isLiveTaePreviewEnabled() && layout) {
            const promptCol = layout.querySelector(".bd-prompt-col");
            if (promptCol) {
                if (panel.parentElement !== promptCol) promptCol.appendChild(panel);
                this.globalPromptLayout?.classList.toggle("bd-rv2v-with-live", layout === this.globalPromptLayout);
                this.segPromptLayout?.classList.toggle("bd-rv2v-with-live", layout === this.segPromptLayout);
                this.globalPromptLayout?.classList.remove("bd-v2v-with-live");
                this.segPromptLayout?.classList.remove("bd-v2v-with-live");
                this._liveSampleHost = "rv2v";
                return;
            }
        }

        this._clearEmbeddedLiveLayoutClasses();
        if (this.outputBarEl && (this._liveSampleHost !== "main" || panel.parentElement !== this.mainBody)) {
            this.outputBarEl.insertAdjacentElement("afterend", panel);
            this._liveSampleHost = "main";
        }
    }

    updateLiveSamplePanel() {
        const panel = this.liveSampleEl;
        if (!panel) return;
        const show = this.needsLiveSamplePanel();
        this._placeLiveSamplePanel();
        panel.classList.toggle("hidden", !show);
        if (!show) {
            panel.classList.remove("receiving");
            return;
        }
        if (!this._liveSampleB64) {
            this.liveSampleImg?.classList.add("hidden");
            this.liveSampleEmpty?.classList.remove("hidden");
            this.liveSampleBadge?.classList.add("hidden");
            if (this.liveSampleMeta) this.liveSampleMeta.textContent = t("liveSample.idleHint");
        }
    }

    clearLiveSamplePreview() {
        this._liveSampleB64 = "";
        this._liveSampleStep = null;
        this._liveSampleTotal = null;
        this._liveSampleSeg = null;
        this.liveSampleEl?.classList.remove("receiving");
        if (this.liveSampleImg) {
            this.liveSampleImg.removeAttribute("src");
            this.liveSampleImg.classList.add("hidden");
        }
        this.liveSampleEmpty?.classList.remove("hidden");
        this.liveSampleBadge?.classList.add("hidden");
        if (this.liveSampleMeta) this.liveSampleMeta.textContent = t("liveSample.idleHint");
    }

    setLiveSamplePreview(detail = {}) {
        if (!this.needsLiveSamplePanel()) return;
        const b64 = detail.image_b64 || detail.imageB64 || "";
        if (!b64) return;
        this._placeLiveSamplePanel();
        this.liveSampleEl?.classList.remove("hidden");
        this._liveSampleB64 = b64;
        this._liveSampleStep = detail.step ?? null;
        this._liveSampleTotal = detail.total_steps ?? detail.totalSteps ?? null;
        this._liveSampleSeg = detail.segment_index ?? detail.segmentIndex ?? null;

        const src = b64.startsWith("data:") ? b64 : `data:image/jpeg;base64,${b64}`;
        if (this.liveSampleImg) {
            this.liveSampleImg.src = src;
            this.liveSampleImg.classList.remove("hidden");
        }
        this.liveSampleEmpty?.classList.add("hidden");
        this.liveSampleEl?.classList.toggle("receiving", !!detail.live);

        const step = this._liveSampleStep;
        const total = this._liveSampleTotal;
        const seg = this._liveSampleSeg;
        let badge = "";
        if (step && total) badge = t("batch.generatingStep", { step, total });
        else if (detail.live) badge = t("batch.generating");
        if (this.liveSampleBadge) {
            this.liveSampleBadge.textContent = badge;
            this.liveSampleBadge.classList.toggle("hidden", !badge);
        }
        if (this.liveSampleMeta) {
            const unit = this.isFl2vMode?.() ? t("unit.shot") : t("unit.segment");
            const segLabel = (seg != null && seg !== "")
                ? t("liveSample.segmentHint", { unit, n: Number(seg) + 1 })
                : "";
            this.liveSampleMeta.textContent = detail.live
                ? (segLabel || t("liveSample.sampling"))
                : (segLabel || t("liveSample.done"));
        }
    }

    /** Keep timeline / 素材组 selection on the segment the run is currently on. */
    _followRunSelection(timelineSeg1Based) {
        const segs = this.timeline?.segments || [];
        if (!segs.length) return;
        const idx = clamp(Math.round(Number(timelineSeg1Based) || 1) - 1, 0, segs.length - 1);
        if (this.selectedIndex === idx) return;
        this.selectedIndex = idx;
        this.updateSelectionUI();
        const seg = segs[idx];
        if (seg && Number.isFinite(seg.start) && !this.isPlaying) {
            const f = Math.max(0, Math.round(Number(seg.start) || 0));
            this.currentFrame = f;
            if (this.seekBar) this.seekBar.value = f;
            this._syncStagePreview?.(f, { force: true });
        }
        this.scheduleRender();
    }

    setRunProgress(detail) {
        if (!this.runStatusEl) return;
        const timelineTotal = this.timeline?.segments?.length || 0;
        const runTotal = Math.max(detail.segment_total || this.getRunProgressSegmentTotal(), 1);
        const runSeg = Math.max(1, detail.segment || 1);
        const timelineSeg = detail.timeline_segment ?? runSeg;
        const partialRun = !!detail.partial_run
            || (this.isRunSelectEnabled?.() && runTotal < timelineTotal);
        const phaseLabel = detail.phase_label || detail.phase || t("run.phase.default");
        const overallPct = detail.overall_max > 0
            ? Math.round((100 * detail.overall_value) / detail.overall_max)
            : 0;
        const phasePct = detail.phase_max > 0
            ? Math.round((100 * detail.phase_value) / detail.phase_max)
            : 0;
        const remain = Math.max(0, runTotal - runSeg);

        if (detail.phase === "finish") {
            this.runStatusEl.className = "bd-run-status done";
            this.runTitleEl.textContent = t("run.titleDone");
            this.runDetailEl.textContent = runTotal
                ? (this.isImageBatch()
                    ? (isVideoBatchTask(this.getTaskKey())
                        ? t("run.detailDoneVideos", { n: runTotal })
                        : t("run.detailDoneImages", { n: runTotal }))
                    : (partialRun
                        ? t("run.detailDoneSegmentsPartial", { n: runTotal })
                        : t("run.detailDoneSegments", { n: runTotal })))
                : t("run.detailDoneGeneric");
            this.runOverallEl.style.width = "100%";
            this.runPhaseEl.style.width = "100%";
            this._runHighlightSeg = -1;
            this._runProgressSegKey = null;
            this._followRunSelection(timelineSeg);
            this.updateRunSelectUI();
            if (this.isImageBatch()) this.renderImageBatchGroups();
            else this.scheduleRender();
            return;
        }

        this.runStatusEl.className = "bd-run-status active";
        // Hide the pre-run "将运行 N 段" chip while progress is live — it sits
        // under the title in the same green accent and reads as a layout glitch.
        this.runSelectBar?.classList.add("hidden");
        this._runHighlightSeg = timelineSeg - 1;
        this._followRunSelection(timelineSeg);
        let title;
        if (detail.phase === "plan") {
            title = runTotal > 1 ? t("run.titlePlanning", { n: runTotal, phase: phaseLabel }) : phaseLabel;
        } else if (this.isImageBatch()) {
            // Partial run: show timeline card number (e.g. group 4), not compact run order.
            title = partialRun
                ? t("run.titleBatchGroupPartial", {
                    timeline: timelineSeg, i: runSeg, n: runTotal, phase: phaseLabel,
                })
                : t("run.titleBatchGroup", { i: runSeg, n: runTotal, phase: phaseLabel });
        } else if (partialRun) {
            title = t("run.titleSegmentPartial", { timeline: timelineSeg, i: runSeg, n: runTotal, phase: phaseLabel });
        } else {
            title = t("run.titleSegment", { i: runSeg, n: runTotal, phase: phaseLabel });
        }
        if (phasePct > 0 && detail.phase !== "plan") {
            title += ` · ${phasePct}%`;
        }
        this.runTitleEl.textContent = title;
        const parts = [];
        if (detail.frames_label) parts.push(detail.frames_label);
        if (detail.task_key) parts.push(detail.task_key);
        parts.push(t("run.detailOverall", { pct: overallPct }));
        if (runTotal > 1) {
            parts.push(this.isImageBatch()
                ? t("run.detailRemainingGroups", { n: remain })
                : t("run.detailRemainingSegments", { n: remain }));
        }
        if (partialRun && timelineTotal > runTotal) {
            parts.push(t("run.detailTimelineTotal", { n: timelineTotal }));
        }
        this.runDetailEl.textContent = parts.join(" · ");
        this.runOverallEl.style.width = `${overallPct}%`;
        this.runPhaseEl.style.width = `${phasePct}%`;
        // Do NOT syncDirectorNodeSize / full batch rebuild every tick — that was the
        // cross-mode (t2v/i2v/r2v/…) infinite-height feedback loop. Status has a fixed
        // min-height; live previews patch in place via minimax_director_preview.
        const segKey = `${timelineSeg}|${detail.phase}|${runSeg}`;
        const segChanged = this._runProgressSegKey !== segKey;
        this._runProgressSegKey = segKey;
        if (this.isImageBatch()) {
            this._syncBatchRunHighlight();
            if (segChanged) healOversizedDirectorNode(this.node, this);
        } else if (segChanged) {
            this.scheduleRender();
            healOversizedDirectorNode(this.node, this);
        }
    }

    clearRunProgress(title, detail) {
        if (!this.runStatusEl) return;
        this.runStatusEl.className = "bd-run-status idle";
        this.runTitleEl.textContent = title || t("run.titleIdle");
        this.runDetailEl.textContent = detail || t("run.detailIdle");
        this.runOverallEl.style.width = "0%";
        this.runPhaseEl.style.width = "0%";
        this._runHighlightSeg = -1;
        this._runProgressSegKey = null;
        this.updateRunSelectUI();
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    setRunError(message) {
        if (!this.runStatusEl) return;
        this.runStatusEl.className = "bd-run-status error";
        this.runTitleEl.textContent = t("run.titleError");
        this.runDetailEl.textContent = message || t("run.detailError");
        if (this.runOverallEl) this.runOverallEl.style.width = "0%";
        if (this.runPhaseEl) this.runPhaseEl.style.width = "0%";
        this._runHighlightSeg = -1;
        this._runProgressSegKey = null;
        this.updateRunSelectUI();
        this.scheduleRender();
    }

    _stopPlay() {
        this.isPlaying = false;
        this._playHandoff = false;
        this._nativePlayFailed = false;
        this._pauseSettling = true;
        cancelAnimationFrame(this._playRaf);
        this._playRaf = null;
        this.stageVideo?.pause();
        this.root.querySelector('[data-a="play"]').textContent = "▶";
        this._resizeObserver?.disconnect();

        const w = this._playCanvasWidth;
        this._releasePlayLayoutLock();

        if (w) this._drawTimelineCanvas(w);
        this._updateTimelineDom({ skipSeek: true });
        this._syncStagePreview(this.currentFrame, { force: true });

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                if (+this.seekBar.value !== this.currentFrame) {
                    this.seekBar.value = this.currentFrame;
                }
                this._observeViewportResize();
                const drawW = this._measureDrawWidth() || this.viewport?.clientWidth || w;
                if (drawW) this._drawTimelineCanvas(drawW);
                this._syncStagePreview(this.currentFrame, { force: true });
                this._pauseSettling = false;
            });
        });
    }

    async _beginNativePlay() {
        const total = this.getTotalFrames();
        if (total < 1) return;
        if (this.currentFrame >= total) this.currentFrame = 0;
        await this._ensureStageReadyForFrame(this.currentFrame);
        if (!this.isPlaying) return;
        const v = this.stageVideo;
        if (!v) return;
        try {
            await v.play();
        } catch {
            // Native play blocked/failed — keep isPlaying but drive via frame clock.
            this._nativePlayFailed = true;
        }
    }

    async _advanceNativePlayToNextClipOrEnd() {
        if (this._playHandoff || !this.isPlaying) return;
        this._playHandoff = true;
        try {
            const total = this.getTotalFrames();
            const range = this._logicalRangeForClip(this._stageClipIndex);
            const next = range.end < total ? range.end : -1;
            if (next >= 0) {
                this.currentFrame = next;
                await this._beginNativePlay();
                return;
            }
            if (this.isLooping) {
                this.currentFrame = 0;
                await this._beginNativePlay();
                return;
            }
            this.currentFrame = Math.max(0, total - 1);
            this._stopPlay();
        } finally {
            this._playHandoff = false;
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this._stopPlay();
            return;
        }
        const total = this.getTotalFrames();
        if (total < 1) return;

        this.isPlaying = true;
        this._nativePlayFailed = false;
        this.root.querySelector('[data-a="play"]').textContent = "⏸";
        this._lockPlayLayout();
        this._resizeObserver?.disconnect();

        if (this.currentFrame >= total) this.currentFrame = 0;
        this.renderTimelineOnly();
        this._updateTimelineDom();

        const useNative = !this._legacyFrames.length && !!this.stageVideo;
        if (useNative) {
            this._beginNativePlay();
        } else {
            this._syncStagePreview(this.currentFrame, { force: true });
        }

        const tick = () => {
            if (!this.isPlaying) return;
            const fps = Math.max(0.001, this.getFrameRate());

            if (useNative && this.stageVideo && !this._nativePlayFailed) {
                const v = this.stageVideo;
                const clipIndex = this._stageClipIndex >= 0 ? this._stageClipIndex : 0;
                const range = this._logicalRangeForClip(clipIndex);
                const lastLogical = Math.max(range.start, range.end - 1);
                const lastTime = this.getFrameMapEntry(lastLogical).frame / fps;
                const atMappedEnd = v.currentTime >= Math.max(0, lastTime - 0.04);
                const hasTimelineEdits = !!(
                    this.getFrameMap().length
                    || deletedSourceRanges(this.timeline.video || {}).length
                );
                // With deletes, file duration still includes removed tails — trust mapped end.
                const atMediaEnd = !hasTimelineEdits && (
                    v.ended || (v.duration > 0 && v.currentTime >= v.duration - 0.04)
                );

                if ((atMappedEnd || atMediaEnd) && !v.seeking && !this._playHandoff) {
                    this.currentFrame = lastLogical;
                    this.renderTimelineOnly();
                    this._updateTimelineDom();
                    this._advanceNativePlayToNextClipOrEnd();
                    if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                    return;
                }

                if (!v.paused) {
                    const srcFrame = Math.max(0, Math.round(v.currentTime * fps));
                    let logical = this._logicalFromStageTime(clipIndex, v.currentTime);
                    const jumpToKept = () => {
                        const nextLogical = this._nextLogicalAfterSourceFrame(clipIndex, srcFrame);
                        if (nextLogical >= 0) {
                            const nextSrc = this.getFrameMapEntry(nextLogical).frame;
                            try { v.currentTime = nextSrc / fps; } catch { /* seek race */ }
                            return nextLogical;
                        }
                        return -1;
                    };
                    // Sparse deleted gap, or mid/leading gap vs mapped source.
                    if (logical < 0) {
                        const next = jumpToKept();
                        if (next < 0) {
                            this.currentFrame = lastLogical;
                            this.renderTimelineOnly();
                            this._updateTimelineDom();
                            this._advanceNativePlayToNextClipOrEnd();
                            if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                            return;
                        }
                        logical = next;
                    } else {
                        const mapped = this.getFrameMapEntry(logical);
                        if (mapped.clip === clipIndex && mapped.frame !== srcFrame) {
                            // leading gap (mapped > src) or mid gap (mapped < src)
                            if (mapped.frame > srcFrame || mapped.frame < srcFrame) {
                                const next = mapped.frame > srcFrame ? logical : jumpToKept();
                                if (next < 0) {
                                    this.currentFrame = clamp(logical, 0, total - 1);
                                    this.renderTimelineOnly();
                                    this._updateTimelineDom();
                                    this._advanceNativePlayToNextClipOrEnd();
                                    if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                                    return;
                                }
                                if (mapped.frame > srcFrame) {
                                    try { v.currentTime = mapped.frame / fps; } catch { /* seek race */ }
                                }
                                logical = next;
                            }
                        }
                    }
                    this.currentFrame = clamp(logical, 0, total - 1);
                    this.renderTimelineOnly();
                    const now = performance.now();
                    if (now - this._lastSeekUiMs > 66) {
                        this._updateTimelineDom();
                        this._lastSeekUiMs = now;
                    }
                }
            } else {
                // Legacy embedded frames (or native play unavailable): step by logical frame.
                this.currentFrame += 1;
                if (this.currentFrame >= total) {
                    if (this.isLooping) this.currentFrame = 0;
                    else {
                        this.currentFrame = total - 1;
                        this._stopPlay();
                        return;
                    }
                }
                this.renderTimelineOnly();
                this._syncStagePreview(this.currentFrame, { force: true });
                const now = performance.now();
                if (now - this._lastSeekUiMs > 80) {
                    this._updateTimelineDom();
                    this._lastSeekUiMs = now;
                }
            }
            this._playRaf = requestAnimationFrame(tick);
        };
        this._playRaf = requestAnimationFrame(tick);
    }
}

function findDirectorNode(nodeId) {
    const id = String(nodeId);
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        if (String(node.id) === id) return node;
    }
    return null;
}

const EXTERNAL_GROUP_NODE_TYPES = new Set([
    "MiniMaxH3DirectorGroupImageToVideo",
    "MiniMaxH3DirectorGroupReferenceToVideo",
]);
const EXTERNAL_COMBINE_NODE_TYPE = "MiniMaxH3DirectorGroupsCombine";

function graphLinkRecord(graph, linkId) {
    if (linkId == null || !graph) return null;
    const links = graph.links;
    if (!links) return null;
    let link = links[linkId];
    if (!link && typeof links.find === "function") {
        link = links.find((l) => l && (l.id === linkId || l[0] === linkId));
    }
    if (!link) return null;
    return {
        originId: link.origin_id ?? link[1],
        originSlot: link.origin_slot ?? link[2],
    };
}

function nodeWidgetValue(node, name) {
    const w = (node?.widgets || []).find((x) => x?.name === name);
    return w?.value;
}

/** Normalize LoadImage-style widget value → relative input path. */
function normalizeImageWidgetPath(value) {
    if (value == null || value === "") return null;
    let val = value;
    if (Array.isArray(val)) val = val[0];
    if (typeof val === "object" && val) {
        const name = String(val.filename || val.name || "").trim();
        if (!name) return null;
        const sub = String(val.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
        return sub ? `${sub}/${name}` : name;
    }
    if (typeof val === "string") {
        const s = val.replace(/\s*\[(input|output|temp)\]\s*$/i, "").trim();
        return s || null;
    }
    return null;
}

function readImageWidgetPath(node) {
    for (const name of ["image", "image_path", "filename"]) {
        const path = normalizeImageWidgetPath(nodeWidgetValue(node, name));
        if (path) return path;
    }
    return null;
}

/** VHS Load Video / Load Video Path use `video`; some loaders use video_path. */
function readVideoWidgetPath(node) {
    for (const name of ["video", "video_path", "file"]) {
        const path = normalizeImageWidgetPath(nodeWidgetValue(node, name));
        if (path) return path;
    }
    return null;
}

/** LoadAudio uses `audio`; VHS audio output often has no file — fall back to sibling video widget. */
function readAudioWidgetPath(node) {
    for (const name of ["audio", "audio_path", "file"]) {
        const path = normalizeImageWidgetPath(nodeWidgetValue(node, name));
        if (path) return path;
    }
    return readVideoWidgetPath(node);
}

function mediaBaseName(path) {
    const s = String(path || "").replace(/\\/g, "/");
    return s.split("/").pop() || s;
}

function parseViewUrlToPath(url) {
    try {
        const u = new URL(String(url || ""), window.location.origin);
        const filename = u.searchParams.get("filename");
        if (!filename) return null;
        const subfolder = (u.searchParams.get("subfolder") || "").replace(/\\/g, "/").replace(/\/$/, "");
        return subfolder ? `${subfolder}/${filename}` : filename;
    } catch {
        return null;
    }
}

function linkedSourceNode(graph, node, inputName) {
    if (!graph || !node) return null;
    const inp = (node.inputs || []).find((i) => i?.name === inputName);
    if (inp?.link == null) return null;
    const rec = graphLinkRecord(graph, inp.link);
    if (!rec) return null;
    return graph.getNodeById?.(rec.originId) || null;
}

/** Resolve an IMAGE input on `node` to a Comfy input-folder relative path. */
function resolveLinkedImageFile(graph, node, inputName, depth = 0) {
    if (!graph || !node || depth > 8) return null;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src) return null;

    const direct = readImageWidgetPath(src);
    if (direct) return direct;

    const imgEl = (src.imageIndex != null ? src.imgs?.[src.imageIndex] : null) || src.imgs?.[0];
    const fromPreview = parseViewUrlToPath(imgEl?.src);
    if (fromPreview) return fromPreview;

    // Walk through IMAGE passthrough nodes (Resize, etc.).
    const imgInputs = (src.inputs || []).filter((i) => String(i?.type || "") === "IMAGE" && i.link != null);
    for (const next of imgInputs) {
        const path = resolveLinkedImageFile(graph, src, next.name, depth + 1);
        if (path) return path;
    }
    return null;
}

/**
 * Resolve ref_video IMAGE-batch wiring to a previewable file path.
 * VHS Load Video exposes frames on IMAGE but the path lives on the `video` widget.
 */
function resolveLinkedVideoFile(graph, node, inputName, depth = 0) {
    if (!graph || !node || depth > 8) return null;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src) return null;

    const direct = readVideoWidgetPath(src);
    if (direct) return direct;

    // Some loaders still put the clip name on filename / image widgets.
    const fallback = readImageWidgetPath(src);
    if (fallback && /\.(mp4|webm|mov|mkv|avi|m4v)$/i.test(fallback)) return fallback;

    const walkTypes = new Set(["IMAGE", "VIDEO"]);
    const nextInputs = (src.inputs || []).filter(
        (i) => walkTypes.has(String(i?.type || "").toUpperCase()) && i.link != null,
    );
    for (const next of nextInputs) {
        const path = resolveLinkedVideoFile(graph, src, next.name, depth + 1);
        if (path) return path;
    }
    return null;
}

/** Resolve AUDIO wiring (LoadAudio / VHS audio out) to a previewable path. */
function resolveLinkedAudioFile(graph, node, inputName, depth = 0) {
    if (!graph || !node || depth > 8) return null;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src) return null;

    const direct = readAudioWidgetPath(src);
    if (direct) return direct;

    const nextInputs = (src.inputs || []).filter(
        (i) => String(i?.type || "").toUpperCase() === "AUDIO" && i.link != null,
    );
    for (const next of nextInputs) {
        const path = resolveLinkedAudioFile(graph, src, next.name, depth + 1);
        if (path) return path;
    }
    return null;
}

function imageRefFromPath(path) {
    if (!path) return null;
    return { imageFile: path, width: 0, height: 0 };
}

function videoRefFromPath(path, index) {
    if (!path) return null;
    return {
        index,
        videoFile: path,
        fileName: mediaBaseName(path),
        type: "input",
        subfolder: "",
        pairedAudioFile: "",
        previewImageFile: "",
        previewImageUrl: "",
        linked: true,
    };
}

/** First-frame poster from Load Video / IMAGE-batch upstream (when file path missing). */
function resolveLinkedVideoPoster(graph, node, inputName, depth = 0) {
    if (!graph || !node || depth > 8) return null;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src) return null;
    const imgEl = (src.imageIndex != null ? src.imgs?.[src.imageIndex] : null) || src.imgs?.[0];
    if (imgEl?.src) {
        return {
            previewImageUrl: imgEl.src,
            previewImageFile: parseViewUrlToPath(imgEl.src) || "",
        };
    }
    const nextInputs = (src.inputs || []).filter(
        (i) => String(i?.type || "").toUpperCase() === "IMAGE" && i.link != null,
    );
    for (const next of nextInputs) {
        const poster = resolveLinkedVideoPoster(graph, src, next.name, depth + 1);
        if (poster) return poster;
    }
    return null;
}

function collectAutogrowVideoRefs(graph, node) {
    const found = new Map();
    const re = /(?:^|\.)ref_video_(\d+)$/;
    for (const inp of node.inputs || []) {
        const m = String(inp?.name || "").match(re);
        if (!m || inp.link == null) continue;
        const idx = parseInt(m[1], 10);
        if (!Number.isFinite(idx) || found.has(idx)) continue;
        patchUpstreamVideoWidgetSync(graph, node, inp.name);
        const path = resolveLinkedVideoFile(graph, node, inp.name);
        const poster = resolveLinkedVideoPoster(graph, node, inp.name);
        if (path) {
            const ref = videoRefFromPath(path, idx);
            if (poster) {
                ref.previewImageFile = poster.previewImageFile || "";
                ref.previewImageUrl = poster.previewImageUrl || "";
            }
            found.set(idx, ref);
        } else if (poster?.previewImageUrl || poster?.previewImageFile) {
            found.set(idx, {
                index: idx,
                videoFile: "",
                fileName: poster.previewImageFile
                    ? mediaBaseName(poster.previewImageFile)
                    : `video_${idx + 1}`,
                type: "input",
                subfolder: "",
                pairedAudioFile: "",
                previewImageFile: poster.previewImageFile || "",
                previewImageUrl: poster.previewImageUrl || "",
                linked: true,
            });
        } else {
            // Linked IMAGE batch without resolvable path/poster — still mark occupied.
            found.set(idx, {
                index: idx,
                videoFile: "",
                fileName: "",
                type: "input",
                subfolder: "",
                pairedAudioFile: "",
                previewImageFile: "",
                previewImageUrl: "",
                linked: true,
            });
        }
    }
    return [...found.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
}

function audioRefFromPath(path, index) {
    if (!path) return null;
    return {
        index,
        audioFile: path,
        fileName: mediaBaseName(path),
        type: "input",
        subfolder: "",
    };
}

function patchUpstreamWidgetSync(graph, node, inputName, widgetNames) {
    if (!graph || !node || !widgetNames?.length) return;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src?.widgets) return;
    const names = new Set(widgetNames);
    for (const w of src.widgets) {
        if (!w || !names.has(w.name)) continue;
        if (w._mmxExternalMediaSyncPatched) continue;
        w._mmxExternalMediaSyncPatched = true;
        const prev = w.callback;
        w.callback = function (...cbArgs) {
            const r = prev?.apply(this, cbArgs);
            queueMicrotask(() => notifyDirectorsSyncExternalGroups());
            return r;
        };
    }
}

function patchUpstreamImageWidgetSync(graph, node, inputName) {
    patchUpstreamWidgetSync(graph, node, inputName, ["image", "image_path", "filename"]);
}

function patchUpstreamVideoWidgetSync(graph, node, inputName) {
    patchUpstreamWidgetSync(graph, node, inputName, ["video", "video_path", "file", "filename"]);
}

function patchUpstreamAudioWidgetSync(graph, node, inputName) {
    patchUpstreamWidgetSync(graph, node, inputName, ["audio", "audio_path", "video", "video_path", "file", "filename"]);
}

/** Collect Autogrow / legacy slots matching `(?:^|\.)prefix_(\\d+)$`. */
function collectAutogrowSlotRefs(graph, node, prefix, resolvePath, toRef, patchSync) {
    const found = new Map();
    const re = new RegExp(`(?:^|\\.)${prefix}_(\\d+)$`);
    for (const inp of node.inputs || []) {
        const m = String(inp?.name || "").match(re);
        if (!m || inp.link == null) continue;
        const idx = parseInt(m[1], 10);
        if (!Number.isFinite(idx) || found.has(idx)) continue;
        patchSync?.(graph, node, inp.name);
        const path = resolvePath(graph, node, inp.name);
        const ref = path ? toRef(path, idx) : null;
        if (ref) found.set(idx, ref);
    }
    return [...found.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
}

function readExternalGroupSpec(node, graph = null) {
    const g = graph || app.graph || app.canvas?.graph;
    const durRaw = Number(nodeWidgetValue(node, "duration_sec"));
    const prompt = String(nodeWidgetValue(node, "prompt") ?? "");
    const cls = node?.comfyClass || node?.type || "";
    const firstImageFile = resolveLinkedImageFile(g, node, "first_frame");
    const lastImageFile = resolveLinkedImageFile(g, node, "last_frame");
    patchUpstreamImageWidgetSync(g, node, "first_frame");
    patchUpstreamImageWidgetSync(g, node, "last_frame");

    let refImages = [];
    let refVideos = [];
    let refAudios = [];
    if (cls === "MiniMaxH3DirectorGroupReferenceToVideo") {
        // Autogrow: ref_images.ref_image_0 / ref_videos.ref_video_0 / …
        refImages = collectAutogrowSlotRefs(
            g, node, "ref_image", resolveLinkedImageFile,
            (path, idx) => ({ index: idx, imageFile: path, imageB64: "" }),
            patchUpstreamImageWidgetSync,
        );
        refVideos = collectAutogrowVideoRefs(g, node);
        const standaloneAudios = collectAutogrowSlotRefs(
            g, node, "ref_audio", resolveLinkedAudioFile,
            audioRefFromPath,
            patchUpstreamAudioWidgetSync,
        );
        // Paired soundtrack for the same-index reference video (ref_video_audio_N).
        const pairedAudios = collectAutogrowSlotRefs(
            g, node, "ref_video_audio", resolveLinkedAudioFile,
            audioRefFromPath,
            patchUpstreamAudioWidgetSync,
        );
        const pairedByIndex = new Map(pairedAudios.map((a) => [a.index, a]));
        for (const vid of refVideos) {
            const paired = pairedByIndex.get(vid.index);
            if (paired?.audioFile) {
                vid.pairedAudioFile = paired.audioFile;
                pairedByIndex.delete(vid.index);
            }
        }
        // Show unpaired video-audio (or standalone) in the 参考音频 strip.
        const audioMap = new Map(standaloneAudios.map((a) => [a.index, a]));
        for (const [idx, paired] of pairedByIndex) {
            if (!audioMap.has(idx)) audioMap.set(idx, paired);
        }
        refAudios = [...audioMap.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
    }

    return {
        nodeId: node?.id ?? null,
        durationSec: Number.isFinite(durRaw) && durRaw > 0 ? durRaw : null,
        prompt,
        firstImageFile,
        lastImageFile,
        refImages,
        refVideos,
        refAudios,
    };
}

/** Autogrow slots are named `groups.group_0`; legacy used `group_0` / `group_01`. */
function combineGroupSlotIndex(name) {
    const m = String(name || "").match(/(?:^|\.)group_(\d+)$/);
    return m ? parseInt(m[1], 10) : null;
}

function isCombineGroupSlot(input) {
    if (!input) return false;
    if (combineGroupSlotIndex(input.name) != null) return true;
    // Fallback: any MMX_DIR_GROUP input on the combine node.
    return String(input.type || "") === "MMX_DIR_GROUP";
}

/**
 * Frontend-only / wiring passthrough nodes that must be skipped when resolving
 * external groups for the Director UI.
 *
 * Official ComfyUI "reroute dots" are often just link path points (origin_id still
 * points at Combine) — so they already work. rgthree inserts a real virtual node
 * `Reroute (rgthree)` into the link chain, which used to stop expansion here.
 */
function isExternalGroupPassthroughNode(node) {
    if (!node) return false;
    const cls = String(node.comfyClass || node.type || "");
    if (/reroute/i.test(cls)) return true;
    // Generic virtual single-input passthrough (frontend-only nodes).
    if (node.isVirtualNode) {
        const linked = (node.inputs || []).filter((i) => i?.link != null);
        if (linked.length === 1) return true;
    }
    return false;
}

function passthroughUpstreamLinkId(node) {
    const linked = (node?.inputs || []).filter((i) => i?.link != null);
    return linked.length ? linked[0].link : null;
}

function expandExternalGroupLink(graph, linkId, depth = 0, mode = "specs") {
    if (linkId == null || depth > 16) return [];
    const rec = graphLinkRecord(graph, linkId);
    if (!rec) return [];
    const node = graph.getNodeById?.(rec.originId);
    if (!node) return [];
    const cls = node.comfyClass || node.type || "";

    // Walk through Reroute / rgthree Reroute / other virtual passthroughs.
    if (isExternalGroupPassthroughNode(node)) {
        const upstream = passthroughUpstreamLinkId(node);
        if (upstream == null) return [];
        return expandExternalGroupLink(graph, upstream, depth + 1, mode);
    }

    if (cls === EXTERNAL_COMBINE_NODE_TYPE) {
        const out = [];
        const slots = (node.inputs || [])
            .filter(isCombineGroupSlot)
            .sort((a, b) => {
                const ai = combineGroupSlotIndex(a.name);
                const bi = combineGroupSlotIndex(b.name);
                if (ai != null && bi != null) return ai - bi;
                return 0;
            });
        for (const input of slots) {
            if (input.link == null) continue;
            out.push(...expandExternalGroupLink(graph, input.link, depth + 1, mode));
        }
        return out;
    }
    if (EXTERNAL_GROUP_NODE_TYPES.has(cls)) {
        return mode === "nodes" ? [node] : [readExternalGroupSpec(node, graph)];
    }
    // Unknown upstream packer — still reserve one slot for run-select/timeline.
    if (mode === "nodes") return [null];
    return [{
        nodeId: null,
        durationSec: null,
        prompt: "",
        firstImageFile: null,
        lastImageFile: null,
        refImages: [],
        refVideos: [],
        refAudios: [],
    }];
}

function collectExternalGroupSpecs(editor) {
    const port = editor?.hasExternalI2vGroups?.()
        ? "i2v_groups"
        : editor?.hasExternalR2vGroups?.()
            ? "r2v_groups"
            : null;
    if (!port) return null;
    const graph = app.graph ?? app.canvas?.graph;
    const inp = editor?.node?.inputs?.find((i) => i?.name === port);
    if (!graph || inp?.link == null) return null;
    const specs = expandExternalGroupLink(graph, inp.link, 0, "specs");
    return specs.length ? specs : null;
}

function collectExternalGroupNodes(editor) {
    const port = editor?.hasExternalI2vGroups?.()
        ? "i2v_groups"
        : editor?.hasExternalR2vGroups?.()
            ? "r2v_groups"
            : null;
    if (!port) return null;
    const graph = app.graph ?? app.canvas?.graph;
    const inp = editor?.node?.inputs?.find((i) => i?.name === port);
    if (!graph || inp?.link == null) return null;
    const nodes = expandExternalGroupLink(graph, inp.link, 0, "nodes");
    return nodes.length ? nodes : null;
}

function notifyDirectorsSyncExternalGroups() {
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        if (!isMiniMaxH3DirectorNode(node)) continue;
        node._minimaxEditor?.syncExternalGroupsTimeline?.();
    }
}

function clearAllDirectorRunStatus() {
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        node._minimaxEditor?.clearRunProgress?.();
    }
}

/** Old workflows may still list removed output slots (e.g. segment_images). */
function isMiniMaxH3DirectorNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return cls === "MiniMaxH3Director" || cls === "ComfyMiniMaxH3Director";
}

function isDirectorNodeDef(nodeType, nodeData) {
    const cls = nodeType?.comfyClass || nodeData?.name || "";
    return cls === "MiniMaxH3Director" || cls === "ComfyMiniMaxH3Director";
}

function stripDeprecatedDirectorOutputs(node) {
    if (!isMiniMaxH3DirectorNode(node) || !node.outputs?.length) return;
    const stale = new Set(["segment_images"]);
    for (let i = node.outputs.length - 1; i >= 0; i--) {
        if (stale.has(node.outputs[i]?.name)) {
            node.removeOutput(i);
        }
    }
}

/** Reorder legacy output links after slot layout changes. */
function migrateDirectorOutputLinks(node) {
    if (!isMiniMaxH3DirectorNode(node)) return;
    const graph = app.graph ?? app.canvas?.graph;
    const links = graph?.links;
    if (!links?.length) return;
    const outputs = node.outputs || [];
    const byName = Object.fromEntries(
        outputs.map((o, i) => [o?.name, i]).filter(([n]) => !!n)
    );

    for (const link of links) {
        if (!link || String(link.origin_id) !== String(node.id)) continue;
        const target = graph.getNodeById?.(link.target_id);
        const input = target?.inputs?.[link.target_slot];
        const inputType = (input?.type || "").toUpperCase();

        // Old layouts had report at slot 1 or 3 as STRING.
        if (inputType === "STRING" && byName.report != null && link.origin_slot !== byName.report) {
            link.origin_slot = byName.report;
            continue;
        }
        // Old layouts had fps last (slot 5) as FLOAT.
        if (inputType === "FLOAT" && byName.fps != null && link.origin_slot !== byName.fps) {
            link.origin_slot = byName.fps;
            continue;
        }
        // Old layouts had frame_count at slot 2 as INT.
        if (inputType === "INT" && byName.frame_count != null && link.origin_slot !== byName.frame_count) {
            link.origin_slot = byName.frame_count;
        }
    }
}

function normalizeDirectorOutputs(node) {
    stripDeprecatedDirectorOutputs(node);
    migrateDirectorOutputLinks(node);
}

app.registerExtension({
    name: "ComfyUI.MiniMaxH3DirectorPlugin",
    async setup() {
        installDirectorClipboardGuard();
        const flushDirectors = () => {
            const graph = app.graph ?? app.canvas?.graph;
            for (const node of graph?._nodes ?? graph?.nodes ?? []) {
                node._minimaxEditor?.flushTimelineSync?.();
            }
        };
        if (app.queuePrompt && !app.queuePrompt._minimaxPatched) {
            const orig = app.queuePrompt.bind(app);
            app.queuePrompt = function (...args) {
                flushDirectors();
                clearAllDirectorRunStatus();
                return orig(...args);
            };
            app.queuePrompt._minimaxPatched = true;
        }

        api.addEventListener("minimax_director_progress", ({ detail }) => {
            findDirectorNode(detail?.node_id)?._minimaxEditor?.setRunProgress?.(detail);
        });

        api.addEventListener("minimax_director_preview", ({ detail }) => {
            const editor = findDirectorNode(detail?.node_id)?._minimaxEditor;
            if (!editor) return;
            if (editor.isImageBatch?.()) {
                setImageBatchPreview(
                    editor,
                    detail?.segment_index ?? 0,
                    detail?.image_b64 || "",
                    {
                        frames: detail?.live ? undefined : detail?.frames,
                        fps: detail?.fps,
                        live: !!detail?.live,
                        step: detail?.step,
                        total_steps: detail?.total_steps,
                    },
                );
                return;
            }
            editor.setLiveSamplePreview?.(detail);
        });

        api.addEventListener("executing", ({ detail }) => {
            if (detail == null) return;
            const node = findDirectorNode(detail);
            const editor = node?._minimaxEditor;
            if (!editor) return;
            editor.flushTimelineSync?.();
            editor.clearLiveSamplePreview?.();
            if (editor.isImageBatch?.()) {
                for (const seg of editor.timeline.segments || []) {
                    seg.previewB64 = "";
                    seg.previewFrames = [];
                    seg.previewLive = false;
                    seg.previewStep = null;
                    seg.previewTotalSteps = null;
                }
                editor.renderImageBatchGroups?.();
            }
            const segTotal = editor.getRunProgressSegmentTotal?.() ?? (editor.timeline?.segments?.length || 1);
            const timelineTotal = editor.timeline?.segments?.length || segTotal;
            editor.setRunProgress({
                node_id: detail,
                segment: 1,
                segment_total: segTotal,
                timeline_segment: 1,
                timeline_segment_total: timelineTotal,
                partial_run: editor.isRunSelectEnabled?.() && segTotal < timelineTotal,
                phase: "plan",
                phase_label: t("executing.parseTimeline"),
                phase_value: 0,
                phase_max: 1,
                overall_value: 0,
                overall_max: Math.max(1, segTotal * 6),
                remaining_segments: Math.max(0, segTotal - 1),
            });
        });

        api.addEventListener("execution_error", ({ detail }) => {
            const node = findDirectorNode(detail?.node_id);
            if (node?._minimaxEditor) {
                node._minimaxEditor.setRunError(detail?.exception_message || t("executing.error"));
            }
        });

        patchDirectorDomWidgetLayout();
        setTimeout(patchDirectorDomWidgetLayout, 500);
    },
    async loadedGraphNode(node) {
        if (!isMiniMaxH3DirectorNode(node)) return;
        normalizeDirectorOutputs(node);
        pruneDirectorDomWidgets(node);
        if (!node._minimaxDomWidget) return;
        finalizeDirectorWidgetOrder(node);
        ensureDirectorDomWidgetWidth(node);
        bindDirectorDomWidgetSizing(node, node._minimaxDomWidget, () => node._minimaxEditor);
        const editor = initDirectorEditor(node);
        editor?.scheduleRender?.();
        // Workflow size is already on node.size — settle fill after widgets arrange.
        scheduleDirectorLayoutSettle(editor);
    },
    async getCustomWidgets() {
        return {
            BDGROUP(node, inputName, inputData) {
                const w = makeGroupHeaderWidget(inputName, inputData);
                if (!node.widgets) node.widgets = [];
                node.widgets.push(w);
                return w;
            },
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeType?.comfyClass || nodeData?.name || "";
        if (EXTERNAL_GROUP_NODE_TYPES.has(cls) || cls === EXTERNAL_COMBINE_NODE_TYPE) {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (...args) {
                const out = onConnectionsChange?.apply(this, args);
                // Combine/Group wiring changes do not fire Director.onConnectionsChange.
                queueMicrotask(() => notifyDirectorsSyncExternalGroups());
                return out;
            };
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const out = onCreated?.apply(this, arguments);
                // Keep Director timeline in sync when duration/prompt widgets change.
                queueMicrotask(() => {
                    for (const w of this.widgets || []) {
                        if (w?.name !== "duration_sec" && w?.name !== "prompt") continue;
                        if (w._mmxExternalSyncPatched) continue;
                        w._mmxExternalSyncPatched = true;
                        const prev = w.callback;
                        w.callback = function (...cbArgs) {
                            const r = prev?.apply(this, cbArgs);
                            // Director→Group write-through sets this to avoid echo wipe.
                            if (!w._mmxSkipExternalSync) notifyDirectorsSyncExternalGroups();
                            return r;
                        };
                    }
                });
                return out;
            };
            return;
        }

        if (!isDirectorNodeDef(nodeType, nodeData)) return;
        if (nodeType.prototype._minimaxDirectorPatched) return;
        nodeType.prototype._minimaxDirectorPatched = true;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            normalizeDirectorOutputs(this);
            applyDirectorWidgetLabels(this);
            // ComfyUI may attach seed's control_after_generate combo after onNodeCreated.
            queueMicrotask(() => applyDirectorWidgetLabels(this));
            setTimeout(() => applyDirectorWidgetLabels(this), 0);
            this.size = [1000, 680];

            const existingDom = pruneDirectorDomWidgets(this);
            // Idempotent: reuse the host if onNodeCreated / graph restore already mounted one.
            if (existingDom?.element) {
                setTimeout(() => {
                    finalizeDirectorWidgetOrder(this);
                    initDirectorEditor(this);
                }, 0);
                return r;
            }

            const container = document.createElement("div");
            container.className = "mmx-host";
            container.style.minHeight = `${getDirectorUiHeight(null)}px`;
            container.style.setProperty("--comfy-widget-min-height", `${getDirectorUiHeight(null)}px`);
            const self = this;
            const widget = this.addDOMWidget(DIRECTOR_DOM_WIDGET_NAME, "director", container, {
                getValue: () => "",
                setValue: () => {},
                getMinHeight: () => getDirectorUiHeight(self._minimaxEditor),
                hideOnZoom: false,
                onDraw() {
                    if (self._minimaxEditor?.isPlaying) return;
                    ensureDirectorDomWidgetWidth(self);
                },
                afterResize: () => {
                    if (self._minimaxEditor?.isPlaying || self._minimaxEditor?._pauseSettling) return;
                    ensureDirectorDomWidgetWidth(self);
                    self._minimaxEditor?.onNodeResize?.();
                },
            });
            bindDirectorDomWidgetSizing(self, widget, () => self._minimaxEditor);
            widget.element = container;
            ensureDirectorDomWidgetWidth(self);
            self._minimaxDomWidget = widget;
            finalizeDirectorWidgetOrder(self);

            setTimeout(() => {
                finalizeDirectorWidgetOrder(self);
                initDirectorEditor(self);
            }, 0);
            return r;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            ensureDirectorDomWidgetWidth(this);
            const out = onResize?.apply(this, arguments);
            if (!this._minimaxEditor?.isPlaying && !this._minimaxEditor?._pauseSettling) {
                this._minimaxEditor?.onNodeResize?.(size);
            }
            return out;
        };

        const onSelected = nodeType.prototype.onSelected;
        nodeType.prototype.onSelected = function () {
            ensureDirectorDomWidgetWidth(this);
            const out = onSelected?.apply(this, arguments);
            // Reselect often lands after graph zoom/layout changes — settle redraw
            // fixes thumbs that were stretched from a mismatched canvas CSS box.
            this._minimaxEditor?.scheduleSettleRender?.();
            this._minimaxEditor?.syncExternalGroupsTimeline?.();
            return out;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            snapshotDirectorSampleWidgets(this);
            lockDirectorSampleSnap(this, 400);
            lockDirectorSigmasInput(this);
            const out = onConnectionsChange?.apply(this, args);
            this._minimaxEditor?.syncExternalGroupsTimeline?.();
            settleDirectorSampleWidgets(this);
            return out;
        };

        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, ...rest) {
            const out = onWidgetChanged?.apply(this, [name, ...rest]);
            if (DIRECTOR_SAMPLE_VALUE_WIDGETS.includes(String(name || ""))) {
                snapshotDirectorSampleWidgets(this);
            }
            return out;
        };

        const onDeselected = nodeType.prototype.onDeselected;
        nodeType.prototype.onDeselected = function () {
            const out = onDeselected?.apply(this, arguments);
            if (this._minimaxEditor?.isPlaying) this._minimaxEditor._stopPlay();
            return out;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            destroyDirectorEditor(this);
            return onRemoved?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            normalizeDirectorOutputs(this);
            const out = onConfigure?.apply(this, arguments);
            setTimeout(() => {
                finalizeDirectorWidgetOrder(this);
                snapshotDirectorSampleWidgets(this);
                syncDirectorSchedulerWidgets(this, { restore: false });
                const ed = initDirectorEditor(this) || this._minimaxEditor;
                if (!ed) return;
                const initTotal = Math.max(0, parseInt(ed.totalFramesWidget?.value || 124, 10));
                const initFps = coerceTimelineFps(ed.frameRateWidget?.value || 24);
                ed.timeline = parseTimeline(ed.timelineWidget?.value, initTotal, initFps);
                ed.syncFrameRateUI(ed.timeline.frameRate);
                ed._directorMode = ed.getDirectorMode();
                if (ed._directorMode === "video") {
                    ed.restoreVideoFromTimeline();
                } else if (ed._directorMode === "prompt_batch" || ed._directorMode === "image_batch") {
                    ensureImageBatchTimeline(ed);
                } else {
                    ed.ensureGenTimeline();
                }
                ed.applyTaskLayout(ed._directorMode);
                ed.populateTaskSelect(ed.globalTask, ed.taskTypeWidget?.value);
                ed.setEditMode(ed.timeline.editMode || "global");
                ed.selectedIndex = 0;
                ed.updateSelectionUI();
                ed.commit(true, { syncTimeline: false });
                ed._externalGroupsSyncSig = null;
                ed.syncExternalGroupsTimeline?.();
                ed.scheduleSettleRender?.();
            }, 80);
            return out;
        };
    },
});
