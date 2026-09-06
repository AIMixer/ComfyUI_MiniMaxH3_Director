import { api } from "../../scripts/api.js";
import { resolveSegmentTaskKey, resolveTaskKey } from "./minimax_gen_timeline.js";
import { t } from "./minimax_i18n.js";

export const H3_NATIVE_FPS = 24;

function uid() {
    if (globalThis.crypto?.randomUUID) return `guide_${crypto.randomUUID()}`;
    return `guide_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 9)}`;
}

function imageRef(value) {
    if (!value || typeof value !== "object") return null;
    const imageFile = String(value.imageFile || value.image_file || value.fileName || "").trim();
    if (!imageFile) return null;
    return {
        imageFile,
        fileName: value.fileName || "",
        type: value.type || "input",
        subfolder: value.subfolder || "",
        width: Number(value.width) || 0,
        height: Number(value.height) || 0,
    };
}

function audioRef(value) {
    if (!value || typeof value !== "object") return null;
    const audioFile = String(value.audioFile || value.audio_file || value.fileName || "").trim();
    if (!audioFile) return null;
    return {
        audioFile,
        fileName: value.fileName || audioFile.split(/[\\/]/).pop() || audioFile,
        type: value.type || "input",
        subfolder: value.subfolder || "",
        durationSec: Math.max(0, Number(value.durationSec ?? value.duration_sec) || 0),
    };
}

function viewUrl(ref) {
    const path = String(ref?.imageFile || "").replace(/\\/g, "/");
    if (!path) return "";
    const slash = path.lastIndexOf("/");
    const filename = slash >= 0 ? path.slice(slash + 1) : path;
    const subfolder = slash >= 0 ? path.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, type: ref?.type || "input" });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

function audioViewUrl(ref) {
    const normalized = audioRef(ref);
    if (!normalized) return "";
    const path = normalized.audioFile.replace(/\\/g, "/");
    const slash = path.lastIndexOf("/");
    const filename = slash >= 0 ? path.slice(slash + 1) : path;
    const subfolder = slash >= 0 ? path.slice(0, slash) : normalized.subfolder;
    const params = new URLSearchParams({ filename, type: normalized.type || "input" });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

function isIntegerText(value) {
    return /^-?\d+$/.test(String(value ?? "").trim());
}

function frameValue(guide) {
    if (guide?._frameDraft != null) {
        return isIntegerText(guide._frameDraft) ? Number(guide._frameDraft) : null;
    }
    const value = Number(guide?.frameIndex);
    return Number.isInteger(value) ? value : null;
}

function startRef(seg) {
    return imageRef(seg?.startImage) || imageRef(seg?.genImage);
}

function endRef(seg) {
    return imageRef(seg?.endImage);
}

export function normalizeAddGuideSegment(seg) {
    if (!seg || typeof seg !== "object") return seg;
    const source = Array.isArray(seg.timedGuides)
        ? seg.timedGuides
        : (Array.isArray(seg.timed_guides) ? seg.timed_guides : []);
    seg.timedGuides = source.map((raw) => {
        const guide = raw && typeof raw === "object" ? raw : {};
        const rawFrame = guide.frameIndex ?? guide.frame_index;
        const numeric = Number(rawFrame);
        // Preserve object identity. UI handlers close over each Guide object;
        // replacing it during validation/prompt refresh makes those handlers
        // write to an orphan while the live timeline keeps the old frame.
        guide.id = String(guide.id || guide.guideId || uid());
        guide.frameIndex = Number.isInteger(numeric) ? numeric : rawFrame;
        guide.image = imageRef(guide.image) || imageRef(guide) || null;
        return guide;
    });
    const audioSource = Array.isArray(seg.timedAudioGuides)
        ? seg.timedAudioGuides
        : (Array.isArray(seg.timed_audio_guides) ? seg.timed_audio_guides : []);
    seg.timedAudioGuides = audioSource.map((raw) => {
        const guide = raw && typeof raw === "object" ? raw : {};
        const rawFrame = guide.frameIndex ?? guide.frame_index;
        const numeric = Number(rawFrame);
        guide.id = String(guide.id || guide.guideId || uid().replace("guide_", "audio_guide_"));
        guide.frameIndex = Number.isInteger(numeric) ? numeric : rawFrame;
        guide.audio = audioRef(guide.audio) || audioRef(guide) || null;
        return guide;
    });
    delete seg.timed_guides;
    delete seg.timed_audio_guides;
    seg.startImage = startRef(seg);
    seg.endImage = endRef(seg);
    return seg;
}

export function sortTimedAudioGuides(seg) {
    normalizeAddGuideSegment(seg);
    seg.timedAudioGuides.sort((a, b) => {
        const af = Number(a.frameIndex);
        const bf = Number(b.frameIndex);
        if (Number.isInteger(af) && Number.isInteger(bf) && af !== bf) return af - bf;
        if (Number.isInteger(af) !== Number.isInteger(bf)) return Number.isInteger(af) ? -1 : 1;
        return String(a.id).localeCompare(String(b.id));
    });
    return seg.timedAudioGuides;
}

export function sortTimedGuides(seg) {
    normalizeAddGuideSegment(seg);
    seg.timedGuides.sort((a, b) => {
        const af = Number(a.frameIndex);
        const bf = Number(b.frameIndex);
        if (Number.isInteger(af) && Number.isInteger(bf) && af !== bf) return af - bf;
        if (Number.isInteger(af) !== Number.isInteger(bf)) return Number.isInteger(af) ? -1 : 1;
        return String(a.id).localeCompare(String(b.id));
    });
    return seg.timedGuides;
}

export function chooseDefaultGuideFrame(frameCount, occupied, options = {}) {
    const count = Math.max(0, Number.parseInt(frameCount, 10) || 0);
    if (!count) return null;
    const last = count - 1;
    const used = new Set((occupied || []).map(Number).filter((v) => Number.isInteger(v) && v >= 0 && v < count));
    const anchors = [...new Set([0, last, ...used])].sort((a, b) => a - b);
    let best = null;
    for (let i = 0; i < anchors.length - 1; i++) {
        const left = anchors[i];
        const right = anchors[i + 1];
        const free = right - left - 1;
        if (free <= 0) continue;
        const candidate = Math.floor((left + right) / 2);
        if (!best || free > best.free || (free === best.free && left < best.left)) {
            best = { free, left, candidate };
        }
    }
    if (best) return best.candidate;
    if (!options.firstPresent && !used.has(0)) return 0;
    if (!options.lastPresent && !used.has(last)) return last;
    return null;
}

function frameError(seg, guide, candidate = frameValue(guide)) {
    const count = Math.max(0, Number.parseInt(seg?.frameCount ?? seg?.length, 10) || 0);
    const last = count - 1;
    if (!Number.isInteger(candidate)) return t("addguide.error.integer");
    if (candidate < 0 || candidate >= count) {
        return t("addguide.error.range", { frame: candidate, last });
    }
    if (startRef(seg) && candidate === 0) return t("addguide.error.firstConflict");
    if (endRef(seg) && candidate === last) return t("addguide.error.lastConflict", { last });
    const duplicate = (seg.timedGuides || []).some((item) => item !== guide && Number(item.frameIndex) === candidate);
    if (duplicate) return t("addguide.error.duplicate", { frame: candidate });
    return "";
}

function audioFrameError(seg, guide, candidate = frameValue(guide)) {
    const count = Math.max(0, Number.parseInt(seg?.frameCount ?? seg?.length, 10) || 0);
    const last = count - 1;
    if (!Number.isInteger(candidate)) return t("addguide.error.integer");
    if (candidate < 0 || candidate >= count) {
        return t("addguide.error.range", { frame: candidate, last });
    }
    const duplicate = (seg.timedAudioGuides || []).some(
        (item) => item !== guide && Number(item.frameIndex) === candidate,
    );
    return duplicate ? t("addguide.error.audioDuplicate", { frame: candidate }) : "";
}

export function getTimedAudioGuideRanges(seg) {
    normalizeAddGuideSegment(seg);
    const count = Math.max(1, Number.parseInt(seg?.frameCount ?? seg?.length, 10) || 1);
    const ordered = [...seg.timedAudioGuides].sort((a, b) => Number(a.frameIndex) - Number(b.frameIndex));
    return ordered.map((guide, index) => {
        const start = Number(guide.frameIndex) || 0;
        const sourceFrames = Math.max(0, Number(audioRef(guide.audio)?.durationSec || 0) * H3_NATIVE_FPS);
        const next = index + 1 < ordered.length ? Number(ordered[index + 1].frameIndex) : count;
        const effectiveFrames = Math.max(0, Math.min(sourceFrames, next - start, count - start));
        return { guide, start, end: start + effectiveFrames, effectiveFrames };
    });
}

export function validateAddGuideSegment(seg) {
    normalizeAddGuideSegment(seg);
    const generalErrors = [];
    const guideErrors = new Map();
    if (!seg.timedGuides.length && !seg.timedAudioGuides.length) {
        generalErrors.push(t("addguide.error.required"));
    }
    const ids = new Set();
    for (const guide of seg.timedGuides) {
        let error = frameError(seg, guide);
        if (!guide.id || ids.has(guide.id)) error ||= t("addguide.error.id");
        ids.add(guide.id);
        if (!imageRef(guide.image)) error ||= t("addguide.error.image");
        if (error) guideErrors.set(guide.id, error);
    }
    const audioErrors = new Map();
    const audioIds = new Set();
    for (const guide of seg.timedAudioGuides) {
        let error = audioFrameError(seg, guide);
        if (!guide.id || audioIds.has(guide.id)) error ||= t("addguide.error.audioId");
        audioIds.add(guide.id);
        if (!audioRef(guide.audio)) error ||= t("addguide.error.audio");
        if (error) audioErrors.set(guide.id, error);
    }
    return {
        valid: !generalErrors.length && !guideErrors.size && !audioErrors.size,
        generalErrors,
        guideErrors,
        audioErrors,
    };
}

export function sanitizeTimedGuides(seg) {
    normalizeAddGuideSegment(seg);
    return sortTimedGuides(seg).map((guide) => ({
        id: String(guide.id),
        frameIndex: Number(guide.frameIndex),
        image: imageRef(guide.image),
    }));
}

export function sanitizeTimedAudioGuides(seg) {
    normalizeAddGuideSegment(seg);
    return sortTimedAudioGuides(seg).map((guide) => ({
        id: String(guide.id),
        frameIndex: Number(guide.frameIndex),
        audio: audioRef(guide.audio),
    }));
}

export function validateAllAddGuideSegments(editor) {
    const globalKey = resolveTaskKey(editor?.getTaskKey?.() || editor?.taskTypeWidget?.value || "");
    const errors = [];
    for (const [index, seg] of (editor?.timeline?.segments || []).entries()) {
        if (resolveSegmentTaskKey(seg, globalKey) !== "addguide") continue;
        const result = validateAddGuideSegment(seg);
        if (!result.valid) {
            const detail = result.generalErrors[0]
                || result.guideErrors.values().next().value
                || result.audioErrors.values().next().value;
            errors.push(t("addguide.error.segment", { n: index + 1, detail }));
        }
    }
    return errors;
}

export function validateAddGuideExternalGroups(editor) {
    const taskKey = resolveTaskKey(editor?.getTaskKey?.() || editor?.taskTypeWidget?.value || "");
    const connected = !!(
        editor?.hasExternalI2vGroups?.()
        || editor?.hasExternalR2vGroups?.()
    );
    return taskKey === "addguide" && connected
        ? [t("batch.notice.addguideExternal")]
        : [];
}

function displayTime(frame) {
    return `${(Math.max(0, Number(frame) || 0) / H3_NATIVE_FPS).toFixed(3)}s`;
}

/** Prompt @ menu entries. Anchors are plain timing prose, not Picture tokens. */
export function getAddGuidePromptMentions(seg) {
    normalizeAddGuideSegment(seg);
    const count = Math.max(1, Number.parseInt(seg?.frameCount ?? seg?.length, 10) || 1);
    const last = count - 1;
    const items = [];
    const append = (name, frame, ref, tag) => {
        const image = imageRef(ref);
        if (!image) return;
        items.push({
            kind: "guide",
            label: `${name} · F${frame} · ${displayTime(frame)}`,
            tag,
            thumb: viewUrl(image),
        });
    };
    append(t("addguide.first"), 0, startRef(seg), "First Frame at 0.000s (F0)");
    const guides = [...(seg.timedGuides || [])].sort((a, b) => Number(a.frameIndex) - Number(b.frameIndex));
    guides.forEach((guide, index) => {
        const frame = Number(guide.frameIndex);
        if (!Number.isInteger(frame)) return;
        append(
            t("addguide.guide", { n: index + 1 }),
            frame,
            guide.image,
            `Guide ${index + 1} at ${displayTime(frame)} (F${frame})`,
        );
    });
    append(
        t("addguide.last"),
        last,
        endRef(seg),
        `Last Frame at ${displayTime(last)} (F${last})`,
    );
    return items;
}

function niceTickFrames(frameCount) {
    const duration = Math.max(0, (frameCount - 1) / H3_NATIVE_FPS);
    const target = Math.max(1, duration / 6);
    const seconds = [0.5, 1, 2, 5, 10, 15, 30, 60, 120].find((value) => value >= target) || 300;
    return Math.max(1, Math.round(seconds * H3_NATIVE_FPS));
}

function commitEditor(editor, render = true) {
    editor.commit?.(true, { syncTimeline: true });
    editor.flushTimelineSync?.();
    if (render) editor.renderImageBatchGroups?.();
    editor.scheduleRender?.();
    editor.updateDomWidgetHeight?.();
}

function renderSlot({
    label,
    ref,
    onUpload,
    onExisting,
    onClear,
    onDelete,
    selected = false,
}) {
    const wrap = document.createElement("div");
    wrap.className = `bd-ag-slot${selected ? " selected" : ""}`;
    const head = document.createElement("div");
    head.className = "bd-ag-slot-head";
    const title = document.createElement("b");
    title.textContent = label;
    head.appendChild(title);
    if (onDelete) {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "bd-ag-guide-delete x";
        remove.textContent = "×";
        remove.title = t("addguide.delete");
        remove.setAttribute("aria-label", t("addguide.delete"));
        remove.onclick = (event) => {
            event.stopPropagation();
            onDelete();
        };
        head.appendChild(remove);
    }
    wrap.appendChild(head);
    const imageWrap = document.createElement("div");
    imageWrap.className = `bd-ag-image-wrap${ref?.imageFile ? " has-img" : " is-empty"}`;
    const image = document.createElement("button");
    image.type = "button";
    image.className = "bd-ag-image";
    if (ref?.imageFile) {
        const img = document.createElement("img");
        img.src = viewUrl(ref);
        img.alt = label;
        image.appendChild(img);
    } else {
        image.textContent = t("addguide.uploadImage");
    }
    image.onclick = (event) => { event.stopPropagation(); onUpload?.(); };
    imageWrap.appendChild(image);
    if (ref?.imageFile && onClear) {
        const clear = document.createElement("button");
        clear.type = "button";
        clear.className = "x";
        clear.textContent = "×";
        clear.title = t("tooltip.fl2vClear");
        clear.setAttribute("aria-label", t("tooltip.fl2vClear"));
        clear.onclick = (event) => {
            event.stopPropagation();
            onClear();
        };
        imageWrap.appendChild(clear);
    }
    if (!ref?.imageFile) {
        const actions = document.createElement("div");
        actions.className = "bd-ag-actions";
        const existing = document.createElement("button");
        existing.type = "button";
        existing.className = "bd-r2v-pick-existing";
        existing.textContent = t("mediaPicker.pickExisting");
        existing.onclick = (event) => { event.stopPropagation(); onExisting?.(); };
        actions.appendChild(existing);
        imageWrap.appendChild(actions);
    }
    wrap.appendChild(imageWrap);
    return wrap;
}

export function appendAddGuideEditor(card, editor, seg, index, handlers = {}) {
    normalizeAddGuideSegment(seg);
    const validation = validateAddGuideSegment(seg);
    const count = Math.max(1, Number.parseInt(seg.frameCount ?? seg.length, 10) || 1);
    const last = count - 1;
    const selectedId = seg._selectedGuideId || seg.timedGuides[0]?.id || "";
    seg._selectedGuideId = selectedId;

    const root = document.createElement("section");
    root.className = "bd-addguide";

    const timeline = document.createElement("div");
    timeline.className = "bd-ag-timeline";
    timeline.setAttribute("aria-label", t("addguide.localHint"));
    const rail = document.createElement("div");
    rail.className = "bd-ag-rail";
    timeline.appendChild(rail);
    const tickStep = niceTickFrames(count);
    const ticks = new Set([0, last]);
    for (let frame = 0; frame <= last; frame += tickStep) ticks.add(frame);
    const orderedTicks = [...ticks]
        .sort((a, b) => a - b)
        .filter((frame) => frame === last || last - frame >= tickStep * 0.4);
    for (const frame of orderedTicks) {
        const tick = document.createElement("div");
        tick.className = "bd-ag-tick";
        tick.classList.toggle("major", frame === 0 || frame === last || frame % (H3_NATIVE_FPS * 2) === 0);
        tick.classList.toggle("at-end", frame === last);
        tick.style.left = `${last > 0 ? (frame / last) * 100 : 0}%`;
        tick.innerHTML = `<span>${(frame / H3_NATIVE_FPS).toFixed(frame % H3_NATIVE_FPS ? 1 : 0)}s</span><i>F${frame}</i>`;
        timeline.appendChild(tick);
    }
    const addMarker = (frame, label, className, guide = null) => {
        const marker = document.createElement("button");
        marker.type = "button";
        marker.className = `bd-ag-marker ${className}`;
        const visualFrame = Math.max(0, Math.min(last, Number(frame) || 0));
        marker.style.left = `${last > 0 ? (visualFrame / last) * 100 : 0}%`;
        marker.title = `${label} · F${frame} · ${displayTime(frame)}`;
        marker.dataset.label = className.includes("endpoint")
            ? label
            : `${label}_${displayTime(frame)}_f${frame}`;
        marker.dataset.guideId = guide?.id || "";
        marker.textContent = className.includes("endpoint") ? label.slice(0, 1) : "◆";
        marker.classList.toggle("at-start", visualFrame === 0);
        marker.classList.toggle("at-end", visualFrame === last);
        if (guide && guide.id === selectedId) marker.classList.add("selected");
        if (guide && validation.guideErrors.has(guide.id)) marker.classList.add("invalid");
        timeline.appendChild(marker);
        return marker;
    };
    if (startRef(seg)) addMarker(0, t("addguide.first"), "endpoint");
    if (endRef(seg)) addMarker(last, t("addguide.last"), "endpoint");
    seg.timedGuides.forEach((guide, guideIndex) => {
        const original = Number(guide.frameIndex);
        const guideLabel = t("addguide.guide", { n: guideIndex + 1 });
        const marker = addMarker(Number.isInteger(original) ? original : 0, guideLabel, "guide", guide);
        marker.onclick = (event) => {
            event.stopPropagation();
            seg._selectedGuideId = guide.id;
            editor.renderImageBatchGroups?.();
        };
        marker.onpointerdown = (event) => {
            event.preventDefault();
            event.stopPropagation();
            marker.setPointerCapture?.(event.pointerId);
            seg._selectedGuideId = guide.id;
            const previous = Number(guide.frameIndex);
            const previousError = frameError(seg, guide, previous);
            const input = [...root.querySelectorAll("[data-guide-frame]")]
                .find((element) => element.dataset.guideFrame === guide.id);
            const time = [...root.querySelectorAll("[data-guide-time]")]
                .find((element) => element.dataset.guideTime === guide.id);
            let pending = previous;
            let error = frameError(seg, guide, pending);
            const move = (moveEvent) => {
                const rect = rail.getBoundingClientRect();
                const ratio = rect.width > 0 ? (moveEvent.clientX - rect.left) / rect.width : 0;
                pending = Math.max(0, Math.min(last, Math.round(ratio * last)));
                error = frameError(seg, guide, pending);
                marker.style.left = `${last > 0 ? (pending / last) * 100 : 0}%`;
                marker.classList.toggle("invalid", !!error);
                marker.title = `${guideLabel} · F${pending} · ${displayTime(pending)}${error ? ` · ${error}` : ""}`;
                marker.dataset.label = `${guideLabel}_${displayTime(pending)}_f${pending}`;
                if (input) input.value = String(pending);
                if (time) time.textContent = displayTime(pending);
            };
            const up = () => {
                marker.removeEventListener("pointermove", move);
                marker.removeEventListener("pointerup", up);
                marker.removeEventListener("pointercancel", up);
                if (error) {
                    marker.style.left = `${last > 0 ? (previous / last) * 100 : 0}%`;
                    if (input) input.value = String(previous);
                    if (time) time.textContent = displayTime(previous);
                    marker.classList.toggle("invalid", !!previousError);
                    marker.title = `${guideLabel} · F${previous} · ${displayTime(previous)}${previousError ? ` · ${previousError}` : ""}`;
                    return;
                }
                guide.frameIndex = pending;
                delete guide._frameDraft;
                sortTimedGuides(seg);
                commitEditor(editor);
            };
            marker.addEventListener("pointermove", move);
            marker.addEventListener("pointerup", up);
            marker.addEventListener("pointercancel", up);
        };
    });
    const updateAudioTimeline = () => {
        for (const range of getTimedAudioGuideRanges(seg)) {
            const bar = [...timeline.querySelectorAll("[data-audio-bar]")]
                .find((element) => element.dataset.audioBar === range.guide.id);
            const handle = [...timeline.querySelectorAll("[data-audio-handle]")]
                .find((element) => element.dataset.audioHandle === range.guide.id);
            const left = last > 0 ? (range.start / last) * 100 : 0;
            const width = last > 0 ? (range.effectiveFrames / last) * 100 : 0;
            if (bar) {
                bar.style.left = `${left}%`;
                bar.style.width = `${Math.max(0, Math.min(100 - left, width))}%`;
                const ordered = [...seg.timedAudioGuides]
                    .sort((a, b) => Number(a.frameIndex) - Number(b.frameIndex));
                const number = ordered.findIndex((item) => item.id === range.guide.id) + 1;
                bar.textContent = `${t("addguide.audioGuide", { n: number })}_${displayTime(range.start)}_f${range.start}`;
            }
            if (handle) {
                handle.style.left = `${left}%`;
                handle.title = `${t("addguide.audioGuide", { n: [...seg.timedAudioGuides]
                    .sort((a, b) => Number(a.frameIndex) - Number(b.frameIndex))
                    .findIndex((item) => item.id === range.guide.id) + 1 })} · F${range.start} · ${displayTime(range.start)}`;
            }
        }
    };
    sortTimedAudioGuides(seg).forEach((guide, guideIndex) => {
        const label = t("addguide.audioGuide", { n: guideIndex + 1 });
        const bar = document.createElement("div");
        bar.className = "bd-ag-audio-bar";
        bar.dataset.audioBar = guide.id;
        bar.textContent = `${label}_${displayTime(guide.frameIndex)}_f${guide.frameIndex}`;
        timeline.appendChild(bar);
        const handle = document.createElement("button");
        handle.type = "button";
        handle.className = "bd-ag-audio-handle";
        handle.dataset.audioHandle = guide.id;
        handle.title = `${label} · F${guide.frameIndex} · ${displayTime(guide.frameIndex)}`;
        if (validation.audioErrors.has(guide.id)) handle.classList.add("invalid");
        handle.onpointerdown = (event) => {
            event.preventDefault();
            event.stopPropagation();
            handle.setPointerCapture?.(event.pointerId);
            const previous = Number(guide.frameIndex);
            let error = "";
            const move = (moveEvent) => {
                const rect = rail.getBoundingClientRect();
                const ratio = rect.width > 0 ? (moveEvent.clientX - rect.left) / rect.width : 0;
                const pending = Math.max(0, Math.min(last, Math.round(ratio * last)));
                error = audioFrameError(seg, guide, pending);
                guide.frameIndex = pending;
                handle.classList.toggle("invalid", !!error);
                const input = [...root.querySelectorAll("[data-audio-guide-frame]")]
                    .find((element) => element.dataset.audioGuideFrame === guide.id);
                const time = [...root.querySelectorAll("[data-audio-guide-time]")]
                    .find((element) => element.dataset.audioGuideTime === guide.id);
                if (input) input.value = String(pending);
                if (time) time.textContent = displayTime(pending);
                updateAudioTimeline();
            };
            const up = () => {
                handle.removeEventListener("pointermove", move);
                handle.removeEventListener("pointerup", up);
                handle.removeEventListener("pointercancel", up);
                if (error) guide.frameIndex = previous;
                sortTimedAudioGuides(seg);
                commitEditor(editor);
            };
            handle.addEventListener("pointermove", move);
            handle.addEventListener("pointerup", up);
            handle.addEventListener("pointercancel", up);
        };
        timeline.appendChild(handle);
    });
    updateAudioTimeline();
    root.appendChild(timeline);

    const strip = document.createElement("div");
    strip.className = "bd-ag-strip";
    strip.setAttribute("aria-label", t("addguide.pictureSection"));
    strip.appendChild(renderSlot({
        label: t("addguide.firstOptional"), ref: startRef(seg),
        onUpload: () => handlers.upload?.("start", null),
        onExisting: () => handlers.pick?.("start", null),
        onClear: () => { seg.startImage = null; seg.genImage = { imageFile: "" }; commitEditor(editor); },
    }));

    seg.timedGuides.forEach((guide, guideIndex) => {
        const error = validation.guideErrors.get(guide.id) || "";
        const wrap = renderSlot({
            label: t("addguide.guide", { n: guideIndex + 1 }),
            ref: imageRef(guide.image),
            selected: guide.id === selectedId,
            onUpload: () => {
                seg._selectedGuideId = guide.id;
                handlers.upload?.("guide", guide.id);
            },
            onExisting: () => {
                seg._selectedGuideId = guide.id;
                handlers.pick?.("guide", guide.id);
            },
            onClear: () => {
                guide.image = null;
                commitEditor(editor);
            },
            onDelete: () => {
                seg.timedGuides = seg.timedGuides.filter((item) => item.id !== guide.id);
                if (seg._selectedGuideId === guide.id) {
                    seg._selectedGuideId = seg.timedGuides[0]?.id || "";
                }
                commitEditor(editor);
            },
        });
        wrap.classList.add("bd-ag-guide-card");
        wrap.dataset.guideId = guide.id;
        // Selection is visual-only. Rebuilding here replaces a just-focused frame
        // input before the user can type, so update cards/markers in place.
        wrap.onclick = () => {
            seg._selectedGuideId = guide.id;
            root.querySelectorAll(".bd-ag-guide-card").forEach((guideCard) => {
                guideCard.classList.toggle("selected", guideCard.dataset.guideId === guide.id);
            });
            root.querySelectorAll(".bd-ag-marker").forEach((marker) => {
                marker.classList.toggle("selected", marker.dataset.guideId === guide.id);
            });
        };
        const frameRow = document.createElement("div");
        frameRow.className = "bd-ag-frame-row";
        const frameLabel = document.createElement("span");
        frameLabel.className = "bd-ag-frame-label";
        frameLabel.textContent = t("addguide.framePosition");
        const controls = document.createElement("div");
        controls.className = "bd-ag-frame-controls";
        const minus = document.createElement("button");
        const input = document.createElement("input");
        const plus = document.createElement("button");
        minus.type = plus.type = "button";
        minus.textContent = "−";
        plus.textContent = "+";
        input.type = "text";
        input.inputMode = "numeric";
        input.title = error || t("addguide.frameTooltip");
        input.dataset.guideFrame = guide.id;
        input.value = String(guide._frameDraft ?? guide.frameIndex);
        const current = frameValue(guide);
        minus.disabled = current == null || !!frameError(seg, guide, current - 1);
        plus.disabled = current == null || !!frameError(seg, guide, current + 1);
        const step = (delta) => {
            const value = Number(guide.frameIndex) + delta;
            if (frameError(seg, guide, value)) return;
            guide.frameIndex = value;
            delete guide._frameDraft;
            sortTimedGuides(seg);
            commitEditor(editor);
        };
        minus.onclick = (event) => { event.stopPropagation(); step(-1); };
        plus.onclick = (event) => { event.stopPropagation(); step(1); };
        const commitInput = () => {
            guide._frameDraft = input.value;
            const candidate = frameValue(guide);
            const nextError = frameError(seg, guide, candidate);
            if (nextError) {
                input.classList.add("invalid");
                input.title = nextError;
                const errorEl = wrap.querySelector(".bd-ag-error");
                if (errorEl) errorEl.textContent = nextError;
                editor.flushTimelineSync?.();
                return;
            }
            guide.frameIndex = candidate;
            delete guide._frameDraft;
            sortTimedGuides(seg);
            commitEditor(editor);
        };
        input.oninput = () => {
            guide._frameDraft = input.value;
            const nextError = frameError(seg, guide);
            input.classList.toggle("invalid", !!nextError);
            input.title = nextError || t("addguide.frameTooltip");
            wrap.classList.toggle("invalid", !!nextError);
            const errorEl = wrap.querySelector(".bd-ag-error");
            if (errorEl) errorEl.textContent = nextError;
        };
        input.onchange = commitInput;
        input.onblur = commitInput;
        input.onkeydown = (event) => {
            event.stopPropagation();
            if (event.key === "Enter") { event.preventDefault(); commitInput(); input.blur(); }
        };
        controls.append(minus, input, plus);
        const time = document.createElement("div");
        time.className = "bd-ag-time";
        time.dataset.guideTime = guide.id;
        time.textContent = Number.isInteger(current) ? displayTime(current) : "—";
        frameRow.append(frameLabel, controls, time);
        wrap.appendChild(frameRow);
        const errorEl = document.createElement("div");
        errorEl.className = "bd-ag-error";
        errorEl.textContent = error;
        wrap.classList.toggle("invalid", !!error);
        wrap.appendChild(errorEl);
        strip.appendChild(wrap);
    });

    const nextFrame = chooseDefaultGuideFrame(
        count,
        seg.timedGuides.map((guide) => Number(guide.frameIndex)),
        { firstPresent: !!startRef(seg), lastPresent: !!endRef(seg) },
    );
    const add = document.createElement("button");
    add.type = "button";
    add.className = "bd-ag-add";
    add.textContent = t("addguide.add");
    add.disabled = nextFrame == null;
    add.title = nextFrame == null ? t("addguide.error.noSpace") : t("addguide.addAt", { frame: nextFrame });
    add.onclick = (event) => {
        event.stopPropagation();
        if (nextFrame == null) return;
        const guide = { id: uid(), frameIndex: nextFrame, image: null };
        seg.timedGuides.push(guide);
        seg._selectedGuideId = guide.id;
        sortTimedGuides(seg);
        commitEditor(editor);
    };
    strip.appendChild(add);
    strip.appendChild(renderSlot({
        label: t("addguide.lastOptional"), ref: endRef(seg),
        onUpload: () => handlers.upload?.("end", null),
        onExisting: () => handlers.pick?.("end", null),
        onClear: () => { seg.endImage = null; commitEditor(editor); },
    }));
    root.appendChild(strip);

    const audioStrip = document.createElement("div");
    audioStrip.className = "bd-ag-audio-strip";
    audioStrip.setAttribute("aria-label", t("addguide.audioSection"));
    sortTimedAudioGuides(seg).forEach((guide, guideIndex) => {
        const ref = audioRef(guide.audio);
        const error = validation.audioErrors.get(guide.id) || "";
        const cardEl = document.createElement("div");
        cardEl.className = "bd-ag-audio-card";
        cardEl.classList.toggle("invalid", !!error);
        const head = document.createElement("div");
        head.className = "bd-ag-slot-head";
        const title = document.createElement("b");
        title.textContent = t("addguide.audioGuide", { n: guideIndex + 1 });
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "bd-ag-audio-delete";
        remove.textContent = "×";
        remove.title = t("addguide.deleteAudioGuide");
        remove.onclick = (event) => {
            event.stopPropagation();
            seg.timedAudioGuides = seg.timedAudioGuides.filter((item) => item.id !== guide.id);
            commitEditor(editor);
        };
        head.append(title, remove);
        cardEl.appendChild(head);
        if (ref) {
            const mediaRow = document.createElement("div");
            mediaRow.className = "bd-ag-audio-row";
            const preview = document.createElement("button");
            preview.type = "button";
            preview.className = "bd-ag-audio-preview";
            const play = document.createElement("span");
            play.className = "bd-ag-audio-play";
            play.textContent = "▶";
            const details = document.createElement("span");
            details.className = "bd-ag-audio-details";
            const name = document.createElement("span");
            name.textContent = ref.fileName;
            const duration = document.createElement("span");
            duration.textContent = t("addguide.audioDuration", { seconds: ref.durationSec.toFixed(3) });
            details.append(name, duration);
            const audio = document.createElement("audio");
            audio.className = "bd-ag-audio-element";
            audio.preload = "metadata";
            audio.src = audioViewUrl(ref);
            audio.onloadedmetadata = () => {
                const seconds = Number(audio.duration);
                if (!Number.isFinite(seconds) || seconds <= 0) return;
                guide.audio.durationSec = seconds;
                duration.textContent = t("addguide.audioDuration", { seconds: seconds.toFixed(3) });
                updateAudioTimeline();
            };
            audio.onended = () => { play.textContent = "▶"; };
            preview.onclick = (event) => {
                event.stopPropagation();
                if (audio.paused) {
                    audio.play().then(() => { play.textContent = "❚❚"; }).catch(() => {});
                } else {
                    audio.pause();
                    play.textContent = "▶";
                }
            };
            preview.append(play, details);
            mediaRow.appendChild(audio);
            const clear = document.createElement("button");
            clear.type = "button";
            clear.className = "bd-ag-audio-clear";
            clear.textContent = "×";
            clear.title = t("addguide.clearAudio");
            clear.onclick = (event) => {
                event.stopPropagation();
                audio.pause?.();
                guide.audio = null;
                commitEditor(editor);
            };
            mediaRow.append(preview, clear);
            cardEl.appendChild(mediaRow);
        } else {
            const actions = document.createElement("div");
            actions.className = "bd-ag-audio-empty";
            const upload = document.createElement("button");
            upload.type = "button";
            upload.textContent = t("addguide.uploadAudio");
            upload.onclick = (event) => { event.stopPropagation(); handlers.uploadAudio?.(guide.id); };
            const existing = document.createElement("button");
            existing.type = "button";
            existing.textContent = t("addguide.chooseAudio");
            existing.onclick = (event) => { event.stopPropagation(); handlers.pickAudio?.(guide.id); };
            actions.append(upload, existing);
            cardEl.appendChild(actions);
        }
        const frameRow = document.createElement("div");
        frameRow.className = "bd-ag-frame-row bd-ag-audio-frame-row";
        const frameLabel = document.createElement("span");
        frameLabel.className = "bd-ag-frame-label";
        frameLabel.textContent = t("addguide.framePosition");
        const controls = document.createElement("div");
        controls.className = "bd-ag-frame-controls bd-ag-audio-frame-controls";
        const minus = document.createElement("button");
        const input = document.createElement("input");
        const plus = document.createElement("button");
        minus.type = plus.type = "button";
        minus.textContent = "−";
        plus.textContent = "+";
        input.type = "text";
        input.inputMode = "numeric";
        input.dataset.audioGuideFrame = guide.id;
        input.value = String(guide.frameIndex);
        const step = (delta) => {
            const candidate = Number(guide.frameIndex) + delta;
            if (audioFrameError(seg, guide, candidate)) return;
            guide.frameIndex = candidate;
            sortTimedAudioGuides(seg);
            commitEditor(editor);
        };
        minus.disabled = !!audioFrameError(seg, guide, Number(guide.frameIndex) - 1);
        plus.disabled = !!audioFrameError(seg, guide, Number(guide.frameIndex) + 1);
        minus.onclick = (event) => { event.stopPropagation(); step(-1); };
        plus.onclick = (event) => { event.stopPropagation(); step(1); };
        const commitInput = () => {
            const candidate = isIntegerText(input.value) ? Number(input.value) : null;
            const nextError = audioFrameError(seg, guide, candidate);
            if (nextError) {
                input.classList.add("invalid");
                input.title = nextError;
                return;
            }
            guide.frameIndex = candidate;
            sortTimedAudioGuides(seg);
            commitEditor(editor);
        };
        input.oninput = () => {
            const candidate = isIntegerText(input.value) ? Number(input.value) : null;
            const nextError = audioFrameError(seg, guide, candidate);
            input.classList.toggle("invalid", !!nextError);
            input.title = nextError || t("addguide.frameTooltip");
        };
        input.onchange = commitInput;
        input.onblur = commitInput;
        input.onkeydown = (event) => {
            event.stopPropagation();
            if (event.key === "Enter") { event.preventDefault(); commitInput(); input.blur(); }
        };
        controls.append(minus, input, plus);
        const time = document.createElement("span");
        time.className = "bd-ag-time";
        time.dataset.audioGuideTime = guide.id;
        time.textContent = displayTime(guide.frameIndex);
        frameRow.append(frameLabel, controls, time);
        cardEl.appendChild(frameRow);
        const errorEl = document.createElement("div");
        errorEl.className = "bd-ag-error";
        errorEl.textContent = error;
        cardEl.appendChild(errorEl);
        audioStrip.appendChild(cardEl);
    });
    const nextAudioFrame = chooseDefaultGuideFrame(
        count,
        seg.timedAudioGuides.map((guide) => Number(guide.frameIndex)),
    );
    const addAudio = document.createElement("button");
    addAudio.type = "button";
    addAudio.className = "bd-ag-add bd-ag-add-audio";
    addAudio.textContent = t("addguide.addAudio");
    addAudio.title = nextAudioFrame == null
        ? t("addguide.error.noAudioSpace")
        : t("addguide.audioTooltip");
    addAudio.disabled = nextAudioFrame == null;
    addAudio.onclick = (event) => {
        event.stopPropagation();
        if (nextAudioFrame == null) return;
        seg.timedAudioGuides.push({
            id: uid().replace("guide_", "audio_guide_"),
            frameIndex: nextAudioFrame,
            audio: null,
        });
        sortTimedAudioGuides(seg);
        commitEditor(editor);
    };
    audioStrip.appendChild(addAudio);
    root.appendChild(audioStrip);
    const warning = document.createElement("div");
    warning.className = "bd-ag-warning";
    warning.textContent = validation.generalErrors[0] || t("addguide.densityHint");
    warning.classList.toggle("invalid", !!validation.generalErrors.length);
    root.appendChild(warning);
    card.appendChild(root);
}

export const ADDGUIDE_STYLES = `
.bd-batch-card.bd-batch-addguide{display:flex;flex-direction:column;gap:8px;align-items:stretch}
.bd-batch-addguide .bd-batch-head{padding-bottom:2px;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:0}
.bd-batch-addguide .bd-batch-prompts{background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:8px 10px;gap:5px}
.bd-batch-addguide .bd-batch-prompts .bd-label{color:#eaeaea;font-size:11px;font-weight:700;letter-spacing:.02em}
.bd-batch-addguide .bd-batch-prompts textarea,.bd-batch-addguide .bd-batch-prompts .bd-token-wrap{min-height:96px}
.bd-batch-addguide .bd-batch-prompts textarea{background:#101010;border-color:#2e2e2e;border-radius:8px;padding:8px;font-size:12px;line-height:1.45}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-addguide{flex:0 0 auto!important;height:auto!important;min-height:0}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-addguide .bd-batch-prompts{flex:0 0 auto;min-height:0;max-height:none;overflow:visible}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-addguide .bd-token-wrap{flex:0 0 auto;min-height:96px;height:auto;overflow:visible}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-addguide .bd-token-editor{flex:0 0 auto;min-height:96px;height:110px;max-height:none;overflow:auto;resize:vertical}
.bd-addguide{--ag-panel:#0d131b;--ag-card:#111923;--ag-line:#2d3b4d;--ag-muted:#8fa0b5;display:flex;flex-direction:column;gap:10px;min-width:0;padding:2px 4px 4px}
.bd-ag-warning{font-size:10px;color:#8d9aaa;line-height:1.45}.bd-ag-warning.invalid,.bd-ag-error{color:#ff7b7b}.bd-ag-error{min-height:15px;font-size:10px;line-height:15px;text-align:center}.bd-ag-error:empty{visibility:hidden}
.bd-ag-timeline{position:relative;height:112px;margin:0 8px 2px;background:transparent;border:0;overflow:visible}
.bd-ag-rail{position:absolute;left:0;right:0;top:24px;height:65px;box-sizing:border-box;border:1px solid #46556d;border-radius:9px}.bd-ag-rail:after{content:"";position:absolute;left:0;right:0;top:31px;border-top:1px solid #46556d}
.bd-ag-tick{position:absolute;top:0;bottom:0;transform:translateX(-50%);pointer-events:none;color:#b7c3d4;font-size:10px;text-align:center}.bd-ag-tick:after{content:"";display:none;position:absolute;top:24px;left:50%;height:65px;border-left:1px solid #2d394a}.bd-ag-tick.major:after{display:block}.bd-ag-tick.at-end:after{left:100%}
.bd-ag-tick span,.bd-ag-tick i{display:block;font-style:normal;white-space:nowrap}.bd-ag-tick i{margin-top:78px;color:#8795a8}
.bd-ag-marker{position:absolute;top:29px;z-index:5;transform:translateX(-50%);width:9px;height:27px;padding:0;border-radius:2px;border:1px solid #62a8ff;background:#326b9c;color:transparent;cursor:ew-resize}.bd-ag-marker:after{content:attr(data-label);position:absolute;left:13px;top:2px;color:#d3e6fb;font-size:10px;font-weight:400;white-space:nowrap;pointer-events:none}.bd-ag-marker.at-start{transform:none}.bd-ag-marker.at-end{transform:translateX(-100%)}
.bd-ag-marker.endpoint{top:27px;width:27px;height:27px;cursor:default;border-radius:5px;border-color:#808b9b;background:#303844;color:#fff;font-size:12px}.bd-ag-marker.endpoint:after{display:none}.bd-ag-marker.endpoint.at-end{transform:translateX(-100%)}.bd-ag-marker.selected{box-shadow:0 0 0 2px #ffcc66}.bd-ag-marker.invalid{border-color:#ff4f5f;background:#681f2a;color:#fff}
.bd-ag-audio-bar{position:absolute;top:62px;height:20px;min-width:2px;z-index:3;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;padding:2px 7px;box-sizing:border-box;border-radius:3px;background:#397c66;color:#dff9ef;font-size:10px;line-height:16px;pointer-events:none}
.bd-ag-audio-handle{position:absolute;top:57px;width:9px;height:31px;z-index:6;transform:translateX(-50%);padding:0;border:1px solid #8ee0bf;border-radius:2px;background:#2c584b;cursor:ew-resize}.bd-ag-audio-handle.invalid{border-color:#ff5968;background:#762632}
.bd-ag-strip,.bd-ag-audio-strip{display:flex;gap:16px;overflow-x:auto;padding:2px 2px 12px;align-items:flex-start;scrollbar-width:auto;scrollbar-color:#3b4757 #141b24}.bd-ag-strip::-webkit-scrollbar,.bd-ag-audio-strip::-webkit-scrollbar{height:10px}.bd-ag-strip::-webkit-scrollbar-track,.bd-ag-audio-strip::-webkit-scrollbar-track{background:#141b24}.bd-ag-strip::-webkit-scrollbar-thumb,.bd-ag-audio-strip::-webkit-scrollbar-thumb{background:#3b4757;border-radius:5px}
.bd-ag-slot{position:relative;display:flex;flex:0 0 274px;height:244px;box-sizing:border-box;flex-direction:column;gap:7px;padding:10px;border:1px solid var(--ag-line);border-radius:12px;background:linear-gradient(160deg,#121b26,#0e151e);box-shadow:0 4px 12px rgba(0,0,0,.16);min-width:0}.bd-ag-slot.selected{border-color:#ffcc66;box-shadow:0 0 0 1px rgba(255,204,102,.3)}.bd-ag-slot.invalid{border-color:#ff5968}
.bd-ag-slot-head{display:flex;align-items:center;justify-content:space-between;gap:6px;min-height:23px}.bd-ag-slot-head>b{font-size:13px;color:#e7ecf5;font-weight:650}.bd-ag-guide-delete{width:22px;height:22px;padding:0;border:0;border-radius:4px;background:rgba(0,0,0,.78);color:#ff8a8a;font-size:18px;font-weight:700;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center}.bd-ag-guide-delete:hover{background:rgba(160,30,30,.95);color:#fff}
.bd-ag-image-wrap{position:relative;width:100%;height:126px;min-width:0;box-sizing:border-box;border:1px dashed #42536a;border-radius:9px;background:#0a1017;overflow:hidden}.bd-ag-image-wrap.is-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:12px}.bd-ag-image{border:1px solid #3a495c;border-radius:6px;background:#141d27;color:#d1dbe7}.bd-ag-image-wrap.has-img .bd-ag-image{width:100%;height:100%;padding:0;border:0;border-radius:8px;overflow:hidden}.bd-ag-image-wrap.is-empty .bd-ag-image,.bd-ag-image-wrap.is-empty .bd-ag-actions{width:132px;height:38px;flex:0 0 38px}.bd-ag-image:hover{border-color:#6584aa;background:#1a2735}.bd-ag-image img{width:100%;height:100%;object-fit:cover}.bd-ag-image-wrap .x{position:absolute;right:5px;top:5px;width:24px;height:24px;padding:0;margin:0;border:0;box-sizing:border-box;display:flex;align-items:center;justify-content:center;border-radius:4px;background:rgba(0,0,0,.78);color:#ff8a8a;font-size:18px;font-weight:700;line-height:1;cursor:pointer;z-index:6;user-select:none;-webkit-user-select:none;font-family:inherit;appearance:none;-webkit-appearance:none}.bd-ag-image-wrap .x:hover{background:rgba(160,30,30,.95);color:#fff}
.bd-ag-actions{display:flex}.bd-ag-actions button{width:100%;height:100%;font-size:11px;padding:5px 7px;border:1px solid #3a495c;border-radius:6px;background:#141d27;color:#d1dbe7}.bd-ag-actions button:hover{border-color:#6584aa;background:#1a2735}
.bd-ag-frame-row{display:grid;grid-template-columns:52px 120px minmax(62px,1fr);gap:7px;align-items:center;min-height:31px}.bd-ag-frame-label{color:#aab6c7;font-size:11px;text-align:right;white-space:nowrap}.bd-ag-frame-controls{display:grid;grid-template-columns:28px 56px 28px;gap:4px}.bd-ag-frame-controls input{width:100%;min-width:0;text-align:center}.bd-ag-frame-controls input.invalid{border-color:#ff5968;color:#ff8c96}.bd-ag-frame-controls button{padding:2px}.bd-ag-time{text-align:left;color:#c1cede;font-size:11px;white-space:nowrap}
.bd-ag-add{flex:0 0 274px;height:244px;box-sizing:border-box;border:1px dashed #55789f;border-radius:12px;background:linear-gradient(160deg,#111d29,#0d1720);color:#9dceff;font-size:13px;font-weight:700}.bd-ag-add:hover{border-color:#7db4ef;background:#142536}.bd-ag-add:disabled{opacity:.4}
.bd-ag-audio-card{position:relative;display:flex;flex:0 0 274px;height:178px;box-sizing:border-box;flex-direction:column;gap:7px;padding:10px;border:1px solid var(--ag-line);border-radius:12px;background:linear-gradient(160deg,#121b26,#0e151e);box-shadow:0 4px 12px rgba(0,0,0,.16);min-width:0}.bd-ag-audio-card.invalid{border-color:#ff5968}.bd-ag-audio-delete,.bd-ag-audio-clear{width:22px;height:22px;padding:0;border:0;border-radius:4px;background:rgba(0,0,0,.78);color:#ff8a8a;font-size:17px;font-weight:700;line-height:1;cursor:pointer}.bd-ag-audio-delete:hover,.bd-ag-audio-clear:hover{background:rgba(160,30,30,.95);color:#fff}
.bd-ag-audio-empty{display:grid;grid-template-columns:1fr 1fr;gap:10px;min-height:64px;align-items:center}.bd-ag-audio-empty button{height:40px;border:1px solid #3b5264;border-radius:7px;background:#13202b;color:#cbe4ee}.bd-ag-audio-empty button:hover{border-color:#69a99b;background:#172b30}.bd-ag-audio-row{display:grid;grid-template-columns:minmax(0,1fr) 26px;gap:8px;align-items:center}.bd-ag-audio-element{display:none}.bd-ag-audio-preview{display:flex;align-items:center;gap:10px;min-width:0;min-height:64px;padding:9px 11px;text-align:left;border:1px solid #466276;border-radius:9px;background:linear-gradient(135deg,#0d1720,#102029);color:#e6edf7;cursor:pointer}.bd-ag-audio-preview:hover{border-color:#70b9a4;background:#122833}.bd-ag-audio-play{font-size:21px;color:#8ee0bf}.bd-ag-audio-details{display:flex;min-width:0;flex-direction:column;gap:4px;font-size:10px}.bd-ag-audio-details span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bd-ag-audio-frame-row{grid-template-columns:52px 120px minmax(62px,1fr)}.bd-ag-audio-frame-controls{grid-template-columns:28px 56px 28px}.bd-ag-add-audio{height:178px;border-color:#4f8876;color:#9de7ce}
`;
