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
        return {
            ...guide,
            id: String(guide.id || guide.guideId || uid()),
            frameIndex: Number.isInteger(numeric) ? numeric : rawFrame,
            image: imageRef(guide.image) || imageRef(guide) || null,
        };
    });
    delete seg.timed_guides;
    seg.startImage = startRef(seg);
    seg.endImage = endRef(seg);
    return seg;
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

export function validateAddGuideSegment(seg) {
    normalizeAddGuideSegment(seg);
    const generalErrors = [];
    const guideErrors = new Map();
    if (!seg.timedGuides.length) generalErrors.push(t("addguide.error.required"));
    const ids = new Set();
    for (const guide of seg.timedGuides) {
        let error = frameError(seg, guide);
        if (!guide.id || ids.has(guide.id)) error ||= t("addguide.error.id");
        ids.add(guide.id);
        if (!imageRef(guide.image)) error ||= t("addguide.error.image");
        if (error) guideErrors.set(guide.id, error);
    }
    return { valid: !generalErrors.length && !guideErrors.size, generalErrors, guideErrors };
}

export function sanitizeTimedGuides(seg) {
    normalizeAddGuideSegment(seg);
    return sortTimedGuides(seg).map((guide) => ({
        id: String(guide.id),
        frameIndex: Number(guide.frameIndex),
        image: imageRef(guide.image),
    }));
}

export function validateAllAddGuideSegments(editor) {
    const globalKey = resolveTaskKey(editor?.getTaskKey?.() || editor?.taskTypeWidget?.value || "");
    const errors = [];
    for (const [index, seg] of (editor?.timeline?.segments || []).entries()) {
        if (resolveSegmentTaskKey(seg, globalKey) !== "addguide") continue;
        const result = validateAddGuideSegment(seg);
        if (!result.valid) {
            const detail = result.generalErrors[0] || result.guideErrors.values().next().value;
            errors.push(t("addguide.error.segment", { n: index + 1, detail }));
        }
    }
    return errors;
}

function displayTime(frame) {
    return `${(Math.max(0, Number(frame) || 0) / H3_NATIVE_FPS).toFixed(3)}s`;
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

function renderSlot({ label, ref, optional = true, onUpload, onExisting, onClear, selected = false }) {
    const wrap = document.createElement("div");
    wrap.className = `bd-ag-slot${selected ? " selected" : ""}`;
    const title = document.createElement("b");
    title.textContent = label;
    wrap.appendChild(title);
    const image = document.createElement("button");
    image.type = "button";
    image.className = "bd-ag-image";
    if (ref?.imageFile) {
        const img = document.createElement("img");
        img.src = viewUrl(ref);
        img.alt = label;
        image.appendChild(img);
    } else {
        image.textContent = optional ? t("addguide.optional") : t("addguide.chooseImage");
    }
    image.onclick = (event) => { event.stopPropagation(); onUpload?.(); };
    wrap.appendChild(image);
    const actions = document.createElement("div");
    actions.className = "bd-ag-actions";
    const existing = document.createElement("button");
    existing.type = "button";
    existing.textContent = t("mediaPicker.pickExisting");
    existing.onclick = (event) => { event.stopPropagation(); onExisting?.(); };
    actions.appendChild(existing);
    if (ref?.imageFile && onClear) {
        const clear = document.createElement("button");
        clear.type = "button";
        clear.textContent = t("tooltip.fl2vClear");
        clear.onclick = (event) => { event.stopPropagation(); onClear(); };
        actions.appendChild(clear);
    }
    wrap.appendChild(actions);
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
    const hint = document.createElement("div");
    hint.className = "bd-ag-hint";
    hint.textContent = t("addguide.localHint");
    root.appendChild(hint);

    const timeline = document.createElement("div");
    timeline.className = "bd-ag-timeline";
    const rail = document.createElement("div");
    rail.className = "bd-ag-rail";
    timeline.appendChild(rail);
    const tickStep = niceTickFrames(count);
    const ticks = new Set([0, last]);
    for (let frame = 0; frame <= last; frame += tickStep) ticks.add(frame);
    for (const frame of [...ticks].sort((a, b) => a - b)) {
        const tick = document.createElement("div");
        tick.className = "bd-ag-tick";
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
        marker.dataset.guideId = guide?.id || "";
        marker.textContent = className.includes("endpoint") ? label.slice(0, 1) : "◆";
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
                if (input) input.value = String(pending);
                if (time) time.textContent = `F${pending} · ${displayTime(pending)}`;
            };
            const up = () => {
                marker.removeEventListener("pointermove", move);
                marker.removeEventListener("pointerup", up);
                marker.removeEventListener("pointercancel", up);
                if (error) {
                    marker.style.left = `${last > 0 ? (previous / last) * 100 : 0}%`;
                    if (input) input.value = String(previous);
                    if (time) time.textContent = `F${previous} · ${displayTime(previous)}`;
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
    root.appendChild(timeline);

    const strip = document.createElement("div");
    strip.className = "bd-ag-strip";
    strip.appendChild(renderSlot({
        label: t("addguide.first"), ref: startRef(seg),
        onUpload: () => handlers.upload?.("start", null),
        onExisting: () => handlers.pick?.("start", null),
        onClear: () => { seg.startImage = null; seg.genImage = { imageFile: "" }; commitEditor(editor); },
    }));

    seg.timedGuides.forEach((guide, guideIndex) => {
        const error = validation.guideErrors.get(guide.id) || "";
        const wrap = renderSlot({
            label: t("addguide.guide", { n: guideIndex + 1 }),
            ref: imageRef(guide.image), optional: false,
            selected: guide.id === selectedId,
            onUpload: () => {
                seg._selectedGuideId = guide.id;
                handlers.upload?.("guide", guide.id);
            },
            onExisting: () => {
                seg._selectedGuideId = guide.id;
                handlers.pick?.("guide", guide.id);
            },
        });
        wrap.classList.add("bd-ag-guide-card");
        wrap.onclick = () => { seg._selectedGuideId = guide.id; editor.renderImageBatchGroups?.(); };
        const frameLabel = document.createElement("span");
        frameLabel.className = "bd-ag-frame-label";
        frameLabel.textContent = t("addguide.framePosition");
        wrap.appendChild(frameLabel);
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
        wrap.appendChild(controls);
        const time = document.createElement("div");
        time.className = "bd-ag-time";
        time.dataset.guideTime = guide.id;
        time.textContent = Number.isInteger(current) ? `F${current} · ${displayTime(current)}` : "F?";
        wrap.appendChild(time);
        const errorEl = document.createElement("div");
        errorEl.className = "bd-ag-error";
        errorEl.textContent = error;
        wrap.classList.toggle("invalid", !!error);
        wrap.appendChild(errorEl);
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "bd-ag-remove";
        remove.textContent = t("addguide.delete");
        remove.onclick = (event) => {
            event.stopPropagation();
            seg.timedGuides = seg.timedGuides.filter((item) => item.id !== guide.id);
            if (seg._selectedGuideId === guide.id) seg._selectedGuideId = seg.timedGuides[0]?.id || "";
            commitEditor(editor);
        };
        wrap.appendChild(remove);
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
        queueMicrotask(() => handlers.upload?.("guide", guide.id));
    };
    strip.appendChild(add);
    strip.appendChild(renderSlot({
        label: t("addguide.last"), ref: endRef(seg),
        onUpload: () => handlers.upload?.("end", null),
        onExisting: () => handlers.pick?.("end", null),
        onClear: () => { seg.endImage = null; commitEditor(editor); },
    }));
    root.appendChild(strip);

    const warning = document.createElement("div");
    warning.className = "bd-ag-warning";
    warning.textContent = validation.generalErrors[0] || t("addguide.densityHint");
    warning.classList.toggle("invalid", !!validation.generalErrors.length);
    root.appendChild(warning);
    card.appendChild(root);
}

export const ADDGUIDE_STYLES = `
.bd-batch-card.bd-batch-addguide{display:flex;flex-direction:column;gap:10px;align-items:stretch}
.bd-addguide{display:flex;flex-direction:column;gap:8px;min-width:0}
.bd-ag-hint,.bd-ag-warning{font-size:10px;color:#8d9aaa;line-height:1.45}
.bd-ag-warning.invalid,.bd-ag-error{color:#ff7b7b}.bd-ag-error{font-size:9px;min-height:13px}
.bd-ag-timeline{position:relative;height:72px;margin:0 12px 2px;border-radius:8px;background:#0d1118;border:1px solid #293243;overflow:visible}
.bd-ag-rail{position:absolute;left:10px;right:10px;top:34px;height:3px;background:#46556d;border-radius:3px}
.bd-ag-tick{position:absolute;top:4px;bottom:4px;transform:translateX(-50%);pointer-events:none;color:#aebbd0;font-size:9px;text-align:center}
.bd-ag-tick:after{content:"";position:absolute;top:24px;left:50%;height:19px;border-left:1px solid #3c485c}
.bd-ag-tick span,.bd-ag-tick i{display:block;font-style:normal;white-space:nowrap}.bd-ag-tick i{margin-top:27px;color:#718097}
.bd-ag-marker{position:absolute;top:25px;z-index:3;transform:translateX(-50%);width:20px;height:20px;padding:0;border-radius:50%;border:1px solid #62a8ff;background:#163658;color:#8fc5ff;cursor:ew-resize}
.bd-ag-marker.endpoint{cursor:default;border-radius:4px;border-color:#808b9b;background:#303844;color:#fff}.bd-ag-marker.selected{box-shadow:0 0 0 2px #ffcc66}.bd-ag-marker.invalid{border-color:#ff4f5f;background:#681f2a;color:#fff}
.bd-ag-strip{display:flex;gap:8px;overflow-x:auto;padding:3px 2px 8px;align-items:stretch}
.bd-ag-slot{position:relative;display:flex;flex:0 0 150px;flex-direction:column;gap:5px;padding:8px;border:1px solid #303846;border-radius:9px;background:#10151d;min-width:0}
.bd-ag-slot.selected{border-color:#ffcc66;box-shadow:0 0 0 1px rgba(255,204,102,.3)}.bd-ag-slot.invalid{border-color:#ff5968}
.bd-ag-slot>b{font-size:11px;color:#e7ecf5}.bd-ag-image{height:92px;border:1px dashed #3a4658;border-radius:7px;background:#0a0e14;color:#8290a4;overflow:hidden;padding:0}.bd-ag-image img{width:100%;height:100%;object-fit:contain}
.bd-ag-actions{display:flex;gap:4px}.bd-ag-actions button,.bd-ag-remove{font-size:9px;padding:3px 5px;flex:1}
.bd-ag-frame-label{color:#8390a3;font-size:9px;text-align:center}
.bd-ag-frame-controls{display:grid;grid-template-columns:26px 1fr 26px;gap:3px}.bd-ag-frame-controls input{width:100%;min-width:0;text-align:center}.bd-ag-frame-controls input.invalid{border-color:#ff5968;color:#ff8c96}.bd-ag-frame-controls button{padding:2px}
.bd-ag-time{text-align:center;color:#a9bad0;font-size:10px}.bd-ag-add{flex:0 0 120px;border:1px dashed #4e6684;border-radius:9px;background:#101923;color:#9dc8ff;min-height:170px}.bd-ag-add:disabled{opacity:.4}
`;
