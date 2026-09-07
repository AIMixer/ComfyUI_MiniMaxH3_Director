import assert from "node:assert/strict";
import fs from "node:fs";


async function loadAddGuideModule() {
    let source = fs.readFileSync("web/js/minimax_addguide.js", "utf8");
    source = source.replace(/import[\s\S]*?from "[^"]+";\r?\n/g, "");
    source = `
        const api = { apiURL: (value) => value };
        const resolveSegmentTaskKey = (seg, globalKey) => seg?.taskKey || globalKey;
        const resolveTaskKey = (value) => value;
        const t = (key, values = {}) => key === "addguide.first"
            ? "First"
            : key === "addguide.last"
                ? "Last"
                : key === "addguide.guide"
                    ? "Guide " + (values.n ?? "")
                    : key;
    ` + source;
    return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}


class FakeElement {
    constructor(tagName) {
        this.tagName = tagName.toUpperCase();
        this.children = [];
        this.dataset = {};
        this.style = {};
        this.className = "";
        this.value = "";
        this._listeners = new Map();
        this.classList = {
            add: (...names) => {
                const current = new Set(
                    this.className.split(/\s+/).filter(Boolean),
                );
                names.forEach((name) => current.add(name));
                this.className = [...current].join(" ");
            },

            toggle: (name, enabled) => {
                const current = new Set(
                    this.className.split(/\s+/).filter(Boolean),
                );
                enabled ? current.add(name) : current.delete(name);
                this.className = [...current].join(" ");
            },

            contains: (name) => (
                this.className
                    .split(/\s+/)
                    .filter(Boolean)
                    .includes(name)
            ),
        };
    }

    appendChild(child) {
        this.children.push(child);
        child.parentElement = this;
        return child;
    }

    append(...children) {
        children.forEach((child) => this.appendChild(child));
    }

    setAttribute() {}

    addEventListener(type, listener) {
        if (!this._listeners.has(type)) this._listeners.set(type, new Set());
        this._listeners.get(type).add(listener);
    }

    removeEventListener(type, listener) {
        this._listeners.get(type)?.delete(listener);
    }

    dispatch(type, event = {}) {
        for (const listener of this._listeners.get(type) || []) listener(event);
    }

    getBoundingClientRect() {
        return { left: 0, width: 100 };
    }

    querySelector(selector) {
        return this.querySelectorAll(selector)[0] || null;
    }

    querySelectorAll(selector) {
        const result = [];
        const visit = (node) => {
            for (const child of node.children) {
                if (selector === "[data-guide-frame]" && child.dataset.guideFrame !== undefined) {
                    result.push(child);
                } else if (selector === "[data-guide-time]" && child.dataset.guideTime !== undefined) {
                    result.push(child);
                } else if (
                    selector.startsWith(".")
                    && child.className.split(/\s+/).includes(selector.slice(1))
                ) {
                    result.push(child);
                }
                visit(child);
            }
        };
        visit(this);
        return result;
    }

    click() {
        const event = {
            propagationStopped: false,
            preventDefault() {},
            stopPropagation() {
                this.propagationStopped = true;
            },
        };
        let target = this;
        while (target) {
            target.onclick?.(event);
            if (event.propagationStopped) break;
            target = target.parentElement;
        }
    }

    blur() {
        this.onblur?.();
    }
}

globalThis.document = {
    createElement: (tagName) => new FakeElement(tagName),
};

const addGuide = await loadAddGuideModule();

{
    const guide = { id: "stable-guide", frame_index: "48", image: { imageFile: "guide.png" } };
    const segment = { frameCount: 243, timedGuides: [guide] };
    addGuide.normalizeAddGuideSegment(segment);
    addGuide.validateAddGuideSegment(segment);
    addGuide.getAddGuidePromptMentions(segment);
    assert.equal(
        segment.timedGuides[0],
        guide,
        "normalization, validation, and prompt refresh must preserve Guide object identity",
    );
    assert.equal(guide.frameIndex, 48);
}

function makeEditor() {
    return {
        renderCount: 0,
        commitCount: 0,
        commit() {
            this.commitCount += 1;
        },
        flushTimelineSync() {},
        renderImageBatchGroups() {
            this.renderCount += 1;
        },
        scheduleRender() {},
        updateDomWidgetHeight() {},
    };
}

{
    const editor = makeEditor();
    const segment = {
        frameCount: 120,
        timedGuides: [],
        timedAudioGuides: [{
            id: "audio-interactions",
            frameIndex: 61,
            audio: { audioFile: "voice.wav", fileName: "voice.wav", durationSec: 3 },
        }],
    };
    const card = new FakeElement("div");
    addGuide.appendAddGuideEditor(card, editor, segment, 0, {});
    const audioCard = card.querySelector(".bd-ag-audio-card");
    assert.equal(
        card.querySelectorAll(".bd-ag-audio-explanation").length,
        0,
        "Audio Guide must not render the removed persistent explanation panel",
    );
    const frameButtons = audioCard.querySelectorAll(".bd-ag-frame-controls")[0].children;
    frameButtons[2].click();
    assert.equal(segment.timedAudioGuides[0].frameIndex, 62, "AG + button must update frameIndex");

    const rerendered = new FakeElement("div");
    addGuide.appendAddGuideEditor(rerendered, editor, segment, 0, {});
    const handle = rerendered.querySelector(".bd-ag-audio-handle");
    handle.onpointerdown({ preventDefault() {}, stopPropagation() {}, pointerId: 1 });
    handle.dispatch("pointermove", { clientX: 20 });
    handle.dispatch("pointerup", {});
    assert.equal(segment.timedAudioGuides[0].frameIndex, 24, "AG handle drag must update frameIndex");

    const clear = rerendered.querySelector(".bd-ag-audio-clear");
    clear.click();
    assert.equal(segment.timedAudioGuides.length, 1, "clearing audio must retain the AG card");
    assert.equal(segment.timedAudioGuides[0].audio, null);
    rerendered.querySelector(".bd-ag-audio-delete").click();
    assert.equal(segment.timedAudioGuides.length, 0, "deleting AG must remove the whole card");
    assert.equal(segment._selectedGuideId, "", "deleting the only selected AG must clear selection");
}

{
    const editor = makeEditor();

    const segment = {
        frameCount: 120,
        timedGuides: [],
        timedAudioGuides: [
            {
                id: "audio-a",
                frameIndex: 0,
                audio: {
                    audioFile: "a.wav",
                    fileName: "a.wav",
                    durationSec: 1,
                },
            },
            {
                id: "audio-b",
                frameIndex: 48,
                audio: {
                    audioFile: "b.wav",
                    fileName: "b.wav",
                    durationSec: 1,
                },
            },
        ],
    };

    const card = new FakeElement("div");

    addGuide.appendAddGuideEditor(
        card,
        editor,
        segment,
        0,
        {},
    );

    const warning = card.querySelector(
        ".bd-ag-audio-warning",
    );

    assert.equal(
        warning.classList.contains("hidden"),
        true,
    );

    const audio = card.querySelector(
        ".bd-ag-audio-element",
    );

    audio.duration = 3;
    audio.onloadedmetadata();

    assert.equal(
        segment.timedAudioGuides[0].audio.durationSec,
        3,
    );

    assert.equal(
        warning.classList.contains("hidden"),
        false,
    );

    assert.equal(
        warning.textContent,
        "addguide.audioTimingWarning",
    );

    assert.equal(
        editor.commitCount,
        0,
        "loadedmetadata must not commit",
    );

    assert.equal(
        editor.renderCount,
        0,
        "loadedmetadata must not rebuild AddGuide DOM",
    );
}

{
    const editor = makeEditor();

    const segment = {
        frameCount: 120,
        timedGuides: [{
            id: "picture-selected-test",
            frameIndex: 48,
            image: {
                imageFile: "guide.png",
                fileName: "guide.png",
            },
        }],
        timedAudioGuides: [{
            id: "audio-selected-test",
            frameIndex: 24,
            audio: {
                audioFile: "voice.wav",
                fileName: "voice.wav",
                durationSec: 1.0,
            },
        }],
    };

    const card = new FakeElement("div");

    addGuide.appendAddGuideEditor(
        card,
        editor,
        segment,
        0,
        {},
    );

    const pictureCard =
        card.querySelector(".bd-ag-guide-card");

    const audioCard =
        card.querySelector(".bd-ag-audio-card");

    const pictureMarker =
        card.querySelector(".bd-ag-marker");

    const audioHandle =
        card.querySelector(".bd-ag-audio-handle");

    assert.equal(
        pictureCard.classList.contains("selected"),
        true,
        "Picture Guide should remain the initial selection when it exists",
    );

    assert.equal(
        pictureMarker.classList.contains("selected"),
        true,
    );

    assert.equal(
        audioCard.classList.contains("selected"),
        false,
    );

    assert.equal(
        audioHandle.classList.contains("selected"),
        false,
    );

    audioCard.click();

    assert.equal(
        segment._selectedGuideId,
        "audio-selected-test",
    );

    assert.equal(
        audioCard.classList.contains("selected"),
        true,
    );

    assert.equal(
        audioHandle.classList.contains("selected"),
        true,
    );

    assert.equal(
        pictureCard.classList.contains("selected"),
        false,
    );

    assert.equal(
        pictureMarker.classList.contains("selected"),
        false,
    );

    pictureCard.click();

    assert.equal(
        segment._selectedGuideId,
        "picture-selected-test",
    );

    assert.equal(
        pictureCard.classList.contains("selected"),
        true,
    );

    assert.equal(
        pictureMarker.classList.contains("selected"),
        true,
    );

    assert.equal(
        audioCard.classList.contains("selected"),
        false,
    );

    assert.equal(
        audioHandle.classList.contains("selected"),
        false,
    );

    audioHandle.onpointerdown({
        preventDefault() {},
        stopPropagation() {},
        pointerId: 1,
    });

    assert.equal(
        segment._selectedGuideId,
        "audio-selected-test",
        "Audio timeline handle must select its Audio Guide",
    );

    assert.equal(
        audioCard.classList.contains("selected"),
        true,
    );

    assert.equal(
        audioHandle.classList.contains("selected"),
        true,
    );

    assert.equal(
        pictureCard.classList.contains("selected"),
        false,
    );

    assert.equal(
        pictureMarker.classList.contains("selected"),
        false,
    );

    audioHandle.dispatch("pointerup", {});
}

{
    const editor = makeEditor();

    const segment = {
        frameCount: 120,
        timedGuides: [],
        timedAudioGuides: [{
            id: "audio-only-selected",
            frameIndex: 24,
            audio: {
                audioFile: "voice.wav",
                fileName: "voice.wav",
                durationSec: 1.0,
            },
        }],
    };

    const card = new FakeElement("div");

    addGuide.appendAddGuideEditor(
        card,
        editor,
        segment,
        0,
        {},
    );

    assert.equal(
        segment._selectedGuideId,
        "audio-only-selected",
    );

    assert.equal(
        card.querySelector(".bd-ag-audio-card")
            .classList.contains("selected"),
        true,
    );

    assert.equal(
        card.querySelector(".bd-ag-audio-handle")
            .classList.contains("selected"),
        true,
    );
}

{
    const editor = makeEditor();
    const segment = {
        frameCount: 243,
        timedGuides: [{ id: "guide-a", frameIndex: 48, image: { imageFile: "guide.png" } }],
    };
    const card = new FakeElement("div");
    addGuide.appendAddGuideEditor(card, editor, segment, 0, {});
    const input = card.querySelector("[data-guide-frame]");

    input.click();
    assert.equal(
        editor.renderCount,
        0,
        "clicking the frame input must not bubble into the Guide card and replace the focused input",
    );
    input.value = "49";
    input.oninput();
    input.onkeydown({ key: "Enter", preventDefault() {}, stopPropagation() {} });
    assert.equal(segment.timedGuides[0].frameIndex, 49);
}

{
    const editor = makeEditor();
    let uploadCount = 0;
    const segment = { frameCount: 243, timedGuides: [] };
    const card = new FakeElement("div");
    addGuide.appendAddGuideEditor(card, editor, segment, 0, {
        upload: () => { uploadCount += 1; },
    });

    card.querySelector(".bd-ag-add").click();
    assert.equal(segment.timedGuides.length, 1);
    assert.equal(uploadCount, 0, "adding an empty Guide must not open upload");
}

{
    const editor = makeEditor();
    const segment = {
        frameCount: 243,
        timedGuides: [{ id: "guide-empty", frameIndex: 48, image: null }],
    };
    const card = new FakeElement("div");
    addGuide.appendAddGuideEditor(card, editor, segment, 0, {});
    assert.deepEqual(
        card.querySelectorAll(".bd-ag-image").map((element) => element.textContent),
        ["addguide.uploadImage", "addguide.uploadImage", "addguide.uploadImage"],
        "First, empty Guide, and Last must use the same upload copy",
    );
    assert.equal(
        card.querySelectorAll(".bd-ag-section-label").length,
        0,
        "the reference layout must not add visible picture/audio section headings",
    );
    assert.equal(
        card.querySelectorAll(".bd-ag-frame-row").length,
        1,
        "a Picture Guide must keep label, controls, and time in one row",
    );
    for (const image of card.querySelectorAll(".bd-ag-image")) {
        assert.ok(
            image.parentElement.querySelector(".bd-ag-actions"),
            "empty upload and existing-image actions must share one media well",
        );
    }
    assert.match(
        addGuide.ADDGUIDE_STYLES,
        /\.bd-ag-slot\{[^}]*flex:0 0 274px;height:244px/,
        "Picture Guide cards must use the reference card footprint",
    );
    assert.match(
        addGuide.ADDGUIDE_STYLES,
        /\.bd-ag-add\{[^}]*flex:0 0 274px;height:244px/,
        "the add-picture card must align with Picture Guide cards",
    );
}

{
    const editor = makeEditor();
    const segment = {
        frameCount: 243,
        timedGuides: [{ id: "guide-delete", frameIndex: 48, image: { imageFile: "guide.png" } }],
        timedAudioGuides: [{
            id: "audio-delete-fallback",
            frameIndex: 24,
            audio: { audioFile: "voice.wav", durationSec: 1 },
        }],
    };
    const card = new FakeElement("div");
    addGuide.appendAddGuideEditor(card, editor, segment, 0, {});
    const imageClear = card.querySelectorAll(".x").find(
        (button) => button.parentElement?.className.includes("bd-ag-image-wrap"),
    );
    const guideDelete = card.querySelector(".bd-ag-guide-delete");

    imageClear.click();
    assert.equal(segment.timedGuides.length, 1, "image clear must preserve the Guide");
    assert.equal(segment.timedGuides[0].image, null, "image clear must only clear the image");
    guideDelete.click();
    assert.equal(segment.timedGuides.length, 0, "card delete must remove the whole Guide");
    assert.equal(
        segment._selectedGuideId,
        "audio-delete-fallback",
        "deleting the selected PG must fall back to the first remaining AG",
    );
}

{
    const anchors = addGuide.getAddGuidePromptMentions({
        frameCount: 243,
        startImage: { imageFile: "first.png" },
        endImage: { imageFile: "last.png" },
        timedGuides: [
            { id: "later", frameIndex: 120, image: { imageFile: "later.png" } },
            { id: "earlier", frameIndex: 48, image: { imageFile: "earlier.png" } },
        ],
    });
    assert.deepEqual(
        anchors.map((item) => item.tag),
        [
            "At 0.000s,",
            "At 2.000s,",
            "At 5.000s,",
            "At 10.083s,",
        ],
    );
    assert.equal(anchors[1].label, "Guide 1 · F48 · 2.000s");
}

{
    const segment = {
        frameCount: 243,
        timedGuides: [{ id: "pg", frameIndex: 48, image: { imageFile: "pg.png" } }],
        timedAudioGuides: [
            { id: "later", frameIndex: 48, audio: { audioFile: "later.wav", durationSec: 4 } },
            { id: "first", frameIndex: 0, audio: { audioFile: "first.wav", durationSec: 4 } },
        ],
    };
    const validation = addGuide.validateAddGuideSegment(segment);
    assert.equal(validation.valid, true, "PG and AG may share F48");
    assert.deepEqual(
        addGuide.getTimedAudioGuideRanges(segment).map((range) => [range.start, range.end]),
        [[0, 48], [48, 144]],
        "the next AG truncates the previous AG while PG positions have no effect",
    );
    segment.timedAudioGuides.find((guide) => guide.id === "later").frameIndex = 72;
    assert.deepEqual(
        addGuide.getTimedAudioGuideRanges(segment).map((range) => [range.start, range.end]),
        [[0, 72], [72, 168]],
    );
    segment.timedAudioGuides.find((guide) => guide.id === "later").frameIndex = 120;
    assert.equal(
        addGuide.getTimedAudioGuideRanges(segment)[0].end,
        96,
        "the previous AG cannot grow past its four-second source",
    );
    const sanitized = addGuide.sanitizeTimedAudioGuides(segment);
    assert.deepEqual(sanitized.map((guide) => guide.id), ["first", "later"]);
    assert.equal(sanitized[0].audio.durationSec, 4);
}

{
    const editor = {
        getTaskKey: () => "addguide",
        hasExternalI2vGroups: () => true,
        hasExternalR2vGroups: () => false,
    };
    assert.deepEqual(
        addGuide.validateAddGuideExternalGroups(editor),
        ["batch.notice.addguideExternal"],
        "queue validation must block AddGuide while an external Group remains connected",
    );
    editor.hasExternalI2vGroups = () => false;
    assert.deepEqual(addGuide.validateAddGuideExternalGroups(editor), []);
}

{
    const segment = {
        frameCount: 120,
        timedGuides: [],
        timedAudioGuides: [{ id: "audio-only", frameIndex: 0, audio: { audioFile: "only.wav", durationSec: 8 } }],
    };
    assert.equal(addGuide.validateAddGuideSegment(segment).valid, true, "audio-only AddGuide is valid");
    assert.equal(addGuide.getTimedAudioGuideRanges(segment)[0].end, 120, "segment end truncates the last AG");
}

{
    const audio = (frameIndex, durationSec, audioFile = "voice.wav") => ({
        frameIndex,
        audio: audioFile ? { audioFile, durationSec } : null,
    });
    assert.equal(addGuide.hasAddGuideAudioTimingConflict({
        frameCount: 120,
        timedAudioGuides: [audio(0, 2), audio(48, 2)],
    }), false, "adjacent Audio Guides must not conflict");
    const overlapping = {
        frameCount: 120,
        timedAudioGuides: [audio(0, 3), audio(48, 2)],
    };
    assert.equal(
        addGuide.hasAddGuideAudioTimingConflict(overlapping),
        true,
        "source duration extending past the next Audio Guide must conflict",
    );
    assert.equal(
        addGuide.validateAddGuideSegment(overlapping).valid,
        true,
        "Audio Guide timing conflicts must remain warnings rather than validation errors",
    );
    assert.equal(addGuide.hasAddGuideAudioTimingConflict({
        frameCount: 120,
        timedAudioGuides: [audio(96, 2)],
    }), true, "source duration extending past the segment end must conflict");
    assert.equal(addGuide.hasAddGuideAudioTimingConflict({
        frameCount: 120,
        timedAudioGuides: [audio("invalid", 10), audio(96, 10, "")],
    }), false, "invalid positions and missing audio must be skipped");

}

console.log("AddGuide UI regression checks: OK");
