import assert from "node:assert/strict";
import fs from "node:fs";


async function loadAddGuideModule() {
    let source = fs.readFileSync("web/js/minimax_addguide.js", "utf8");
    source = source.replace(/import[\s\S]*?from "[^"]+";\r?\n/g, "");
    source = `
        const api = { apiURL: (value) => value };
        const resolveSegmentTaskKey = () => "addguide";
        const resolveTaskKey = () => "addguide";
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
        this.classList = {
            add: (...names) => {
                const current = new Set(this.className.split(/\s+/).filter(Boolean));
                names.forEach((name) => current.add(name));
                this.className = [...current].join(" ");
            },
            toggle: (name, enabled) => {
                const current = new Set(this.className.split(/\s+/).filter(Boolean));
                enabled ? current.add(name) : current.delete(name);
                this.className = [...current].join(" ");
            },
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
        commit() {},
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
}

{
    const editor = makeEditor();
    const segment = {
        frameCount: 243,
        timedGuides: [{ id: "guide-delete", frameIndex: 48, image: { imageFile: "guide.png" } }],
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
            "First Frame at 0.000s (F0)",
            "Guide 1 at 2.000s (F48)",
            "Guide 2 at 5.000s (F120)",
            "Last Frame at 10.083s (F242)",
        ],
    );
}

console.log("AddGuide UI regression checks: OK");
