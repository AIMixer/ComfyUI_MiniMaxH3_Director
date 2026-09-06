import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const batchSource = fs.readFileSync("web/js/minimax_image_batch.js", "utf8");
const timelineSource = fs.readFileSync("web/js/minimax_timeline.js", "utf8");
const noop = () => {};
const context = vm.createContext({
    resolveTaskKey: (value) => value,
    t: (key) => key,
    flushBatchPromptInputs: noop, flushBatchDurationInputs: noop, stopAllPlayers: noop,
    imageBatchVariant: () => "prompt", isVideoBatchTask: () => true,
    teardownPromptImageMentions: noop, syncBatchDetailModeButton: noop,
    renderBatchGroupPicker: noop, isBatchDetailSolo: () => false,
    updateR2vToolbarBtns: noop, updateFl2vToolbarBtns: noop, refreshPromptTokenEditors: noop,
    collectExternalGroupSpecs: () => { throw new Error("must not read ignored Groups"); },
    collectExternalGroupNodes: () => { throw new Error("must not write ignored Groups"); },
});
const renderSource = batchSource.match(/export function renderImageBatchGroups\(editor\) \{[\s\S]*?\n\}/)[0];
const render = vm.runInContext(`(${renderSource.replace("export ", "")})`, context);
const methods = {};
for (const name of ["syncExternalGroupsTimeline", "writeExternalGroupPrompt", "updateExternalGroupsBanner"]) {
    const source = timelineSource.match(new RegExp(`    ${name}\\([^]*?\\n    \\}`))[0];
    methods[name] = vm.runInContext(`({${source}})`, context)[name];
}
function element() {
    const classes = new Set();
    return { textContent: "", classList: {
        add: (name) => classes.add(name), remove: (name) => classes.delete(name),
        toggle: (name, active) => active ? classes.add(name) : classes.delete(name),
        contains: (name) => classes.has(name),
    }, setAttribute: noop };
}
for (const mode of ["addguide", "mixed", "t2v"]) {
    for (const ports of [[false, false], [true, false], [false, true], [true, true]]) {
        const add = element();
        const editor = {
            getTaskKey: () => mode,
            hasExternalI2vGroups: () => ports[0], hasExternalR2vGroups: () => ports[1],
            batchList: {}, batchPanel: { querySelector: () => add },
            batchI2vNotice: element(), timeline: { segments: [] },
        };
        render(editor);
        const connected = ports.some(Boolean);
        assert.equal(editor.batchI2vNotice.classList.contains("visible"),
            connected && mode !== "t2v");
        if (connected && mode !== "t2v") {
            assert.equal(editor.batchI2vNotice.textContent, `batch.notice.${mode}External`);
        }
        assert.equal(add.disabled, mode !== "addguide" && connected);
        if (mode === "addguide") {
            editor.externalGroupsMsgEl = element();
            editor.root = element();
            editor.updateExternalGroupsBanner = () => methods.updateExternalGroupsBanner.call(editor);
            let warningRefreshes = 0;
            editor.renderImageBatchGroups = () => { warningRefreshes += 1; };
            editor.timeline.segments.push({ prompt: "local draft", timedGuides: [{ id: "keep" }] });
            const before = JSON.stringify(editor.timeline);
            methods.syncExternalGroupsTimeline.call(editor);
            methods.writeExternalGroupPrompt.call(editor, 0, "changed draft");
            assert.equal(warningRefreshes, 1, "connection sync must refresh the warning");
            assert.equal(JSON.stringify(editor.timeline), before);
            assert.equal(editor.externalGroupsMsgEl.classList.contains("hidden"), true);
        }
    }
}
console.log("External Group warning and AddGuide data protection: OK");
