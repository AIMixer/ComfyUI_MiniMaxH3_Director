import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import test from "node:test";

// Point at a clean upstream checkout to run the same behavioral contract there.
const sourceRoot = process.env.UPSTREAM_UI_SOURCE_ROOT || process.cwd();
const read = (name) => fs.readFileSync(path.join(sourceRoot, "web/js", name), "utf8");
const batchSource = read("minimax_image_batch.js");
const timelineSource = read("minimax_timeline.js");
const noop = () => {};
function functionSource(source, name) {
    const match = source.match(new RegExp(`(?:export )?function ${name}\\([^]*?\\n\\}`));
    assert.ok(match, `function ${name} must exist`);
    return match[0].replace(/^export /, "");
}
function timelineMethod(name, context) {
    const match = timelineSource.match(new RegExp(`    ${name}\\([^]*?\\n    \\}`));
    assert.ok(match, `timeline method ${name} must exist`);
    return vm.runInContext(`({${match[0]}})`, context)[name];
}
function batchContext(extra = {}) {
    const context = vm.createContext({
        resolveTaskKey: (value) => value,
        isVideoBatchTask: (value) => ["t2v", "i2v", "r2v"].includes(value),
        clearTimeout, document: { activeElement: null },
        roundDurationSec: (value) => Math.round(value * 10) / 10,
        ...extra,
    });
    for (const name of ["liveBatchSegmentFromEl", "flushBatchPromptInputs", "flushBatchDurationInputs"]) {
        vm.runInContext(functionSource(batchSource, name), context);
    }
    return context;
}

test("pack import protects segment and global prompts from old editors, then restores editing", () => {
    const context = batchContext({
        teardownPromptImageMentions: noop,
        coerceTimelineFps: Number,
        parseTimeline: JSON.parse,
        ensureImageBatchTimeline: (editor) => context.flushBatchPromptInputs(editor),
        snapshotDirectorSampleWidgets: noop,
    });
    const applyImport = timelineMethod("applyImportedTimeline", context);
    const staleInput = {
        value: "",
        getAttribute: (name) => name === "data-batch-seg-id" ? "same-id" : "0",
    };
    const editor = {
        _batchWsMem: { t2v: { prompt: "stale workspace" } },
        _videoWsMem: { v2v: { prompt: "stale workspace" } },
        // Keep an old DOM handle reachable to exercise the flush guard even
        // while detached controls still survive a synchronous import callback.
        batchList: { innerHTML: "old editor", querySelectorAll: () => [staleInput] },
        timelineWidget: { value: "" }, taskTypeWidget: { value: "t2v" },
        globalTask: {}, globalPromptWidget: {}, globalPrompt: {},
        widget: () => null,
        getDirectorMode: () => "prompt_batch",
        applyTaskLayout: noop, populateTaskSelect: noop, setEditMode: noop,
        updateSelectionUI: noop,
        commit() { context.flushBatchPromptInputs(this); },
    };
    const pack = {
        frameRate: 24, global: { taskType: "t2v", prompt: "imported global" },
        segments: [{ id: "same-id", prompt: "imported shot" }],
    };
    applyImport.call(editor, pack);
    assert.equal(editor.timeline.segments[0].prompt, "imported shot");
    assert.equal(editor.globalPrompt.value, "imported global");
    assert.equal(editor.globalPromptWidget.value, "imported global");
    assert.equal(editor.batchList.innerHTML, "");
    assert.equal(Object.keys(editor._batchWsMem).length, 0);
    assert.equal(Object.keys(editor._videoWsMem).length, 0);
    assert.equal(editor._suspendPromptFlush, false);
    staleInput.value = "new user edit";
    context.flushBatchPromptInputs(editor);
    assert.equal(editor.timeline.segments[0].prompt, "new user edit");
    editor.commit = () => { throw new Error("render failed"); };
    assert.throws(() => applyImport.call(editor, pack), /render failed/);
    assert.equal(editor._suspendPromptFlush, false, "failed imports must release the editing guard");
});

test("external group duration mirrors cannot overwrite freshly synchronized durations", () => {
    const writes = [];
    const context = batchContext({
        applyBatchSegmentDuration: (editor, index, value) => writes.push([index, value]),
    });
    const input = {
        value: "3", disabled: false, readOnly: false,
        getAttribute: (name) => name === "data-batch-seg-id" ? "group-a" : "0",
    };
    const editor = {
        getTaskKey: () => "r2v",
        timeline: { segments: [{ id: "group-a", durationSec: 7 }] },
        batchList: { querySelectorAll: (selector) => selector.startsWith("input") ? [input] : [] },
    };
    for (const port of ["hasExternalI2vGroups", "hasExternalR2vGroups"]) {
        editor[port] = () => true;
        context.flushBatchDurationInputs(editor);
        assert.equal(writes.length, 0);
        assert.equal(editor.timeline.segments[0].durationSec, 7);
        delete editor[port];
    }
    for (const flag of ["disabled", "readOnly"]) {
        input[flag] = true;
        context.flushBatchDurationInputs(editor);
        assert.equal(writes.length, 0);
        input[flag] = false;
    }
    context.flushBatchDurationInputs(editor);
    assert.deepEqual(writes, [[0, 3]], "ordinary local duration edits still apply");
});

test("external witness tracks graph prompt, duration and reference sizing without false local-draft mismatches", async () => {
    const source = read("minimax_external_witness.js");
    const witness = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
    const group = {
        id: 1, type: "MiniMaxH3DirectorR2VGroup", inputs: [],
        widgets: [{ name: "prompt", value: "graph prompt" }, { name: "duration", value: 7 }],
    };
    const graph = { _nodes: [group], links: { 10: { origin_id: 1, origin_slot: 0 } } };
    const director = { graph, inputs: [{ name: "r2v_groups", link: 10 }] };
    witness.setExternalGroupSpecsProvider(() => [{
        node: group, slot: "group_0", prompt: group.widgets[0].value, durationSec: group.widgets[1].value,
    }]);
    const timeline = {
        frameRate: 24, output: { refImageSize: "match" },
        global: { commonEnabled: false, prompt: "ignored common prompt" },
        segments: [{ index: 0, prompt: "ignored local prompt", durationSec: 3, refImageSize: "match" }],
        runSelectEnabled: true, runSelection: [0],
    };
    const capture = () => {
        witness.attachExternalGroupsWitness(director, timeline);
        return structuredClone(timeline.externalGroupsWitness);
    };
    const baseline = capture();
    assert.equal(baseline.groups[0].dur, 7);
    assert.equal(baseline.fps, 24);
    assert.deepEqual(baseline.sel, { on: true, idx: [0] });
    timeline.segments[0].prompt = "changed local prompt";
    timeline.segments[0].durationSec = 12;
    timeline.global.prompt = "changed disabled common prompt";
    assert.deepEqual(capture(), baseline);
    timeline.segments[0].refImageSize = "original";
    assert.notEqual(capture().timeline, baseline.timeline);
    timeline.segments[0].refImageSize = "match";
    timeline.global.commonEnabled = true;
    const common = capture();
    timeline.global.prompt = "new enabled common prompt";
    assert.notEqual(capture().timeline, common.timeline);
    group.widgets[0].value = "new graph prompt";
    const promptChanged = capture();
    assert.notEqual(promptChanged.groups[0].prompt, baseline.groups[0].prompt);
    assert.notEqual(promptChanged.facets.prompt, baseline.facets.prompt);
    group.widgets[1].value = 9;
    const durationChanged = capture();
    assert.equal(durationChanged.groups[0].dur, 9);
    assert.notEqual(durationChanged.facets.length, baseline.facets.length);
    const serialized = witness.injectExternalGroupsWitness(director, JSON.stringify(timeline));
    assert.equal(witness.injectExternalGroupsWitness(director, serialized), serialized);
    director.inputs[0].link = null;
    assert.equal("externalGroupsWitness" in JSON.parse(witness.injectExternalGroupsWitness(director, serialized)), false);
});

test("Refine independent seed visibility, serialization and source-aspect canvas remain intact", () => {
    const context = vm.createContext({ app: { registerExtension: noop }, api: {}, CUSTOM_ASPECT_RATIO: "Custom" });
    vm.runInContext(functionSource(read("minimax_gen_timeline.js"), "snapResolutionDim"), context);
    vm.runInContext(read("minimax_refine.js").replace(/import[\s\S]*?from "[^"]+";\r?\n/g, ""), context);
    const values = {
        mode: "upscale", seed_mode: "independent", seed: 123456, control_after_generate: "fixed",
        aspect_ratio: "16:9 (宽屏)", megapixels: 2, width: 1280, height: 720,
        upscale_method: "h3_latent", sampler: "euler", passes: 1,
    };
    const node = { widgets: Object.entries(values).map(([name, value]) => ({ name, value })) };
    const widget = (name) => node.widgets.find((entry) => entry.name === name);
    context.migrateRefineWidgets(node);
    context.syncRefineWidgetVisibility(node);
    assert.equal(widget("seed").hidden, false);
    assert.equal(widget("control_after_generate").hidden, false);
    assert.equal(widget("aspect_ratio").value, "跟随导演台");
    assert.equal(widget("aspect_ratio").hidden, true);
    assert.equal(widget("megapixels").hidden, false);
    const seedModeIndex = node.widgets.indexOf(widget("seed_mode"));
    assert.equal(node.widgets[seedModeIndex + 1], widget("seed"));
    assert.equal(node.widgets[seedModeIndex + 2], widget("control_after_generate"));
    const serialized = context.collectRefineWitness(node);
    assert.equal(serialized.seed_mode, "independent");
    assert.equal(serialized.seed, 123456);
    assert.equal(serialized.megapixels, 2);
    widget("seed_mode").value = "inherit";
    context.syncRefineWidgetVisibility(node);
    assert.equal(widget("seed").hidden, true);
    assert.equal(widget("seed").value, 123456, "mode toggles preserve the independent seed");
    widget("seed_mode").value = "independent";
    widget("mode").value = "latent_upscale";
    context.syncRefineWidgetVisibility(node);
    assert.equal(widget("seed").hidden, true);
    assert.equal(widget("sampler").hidden, true);
    for (const [width, height, mp] of [[864, 480, 2], [480, 864, 2], [2048, 1024, 0.1]]) {
        const canvas = context.canvasFromSourceMegapixels(width, height, mp);
        assert.equal(canvas.width % 32, 0);
        assert.equal(canvas.height % 32, 0);
        assert.ok(canvas.width >= width && canvas.height >= height, "Refine must not shrink the first-pass canvas");
        assert.ok(Math.abs(canvas.width / canvas.height - width / height) < 0.06);
    }
});
