/** MiniMax H3 Director Sampling — hide Director's built-in sampling widgets
 * when a Sampling node is wired into the `sampling` port. */

import { app } from "../../scripts/app.js";

const DIRECTOR_CLASSES = new Set(["MiniMaxH3Director", "ComfyMiniMaxH3Director"]);
const SAMPLING_WIDGETS = [
    "cfg",
    "seed",
    "steps",
    "sampler",
    "scheduler",
    "shift_video",
    "shift_audio",
];

function isDirectorNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return DIRECTOR_CLASSES.has(cls);
}

function widgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setWidgetVisible(node, name, visible) {
    const w = widgetByName(node, name);
    if (!w) return;
    if (w.hidden === !visible) {
        w.hidden = visible ? false : true;
    }
    if (!w.options) w.options = {};
    if (w.options.hidden !== !visible) w.options.hidden = visible ? false : true;
    if (visible) {
        if (w.element) w.element.style.display = "";
    } else {
        if (w.element) w.element.style.display = "none";
    }
}

function samplingLinkConnected(node) {
    const inp = node?.inputs?.find((i) => i?.name === "sampling");
    return inp != null && inp.link != null;
}

function syncDirectorSamplingWidgets(node) {
    if (!isDirectorNode(node)) return;
    const sampling = samplingLinkConnected(node);
    for (const name of SAMPLING_WIDGETS) {
        setWidgetVisible(node, name, !sampling);
    }
    try {
        node.setDirtyCanvas?.(true, true);
    } catch {
        /* ignore */
    }
}

function refreshSamplingNodes() {
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        syncDirectorSamplingWidgets(node);
    }
}

app.registerExtension({
    name: "ComfyUI.MiniMaxH3DirectorSampling",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeData?.name || "";
        if (!DIRECTOR_CLASSES.has(cls)) return;
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const out = onConnectionsChange?.apply(this, args);
            syncDirectorSamplingWidgets(this);
            return out;
        };
    },
    nodeCreated(node) {
        if (!isDirectorNode(node)) return;
        setTimeout(() => syncDirectorSamplingWidgets(node), 0);
    },
    loadedGraphNode(node) {
        if (!isDirectorNode(node)) return;
        setTimeout(() => syncDirectorSamplingWidgets(node), 0);
    },
    afterConfigureGraph() {
        refreshSamplingNodes();
        setTimeout(refreshSamplingNodes, 100);
    },
});