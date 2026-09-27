/** MiniMax H3 Director Semantic Bridge — first-pass cache witness. */

const SEMANTIC_BRIDGE_CLASS = "MiniMaxH3DirectorSemanticBridge";
const BYPASS_MODES = new Set([2, 4]);

function isSemanticBridgeNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return cls === SEMANTIC_BRIDGE_CLASS;
}

function widgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function widgetValue(w) {
    if (!w) return undefined;
    const v = w.value;
    if (v && typeof v === "object") {
        if (typeof v.content === "string") return v.content;
        if (typeof v.value === "string") return v.value;
    }
    return v;
}

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

function linkedSourceNode(graph, node, inputName) {
    if (!graph || !node) return null;
    const inp = (node.inputs || []).find((i) => i?.name === inputName);
    if (inp?.link == null) return null;
    const rec = graphLinkRecord(graph, inp.link);
    if (!rec) return null;
    return graph.getNodeById?.(rec.originId) || null;
}

function isPassthroughNode(node) {
    if (!node) return false;
    const cls = String(node.comfyClass || node.type || "");
    if (/reroute/i.test(cls)) return true;
    if (node.isVirtualNode) {
        const linked = (node.inputs || []).filter((i) => i?.link != null);
        if (linked.length === 1) return true;
    }
    return false;
}

function firstLinkedInputName(node) {
    const inp = (node?.inputs || []).find((i) => i?.link != null);
    return inp?.name || null;
}

function nodeMuted(node) {
    return BYPASS_MODES.has(Number(node?.mode ?? 0));
}

function resolveSemanticBridgeNode(director) {
    const graph = director?.graph;
    if (!graph) return null;
    let node = director;
    let inputName = "semantic_bridge";
    for (let hop = 0; hop < 16; hop += 1) {
        const src = linkedSourceNode(graph, node, inputName);
        if (!src || nodeMuted(src)) return null;
        if (isSemanticBridgeNode(src)) return src;
        if (isPassthroughNode(src)) {
            node = src;
            inputName = firstLinkedInputName(src);
            if (!inputName) return null;
            continue;
        }
        const emits = (src.outputs || []).some(
            (out) => String(out?.type || "") === "MMX_DIR_SEMANTIC_BRIDGE",
        );
        return emits ? src : null;
    }
    return null;
}

function widgetStr(node, name, fallback) {
    const v = widgetValue(widgetByName(node, name));
    if (v == null || v === "") return fallback;
    return String(v);
}

function widgetNum(node, name, fallback) {
    const n = Number(widgetValue(widgetByName(node, name)));
    return Number.isFinite(n) ? n : fallback;
}

function widgetBool(node, name, fallback) {
    const v = widgetValue(widgetByName(node, name));
    if (v === true || v === false) return v;
    if (v == null || v === "") return fallback;
    if (v === 1 || v === "1" || v === "true") return true;
    if (v === 0 || v === "0" || v === "false") return false;
    return Boolean(v);
}

/**
 * Pack the graph-wired Semantic Bridge node so the first-pass cache panel
 * can compare the same sb_* fingerprint keys the run writes.
 * Unconnected / bypassed → null.
 */
export function collectSemanticBridgeWitness(director) {
    const src = resolveSemanticBridgeNode(director);
    if (!src) return null;
    return {
        enabled: true,
        adapter: widgetStr(src, "adapter", ""),
        alpha: widgetNum(src, "alpha", 0.15),
        magnitude_match: widgetBool(src, "magnitude_match", true),
    };
}
