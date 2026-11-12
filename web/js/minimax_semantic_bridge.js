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

/**
 * Link lookup that tolerates every table shape the frontend has shipped:
 * newer builds keep `graph.links` a `Map` (and `graph._links` as well), older
 * ones a plain object keyed by link id, some a plain array. Indexing a Map
 * silently returns undefined, which used to make the whole wiring walk fail and
 * the cache panel report「不匹配」for every segment.
 */
function graphLinkRecord(graph, linkId) {
    if (linkId == null || !graph) return null;
    for (const pool of [graph.links, graph._links]) {
        if (!pool) continue;
        let link = null;
        if (typeof pool.get === "function") {
            link = pool.get(linkId) ?? pool.get(String(linkId));
        } else if (Array.isArray(pool)) {
            link = pool.find((l) => l && (l.id === linkId || l[0] === linkId)) ?? null;
        } else {
            link = pool[linkId] ?? pool[String(linkId)] ?? null;
        }
        if (!link) continue;
        const originId = link.origin_id ?? link.originId ?? link[1];
        if (originId == null) continue;
        return {
            originId,
            originSlot: link.origin_slot ?? link.originSlot ?? link[2],
        };
    }
    return null;
}

/** Node lookup that still works when the graph has no `getNodeById`. */
function findGraphNode(graph, id) {
    if (!graph || id == null) return null;
    const wanted = String(id);
    if (typeof graph.getNodeById === "function") {
        const hit = graph.getNodeById(wanted) ?? graph.getNodeById(id);
        if (hit) return hit;
    }
    for (const candidate of graph._nodes ?? graph.nodes ?? []) {
        if (String(candidate?.id) === wanted) return candidate;
    }
    return null;
}

function linkedSourceNode(graph, node, inputName) {
    if (!graph || !node) return null;
    const inp = (node.inputs || []).find((i) => i?.name === inputName);
    if (inp?.link == null) return null;
    const rec = graphLinkRecord(graph, inp.link);
    if (!rec) return null;
    return findGraphNode(graph, rec.originId);
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
