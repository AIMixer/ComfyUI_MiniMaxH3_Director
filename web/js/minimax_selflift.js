/** MiniMax H3 Director SelfLift — split-mode widget visibility + cache witness. */

import { app } from "../../scripts/app.js";

const SELFLIFT_CLASS = "MiniMaxH3DirectorSelfLift";
const DIRECTOR_CLASSES = new Set(["MiniMaxH3Director", "ComfyMiniMaxH3Director"]);
const BYPASS_MODES = new Set([2, 4]);

function isSelfLiftNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return cls === SELFLIFT_CLASS;
}

function widgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function getWidgetValueStore() {
    try {
        const pinia = app.vueApp?.config?.globalProperties?.$pinia;
        return pinia?._s?.get?.("widgetValue") || null;
    } catch {
        return null;
    }
}

function eachVueWidgetState(node, w, fn) {
    if (!w || !node || typeof fn !== "function") return;
    const store = getWidgetValueStore();
    if (!store) return;
    const graphId = node.graph?.rootGraph?.id ?? node.graph?.id;
    const states = [];
    if (typeof store.getNodeWidgets === "function" && graphId != null) {
        try {
            const found = store.getNodeWidgets(graphId, node.id);
            if (Array.isArray(found)) states.push(...found);
        } catch {
            /* ignore */
        }
    }
    if (w.widgetId && typeof store.getWidget === "function") {
        try {
            const st = store.getWidget(w.widgetId);
            if (st) states.push(st);
        } catch {
            /* ignore */
        }
    }
    for (const st of states) {
        if (!st || (st.name && st.name !== w.name)) continue;
        fn(st);
    }
}

function clearVueHiddenState(node, w) {
    eachVueWidgetState(node, w, (st) => {
        if (st.options && typeof st.options === "object") {
            st.options.hidden = false;
            delete st.options.hidden;
        }
        if ("hidden" in st) st.hidden = false;
    });
}

function setVueHiddenState(node, w, hidden) {
    eachVueWidgetState(node, w, (st) => {
        if (!st.options || typeof st.options !== "object") st.options = {};
        st.options.hidden = hidden;
        if ("hidden" in st) st.hidden = hidden;
    });
}

function restoreDefaultLabel(node, w) {
    if (!w) return;
    delete w.label;
    if (w.options) delete w.options.label;
    eachVueWidgetState(node, w, (st) => {
        if (st.options && typeof st.options === "object") delete st.options.label;
        if ("label" in st) delete st.label;
    });
}

function unhideWidget(node, w) {
    if (!w) return;
    w.hidden = false;
    if (w.options) {
        const next = { ...w.options };
        delete next.hidden;
        next.hidden = false;
        w.options = next;
    } else {
        w.options = { hidden: false };
    }
    w.options.hidden = false;
    delete w.options.hidden;
    if (w._mmxOrigComputeSize) {
        w.computeSize = w._mmxOrigComputeSize;
        delete w._mmxOrigComputeSize;
    } else if (w.computeSize && w.computeSize._mmxSelfLiftHide) {
        delete w.computeSize;
    }
    if (w.element) {
        w.element.style.display = "";
        w.element.hidden = false;
    }
    clearVueHiddenState(node, w);
}

function restoreConvertedWidget(node, name) {
    const w = widgetByName(node, name);
    if (w) return w;
    const input = (node.inputs || []).find((i) => i?.name === name);
    if (!input) return null;
    if (typeof node.convertToWidget === "function") {
        try {
            node.convertToWidget(input);
        } catch {
            /* older / newer frontends differ */
        }
    }
    return widgetByName(node, name);
}

function insertWidgetAfter(node, widget, afterName) {
    const list = node.widgets;
    if (!list || !widget) return;
    const cur = list.indexOf(widget);
    if (cur >= 0) list.splice(cur, 1);
    const after = list.findIndex((item) => item?.name === afterName);
    list.splice(after >= 0 ? after + 1 : list.length, 0, widget);
}

function nodeInputSpec(node, name) {
    const data = node?.constructor?.nodeData?.input;
    return data?.required?.[name] || data?.optional?.[name] || null;
}

function ensureIntWidget(node, name, value, options, afterName) {
    let w = restoreConvertedWidget(node, name) || widgetByName(node, name);
    if (!w) {
        const spec = nodeInputSpec(node, name) || ["INT", { default: value, ...options }];
        const factory = window.comfyAPI?.widgets?.ComfyWidgets?.INT;
        if (typeof factory === "function") {
            try {
                const out = factory(node, name, spec, app);
                w = out?.widget || widgetByName(node, name);
            } catch {
                /* older / Vue frontends differ */
            }
        }
    }
    if (!w && typeof node.addWidget === "function") {
        w = node.addWidget("number", name, value, () => {}, options);
        if (w) w.serialize = true;
    }
    if (w) {
        unhideWidget(node, w);
        if (w.value == null || w.value === "") w.value = value;
        if (!w.type || w.type === "converted-widget") w.type = "number";
        if (!w.options) w.options = {};
        Object.assign(w.options, options);
        delete w.options.hidden;
        w.hidden = false;
        insertWidgetAfter(node, w, afterName);
        clearVueHiddenState(node, w);
    }
    return w;
}

function bumpWidgetList(node) {
    if (!Array.isArray(node.widgets)) return;
    node.widgets = node.widgets.slice();
}

function visibleWidgetRowHeight(w) {
    if (!w || w.hidden || w.options?.hidden) return 0;
    if (w.computeSize?._mmxSelfLiftHide) return 0;
    if (w._bdGroupHeader || w.type === "BDGROUP") return 40;
    return 30;
}

function fitSelfLiftNode(node) {
    if (!node?.setSize) return;
    const width = Math.max(Number(node.size?.[0]) || 0, 280);
    let contentH = 62;
    for (const w of node.widgets || []) contentH += visibleWidgetRowHeight(w);
    contentH += 20;
    const computed = Number(node.computeSize?.(width)?.[1]) || 0;
    const need = Math.max(contentH, computed + 24);
    const cur = Number(node.size?.[1]) || 0;
    if (Math.abs(need - cur) > 2) {
        node.setSize([width, need]);
        node.onResize?.(node.size);
    }
}

function setWidgetVisible(node, name, visible) {
    const w = widgetByName(node, name);
    if (!w) return;
    if (visible) {
        unhideWidget(node, w);
        return;
    }
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    setVueHiddenState(node, w, true);
    if (!w._mmxOrigComputeSize && typeof w.computeSize === "function" && !w.computeSize._mmxSelfLiftHide) {
        w._mmxOrigComputeSize = w.computeSize.bind(w);
    }
    const hide = () => [0, -4];
    hide._mmxSelfLiftHide = true;
    w.computeSize = hide;
    if (w.element) w.element.style.display = "none";
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

function resolveSelfLiftNode(director) {
    const graph = director?.graph;
    if (!graph) return null;
    let node = director;
    let inputName = "selflift";
    for (let hop = 0; hop < 16; hop += 1) {
        const src = linkedSourceNode(graph, node, inputName);
        if (!src || nodeMuted(src)) return null;
        if (isSelfLiftNode(src)) return src;
        if (isPassthroughNode(src)) {
            node = src;
            inputName = firstLinkedInputName(src);
            if (!inputName) return null;
            continue;
        }
        const emits = (src.outputs || []).some(
            (out) => String(out?.type || "") === "MMX_DIR_SELFLIFT",
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

function hiresModelLinked(node) {
    const src = linkedSourceNode(node?.graph, node, "model_hires");
    return Boolean(src && !nodeMuted(src));
}

/**
 * Pack the graph-wired SelfLift node so the first-pass cache panel can compare
 * the same sl_* fingerprint keys the run writes. Unconnected / bypassed → null.
 */
export function collectSelfLiftWitness(director) {
    const src = resolveSelfLiftNode(director);
    if (!src) return null;
    return {
        enabled: true,
        split_mode: widgetStr(src, "split_mode", "highres_steps"),
        highres_steps: widgetNum(src, "highres_steps", 2),
        transition_step: widgetNum(src, "transition_step", 6),
        lowres_scale: widgetNum(src, "lowres_scale", 0.5),
        sampler_mode: widgetStr(src, "sampler_mode", "euler"),
        native_low_carry: widgetBool(src, "native_low_carry", true),
        latent_upscale_model: widgetStr(src, "latent_upscale_model", ""),
        latent_upsample: widgetStr(src, "latent_upsample", "bilinear"),
        rho: widgetNum(src, "rho", 0),
        w_min: widgetNum(src, "w_min", 0.5),
        w_max: widgetNum(src, "w_max", 1),
        enable_latent_chunking: widgetBool(src, "enable_latent_chunking", false),
        enable_tiling: widgetBool(src, "enable_tiling", false),
        tile_count: widgetNum(src, "tile_count", 2),
        tile_overlap: widgetNum(src, "tile_overlap", 128),
        sample_model: hiresModelLinked(src) ? true : null,
    };
}

function syncSelfLiftWidgets(node) {
    if (!isSelfLiftNode(node)) return;
    const split = String(widgetValue(widgetByName(node, "split_mode")) || "highres_steps");
    const trans = split === "transition_step";
    setWidgetVisible(node, "highres_steps", !trans);
    setWidgetVisible(node, "transition_step", trans);
    const rho = Number(widgetValue(widgetByName(node, "rho")) || 0);
    const pixel = rho > 1e-8;
    setWidgetVisible(node, "w_min", pixel);
    setWidgetVisible(node, "w_max", pixel);
    // Keep the widgets in the list (Vue drops them if they vanish), then hide
    // by flag so turning tiling off collapses tile_count / tile_overlap.
    ensureIntWidget(node, "tile_count", 2, { min: 1, max: 8, step: 1, precision: 0 }, "enable_tiling");
    ensureIntWidget(node, "tile_overlap", 128, { min: 0, max: 2048, step: 64, precision: 0 }, "tile_count");
    const tiling = widgetBool(node, "enable_tiling", false);
    setWidgetVisible(node, "tile_count", tiling);
    setWidgetVisible(node, "tile_overlap", tiling);
    for (const name of ["enable_tiling", "tile_count", "tile_overlap"]) {
        restoreDefaultLabel(node, widgetByName(node, name));
    }
    bumpWidgetList(node);
    try {
        fitSelfLiftNode(node);
    } catch {
        /* ignore */
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function hookSelfLiftWidget(node, name) {
    const w = widgetByName(node, name);
    if (!w || w._mmxSelfLiftHooked) return;
    w._mmxSelfLiftHooked = true;
    const orig = w.callback;
    w.callback = function (...args) {
        if (typeof orig === "function") orig.apply(this, args);
        syncSelfLiftWidgets(node);
    };
}

function installSelfLiftWidgets(node) {
    if (!isSelfLiftNode(node)) return;
    for (const name of ["split_mode", "rho", "enable_tiling"]) {
        hookSelfLiftWidget(node, name);
    }
    syncSelfLiftWidgets(node);
}

function scheduleSelfLiftRefresh(node) {
    installSelfLiftWidgets(node);
    queueMicrotask(() => installSelfLiftWidgets(node));
    setTimeout(() => installSelfLiftWidgets(node), 0);
    setTimeout(() => installSelfLiftWidgets(node), 80);
    setTimeout(() => installSelfLiftWidgets(node), 250);
}

function graphNodes() {
    const graph = app.graph ?? app.canvas?.graph;
    return graph?._nodes ?? graph?.nodes ?? [];
}

function refreshLinkedDirectorCache(delay = 120) {
    for (const node of graphNodes()) {
        const cls = node?.comfyClass || node?.type || "";
        if (!DIRECTOR_CLASSES.has(cls)) continue;
        if (typeof node._mmxRefreshFirstPassCache === "function") {
            node._mmxRefreshFirstPassCache(delay);
        }
    }
}

app.registerExtension({
    name: "minimax.h3.director.selflift",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== SELFLIFT_CLASS) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            scheduleSelfLiftRefresh(this);
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            scheduleSelfLiftRefresh(this);
            return result;
        };
        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (...args) {
            const result = onWidgetChanged?.apply(this, args);
            syncSelfLiftWidgets(this);
            refreshLinkedDirectorCache();
            return result;
        };
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = onConnectionsChange?.apply(this, args);
            refreshLinkedDirectorCache();
            return result;
        };
    },
    nodeCreated(node) {
        scheduleSelfLiftRefresh(node);
    },
    loadedGraphNode(node) {
        scheduleSelfLiftRefresh(node);
    },
    afterConfigureGraph() {
        for (const node of graphNodes()) {
            scheduleSelfLiftRefresh(node);
        }
    },
});
