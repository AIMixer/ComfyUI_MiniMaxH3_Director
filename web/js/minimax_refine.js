/** MiniMax H3 Director Refine — show canvas widgets like Director output bar. */

import { app } from "../../scripts/app.js";
import {
    CUSTOM_ASPECT_RATIO,
    resolutionFromSelector,
    snapResolutionDim,
} from "./minimax_gen_timeline.js";

const REFINE_CLASS = "MiniMaxH3DirectorRefine";
const FOLLOW_DIRECTOR_ASPECT = "跟随导演台";

function isRefineNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return cls === REFINE_CLASS;
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

function setWidgetVisible(node, name, visible) {
    const w = widgetByName(node, name);
    if (!w) return;
    w.hidden = !visible;
    if (!w.options) w.options = {};
    w.options.hidden = !visible;
    if (visible) {
        if (w._mmxOrigComputeSize) {
            w.computeSize = w._mmxOrigComputeSize;
            delete w._mmxOrigComputeSize;
        } else if (w.computeSize) {
            delete w.computeSize;
        }
        if (w.element) w.element.style.display = "";
    } else {
        if (!w._mmxOrigComputeSize && typeof w.computeSize === "function") {
            w._mmxOrigComputeSize = w.computeSize.bind(w);
        }
        w.computeSize = () => [0, -4];
        if (w.element) w.element.style.display = "none";
    }
}

function isCustomAspect(value) {
    const v = String(value ?? "").trim();
    return v === CUSTOM_ASPECT_RATIO || v === "Custom" || v.startsWith("自定义");
}

const ASPECT_CHOICES = new Set([
    FOLLOW_DIRECTOR_ASPECT,
    "Follow Director",
    CUSTOM_ASPECT_RATIO,
    "Custom",
    "1:1 (方形)",
    "2:3 (竖版照片)",
    "3:2 (横版照片)",
    "3:4 (竖版标准)",
    "4:3 (标准)",
    "9:16 (竖屏)",
    "16:9 (宽屏)",
    "21:9 (超宽)",
]);

const UPSCALE_METHOD_VALUES = new Set(["lanczos", "nvidia_rtx_vsr", "latent"]);
const SEED_MODE_VALUES = new Set(["inherit", "offset"]);

function looksLikeUpscaleMethod(value) {
    return UPSCALE_METHOD_VALUES.has(String(value ?? "").trim().toLowerCase());
}

function looksLikeSeedMode(value) {
    return SEED_MODE_VALUES.has(String(value ?? "").trim().toLowerCase());
}

function looksLikeNumber(value) {
    if (typeof value === "number") return Number.isFinite(value);
    if (typeof value === "string" && value.trim() !== "") {
        return Number.isFinite(Number(value));
    }
    return false;
}

function moveWidgetAfter(node, name, afterName) {
    const widgets = node.widgets;
    if (!Array.isArray(widgets)) return;
    const from = widgets.findIndex((w) => w.name === name);
    const after = widgets.findIndex((w) => w.name === afterName);
    if (from < 0 || after < 0 || from === after + 1) return;
    const [widget] = widgets.splice(from, 1);
    const insertAt = widgets.findIndex((w) => w.name === afterName) + 1;
    widgets.splice(insertAt, 0, widget);
}

function looksLikeAspectChoice(value) {
    const v = String(value ?? "").trim();
    return ASPECT_CHOICES.has(v) || v === FOLLOW_DIRECTOR_ASPECT || v === "Follow Director";
}

function clampPasses(value) {
    const n = Math.round(Number(value));
    if (!Number.isFinite(n) || n < 1) return 1;
    return Math.min(9999, n);
}

function migrateRefineWidgetOrder(node) {
    // A: mode, denoise, steps, seed, aspect, mp, width, height, skip, method
    // B: mode, method, denoise, steps, seed, aspect, mp, width, height, skip
    // C: mode, method, denoise, steps, passes, seed, aspect, mp, width, height, skip
    const methodW = widgetByName(node, "upscale_method");
    const denoiseW = widgetByName(node, "denoise");
    const stepsW = widgetByName(node, "steps");
    const passesW = widgetByName(node, "passes");
    const seedW = widgetByName(node, "seed_mode");
    const aspectW = widgetByName(node, "aspect_ratio");
    const mpW = widgetByName(node, "megapixels");
    const widthW = widgetByName(node, "width");
    const heightW = widgetByName(node, "height");
    const skipW = widgetByName(node, "skip_fl2v");
    if (!methodW || !denoiseW || !stepsW || !skipW) return;

    const methodV = widgetValue(methodW);
    const denoiseV = widgetValue(denoiseW);
    const stepsV = widgetValue(stepsW);
    const passesV = passesW ? widgetValue(passesW) : undefined;
    const seedV = seedW ? widgetValue(seedW) : undefined;
    const aspectV = aspectW ? widgetValue(aspectW) : undefined;
    const mpV = mpW ? widgetValue(mpW) : undefined;
    const widthV = widthW ? widgetValue(widthW) : undefined;
    const heightV = heightW ? widgetValue(heightW) : undefined;
    const skipV = widgetValue(skipW);

    const apply = (vals) => {
        methodW.value = looksLikeUpscaleMethod(vals.method) ? vals.method : "lanczos";
        denoiseW.value = Number(vals.denoise);
        stepsW.value = Number(vals.steps);
        if (passesW) passesW.value = clampPasses(vals.passes);
        if (seedW) seedW.value = looksLikeSeedMode(vals.seed) ? vals.seed : "inherit";
        if (aspectW && vals.aspect !== undefined) aspectW.value = vals.aspect;
        if (mpW && vals.mp !== undefined) mpW.value = vals.mp;
        if (widthW && vals.width !== undefined) widthW.value = vals.width;
        if (heightW && vals.height !== undefined) heightW.value = vals.height;
        if (vals.skip === true || vals.skip === false) skipW.value = vals.skip;
    };

    if (looksLikeUpscaleMethod(methodV)) {
        // B → C: passes slot received seed_mode
        if (passesW && looksLikeSeedMode(passesV)) {
            apply({
                method: methodV,
                denoise: denoiseV,
                steps: stepsV,
                passes: 1,
                seed: passesV,
                aspect: seedV,
                mp: aspectV,
                width: mpV,
                height: widthV,
                skip: heightV,
            });
        } else if (passesW) {
            passesW.value = clampPasses(passesV);
        }
        return;
    }

    const shiftedA =
        looksLikeNumber(methodV) &&
        (looksLikeSeedMode(stepsV) || looksLikeUpscaleMethod(skipV) || looksLikeUpscaleMethod(heightV));
    if (!shiftedA) {
        if (passesW) passesW.value = clampPasses(passesV);
        return;
    }

    if (passesW && looksLikeAspectChoice(passesV)) {
        // A save loaded onto C widgets (aspect landed on passes)
        apply({
            method: heightV,
            denoise: methodV,
            steps: denoiseV,
            passes: 1,
            seed: stepsV,
            aspect: passesV,
            mp: seedV,
            width: aspectV,
            height: mpV,
            skip: widthV,
        });
        return;
    }

    // A save loaded onto B widgets (no passes yet / passes still default)
    apply({
        method: skipV,
        denoise: methodV,
        steps: denoiseV,
        passes: 1,
        seed: stepsV,
        aspect: seedV,
        mp: aspectV,
        width: mpV,
        height: widthV,
        skip: heightV,
    });
}

function migrateRefineWidgets(node) {
    migrateRefineWidgetOrder(node);
    moveWidgetAfter(node, "upscale_method", "mode");
    moveWidgetAfter(node, "passes", "steps");
    const aspectW = widgetByName(node, "aspect_ratio");
    const mpW = widgetByName(node, "megapixels");
    const widthW = widgetByName(node, "width");
    const heightW = widgetByName(node, "height");
    if (aspectW && !ASPECT_CHOICES.has(widgetValue(aspectW))) {
        aspectW.value = FOLLOW_DIRECTOR_ASPECT;
    }
    if (mpW) {
        const n = Number(widgetValue(mpW));
        if (!Number.isFinite(n) || n < 0.1) mpW.value = 1.0;
    }
    if (widthW) {
        const n = Number(widgetValue(widthW));
        if (!Number.isFinite(n) || n < 32) widthW.value = 1280;
    }
    if (heightW) {
        const n = Number(widgetValue(heightW));
        if (!Number.isFinite(n) || n < 32) heightW.value = 720;
    }
}

function isFollowAspect(value) {
    const v = String(value ?? "").trim();
    if (v === "0" || v === "0.0") return true;
    return !v || v === FOLLOW_DIRECTOR_ASPECT || v === "Follow Director";
}

function readMode(node) {
    const named = widgetByName(node, "mode");
    const raw = String(widgetValue(named) ?? "").toLowerCase();
    if (raw.includes("upscale")) return "upscale";
    if (raw.includes("refine")) return "refine";
    for (const w of node.widgets || []) {
        const s = String(widgetValue(w) ?? "").toLowerCase();
        if (s === "upscale") return "upscale";
        if (s === "refine") return "refine";
    }
    return null;
}

function syncRefineComputedSize(node) {
    const aspectW = widgetByName(node, "aspect_ratio");
    const mpW = widgetByName(node, "megapixels");
    const widthW = widgetByName(node, "width");
    const heightW = widgetByName(node, "height");
    if (!aspectW || isFollowAspect(widgetValue(aspectW)) || isCustomAspect(widgetValue(aspectW))) return;
    const resolved = resolutionFromSelector(widgetValue(aspectW), widgetValue(mpW) ?? 1.0);
    if (!resolved) return;
    if (widthW) widthW.value = resolved.width;
    if (heightW) heightW.value = resolved.height;
}

function syncRefineWidgetVisibility(node) {
    const mode = readMode(node);
    const upscale = mode === "upscale";
    const aspect = widgetValue(widgetByName(node, "aspect_ratio"));
    const follow = isFollowAspect(aspect);
    const custom = isCustomAspect(aspect);
    setWidgetVisible(node, "aspect_ratio", upscale);
    setWidgetVisible(node, "megapixels", upscale && !follow && !custom);
    setWidgetVisible(node, "width", upscale && custom);
    setWidgetVisible(node, "height", upscale && custom);
    setWidgetVisible(node, "upscale_method", upscale);
    setWidgetVisible(node, "target_width", false);
    setWidgetVisible(node, "target_height", false);
    if (upscale && !follow && !custom) syncRefineComputedSize(node);
    try {
        const size = node.computeSize?.();
        if (Array.isArray(size) && size.length >= 2) {
            node.setSize?.([node.size?.[0] || size[0], size[1]]);
        }
    } catch {
        /* ignore */
    }
    node.setDirtyCanvas?.(true, true);
}

function hookWidget(node, name, fn) {
    if (!node._mmxRefineHooked) node._mmxRefineHooked = new Set();
    if (node._mmxRefineHooked.has(name)) return;
    const w = widgetByName(node, name);
    if (!w) return;
    node._mmxRefineHooked.add(name);
    const prev = w.callback;
    w.callback = function (...args) {
        const r = prev?.apply(this, args);
        fn();
        return r;
    };
}

function installRefineResolutionUI(node) {
    const onAspect = () => {
        const aspectW = widgetByName(node, "aspect_ratio");
        const widthW = widgetByName(node, "width");
        const heightW = widgetByName(node, "height");
        if (aspectW && isCustomAspect(widgetValue(aspectW)) && widthW && heightW) {
            widthW.value = snapResolutionDim(widgetValue(widthW) || 1280);
            heightW.value = snapResolutionDim(widgetValue(heightW) || 720);
        }
        syncRefineWidgetVisibility(node);
    };
    hookWidget(node, "mode", () => syncRefineWidgetVisibility(node));
    hookWidget(node, "aspect_ratio", onAspect);
    hookWidget(node, "megapixels", () => syncRefineComputedSize(node));
    hookWidget(node, "width", () => {
        const w = widgetByName(node, "width");
        if (w) w.value = snapResolutionDim(widgetValue(w));
    });
    hookWidget(node, "height", () => {
        const w = widgetByName(node, "height");
        if (w) w.value = snapResolutionDim(widgetValue(w));
    });
    if (!node._mmxRefineOnWidgetChanged) {
        node._mmxRefineOnWidgetChanged = true;
        const prev = node.onWidgetChanged;
        node.onWidgetChanged = function (name, ...rest) {
            const r = prev?.apply(this, [name, ...rest]);
            if (name === "mode" || name === "aspect_ratio" || name === "megapixels") {
                migrateRefineWidgets(this);
                syncRefineWidgetVisibility(this);
            }
            return r;
        };
    }
}

function refreshRefineNode(node) {
    if (!isRefineNode(node)) return;
    installRefineResolutionUI(node);
    migrateRefineWidgets(node);
    syncRefineWidgetVisibility(node);
}

function refreshAllRefineNodes() {
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        refreshRefineNode(node);
    }
}

function scheduleRefineRefresh(node) {
    refreshRefineNode(node);
    queueMicrotask(() => refreshRefineNode(node));
    setTimeout(() => refreshRefineNode(node), 0);
    setTimeout(() => refreshRefineNode(node), 80);
    setTimeout(() => refreshRefineNode(node), 250);
}

app.registerExtension({
    name: "ComfyUI.MiniMaxH3DirectorRefine",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== REFINE_CLASS) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const r = onNodeCreated?.apply(this, args);
            scheduleRefineRefresh(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const r = onConfigure?.apply(this, args);
            scheduleRefineRefresh(this);
            return r;
        };
    },
    nodeCreated(node) {
        scheduleRefineRefresh(node);
    },
    loadedGraphNode(node) {
        scheduleRefineRefresh(node);
    },
    afterConfigureGraph() {
        refreshAllRefineNodes();
        setTimeout(refreshAllRefineNodes, 100);
    },
});
