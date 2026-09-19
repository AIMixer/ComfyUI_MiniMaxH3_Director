/** MiniMax H3 Director Refine — show canvas widgets like Director output bar. */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    CUSTOM_ASPECT_RATIO,
    resolutionFromSelector,
    snapResolutionDim,
} from "./minimax_gen_timeline.js";
import { injectExternalGroupsWitness } from "./minimax_external_witness.js";
import { collectSelfLiftWitness } from "./minimax_selflift.js";
import { collectSemanticBridgeWitness } from "./minimax_semantic_bridge.js";

const REFINE_CLASS = "MiniMaxH3DirectorRefine";
const SELFLIFT_CLASS = "MiniMaxH3DirectorSelfLift";
const SEMANTIC_BRIDGE_CLASS = "MiniMaxH3DirectorSemanticBridge";
const DIRECTOR_CLASSES = new Set(["MiniMaxH3Director", "ComfyMiniMaxH3Director"]);
const CACHE_STATUS_WIDGET = "first_pass_cache_status";
const FOLLOW_DIRECTOR_ASPECT = "跟随导演台";
const PACKER_CLASSES = new Set([SELFLIFT_CLASS, SEMANTIC_BRIDGE_CLASS]);

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

const UPSCALE_METHOD_VALUES = new Set(["lanczos", "nvidia_rtx_vsr", "h3_latent"]);
const SEED_MODE_VALUES = new Set(["inherit", "offset"]);
const SAMPLER_HINTS = new Set([
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
    "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
    "deis", "res_multistep", "res_multistep_ancestral", "gradient_estimation",
    "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
    "uni_pc", "uni_pc_bh2", "ddim",
]);

function looksLikeUpscaleMethod(value) {
    return UPSCALE_METHOD_VALUES.has(String(value ?? "").trim().toLowerCase());
}

function looksLikeSampler(value) {
    return SAMPLER_HINTS.has(String(value ?? "").trim().toLowerCase());
}

function clampPasses(value) {
    const n = Math.round(Number(value));
    if (!Number.isFinite(n) || n < 1) return 1;
    return Math.min(9999, n);
}

function migrateRefineWidgetOrder(node) {
    const samplerW = widgetByName(node, "sampler");
    const passesW = widgetByName(node, "passes");
    const methodW = widgetByName(node, "upscale_method");
    if (samplerW && !looksLikeSampler(widgetValue(samplerW))) {
        samplerW.value = "euler";
    }
    if (passesW) {
        passesW.value = clampPasses(widgetValue(passesW));
    }
    if (methodW && !looksLikeUpscaleMethod(widgetValue(methodW))) {
        methodW.value = "h3_latent";
    }
}

function migrateLegacyPrePassesValues(node) {
    const seedW = widgetByName(node, "seed_mode");
    const aspectW = widgetByName(node, "aspect_ratio");
    const mpW = widgetByName(node, "megapixels");
    const widthW = widgetByName(node, "width");
    const heightW = widgetByName(node, "height");
    const skipW = widgetByName(node, "skip_fl2v");
    const rawSeed = widgetValue(seedW);
    if (!seedW || SEED_MODE_VALUES.has(String(rawSeed ?? "").trim().toLowerCase())) return;

    // Workflows saved before `passes` was inserted load every following value
    // one slot early: seed_mode gets the aspect ratio, aspect gets MP, etc.
    if (ASPECT_CHOICES.has(rawSeed)) {
        const rawAspect = widgetValue(aspectW);
        const rawMp = widgetValue(mpW);
        const rawWidth = widgetValue(widthW);
        const rawHeight = widgetValue(heightW);
        seedW.value = "inherit";
        if (aspectW) aspectW.value = rawSeed;
        const mp = Number(rawAspect);
        if (mpW && Number.isFinite(mp) && mp >= 0.1 && mp <= 16) mpW.value = mp;
        const width = Number(rawMp);
        if (widthW && Number.isFinite(width) && width >= 32 && width <= 8192) widthW.value = width;
        const height = Number(rawWidth);
        if (heightW && Number.isFinite(height) && height >= 32 && height <= 8192) heightW.value = height;
        if (skipW && (rawHeight === true || rawHeight === false)) skipW.value = rawHeight;
        node._mmxRecoveredLegacyRefineValues = true;
        return;
    }
    seedW.value = "inherit";
}

function migrateRefineWidgets(node) {
    migrateLegacyPrePassesValues(node);
    migrateRefineWidgetOrder(node);
    const seedW = widgetByName(node, "seed_mode");
    const aspectW = widgetByName(node, "aspect_ratio");
    const mpW = widgetByName(node, "megapixels");
    const widthW = widgetByName(node, "width");
    const heightW = widgetByName(node, "height");
    if (seedW && !SEED_MODE_VALUES.has(String(widgetValue(seedW) ?? "").trim().toLowerCase())) {
        seedW.value = "inherit";
    }
    if (aspectW && !ASPECT_CHOICES.has(widgetValue(aspectW))) {
        aspectW.value = FOLLOW_DIRECTOR_ASPECT;
    }
    if (mpW) {
        const n = Number(widgetValue(mpW));
        if (!Number.isFinite(n) || n < 0.1 || n > 16) mpW.value = 1.0;
    }
    if (widthW) {
        const n = Number(widgetValue(widthW));
        if (!Number.isFinite(n) || n < 32 || n > 8192) widthW.value = 1280;
    }
    if (heightW) {
        const n = Number(widgetValue(heightW));
        if (!Number.isFinite(n) || n < 32 || n > 8192) heightW.value = 720;
    }
    setWidgetVisible(node, "schedule", false);
    setWidgetVisible(node, "denoise", false);
    setWidgetVisible(node, "steps", false);
    setWidgetVisible(node, "sigmas_text", false);
    setWidgetVisible(node, "sigmas", false);
    setWidgetVisible(node, "h3_latent_model", false);
    setWidgetVisible(node, "upscale_model", false);
}

function isFollowAspect(value) {
    const v = String(value ?? "").trim();
    if (v === "0" || v === "0.0") return true;
    return !v || v === FOLLOW_DIRECTOR_ASPECT || v === "Follow Director";
}

function readMode(node) {
    const named = widgetByName(node, "mode");
    const raw = String(widgetValue(named) ?? "").toLowerCase();
    if (raw.includes("latent_upscale") || raw.includes("latent")) return "latent_upscale";
    if (raw.includes("upscale")) return "upscale";
    if (raw.includes("refine")) return "refine";
    for (const w of node.widgets || []) {
        const s = String(widgetValue(w) ?? "").toLowerCase();
        if (s === "latent_upscale") return "latent_upscale";
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

function readUpscaleMethod(node) {
    return String(widgetValue(widgetByName(node, "upscale_method")) ?? "").trim().toLowerCase();
}

function boolWidgetValue(node, name) {
    const value = widgetValue(widgetByName(node, name));
    return value === true || value === 1 || String(value).toLowerCase() === "true";
}

function graphNodes() {
    const graph = app.graph ?? app.canvas?.graph;
    return graph?._nodes ?? graph?.nodes ?? [];
}

function connectedDirector(refineNode) {
    const graph = refineNode?.graph ?? app.graph ?? app.canvas?.graph;
    for (const candidate of graphNodes()) {
        const cls = candidate?.comfyClass || candidate?.type || "";
        if (!DIRECTOR_CLASSES.has(cls)) continue;
        const input = candidate.inputs?.find((item) => item?.name === "refine");
        if (input?.link == null) continue;
        const link = graph?.links?.[input.link] ?? graph?._links?.[input.link];
        if (String(link?.origin_id) === String(refineNode.id)) return candidate;
    }
    return null;
}

function directorValue(node, name, fallback) {
    const value = widgetValue(widgetByName(node, name));
    return value == null || value === "" ? fallback : value;
}

/** Fingerprint key → what the user actually changed. */
const CACHE_DIFF_LABELS = {
    seed: "seed",
    start: "片段起点",
    end: "片段终点（时间范围变化）",
    prompt: "提示词",
    negative: "反向提示词",
    task_key: "生成模式",
    width: "宽度",
    height: "高度",
    frame_rate: "帧率",
    output_mode: "输出模式",
    refs: "参考图片",
    ref_audios: "参考音频",
    ref_videos: "参考视频",
    ref_video: "参考视频",
    ref_video_start: "参考视频起点",
    source_video: "源视频",
    continuity: "段间连续性",
    continuity_overlap: "上下文帧数",
    continuity_mode: "引导方式",
    continuity_redraw: "重绘幅度",
    continuity_keep_tail: "保完整",
    cfg: "CFG",
    steps: "一采步数",
    sampler: "一采采样器",
    scheduler: "调度器",
    sigmas: "一采噪声表",
    sigmas_source: "一采 SIGMAS 接线",
    shift_video: "视频 shift",
    shift_audio: "音频 shift",
    "<invalid-meta>": "缓存信息损坏",
    external_wiring: "外接组接线",
    external_prompt: "外接组提示词",
    external_length: "外接组时长",
    external_shift: "外接组时间轴推移",
    external_media: "外接组参考素材",
    external_other: "外接组其他参数",
    external_groups_off: "外接组（缓存写入后接线已断开）",
    "<unverified-external>": "未记录外接组（旧版写入）",
    selflift: "SelfLift",
    sl_split: "SelfLift 分段方式",
    sl_high: "SelfLift 高清步数",
    sl_trans: "SelfLift 过渡步",
    sl_scale: "SelfLift 低清倍率",
    sl_model: "SelfLift 3D 权重",
    sl_samp: "SelfLift 采样器",
    sl_carry: "SelfLift 低清承接",
    sl_rho: "SelfLift rho",
    sl_wmin: "SelfLift w_min",
    sl_wmax: "SelfLift w_max",
    sl_up: "SelfLift 插值",
    sl_chunk: "SelfLift 时间分块",
    sl_tile: "SelfLift 空间分块",
    sl_tiles: "SelfLift 分块数",
    sl_overlap: "SelfLift 分块重叠",
    sl_hires_model: "SelfLift 高清模型",
    semantic_bridge: "Semantic Bridge",
    sb_adapter: "Semantic Bridge 权重",
    sb_alpha: "Semantic Bridge alpha",
    sb_mag: "Semantic Bridge magnitude_match",
    refine: "Refine",
    refine_mode: "二采模式",
    refine_passes: "二采次数",
    refine_seed_mode: "二采 seed",
    refine_target: "二采目标画布",
    refine_sampler: "二采采样器",
    refine_sigmas: "二采噪声表",
    refine_sigmas_wired: "二采 SIGMAS 接线",
    refine_sample_model: "二采模型",
    refine_skip_fl2v: "跳过 fl2v 二采",
};

function diffLabel(key) {
    return CACHE_DIFF_LABELS[key] || key;
}

/**
 * One compact line naming the segments that cannot be reused. 1 group = 1
 * segment = 1 cache slot, and a segment is compared against its own group only,
 * so an edit to one group leaves the others matching — listing every segment
 * would bury that (and the rest of the panel) under a dozen lines.
 */
function externalMismatchLine(data) {
    const rows = Array.isArray(data?.segments) ? data.segments : [];
    const bad = rows.filter((row) => !row?.matches);
    if (!bad.length) return "";
    const reasons = (row) => {
        const keys = (Array.isArray(row?.diff_keys) ? row.diff_keys : [])
            .filter((key) => key !== "<missing-cache>");
        return keys.length ? keys.map(diffLabel).join("、") : "无缓存";
    };
    if (bad.length === rows.length && rows.length > 1) {
        const all = [...new Set(bad.flatMap((row) => reasons(row).split("、")))].join("、");
        return `不匹配：全部 ${rows.length} 段（${all}）`;
    }
    const parts = bad.slice(0, 4).map(
        (row) => `${row?.slot || `第 ${row?.segment} 段`}（${reasons(row)}）`,
    );
    if (bad.length > parts.length) parts.push(`…共 ${bad.length} 段`);
    return `不匹配：${parts.join("、")}`;
}

function directorHasSigmasLink(node) {
    const inp = (node?.inputs || []).find((i) => String(i.name) === "sigmas");
    if (!inp) return false;
    if (inp.link != null) return true;
    return Array.isArray(inp.links) && inp.links.length > 0;
}

function inputLinked(node, name) {
    const inp = (node?.inputs || []).find((i) => String(i.name) === name);
    if (!inp) return false;
    if (inp.link != null) return true;
    return Array.isArray(inp.links) && inp.links.length > 0;
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

function collectRefineWitness(refine) {
    if (!refine) return null;
    return {
        enabled: true,
        mode: readMode(refine) || "refine",
        upscale_method: readUpscaleMethod(refine) || "h3_latent",
        sampler: widgetStr(refine, "sampler", ""),
        passes: widgetNum(refine, "passes", 1),
        seed_mode: widgetStr(refine, "seed_mode", "inherit"),
        aspect_ratio: widgetStr(refine, "aspect_ratio", FOLLOW_DIRECTOR_ASPECT),
        megapixels: widgetNum(refine, "megapixels", 1),
        width: widgetNum(refine, "width", 0),
        height: widgetNum(refine, "height", 0),
        skip_fl2v: widgetByName(refine, "skip_fl2v") == null
            ? true
            : boolWidgetValue(refine, "skip_fl2v"),
        confirm_first_pass: boolWidgetValue(refine, "confirm_first_pass"),
        enable_latent_chunking: boolWidgetValue(refine, "enable_latent_chunking"),
        enable_tiling: boolWidgetValue(refine, "enable_tiling"),
        tile_count: widgetNum(refine, "tile_count", 2),
        tile_overlap: widgetNum(refine, "tile_overlap", 128),
        latent_upscale_model: widgetStr(refine, "latent_upscale_model", ""),
        has_sample_model: inputLinked(refine, "refine_model") || inputLinked(refine, "model"),
        has_upscale_model: inputLinked(refine, "upscale_model"),
        has_sigmas_tensor: inputLinked(refine, "sigmas"),
    };
}

const DIFF_PRIORITY = [
    "seed",
    "semantic_bridge",
    "sb_adapter",
    "sb_alpha",
    "sb_mag",
    "prompt",
    "end",
    "height",
    "width",
    "refs",
    "ref_max",
];

function sortDiffKeys(keys) {
    return [...keys].sort((a, b) => {
        const ia = DIFF_PRIORITY.indexOf(a);
        const ib = DIFF_PRIORITY.indexOf(b);
        if (ia === -1 && ib === -1) return String(a).localeCompare(String(b));
        if (ia === -1) return 1;
        if (ib === -1) return -1;
        return ia - ib;
    });
}

function cacheStatusPayload(director, refine) {
    try {
        director?._minimaxEditor?._writeTimelineWidget?.();
    } catch {
        /* best effort */
    }
    return {
        node_id: String(director.id),
        // Keep the graph-wired external-group witness current: the status route
        // runs on the backend where i2v_groups / r2v_groups links are invisible.
        timeline_data: injectExternalGroupsWitness(
            director,
            String(directorValue(director, "timeline_data", "")),
        ),
        task_type: String(directorValue(director, "task_type", "")),
        global_prompt: String(directorValue(director, "global_prompt", "")),
        total_frames: Number(directorValue(director, "total_frames", 124)),
        frame_rate: Number(directorValue(director, "frame_rate", 24)),
        width: Number(directorValue(director, "width", 864)),
        height: Number(directorValue(director, "height", 480)),
        ref_max_size: Number(directorValue(director, "ref_max_size", 864)),
        seed: Number(directorValue(director, "seed", 0)),
        cfg: Number(directorValue(director, "cfg", 1)),
        steps: Number(directorValue(director, "steps", 25)),
        sampler: String(directorValue(director, "sampler", "")),
        scheduler: String(directorValue(director, "scheduler", "")),
        shift_video: Number(directorValue(director, "shift_video", 12)),
        shift_audio: Number(directorValue(director, "shift_audio", 3)),
        sigmas_linked: directorHasSigmasLink(director),
        // SelfLift is a graph-wired pack, not a Director widget. The run writes
        // sl_* into first-pass meta; the panel must send the same pack or it
        // always reports those keys as diffs.
        selflift: collectSelfLiftWitness(director),
        semantic_bridge: collectSemanticBridgeWitness(director),
        refine: collectRefineWitness(refine),
    };
}

function renderCacheStatus(node, data, kind = "normal") {
    const ui = node._mmxFirstPassCacheUI;
    if (!ui) return;
    const colors = {
        normal: "var(--input-text, #ddd)",
        ok: "#65d68a",
        warn: "#f0bd58",
        error: "#ef7777",
        muted: "#aaa",
    };
    ui.body.style.color = colors[kind] || colors.normal;
    if (typeof data === "string") {
        ui.body.textContent = data;
        return;
    }
    const total = Number(data?.segment_total || 0);
    const cached = Number(data?.cached_count || 0);
    const matched = Number(data?.matched_count || 0);
    const seeds = Array.isArray(data?.cached_seeds) && data.cached_seeds.length
        ? data.cached_seeds.join(", ")
        : "—";
    const diffLabels = CACHE_DIFF_LABELS;
    const diffs = sortDiffKeys(
        (Array.isArray(data?.diff_keys) ? data.diff_keys : [])
            .filter((key) => key !== "<missing-cache>"),
    )
        .slice(0, 12)
        .map((key) => diffLabels[key] || key);
    const selTotal = data?.selected_total;
    const selMatched = data?.selected_matched;
    const selActive = Number.isFinite(selTotal) && Number(selTotal) !== total;
    const lines = [
        `一采缓存：${data?.exists ? `存在（${cached}/${total} 段）` : "不存在"}`,
        `一采匹配：${data?.matches ? `是（${matched}/${total} 段）` : "否"}`
            + (selActive ? ` · 选中 ${selMatched ?? 0}/${selTotal ?? 0}` : ""),
    ];
    const confirmOn = Boolean(data?.confirm_first_pass);
    const canConfirm = Boolean(data?.can_confirm_refine);
    const confirmReason = String(data?.confirm_refine_reason || "");
    if (!confirmOn) {
        lines.push("确认二采：未开启「先确认一采」— Queue 会一采+二采连续跑");
    } else if (canConfirm) {
        lines.push("确认二采：可以 — 一采指纹匹配（含 Semantic Bridge），Queue 将跳过一采只跑二采");
    } else if (confirmReason === "refine_skipped") {
        lines.push("确认二采：不可以 — 当前选中段不会跑二采（如 skip_fl2v）");
    } else {
        lines.push("确认二采：不可以 — 一采指纹不匹配，Queue 只会写一采 / 重跑一采");
    }
    lines.push(`缓存 seed：${seeds}`);
    lines.push(`当前 seed：${data?.current_seed ?? "—"}`);
    const finalCached = Number(data?.final_cached_count || 0);
    const finalMatched = Number(data?.final_matched_count || 0);
    lines.push(
        `成片匹配：${finalMatched}/${total} 段`
            + (finalCached !== finalMatched ? `（磁盘仍有 ${finalCached} 段旧成片，不会当这一轮二采复用）` : ""),
    );
    if (diffs.length) lines.push(`一采差异：${diffs.join("、")}`);
    if (data?.mode === "external_groups") {
        // External groups are not on the timeline: each segment is compared
        // against its own group (prompt / duration / its reference slots) plus
        // the plan-level knobs.
        lines.push("核对方式：外接组");
        const mismatch = externalMismatchLine(data);
        if (mismatch) lines.push(mismatch);
    }
    const unverified = Number(data?.unverified_count || 0);
    if (unverified > 0) {
        lines.push(`提示：${unverified} 段缓存未记录外接组信息（旧版写入），这些段会重采一次`);
    }
    const staleExternal = Number(data?.stale_external_count || 0);
    if (staleExternal > 0) {
        lines.push(
            `提示：${staleExternal} 段缓存由外接组计划写入，当前未检测到外接组接线，这些段会重采一采`,
        );
    }
    ui.body.textContent = lines.join("\n");
}

async function refreshFirstPassCacheStatus(node) {
    if (!isRefineNode(node)) return;
    ensureFirstPassCacheUI(node);
    const director = connectedDirector(node);
    if (!director) {
        renderCacheStatus(node, "未找到相连的 MiniMax H3 Director。", "warn");
        return;
    }
    const seq = (node._mmxCacheStatusSeq || 0) + 1;
    node._mmxCacheStatusSeq = seq;
    renderCacheStatus(node, "正在检查分段缓存…", "muted");
    try {
        const response = await api.fetchApi("/minimax/director/first_pass_cache_status", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(cacheStatusPayload(director, node)),
        });
        const data = await response.json();
        if (seq !== node._mmxCacheStatusSeq) return;
        if (!response.ok || data?.error) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        const confirmOn = Boolean(data?.confirm_first_pass);
        const tone = confirmOn
            ? (data.can_confirm_refine ? "ok" : (data.exists ? "warn" : "muted"))
            : (data.matches ? "ok" : (data.exists ? "warn" : "muted"));
        renderCacheStatus(node, data, tone);
    } catch (error) {
        if (seq !== node._mmxCacheStatusSeq) return;
        renderCacheStatus(node, `缓存检查失败：${error?.message || error}`, "error");
    }
}

function scheduleCacheStatusRefresh(node, delay = 120) {
    clearTimeout(node._mmxCacheStatusTimer);
    node._mmxCacheStatusTimer = setTimeout(() => refreshFirstPassCacheStatus(node), delay);
}

function refreshCacheStatusForDirector(director, delay = 120) {
    for (const node of graphNodes()) {
        if (
            isRefineNode(node)
            && connectedDirector(node) === director
        ) {
            scheduleCacheStatusRefresh(node, delay);
        }
    }
}

function ensureFirstPassCacheUI(node) {
    if (node._mmxFirstPassCacheUI || typeof node.addDOMWidget !== "function") return;
    const root = document.createElement("div");
    root.style.cssText = [
        "box-sizing:border-box",
        "margin:4px 8px",
        "padding:8px 10px",
        "border:1px solid var(--border-color, #555)",
        "border-radius:6px",
        "background:rgba(0,0,0,.16)",
        "font:12px/1.45 sans-serif",
    ].join(";");
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;margin-bottom:5px";
    const title = document.createElement("strong");
    title.textContent = "分段缓存状态";
    const makeHeaderButton = (text, onClick) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = text;
        btn.style.cssText = "padding:2px 8px;cursor:pointer";
        btn.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            onClick();
        });
        return btn;
    };
    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;align-items:center;gap:6px";
    const clearBtn = makeHeaderButton("清理缓存", () => clearSegmentCache(node));
    const refresh = makeHeaderButton("重新检查", () => refreshFirstPassCacheStatus(node));
    buttons.append(clearBtn, refresh);
    const body = document.createElement("div");
    body.style.cssText = "white-space:pre-wrap;word-break:break-word;user-select:text;cursor:text";
    body.textContent = "等待检查…";
    for (const eventName of ["pointerdown", "mousedown", "click"]) {
        body.addEventListener(eventName, (event) => event.stopPropagation());
    }
    header.append(title, buttons);
    root.append(header, body);
    const widget = node.addDOMWidget(CACHE_STATUS_WIDGET, "cache_status", root, {
        getValue: () => "",
        setValue: () => {},
        getMinHeight: () => 188,
        hideOnZoom: false,
    });
    // Status is derived UI, not a positional backend widget value.
    widget.serialize = false;
    if (!widget.options) widget.options = {};
    widget.options.serialize = false;
    node._mmxFirstPassCacheUI = { root, body, refresh, widget };
}

async function clearSegmentCache(node) {
    const director = connectedDirector(node);
    if (!director) {
        renderCacheStatus(node, "未找到相连的 MiniMax H3 Director。", "warn");
        return;
    }
    if (!window.confirm("确定清空这个节点的分段缓存吗？一采和成片都会删除，需要重新生成。")) {
        return;
    }
    renderCacheStatus(node, "正在清空缓存…", "muted");
    try {
        const response = await api.fetchApi("/minimax/director/clear_segment_cache", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ node_id: String(director.id), kind: "all" }),
        });
        const data = await response.json();
        if (!response.ok || data?.error) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        renderCacheStatus(node, `缓存已清空（删除 ${data.removed} 个文件）。`, "ok");
        scheduleCacheStatusRefresh(node, 200);
    } catch (error) {
        renderCacheStatus(node, `清空缓存失败：${error?.message || error}`, "error");
    }
}

function syncRefineWidgetVisibility(node) {
    const mode = readMode(node);
    const upscale = mode === "upscale";
    const latentOnly = mode === "latent_upscale";
    const needsCanvas = upscale || latentOnly;
    const aspect = widgetValue(widgetByName(node, "aspect_ratio"));
    const follow = isFollowAspect(aspect);
    const custom = isCustomAspect(aspect);
    setWidgetVisible(node, "aspect_ratio", needsCanvas);
    setWidgetVisible(node, "megapixels", needsCanvas && !follow && !custom);
    setWidgetVisible(node, "width", needsCanvas && custom);
    setWidgetVisible(node, "height", needsCanvas && custom);
    const method = readUpscaleMethod(node);
    const showH3Model = latentOnly || (upscale && method === "h3_latent");
    setWidgetVisible(node, "upscale_method", upscale);
    setWidgetVisible(node, "latent_upscale_model", showH3Model);
    setWidgetVisible(node, "enable_latent_chunking", showH3Model);
    setWidgetVisible(node, "h3_latent_model", false);
    setWidgetVisible(node, "upscale_model", false);
    setWidgetVisible(node, "schedule", false);
    setWidgetVisible(node, "denoise", false);
    setWidgetVisible(node, "steps", false);
    setWidgetVisible(node, "sigmas_text", false);
    setWidgetVisible(node, "sigmas", false);
    setWidgetVisible(node, "sampler", !latentOnly);
    setWidgetVisible(node, "passes", !latentOnly);
    setWidgetVisible(node, "seed_mode", !latentOnly);
    setWidgetVisible(node, "enable_tiling", !latentOnly);
    const tilingOn = !latentOnly && Boolean(widgetValue(widgetByName(node, "enable_tiling")));
    setWidgetVisible(node, "tile_count", tilingOn);
    setWidgetVisible(node, "tile_overlap", tilingOn);
    setWidgetVisible(node, "target_width", false);
    setWidgetVisible(node, "target_height", false);
    ensureFirstPassCacheUI(node);
    setWidgetVisible(node, CACHE_STATUS_WIDGET, true);
    if (needsCanvas && !follow && !custom) syncRefineComputedSize(node);
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
    hookWidget(node, "upscale_method", () => syncRefineWidgetVisibility(node));
    hookWidget(node, "enable_tiling", () => syncRefineWidgetVisibility(node));
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
    hookWidget(node, "confirm_first_pass", () => {
        syncRefineWidgetVisibility(node);
        scheduleCacheStatusRefresh(node, 0);
    });
    if (!node._mmxRefineOnWidgetChanged) {
        node._mmxRefineOnWidgetChanged = true;
        const prev = node.onWidgetChanged;
        node.onWidgetChanged = function (name, ...rest) {
            const r = prev?.apply(this, [name, ...rest]);
            if (name === "mode" || name === "upscale_method" || name === "aspect_ratio" || name === "megapixels" || name === "enable_tiling") {
                migrateRefineWidgets(this);
                syncRefineWidgetVisibility(this);
            }
            scheduleCacheStatusRefresh(this);
            return r;
        };
    }
}

function refreshRefineNode(node) {
    if (!isRefineNode(node)) return;
    installRefineResolutionUI(node);
    migrateRefineWidgets(node);
    syncRefineWidgetVisibility(node);
    scheduleCacheStatusRefresh(node);
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
        if (DIRECTOR_CLASSES.has(nodeData?.name)) {
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (...args) {
                const result = onWidgetChanged?.apply(this, args);
                refreshCacheStatusForDirector(this);
                return result;
            };
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (...args) {
                const result = onConnectionsChange?.apply(this, args);
                refreshCacheStatusForDirector(this);
                return result;
            };
            return;
        }
        if (PACKER_CLASSES.has(nodeData?.name)) {
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (...args) {
                const result = onWidgetChanged?.apply(this, args);
                refreshAllRefineNodes();
                return result;
            };
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (...args) {
                const result = onConnectionsChange?.apply(this, args);
                refreshAllRefineNodes();
                return result;
            };
            return;
        }
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
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const r = onConnectionsChange?.apply(this, args);
            syncRefineWidgetVisibility(this);
            scheduleCacheStatusRefresh(this);
            return r;
        };
    },
    nodeCreated(node) {
        const cls = node?.comfyClass || node?.type || "";
        if (DIRECTOR_CLASSES.has(cls)) {
            node._mmxRefreshFirstPassCache = (delay = 0) => {
                refreshCacheStatusForDirector(node, delay);
            };
            return;
        }
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

api.addEventListener?.("executed", () => {
    for (const node of graphNodes()) {
        if (isRefineNode(node)) {
            scheduleCacheStatusRefresh(node, 250);
        }
    }
});
