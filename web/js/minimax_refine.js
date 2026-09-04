/** MiniMax H3 Director Refine — show canvas widgets like Director output bar. */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    CUSTOM_ASPECT_RATIO,
    resolutionFromSelector,
    snapResolutionDim,
} from "./minimax_gen_timeline.js";

const REFINE_CLASS = "MiniMaxH3DirectorRefine";
const DIRECTOR_CLASSES = new Set(["MiniMaxH3Director", "ComfyMiniMaxH3Director"]);
const CACHE_STATUS_WIDGET = "first_pass_cache_status";
const FOLLOW_DIRECTOR_ASPECT = "跟随导演台";

function isRefineNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return cls === REFINE_CLASS;
}

function isDirectorNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return DIRECTOR_CLASSES.has(cls);
}

/** Cache UI lives on Refine (status+manager) or directly on the Director. */
function cacheUI(node) {
    return node?._mmxFirstPassCacheUI || node?._mmxDirectorCacheUI || null;
}

function directorForCacheNode(node) {
    return isRefineNode(node) ? connectedDirector(node) : (isDirectorNode(node) ? node : null);
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

function directorHasSigmasLink(node) {
    const inp = (node?.inputs || []).find((i) => String(i.name) === "sigmas");
    if (!inp) return false;
    if (inp.link != null) return true;
    return Array.isArray(inp.links) && inp.links.length > 0;
}

function cacheStatusPayload(director) {
    try {
        director?._minimaxEditor?._writeTimelineWidget?.();
    } catch {
        /* best effort */
    }
    return {
        node_id: String(director.id),
        timeline_data: String(directorValue(director, "timeline_data", "")),
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
    const diffLabels = {
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
        cfg: "CFG",
        steps: "一采步数",
        sampler: "一采采样器",
        scheduler: "调度器",
        sigmas: "一采噪声表",
        sigmas_source: "一采 SIGMAS 接线",
        shift_video: "视频 shift",
        shift_audio: "音频 shift",
        "<invalid-meta>": "缓存信息损坏",
    };
    const diffs = Array.isArray(data?.diff_keys)
        ? data.diff_keys
            .filter((key) => key !== "<missing-cache>")
            .slice(0, 8)
            .map((key) => diffLabels[key] || key)
        : [];
    const selTotal = data?.selected_total;
    const selMatched = data?.selected_matched;
    const selActive = Number.isFinite(selTotal) && Number(selTotal) !== total;
    const lines = [
        `一采缓存：${data?.exists ? `存在（${cached}/${total} 段）` : "不存在"}`,
        `当前匹配：${data?.matches ? `是（${matched}/${total} 段）` : "否"}`
            + (selActive ? ` · 选中 ${selMatched ?? 0}/${selTotal ?? 0}` : ""),
    ];
    lines.push(`缓存 seed：${seeds}`);
    lines.push(`当前 seed：${data?.current_seed ?? "—"}`);
    const finalCached = Number(data?.final_cached_count || 0);
    lines.push(`成片缓存：${finalCached}/${total} 段（含音频；部分重跑会接这里）`);
    if (diffs.length) lines.push(`差异：${diffs.join(", ")}`);
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
            body: JSON.stringify(cacheStatusPayload(director)),
        });
        const data = await response.json();
        if (seq !== node._mmxCacheStatusSeq) return;
        if (!response.ok || data?.error) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        renderCacheStatus(node, data, data.matches ? "ok" : (data.exists ? "warn" : "muted"));
    } catch (error) {
        if (seq !== node._mmxCacheStatusSeq) return;
        renderCacheStatus(node, `缓存检查失败：${error?.message || error}`, "error");
    }
}

function scheduleCacheStatusRefresh(node, delay = 120) {
    clearTimeout(node._mmxCacheStatusTimer);
    node._mmxCacheStatusTimer = setTimeout(() => refreshFirstPassCacheStatus(node), delay);
}

function formatBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function resizeCacheWidget(node) {
    try {
        const size = node.computeSize?.();
        if (Array.isArray(size) && size.length >= 2) {
            node.setSize?.([node.size?.[0] || size[0], size[1]]);
        }
        node.setDirtyCanvas?.(true, true);
    } catch {
        /* ignore */
    }
}

function makeSmallButton(text, onClick) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = text;
    btn.style.cssText = "padding:1px 7px;cursor:pointer;font-size:11px";
    btn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick();
    });
    return btn;
}

async function toggleCacheManager(node) {
    if (isRefineNode(node)) ensureFirstPassCacheUI(node);
    else ensureDirectorCacheManagerUI(node);
    const ui = cacheUI(node);
    if (!ui) return;
    ui.managerOpen = !ui.managerOpen;
    ui.manager.style.display = ui.managerOpen ? "" : "none";
    if (ui.manageBtn) ui.manageBtn.textContent = ui.managerOpen ? "收起管理" : "缓存管理";
    resizeCacheWidget(node);
    if (ui.managerOpen) await refreshCacheManager(node);
}

async function refreshCacheManager(node) {
    const ui = cacheUI(node);
    if (!ui?.managerOpen) return;
    const director = directorForCacheNode(node);
    if (!director) {
        ui.managerSummary.textContent = "未找到关联的 MiniMax H3 Director。";
        resizeCacheWidget(node);
        return;
    }
    ui.managerSummary.textContent = "正在读取缓存明细…";
    try {
        const response = await api.fetchApi("/minimax/director/segment_cache_detail", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ node_id: String(director.id) }),
        });
        const data = await response.json();
        if (!response.ok || data?.error) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        renderCacheManager(node, data);
    } catch (error) {
        ui.managerSummary.textContent = `读取缓存失败：${error?.message || error}`;
    }
    refreshCacheOverview(node);
    resizeCacheWidget(node);
}

function cacheBadge(text, active) {
    const badge = document.createElement("span");
    badge.textContent = text;
    badge.style.cssText = [
        "padding:0 5px",
        "border-radius:3px",
        "font-size:11px",
        `color:${active ? "#65d68a" : "#777"}`,
        `border:1px solid ${active ? "#3d7a52" : "#555"}`,
    ].join(";");
    return badge;
}

function renderCacheManager(node, data) {
    const ui = cacheUI(node);
    if (!ui) return;
    const list = ui.managerList;
    list.textContent = "";
    ui._mmxCacheChecked = new Set();
    const segments = Array.isArray(data?.segments) ? data.segments : [];
    const others = Array.isArray(data?.other_files) ? data.other_files : [];
    if (!segments.length && !others.length) {
        ui.managerSummary.textContent = "本节点暂无缓存文件。";
        const empty = document.createElement("div");
        empty.style.cssText = "color:#999;padding:4px 2px";
        empty.textContent = "（运行一次生成后会在这里按素材组列出缓存）";
        list.appendChild(empty);
        return;
    }
    ui.managerSummary.textContent =
        `共 ${segments.length} 个素材组 · ${data.file_count} 个文件 · ${formatBytes(data.total_size)}`;
    for (const seg of segments) {
        const row = document.createElement("div");
        row.style.cssText = [
            "display:flex",
            "align-items:center",
            "gap:6px",
            "padding:3px 6px",
            "border:1px solid var(--border-color, #444)",
            "border-radius:4px",
            "background:rgba(255,255,255,.03)",
        ].join(";");

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.style.cssText = "margin:0;flex:none";
        checkbox.addEventListener("change", () => {
            if (checkbox.checked) ui._mmxCacheChecked.add(seg.segment_index);
            else ui._mmxCacheChecked.delete(seg.segment_index);
        });

        const title = document.createElement("span");
        title.textContent = `第${seg.segment}组`;
        title.style.cssText = "min-width:52px;flex:none;font-weight:600;color:var(--input-text,#ddd)";

        const promptText = String(seg.meta?.prompt || seg.pre_meta?.prompt || "").trim();
        const info = document.createElement("span");
        info.style.cssText = "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#bbb";
        info.textContent = promptText || "（无提示词记录）";
        info.title = promptText;

        const badges = document.createElement("span");
        badges.style.cssText = "display:flex;gap:4px;flex:none";
        badges.append(
            cacheBadge("一采", seg.has_first_pass),
            cacheBadge("成片", seg.has_final),
            cacheBadge("音频", seg.has_audio),
        );

        const size = document.createElement("span");
        size.textContent = formatBytes(seg.total_size);
        size.style.cssText = "flex:none;min-width:64px;text-align:right;color:#bbb";
        size.title = (Array.isArray(seg.files) ? seg.files : [])
            .map((f) => `${f.name}  ${formatBytes(f.size)}`)
            .join("\n");

        const time = document.createElement("span");
        time.textContent = seg.mtime ? new Date(seg.mtime * 1000).toLocaleString() : "";
        time.style.cssText = "flex:none;color:#888;font-size:11px";

        row.append(
            checkbox,
            title,
            badges,
            info,
            size,
            time,
            makeSmallButton("删除", () => deleteSegmentCacheEntries(node, [seg.segment_index])),
        );
        list.appendChild(row);
    }
    if (others.length) {
        const note = document.createElement("div");
        note.style.cssText = "color:#999;padding:2px 2px 0";
        note.textContent = `另有 ${others.length} 个未命名临时/残留文件（下次写入缓存会自动清理）。`;
        list.appendChild(note);
    }
}

async function deleteSegmentCacheEntries(node, indices) {
    const director = directorForCacheNode(node);
    if (!director) return;
    const list = [...new Set(indices.map((i) => Number(i)).filter((i) => Number.isFinite(i) && i >= 0))];
    if (!list.length) return;
    const label = list.length === 1 ? `第${list[0] + 1}组` : `选中的 ${list.length} 个素材组`;
    if (!window.confirm(`确定删除${label}的缓存吗？一采和成片都会删除，需要重新生成。`)) {
        return;
    }
    const ui = cacheUI(node);
    if (ui) ui.managerSummary.textContent = "正在删除缓存…";
    try {
        let removed = 0;
        for (const idx of list) {
            const response = await api.fetchApi("/minimax/director/clear_segment_cache", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: String(director.id),
                    kind: "all",
                    segment_index: idx,
                }),
            });
            const data = await response.json();
            if (!response.ok || data?.error) {
                throw new Error(data?.error || `HTTP ${response.status}`);
            }
            removed += Number(data.removed || 0);
        }
        if (ui) ui.managerSummary.textContent = `已删除${label}的缓存（${removed} 个文件）。`;
    } catch (error) {
        if (ui) ui.managerSummary.textContent = `删除缓存失败：${error?.message || error}`;
    }
    await refreshCacheManager(node);
    scheduleCacheStatusRefresh(node, 200);
}

async function refreshCacheOverview(node) {
    const ui = cacheUI(node);
    if (!ui?.managerOpen) return;
    try {
        const response = await api.fetchApi("/minimax/director/cache_overview");
        const data = await response.json();
        if (!response.ok || data?.error) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        renderCacheOverview(node, data);
    } catch (error) {
        ui.overviewList.textContent = `读取全局缓存失败：${error?.message || error}`;
    }
    resizeCacheWidget(node);
}

function renderCacheOverview(node, data) {
    const ui = cacheUI(node);
    if (!ui) return;
    const list = ui.overviewList;
    list.textContent = "";
    const director = directorForCacheNode(node);
    const currentId = director ? String(director.id) : "";
    const groups = (Array.isArray(data?.groups) ? data.groups : []).filter(
        (g) => String(g.node_id) !== currentId && Number(g.file_count) > 0,
    );
    if (!groups.length) {
        list.textContent = "";
        const empty = document.createElement("div");
        empty.style.cssText = "color:#999";
        empty.textContent = "没有其他节点的缓存。";
        list.appendChild(empty);
        return;
    }
    for (const group of groups) {
        const row = document.createElement("div");
        row.style.cssText = [
            "display:flex",
            "align-items:center",
            "gap:6px",
            "padding:2px 6px",
            "border:1px solid var(--border-color, #444)",
            "border-radius:4px",
        ].join(";");
        const name = document.createElement("span");
        name.textContent = `节点 #${group.node_id}`;
        name.style.cssText = "flex:1;min-width:0;color:#bbb";
        name.title = "该节点可能已从工作流删除；删除后缓存不可恢复";
        const size = document.createElement("span");
        size.textContent = `${group.file_count} 个文件 · ${formatBytes(group.total_size)}`;
        size.style.cssText = "flex:none;color:#888;font-size:11px";
        row.append(
            name,
            size,
            makeSmallButton("删除", () => deleteNodeCacheGroup(node, group.node_id)),
        );
        list.appendChild(row);
    }
}

async function deleteNodeCacheGroup(node, nodeId) {
    if (!window.confirm(`确定删除节点 #${nodeId} 的全部缓存吗？需要重新生成才能恢复。`)) {
        return;
    }
    const ui = cacheUI(node);
    try {
        const response = await api.fetchApi("/minimax/director/clear_segment_cache", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ node_id: String(nodeId), kind: "all" }),
        });
        const data = await response.json();
        if (!response.ok || data?.error) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        if (ui) ui.managerSummary.textContent = `已删除节点 #${nodeId} 的缓存（${data.removed} 个文件）。`;
    } catch (error) {
        if (ui) ui.managerSummary.textContent = `删除缓存失败：${error?.message || error}`;
    }
    await refreshCacheManager(node);
    scheduleCacheStatusRefresh(node, 200);
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
    const manageBtn = makeHeaderButton("缓存管理", () => toggleCacheManager(node));
    const clearBtn = makeHeaderButton("清理缓存", () => clearSegmentCache(node));
    const refresh = makeHeaderButton("重新检查", () => refreshFirstPassCacheStatus(node));
    buttons.append(manageBtn, clearBtn, refresh);
    const body = document.createElement("div");
    body.style.cssText = "white-space:pre-wrap;word-break:break-word;user-select:text;cursor:text";
    body.textContent = "等待检查…";

    const manager = document.createElement("div");
    manager.style.cssText = "display:none;margin-top:6px;border-top:1px solid var(--border-color, #555);padding-top:6px";
    const managerHead = document.createElement("div");
    managerHead.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:6px;margin-bottom:4px";
    const managerSummary = document.createElement("span");
    managerSummary.style.cssText = "color:var(--input-text, #ddd);font-size:11px";
    const managerActions = document.createElement("div");
    managerActions.style.cssText = "display:flex;gap:6px;flex:none";
    const deleteSelectedBtn = makeSmallButton("删除选中", () => {
        const ui = node._mmxFirstPassCacheUI;
        deleteSegmentCacheEntries(node, [...(ui?._mmxCacheChecked || [])]);
    });
    managerActions.append(
        deleteSelectedBtn,
        makeSmallButton("刷新", () => refreshCacheManager(node)),
    );
    managerHead.append(managerSummary, managerActions);
    const managerList = document.createElement("div");
    managerList.style.cssText = "max-height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:4px";
    const overviewTitle = document.createElement("div");
    overviewTitle.style.cssText = "margin-top:8px;color:#999;font-size:11px";
    overviewTitle.textContent = "其他节点的缓存（含已从工作流删除的节点）：";
    const overviewList = document.createElement("div");
    overviewList.style.cssText = "display:flex;flex-direction:column;gap:4px;margin-top:3px";
    manager.append(managerHead, managerList, overviewTitle, overviewList);

    for (const eventName of ["pointerdown", "mousedown", "click", "wheel"]) {
        body.addEventListener(eventName, (event) => event.stopPropagation());
        manager.addEventListener(eventName, (event) => event.stopPropagation());
    }
    header.append(title, buttons);
    root.append(header, body, manager);
    const widget = node.addDOMWidget(CACHE_STATUS_WIDGET, "cache_status", root, {
        getValue: () => "",
        setValue: () => {},
        getMinHeight: () => Math.max(148, root.scrollHeight + 20),
        hideOnZoom: false,
    });
    // Status is derived UI, not a positional backend widget value.
    widget.serialize = false;
    if (!widget.options) widget.options = {};
    widget.options.serialize = false;
    node._mmxFirstPassCacheUI = {
        root,
        body,
        refresh,
        widget,
        manageBtn,
        manager,
        managerSummary,
        managerList,
        overviewList,
        managerOpen: false,
    };
}

const DIRECTOR_CACHE_WIDGET = "minimax_director_cache_manager";

/** Cache manager on the Director node itself — no Refine node required. */
function ensureDirectorCacheManagerUI(node) {
    if (!isDirectorNode(node) || node._mmxDirectorCacheUI || typeof node.addDOMWidget !== "function") return;
    if ((node.widgets || []).some((w) => w?.name === DIRECTOR_CACHE_WIDGET)) return;
    const root = document.createElement("div");
    root.style.cssText = [
        "box-sizing:border-box",
        "margin:2px 8px",
        "padding:6px 10px",
        "border:1px solid var(--border-color, #555)",
        "border-radius:6px",
        "background:rgba(0,0,0,.16)",
        "font:12px/1.45 sans-serif",
    ].join(";");
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between";
    const title = document.createElement("strong");
    title.textContent = "分段缓存";
    title.style.cssText = "color:var(--input-text, #ddd)";
    const manageBtn = makeSmallButton("缓存管理", () => toggleCacheManager(node));
    manageBtn.style.fontSize = "12px";
    manageBtn.style.padding = "2px 10px";
    header.append(title, manageBtn);

    const manager = document.createElement("div");
    manager.style.cssText = "display:none;margin-top:6px;border-top:1px solid var(--border-color, #555);padding-top:6px";
    const managerHead = document.createElement("div");
    managerHead.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:6px;margin-bottom:4px";
    const managerSummary = document.createElement("span");
    managerSummary.style.cssText = "color:var(--input-text, #ddd);font-size:11px";
    const managerActions = document.createElement("div");
    managerActions.style.cssText = "display:flex;gap:6px;flex:none";
    const deleteSelectedBtn = makeSmallButton("删除选中", () => {
        const ui = node._mmxDirectorCacheUI;
        deleteSegmentCacheEntries(node, [...(ui?._mmxCacheChecked || [])]);
    });
    managerActions.append(
        deleteSelectedBtn,
        makeSmallButton("刷新", () => refreshCacheManager(node)),
    );
    managerHead.append(managerSummary, managerActions);
    const managerList = document.createElement("div");
    managerList.style.cssText = "max-height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:4px";
    const overviewTitle = document.createElement("div");
    overviewTitle.style.cssText = "margin-top:8px;color:#999;font-size:11px";
    overviewTitle.textContent = "其他节点的缓存（含已从工作流删除的节点）：";
    const overviewList = document.createElement("div");
    overviewList.style.cssText = "display:flex;flex-direction:column;gap:4px;margin-top:3px";
    manager.append(managerHead, managerList, overviewTitle, overviewList);

    for (const eventName of ["pointerdown", "mousedown", "click", "wheel"]) {
        manager.addEventListener(eventName, (event) => event.stopPropagation());
    }
    root.append(header, manager);
    const widget = node.addDOMWidget(DIRECTOR_CACHE_WIDGET, "cache_manager", root, {
        getValue: () => "",
        setValue: () => {},
        getMinHeight: () => {
            const ui = node._mmxDirectorCacheUI;
            return ui?.managerOpen ? Math.max(60, root.scrollHeight + 16) : 34;
        },
        hideOnZoom: false,
    });
    widget.serialize = false;
    if (!widget.options) widget.options = {};
    widget.options.serialize = false;
    node._mmxDirectorCacheUI = {
        root,
        manageBtn,
        manager,
        managerSummary,
        managerList,
        overviewList,
        managerOpen: false,
        widget,
    };
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
        if (node._mmxFirstPassCacheUI?.managerOpen) {
            refreshCacheManager(node);
        }
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
            if (name === "mode" || name === "upscale_method" || name === "aspect_ratio" || name === "megapixels") {
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
            setTimeout(() => {
                ensureDirectorCacheManagerUI(node);
                resizeCacheWidget(node);
            }, 0);
            return;
        }
        scheduleRefineRefresh(node);
    },
    loadedGraphNode(node) {
        if (isDirectorNode(node)) {
            setTimeout(() => {
                ensureDirectorCacheManagerUI(node);
                resizeCacheWidget(node);
            }, 0);
        }
        scheduleRefineRefresh(node);
    },
    afterConfigureGraph() {
        refreshAllRefineNodes();
        for (const node of graphNodes()) {
            if (isDirectorNode(node)) ensureDirectorCacheManagerUI(node);
        }
        setTimeout(refreshAllRefineNodes, 100);
    },
});

api.addEventListener?.("executed", () => {
    for (const node of graphNodes()) {
        if (isRefineNode(node)) {
            scheduleCacheStatusRefresh(node, 250);
        }
        if (cacheUI(node)?.managerOpen) {
            const target = node;
            setTimeout(() => refreshCacheManager(target), 400);
        }
    }
});
