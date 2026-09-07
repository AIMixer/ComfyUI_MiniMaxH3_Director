/**
 * MiniMaxH3Grade 调色台面板（v52）。
 *
 * 纯下游后期制作：不触碰采样/二采/缓存/版本号。面板提供：
 * - 内置预览播放（读取 ui.videos，段标签切换时自动定位时间轴）；
 * - 两级手工参数：全片统一 + 每段覆盖（bright/contrast/sat/hue/soften）；
 * - 自动优化按钮（接缝漂移 / 对比度均衡 × 全局 / 本段）→ 触发单节点重执行，
 *   由后端 auto_analyze 返回建议值，一键写入滑块；
 * - 快照历史（撤销/重做）+ A/B 参数对比 + 尖峰自动修复开关。
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const GRADE_NODE_TYPE = "MiniMaxH3Grade";
const GRADE_DOM_WIDGET_NAME = "minimax_grade_ui";

const GRADE_PARAM_DEFS = [
    { key: "bright", label: "亮度", min: -0.2, max: 0.2, step: 0.002, def: 0.0, digits: 3 },
    { key: "contrast", label: "对比度", min: 0.5, max: 1.5, step: 0.002, def: 1.0, digits: 3 },
    { key: "sat", label: "饱和度", min: 0.5, max: 1.5, step: 0.002, def: 1.0, digits: 3 },
    { key: "hue", label: "色相°", min: -20, max: 20, step: 0.1, def: 0.0, digits: 1 },
    { key: "soften", label: "柔化", min: 0, max: 1, step: 0.01, def: 0.0, digits: 2 },
];
const GRADE_PARAM_KEYS = GRADE_PARAM_DEFS.map((d) => d.key);

function defaultGradeParams() {
    const p = {};
    for (const d of GRADE_PARAM_DEFS) p[d.key] = d.def;
    return p;
}

function cloneParams(p) {
    const out = {};
    for (const k of GRADE_PARAM_KEYS) out[k] = p ? Number(p[k]) : undefined;
    return out;
}

function clampParam(key, v) {
    const d = GRADE_PARAM_DEFS.find((x) => x.key === key);
    if (!d) return v;
    const n = Number(v);
    if (!Number.isFinite(n)) return d.def;
    return Math.min(d.max, Math.max(d.min, n));
}

function fmtParam(key, v) {
    const d = GRADE_PARAM_DEFS.find((x) => x.key === key);
    return Number(v).toFixed(d ? d.digits : 3);
}

/** 与后端 director/grade_core.py parse_timeline_segments 同构的时间轴解析。 */
function parseSegmentsFromTimeline(raw) {
    let data = raw;
    if (typeof raw === "string") {
        if (!raw) return [];
        try {
            data = JSON.parse(raw);
        } catch {
            return [];
        }
    }
    if (!data || typeof data !== "object") return [];
    let rows = null;
    for (const key of ["segments", "shots", "r2vGroups", "fl2vGroups"]) {
        if (Array.isArray(data[key]) && data[key].length) {
            rows = data[key];
            break;
        }
    }
    if (!rows && data.output && Array.isArray(data.output.segments)) rows = data.output.segments;
    if (!Array.isArray(rows)) return [];
    const fps = Number(data.fps || data.frameRate || 24) || 24;
    const out = [];
    for (const row of rows) {
        if (!row || typeof row !== "object") continue;
        let fc = row.frameCount ?? row.length ?? row.frames;
        if (fc == null && row.durationSec != null) {
            fc = Math.max(1, Math.round(Number(row.durationSec) * fps));
        }
        if (fc != null) out.push(Math.max(1, parseInt(String(fc), 10) || 1));
    }
    return out;
}

function hideGradeWidget(w) {
    if (!w) return;
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    if (w.computeSize) w.computeSize = () => [0, 0];
    if (w.element) w.element.style.display = "none";
}

/**
 * 兼容各前端版本的取链接：graph.links 在旧版是数组、新版 litegraph 是
 * Map、极老的工作流还有按 id 索引的对象形态。链接记录本身也可能是
 * 数组 [id, origin_id, origin_slot, target_id, target_slot, type]。
 */
function graphLinkById(graph, linkId) {
    if (!graph || linkId == null) return null;
    const links = graph.links;
    if (!links) return null;
    let link = null;
    if (typeof links.get === "function") {
        link = links.get(Number(linkId));
        if (link == null) link = links.get(String(linkId));
    } else if (Array.isArray(links)) {
        link = links.find((l) => l && (l.id === linkId || l[0] === linkId)) || null;
    } else {
        link = links[linkId] || null;
    }
    if (!link) return null;
    return {
        id: link.id ?? link[0],
        originId: link.origin_id ?? link[1],
        originSlot: link.origin_slot ?? link[2],
        targetId: link.target_id ?? link[3],
        targetSlot: link.target_slot ?? link[4],
    };
}

/**
 * 实时调色/自动分析：POST 插件路由，不走 ComfyUI 队列。
 * ComfyUI-XS 等衍生版不支持部分 prompt 重执行（validate_inputs 强制要求
 * 所有引用节点在场），此路由读调色台上次真实执行时缓存的原始画面重算，
 * 秒级返回预览，不触发任何下游保存节点。
 */
async function postGradePreview(panel, opts = {}) {
    const analysis = !!opts.analysis;
    try {
        panel.setBusy(true);
        const spec = panel.serialize(true);
        if (!analysis) spec.auto_requests = null; // 执行优化不重复分析
        const fpsW = panel.node.widgets?.find((w) => w.name === "fps");
        const res = await api.fetchApi("/minimax/grade/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                node_id: String(panel.node.id),
                spec,
                fps: Number(fpsW?.value || 24) || 24,
                skip_preview: !!analysis, // 纯分析不重算预览（点「执行优化」才算）
            }),
        });
        const data = await res.json().catch(() => ({}));
        panel.setBusy(false);
        if (!res.ok || !data?.ok) {
            throw new Error(data?.message || data?.error || `HTTP ${res.status}`);
        }
        if (data.video) {
            panel.setVideoByName(data.video);
            if (!analysis) {
                panel.showNotes([
                    `✓ 实时预览已更新 ${new Date().toLocaleTimeString()}（看面板内的播放器，节点下方的原生预览只在完整运行时更新）`,
                ]);
            }
        }
        if (data.auto_results) {
            panel.applyAutoResults(data.auto_results);
            if (analysis && panel.state.auto_requests?.mode !== "drift_report") {
                // 一键自动优化：分析完成后直接把建议写入滑块并实时重算预览。
                // 漂移参考表不写参数，跳过。
                panel.applySuggestions();
            }
        }
        panel._afterPreviewRun?.();
    } catch (err) {
        panel.setBusy(false);
        panel._afterPreviewRun?.();
        panel.showNotes(["实时计算失败：" + (err?.message || err)]);
    }
}

/**
 * 队列出片：提交完整图（XS 衍生版校验要求所有引用节点在场）。
 * 未变更节点（导演台/二采/加载器）走服务端缓存不重跑，只有调色台与
 * 下游 CreateVideo/SaveVideo 重新执行。
 */
async function queueFullGraph(panel) {
    try {
        const graph = app.graph ?? app.canvas?.graph;
        const nodes = graph?._nodes ?? graph?.nodes ?? [];
        const p = {};
        for (const n of nodes) {
            if (n.mode === 2 || n.mode === 4) continue; // 跳过 bypass/mute
            const entry = { class_type: n.type, inputs: {} };
            for (const input of n.inputs || []) {
                if (input.link != null) {
                    const link = graphLinkById(graph, input.link);
                    if (link && link.originId != null) {
                        entry.inputs[input.name] = [String(link.originId), link.originSlot ?? 0];
                        continue;
                    }
                }
                const w = n.widgets?.find((x) => x.name === input.name);
                if (w) entry.inputs[input.name] = w.value;
            }
            p[String(n.id)] = entry;
        }
        panel.setBusy(true);
        const res = await api.fetchApi("/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: p,
                client_id: api.clientId,
                extra_data: {},
            }),
        });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(`服务器返回 ${res.status}: ${text}`);
        }
    } catch (err) {
        panel.setBusy(false);
        panel.showNotes([
            "队列出片失败：" + (err?.message || err) + "（若缓存被清空，请先完整运行一次工作流）",
        ]);
    }
}

function injectGradeStyles() {
    if (document.getElementById("mmx-grade-styles")) return;
    const style = document.createElement("style");
    style.id = "mmx-grade-styles";
    style.textContent = `
.mmx-grade-host { padding: 6px; font-size: 12px; color: var(--fg-color, #ddd); user-select: none; }
.mmx-grade-host .mg-head { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin-bottom: 6px; }
.mmx-grade-host .mg-title { font-weight: 700; margin-right: 6px; }
.mmx-grade-host button.mg-btn {
    background: var(--comfy-input-bg, #2a2a2e); color: var(--fg-color, #ddd);
    border: 1px solid var(--border-color, #555); border-radius: 4px; padding: 3px 8px;
    cursor: pointer; font-size: 11px;
}
.mmx-grade-host button.mg-btn:hover { border-color: #8ab4f8; }
.mmx-grade-host button.mg-btn:disabled { opacity: 0.45; cursor: default; }
.mmx-grade-host button.mg-btn.mg-on { background: #1e3a5f; border-color: #8ab4f8; }
.mmx-grade-host .mg-status { margin-left: auto; color: #c9a227; min-width: 60px; text-align: right; }
.mmx-grade-host .mg-preview { position: relative; margin-bottom: 6px; background: #000; border-radius: 6px; overflow: hidden; }
.mmx-grade-host .mg-preview video { width: 100%; max-height: 260px; display: block; background: #000; }
.mmx-grade-host .mg-preview-empty { padding: 34px 8px; text-align: center; color: #888; }
.mmx-grade-host .mg-preview-bar { display: flex; gap: 4px; align-items: center; padding: 3px 6px; background: rgba(0,0,0,0.4); }
.mmx-grade-host .mg-tabs { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 6px; }
.mmx-grade-host .mg-tab {
    padding: 3px 9px; border: 1px solid var(--border-color, #555); border-radius: 4px;
    cursor: pointer; background: var(--comfy-input-bg, #2a2a2e);
}
.mmx-grade-host .mg-tab.mg-active { background: #1e3a5f; border-color: #8ab4f8; font-weight: 600; }
.mmx-grade-host .mg-row { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
.mmx-grade-host .mg-row label.mg-lab { width: 52px; flex: 0 0 auto; text-align: right; }
.mmx-grade-host .mg-row input[type=range] { flex: 1 1 auto; min-width: 90px; }
.mmx-grade-host .mg-row .mg-val { width: 44px; flex: 0 0 auto; text-align: left; font-variant-numeric: tabular-nums; }
.mmx-grade-host .mg-num {
    width: 68px; flex: 0 0 auto; background: var(--comfy-input-bg, #2a2a2e);
    color: var(--fg-color, #ddd); border: 1px solid var(--border-color, #555);
    border-radius: 4px; font-size: 11px; padding: 1px 4px; font-variant-numeric: tabular-nums;
}
.mmx-grade-host .mg-sec { border-top: 1px solid var(--border-color, #444); margin-top: 6px; padding-top: 6px; }
.mmx-grade-host .mg-sec-title { font-weight: 600; margin-bottom: 4px; }
.mmx-grade-host .mg-auto { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.mmx-grade-host .mg-notes { margin-top: 5px; color: #b8b8b8; max-height: 90px; overflow-y: auto; user-select: text; cursor: text; }
.mmx-grade-host .mg-notes div { padding: 1px 0; }
.mmx-grade-host .mg-check { display: inline-flex; align-items: center; gap: 3px; cursor: pointer; }
.mmx-grade-host select.mg-preset, .mmx-grade-host select {
    background: var(--comfy-input-bg, #2a2a2e); color: var(--fg-color, #ddd);
    border: 1px solid var(--border-color, #555); border-radius: 4px;
    font-size: 11px; max-width: 230px; padding: 2px 4px;
}
`;
    document.head.appendChild(style);
}

class GradePanel {
    constructor(node) {
        this.node = node;
        this.state = {
            global: defaultGradeParams(),
            per_segment: [],
            spike_fix: true,
            auto_requests: null,
            segments: [],
            segment_bounds: null, // 导演台导出的真实每段帧数（pin 裁切后）
            preview_slot: 0, // 预览文件 LRU 槽位（0/1/2 轮换，覆盖最早的）
            save_ref_copy: false, // 仅出片时瞬时置位：写 input 目录参考副本
        };
        this.activeTab = 0; // 0 = 全片，1..N = 段
        this.undoStack = [];
        this.redoStack = [];
        this.abSnap = null; // A/B：存放 A 快照；当前滑块即 B
        this.lastSuggestions = null;
        this.lastNotes = [];
        this.lastRefCopy = null; // 最近一次出片写的 input 参考副本名
        this.uiLocked = false;
        this._rerunDebounce = null;
        this._rerunActive = false;
        this._rerunPending = false;
        this._saveQueued = false;
        this.video = null;
        this.placeholderEl = null;
        this.statusEl = null;
        this.notesEl = null;
        this.tabsEl = null;
        this.slidersEl = null;
        this.abBtn = null;
        this.applyBtn = null;
        this.specWidget = null;
        this.rootEl = null;
        this.presets = []; // 参数库列表（服务端保存）
        this.presetSel = null;
        this._videoObjectUrl = null; // 当前预览的 objectURL（切换时回收）
    }

    /* ── 状态读写 ─────────────────────────────────────────────── */

    serialize(includeBounds = true) {
        const out = {
            global: cloneParams(this.state.global),
            per_segment: (this.state.per_segment || []).map((p) => cloneParams(p)),
            spike_fix: !!this.state.spike_fix,
            auto_requests: this.state.auto_requests ? { ...this.state.auto_requests } : null,
            segments: (this.state.segments || []).slice(),
            preview_slot: (parseInt(this.state.preview_slot, 10) || 0) % 3,
            save_ref_copy: !!this.state.save_ref_copy,
        };
        if (includeBounds && Array.isArray(this.state.segment_bounds)) {
            out.segment_bounds = this.state.segment_bounds.slice();
        }
        return out;
    }

    loadFromWidget() {
        this.specWidget = this.node.widgets?.find((w) => w.name === "grade_spec");
        const raw = this.specWidget?.value;
        let s = null;
        try {
            s = raw ? JSON.parse(raw) : null;
        } catch {
            s = null;
        }
        if (!s || typeof s !== "object") s = {};
        const g = s.global && typeof s.global === "object" ? s.global : {};
        this.state.global = defaultGradeParams();
        for (const k of GRADE_PARAM_KEYS) {
            if (g[k] != null) this.state.global[k] = clampParam(k, g[k]);
        }
        this.state.per_segment = [];
        if (Array.isArray(s.per_segment)) {
            for (const row of s.per_segment) {
                const p = defaultGradeParams();
                if (row && typeof row === "object") {
                    for (const k of GRADE_PARAM_KEYS) {
                        if (row[k] != null) p[k] = clampParam(k, row[k]);
                    }
                }
                this.state.per_segment.push(p);
            }
        }
        this.state.spike_fix = s.spike_fix == null ? true : !!s.spike_fix;
        this.state.auto_requests =
            s.auto_requests && typeof s.auto_requests === "object" ? { ...s.auto_requests } : null;
        if (Array.isArray(s.segments)) this.state.segments = s.segments.map((x) => Math.max(1, parseInt(String(x), 10) || 1));
        this.state.segment_bounds = Array.isArray(s.segment_bounds)
            ? s.segment_bounds.map((x) => Math.max(1, parseInt(String(x), 10) || 1))
            : null;
        this.state.preview_slot = (parseInt(s.preview_slot, 10) || 0) % 3;
        // save_ref_copy 是瞬时标志，重载工作流时强制复位，避免每轮都写参考副本。
        this.state.save_ref_copy = false;
        // 用户要求：每次打开浏览器/F5 后滑块重置为默认——上次的修正值不延续，
        // 防止忘记归零导致新一轮被默认修改。需要复用参数时用参数库显式读取。
        this.state.global = defaultGradeParams();
        this.state.per_segment = [];
        this.state.auto_requests = null;
    }

    syncToWidget() {
        this.specWidget = this.node.widgets?.find((w) => w.name === "grade_spec");
        if (this.specWidget) this.specWidget.value = JSON.stringify(this.serialize());
        this.node.setDirtyCanvas?.(true, false);
    }

    resolveSegments() {
        // 优先用导演台导出的真实分段（pin 裁切后），其次时间轴，最后持久化值。
        if (Array.isArray(this.state.segment_bounds) && this.state.segment_bounds.length) {
            return this.state.segment_bounds.slice();
        }
        const tw = this.node.widgets?.find((w) => w.name === "timeline_data");
        let segs = parseSegmentsFromTimeline(tw?.value);
        if (!segs.length && Array.isArray(this.state.segments)) segs = this.state.segments.slice();
        return segs;
    }

    /** images 输入实际连到的上游节点（通常是导演台）。 */
    resolveDirectorNode() {
        const graph = app.graph ?? app.canvas?.graph;
        const input = this.node.inputs?.find((i) => i.name === "images");
        if (!input || input.link == null) return null;
        const link = graphLinkById(graph, input.link);
        if (!link || link.originId == null) return null;
        const nodes = graph?._nodes ?? graph?.nodes ?? [];
        return nodes.find((n) => String(n.id) === String(link.originId)) || null;
    }

    /**
     * 导演台不输出 timeline_data（只存在其面板 widget 里）——这里直接读
     * 上游导演台节点的 timeline_data widget，把段边界同步进本节点 spec，
     * 保证分段标签 / 自动定位 / 尖峰修复在第一次运行前就可用。
     */
    harvestSegments() {
        const dir = this.resolveDirectorNode();
        if (!dir) return false;
        const tw = dir.widgets?.find((w) => w.name === "timeline_data");
        const segs = parseSegmentsFromTimeline(tw?.value);
        if (!segs.length) return false;
        const cur = JSON.stringify(this.state.segments || []);
        const next = JSON.stringify(segs);
        if (cur === next) return false;
        this.state.segments = segs;
        this.syncToWidget();
        this.renderTabs();
        return true;
    }

    /**
     * 从上游导演台的 report 输出里抓 [grade-bounds] 行：那是 pin 相位对齐
     * 裁切后的真实每段帧数（时间轴 frameCount 会差数帧）。导演台执行完成后
     * 该消息会出现在 app.nodeOutputs，抓取后持久化进 spec。
     */
    harvestBounds() {
        const dir = this.resolveDirectorNode();
        if (!dir) return false;
        const msg = app.nodeOutputs?.[String(dir.id)];
        const report = msg?.report;
        if (typeof report !== "string" || !report) return false;
        const m = report.match(/\[grade-bounds\]\s*(\{.*?\})/);
        if (!m) return false;
        try {
            const data = JSON.parse(m[1]);
            const bounds = Array.isArray(data?.segments)
                ? data.segments.map((x) => Math.max(1, parseInt(String(x), 10) || 1))
                : [];
            if (!bounds.length) return false;
            const cur = JSON.stringify(this.state.segment_bounds || []);
            const next = JSON.stringify(bounds);
            if (cur === next) return false;
            this.state.segment_bounds = bounds;
            this.syncToWidget();
            this.renderTabs();
            return true;
        } catch {
            return false;
        }
    }

    pushUndo() {
        const snap = JSON.stringify(this.serialize(false));
        if (this.undoStack.length && this.undoStack[this.undoStack.length - 1] === snap) return;
        this.undoStack.push(snap);
        if (this.undoStack.length > 40) this.undoStack.shift();
        this.redoStack.length = 0;
        this.syncToWidget();
    }

    applySnapshot(snap) {
        try {
            const s = JSON.parse(snap);
            if (!s || typeof s !== "object") return;
            this.loadSnapshot(s);
        } catch {
            /* ignore */
        }
    }

    loadSnapshot(s) {
        if (s.global && typeof s.global === "object") {
            this.state.global = defaultGradeParams();
            for (const k of GRADE_PARAM_KEYS) {
                if (s.global[k] != null) this.state.global[k] = clampParam(k, s.global[k]);
            }
        }
        if (Array.isArray(s.per_segment)) {
            this.state.per_segment = s.per_segment.map((row) => {
                const p = defaultGradeParams();
                if (row && typeof row === "object") {
                    for (const k of GRADE_PARAM_KEYS) {
                        if (row[k] != null) p[k] = clampParam(k, row[k]);
                    }
                }
                return p;
            });
        }
        if (s.spike_fix != null) this.state.spike_fix = !!s.spike_fix;
        if (s.segments) this.state.segments = s.segments.map((x) => Math.max(1, parseInt(String(x), 10) || 1));
        this.renderAll();
        this.syncToWidget();
    }

    undo() {
        const snap = this.undoStack.pop();
        if (!snap) return;
        this.redoStack.push(JSON.stringify(this.serialize(false)));
        this.applySnapshot(snap);
    }

    redo() {
        const snap = this.redoStack.pop();
        if (!snap) return;
        this.undoStack.push(JSON.stringify(this.serialize(false)));
        this.applySnapshot(snap);
    }

    toggleAB() {
        if (!this.abSnap) {
            this.abSnap = JSON.stringify(this.serialize(false));
            this.abBtn.textContent = "A/B：A已存 · 当前B";
        } else {
            const cur = JSON.stringify(this.serialize(false));
            this.applySnapshot(this.abSnap);
            this.abSnap = cur;
            this.abBtn.textContent = "A/B：回到A · 按再回B";
        }
    }

    /* ── DOM ──────────────────────────────────────────────────── */

    mount() {
        injectGradeStyles();
        const container = document.createElement("div");
        container.className = "mmx-grade-host";
        const widget = this.node.addDOMWidget(GRADE_DOM_WIDGET_NAME, "grade", container, {
            getValue: () => "",
            setValue: () => {},
            getMinHeight: () => 470,
            hideOnZoom: false,
        });
        widget.element = container;
        this.domWidget = widget;
        this.buildDOM(container);
        this.rootEl = container;
        this.loadFromWidget();
        this.renderAll();
        this.refreshPresets();
        // 刷新/重开浏览器后自动加载磁盘上最新的预览（LRU 槽位），
        // 文件不存在时 blob 加载失败会自动回到占位提示。
        this.tryLoadPreviewFile();
        setTimeout(() => {
            this.harvestSegments();
            this.harvestBounds();
        }, 0);
    }

    /** 面板级锁：计算中禁用所有交互元素（新渲染的元素也要带上锁状态）。 */
    _lockEl(el) {
        if (el && this.uiLocked) el.disabled = true;
        return el;
    }

    el(tag, cls, text) {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }

    buildDOM(root) {
        const head = this.el("div", "mg-head");
        head.appendChild(this.el("span", "mg-title", "调色台 v52"));
        this.abBtn = this.el("button", "mg-btn", "A/B：存A");
        this.abBtn.title = "参数快照 A/B 对比：第1次存A（当前为B）；再按回到A，第三次回到B";
        this.abBtn.addEventListener("click", () => this.toggleAB());
        head.appendChild(this.abBtn);
        const undoBtn = this.el("button", "mg-btn", "撤销");
        undoBtn.title = "撤销上一次参数修改（快照栈，最多40步）";
        undoBtn.addEventListener("click", () => this.undo());
        head.appendChild(undoBtn);
        const redoBtn = this.el("button", "mg-btn", "重做");
        redoBtn.addEventListener("click", () => this.redo());
        head.appendChild(redoBtn);
        const resetBtn = this.el("button", "mg-btn", "本页恢复默认");
        resetBtn.addEventListener("click", () => {
            const scope = this.currentScope();
            for (const k of GRADE_PARAM_KEYS) scope[k] = defaultGradeParams()[k];
            this.renderSliders();
            this.pushUndo();
        });
        head.appendChild(resetBtn);
        const execBtn = this.el("button", "mg-btn", "▶ 执行优化");
        execBtn.title = "把当前所有滑块参数渲染成预览（唯一计算入口；改参数不自动算，调完一起点这里）";
        execBtn.addEventListener("click", () => this._doRerun());
        head.appendChild(execBtn);
        const spikeLabel = this.el("label", "mg-check");
        this.spikeCb = document.createElement("input");
        this.spikeCb.type = "checkbox";
        this.spikeCb.checked = !!this.state.spike_fix;
        this.spikeCb.addEventListener("change", () => {
            this.state.spike_fix = this.spikeCb.checked;
            this.pushUndo();
        });
        spikeLabel.appendChild(this.spikeCb);
        spikeLabel.appendChild(document.createTextNode("尖峰修复"));
        spikeLabel.title = "自动修复段接缝处的单帧亮度尖峰（前后帧一致、中间一帧离群时拉回局部趋势）";
        head.appendChild(spikeLabel);
        this.statusEl = this.el("span", "mg-status", "");
        head.appendChild(this.statusEl);
        root.appendChild(head);

        // 预览
        const preview = this.el("div", "mg-preview");
        this.placeholderEl = this.el("div", "mg-preview-empty", "运行后在此预览成片（连接 images → 队列）");
        preview.appendChild(this.placeholderEl);
        this.video = document.createElement("video");
        this.video.controls = true;
        this.video.playsInline = true;
        this.video.loop = true;
        this.video.style.display = "none";
        this.video.addEventListener("error", () => {
            // 文件不存在（预览开关关闭/尚未运行）→ 回到占位提示。
            this.video.style.display = "none";
            if (this.placeholderEl) this.placeholderEl.style.display = "";
        });
        preview.appendChild(this.video);
        root.appendChild(preview);

        // 段标签
        this.tabsEl = this.el("div", "mg-tabs");
        root.appendChild(this.tabsEl);

        // 滑块区
        this.slidersEl = this.el("div", "mg-sliders");
        root.appendChild(this.slidersEl);

        // 自动优化
        const auto = this.el("div", "mg-sec");
        auto.appendChild(this.el("div", "mg-sec-title", "自动优化（建议值 → 一键写入滑块）"));
        const autoRow = this.el("div", "mg-auto");
        const mk = (label, mode, global) => {
            const b = this.el("button", "mg-btn", label);
            b.title = global
                ? "测量所有接缝取均值（对比度均衡：以段1为参考）"
                : "只测量当前选中段的入口接缝（段1为参考段，无可分析入口）";
            b.addEventListener("click", () => this.requestAuto(mode, global ? -1 : this.activeTab - 1));
            this.autoButtons.push({ b, global });
            autoRow.appendChild(b);
        };
        this.autoButtons = [];
        mk("接缝漂移·全局", "seam", true);
        mk("接缝漂移·本段", "seam", false);
        mk("对比度均衡·全局", "contrast", true);
        mk("对比度均衡·本段", "contrast", false);
        const driftBtn = this.el("button", "mg-btn", "各段漂移值");
        driftBtn.title = "计算各段入口的亮度/对比度/饱和度/色相漂移原始值（仅供参考，不写参数）";
        driftBtn.addEventListener("click", () => this.requestAuto("drift_report", -1));
        autoRow.appendChild(driftBtn);
        this.applyBtn = this.el("button", "mg-btn", "应用建议");
        this.applyBtn.disabled = true;
        this.applyBtn.title = "把最近一次分析的建议值写入对应滑块（队列运行后生效）";
        this.applyBtn.addEventListener("click", () => this.applySuggestions());
        autoRow.appendChild(this.applyBtn);
        const saveBtn = this.el("button", "mg-btn", "▶ 队列出片");
        saveBtn.title =
            "只重跑调色台及其下游（CreateVideo → SaveVideo），上游导演台走缓存不重新生成；" +
            "SaveVideo 会把最终成片写到 ComfyUI 的 output 目录。首次使用前请先完整运行过一次工作流。";
        saveBtn.addEventListener("click", () => this.queueSave());
        autoRow.appendChild(saveBtn);
        auto.appendChild(autoRow);
        // 参数库：保存（自动命名 日期_次数）/ 列表读取 / 删除。
        const presetRow = this.el("div", "mg-auto");
        presetRow.appendChild(this.el("span", "", "参数库："));
        const savePBtn = this._lockEl(this.el("button", "mg-btn", "存参数"));
        savePBtn.title = "把当前滑块参数存进参数库（命名 = 日期 + 当天保存次数，如 20260206_01）";
        savePBtn.addEventListener("click", () => this.savePreset());
        presetRow.appendChild(savePBtn);
        this.presetSel = this._lockEl(document.createElement("select"));
        this.presetSel.title = "已保存的参数列表";
        presetRow.appendChild(this.presetSel);
        const loadPBtn = this._lockEl(this.el("button", "mg-btn", "读取"));
        loadPBtn.title = "把列表里选中的参数写入滑块并实时重算预览";
        loadPBtn.addEventListener("click", () => this.loadPreset());
        presetRow.appendChild(loadPBtn);
        const delPBtn = this._lockEl(this.el("button", "mg-btn", "删除"));
        delPBtn.title = "删除列表里选中的参数";
        delPBtn.addEventListener("click", () => this.deletePreset());
        presetRow.appendChild(delPBtn);
        auto.appendChild(presetRow);
        this.notesEl = this.el("div", "mg-notes");
        auto.appendChild(this.notesEl);
        root.appendChild(auto);
    }

    currentScope() {
        if (this.activeTab === 0) return this.state.global;
        while (this.state.per_segment.length < this.activeTab) this.state.per_segment.push(defaultGradeParams());
        return this.state.per_segment[this.activeTab - 1];
    }

    scopeLabel(scope) {
        return this.activeTab === 0 ? "全片" : "段" + this.activeTab;
    }

    renderTabs() {
        this.tabsEl.replaceChildren();
        const segs = this.resolveSegments();
        const tabs = ["全片"];
        for (let i = 0; i < segs.length; i++) tabs.push("段" + (i + 1));
        for (const ab of this.autoButtons || []) {
            if (!ab.global) ab.b.disabled = this.activeTab === 0;
        }
        tabs.forEach((label, i) => {
            const t = this._lockEl(this.el("button", "mg-tab" + (i === this.activeTab ? " mg-active" : ""), label));
            t.title =
                i === 0 ? "全片统一参数" : `段${i}（帧 ${segs.slice(0, i - 1).reduce((a, b) => a + b, 0)} 起）`;
            t.addEventListener("click", () => {
                this.activeTab = i;
                this.renderTabs();
                this.renderSliders();
                this.seekToSegment(i);
            });
            this.tabsEl.appendChild(t);
        });
        if (!segs.length && this.state.auto_requests == null) {
            this.showNotes([
                "未检测到分段：连接导演台的 timeline_data 输出，或先运行一次工作流以同步段边界。",
            ]);
        }
    }

    seekToSegment(tab) {
        if (!this.video || !this.video.src || !this.video.duration || tab <= 0) return;
        const segs = this.resolveSegments();
        if (!segs.length) return;
        const start = segs.slice(0, tab - 1).reduce((a, b) => a + b, 0);
        const fps = Number(this.node.widgets?.find((w) => w.name === "fps")?.value || 24) || 24;
        try {
            this.video.currentTime = Math.min(start / fps, this.video.duration - 0.05);
        } catch {
            /* ignore */
        }
    }

    renderSliders() {
        this.slidersEl.replaceChildren();
        const scope = this.currentScope();
        for (const d of GRADE_PARAM_DEFS) {
            const row = this.el("div", "mg-row");
            row.appendChild(this.el("label", "mg-lab", d.label));
            const input = this._lockEl(document.createElement("input"));
            input.type = "range";
            input.min = d.min;
            input.max = d.max;
            input.step = d.step;
            input.value = scope[d.key];
            // 数值输入框：点击可直接输入精确数字（滑块难控制精确值）。
            const num = this._lockEl(document.createElement("input"));
            num.type = "number";
            num.className = "mg-num";
            num.min = d.min;
            num.max = d.max;
            num.step = d.step;
            num.value = Number(scope[d.key]);
            num.title = "点击输入精确数值，回车/失焦生效（点「执行优化」统一计算）";
            input.addEventListener("input", () => {
                scope[d.key] = clampParam(d.key, input.value);
                num.value = scope[d.key];
            });
            input.addEventListener("change", () => {
                // 只记录参数与快照，不实时计算（点「执行优化」统一计算）。
                this.pushUndo();
            });
            num.addEventListener("change", () => {
                scope[d.key] = clampParam(d.key, num.value);
                input.value = scope[d.key];
                this.pushUndo();
            });
            num.addEventListener("keydown", (e) => {
                if (e.key === "Enter") num.blur();
            });
            row.appendChild(input);
            row.appendChild(num);
            this.slidersEl.appendChild(row);
        }
    }

    renderAll() {
        this.spikeCb.checked = !!this.state.spike_fix;
        this.renderTabs();
        this.renderSliders();
    }

    showNotes(lines) {
        this.lastNotes = Array.isArray(lines) ? lines.slice() : [String(lines)];
        this.notesEl.replaceChildren();
        for (const line of this.lastNotes) this.notesEl.appendChild(this.el("div", "", line));
    }

    setBusy(busy) {
        this.uiLocked = !!busy;
        if (this.statusEl) this.statusEl.textContent = busy ? "⏳ 计算中…" : "";
        if (this.rootEl) {
            for (const el of this.rootEl.querySelectorAll("button, input, select")) {
                el.disabled = !!busy;
            }
        }
    }

    /**
     * 加载磁盘预览：优先当前槽位，逐个回退 0/1/2 直到命中；全部缺失时
     * 显示占位说明。quiet=true 时不打扰提示区（运行结束时用）。
     */
    tryLoadPreviewFile(opts = {}) {
        const cur = (parseInt(this.state.preview_slot, 10) || 0) % 3;
        this._loadPreviewSlots([cur, (cur + 2) % 3, (cur + 1) % 3], 0, opts);
    }

    async _loadPreviewSlots(slots, idx, opts) {
        if (idx >= slots.length) {
            this.video.style.display = "none";
            if (this.placeholderEl) {
                this.placeholderEl.style.display = "";
                this.placeholderEl.textContent =
                    "未找到预览文件——拖动一次滑块（实时重算）或运行一次完整流程即可生成";
            }
            return;
        }
        const name = `minimax_grade_preview_${this.node.id}_${slots[idx]}.mp4`;
        const ok = await this.setVideoByName(name, { silent: true });
        if (!ok) {
            this._loadPreviewSlots(slots, idx + 1, opts);
            return;
        }
        if (!opts.quiet) {
            this.showNotes([`已加载磁盘上的预览：${name}（滑块已归零，调完参数点「▶ 执行优化」重算）`]);
        }
    }

    /* ── 执行优化（唯一计算入口：改参数不自动算，点按钮统一计算） ──── */

    _doRerun() {
        if (this._rerunActive) {
            this._rerunPending = true; // 正在算：合并为"算完再算一次"
            return;
        }
        this._rerunActive = true;
        // 预览 LRU 槽位轮换：只保留最近 3 个预览文件，新计算覆盖最早的。
        this.state.preview_slot = ((parseInt(this.state.preview_slot, 10) || 0) + 1) % 3;
        this.syncToWidget();
        this.showNotes(["执行优化计算中…（只更新预览，不写保存文件）"]);
        postGradePreview(this, { analysis: false });
    }

    _afterPreviewRun() {
        this._rerunActive = false;
        if (this._rerunPending) {
            this._rerunPending = false;
            this._doRerun();
        }
    }

    /**
     * 由文件名设置面板预览（实时路由 / executed 消息 / 兜底共用）。
     * fetch → blob → objectURL：强制加载新内容，绕开浏览器/代理对同名
     * mp4 的缓存。返回是否成功。
     */
    async setVideoByName(name, opts = {}) {
        if (!name) return false;
        try {
            const url =
                `/view?filename=${encodeURIComponent(name)}&type=output&subfolder=&t=${Date.now()}`;
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const blob = await res.blob();
            if (!blob.size) throw new Error("空文件");
            if (this._videoObjectUrl) URL.revokeObjectURL(this._videoObjectUrl);
            this._videoObjectUrl = URL.createObjectURL(blob);
            this.video.src = this._videoObjectUrl;
            this.video.style.display = "";
            if (this.placeholderEl) this.placeholderEl.style.display = "none";
            this.video.load();
            if (!opts.silent) {
                console.log(
                    `[MiniMaxH3Grade] 预览已更新（${new Date().toLocaleTimeString()}）：`,
                    name,
                    blob.size,
                    "bytes",
                );
            }
            return true;
        } catch (err) {
            if (!opts.silent) {
                console.warn("[MiniMaxH3Grade] 预览加载失败：", name, err?.message || err);
            }
            return false;
        }
    }

    /** 自动分析结果 → 建议值 + 提示区（实时路由 / executed 消息共用）。 */
    applyAutoResults(ar) {
        if (!ar || typeof ar !== "object") return;
        this.lastSuggestions = ar.suggestions || null;
        const notes = Array.isArray(ar.notes) ? ar.notes.slice() : [];
        // 各段漂移参考表（原始实测值，与后端日志"色调角"同口径）。
        if (Array.isArray(ar.drift_report) && ar.drift_report.length) {
            notes.push("── 各段间漂移参考（上段尾 ↔ 本段头，同内容相邻帧）──");
            for (const r of ar.drift_report) {
                const b = r.bright >= 0 ? `+${r.bright}` : `${r.bright}`;
                const h = r.hue >= 0 ? `+${r.hue}` : `${r.hue}`;
                notes.push(
                    `段${r.seg}入口: 亮度 ${b}% | 对比度 ×${r.contrast} | 饱和度 ×${r.sat} | 色相 ${h}°${
                        r.hue_flag ? "（>15°，疑似镜头变化）" : ""
                    }`,
                );
            }
        }
        notes.push("建议已生成——点「应用建议」写入滑块，再点「▶ 执行优化」查看效果");
        this.showNotes(notes);
        this.applyBtn.disabled = !this.lastSuggestions;
    }

    _onRunIdle() {
        this.setBusy(false);
        // 运行结束后一切归零：丢弃待算的实时预览请求，滑块重置为默认。
        this._rerunActive = false;
        this._rerunPending = false;
        if (this._saveQueued) {
            this._saveQueued = false;
            this.state.save_ref_copy = false;
            this.syncToWidget();
            this.showSaveResults();
        }
        // 点运行之后滑块重置为默认：修正值不延续到下一次运行，
        // 需要复用上次参数时从参数库显式读取。
        this.state.global = defaultGradeParams();
        this.state.per_segment = [];
        this.state.auto_requests = null;
        this.renderSliders();
        this.syncToWidget();
    }

    /** 出片完成后：显示所有保存文件的路径（下游 SaveVideo + input 参考副本）。 */
    showSaveResults() {
        const lines = [];
        if (this.lastRefCopy) {
            lines.push(`下一轮二采/三采参考副本：input\\${this.lastRefCopy}`);
        }
        for (const n of this.collectDownstream()) {
            if (!/SaveVideo/i.test(String(n.type || ""))) continue;
            const msg = app.nodeOutputs?.[String(n.id)];
            const u = (msg && typeof msg === "object" && msg.ui) || {};
            const vids = u.videos ?? u.animated ?? [];
            for (const v of vids) {
                const name = v?.filename ?? v?.name;
                if (name) {
                    const sub = v.subfolder ? `${v.subfolder}\\` : "";
                    lines.push(`成片已保存：output\\${sub}${name}`);
                }
            }
        }
        if (lines.length) {
            this.showNotes(lines);
        } else {
            this.showNotes(["出片完成（未检测到下游保存节点的输出路径）"]);
        }
    }

    /* ── 参数库（保存/读取/删除） ───────────────────────────────── */

    async refreshPresets() {
        try {
            const res = await api.fetchApi("/minimax/grade/presets");
            const data = await res.json().catch(() => ({}));
            this.presets = Array.isArray(data?.presets) ? data.presets : [];
        } catch {
            this.presets = [];
        }
        this.renderPresetSelect();
    }

    renderPresetSelect() {
        if (!this.presetSel) return;
        this.presetSel.replaceChildren();
        const none = document.createElement("option");
        none.value = "";
        none.textContent = "参数库（空）";
        this.presetSel.appendChild(none);
        for (const p of this.presets) {
            const o = document.createElement("option");
            o.value = p.name;
            o.textContent = `${p.name} (${p.created || ""})`;
            this.presetSel.appendChild(o);
        }
    }

    async savePreset() {
        try {
            const res = await api.fetchApi("/minimax/grade/presets/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ params: this.serialize(false) }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok || !data?.ok) throw new Error(data?.error || `HTTP ${res.status}`);
            await this.refreshPresets();
            if (this.presetSel) this.presetSel.value = data.name;
            this.showNotes([`参数已保存：${data.name}（${data.created}）——刷新/运行后滑块会归零，需要时从这里读取`]);
        } catch (err) {
            this.showNotes(["保存参数失败：" + (err?.message || err)]);
        }
    }

    loadPreset() {
        const name = this.presetSel?.value;
        const preset = this.presets.find((p) => p.name === name);
        if (!preset?.params) {
            this.showNotes(["请先在参数库列表里选择要读取的参数"]);
            return;
        }
        this.loadSnapshot(preset.params);
        this.showNotes([`已读取参数：${name} → 点「▶ 执行优化」查看效果`]);
    }

    async deletePreset() {
        const name = this.presetSel?.value;
        if (!name) {
            this.showNotes(["请先在参数库列表里选择要删除的参数"]);
            return;
        }
        try {
            const res = await api.fetchApi("/minimax/grade/presets/delete", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok || !data?.ok) throw new Error(data?.error || `HTTP ${res.status}`);
            await this.refreshPresets();
            this.showNotes([`已删除参数：${name}`]);
        } catch (err) {
            this.showNotes(["删除参数失败：" + (err?.message || err)]);
        }
    }

    /** 顺着本节点输出连线收集全部下游节点（调色台 → CreateVideo → SaveVideo 等）。 */
    collectDownstream() {
        const graph = app.graph ?? app.canvas?.graph;
        const nodes = graph?._nodes ?? graph?.nodes ?? [];
        const byId = {};
        for (const n of nodes) byId[String(n.id)] = n;
        const seen = new Set([String(this.node.id)]);
        const out = [this.node];
        const queue = [this.node];
        while (queue.length) {
            const cur = queue.shift();
            for (const o of cur.outputs || []) {
                for (const lid of o.links || []) {
                    const link = graphLinkById(graph, lid);
                    if (!link || link.targetId == null) continue;
                    const key = String(link.targetId);
                    if (seen.has(key)) continue;
                    const dst = byId[key];
                    if (!dst) continue;
                    seen.add(key);
                    out.push(dst);
                    queue.push(dst);
                }
            }
        }
        return out;
    }

    /**
     * 一键出片：提交完整图（XS 衍生版校验要求所有引用节点在场）。
     * 导演台/二采等未变节点走服务端缓存不重跑，只有调色台与下游
     * CreateVideo/SaveVideo 重新执行并写最终文件。
     */
    queueSave() {
        const set = this.collectDownstream();
        const extra = set.length - 1;
        if (extra <= 0) {
            this.showNotes([
                "未发现下游节点：请把调色台 images/audio 输出接到 CreateVideo → SaveVideo，再点出片。",
            ]);
            return;
        }
        const chain = set
            .map((n) => n.title || n.type)
            .join(" → ");
        // 出片时同时把调色后的全分辨率画面写进 input 目录，
        // 供下一轮二采/三采（导演台 passes）当参考视频引用。
        this.state.save_ref_copy = true;
        this._saveQueued = true;
        this.syncToWidget();
        this.showNotes([`队列出片（下游链：${chain}）… 完成后显示保存路径`]);
        queueFullGraph(this);
    }

    /* ── 自动优化 ─────────────────────────────────────────────── */

    requestAuto(mode, target) {
        this.state.auto_requests = { mode, target };
        this.syncToWidget();
        const modeLabel =
            mode === "seam" ? "接缝漂移" : mode === "contrast" ? "对比度均衡" : "各段漂移值";
        this.showNotes([`已请求${modeLabel}分析${target < 0 ? "（全局）" : "（本段）"}…`]);
        postGradePreview(this, { analysis: true });
    }

    applySuggestions() {
        if (!this.lastSuggestions) return;
        const s = this.lastSuggestions;
        const changed = [];
        if (s.global && typeof s.global === "object") {
            for (const k of GRADE_PARAM_KEYS) {
                if (s.global[k] != null) {
                    this.state.global[k] = clampParam(k, s.global[k]);
                    changed.push(`全片.${k}`);
                }
            }
        }
        if (Array.isArray(s.per_segment)) {
            s.per_segment.forEach((row, i) => {
                if (!row || typeof row !== "object") return;
                while (this.state.per_segment.length <= i) this.state.per_segment.push(defaultGradeParams());
                for (const k of GRADE_PARAM_KEYS) {
                    if (row[k] != null) {
                        this.state.per_segment[i][k] = clampParam(k, row[k]);
                        changed.push(`段${i + 1}.${k}`);
                    }
                }
            });
        }
        const notes = (this.lastNotes || []).slice();
        notes.push(changed.length ? `✓ 建议已写入滑块（${changed.join("、")}）→ 点「▶ 执行优化」查看效果` : "建议与当前值一致，无需修改");
        this.showNotes(notes);
        this.renderSliders();
        this.pushUndo();
    }

    /* ── 执行回调 ─────────────────────────────────────────────── */

    handleExecuted(message) {
        this._afterPreviewRun();
        this.setBusy(false);
        const ui = message && typeof message === "object" ? message.ui : null;
        const warn = typeof ui?.warning === "string" && ui.warning ? [ui.warning] : [];
        if (ui) {
            if (typeof ui.ref_copy?.path === "string" && ui.ref_copy.path) {
                this.lastRefCopy = ui.ref_copy.path;
            }
            const vids = Array.isArray(ui.videos) && ui.videos.length ? ui.videos : Array.isArray(ui.animated) ? ui.animated : [];
            if (vids.length) {
                const v = vids[0];
                const name = v?.filename ?? v?.name;
                if (name) {
                    this.setVideoByName(name);
                    console.log("[MiniMaxH3Grade] 已收到执行结果，预览视频：", name);
                }
            }
            if (Array.isArray(ui.segments) && ui.segments.length) {
                this.state.segments = ui.segments.map((x) => Math.max(1, parseInt(String(x), 10) || 1));
                this.syncToWidget();
                this.renderTabs();
            }
            if (typeof ui.auto_results === "string" && ui.auto_results) {
                try {
                    const ar = JSON.parse(ui.auto_results);
                    const notes = (warn || []).concat(Array.isArray(ar?.notes) ? ar.notes.slice() : []);
                    notes.push("点击「应用建议」写入滑块");
                    this.lastSuggestions = (ar && ar.suggestions) || null;
                    this.showNotes(notes);
                    this.applyBtn.disabled = !this.lastSuggestions;
                } catch {
                    this.lastSuggestions = null;
                }
            } else if (warn.length) {
                this.showNotes(warn);
            }
        }
        this.harvestBounds();
    }
}

app.registerExtension({
    name: "ComfyUI.MiniMaxH3DirectorPlugin.GradePanel",
    async setup() {
        injectGradeStyles();
        const allNodes = () => {
            const graph = app.graph ?? app.canvas?.graph;
            return graph?._nodes ?? graph?.nodes ?? [];
        };
        const flushGradeHarvest = () => {
            for (const n of allNodes()) {
                n._gradePanel?.harvestSegments?.();
                n._gradePanel?.harvestBounds?.();
            }
        };
        if (app.queuePrompt && !app.queuePrompt._minimaxGradeFlushPatched) {
            const orig = app.queuePrompt.bind(app);
            app.queuePrompt = function (...args) {
                flushGradeHarvest();
                return orig(...args);
            };
            app.queuePrompt._minimaxGradeFlushPatched = true;
        }
        api.addEventListener("executing", ({ detail }) => {
            if (detail == null) {
                for (const n of allNodes()) {
                    // 整次运行结束：复位锁定 + 处理出片收尾（清参考副本标志、
                    // 显示保存路径）+ 按固定文件名兜底加载预览。
                    n._gradePanel?._onRunIdle?.();
                    n._gradePanel?.tryLoadPreviewFile?.({ quiet: true });
                }
                flushGradeHarvest();
                return;
            }
            for (const n of allNodes()) {
                if (String(n.id) === String(detail)) n._gradePanel?.setBusy(true);
            }
        });
        // 导演台完成时 nodeOutputs 会带 report（含 [grade-bounds]）——立刻同步真实分段。
        // 同时把 executed 消息分发给调色台面板（不依赖节点原型 onExecuted，
        // 新版前端 1.23 的执行分发路径可能不经过它）。
        api.addEventListener("executed", ({ detail }) => {
            if (detail?.node != null) {
                for (const n of allNodes()) {
                    if (String(n.id) === String(detail.node)) {
                        n._gradePanel?.handleExecuted(detail?.output ?? {});
                        break;
                    }
                }
            }
            flushGradeHarvest();
        });
        api.addEventListener("execution_error", ({ detail }) => {
            if (detail?.node_id == null) return;
            for (const n of allNodes()) {
                if (String(n.id) === String(detail.node_id)) {
                    const p = n._gradePanel;
                    if (p) {
                        p._rerunActive = false;
                        p._rerunPending = false;
                        p.setBusy(false);
                        p.showNotes([detail?.exception_message || "执行失败"]);
                    }
                }
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if ((nodeType?.comfyClass || nodeData?.name || "") !== GRADE_NODE_TYPE) return;
        if (nodeType.prototype._minimaxGradePatched) return;
        nodeType.prototype._minimaxGradePatched = true;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            for (const w of this.widgets || []) {
                if (w.name === "grade_spec" || w.name === "timeline_data") hideGradeWidget(w);
            }
            this._gradePanel?.destroy?.();
            this._gradePanel = new GradePanel(this);
            this._gradePanel.mount();
            this.size = [640, 760];
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const r = onConfigure?.apply(this, arguments);
            const panel = this._gradePanel;
            if (panel) {
                panel.loadFromWidget();
                panel.renderAll();
                panel.harvestSegments();
                panel.harvestBounds();
            }
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const r = onConnectionsChange?.apply(this, arguments);
            this._gradePanel?.harvestSegments?.();
            this._gradePanel?.harvestBounds?.();
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            this._gradePanel?.handleExecuted(message ?? {});
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const r = onRemoved?.apply(this, arguments);
            this._gradePanel?.destroy?.();
            this._gradePanel = null;
            return r;
        };
    },
});

// 版本标记：F12 控制台应能看到这一行，用于确认浏览器加载的是最新 JS。
console.log("[MiniMaxH3Grade] 调色台面板 v8.6-grade 已加载（rebasedV2 重建分支）");
