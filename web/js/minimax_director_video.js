/* MiniMax H3 Director — 出片即时预览面板（提示词进度条 + 单文件预览）
 *
 * 设计（2026-10-03 三次改版，最终形态）：
 *   ① 跨运行拼 merged_latest.mp4 —— 每次运行/刷新都重拼，磁盘累积；缺段拼坏就黑屏。
 *   ② 前端播放列表逐段连播 —— 段间换 <video> 源有可见闪烁。
 *   ③ 流式（fragmented mp4 边拼边吐，不落盘）—— 不闪了，但 fMP4 不可随机 seek：
 *      拖动要重拉一条 ffmpeg 流，1–3 秒延迟，且 duration 未知、原生时间条不准。
 *   ④ 现在：GET /minimax/director/preview_file 把整条时间轴（缺失段用代码生成的
 *      占位片补位）拼成**一份普通 mp4 + faststart**，按内容指纹缓存在
 *      output/minimax_preview_cache/，前端只挂一个 <video src>：
 *        * seek 由浏览器走 HTTP range 本地完成 → 毫秒级，无重拉；
 *        * 普通 mp4 → duration 正确，原生时间条可用；
 *        * 内容没变直接命中既有文件（零 ffmpeg 开销）；改过一段才生成新 key；
 *        * 落盘累积由后端 prune_preview_cache 兜住（TTL / 数量 / 总大小 / 残留）。
 *      代价：首次出片多一次 concat 落盘（copy 模式通常 <1s，后端另有后台预热）。
 *
 * 布局（自上而下）：
 *   导演台时间轴 canvas（原生）
 *   ── 提示词进度条：嵌在时间轴 viewport 内、canvas 正下方，宽度 = canvas 宽，
 *      每块 left/width 用原生 frameToX(seg.start/length) 计算，与时间轴逐像素对齐。
 *   ── 出片预览面板：<video controls> 播整条预览文件；播放头由 JS 按时间→帧映射驱动。
 *      **不自动播放** —— 出片 / 换源 / 刷新后一律停在原位（首帧照画），等用户点播放键。
 *      唯一例外是「续播」：换源前用户本来就在播（点过播放键），换源后接着播，
 *      否则多段连续运行时每写盘一段预览都会被打断。
 *
 * 事件（后端 director/progress.py report_director_video）：
 *   kind="playlist" → detail.entries = [{index, frames, status, subfolder, filename}]
 *                     前端用它画进度条（含缺失段红斜纹）+ 请求预览文件。
 *   kind="segment"  → 单段刚写完（前端当成只有一条的播放列表）
 * 刷新恢复：POST /minimax/director/preview_plan（body 带**当前时间轴的段形状**：
 *   segments[{index,frames}] + frame_rate + task_type）→ {entries:[...]}。
 *   后端按这条时间轴展开：该段在任务表里有登记 → 真实片段；否则补「第 N 段缺失」占位片。
 *   拿不到形状 / 后端未重启 → 退回 GET /minimax/director/stitch_latest（只读任务表，
 *   段数只等于登记过的段数，仅作兜底）。
 *
 * 依赖 minimax_timeline.js 的符号（ES module，必须 import 而非读全局）：
 *   MiniMaxH3DirectorEditor / inputViewUrl
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { MiniMaxH3DirectorEditor, inputViewUrl } from "./minimax_timeline.js";

(function () {
    "use strict";

    var NODE_TYPES = new Set(["MiniMaxH3Director", "ComfyMiniMaxH3Director"]);

    function isDirNode(node) {
        return !!node && NODE_TYPES.has(String(node.type || ""));
    }

    function findDirectorNodeById(nodeId) {
        var g = (app && (app.graph || (app.canvas && app.canvas.graph))) || null;
        var nodes = g ? (g._nodes || g.nodes || []) : [];
        for (var i = 0; i < nodes.length; i++) {
            if (String(nodes[i].id) === String(nodeId)) return nodes[i];
        }
        return null;
    }

    function directorEditorClass() {
        return MiniMaxH3DirectorEditor || null;
    }

    function apiURL(path) {
        if (api && typeof api.apiURL === "function") {
            return api.apiURL(path);
        }
        return path;
    }

    /** output 目录下的文件 → ComfyUI /view URL（subfolder 用 / 分隔）。 */
    function outputViewUrl(filename, subfolder) {
        var params = new URLSearchParams({ filename: String(filename || ""), type: "output" });
        if (subfolder) params.set("subfolder", String(subfolder).replace(/\\/g, "/"));
        return apiURL("/view?" + params.toString());
    }

    function fmtTime(sec) {
        sec = Math.max(0, Number(sec) || 0);
        var h = Math.floor(sec / 3600);
        var m = Math.floor((sec % 3600) / 60);
        var s = Math.floor(sec % 60);
        var cs = Math.floor((sec % 1) * 100);
        var mm = String(m).padStart(2, "0");
        var ss = String(s).padStart(2, "0");
        var base = h > 0 ? h + ":" + mm + ":" + ss : mm + ":" + ss;
        return sec < 60 ? base + "." + String(cs).padStart(2, "0") : base;
    }

    function segTaskLabel(seg) {
        var raw = seg && (seg.taskType || seg.task_type);
        var txt = String(raw || "").trim();
        if (!txt) return "片段";
        var m = txt.match(/—\s*(.+)$/) || txt.match(/-\s*(.+)$/);
        return m ? m[1].trim() : txt;
    }

    function summarize(txt, max) {
        var s = String(txt || "").replace(/\s+/g, " ").trim();
        return s.length > max ? s.slice(0, max) + "…" : s;
    }

    function hasVideoSrc(v) {
        return !!v && !!String((v.src != null ? v.src : "") || v.getAttribute("src") || "");
    }

    function frameCountOf(seg) {
        if (!seg) return 0;
        return Number(seg.frameCount) || Number(seg.length) || 0;
    }

    var STYLES = [
        ".mmx-vp{display:flex;flex-direction:column;gap:6px;margin:8px 10px 10px;padding:8px 10px;border:1px solid #3f789e;border-radius:8px;background:rgba(63,120,158,.08);box-sizing:border-box;min-width:0;}",
        ".mmx-vp-head{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600;color:#cfe3f5;}",
        ".mmx-vp-badge{font-size:12px;font-weight:400;color:#8fd0ff;background:rgba(63,120,158,.18);padding:1px 8px;border-radius:999px;}",
        ".mmx-vp-tools{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:11px;font-weight:400;color:#9fb4c8;}",
        ".mmx-vp-dl{display:inline-flex;align-items:center;padding:1px 9px;border:1px solid rgba(143,208,255,.45);border-radius:6px;background:rgba(63,120,158,.18);color:#8fd0ff;font-size:11px;line-height:1.6;text-decoration:none;cursor:pointer;}",
        ".mmx-vp-dl:hover{background:rgba(63,120,158,.45);color:#e8f4ff;border-color:rgba(143,208,255,.75);}",
        ".mmx-vp-dl[aria-disabled='true']{opacity:.4;pointer-events:none;}",
        ".mmx-vp-video{width:100%;max-height:240px;background:#000;border-radius:6px;display:block;}",
        ".mmx-vp-empty{font-size:13px;color:#8aa;padding:14px 4px;}",
        // ---- 提示词进度条：嵌在时间轴 viewport 底部，与导演台分段逐像素对齐 ----
        ".mmx-vp-trackwrap{width:100%;}",
        ".mmx-vp-track{position:relative;display:block;height:34px;overflow:hidden;background:#151d29;border-top:1px solid #2a394d;cursor:pointer;touch-action:none;}",
        ".mmx-vp-block{position:absolute;top:0;bottom:0;min-width:3px;box-sizing:border-box;border-right:1px solid rgba(10,14,20,.9);overflow:hidden;}",
        ".mmx-vp-block-inner{position:absolute;inset:0;display:flex;align-items:center;gap:4px;padding:0 4px;box-sizing:border-box;background:linear-gradient(180deg,rgba(63,120,158,.55),rgba(63,120,158,.32));overflow:hidden;}",
        ".mmx-vp-block:nth-child(odd) .mmx-vp-block-inner{background:linear-gradient(180deg,rgba(72,101,140,.55),rgba(72,101,140,.32));}",
        ".mmx-vp-block:hover .mmx-vp-block-inner{filter:brightness(1.25);}",
        // 已播放部分：深一层，一眼看出「进度走到哪儿了」
        ".mmx-vp-block.mmx-vp-past .mmx-vp-block-inner{filter:brightness(.72) saturate(.85);}",
        ".mmx-vp-block.mmx-vp-active .mmx-vp-block-inner{background:linear-gradient(180deg,rgba(255,210,74,.42),rgba(255,210,74,.22));}",
        ".mmx-vp-block.mmx-vp-active::after{content:\"\";position:absolute;left:0;right:0;bottom:0;top:0;border:1px solid rgba(255,210,74,.85);pointer-events:none;}",
        // 缺失段（文件丢失 / 未生成）：红斜纹 + 警示，一眼看出「这段要重跑」
        ".mmx-vp-block.mmx-vp-missing .mmx-vp-block-inner{background:repeating-linear-gradient(45deg,rgba(190,52,52,.55),rgba(190,52,52,.55) 7px,rgba(120,30,30,.55) 7px,rgba(120,30,30,.55) 14px);}",
        ".mmx-vp-block.mmx-vp-missing.mmx-vp-active .mmx-vp-block-inner{background:repeating-linear-gradient(45deg,rgba(255,120,90,.6),rgba(255,120,90,.6) 7px,rgba(190,52,52,.6) 7px,rgba(190,52,52,.6) 14px);}",
        ".mmx-vp-block-warn{font-size:10px;line-height:1.1;color:#ffd9d2;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}",
        // 标红段上的「恢复」按钮：选一个现成 mp4 上传并绑到该段（不校验提示词一致）。
        ".mmx-vp-block.mmx-vp-missing .mmx-vp-block-inner{padding-right:42px;}",
        ".mmx-vp-recover{position:absolute;right:2px;top:2px;z-index:7;padding:2px 5px;border:1px solid rgba(255,205,190,.7);border-radius:4px;background:rgba(28,18,18,.82);color:#ffd9d2;font-size:10px;font-weight:600;line-height:1.2;cursor:pointer;white-space:nowrap;}",
        ".mmx-vp-recover:hover:not([disabled]){background:rgba(190,52,52,.9);border-color:#ffd9d2;color:#fff;}",
        ".mmx-vp-recover[disabled]{opacity:.6;cursor:progress;}",
        ".mmx-vp-thumb{width:22px;height:22px;flex:0 0 22px;object-fit:cover;border-radius:3px;background:#0d1219;}",
        ".mmx-vp-block-meta{display:flex;flex-direction:column;min-width:0;gap:1px;}",
        ".mmx-vp-block-label{font-size:10px;line-height:1.15;color:#eaf3fb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}",
        ".mmx-vp-block-label.mmx-vp-n{color:#ffd98a;font-weight:600;}",
        ".mmx-vp-block-prompt{font-size:10px;line-height:1.15;color:#a9c3da;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}",
        ".mmx-vp-playhead{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:#ffd24a;z-index:6;pointer-events:none;box-shadow:0 0 5px rgba(255,210,74,.9);}",
        ".mmx-vp-playhead::before{content:\"\";position:absolute;top:0;left:-3px;border-left:4px solid transparent;border-right:4px solid transparent;border-top:5px solid #ffd24a;}",
        ".mmx-vp-hint{font-size:11px;color:#7fa3c4;}"
    ].join("\n");

    function OutputVideoPreview(editor) {
        this.editor = editor;
        this.trackItems = [];
        this.totalSec = 0;
        this.fps = 24;
        this._restoring = false;
        this._scrubbing = false;
        this._lastFrame = -1;
        this._lastPlayFrame = null;
        // 预览文件状态：当前 <video src>（/view?...&filename=preview_<key>.mp4）与
        // 「载入后要恢复到的播放秒数」。同一 URL 不重设 src，避免白白重载一次。
        this._src = "";
        this._pendingResume = 0;
        this._hintBackup = null;
        this._buildDom();
        this._bind();
    }

    OutputVideoPreview.prototype._buildDom = function () {
        var el = document.createElement("div");
        el.className = "mmx-vp";
        el.style.display = "none";
        el.innerHTML =
            "<style>" + STYLES + "</style>" +
            '<div class="mmx-vp-head">' +
            '  <span class="mmx-vp-title">出片预览</span>' +
            '  <span class="mmx-vp-badge hidden"></span>' +
            '  <span class="mmx-vp-tools">' +
            '    <a class="mmx-vp-dl" href="#" aria-disabled="true" title="把这份预览文件下到本地 —— 与页面正在播的是同一个文件">下载</a>' +
            "  </span>" +
            "</div>" +
            '<div class="mmx-vp-empty">等待出片 — 运行完成后这里会出现整条预览，点播放键观看</div>' +
            '<video class="mmx-vp-video" controls playsinline preload="auto" muted></video>' +
            '<div class="mmx-vp-hint hidden"></div>';
        this.el = el;
        this.video = el.querySelector(".mmx-vp-video");
        this.video.style.display = "none";
        // 默认静音。**不自动播放**：不再依赖「静音才允许 autoplay」那条浏览器策略，
        // 保持静音是为了不让用户点播放键时突然出声（要听声自己点喇叭，
        // 开过之后 _userWantsSound=true，后续换源沿用有声）。
        this.video.muted = true;
        this._userWantsSound = false;
        this.badge = el.querySelector(".mmx-vp-badge");
        this.empty = el.querySelector(".mmx-vp-empty");
        this.hintEl = el.querySelector(".mmx-vp-hint");
        this.dlEl = el.querySelector(".mmx-vp-dl");
        // 进度条不放在面板里：它要嵌进时间轴 viewport（canvas 正下方）与分段对齐。
        var tw = document.createElement("div");
        tw.className = "mmx-vp-trackwrap hidden";
        tw.innerHTML =
            '<div class="mmx-vp-track" title="提示词进度条：与上方时间轴分段对齐 — 拖动跳转，双击某条跳到它开头"></div>';
        this.trackWrap = tw;
        this.trackEl = tw.querySelector(".mmx-vp-track");
    };

    OutputVideoPreview.prototype._bind = function () {
        var self = this;
        var v = this.video;
        // 面板内的拖动/点击不要冒泡到画布，否则会被当成「拖动节点」。
        ["pointerdown", "mousedown", "wheel", "dblclick"].forEach(function (evt) {
            self.el.addEventListener(evt, function (ev) { ev.stopPropagation(); }, { passive: true });
        });
        // 进度条嵌在时间轴 viewport 里（面板之外），必须自己截断冒泡，
        // 否则拖动/点击会被上层当成画布交互（拖动节点）。
        ["pointerdown", "mousedown", "wheel", "dblclick"].forEach(function (evt) {
            self.trackWrap.addEventListener(evt, function (ev) { ev.stopPropagation(); }, { passive: true });
        });
        v.addEventListener("timeupdate", function () { self._updatePlayhead(); });
        v.addEventListener("loadedmetadata", function () {
            self._applyResume();       // 换 src / 重载后回到原位（多段连跑不跳回开头）
            self._updatePlayhead();
        });
        // 首帧兜底：**不自动播放**，但要把第一帧画出来，避免 <video> 在 background:#000 下
        // 整块黑屏 —— 用户得先看见封面，才知道这里出了片、可以点播放。
        v.addEventListener("loadeddata", function () { self._ensureFirstFrame(); });
        // 记住用户是否手动开过声：开过则后续视频带声音播（用户已与页面交互，autoplay 允许）。
        v.addEventListener("volumechange", function () {
            self._userWantsSound = !v.muted;
        });
        v.addEventListener("seeking", function () { self._updatePlayhead(); });
        v.addEventListener("seeked", function () { self._updatePlayhead(); });

        // 提示词进度条拖动 / 点击 seek。进度条与时间轴帧对齐（不是时间等比），
        // 所以位置 → 全局帧 → 全局秒 → 直接设 <video>.currentTime：预览是普通 mp4
        // （faststart + range），浏览器本地跳转毫秒级，拖动中也能实时跟手。
        var seekFromEvent = function (ev) {
            if (!self.trackItems.length) return;
            var frame = self._frameFromClientX(ev.clientX);
            if (frame == null) return;
            self._seekToFrame(frame);
        };
        this.trackEl.addEventListener("pointerdown", function (ev) {
            self._scrubbing = true;
            try { self.trackEl.setPointerCapture(ev.pointerId); } catch (e) { /* noop */ }
            seekFromEvent(ev);
            ev.preventDefault();
        });
        this.trackEl.addEventListener("pointermove", function (ev) {
            if (self._scrubbing) seekFromEvent(ev);
        });
        var endScrub = function (ev) {
            if (!self._scrubbing) return;
            self._scrubbing = false;
            try { self.trackEl.releasePointerCapture(ev.pointerId); } catch (e) { /* noop */ }
            seekFromEvent(ev);
        };
        this.trackEl.addEventListener("pointerup", endScrub);
        this.trackEl.addEventListener("pointercancel", endScrub);
        // 双击某个块 → 跳到该条提示词的开头
        this.trackEl.addEventListener("dblclick", function (ev) {
            var frame = self._frameFromClientX(ev.clientX);
            if (frame == null) return;
            var idx = self._itemIdxAtFrame(frame);
            if (idx < 0) return;
            self._seekToFrame(self.trackItems[idx].gStart);
        });
    };

    OutputVideoPreview.prototype._timeline = function () {
        return (this.editor && this.editor.timeline) || {};
    };

    // ---------- 素材 ↔ 时间：两边互相对表 ----------

    /** 段的提示词：段级优先，回退全局提示词（editMode=global 时段级是空的）。 */
    OutputVideoPreview.prototype._segPrompt = function (seg) {
        var own = String((seg && (seg.prompt || seg.text || seg.positive)) || "").trim();
        if (own) return own;
        var g = this._timeline().global || {};
        return String(g.prompt || "").trim();
    };

    /** 素材缩略图：refs → 生成图 genImage → imageFile，逐个候选降级。 */
    OutputVideoPreview.prototype._thumbUrl = function (seg) {
        if (!seg) return null;
        var cands = [];
        var refs = Array.isArray(seg.refs) ? seg.refs : [];
        for (var i = 0; i < refs.length; i++) {
            var r = refs[i] || {};
            var rf = r.imageFile || r.fileName || "";
            if (rf) cands.push([rf, r.type || "input"]);
        }
        var gi = seg.genImage || {};
        if (gi.imageFile || gi.fileName) cands.push([gi.imageFile || gi.fileName, "input"]);
        if (seg.imageFile) cands.push([seg.imageFile, "input"]);
        if (seg.firstImageFile) cands.push([seg.firstImageFile, "input"]);
        for (var j = 0; j < cands.length; j++) {
            try {
                var u = inputViewUrl(cands[j][0], cands[j][1]);
                if (u) return u;
            } catch (e) { /* 换下一个候选 */ }
        }
        return null;
    };

    /** 全局帧落在哪条素材上（-1 = 落在没导出的段 / 还没有素材轴）。 */
    OutputVideoPreview.prototype._itemIdxAtFrame = function (frame) {
        for (var i = 0; i < this.trackItems.length; i++) {
            var it = this.trackItems[i];
            if (frame >= it.gStart && frame < it.gStart + it.gLen) return i;
        }
        return -1;
    };

    OutputVideoPreview.prototype._timelineTotalFrames = function () {
        var ed = this.editor;
        if (ed && typeof ed.getTotalFrames === "function") return Number(ed.getTotalFrames()) || 0;
        return Number(this._timeline().totalFrames) || 0;
    };

    /** 进度条上一点（clientX）→ 导演台全局帧（用原生 xToFrame，与时间轴同判）。 */
    OutputVideoPreview.prototype._frameFromClientX = function (clientX) {
        if (!this.trackEl) return null;
        var rect = this.trackEl.getBoundingClientRect();
        if (!rect.width) return null;
        var x = Math.min(rect.width, Math.max(0, clientX - rect.left));
        var ed = this.editor;
        if (ed && typeof ed.xToFrame === "function") return ed.xToFrame(x, rect.width);
        var total = this._timelineTotalFrames();
        if (!total) return null;
        return Math.min(total, Math.round((x / rect.width) * total));
    };

    /**
     * 对齐核心：进度条横宽 = 时间轴 canvas 宽，每块的 left/width 用原生
     * frameToX(seg.start / seg.length) 计算 → 与上方绿色时间轴的分段逐像素一致。
     * 时间轴放大变宽时由 ResizeObserver 触发重排。
     */
    OutputVideoPreview.prototype._layoutTrack = function () {
        if (!this.trackEl || !this.trackItems.length) return;
        var ed = this.editor;
        var W = (ed && ed.canvas && ed.canvas.clientWidth) || 0;
        if (!W) return;
        this._drawW = W;
        this.trackEl.style.width = W + "px";
        var total = this._timelineTotalFrames() || 1;
        var x = function (frame) {
            if (ed && typeof ed.frameToX === "function") return ed.frameToX(frame, W);
            return (frame / Math.max(1, total)) * W;
        };
        for (var i = 0; i < this.trackItems.length; i++) {
            var it = this.trackItems[i];
            if (!it._block) continue;
            var x0 = x(it.gStart);
            var x1 = x(it.gStart + it.gLen);
            it._block.style.left = x0.toFixed(2) + "px";
            it._block.style.width = Math.max(2, x1 - x0).toFixed(2) + "px";
        }
        var ph = this.trackEl.querySelector(".mmx-vp-playhead");
        if (ph && this._lastPlayFrame != null) {
            ph.style.left = Math.min(W, Math.max(0, x(this._lastPlayFrame))).toFixed(2) + "px";
        }
    };

    /** 时间轴 canvas 尺寸变化（缩放 / 节点宽度变化）→ 重排进度条。 */
    OutputVideoPreview.prototype._setupResize = function () {
        if (typeof ResizeObserver === "undefined") return;
        var canvas = this.editor && this.editor.canvas;
        if (!canvas) return;
        try {
            if (!this._ro) {
                var self = this;
                var raf = 0;
                this._ro = new ResizeObserver(function () {
                    if (raf) return;
                    raf = requestAnimationFrame(function () {
                        raf = 0;
                        self._layoutTrack();
                        self._updatePlayhead();
                    });
                });
            }
            this._ro.disconnect();
            this._ro.observe(canvas);
        } catch (e) { /* noop */ }
    };

    /** 挂 DOM：进度条进时间轴 viewport（canvas 之后），面板紧随 viewport 之后。 */
    OutputVideoPreview.prototype._mountDom = function () {
        var ed = this.editor;
        if (!ed) return;
        try {
            if (ed.viewport && this.trackWrap && this.trackWrap.parentElement !== ed.viewport) {
                ed.viewport.appendChild(this.trackWrap);
            }
            // 必须精确卡在 viewport 正后方（进度条正下方）。
            // 不能只看 parentElement 是否一致：早期 fallback 挂到 mainBody 末尾时
            // 爸爸也是 mainBody，旧判断会误判「已挂好」而永远不再纠正位置。
            if (ed.viewport && ed.viewport.parentElement) {
                if (this.el.parentElement !== ed.viewport.parentElement
                    || this.el.previousElementSibling !== ed.viewport) {
                    ed.viewport.insertAdjacentElement("afterend", this.el);
                }
            } else if (!this.el.isConnected && ed.mainBody) {
                ed.mainBody.appendChild(this.el);
            }
        } catch (e) {
            try { if (ed.mainBody) ed.mainBody.appendChild(this.el); } catch (e2) { /* noop */ }
        }
        this._setupResize();
    };

    // ---------- 单文件预览播放 ----------
    //
    // 后端 /minimax/director/preview_file 把整条时间轴（含缺失段占位片）拼成**一份普通
    // mp4（faststart）**并按内容指纹缓存，前端只挂一个 src：
    //   * 段间零换源 → 不闪；
    //   * 普通 mp4 + HTTP range → 拖动直接设 currentTime，毫秒级，不再重拉 ffmpeg 流；
    //   * 内容没变 → 后端命中缓存，秒回（URL 不变时前端也不重设 src）。

    /** 当前导演台节点 id（预览路由用它命中本进程缓存的播放列表）。 */
    OutputVideoPreview.prototype._nodeId = function () {
        var ed = this.editor;
        var n = ed && ed.node;
        return (n && n.id != null) ? n.id : null;
    };

    /** 问后端要「预览文件」的 /view URL（内容未变时后端直接命中缓存）。 */
    OutputVideoPreview.prototype._fetchPreviewFile = function () {
        var params = new URLSearchParams();
        var nid = this._nodeId();
        if (nid != null) params.set("node_id", String(nid));
        // 顺带把**当前时间轴的段形状**带上（紧凑串 "0:124,1:124"）。后端据此算出
        // 「工作流身份指纹」：多个工作流各放一个导演节点时（id 都可能正好是 5），
        // 别条时间轴的预览快照不会被复用 —— 否则这里会拿到别的片子的条目。
        var shape = this._timelineShape();
        if (shape && shape.segments && shape.segments.length) {
            var parts = [];
            for (var i = 0; i < shape.segments.length; i++) {
                var s = shape.segments[i];
                parts.push((s.index != null ? s.index : i) + ":" + (s.frames || 0));
            }
            params.set("shape", parts.join(","));
            if (shape.task_type) params.set("task_type", shape.task_type);
            if (shape.frame_rate) params.set("frame_rate", String(shape.frame_rate));
        }
        return fetch(apiURL("/minimax/director/preview_file?" + params.toString()), { cache: "no-store" })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (d) {
                if (!d || !d.filename) return null;
                return outputViewUrl(d.filename, d.subfolder);
            })
            .catch(function () { return null; });
    };

    /** 载入（或重载）整条时间轴的预览文件，可选回到 resumeSec 秒。 */
    OutputVideoPreview.prototype._loadPreview = function (resumeSec) {
        var self = this;
        var v = this.video;
        if (!v || !this.trackItems.length) return;
        var sec = Math.max(0, Number(resumeSec) || 0);
        this._pendingResume = sec;
        this._setLoading(true);
        this._fetchPreviewFile().then(function (url) {
            self._setLoading(false);
            // 下载按钮始终指向「页面正在播的那一份」；拿不到预览文件就置灰。
            self._setDownloadUrl(url || null);
            if (!url) return;
            if (self._src === url) {
                // 同一份预览文件：只把播放位置对齐，不重设 src（避免白白重载一次）。
                // 不动播放状态 —— 本来在播的继续播，本来暂停的保持暂停（**不自动播放**）。
                self._applyResume();
                self._updatePlayhead();
                return;
            }
            // 换源前记下「用户是不是正在播」：只有这一种情况换源后接着播（续播），
            // 否则一律停在原位（resumeSec / 0）等用户点播放键 —— 绝不「出片即播」。
            var wasPlaying = !!(v && !v.paused && hasVideoSrc(v));
            self._src = url;
            v.muted = !self._userWantsSound;
            v.style.display = "";
            v.src = url;
            self._lastFrame = -1;
            try { v.load(); } catch (e) { /* noop */ }
            self._applyResume();
            if (wasPlaying) self._play();
        });
    };

    /**
     * 另存为用的文件名：可读名 + 时间戳。
     *
     * 预览文件本身叫 ``preview_<内容指纹>.mp4`` —— 内容寻址、天然去重、不重复拼，
     * 当缓存 key 正合适，但拿去当交付文件名就是一串哈希。下载按钮现在是把整片
     * 拿出图面的主要出口，所以另存时换成「工作流名_日期时间.mp4」。
     */
    OutputVideoPreview.prototype._downloadName = function () {
        var base = "";
        try {
            var mgr = (typeof app !== "undefined" && app) ? app.workflowManager : null;
            var wf = mgr && mgr.activeWorkflow;
            var g = (typeof app !== "undefined" && app
                && (app.graph || (app.canvas && app.canvas.graph))) || null;
            base = (wf && (wf.name || wf.filename)) || (g && g.title) || "";
        } catch (e) { base = ""; }
        base = String(base || "").replace(/\.json$/i, "");
        base = base.replace(/[\\/:*?"<>|\u0000-\u001f]+/g, "_");
        base = base.replace(/\s+/g, "_").replace(/^[_.]+|[_.]+$/g, "");
        if (!base) base = "director";
        var d = new Date();
        var p = function (n) { return (n < 10 ? "0" : "") + n; };
        var ts = d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate())
            + "_" + p(d.getHours()) + p(d.getMinutes());
        return base + "_" + ts + ".mp4";
    };

    /**
     * 下载按钮指向**正在播的那一份预览文件**：同源 URL + <a download>，
     * 另存下来的字节与页面里看到的一致。
     *
     * 这版 ComfyUI 的 /view 没有 download 参数（不会回 Content-Disposition:
     * attachment），因此靠 <a download> 触发另存为：同源请求，浏览器认这个属性。
     */
    OutputVideoPreview.prototype._setDownloadUrl = function (url) {
        var a = this.dlEl;
        if (!a) return;
        if (!url) {
            a.setAttribute("aria-disabled", "true");
            a.removeAttribute("href");
            a.removeAttribute("download");
            a.removeAttribute("data-save-as");
            return;
        }
        var name = this._downloadName();
        a.href = url;
        a.setAttribute("download", name);
        a.setAttribute("data-save-as", name);
        a.setAttribute("title", "另存为 " + name + " —— 与页面正在播的是同一个文件");
        a.removeAttribute("aria-disabled");
    };

    /** 应用挂起的恢复位置（metadata 还没到就留到 loadedmetadata 再用）。 */
    OutputVideoPreview.prototype._applyResume = function () {
        var v = this.video;
        if (!v) return;
        var t = Math.max(0, Number(this._pendingResume) || 0);
        if (t <= 0.05) { this._pendingResume = 0; return; }
        if (v.readyState >= 1 && isFinite(v.duration) && v.duration > 0) {
            this._pendingResume = 0;
            try { v.currentTime = Math.min(t, Math.max(0, v.duration - 0.05)); } catch (e) { /* noop */ }
        }
    };

    /** 载入中提示（首次出片要拼一次；命中缓存则一闪而过）。 */
    OutputVideoPreview.prototype._setLoading = function (on) {
        var h = this.hintEl;
        if (!h) return;
        if (on) {
            if (this._hintBackup == null) this._hintBackup = h.textContent;
            h.textContent = "正在准备预览文件（按内容缓存，内容没变则秒开）…";
            h.classList.remove("hidden");
        } else if (this._hintBackup != null) {
            h.textContent = this._hintBackup;
            this._hintBackup = null;
        }
    };

    /** 全局帧 → 全局秒（条目累积时长的分段线性映射）。 */
    OutputVideoPreview.prototype._timeAtFrame = function (frame) {
        frame = Math.max(0, Math.round(Number(frame) || 0));
        var items = this.trackItems;
        for (var i = 0; i < items.length; i++) {
            var it = items[i];
            if (frame >= it.gStart && frame < it.gStart + it.gLen) {
                var r = it.gLen > 0 ? (frame - it.gStart) / it.gLen : 0;
                return it.startSec + r * (it.endSec - it.startSec);
            }
        }
        return items.length ? items[items.length - 1].endSec : 0;
    };

    /** 全局秒 → 全局帧（_timeAtFrame 的逆映射）。 */
    OutputVideoPreview.prototype._frameAtTime = function (ct) {
        var t = Math.max(0, Number(ct) || 0);
        var items = this.trackItems;
        for (var i = 0; i < items.length; i++) {
            var it = items[i];
            var last = (i === items.length - 1);
            if (t < it.endSec || last) {
                if (t <= it.startSec) return it.gStart;
                var dur = it.endSec - it.startSec;
                var r = dur > 0 ? Math.min(1, Math.max(0, (t - it.startSec) / dur)) : 0;
                return Math.round(it.gStart + r * it.gLen);
            }
        }
        return 0;
    };

    /** 播放时间落在哪条（进度条高亮用）。 */
    OutputVideoPreview.prototype._itemIdxAtTime = function (ct) {
        var t = Math.max(0, Number(ct) || 0);
        var items = this.trackItems;
        for (var i = 0; i < items.length; i++) {
            if (t < items[i].endSec) return i;
        }
        return items.length ? items.length - 1 : -1;
    };

    /**
     * 定位到全局帧：预览文件是普通 mp4（faststart + range），浏览器本地随机 seek
     * 毫秒级 —— 直接设 currentTime 即可，不必（也不能）重拉后端流。
     */
    OutputVideoPreview.prototype._seekToFrame = function (frame) {
        frame = Math.max(0, Math.round(Number(frame) || 0));
        this._lastPlayFrame = frame;
        var v = this.video;
        if (v && hasVideoSrc(v) && isFinite(v.duration) && v.duration > 0) {
            var t = this._timeAtFrame(frame);
            try { v.currentTime = Math.min(t, Math.max(0, v.duration - 0.05)); } catch (e) { /* noop */ }
        }
        this._updatePlayhead();
    };

    /**
     * 播放（暴露给内部唯一入口）。
     * 🔴 **不是「出片即看」的自动播放** —— 只在两种情况下调用：
     *   ① 用户点了播放键（原生 controls，浏览器自己处理，不走这里）；
     *   ② 换源前的「续播」：用户本来就在播，换源后接着播（见 _loadPreview）。
     * 其它任何时机（出片、刷新、恢复素材、段落写盘）都必须保持暂停。
     */
    OutputVideoPreview.prototype._play = function () {
        var v = this.video;
        if (!hasVideoSrc(v)) return;
        try {
            var pr = v.play();
            if (pr && pr.catch) pr.catch(function () { /* 被策略拦截：loadeddata 会兜底首帧 */ });
        } catch (e) { /* noop */ }
    };

    /** 没自动播放时，把首帧画出来，避免 <video> 在 background:#000 下整块黑屏。 */
    OutputVideoPreview.prototype._ensureFirstFrame = function () {
        var v = this.video;
        if (!v || !v.videoWidth) return;
        var ct = Number(v.currentTime) || 0;
        if (v.paused && ct < 0.05 && !(Number(this._pendingResume) > 0)) {
            try { v.currentTime = 0.01; } catch (e) { /* noop */ }
        }
    };

    // ---------- 详情 → 播放列表条目 ----------

    /** 事件 / 恢复数据 → 统一的播放列表条目 [{index,frames,status,subfolder,filename}]。 */
    OutputVideoPreview.prototype._entriesFromDetail = function (detail) {
        detail = detail || {};
        var norm = function (e, fallbackStatus) {
            return {
                index: Number(e && e.index),
                frames: Number(e && e.frames) || 0,
                status: String((e && e.status) || fallbackStatus || "ok"),
                subfolder: (e && e.subfolder != null) ? String(e.subfolder) : null,
                filename: (e && e.filename != null && String(e.filename) !== "")
                    ? String(e.filename) : null,
            };
        };
        if (Array.isArray(detail.entries) && detail.entries.length) {
            return detail.entries.map(function (e) { return norm(e, "ok"); });
        }
        // 单段事件：就是「只有一条的播放列表」
        if (detail.kind === "segment" && detail.segment_index != null) {
            return [norm({
                index: Number(detail.segment_index),
                frames: Number(detail.frame_count) || 0,
                status: "ok",
                subfolder: detail.subfolder,
                filename: detail.filename,
            }, "ok")];
        }
        // 旧版事件兜底：track / segments 只给布局，单文件引用落到第一条。
        var src = (Array.isArray(detail.track) && detail.track.length) ? detail.track
            : ((Array.isArray(detail.segments) && detail.segments.length) ? detail.segments : null);
        if (src) {
            return src.map(function (d, i) {
                var e = norm({ index: Number(d.index), frames: Number(d.frames) || 0, status: d.status }, "ok");
                if (i === 0 && detail.filename) {
                    e.subfolder = detail.subfolder;
                    e.filename = detail.filename;
                }
                return e;
            });
        }
        if (detail.filename) {
            return [norm({
                index: 0, frames: Number(detail.frame_count) || 0, status: "ok",
                subfolder: detail.subfolder, filename: detail.filename,
            }, "ok")];
        }
        return [];
    };

    /** 条目 → 时间轴物料（含每条的 clipUrl），无条目时按勾选/全量时间轴摆空轴。 */
    OutputVideoPreview.prototype._computeTrackFromEntries = function (entries) {
        var t = this._timeline();
        var segs = Array.isArray(t.segments) ? t.segments : [];
        var fps = Number((this.editor && typeof this.editor.getFrameRate === "function")
            ? this.editor.getFrameRate() : t.frameRate) || 24;
        this.fps = fps;
        var totalFrames = Number(t.totalFrames)
            || segs.reduce(function (s, x) { return s + frameCountOf(x); }, 0)
            || 1;

        if (!entries || !entries.length) {
            // 「只算跑过的段」开关已移除：固定按 runSelection 摆轴 —— 这正是该开关
            // 原先勾选时的分支（默认就是勾选的），去掉开关不改变默认行为。
            var sel = Array.isArray(t.runSelection)
                ? t.runSelection.filter(function (i) { return i >= 0 && i < segs.length; })
                : [];
            var idxs = (sel.length && sel.length < segs.length)
                ? sel : segs.map(function (_, i) { return i; });
            entries = idxs.map(function (i) {
                return { index: i, frames: frameCountOf(segs[i]), status: "ok", subfolder: null, filename: null };
            });
        }

        var acc = 0;
        var items = entries.map(function (e) {
            var i = Number(e.index);
            var seg = (i >= 0 && i < segs.length) ? segs[i] : null;
            var frames = Number(e.frames) || frameCountOf(seg);
            var dur = frames > 0 ? frames / fps : (Number(seg && seg.durationSec) || 0);
            // 该段在导演台时间轴上的全局帧区间（用于和上方的播放头互相对表）。
            var gStart = 0;
            var gLen = frames;
            if (seg && seg.length != null) {
                gStart = Number(seg.start) || 0;
                gLen = Number(seg.length) || frames;
            }
            var item = {
                seg: seg || {},
                i: i,
                status: e.status || "ok",
                startSec: acc,
                durSec: dur,
                endSec: acc + dur,
                gStart: gStart,
                gLen: gLen > 0 ? gLen : frames,
            };
            acc += dur;
            return item;
        });
        this.trackItems = items;
        this.totalSec = acc || (totalFrames / fps);
    };

    OutputVideoPreview.prototype._badgeText = function (kind, detail) {
        if (kind === "playlist") {
            var miss = 0;
            for (var i = 0; i < this.trackItems.length; i++) {
                if (this.trackItems[i].status === "missing") miss++;
            }
            var n = this.trackItems.length;
            return "完整时间轴 · " + n + " 段" + (miss ? " · " + miss + " 段需重新生成" : "");
        }
        if (kind === "segment") {
            var idx = Number(detail && detail.segment_index);
            return "第 " + (isFinite(idx) ? idx + 1 : "?") + " 段";
        }
        return "完整视频 · " + fmtTime(this.totalSec);
    };

    OutputVideoPreview.prototype.show = function (detail) {
        detail = detail || this._lastDetail;
        if (!detail) { this._setDownloadUrl(null); return; }
        var entries = this._entriesFromDetail(detail);
        if (!entries.length && !detail.filename) { this._setDownloadUrl(null); return; }
        this._lastDetail = detail;
        this._computeTrackFromEntries(entries);
        if (!this.trackItems.length) { this._setDownloadUrl(null); return; }
        this.el.style.display = "";
        this.empty.classList.add("hidden");
        this.badge.classList.remove("hidden");
        var kind = detail.kind || "segment";
        this.badge.textContent = this._badgeText(kind, detail);
        this._renderTrack();
        // 整条时间轴一份预览文件（缺段已用占位片补齐）——段间零换源、可随机 seek。
        // 多段连续运行时每次出片都会重新下发播放列表：保留当前播放位置，
        // 别让预览每跑完一段就跳回开头。
        var resumeSec = 0;
        if (this.video && hasVideoSrc(this.video)) {
            var ct0 = Number(this.video.currentTime) || 0;
            if (isFinite(ct0) && ct0 > 0.5) resumeSec = ct0;
        }
        this._loadPreview(resumeSec);
        this._updatePlayhead();
        this._notifyNodeHeight();
    };

    /** 当前时间轴的段形状（发给后端按它展开播放列表：段号 + 每段帧数）。 */
    OutputVideoPreview.prototype._timelineShape = function () {
        var ed = this.editor;
        var t = this._timeline();
        var segs = Array.isArray(t.segments) ? t.segments : [];
        if (!segs.length) {
            // 极端时序（DOM 刚建好、timeline 还没 parse 完）时，从 widget 原文兜一次。
            try {
                var raw = ed && ed.timelineWidget && ed.timelineWidget.value;
                var parsed = raw ? JSON.parse(raw) : null;
                if (parsed && Array.isArray(parsed.segments) && parsed.segments.length) {
                    t = parsed;
                    segs = parsed.segments;
                }
            } catch (e2) { /* 当作拿不到形状，走旧降级路 */ }
        }
        var fps = Number((ed && typeof ed.getFrameRate === "function")
            ? ed.getFrameRate() : t.frameRate) || Number(t.frameRate) || 24;
        var shape = [];
        for (var i = 0; i < segs.length; i++) {
            shape.push({ index: i, frames: Math.max(0, frameCountOf(segs[i])) });
        }
        var task = "";
        try {
            task = (ed && typeof ed.getTaskKey === "function" && ed.getTaskKey())
                || (ed && ed.taskTypeWidget && ed.taskTypeWidget.value)
                || (ed && ed.globalTask && ed.globalTask.value)
                || "";
        } catch (e) { task = ""; }
        return { segments: shape, frame_rate: fps, task_type: task };
    };

    /** 旧降级路：拿不到前端形状时，从任务表 / 最近一次导出目录自建列表。 */
    OutputVideoPreview.prototype._restoreFromServer = function () {
        var self = this;
        return fetch(apiURL("/minimax/director/stitch_latest"), { cache: "no-store" })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                if (data && Array.isArray(data.entries) && data.entries.length) {
                    self.show({ kind: "playlist", entries: data.entries });
                    return null;
                }
                // 旧后端 / 无任务表：退回最近一次运行目录，按 seg 文件自建播放列表。
                return fetch(apiURL("/minimax/director/latest_seg_export"), { cache: "no-store" })
                    .then(function (r) { return r.ok ? r.json() : null; })
                    .then(function (d2) {
                        if (!d2 || !d2.run_dir) return;
                        var sub = "minimax_seg_export/" + d2.run_dir;
                        var files = (d2.files || []).slice().sort();
                        var list = [];
                        for (var i = 0; i < files.length; i++) {
                            var m = /^seg_(\d{4})\.mp4$/.exec(files[i]);
                            if (m) {
                                list.push({
                                    index: parseInt(m[1], 10),
                                    frames: 0,
                                    status: "ok",
                                    subfolder: sub,
                                    filename: files[i],
                                });
                            }
                        }
                        if (list.length) self.show({ kind: "playlist", entries: list });
                    });
            });
    };

    OutputVideoPreview.prototype.restoreLatest = function () {
        if (this._restoring) return;
        this._restoring = true;
        var self = this;
        var done = function () { self._restoring = false; };
        // 首选：把**当前时间轴的段形状**发给后端，按这条时间轴展开播放列表 —— 有登记的
        // 段给真实片段，其余按时长补「第 N 段缺失」占位片。段数/每段帧数只有浏览器端
        // 知道，后端自己推不出来：只读任务表会塌成「只剩跑过的那几段」（16 段变 17 秒），
        // 按段号扫盘又会把**别的**提示词分组的素材拼进来（实测 83.38s 被拼成 5:39）。
        var shape = this._timelineShape();
        if (!shape.segments.length) {
            this._restoreFromServer().catch(function () { /* 静默 */ }).finally(done);
            return;
        }
        fetch(apiURL("/minimax/director/preview_plan"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                node_id: this._nodeId(),
                task_type: shape.task_type,
                frame_rate: shape.frame_rate,
                segments: shape.segments
            }),
            cache: "no-store"
        })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                if (data && Array.isArray(data.entries) && data.entries.length) {
                    self.show({ kind: "playlist", entries: data.entries });
                    return null;
                }
                return self._restoreFromServer();
            })
            .catch(function () { return self._restoreFromServer(); })
            .catch(function () { /* 后端未重启 / 无导出目录时静默 */ })
            .finally(done);
    };

    OutputVideoPreview.prototype._renderTrack = function () {
        if (!this.trackItems.length) {
            this.trackWrap.classList.add("hidden");
            this.hintEl.classList.add("hidden");
            return;
        }
        this.trackWrap.classList.remove("hidden");
        this.trackEl.innerHTML = "";
        var self = this;
        this.trackItems.forEach(function (item) {
            var missing = item.status === "missing";
            var block = document.createElement("div");
            block.className = "mmx-vp-block" + (missing ? " mmx-vp-missing" : "");
            var prompt = self._segPrompt(item.seg);
            block.title = "#" + (item.i + 1) + "  " + fmtTime(item.startSec) + " – " + fmtTime(item.endSec)
                + "  " + segTaskLabel(item.seg)
                + (missing
                    ? "\n⚠ 文件丢失，需重新生成（播放占位片）\n点右上角「恢复」可上传现成的 mp4 直接补上这一段"
                    : (prompt ? "\n" + summarize(prompt, 220) : ""));
            var inner = document.createElement("div");
            inner.className = "mmx-vp-block-inner";
            if (missing) {
                // 缺失段：不画缩略图，直接放「需重新生成」标记 + 「恢复」按钮（手上
                // 已有这段素材时，不必重跑生成 —— 上传后直接绑到这一段）。
                var warn = document.createElement("span");
                warn.className = "mmx-vp-block-warn";
                warn.textContent = "⚠ 需重新生成";
                inner.appendChild(warn);
            } else {
                var thumb = self._thumbUrl(item.seg);
                if (thumb) {
                    var img = document.createElement("img");
                    img.src = thumb;
                    img.className = "mmx-vp-thumb";
                    img.alt = "";
                    inner.appendChild(img);
                }
                var meta = document.createElement("div");
                meta.className = "mmx-vp-block-meta";
                var n = document.createElement("span");
                n.className = "mmx-vp-block-label mmx-vp-n";
                n.textContent = "#" + (item.i + 1) + " " + fmtTime(item.startSec);
                meta.appendChild(n);
                var p = document.createElement("span");
                p.className = "mmx-vp-block-prompt";
                p.textContent = prompt ? summarize(prompt, 24) : segTaskLabel(item.seg);
                meta.appendChild(p);
                inner.appendChild(meta);
            }
            block.appendChild(inner);
            if (missing) block.appendChild(self._recoverButton(item));
            item._block = block;
            self.trackEl.appendChild(block);
        });
        var ph = document.createElement("div");
        ph.className = "mmx-vp-playhead";
        this.trackEl.appendChild(ph);
        this._layoutTrack();
        this.hintEl.textContent = (this.trackItems.length === 1)
            ? "只有 1 条提示词：进度条即该段"
            : "进度条与时间轴分段对齐：拖动跳转，双击某条跳到它开头";
        this.hintEl.classList.remove("hidden");
        this._notifyNodeHeight();
    };

    // ---------- 标红段「恢复素材」：上传一个 mp4，直接绑到这一段 ----------
    //
    // 任务表是按「提示词 + 负向 + 参考素材 + 帧数」的**内容指纹**定位段的。改过时长
    // 或提示词的旧素材指纹必然对不上，于是那些段在预览里全成了红斜纹占位片 —— 而素材
    // 其实还在磁盘上。指纹证明不了的事由人担保：点「恢复」选个 mp4，后端把文件收进
    // output/minimax_seg_export/recovered_<ts>/ 并写一条 bind#<idx> 绑定（优先级低于
    // 内容指纹、高于占位片）。下次真跑这一段，绑定会被真实产物自动覆盖。

    /** 段的「恢复」按钮（进度条整条是拖动 seek 命中区，按钮必须自己吃掉指针事件）。 */
    OutputVideoPreview.prototype._recoverButton = function (item) {
        var self = this;
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "mmx-vp-recover";
        btn.textContent = "恢复";
        btn.title = "第 " + (item.i + 1) + " 段缺失：点这里选一个现成的 mp4，"
            + "上传后直接绑定到这一段（不校验提示词是否一致）";
        ["pointerdown", "mousedown", "click", "dblclick", "wheel"].forEach(function (evt) {
            btn.addEventListener(evt, function (ev) { ev.stopPropagation(); });
        });
        btn.addEventListener("click", function (ev) {
            ev.preventDefault();
            ev.stopPropagation();
            if (btn.disabled) return;
            self._pickRecoverFile(item, btn);
        });
        return btn;
    };

    /** 弹系统选文件框（hidden input 用完即弃，不污染 DOM）。 */
    OutputVideoPreview.prototype._pickRecoverFile = function (item, btn) {
        var self = this;
        var picker = document.createElement("input");
        picker.type = "file";
        picker.accept = "video/mp4,video/*,.mp4,.mov,.mkv,.webm,.m4v";
        picker.style.display = "none";
        document.body.appendChild(picker);
        picker.addEventListener("change", function () {
            var file = (picker.files && picker.files[0]) || null;
            try { picker.remove(); } catch (e) { /* noop */ }
            if (!file) return;
            self._recoverFromFile(item, file, btn);
        });
        picker.click();
    };

    /** 分块上传（与时间轴参考素材同一接口，8MB 一片，避免单次巨body）。 */
    OutputVideoPreview.prototype._uploadRecoverChunked = function (file, onProgress) {
        var CHUNK = 8 * 1024 * 1024;
        var total = Math.max(1, Math.ceil(file.size / CHUNK));
        var uploadId = (window.crypto && crypto.randomUUID)
            ? crypto.randomUUID()
            : ("mmxrec-" + Date.now() + "-" + Math.random().toString(16).slice(2));
        var name = String(file.name || "").trim() || ("recovered_" + Date.now() + ".mp4");
        var step = function (i) {
            if (i >= total) return Promise.resolve(null);
            var body = new FormData();
            body.append("upload_id", uploadId);
            body.append("chunk_index", String(i));
            body.append("total_chunks", String(total));
            body.append("filename", name);
            body.append("chunk", file.slice(i * CHUNK, Math.min(file.size, (i + 1) * CHUNK)), name + ".part");
            return fetch(apiURL("/minimax/director/upload_chunk"), { method: "POST", body: body })
                .then(function (r) {
                    if (!r.ok) {
                        return r.text().then(function (t) { throw new Error(t || ("HTTP " + r.status)); });
                    }
                    return r.json();
                })
                .then(function (d) {
                    if (typeof onProgress === "function") onProgress(i + 1, total);
                    return (d && d.name) ? d : step(i + 1);
                });
        };
        return step(0);
    };

    /** 上传 → 绑定 → 就地刷新这一段。 */
    OutputVideoPreview.prototype._recoverFromFile = function (item, file, btn) {
        var self = this;
        var label = String(file.name || "文件");
        if (btn) { btn.disabled = true; btn.textContent = "上传中"; }
        var restoreBtn = function () {
            if (!btn) return;
            btn.disabled = false;
            btn.textContent = "恢复";
        };
        this._setHint("正在上传「" + summarize(label, 40) + "」并绑定到第 " + (item.i + 1) + " 段…", 0);
        this._uploadRecoverChunked(file, function (done, total) {
            if (btn && total > 1) btn.textContent = Math.round((done / total) * 100) + "%";
        })
            .then(function (up) {
                if (!up || !up.name) throw new Error("上传未完成");
                return self._bindRecovered(item, {
                    filename: up.name,
                    type: up.type || "input",
                    subfolder: up.subfolder || ""
                });
            })
            .then(function (data) {
                restoreBtn();
                self._applyRecoveredEntry(item.i, data);
            })
            .catch(function (err) {
                restoreBtn();
                self._setHint("「" + summarize(label, 40) + "」恢复失败："
                    + ((err && err.message) ? err.message : err), 6000);
            });
    };

    /** 后端绑定：把一个已有文件登记成该段的 bind#<idx> 素材。 */
    OutputVideoPreview.prototype._bindRecovered = function (item, payload) {
        var self = this;
        var ed = this.editor;
        var taskType = "";
        try {
            taskType = (ed && typeof ed.getTaskKey === "function" && ed.getTaskKey())
                || (ed && ed.taskTypeWidget && ed.taskTypeWidget.value)
                || (ed && ed.globalTask && ed.globalTask.value)
                || "";
        } catch (e) { taskType = ""; }
        var body = {
            node_id: self._nodeId(),
            index: item.i,
            task_type: taskType,
            filename: payload.filename,
            type: payload.type,
            subfolder: payload.subfolder
        };
        return fetch(apiURL("/minimax/director/recover_segment"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        }).then(function (r) {
            return r.text().then(function (text) {
                if (!r.ok) throw new Error(text || ("HTTP " + r.status));
                try { return JSON.parse(text); } catch (e) { return {}; }
            });
        });
    };

    /** 本地把这一段标成 ok 并重拼预览（不等后端重下整条列表）。 */
    OutputVideoPreview.prototype._applyRecoveredEntry = function (idx, data) {
        var found = false;
        var entries = [];
        for (var i = 0; i < this.trackItems.length; i++) {
            var it = this.trackItems[i];
            if (it.i === idx) {
                found = true;
                var frames = Number(data && data.frames) || 0;
                if (!frames) frames = Math.max(1, Math.round(it.durSec * (this.fps || 24)));
                entries.push({ index: idx, frames: frames, status: "ok" });
            } else {
                entries.push({
                    index: it.i,
                    frames: Math.max(1, Math.round(it.durSec * (this.fps || 24))),
                    status: it.status
                });
            }
        }
        if (!found) return;
        var resume = Number(this.video && this.video.currentTime) || 0;
        this._computeTrackFromEntries(entries);
        this._renderTrack();
        var sec = Number(data && data.frames) > 0
            ? fmtTime(Number(data.frames) / (this.fps || 24)) : "";
        this._setHint("已恢复第 " + (idx + 1) + " 段素材（" + (data && data.filename ? data.filename : "已绑定")
            + (sec ? "，实际时长 " + sec : "") + "）", 5000);
        // 内容变了 → 预览文件必然是新的一份：清掉 _src，强制换源重载。
        this._src = "";
        this._loadPreview(resume);
    };

    /** 提示条临时文案（ms<=0 表示常驻到下次覆盖）。 */
    OutputVideoPreview.prototype._setHint = function (text, ms) {
        var h = this.hintEl;
        if (!h) return;
        var self = this;
        if (this._hintTimer) { clearTimeout(this._hintTimer); this._hintTimer = null; }
        if (!this._hintActive) {
            this._hintBackup = h.textContent;
            this._hintActive = true;
        }
        h.textContent = text;
        h.classList.remove("hidden");
        if (!(Number(ms) > 0)) return;
        this._hintTimer = setTimeout(function () {
            self._hintTimer = null;
            self._hintActive = false;
            if (self._hintBackup != null) {
                h.textContent = self._hintBackup;
                self._hintBackup = null;
            }
        }, Number(ms));
    };

    OutputVideoPreview.prototype._updatePlayhead = function () {
        if (!this.trackEl || this.trackWrap.classList.contains("hidden")) return;
        var ct = (this.video && typeof this.video.currentTime === "number") ? this.video.currentTime : 0;
        // 预览文件从 0 秒起就是整条时间轴 → 播放时间即全局时间。
        var gt = ct;
        var frame = this._frameAtTime(gt);
        this._lastPlayFrame = frame;
        // 播放头按帧定位（与时间轴播放头同一个 x 算法），不是按时间百分比。
        var W = this._drawW
            || (this.editor && this.editor.canvas && this.editor.canvas.clientWidth)
            || this.trackEl.getBoundingClientRect().width;
        if (W) {
            var ed = this.editor;
            var x = (ed && typeof ed.frameToX === "function")
                ? ed.frameToX(frame, W)
                : (frame / Math.max(1, this._timelineTotalFrames() || 1)) * W;
            var ph = this.trackEl.querySelector(".mmx-vp-playhead");
            if (ph) ph.style.left = Math.min(W, Math.max(0, x)).toFixed(2) + "px";
        }
        var idx = this._itemIdxAtTime(gt);
        var kids = this.trackEl.children;
        for (var i = 0; i < kids.length; i++) {
            var b = kids[i];
            if (b.classList.contains("mmx-vp-playhead")) continue;
            b.classList.toggle("mmx-vp-active", i === idx);
            b.classList.toggle("mmx-vp-past", i < idx);
        }
        // 顺带把上方导演台的播放头与选中的提示词带过来。
        // 「联动导演台」开关已移除：固定联动 —— 这正是该开关原先勾选时的分支。
        this._syncEditorFrame(frame);
    };

    /** 把播放位置同步给上方导演台：播放头 + 选中该条提示词。 */
    OutputVideoPreview.prototype._syncEditorFrame = function (frame) {
        var ed = this.editor;
        if (!ed) return;
        if (ed.isPlaying) return;              // 编辑器自己在播，不抢它的播放头
        frame = Math.max(0, Math.round(Number(frame) || 0));
        if (typeof ed.getTotalFrames === "function") {
            var total = ed.getTotalFrames();
            if (total > 0) frame = Math.min(frame, total - 1);
        }
        if (frame === this._lastFrame) return;
        this._lastFrame = frame;
        try {
            ed.currentFrame = frame;
            if (ed.seekBar) {
                ed.seekBar.max = Math.max(0, (typeof ed.getTotalFrames === "function"
                    ? ed.getTotalFrames() : frame) - 1);
                ed.seekBar.value = frame;
            }
            if (typeof ed._updateTimelineDom === "function") {
                ed._updateTimelineDom({ skipSeek: true });
            }
            // 落在哪一段 → 选中它：底部提示词区会自动切到这条素材。
            var segs = (ed.timeline && ed.timeline.segments) || [];
            for (var i = 0; i < segs.length; i++) {
                var s = segs[i] || {};
                var st = Number(s.start) || 0;
                var ln = Number(s.length) || 0;
                if (frame >= st && frame < st + ln) {
                    if (ed.selectedIndex !== i) {
                        ed.selectedIndex = i;
                        if (typeof ed.updateSelectionUI === "function") ed.updateSelectionUI();
                    }
                    break;
                }
            }
            this._ensurePlayheadVisible(frame);
            if (typeof ed.scheduleRender === "function") ed.scheduleRender();
        } catch (e) { /* 编辑器结构变化时不影响预览 */ }
    };

    /** 放大后的时间轴横向滚动时，把播放头拽回视野。 */
    OutputVideoPreview.prototype._ensurePlayheadVisible = function (frame) {
        var ed = this.editor;
        if (!ed || !ed.viewport) return;
        try {
            if (typeof ed.getTimelineZoom === "function" && ed.getTimelineZoom() <= 1) return;
            var w = (typeof ed._measureDrawWidth === "function") ? ed._measureDrawWidth() : 0;
            if (!w || typeof ed.frameToX !== "function") return;
            var x = ed.frameToX(frame, w);
            var vp = ed.viewport;
            var pad = 24;
            if (x < vp.scrollLeft + pad || x > vp.scrollLeft + vp.clientWidth - pad) {
                vp.scrollLeft = Math.max(0, x - vp.clientWidth / 2);
            }
        } catch (e) { /* noop */ }
    };

    /** 反向：上方时间轴 seek / 点段 → 视频跳到同一素材位置。 */
    OutputVideoPreview.prototype.seekVideoToFrame = function (frame) {
        if (!this.trackItems.length) return;
        this._seekToFrame(frame);
    };

    OutputVideoPreview.prototype._notifyNodeHeight = function () {
        var ed = this.editor;
        if (!ed) return;
        try { if (typeof ed.updateDomWidgetHeight === "function") ed.updateDomWidgetHeight(); } catch (e) { /* noop */ }
        var node = ed.node;
        if (node && node.computeSize && node.size) {
            try {
                var ideal = node.computeSize();
                if (ideal && ideal[1] != null && (node.size[1] || 0) < ideal[1] - 2) {
                    node.setSize([node.size[0], ideal[1]]);
                    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
                }
            } catch (e) { /* noop */ }
        }
    };

    function mountPreview(editor) {
        if (!editor) return null;
        var pv = editor._videoPreview;
        if (pv) {
            // buildDOM 可能重建过 DOM（viewport / mainBody 都换新了），重新挂回去。
            pv._mountDom();
            return pv;
        }
        pv = new OutputVideoPreview(editor);
        editor._videoPreview = pv;
        pv._mountDom();
        // 页面刷新后恢复最近一次导出（后端路由需已重启生效，失败静默）。
        pv.restoreLatest();
        return pv;
    }

    function patchEditorBuild() {
        var Cls = directorEditorClass();
        if (!Cls || Cls.prototype._mmxVpPatched) return;
        Cls.prototype._mmxVpPatched = true;
        var origBuild = Cls.prototype.buildDOM;
        Cls.prototype.buildDOM = function () {
            var r = origBuild ? origBuild.apply(this, arguments) : undefined;
            try { mountPreview(this); } catch (e) { /* noop */ }
            return r;
        };
        // 拖动上方时间轴的播放头 → 下方视频跟着跳到同一条素材。
        var origSeek = Cls.prototype.seekToFrame;
        if (typeof origSeek === "function") {
            Cls.prototype.seekToFrame = function (frame, opts) {
                var first = origSeek.apply(this, arguments);
                try {
                    var pv = this._videoPreview;
                    // 「联动导演台」开关已移除 → 固定联动（即原先勾选时的行为）。
                    if (pv) pv.seekVideoToFrame(frame);
                } catch (e) { /* noop */ }
                void opts;
                return first;
            };
        }
    }

    function setup() {
        patchEditorBuild();
        api.addEventListener("minimax_director_video", function (ev) {
            var detail = ev && ev.detail;
            if (!detail || !detail.node_id) return;
            var node = findDirectorNodeById(detail.node_id);
            var ed = node && node._minimaxEditor;
            if (!ed) return;
            var pv = mountPreview(ed);
            if (pv) {
                // 后端推两类预览事件：
                //   - kind="playlist"：整条时间轴的播放列表（每段一条有效片段 / 占位片）；
                //   - kind="segment" ：单段刚写完（前端当成一条）。
                pv.show(detail);
            }
        });
        // 已存在 / 新拖入的导演台节点：编辑器挂好后补挂预览面板。
        setTimeout(function () {
            var g = (app && (app.graph || (app.canvas && app.canvas.graph))) || null;
            var nodes = g ? (g._nodes || g.nodes || []) : [];
            for (var i = 0; i < nodes.length; i++) {
                if (isDirNode(nodes[i]) && nodes[i]._minimaxEditor) mountPreview(nodes[i]._minimaxEditor);
            }
        }, 1200);
    }

    app.registerExtension({
        name: "MiniMaxH3Director.OutputVideoPreview",
        setup: setup,
        async loadedGraphNode(node) {
            if (!isDirNode(node)) return;
            var ed = node && node._minimaxEditor;
            if (ed) mountPreview(ed);
        },
    });
})();
