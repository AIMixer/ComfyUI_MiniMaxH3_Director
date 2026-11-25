/** The waveform and the picture strip share the same frame transform. */
import { api } from "../../scripts/api.js";
import { getLocale } from "./minimax_i18n.js";

export const MASTER_LANE_H = 96;
export const MASTER_KEY = "masterAudio";
export const MASTER_MAX_SHOT = 498;
const MASTER_EXT = /\.(wav|mp3|flac|m4a|ogg|aac)$/i;
const MAX_BYTES = 160 * 1024 * 1024;

export function masterAvailable(editor) {
    return ["r2v", "mixed"].includes(editor?.getTaskKey?.());
}

export function masterTiming(editor) {
    return masterAvailable(editor) && !!editor?.timeline?.[MASTER_KEY]?.audioFile;
}

export function masterDurationToFrames(seconds, fps) {
    const frames = Math.max(5, Math.min(MASTER_MAX_SHOT, Math.round(Number(seconds) * fps) || 5));
    return { frames, durationSec: frames / fps };
}

export function masterViewFrames(pictures, duration, fps) {
    return Math.max(pictures, Math.ceil(duration * fps - 1e-8), 1);
}

const DISPLAY_GAINS = [1, 2, 4, 8];

// When zoomed out keep a transient's peak but average RMS as energy, not amplitude.
function waveformLevel(info, start, end) {
    const { peaks, rms, samples, stride, scale = 255 } = info;
    let peak = 0, energy = 0, weight = 0;
    const first = Math.max(0, Math.floor(start / stride));
    const last = Math.min(peaks.length, Math.ceil(end / stride));
    for (let i = first; i < last; i++) {
        const overlap = Math.min(end, (i + 1) * stride, samples) - Math.max(start, i * stride);
        if (overlap <= 0) continue;
        peak = Math.max(peak, peaks[i]);
        if (rms) energy += rms[i] ** 2 * overlap;
        weight += overlap;
    }
    return { peak: peak / scale, rms: rms && weight ? Math.sqrt(energy / weight) / scale : null };
}

const dbfs = amplitude => amplitude > 0 ? `${(20 * Math.log10(amplitude)).toFixed(1)} dBFS` : "−∞ dBFS";

const tr = (zh, en) => getLocale() === "en" ? en : zh;
const time = (s) => {
    const cs = Math.round(s * 100);
    return `${Math.floor(cs / 6000)}:${((cs % 6000) / 100).toFixed(2).padStart(5, "0")}`;
};
const btn = (handler) => {
    const b = document.createElement("button");
    b.type = "button";
    b.addEventListener("click", handler);
    return b;
};

export class MasterAudioTrack {
    constructor(editor, { upload, afterEdit }) {
        this.ed = editor;
        this.upload = upload;
        this.afterEdit = afterEdit;
        this.path = "";
        this.followAudio = false;
        this.showBeats = true;
        this.media = new Audio();
        this.media.preload = "metadata";
        this.root = document.createElement("div");
        this.root.className = "bd-controls bd-master-controls";
        Object.assign(this.root.style, { padding: "6px 8px", gap: "8px", flexWrap: "wrap", alignItems: "center" });
        const file = this.file = document.createElement("input");
        file.type = "file";
        file.accept = ".wav,.mp3,.flac,.m4a,.ogg,.aac";
        file.hidden = true;
        file.onchange = () => {
            const picked = file.files[0];
            file.value = "";
            if (picked) void this.pick(picked);
        };
        this.uploadBtn = btn(() => file.click());
        this.playBtn = btn(() => void this.toggle());
        this.removeBtn = btn(() => {
            delete editor.timeline[MASTER_KEY];
            this.error = "";
            this.afterEdit();
        });
        this.fitBtn = btn(() => {
            const segments = editor.timeline.segments;
            const last = segments.at(-1);
            const frames = this.fitFrameCount();
            if (!last || frames < 5 || frames > MASTER_MAX_SHOT) return;
            last.durationSec = frames / editor.getFrameRate();
            this.afterEdit();
        });
        this.gainIndex = 0;
        this.gainBtn = btn(() => {
            this.gainIndex = (this.gainIndex + 1) % DISPLAY_GAINS.length;
            editor.scheduleRender();
        });
        this.beatBtn = btn(() => {
            this.showBeats = !this.showBeats;
            editor.scheduleRender();
        });
        this.status = document.createElement("span");
        this.status.setAttribute("aria-live", "polite");
        Object.assign(this.status.style, { color: "#9cb0b6", fontSize: "12px", overflowWrap: "anywhere" });
        this.root.append(this.uploadBtn, this.playBtn, this.fitBtn, this.removeBtn, this.gainBtn, this.beatBtn, this.status, file);
        editor.viewport.after(this.root);
        for (const name of ["mousedown", "pointerdown", "keydown", "wheel"]) {
            this.root.addEventListener(name, e => e.stopPropagation());
        }
        this.media.addEventListener("playing", () => this.startPlayhead());
        for (const event of ["pause", "ended", "seeked"]) {
            this.media.addEventListener(event, () => {
                if (this.destroyed || !this.followAudio) return;
                if (this.media.paused) this.stopPlayhead();
                this.syncPictureFrame();
                editor.scheduleRender();
            });
        }
    }

    async inspect(path, signal) {
        const response = await api.fetchApi(`/minimax/director/master_peaks?${new URLSearchParams({filename: path})}`, { signal });
        if (!response.ok) throw new Error(await response.text());
        return await response.json();
    }

    async pick(file) {
        if (!MASTER_EXT.test(file.name) || !file.size || file.size > MAX_BYTES) {
            this.error = tr("请选择 160 MiB 以内的 WAV / MP3 / FLAC / M4A / OGG / AAC。", "Choose a WAV / MP3 / FLAC / M4A / OGG / AAC file under 160 MiB.");
            this.ed.scheduleRender();
            return;
        }
        this.error = "";
        this.uploading = true;
        this.ed.scheduleRender();
        try {
            const identity = crypto.randomUUID().replaceAll("-", "");
            const ext = file.name.match(MASTER_EXT)[0].toLowerCase();
            const named = new File([file], `master_${identity}${ext}`, {type: file.type});
            const uploaded = await this.upload(named);
            const path = [uploaded.subfolder, uploaded.name || uploaded.filename].filter(Boolean).join("/");
            const info = await this.inspect(path);
            if (this.destroyed) return;
            this.pendingInfo = {path, info};
            // Keep old edits at their exact displayed frame count when enabling master timing.
            if (!masterTiming(this.ed) && masterAvailable(this.ed)) {
                for (const seg of this.ed.timeline.segments || []) {
                    const frames = Number(seg.frameCount ?? seg.length);
                    if (frames >= 5) seg.durationSec = frames / this.ed.getFrameRate();
                }
            }
            this.ed.timeline[MASTER_KEY] = {audioFile: path, label: file.name};
            this.afterEdit();
        } catch (err) {
            if (!this.destroyed) this.error = err.message;
        } finally {
            this.uploading = false;
            if (!this.destroyed) this.ed.scheduleRender();
        }
    }

    sync() {
        const active = masterTiming(this.ed);
        const path = active ? this.ed.timeline[MASTER_KEY].audioFile : "";
        if (path === this.path) return;
        this.abort?.abort();
        this.path = path;
        this.leaveAudioClock();
        this.info = null;
        this.columnCache = null;
        this.loadError = "";
        this.media.removeAttribute("src");
        this.media.load();
        if (!path) return;
        const i = path.lastIndexOf("/");
        this.media.src = api.apiURL(`/view?${new URLSearchParams({ filename: path.slice(i + 1), subfolder: i < 0 ? "" : path.slice(0, i), type: "input" })}`);
        if (this.pendingInfo?.path === path) {
            this.info = this.pendingInfo.info;
            this.pendingInfo = null;
            return;
        }
        const controller = this.abort = new AbortController();
        this.inspect(path, controller.signal).then(info => {
            if (controller.signal.aborted || this.destroyed) return;
            this.info = info;
            this.ed.scheduleRender();
        }).catch(err => {
            if (controller.signal.aborted || this.destroyed) return;
            this.loadError = err.message;
            this.ed.scheduleRender();
        });
    }

    fitFrameCount() {
        const last = this.ed.timeline.segments?.at(-1);
        if (!this.info || !last) return 0;
        return Math.ceil(this.info.duration * this.ed.getFrameRate() - 1e-8) - last.start;
    }

    update() {
        this.root.style.display = masterAvailable(this.ed) ? "flex" : "none";
        this.sync();
        if (this.ed.outAudioMode) {
            const mode = this.ed.outAudioMode;
            mode.disabled = !!this.path;
            if (this.path) mode.value = "master";
            else if (mode.value === "master") mode.value = this.ed.timeline.output?.audioMode || "generate";
            this.ed.outAudioMode.title = this.path ? tr("使用下方全局音轨输出原声", "The global track supplies output audio") : "";
        }
        const record = this.ed.timeline[MASTER_KEY];
        this.uploadBtn.textContent = this.uploading ? tr("读取中…", "Loading…") :
            (this.path ? tr("更换整轨", "Replace track") : tr("添加全局音轨", "Add master audio"));
        const externalLocked = !!(this.ed.hasExternalI2vGroups?.() || this.ed.hasExternalR2vGroups?.());
        this.uploadBtn.disabled = this.uploading || externalLocked;
        this.removeBtn.textContent = tr("移除", "Remove");
        this.removeBtn.hidden = !this.path;
        this.removeBtn.disabled = this.uploading;
        this.playBtn.hidden = !this.path;
        this.playBtn.disabled = !this.info;
        this.playBtn.textContent = this.media.paused ? tr("试听", "Listen") : tr("暂停", "Pause");
        this.gainBtn.hidden = !this.info;
        this.gainBtn.textContent = `${tr("波形显示", "Waveform view")} ×${DISPLAY_GAINS[this.gainIndex]}`;
        this.gainBtn.title = tr("点击切换 ×1 / ×2 / ×4 / ×8，只放大波形显示，不改变音量。", "Cycle ×1 / ×2 / ×4 / ×8. Display only; playback and export levels are unchanged.");
        this.beatBtn.hidden = !this.info;
        const count = this.info?.beats?.length || 0;
        this.beatBtn.disabled = count === 0;
        this.beatBtn.setAttribute("aria-pressed", String(this.showBeats && count > 0));
        this.beatBtn.textContent = count
            ? `${tr("卡点提示", "Cut markers")} ${this.showBeats ? `· ${count}` : tr("· 关", "· Off")}`
            : tr("未检测到明显卡点", "No clear onsets");
        this.beatBtn.title = tr("根据音乐的重音和起音自动标记候选切点，不会移动分镜。", "Suggests cuts from audio onsets; does not move any shots.");
        this.fitBtn.hidden = !this.path;
        this.fitBtn.textContent = tr("末镜贴齐曲尾", "Fit last shot to audio end");
        const frames = this.fitFrameCount();
        this.fitBtn.disabled = frames < 5 || frames > MASTER_MAX_SHOT || externalLocked;
        if (this.error || this.loadError) {
            this.status.textContent = this.error || this.loadError;
        } else if (externalLocked) {
            this.status.textContent = tr("全局音轨需要在导演台内编辑素材组，暂不支持外部素材组。", "Edit groups in the Director to use master audio; wired groups are not supported.");
        } else if (!this.path) {
            this.status.textContent = tr("先铺整首歌，再沿波形划分分镜。", "Lay in the full song, then line up shots to the waveform.");
        } else if (!this.info) {
            this.status.textContent = tr("读取波形…", "Reading waveform…");
        } else if (this.ed.getFrameRate() !== 24) {
            this.status.textContent = tr("全局音轨请用 24 fps。", "Master audio requires 24 fps.");
        } else {
            const fps = this.ed.getFrameRate();
            const video = this.ed.getTotalFrames() / fps;
            const delta = this.info.duration - video;
            const coverage = delta > 1/fps ? tr(`尚有 ${delta.toFixed(2)} 秒未铺画面（导出只取画面覆盖部分）`, `${delta.toFixed(2)}s has no picture yet (export covers picture only)`) :
                delta < -1/fps ? tr(`画面超出曲尾 ${(-delta).toFixed(2)} 秒（尾部静音）`, `Picture exceeds audio by ${(-delta).toFixed(2)}s (silence at end)`) : tr("曲尾已对齐", "Ends aligned");
            this.status.textContent = `${record.label || this.path} · ${time(this.info.duration)} · ${coverage} · ` +
                tr("原声输出；段间引导暂停", "Master audio output; continuity disabled");
        }
    }

    async toggle() {
        if (!this.media.paused) { this.media.pause(); return; }
        if (!this.info) return;
        if (this.ed.isPlaying) this.ed._stopPlay();
        if (!this.followAudio && !this.seek(this.ed.currentFrame)) this.seek(0);
        if (this.media.ended || this.media.currentTime >= this.info.duration) this.seek(0);
        try { await this.media.play(); }
        catch { this.error = tr("浏览器不能试听这个编码，可换用 WAV 或 MP3。", "This codec cannot be previewed in the browser; try WAV or MP3."); this.ed.scheduleRender(); }
    }

    playheadFrame(fps) {
        return this.followAudio ? Math.min(this.media.currentTime, this.info?.duration ?? Infinity) * fps : null;
    }

    syncPictureFrame() {
        // The red line can reach the whole song, including portions not covered by picture yet.
        const frame = this.playheadFrame(this.ed.getFrameRate());
        if (frame === null) return;
        // Media seeks may report a timestamp rounded down to microseconds.
        const pictureFrame = Math.min(Math.floor(frame + 1e-4), Math.max(0, this.ed.getTotalFrames() - 1));
        if (pictureFrame === this.ed.currentFrame) return;
        this.ed.currentFrame = pictureFrame;
        this.ed._updateTimelineDom();
    }

    startPlayhead() {
        if (this.destroyed || this.playheadRaf || !this.followAudio) return;
        let lastDraw = -Infinity;
        const tick = now => {
            this.playheadRaf = null;
            if (this.destroyed || !this.followAudio || this.media.paused) return;
            if (now - lastDraw >= 32) {
                lastDraw = now;
                this.syncPictureFrame();
                this.ed.renderTimelineOnly();
            }
            this.playheadRaf = requestAnimationFrame(tick);
        };
        this.playheadRaf = requestAnimationFrame(tick);
        this.ed.scheduleRender();
    }

    stopPlayhead() {
        if (this.playheadRaf) cancelAnimationFrame(this.playheadRaf);
        this.playheadRaf = null;
    }

    leaveAudioClock() {
        this.followAudio = false;
        this.stopPlayhead();
        this.media.pause();
    }

    seek(frame) {
        if (!this.info) return false;
        if (this.ed.isPlaying) this.ed._stopPlay();
        const fps = this.ed.getFrameRate();
        const seconds = frame / fps;
        if (seconds > this.info.duration && this.info.duration * fps < this.ed.getTotalFrames()) {
            this.leaveAudioClock();
            return false; // The picture has a silent tail; use its own playhead there.
        }
        this.followAudio = true;
        this.media.currentTime = Math.max(0, Math.min(this.info.duration, seconds));
        this.syncPictureFrame();
        this.ed.scheduleRender();
        return true;
    }

    describeTime(seconds) {
        const info = this.info;
        if (!info || seconds < 0 || seconds >= info.duration) return "";
        const span = info.samples / info.duration;
        const level = waveformLevel(info, (seconds - 0.05) * span, (seconds + 0.05) * span);
        const avg = level.rms == null ? "" : `${tr("平均", "RMS")} ${dbfs(level.rms)} · `;
        return `${time(seconds)} · ${avg}${tr("峰值", "Peak")} ${dbfs(level.peak)}`;
    }

    waveformColumns(audioWidth) {
        const width = Math.max(0, Math.ceil(audioWidth));
        const cache = this.columnCache;
        if (cache?.info === this.info && cache.audioWidth === audioWidth) return cache;
        const columns = { info: this.info, audioWidth, peak: new Float32Array(width), rms: new Float32Array(width) };
        const { samples } = this.info;
        for (let x = 0; x < width; x++) {
            const level = waveformLevel(this.info, x / audioWidth * samples, Math.min(1, (x + 1) / audioWidth) * samples);
            columns.peak[x] = level.peak;
            columns.rms[x] = level.rms ?? 0;
        }
        this.columnCache = columns;
        return columns;
    }

    paint(ctx, width, y, spanFrames, pictureFrames, segments, fps) {
        const h = MASTER_LANE_H;
        ctx.save();
        ctx.fillStyle = "#14242a";
        ctx.fillRect(0, y, width, h);
        ctx.strokeStyle = "#47646a";
        ctx.strokeRect(0.5, y + 0.5, width - 1, h - 1);
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.font = "11px sans-serif";
        ctx.fillStyle = "#c8ede9";
        ctx.fillText(tr("全局音轨", "MASTER AUDIO"), 7, y + 5);
        const mid = y + 56;
        if (this.info) {
            const { duration, rms } = this.info;
            const audioWidth = width * (duration * fps / spanFrames);
            const columns = this.waveformColumns(audioWidth);
            const gain = DISPLAY_GAINS[this.gainIndex];
            // Mild visual compression makes quiet detail legible even at ×1.
            const height = v => Math.min(1, v * gain) ** 0.68 * 34;
            const drawEnvelope = (data, color) => {
                ctx.strokeStyle = color;
                ctx.beginPath();
                for (let x = 0; x < data.length; x++) {
                    const a = height(data[x]);
                    if (a < 0.20) continue;
                    ctx.moveTo(x + 0.5, mid - a);
                    ctx.lineTo(x + 0.5, mid + a);
                }
                ctx.stroke();
            };
            ctx.fillStyle = "#496467";
            ctx.fillRect(0, mid, audioWidth, 1);
            drawEnvelope(columns.peak, "#4b9188");
            drawEnvelope(columns.rms, "#9aefd2");
            if (rms && width > 260) {
                ctx.fillStyle = "#4b9188";
                ctx.fillRect(width - 134, y + 7, 9, 7);
                ctx.fillStyle = "#b7d1cf";
                ctx.fillText(tr("峰值", "Peak"), width - 120, y + 5);
                ctx.fillStyle = "#9aefd2";
                ctx.fillRect(width - 74, y + 7, 9, 7);
                ctx.fillStyle = "#d1f4ea";
                ctx.fillText(tr("平均强度", "RMS"), width - 60, y + 5);
            }
            const voiced = width * pictureFrames / spanFrames;
            if (voiced < audioWidth) {
                ctx.fillStyle = "rgba(10,12,24,.38)";
                ctx.fillRect(voiced, y + 18, audioWidth - voiced, h - 19);
            }
            if (this.showBeats && this.info.beats?.length) {
                ctx.fillStyle = "#ffce72";
                ctx.beginPath();
                let previous = -Infinity;
                for (const second of this.info.beats) {
                    const x = second * fps / spanFrames * width;
                    if (x - previous < 6) continue;
                    previous = x;
                    ctx.moveTo(x - 2.5, y + 17);
                    ctx.lineTo(x + 2.5, y + 17);
                    ctx.lineTo(x, y + 23);
                    ctx.closePath();
                }
                ctx.fill();
            }
            const seconds = (this.playheadFrame(fps) ?? this.ed.currentFrame) / fps;
            const seekX = seconds * fps / spanFrames * width;
            ctx.fillStyle = "#ffaaaa";
            ctx.fillText(time(seconds), Math.max(75, Math.min(width - (rms && width > 260 ? 188 : 53), seekX + 4)), y + 5);
        } else {
            ctx.fillStyle = "#779499";
            ctx.fillText(this.path ? tr("等待波形", "Waiting for waveform") : tr("从下方添加完整歌曲", "Add the complete song below"), 7, y + 30);
        }
        if (this.path) {
            ctx.strokeStyle = "rgba(255,255,255,.55)";
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            for (const seg of segments) {
                const x = seg.start * width / spanFrames;
                if (x <= 0 || x >= width) continue;
                ctx.moveTo(x, y + 18);
                ctx.lineTo(x, y + h);
            }
            ctx.stroke();
        }
        ctx.restore();
    }

    destroy() {
        this.destroyed = true;
        this.abort?.abort();
        this.leaveAudioClock();
        this.media.removeAttribute("src");
        this.media.load();
    }
}
