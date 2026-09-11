/** Re-run loop for the MiniMax H3 Director.
 *
 * With "Ask to re-run when done" on, a finished run opens a small panel:
 * re-run now with a new seed, or stop. With "Auto re-run after (s)" above 0
 * the panel counts down and re-runs by itself, so a long batch keeps making
 * new takes while nobody is at the machine, until Stop is pressed or the
 * toggle is turned off.
 */

import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { t } from "./minimax_i18n.js";

const SEED_MAX = 2 ** 48;
const _finished = new Set();
let _take = 1;
let _closePanel = null;

function tf(key, vars = {}) {
    let text = t(key);
    for (const [name, value] of Object.entries(vars)) text = text.split(`{${name}}`).join(String(value));
    return text;
}

function el(tag, cls = "", text = null) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
}

function widget(node, name) {
    return node?.widgets?.find((w) => w.name === name) || null;
}

function directorNode(id) {
    const graph = app.graph;
    if (!graph?.getNodeById) return null;
    const numeric = Number(id);
    return graph.getNodeById(Number.isFinite(numeric) ? numeric : id) || null;
}

function toast(text, tone = "ok") {
    ensureRerunStyles();
    const node = el("div", `mmx-rr-toast is-${tone}`, text);
    document.body.appendChild(node);
    setTimeout(() => node.classList.add("out"), 2600);
    setTimeout(() => node.remove(), 3000);
}

export function closeRerunPanel() {
    _closePanel?.();
}

async function requeue(nodes) {
    for (const node of nodes) {
        const seed = widget(node, "seed");
        if (!seed) continue;
        seed.value = Math.floor(Math.random() * SEED_MAX);
        seed.callback?.(seed.value);
    }
    app.graph?.setDirtyCanvas?.(true, true);
    _take += 1;
    try {
        await app.queuePrompt(0, 1);
        toast(tf("rr.rerunning", { n: _take }));
    } catch (err) {
        _take = Math.max(1, _take - 1);
        toast(`${t("rr.error")}: ${err?.message || err}`, "bad");
    }
}

function openRerunPanel(nodes) {
    closeRerunPanel();
    ensureRerunStyles();
    const seconds = Math.max(0, Math.round(Number(widget(nodes[0], "rerun_after_seconds")?.value) || 0));
    const panel = el("div", "mmx-rr-panel");
    const title = el("div", "mmx-rr-title");
    title.append(el("span", "mmx-rr-dot"), el("span", "", t("rr.finished")), el("span", "mmx-rr-take", tf("rr.take", { n: _take })));
    const sub = el("div", "mmx-rr-sub", seconds ? tf("rr.autoIn", { n: seconds }) : t("rr.ask"));
    const bar = el("div", "mmx-rr-bar");
    const fill = el("div", "mmx-rr-fill");
    bar.appendChild(fill);
    bar.hidden = !seconds;
    const actions = el("div", "mmx-rr-actions");
    const go = el("button", "mmx-rr-go", `▶ ${t("rr.rerunNow")}`);
    const stop = el("button", "mmx-rr-stop", `■ ${t("rr.stop")}`);
    go.type = "button";
    stop.type = "button";
    actions.append(go, stop);
    panel.append(title, sub, bar, actions);
    for (const type of ["pointerdown", "mousedown", "click", "wheel", "keydown", "keyup"]) {
        panel.addEventListener(type, (e) => e.stopPropagation());
    }

    let timer = 0;
    const close = () => {
        if (_closePanel !== close) return;
        _closePanel = null;
        clearInterval(timer);
        panel.classList.add("out");
        setTimeout(() => panel.remove(), 220);
    };
    const rerun = () => {
        close();
        requeue(nodes);
    };
    go.onclick = rerun;
    stop.onclick = () => {
        close();
        _take = 1;
        toast(t("rr.stopped"), "muted");
    };
    if (seconds) {
        const started = Date.now();
        const deadline = started + seconds * 1000;
        const tick = () => {
            const now = Date.now();
            sub.textContent = tf("rr.autoIn", { n: Math.max(0, Math.ceil((deadline - now) / 1000)) });
            fill.style.width = `${Math.min(100, (100 * (now - started)) / (seconds * 1000))}%`;
            if (now >= deadline) rerun();
        };
        timer = setInterval(tick, 250);
        tick();
    }
    document.body.appendChild(panel);
    _closePanel = close;
}

/** Hook the Director's finish event and ComfyUI's prompt events once. */
export function installDirectorRerun() {
    if (installDirectorRerun.done) return;
    installDirectorRerun.done = true;
    api.addEventListener("execution_start", () => {
        _finished.clear();
        closeRerunPanel();
    });
    api.addEventListener("minimax_director_progress", ({ detail }) => {
        if (detail?.phase === "finish" && detail?.node_id != null) _finished.add(String(detail.node_id));
    });
    api.addEventListener("execution_success", () => {
        const nodes = [..._finished]
            .map(directorNode)
            .filter((node) => node && widget(node, "rerun_when_done")?.value);
        _finished.clear();
        if (nodes.length) openRerunPanel(nodes);
    });
    for (const type of ["execution_error", "execution_interrupted"]) {
        api.addEventListener(type, () => {
            _finished.clear();
            _take = 1;
        });
    }
}

export function ensureRerunStyles() {
    if (document.getElementById("mmx-rerun-css")) return;
    const style = document.createElement("style");
    style.id = "mmx-rerun-css";
    style.textContent = `
.mmx-rr-panel { position:fixed; right:18px; bottom:18px; z-index:10030; width:330px; box-sizing:border-box; padding:14px;
    border-radius:14px; background:linear-gradient(180deg, #161616, #0d0d0d); border:1px solid #2c4a37; color:#eee;
    font:12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
    box-shadow:0 18px 44px rgba(0,0,0,.6), 0 0 0 1px rgba(79,255,143,.08);
    animation:mmx-rr-in .24s cubic-bezier(.2,.9,.3,1.15); transition:opacity .2s, transform .2s; }
.mmx-rr-panel.out { opacity:0; transform:translateY(10px); }
@keyframes mmx-rr-in { from { opacity:0; transform:translateY(14px) scale(.98); } }
.mmx-rr-title { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:800; }
.mmx-rr-dot { width:8px; height:8px; border-radius:50%; background:#4fff8f; box-shadow:0 0 10px #4fff8f;
    animation:mmx-rr-pulse 1.4s ease-in-out infinite; }
@keyframes mmx-rr-pulse { 50% { opacity:.35; } }
.mmx-rr-take { margin-left:auto; font-size:10px; font-weight:700; color:#8fcfaa; border:1px solid #2c4a37; border-radius:999px; padding:1px 7px; }
.mmx-rr-sub { margin:7px 0 10px; color:#9fdcb8; font-size:11.5px; font-variant-numeric:tabular-nums; }
.mmx-rr-bar { height:4px; margin-bottom:12px; border-radius:4px; background:#1d2a22; overflow:hidden; }
.mmx-rr-bar[hidden] { display:none; }
.mmx-rr-fill { height:100%; width:0; background:linear-gradient(90deg, #4fff8f, #b6ffd3); transition:width .25s linear; }
.mmx-rr-actions { display:flex; gap:8px; }
.mmx-rr-go { flex:1 1 auto; background:#4fff8f; color:#062b14; border:none; border-radius:9px; padding:9px 10px;
    font-weight:800; font-size:12px; cursor:pointer; font-family:inherit; }
.mmx-rr-go:hover { filter:brightness(1.08); }
.mmx-rr-stop { background:transparent; color:#e06c6c; border:1px solid #5a2a2a; border-radius:9px; padding:9px 14px;
    font-weight:700; font-size:12px; cursor:pointer; font-family:inherit; }
.mmx-rr-stop:hover { background:rgba(224,108,108,.1); }
.mmx-rr-toast { position:fixed; right:18px; bottom:18px; z-index:10030; padding:8px 12px; border-radius:9px;
    font:800 11.5px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif; box-shadow:0 8px 20px rgba(0,0,0,.45);
    transition:opacity .35s, transform .35s; }
.mmx-rr-toast.is-ok { background:#4fff8f; color:#062b14; }
.mmx-rr-toast.is-muted { background:#262626; color:#ddd; }
.mmx-rr-toast.is-bad { background:#5a1f1f; color:#ffd6d6; }
.mmx-rr-toast.out { opacity:0; transform:translateY(8px); }`;
    document.head.appendChild(style);
}

installDirectorRerun();
