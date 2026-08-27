/** Director prompt @-mentions + official-tag token chips (<Picture N> / <Video K> / <Audio J>).
 *
 * Canonical storage is always the official MiniMax tag string (textarea.value).
 * The visible surface is a contenteditable that renders atomic bd-token chips.
 */

import { api } from "../../scripts/api.js";
import {
    refAudioLabel,
    refAudioPromptTag,
    refImageLabel,
    refImagePromptTag,
    refVideoLabel,
    refVideoPromptTag,
    resolveTaskKey,
} from "./minimax_gen_timeline.js";
import { t } from "./minimax_i18n.js";

const TAG_RE = /<(Picture|Video|Audio)\s+(\d+)\s*>/gi;
const TOKEN_CLASS = "bd-token";

const MENTION_STYLES = `
.bd-mention-menu{position:fixed;z-index:10050;min-width:210px;max-width:300px;max-height:240px;overflow:auto;background:#252525;border:1px solid #444;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.45);padding:4px 0}
.bd-mention-menu.hidden{display:none!important}
.bd-mention-title{padding:6px 10px 4px;font-size:10px;color:#888;user-select:none}
.bd-mention-item{display:flex;align-items:center;gap:8px;padding:6px 10px;cursor:pointer;font-size:11px;color:#ddd}
.bd-mention-item:hover,.bd-mention-item.active{background:#333;color:#fff}
.bd-mention-item img,.bd-mention-thumb{
  width:36px;height:36px;object-fit:cover;border-radius:4px;flex-shrink:0;background:#111;border:1px solid #333;box-sizing:border-box
}
.bd-mention-thumb{
  display:inline-flex;align-items:center;justify-content:center;font-size:16px;line-height:1;color:#9ad
}
.bd-mention-thumb.bd-mention-thumb-video{color:#7db7ff;background:#152030;border-color:#2a4a6a}
.bd-mention-thumb.bd-mention-thumb-audio{color:#e0b06a;background:#2a2010;border-color:#5a4530}
.bd-mention-item .bd-mention-label{font-weight:600;color:#4fff8f}
.bd-mention-empty{padding:10px 12px;font-size:11px;color:#888;text-align:center;line-height:1.4}

.bd-token-wrap{position:relative;display:flex;flex-direction:column;min-width:0;min-height:0;flex:1 1 auto;width:100%;box-sizing:border-box}
.bd-token-wrap>.bd-token-source{
  position:absolute!important;width:1px!important;height:1px!important;padding:0!important;margin:-1px!important;
  overflow:hidden!important;clip:rect(0,0,0,0)!important;border:0!important;opacity:0!important;pointer-events:none!important;
  resize:none!important;min-height:0!important;flex:none!important
}
.bd-token-editor{
  width:100%;min-height:96px;flex:1 1 auto;box-sizing:border-box;
  background:#181818;border:1px solid #333;border-radius:6px;color:#eee;padding:8px;
  font-size:12px;font-family:inherit;line-height:1.45;outline:none;overflow:auto;white-space:pre-wrap;word-break:break-word;
  resize:vertical
}
.bd-token-editor:focus{border-color:#4a7a5a;box-shadow:0 0 0 1px rgba(79,255,143,.18)}
.bd-token-editor:empty:before{content:attr(data-placeholder);color:#666;pointer-events:none}
.bd-token-resize-handle{
  position:absolute;left:50%;bottom:0;z-index:4;width:96px;max-width:40%;height:18px;
  transform:translateX(-50%);display:flex;align-items:flex-end;justify-content:center;
  padding:0;border:0;background:transparent;outline:none;border-radius:8px 8px 0 0;
  cursor:ns-resize;touch-action:none;user-select:none
}
.bd-token-resize-handle::after{
  content:"";width:54px;max-width:72%;height:3px;margin-bottom:3px;border-radius:999px;
  background:#52665a;box-shadow:0 -4px 0 rgba(82,102,90,.65)
}
.bd-token-resize-handle:hover::after,
.bd-token-resize-handle:focus-visible::after,
.bd-token-wrap.bd-token-resizing .bd-token-resize-handle::after{
  background:#4fff8f;box-shadow:0 -4px 0 rgba(79,255,143,.45)
}
body.bd-token-resizing{cursor:ns-resize!important;user-select:none!important}
.bd-rv2v-layout .bd-token-editor,.bd-v2v-layout .bd-token-editor{
  min-height:220px;background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45
}
.bd-v2v-layout .bd-token-editor{min-height:180px}
.bd-batch-prompts .bd-token-editor{
  min-height:88px;background:#181818;border:1px solid #333;border-radius:4px;padding:6px;font-size:11px;line-height:1.35
}
.bd-batch-plain .bd-batch-prompts .bd-token-editor,.bd-batch-source .bd-batch-prompts .bd-token-editor{
  min-height:120px;background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45
}
.bd-batch-r2v .bd-batch-prompts .bd-token-editor{
  min-height:360px;height:100%;background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45
}

.bd-token{
  display:inline-flex;align-items:center;gap:4px;max-width:100%;
  margin:0 2px;padding:1px 7px 1px 3px;border-radius:999px;vertical-align:baseline;
  border:1.5px solid #3dcc7a;background:rgba(61,204,122,.12);color:#c8ffd9;
  font-size:11px;font-weight:600;line-height:1.4;user-select:none;cursor:default;white-space:nowrap
}
.bd-token[contenteditable="false"]{-webkit-user-modify:read-only}
.bd-token.bd-token-image{
  border-color:#3dcc7a;background:rgba(61,204,122,.14);color:#c8ffd9;
  box-shadow:0 0 0 1px rgba(61,204,122,.22)
}
.bd-token.bd-token-video{
  border-color:#4d9fff;background:rgba(77,159,255,.16);color:#d4e9ff;
  box-shadow:0 0 0 1px rgba(77,159,255,.28)
}
.bd-token.bd-token-audio{
  border-color:#e8a23a;background:rgba(232,162,58,.16);color:#ffe6bf;
  box-shadow:0 0 0 1px rgba(232,162,58,.28)
}
.bd-token.is-missing{opacity:.62;border-style:dashed}
.bd-token.is-missing .bd-token-label{text-decoration:line-through;text-decoration-color:rgba(255,255,255,.35)}
.bd-token-thumb{
  width:16px;height:16px;border-radius:3px;object-fit:cover;flex-shrink:0;background:#111;border:1px solid rgba(0,0,0,.35)
}
.bd-token-glyph{
  width:16px;height:16px;border-radius:3px;flex-shrink:0;display:inline-flex;align-items:center;justify-content:center;
  font-size:10px;line-height:1;background:rgba(0,0,0,.35);color:inherit
}
.bd-token-image .bd-token-glyph{background:rgba(61,204,122,.28);color:#8dffb8}
.bd-token-video .bd-token-glyph{background:rgba(77,159,255,.32);color:#9cc8ff}
.bd-token-audio .bd-token-glyph{background:rgba(232,162,58,.32);color:#ffd48a}
.bd-token-label{max-width:7em;overflow:hidden;text-overflow:ellipsis}
`;

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;
    const el = document.createElement("style");
    el.textContent = MENTION_STYLES;
    document.head.appendChild(el);
}

function inputViewUrl(filename, type = "input") {
    const subfolder = filename.includes("/") ? filename.slice(0, filename.lastIndexOf("/")) : "";
    const base = subfolder ? filename.slice(subfolder.length + 1) : filename;
    const params = new URLSearchParams({ filename: base, type });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

function refThumbUrl(ref) {
    if (ref?.imageFile) return inputViewUrl(ref.imageFile, "input");
    if (ref?.imageB64) {
        return ref.imageB64.startsWith("data:") ? ref.imageB64 : `data:image/png;base64,${ref.imageB64}`;
    }
    return "";
}

function videoThumbUrl(ref) {
    if (ref?.previewImageUrl) return String(ref.previewImageUrl);
    if (ref?.previewImageFile) return inputViewUrl(ref.previewImageFile, "input");
    return "";
}

/** Square leading preview for the @ menu: real thumb, else kind icon. */
function makeMentionMenuThumb(item) {
    if (item?.thumb) {
        const img = document.createElement("img");
        img.src = item.thumb;
        img.alt = item.label || "";
        return img;
    }
    const thumb = document.createElement("span");
    const kind = item?.kind || "image";
    thumb.className = `bd-mention-thumb bd-mention-thumb-${kind}`;
    thumb.setAttribute("aria-hidden", "true");
    // Simple geometric icons — match image thumb size, no external assets.
    if (kind === "video") thumb.textContent = "▶";
    else if (kind === "audio") thumb.textContent = "♪";
    else thumb.textContent = "▣";
    return thumb;
}

function listAvailableMentions(refs, audios, videos) {
    const items = [];
    for (const r of [...(refs || [])]
        .filter((x) => x?.imageFile || x?.imageB64)
        .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0))) {
        const index = Number(r.index ?? r.slot ?? 0);
        items.push({
            index,
            kind: "image",
            label: refImageLabel(index),
            tag: refImagePromptTag(index),
            thumb: refThumbUrl(r),
        });
    }
    for (const v of [...(videos || [])]
        .filter((x) => x?.videoFile || x?.fileName || x?.previewImageFile || x?.previewImageUrl)
        .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0))) {
        const index = Number(v.index ?? v.slot ?? 0);
        items.push({
            index,
            kind: "video",
            label: v.mentionLabel || v.displayName || refVideoLabel(index),
            tag: refVideoPromptTag(index),
            thumb: videoThumbUrl(v),
        });
    }
    for (const a of [...(audios || [])]
        .filter((x) => x?.audioFile || x?.fileName)
        .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0))) {
        const index = Number(a.index ?? a.slot ?? 0);
        items.push({
            index,
            kind: "audio",
            label: refAudioLabel(index),
            tag: refAudioPromptTag(index),
            thumb: "",
        });
    }
    return items;
}

function sourceVideoMention(editor, seg = null) {
    const taskKey = resolveTaskKey(
        seg?.taskType
        || seg?.task_type
        || editor?.timeline?.global?.taskType
        || editor?.getTaskKey?.()
        || "",
    );
    if (taskKey !== "v2v" && taskKey !== "rv2v") return [];

    const clips = Array.isArray(editor?.timeline?.videoClips)
        ? editor.timeline.videoClips
        : [];
    const clipId = seg?.videoClipId || seg?.video_clip_id;
    const source = (
        (clipId ? clips.find((clip) => clip?.id === clipId) : null)
        || clips[0]
        || editor?.timeline?.video
        || {}
    );
    const videoFile = source.videoFile || source.fileName || "";
    if (!videoFile) return [];
    const fileName = source.fileName || String(videoFile).split(/[\\/]/).pop() || videoFile;
    return [{
        ...source,
        index: 0,
        videoFile,
        fileName,
        mentionLabel: `${refVideoLabel(0)} · ${fileName}`,
        isTimelineSource: true,
    }];
}

function promptVideosFor(editor, seg, extraVideos) {
    const source = sourceVideoMention(editor, seg);
    if (!source.length) return extraVideos || [];
    // v2v/rv2v reserves <Video 1> for the segment's timeline source.
    return [
        ...source,
        ...(extraVideos || []).filter((video) => Number(video?.index ?? video?.slot ?? 0) !== 0),
    ];
}

function kindFromTagType(type) {
    const k = String(type || "").toLowerCase();
    if (k === "picture") return "image";
    if (k === "video") return "video";
    if (k === "audio") return "audio";
    return "image";
}

function tagFor(kind, ordinal1) {
    const n = Math.max(1, Number(ordinal1) || 1);
    if (kind === "video") return refVideoPromptTag(n - 1);
    if (kind === "audio") return refAudioPromptTag(n - 1);
    return refImagePromptTag(n - 1);
}

function labelFor(kind, ordinal1) {
    const n = Math.max(1, Number(ordinal1) || 1);
    if (kind === "video") return refVideoLabel(n - 1);
    if (kind === "audio") return refAudioLabel(n - 1);
    return refImageLabel(n - 1);
}

function findMentionItem(items, kind, ordinal1) {
    const index0 = Math.max(1, Number(ordinal1) || 1) - 1;
    return (items || []).find((it) => it.kind === kind && Number(it.index) === index0) || null;
}

function appendTextWithBreaks(parent, text) {
    const parts = String(text || "").split("\n");
    parts.forEach((part, i) => {
        if (part) parent.appendChild(document.createTextNode(part));
        if (i < parts.length - 1) parent.appendChild(document.createElement("br"));
    });
}

function makeTokenChip(kind, ordinal1, mediaItem, { onActivate } = {}) {
    const tag = tagFor(kind, ordinal1);
    const chip = document.createElement("span");
    chip.className = `${TOKEN_CLASS} bd-token-${kind}`;
    chip.contentEditable = "false";
    chip.dataset.tag = tag;
    chip.dataset.kind = kind;
    chip.dataset.ordinal = String(Math.max(1, Number(ordinal1) || 1));
    chip.title = tag;

    const missing = !mediaItem;
    chip.classList.toggle("is-missing", missing);

    if (kind === "image" && mediaItem?.thumb) {
        const img = document.createElement("img");
        img.className = "bd-token-thumb";
        img.src = mediaItem.thumb;
        img.alt = "";
        chip.appendChild(img);
    } else {
        const glyph = document.createElement("span");
        glyph.className = "bd-token-glyph";
        glyph.textContent = kind === "video" ? "▶" : kind === "audio" ? "♪" : "▣";
        chip.appendChild(glyph);
    }

    const label = document.createElement("span");
    label.className = "bd-token-label";
    label.textContent = mediaItem?.label || labelFor(kind, ordinal1);
    chip.appendChild(label);

    chip.addEventListener("pointerdown", (event) => {
        // Place caret before/after chip; keep chip atomic.
        event.preventDefault();
        const editor = chip.closest(".bd-token-editor");
        if (!editor) return;
        const range = document.createRange();
        const rect = chip.getBoundingClientRect();
        const before = event.clientX < rect.left + rect.width / 2;
        if (before) range.setStartBefore(chip);
        else range.setStartAfter(chip);
        range.collapse(true);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        editor.focus();
        if (typeof onActivate === "function") {
            onActivate({ kind, ordinal: Number(chip.dataset.ordinal) || 1, tag });
        }
    });

    return chip;
}

/** Serialize editor DOM → official prompt string. */
export function serializeTokenEditor(editor) {
    if (!editor) return "";
    let out = "";
    const visit = (node) => {
        if (!node) return;
        if (node.nodeType === Node.TEXT_NODE) {
            out += node.textContent || "";
            return;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        if (node.classList?.contains(TOKEN_CLASS)) {
            out += node.dataset.tag || "";
            return;
        }
        if (node.tagName === "BR") {
            out += "\n";
            return;
        }
        const block = ["DIV", "P"].includes(node.tagName);
        if (block && out && !out.endsWith("\n")) out += "\n";
        for (const child of node.childNodes || []) visit(child);
    };
    for (const child of editor.childNodes || []) visit(child);
    return out;
}

/** Build editor DOM from official prompt string. */
export function hydrateTokenEditor(editor, text, getMedia, options = {}) {
    if (!editor) return;
    const media = typeof getMedia === "function" ? getMedia() : {};
    const items = listAvailableMentions(media.refs, media.audios, media.videos);
    const source = String(text ?? "");
    editor.textContent = "";
    let cursor = 0;
    TAG_RE.lastIndex = 0;
    let match;
    while ((match = TAG_RE.exec(source))) {
        if (match.index > cursor) {
            appendTextWithBreaks(editor, source.slice(cursor, match.index));
        }
        const kind = kindFromTagType(match[1]);
        const ordinal1 = Number(match[2]) || 1;
        const item = findMentionItem(items, kind, ordinal1);
        editor.appendChild(makeTokenChip(kind, ordinal1, item, options));
        cursor = match.index + match[0].length;
    }
    if (cursor < source.length) appendTextWithBreaks(editor, source.slice(cursor));
    // Leave truly empty so CSS :empty placeholder works; caret still works on empty contenteditable.
}

function refreshTokenStates(editor, getMedia) {
    if (!editor) return;
    const media = typeof getMedia === "function" ? getMedia() : {};
    const items = listAvailableMentions(media.refs, media.audios, media.videos);
    for (const chip of editor.querySelectorAll(`.${TOKEN_CLASS}`)) {
        const kind = chip.dataset.kind || "image";
        const ordinal1 = Number(chip.dataset.ordinal) || 1;
        const item = findMentionItem(items, kind, ordinal1);
        chip.classList.toggle("is-missing", !item);
        chip.title = chip.dataset.tag || "";
        const label = chip.querySelector(".bd-token-label");
        if (label) label.textContent = item?.label || labelFor(kind, ordinal1);
        const thumb = chip.querySelector(".bd-token-thumb");
        if (kind === "image") {
            if (item?.thumb) {
                if (thumb) {
                    if (thumb.src !== item.thumb) thumb.src = item.thumb;
                } else {
                    const glyph = chip.querySelector(".bd-token-glyph");
                    const img = document.createElement("img");
                    img.className = "bd-token-thumb";
                    img.src = item.thumb;
                    img.alt = "";
                    if (glyph) glyph.replaceWith(img);
                    else chip.insertBefore(img, label);
                }
            }
        }
    }
}

/** Serialized prompt offset for a DOM boundary inside the token editor. */
function serializedBoundaryOffset(editor, container, offset) {
    if (!editor || !container || !editor.contains(container)) {
        return serializeTokenEditor(editor).length;
    }
    const pre = document.createRange();
    pre.selectNodeContents(editor);
    try {
        pre.setEnd(container, offset);
    } catch {
        return serializeTokenEditor(editor).length;
    }
    const walkerRoot = document.createElement("div");
    walkerRoot.appendChild(pre.cloneContents());
    return serializeTokenEditor(walkerRoot).length;
}

/** Selection [start, end) in serialized prompt space (end === start when collapsed). */
function serializedSelectionOffsets(editor) {
    const sel = window.getSelection();
    const len = serializeTokenEditor(editor).length;
    if (!sel || !sel.rangeCount || !editor.contains(sel.anchorNode)) {
        return { start: len, end: len };
    }
    const range = sel.getRangeAt(0);
    let start = serializedBoundaryOffset(editor, range.startContainer, range.startOffset);
    let end = sel.isCollapsed
        ? start
        : serializedBoundaryOffset(editor, range.endContainer, range.endOffset);
    if (end < start) [start, end] = [end, start];
    return { start, end };
}

function serializedCaretOffset(editor) {
    return serializedSelectionOffsets(editor).start;
}

function setCaretBySerializedOffset(editor, offset) {
    const target = Math.max(0, Number(offset) || 0);
    let seen = 0;
    const sel = window.getSelection();
    const range = document.createRange();

    const place = (node, at) => {
        range.setStart(node, at);
        range.collapse(true);
        sel.removeAllRanges();
        sel.addRange(range);
    };

    const walk = (node) => {
        if (seen >= target) return true;
        if (node.nodeType === Node.TEXT_NODE) {
            const len = (node.textContent || "").length;
            if (seen + len >= target) {
                place(node, target - seen);
                seen = target;
                return true;
            }
            seen += len;
            return false;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return false;
        if (node.classList?.contains(TOKEN_CLASS)) {
            const tag = node.dataset.tag || "";
            if (seen + tag.length >= target) {
                // Caret after chip if offset lands inside tag span.
                range.setStartAfter(node);
                range.collapse(true);
                sel.removeAllRanges();
                sel.addRange(range);
                seen = target;
                return true;
            }
            seen += tag.length;
            return false;
        }
        if (node.tagName === "BR") {
            if (seen + 1 >= target) {
                range.setStartAfter(node);
                range.collapse(true);
                sel.removeAllRanges();
                sel.addRange(range);
                seen = target;
                return true;
            }
            seen += 1;
            return false;
        }
        for (const child of node.childNodes || []) {
            if (walk(child)) return true;
        }
        return false;
    };

    for (const child of editor.childNodes || []) {
        if (walk(child)) return;
    }
    // End of editor
    range.selectNodeContents(editor);
    range.collapse(false);
    sel.removeAllRanges();
    sel.addRange(range);
}

function textBeforeCaret(editor) {
    const { start: offset } = serializedSelectionOffsets(editor);
    const full = serializeTokenEditor(editor);
    return { full, offset, before: full.slice(0, offset), after: full.slice(offset) };
}

function insertAtCaret(editor, insertText, getMedia, options, { replaceFrom = null } = {}) {
    const { start: selStart, end: selEnd } = serializedSelectionOffsets(editor);
    const full = serializeTokenEditor(editor);
    // Replace the active selection (or @-query range). Previously paste only
    // inserted at selStart and kept the selected text → duplicate prompts (#8).
    const start = replaceFrom != null ? replaceFrom : selStart;
    const next = full.slice(0, start) + insertText + full.slice(selEnd);
    const caret = start + insertText.length;
    hydrateTokenEditor(editor, next, getMedia, options);
    setCaretBySerializedOffset(editor, caret);
    return next;
}

function editorHasRawTagsInTextNodes(editor) {
    for (const node of editor.childNodes || []) {
        if (node.nodeType === Node.TEXT_NODE && TAG_RE.test(node.textContent || "")) {
            TAG_RE.lastIndex = 0;
            return true;
        }
        if (node.nodeType === Node.ELEMENT_NODE && !node.classList?.contains(TOKEN_CLASS)) {
            if (node.tagName !== "BR") {
                for (const child of node.childNodes || []) {
                    if (child.nodeType === Node.TEXT_NODE && TAG_RE.test(child.textContent || "")) {
                        TAG_RE.lastIndex = 0;
                        return true;
                    }
                }
            }
        }
    }
    TAG_RE.lastIndex = 0;
    return false;
}

/** Viewport rect of the current caret (contenteditable), or null. */
function getCaretClientRect(editor) {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount || !editor?.contains?.(sel.focusNode)) return null;
    const range = sel.getRangeAt(0).cloneRange();
    range.collapse(true);
    const rects = range.getClientRects();
    if (rects?.length) return rects[0];
    const br = range.getBoundingClientRect();
    if (br && (br.top || br.left || br.bottom || br.right)) return br;

    // Collapsed caret at empty edges often has a zero rect — probe with a marker.
    const caretOffset = serializedCaretOffset(editor);
    const marker = document.createElement("span");
    marker.textContent = "\u200b";
    range.insertNode(marker);
    const probed = marker.getBoundingClientRect();
    marker.remove();
    editor.normalize?.();
    setCaretBySerializedOffset(editor, caretOffset);
    if (probed && (probed.top || probed.left || probed.bottom || probed.right)) return probed;
    return null;
}

/** Place the @-menu next to the caret (not under the whole textbox). */
function positionMenu(menu, editor) {
    const editorRect = editor.getBoundingClientRect();
    const caret = getCaretClientRect(editor);
    const pad = 8;
    const gap = 4;
    menu.style.maxWidth = `${Math.max(210, Math.min(320, editorRect.width))}px`;
    // Must be measurable for flip-above logic.
    menu.classList.remove("hidden");
    const menuH = menu.offsetHeight || 180;
    const menuW = menu.offsetWidth || 210;

    let left = caret ? caret.left : editorRect.left + pad;
    let top = caret ? caret.bottom + gap : editorRect.top + pad;

    // Flip above caret when not enough room below.
    if (top + menuH > window.innerHeight - pad) {
        const above = (caret ? caret.top : editorRect.top) - menuH - gap;
        if (above >= pad) top = above;
        else top = Math.max(pad, window.innerHeight - pad - menuH);
    }
    // Keep horizontally near caret, clamped to the viewport.
    left = Math.min(Math.max(pad, left), window.innerWidth - pad - menuW);
    menu.style.left = `${Math.round(left)}px`;
    menu.style.top = `${Math.round(top)}px`;
}

/**
 * Native CSS resize handles are unreliable inside ComfyUI's transformed canvas.
 * This explicit grip converts viewport pointer movement back into the editor's
 * unscaled CSS height, so dragging remains accurate at every canvas zoom level.
 */
function installTokenResizeHandle(wrap, editor) {
    if (!wrap || !editor || wrap.querySelector(":scope > .bd-token-resize-handle")) return;

    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "bd-token-resize-handle";
    handle.dataset.role = "prompt-resize-handle";
    handle.setAttribute("aria-label", t("tooltip.promptEditorResize"));
    handle.dataset.i18nTitle = "tooltip.promptEditorResize";
    handle.title = t("tooltip.promptEditorResize");
    wrap.appendChild(handle);

    const applyHeight = (height, minHeight = 96) => {
        const nextHeight = Math.max(minHeight, Math.round(height));
        editor.style.height = `${nextHeight}px`;
        wrap.style.height = `${nextHeight}px`;
    };

    handle.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
        event.preventDefault();
        event.stopPropagation();
        const computed = getComputedStyle(editor);
        const current = Number.parseFloat(computed.height) || editor.offsetHeight || 96;
        const minHeight = Number.parseFloat(computed.minHeight) || 96;
        const step = event.shiftKey ? 80 : 24;
        applyHeight(current + (event.key === "ArrowDown" ? step : -step), minHeight);
    });

    handle.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
    });

    handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();

        const computed = getComputedStyle(editor);
        const rect = editor.getBoundingClientRect();
        const startHeight = Number.parseFloat(computed.height) || editor.offsetHeight || 96;
        const minHeight = Number.parseFloat(computed.minHeight) || 96;
        const scaleY = rect.height > 0 ? rect.height / startHeight : 1;
        const startY = event.clientY;
        const pointerId = event.pointerId;

        wrap.classList.add("bd-token-resizing");
        document.body.classList.add("bd-token-resizing");
        try {
            handle.setPointerCapture(pointerId);
        } catch {
            /* Window listeners below keep dragging active without pointer capture. */
        }

        const onMove = (moveEvent) => {
            if (moveEvent.pointerId !== pointerId) return;
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            applyHeight(
                startHeight + (moveEvent.clientY - startY) / Math.max(scaleY, 0.01),
                minHeight
            );
        };

        const stop = (endEvent) => {
            if (endEvent.pointerId !== pointerId) return;
            endEvent.preventDefault();
            endEvent.stopPropagation();
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", stop);
            window.removeEventListener("pointercancel", stop);
            wrap.classList.remove("bd-token-resizing");
            document.body.classList.remove("bd-token-resizing");
            try {
                handle.releasePointerCapture(pointerId);
            } catch {
                /* Ignore when capture already ended. */
            }
        };

        window.addEventListener("pointermove", onMove, { passive: false });
        window.addEventListener("pointerup", stop);
        window.addEventListener("pointercancel", stop);
    });
}

function ensureTokenShell(textarea) {
    if (textarea.dataset.tokenShell === "1" && textarea.__bdTokenEditor) {
        return textarea.__bdTokenEditor;
    }
    injectStyles();
    const wrap = document.createElement("div");
    wrap.className = "bd-token-wrap";
    const parent = textarea.parentNode;
    parent.insertBefore(wrap, textarea);
    wrap.appendChild(textarea);
    textarea.classList.add("bd-token-source");
    textarea.dataset.tokenShell = "1";
    textarea.setAttribute("tabindex", "-1");
    textarea.setAttribute("aria-hidden", "true");

    const editor = document.createElement("div");
    editor.className = "bd-token-editor";
    editor.contentEditable = "true";
    editor.dataset.placeholder = textarea.getAttribute("placeholder") || "";
    editor.setAttribute("role", "textbox");
    editor.setAttribute("aria-multiline", "true");
    // Mirror key classes used for layout (bd-prompt etc.)
    for (const cls of textarea.classList) {
        if (cls === "bd-token-source") continue;
        editor.classList.add(cls);
    }
    wrap.appendChild(editor);
    // R2V uses the growable card/list layout; other modes keep their existing flex sizing.
    if (textarea.closest(".bd-batch-r2v")) {
        installTokenResizeHandle(wrap, editor);
    }
    textarea.__bdTokenEditor = editor;
    textarea.__bdTokenWrap = wrap;

    // Keep textarea.value as source of truth; mirror programmatic writes into editor.
    const nativeDesc = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
    Object.defineProperty(textarea, "value", {
        configurable: true,
        enumerable: true,
        get() {
            return nativeDesc.get.call(this);
        },
        set(v) {
            nativeDesc.set.call(this, v == null ? "" : String(v));
            if (this.__bdTokenSyncing) return;
            const api = this.__bdTokenApi;
            if (api?.hydrateFromValue) api.hydrateFromValue(String(v ?? ""));
        },
    });

    // Placeholder attribute sync
    const mo = new MutationObserver(() => {
        editor.dataset.placeholder = textarea.getAttribute("placeholder") || "";
    });
    mo.observe(textarea, { attributes: true, attributeFilter: ["placeholder"] });
    textarea.__bdTokenPlaceholderObserver = mo;

    return editor;
}

function writeTextareaValue(textarea, value) {
    const nativeDesc = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
    textarea.__bdTokenSyncing = true;
    try {
        nativeDesc.set.call(textarea, value);
    } finally {
        textarea.__bdTokenSyncing = false;
    }
}

/**
 * Wire @-mention dropdown + token chip editor on a prompt textarea.
 * Typing `@` lists uploaded reference images / audios / videos; pick one to insert official tags.
 */
export function wirePromptImageMentions(editorHost, textarea, getMedia) {
    if (!textarea || textarea.dataset.mentionWired) return;
    textarea.dataset.mentionWired = "1";
    injectStyles();

    const rich = ensureTokenShell(textarea);
    const chipOpts = {
        onActivate: ({ kind, ordinal }) => {
            try {
                editorHost?.highlightRefSlot?.(kind, ordinal - 1);
            } catch {
                /* optional hook */
            }
            const scope = rich.closest(".bd-batch-card")
                || editorHost?.root
                || document;
            const sel = scope.querySelector?.(
                `[data-ref-kind="${kind}"][data-ref-index="${ordinal - 1}"]`
            );
            if (sel?.scrollIntoView) {
                sel.classList.add("bd-ref-flash");
                sel.scrollIntoView({ block: "nearest", behavior: "smooth" });
                setTimeout(() => sel.classList.remove("bd-ref-flash"), 900);
            }
        },
    };

    const hydrateFromValue = (value) => {
        const caret = document.activeElement === rich ? serializedCaretOffset(rich) : null;
        hydrateTokenEditor(rich, value, getMedia, chipOpts);
        if (caret != null && document.activeElement === rich) {
            setCaretBySerializedOffset(rich, caret);
        }
    };

    hydrateFromValue(textarea.value || "");

    let menu = null;
    let mentionStart = -1;
    let activeIndex = 0;
    let filtered = [];
    let composing = false;
    let rehydrateTimer = null;

    const syncToTextarea = ({ emitInput = true } = {}) => {
        const next = serializeTokenEditor(rich);
        if (next === textarea.value) {
            refreshTokenStates(rich, getMedia);
            return next;
        }
        writeTextareaValue(textarea, next);
        if (emitInput) textarea.dispatchEvent(new Event("input", { bubbles: true }));
        return next;
    };

    const ensureMenu = () => {
        if (menu) return menu;
        menu = document.createElement("div");
        menu.className = "bd-mention-menu hidden";
        menu.setAttribute("role", "listbox");
        // Keep editor focus while interacting with the menu / scrollbar.
        menu.addEventListener("mousedown", (e) => {
            e.preventDefault();
        });
        document.body.appendChild(menu);
        return menu;
    };

    const closeMenu = () => {
        mentionStart = -1;
        filtered = [];
        activeIndex = 0;
        if (menu) menu.classList.add("hidden");
    };

    const renderMenu = (query) => {
        const m = ensureMenu();
        const media = typeof getMedia === "function" ? getMedia() : {};
        const all = listAvailableMentions(media.refs, media.audios, media.videos);
        const q = (query || "").toLowerCase();
        filtered = all.filter((item) => {
            if (!q) return true;
            const label = String(item.label || "").toLowerCase();
            const tag = String(item.tag || "").toLowerCase();
            return label.includes(q) || tag.includes(q)
                || (item.kind === "image" && `picture ${item.index + 1}`.includes(q))
                || (item.kind === "video" && `video ${item.index + 1}`.includes(q))
                || (item.kind === "audio" && `audio ${item.index + 1}`.includes(q));
        });
        m.innerHTML = "";
        const title = document.createElement("div");
        title.className = "bd-mention-title";
        title.textContent = t("mention.title");
        m.appendChild(title);

        if (!filtered.length) {
            const empty = document.createElement("div");
            empty.className = "bd-mention-empty";
            empty.textContent = all.length ? t("mention.emptyFilter") : t("mention.emptyNoUpload");
            m.appendChild(empty);
        } else {
            filtered.forEach((item, i) => {
                const row = document.createElement("div");
                row.className = `bd-mention-item${i === activeIndex ? " active" : ""}`;
                row.dataset.index = String(i);
                row.appendChild(makeMentionMenuThumb(item));
                const label = document.createElement("span");
                label.innerHTML = `<span class="bd-mention-label">${item.label}</span>`;
                row.appendChild(label);
                row.onmousedown = (e) => {
                    e.preventDefault();
                    insertMention(item.tag);
                };
                m.appendChild(row);
            });
        }
        positionMenu(m, rich);
        m.classList.remove("hidden");
        scrollActiveIntoView();
    };

    const scrollActiveIntoView = () => {
        if (!menu || menu.classList.contains("hidden")) return;
        const row = menu.querySelector(`.bd-mention-item[data-index="${activeIndex}"]`);
        if (!row) return;
        // Adjust only the menu scroller — scrollIntoView would move page ancestors
        // and trip the capture scroll listener that closes the menu.
        const menuRect = menu.getBoundingClientRect();
        const rowRect = row.getBoundingClientRect();
        if (rowRect.bottom > menuRect.bottom) {
            menu.scrollTop += rowRect.bottom - menuRect.bottom + 4;
        } else if (rowRect.top < menuRect.top) {
            menu.scrollTop -= menuRect.top - rowRect.top + 4;
        }
    };

    /** Move highlight without rebuilding the list (keeps scroll position). */
    const moveActive = (delta) => {
        if (!filtered.length || !menu || menu.classList.contains("hidden")) return;
        const prev = activeIndex;
        activeIndex = (activeIndex + delta + filtered.length) % filtered.length;
        menu.querySelector(`.bd-mention-item[data-index="${prev}"]`)?.classList.remove("active");
        const next = menu.querySelector(`.bd-mention-item[data-index="${activeIndex}"]`);
        next?.classList.add("active");
        scrollActiveIntoView();
    };

    const insertMention = (tag) => {
        const { offset } = textBeforeCaret(rich);
        const start = mentionStart >= 0 ? mentionStart : offset;
        insertAtCaret(rich, `${tag} `, getMedia, chipOpts, { replaceFrom: start });
        closeMenu();
        syncToTextarea({ emitInput: true });
        rich.focus();
    };

    const openIfMention = () => {
        const { before, offset } = textBeforeCaret(rich);
        const match = before.match(/@([^\s@<]*)$/);
        if (!match) {
            closeMenu();
            return;
        }
        mentionStart = offset - match[0].length;
        activeIndex = 0;
        renderMenu(match[1]);
    };

    const scheduleTagRehydrate = () => {
        if (composing) return;
        clearTimeout(rehydrateTimer);
        rehydrateTimer = setTimeout(() => {
            if (composing || document.activeElement !== rich) return;
            if (!editorHasRawTagsInTextNodes(rich)) {
                syncToTextarea({ emitInput: true });
                return;
            }
            const caret = serializedCaretOffset(rich);
            const text = serializeTokenEditor(rich);
            hydrateTokenEditor(rich, text, getMedia, chipOpts);
            setCaretBySerializedOffset(rich, caret);
            syncToTextarea({ emitInput: true });
        }, 80);
    };

    rich.addEventListener("compositionstart", () => { composing = true; });
    rich.addEventListener("compositionend", () => {
        composing = false;
        scheduleTagRehydrate();
        openIfMention();
    });

    rich.addEventListener("input", () => {
        if (composing) return;
        scheduleTagRehydrate();
        openIfMention();
    });

    rich.addEventListener("click", () => {
        refreshTokenStates(rich, getMedia);
        openIfMention();
    });

    rich.addEventListener("keyup", (e) => {
        if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) openIfMention();
    });

    rich.addEventListener("paste", (e) => {
        // Block Comfy canvas node-paste (clipboard still holds last copied nodes).
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation?.();
        const text = (e.clipboardData || window.clipboardData)?.getData("text/plain") || "";
        insertAtCaret(rich, text.replace(/\r\n/g, "\n"), getMedia, chipOpts);
        syncToTextarea({ emitInput: true });
        openIfMention();
    });

    rich.addEventListener("keydown", (e) => {
        // contenteditable is not INPUT/TEXTAREA — Comfy treats Ctrl+V as graph paste.
        if ((e.ctrlKey || e.metaKey) && ["v", "c", "x"].includes(e.key?.toLowerCase?.())) {
            e.stopPropagation();
        }
        if (!menu?.classList.contains("hidden") && filtered.length) {
            if (e.key === "ArrowDown") {
                e.preventDefault();
                moveActive(1);
                return;
            }
            if (e.key === "ArrowUp") {
                e.preventDefault();
                moveActive(-1);
                return;
            }
            if (e.key === "Enter" || e.key === "Tab") {
                e.preventDefault();
                insertMention(filtered[activeIndex].tag);
                return;
            }
            if (e.key === "Escape") {
                e.preventDefault();
                closeMenu();
                return;
            }
        }

        // Atomic backspace/delete against chips.
        if (e.key === "Backspace" || e.key === "Delete") {
            const sel = window.getSelection();
            if (!sel || !sel.isCollapsed || !sel.rangeCount) return;
            const range = sel.getRangeAt(0);
            if (e.key === "Backspace") {
                let node = range.startContainer;
                let offset = range.startOffset;
                if (node === rich && offset > 0) {
                    const prev = rich.childNodes[offset - 1];
                    if (prev?.classList?.contains(TOKEN_CLASS)) {
                        e.preventDefault();
                        prev.remove();
                        syncToTextarea({ emitInput: true });
                        return;
                    }
                }
                if (node.nodeType === Node.TEXT_NODE && offset === 0) {
                    const prev = node.previousSibling;
                    if (prev?.classList?.contains(TOKEN_CLASS)) {
                        e.preventDefault();
                        prev.remove();
                        syncToTextarea({ emitInput: true });
                    }
                }
            } else if (e.key === "Delete") {
                let node = range.startContainer;
                let offset = range.startOffset;
                if (node === rich) {
                    const next = rich.childNodes[offset];
                    if (next?.classList?.contains(TOKEN_CLASS)) {
                        e.preventDefault();
                        next.remove();
                        syncToTextarea({ emitInput: true });
                        return;
                    }
                }
                if (node.nodeType === Node.TEXT_NODE && offset === (node.textContent || "").length) {
                    const next = node.nextSibling;
                    if (next?.classList?.contains(TOKEN_CLASS)) {
                        e.preventDefault();
                        next.remove();
                        syncToTextarea({ emitInput: true });
                    }
                }
            }
        }
    });

    rich.addEventListener("blur", () => {
        syncToTextarea({ emitInput: true });
        refreshTokenStates(rich, getMedia);
    });

    // document/window listeners outlive the textarea. Batch cards rebuild with
    // innerHTML, so unnamed handlers would accumulate and pin the whole editor.
    const onDocMouseDown = (e) => {
        if (!menu || menu.classList.contains("hidden")) return;
        if (e.target === rich || rich.contains?.(e.target) || menu.contains(e.target)) return;
        closeMenu();
    };
    const onWinScroll = (e) => {
        if (!menu || menu.classList.contains("hidden")) return;
        const t = e.target;
        if (t === menu || menu.contains(t)) return;
        closeMenu();
    };

    document.addEventListener("mousedown", onDocMouseDown);
    // Capture: close when the page/list moves, but ignore scrolls inside the menu.
    window.addEventListener("scroll", onWinScroll, true);
    window.addEventListener("resize", closeMenu);

    let tornDown = false;
    const teardown = () => {
        if (tornDown) return;
        tornDown = true;
        clearTimeout(rehydrateTimer);
        document.removeEventListener("mousedown", onDocMouseDown);
        window.removeEventListener("scroll", onWinScroll, true);
        window.removeEventListener("resize", closeMenu);
        menu?.remove();
        menu = null;
        textarea.__bdTokenPlaceholderObserver?.disconnect();
        delete textarea.__bdTokenPlaceholderObserver;
        delete textarea.dataset.mentionWired;
        delete textarea.__bdTokenApi;
    };

    textarea.__bdTokenApi = {
        hydrateFromValue,
        refreshMedia: () => refreshTokenStates(rich, getMedia),
        editor: rich,
        sync: () => syncToTextarea({ emitInput: false }),
        teardown,
    };
}

/**
 * Drop document/window listeners, body menus, and observers for token editors
 * under ``root``. Call before wiping a list with innerHTML, and on editor destroy.
 */
export function teardownPromptImageMentions(root = document) {
    const areas = root?.querySelectorAll?.("textarea.bd-token-source") || [];
    for (const ta of areas) {
        ta.__bdTokenApi?.teardown?.();
    }
}

/** Refresh chip missing/thumb state after refs change (optional call sites). */
export function refreshPromptTokenEditors(root = document) {
    const areas = root.querySelectorAll?.("textarea.bd-token-source") || [];
    for (const ta of areas) {
        ta.__bdTokenApi?.refreshMedia?.();
    }
}

/** Attach @-mention + tokens to global + segment positive prompt fields. */
export function mountPromptImageMentions(editor) {
    if (!editor) return;
    wirePromptImageMentions(editor, editor.globalPrompt, () => ({
        refs: editor.timeline?.global?.refs || [],
        audios: editor.timeline?.global?.refAudios || [],
        videos: promptVideosFor(editor, null, editor.timeline?.global?.refVideos || []),
    }));
    wirePromptImageMentions(editor, editor.segPrompt, () => {
        const seg = editor.timeline?.segments?.[editor.selectedIndex];
        return {
            refs: seg?.refs || [],
            audios: seg?.refAudios || [],
            videos: promptVideosFor(editor, seg, seg?.refVideos || []),
        };
    });
}
