/**
 * External Director Group witness — what the first-pass cache panel can see.
 *
 * `i2v_groups` / `r2v_groups` reach the Director as tensors at execute time, so
 * the cache-status request (which only carries Director widget values) cannot
 * rebuild the executed segments. Without a witness it rebuilt the expected
 * fingerprint from the UI timeline — where external-group prompts, durations
 * and reference media never live — and therefore always reported 不匹配.
 *
 * The witness digests
 *   1. every node upstream of the connected group input (id, class, bypass
 *      mode, live widget values) plus the link topology between them,
 *   2. the plan-relevant subset of the timeline payload, and
 *   3. one record per graph-wired group (= one segment = one cache slot),
 * and travels inside `timeline_data`. That widget is already sent both by the
 * cache-status panel and by the Director node at execute time, so both sides
 * compare the very same value: any wiring change (prompt, duration, ref slot,
 * reference file, node bypass) invalidates the cached segment.
 *
 * `facets` buckets the graph by category — wiring / prompt / length / media /
 * other, plus the timeline projection — so the panel can name the changed item
 * ("外接组提示词") instead of listing every key that could have changed. An
 * upstream node's widgets are bucketed through the input they feed as well, so a
 * multiline text node whose widget is named `value` still counts as a prompt.
 * `groups[i]` does the same per segment: `dur` is the group's own duration in
 * seconds (the panel derives its frame count from it) and `prompt` is a digest
 * of its resolved prompt text, independent of widget naming.
 *
 * Group discovery is delegated to `minimax_timeline.js`
 * (`setExternalGroupSpecsProvider`), which already walks Combine slots /
 * reroutes and knows the packer nodes — the witness and the Director UI must
 * agree on the segment order the run will use.
 *
 * Cost: only runs when a group input is connected, walks the group upstream
 * chain (a handful of nodes) and hashes a few KB — well under a millisecond.
 */

export const EXTERNAL_WITNESS_KEY = "externalGroupsWitness";

const GROUP_PORTS = ["i2v_groups", "r2v_groups"];
const WITNESS_VERSION = 4;
const FACET_NAMES = ["wiring", "prompt", "length", "media", "other"];
const MAX_NODES = 500;
const MAX_DEPTH = 48;
const MAX_GROUPS = 128;
const MAX_STRING = 400;
const MAX_ARRAY = 64;
const MAX_OBJECT_DEPTH = 8;

/**
 * Timeline keys an external-group run really reads. Deliberately narrow: once
 * groups are wired in, the UI card's own rows, prompt, duration and total
 * frames are ignored, so projecting them would invent mismatches for edits that
 * change nothing. What is left is the row-level state the group path resolves
 * per segment (`refImageSize`, `continuityFromPrev`), the continuity switch, and —
 * only when the common block is on — the shared reference media it merges in.
 */
const TIMELINE_TOP_KEYS = [
    "frameRate",
    "width",
    "height",
    "refMaxSize",
    "continuousReference",
];
const TIMELINE_GLOBAL_KEYS = ["commonEnabled", "continuousReference"];
const TIMELINE_COMMON_KEYS = ["prompt", "refs", "refVideos", "refAudios"];
const TIMELINE_OUTPUT_KEYS = ["refImageSize"];
const TIMELINE_ROW_ARRAYS = ["segments"];
const TIMELINE_ROW_KEYS = ["index", "refImageSize", "continuityFromPrev"];

/**
 * Widget / input name buckets behind the named facets. Chinese aliases matter:
 * third-party packers name the duration widget 「时长_秒」 and their media input
 * 「素材」, which an English-only matcher silently drops into the catch-all.
 */
const PROMPT_HINT_RE =
    /prompt|text|caption|descri|positive|negative|instruct|lyric|tag|note|script|提示词|文本|描述|台词|旁白|字幕/i;
const LENGTH_HINT_RE =
    /frame|duration|length|second|sec|time|fps|count|帧|秒|时长|长度/i;
const MEDIA_HINT_RE =
    /image|img|picture|photo|video|vid|audio|aud|sound|file|path|media|mask|ref|asset|图片|图像|视频|音频|素材|参考|角色卡/i;

/** Provider installed by minimax_timeline.js (group chain traversal lives there). */
let groupSpecsProvider = null;

/**
 * Register the group-chain provider. The callback receives the Director node
 * and returns `[{slot, node, nodeLabel, durationSec, prompt}]` in run order, or
 * null when nothing is wired.
 */
export function setExternalGroupSpecsProvider(provider) {
    groupSpecsProvider = typeof provider === "function" ? provider : null;
}

/** FNV-1a over UTF-16 code units: tiny, dependency free, stable across sessions. */
function digest32(text) {
    let hash = 0x811c9dc5;
    const value = String(text ?? "");
    for (let i = 0; i < value.length; i += 1) {
        const code = value.charCodeAt(i);
        hash ^= code & 0xff;
        hash = Math.imul(hash, 0x01000193) >>> 0;
        hash ^= (code >>> 8) & 0xff;
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash.toString(16).padStart(8, "0");
}

/**
 * Structural clone with stable ordering. Long strings (base64 previews, whole
 * prompts) collapse to `#length:digest` so the witness stays small while still
 * changing whenever the content changes.
 */
function stableValue(value, depth = 0) {
    if (value == null) return null;
    const type = typeof value;
    if (type === "number") return Number.isFinite(value) ? value : String(value);
    if (type === "boolean") return value;
    if (type === "string") {
        return value.length > MAX_STRING ? `#${value.length}:${digest32(value)}` : value;
    }
    if (type === "bigint") return String(value);
    if (type === "function" || type === "symbol" || type === "undefined") return null;
    if (Array.isArray(value)) {
        const items = value.slice(0, MAX_ARRAY).map((item) => stableValue(item, depth + 1));
        if (value.length > MAX_ARRAY) items.push(`#more:${value.length}`);
        return items;
    }
    if (depth > MAX_OBJECT_DEPTH) return String(value);
    const out = {};
    for (const key of Object.keys(value).sort()) {
        out[key] = stableValue(value[key], depth + 1);
    }
    return out;
}

function stableStringify(value) {
    try {
        return JSON.stringify(stableValue(value));
    } catch {
        return "";
    }
}

function pickKeys(source, keys) {
    const out = {};
    if (!source || typeof source !== "object") return out;
    for (const key of keys) {
        if (key in source) out[key] = source[key];
    }
    return out;
}

/** Plan-relevant projection of a timeline payload (witness key excluded). */
function timelineProjection(timeline) {
    const out = pickKeys(timeline, TIMELINE_TOP_KEYS);
    const globalBlock = timeline?.global;
    out.global = pickKeys(globalBlock, TIMELINE_GLOBAL_KEYS);
    if (out.global.commonEnabled) {
        Object.assign(out.global, pickKeys(globalBlock, TIMELINE_COMMON_KEYS));
    }
    out.output = pickKeys(timeline?.output, TIMELINE_OUTPUT_KEYS);
    for (const name of TIMELINE_ROW_ARRAYS) {
        const rows = timeline?.[name];
        out[name] = Array.isArray(rows)
            ? rows.map((row) => (row && typeof row === "object" ? pickKeys(row, TIMELINE_ROW_KEYS) : row))
            : [];
    }
    return out;
}

function graphOf(node) {
    return node?.graph
        ?? globalThis.app?.graph
        ?? globalThis.app?.canvas?.graph
        ?? null;
}

/** Link lookup that tolerates Map / array / plain-object link tables. */
function linkRecord(graph, linkId) {
    if (linkId == null || !graph) return null;
    for (const pool of [graph.links, graph._links]) {
        if (!pool) continue;
        let record = null;
        if (typeof pool.get === "function") {
            record = pool.get(linkId) ?? pool.get(String(linkId));
        } else if (Array.isArray(pool)) {
            record = pool.find((item) => item && String(item.id) === String(linkId));
        } else {
            record = pool[linkId] ?? pool[String(linkId)];
        }
        if (!record) continue;
        const originId = record.origin_id ?? record.originId;
        if (originId == null) return null;
        return {
            originId: String(originId),
            originSlot: record.origin_slot ?? record.originSlot ?? 0,
        };
    }
    return null;
}

function findNode(graph, id) {
    const wanted = String(id);
    if (typeof graph?.getNodeById === "function") {
        const direct = graph.getNodeById(wanted) ?? graph.getNodeById(Number(wanted));
        if (direct) return direct;
    }
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        if (String(node?.id) === wanted) return node;
    }
    return null;
}

/** Widgets that carry state (skips non-serialized widgets and buttons). */
function liveWidgets(node) {
    const out = [];
    for (const widget of node?.widgets || []) {
        if (!widget || widget.options?.serialize === false) continue;
        if (String(widget.type || "").toLowerCase() === "button") continue;
        out.push(widget);
    }
    return out;
}

/** Live widget values of one node. */
function widgetDigest(node) {
    return stableStringify(
        liveWidgets(node).map((widget) => [String(widget.name ?? ""), stableValue(widget.value)]),
    );
}

/**
 * Which named facet a widget belongs to. Everything that is not a prompt, a
 * length or reference media lands in "other" so a change is always localizable.
 */
function widgetFacet(name) {
    if (!name) return "other";
    if (PROMPT_HINT_RE.test(name)) return "prompt";
    if (LENGTH_HINT_RE.test(name)) return "length";
    if (MEDIA_HINT_RE.test(name)) return "media";
    return "other";
}

/** Which named facet an input link belongs to ("" = wiring). */
function wireFacet(name) {
    if (!name) return "";
    if (PROMPT_HINT_RE.test(name)) return "prompt";
    if (MEDIA_HINT_RE.test(name)) return "media";
    return "";
}

/**
 * Facet of an input name, or "" when the name says nothing. Used to bucket the
 * widgets of an *upstream* node by the input it feeds: the text node wired into
 * `prompt` is very often named `value` (PrimitiveStringMultiline) or `string`,
 * so its own widget name alone would drop a prompt edit into the catch-all
 * "other" bucket instead of naming it「外接组提示词」.
 */
function facetForName(name) {
    if (!name) return "";
    if (PROMPT_HINT_RE.test(name)) return "prompt";
    if (LENGTH_HINT_RE.test(name)) return "length";
    if (MEDIA_HINT_RE.test(name)) return "media";
    return "";
}

/** Priority when a node is reachable through several inputs at once. */
const FACET_RANK = { prompt: 3, length: 2, media: 1, "": 0 };

/**
 * Bucket of one widget. The widget's own name wins; the role of the input the
 * node feeds is the fallback, so an unknown name is still localizable.
 */
function widgetFacetIn(name, role) {
    const byName = widgetFacet(name);
    if (byName !== "other" || !role) return byName;
    return role;
}

/**
 * Upstream chain of one node.
 *
 * `entries` is the aggregate identity, kept byte-identical to the first version
 * so witnesses already written into existing caches still compare equal.
 * `facets` buckets the same information by category for the panel's diff line,
 * including the widgets of upstream nodes, bucketed through the input they feed.
 */
function graphEntries(graph, startNode) {
    const seen = new Set();
    const entries = [];
    const facets = { wiring: [], prompt: [], length: [], media: [], other: [] };
    // Node id -> facet of the input it feeds. Priority-merged so the result does
    // not depend on which branch of the walk reaches the node first (a text node
    // feeding both `medias` and `prompt` is a prompt source).
    const roles = new Map();
    const rememberRole = (id, role) => {
        if (!role) return;
        const key = `n${id}`;
        const prev = roles.get(key) || "";
        if ((FACET_RANK[role] || 0) > (FACET_RANK[prev] || 0)) roles.set(key, role);
    };
    const stack = startNode ? [{ node: startNode, depth: 0 }] : [];
    while (stack.length && entries.length < MAX_NODES) {
        const { node, depth } = stack.pop();
        if (!node || depth > MAX_DEPTH) continue;
        const key = `n${node.id}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const cls = String(node.comfyClass || node.type || "");
        const label = `${node.id}|${cls}`;
        const mode = `m${Number(node.mode ?? 0)}`;
        const wires = [];
        const plainWires = [];
        for (const input of node.inputs || []) {
            if (input?.link == null) continue;
            const name = String(input.name ?? "");
            const record = linkRecord(graph, input.link);
            if (!record) {
                wires.push(`${name}<missing`);
                plainWires.push(`${name}<missing`);
                continue;
            }
            const wire = `${name}<${record.originId}:${record.originSlot}`;
            wires.push(wire);
            const bucket = wireFacet(name);
            if (bucket) facets[bucket].push(`${label}|${wire}`);
            else plainWires.push(wire);
            const source = findNode(graph, record.originId);
            if (source) {
                rememberRole(source.id, facetForName(name));
                stack.push({ node: source, depth: depth + 1 });
            }
        }
        wires.sort();
        plainWires.sort();
        entries.push([
            String(node.id),
            cls,
            mode,
            wires.join(","),
            widgetDigest(node),
        ].join("|"));
        facets.wiring.push(`${label}|${mode}|${plainWires.join(",")}`);
        const role = roles.get(key) || "";
        for (const widget of liveWidgets(node)) {
            const bucket = widgetFacetIn(String(widget.name ?? ""), role);
            facets[bucket].push(
                `${label}|${widget.name}=${stableStringify(stableValue(widget.value))}`,
            );
        }
    }
    return { entries, facets };
}

function digestEntries(entries) {
    return digest32(entries.slice().sort().join("\n"));
}

function facetDigests(facets) {
    const out = {};
    for (const name of FACET_NAMES) {
        out[name] = digest32((facets?.[name] || []).slice().sort().join("\n"));
    }
    return out;
}

function connectedGroupPort(node) {
    for (const name of GROUP_PORTS) {
        const input = (node?.inputs || []).find((item) => String(item?.name) === name);
        if (input && input.link != null) return { port: name, link: input.link };
    }
    return null;
}

/**
 * One record per graph-wired group, in the order the Director executes them:
 * `{slot, node, dur, prompt, sub, facets}`.
 *
 * `dur` is the group's own duration in seconds — the cache panel derives the
 * segment's frame count from it exactly like the backend does. `prompt` is a
 * digest of the resolved prompt text, so a prompt edit is named even when the
 * packer names its widget something the facet heuristics do not know.
 */
function groupRecords(graph, node) {
    if (typeof groupSpecsProvider !== "function") return null;
    let specs = null;
    try {
        specs = groupSpecsProvider(node);
    } catch {
        return null;
    }
    if (!Array.isArray(specs) || !specs.length) return null;
    return specs.slice(0, MAX_GROUPS).map((spec, index) => {
        const leaf = spec?.node || null;
        const dur = Number(spec?.durationSec);
        const record = {
            slot: String(spec?.slot || `group_${index}`),
            node: String(
                spec?.nodeLabel
                || (leaf ? `${leaf.id}:${leaf.comfyClass || leaf.type || ""}` : ""),
            ),
            dur: Number.isFinite(dur) && dur > 0 ? dur : null,
            prompt: digest32(stableStringify(String(spec?.prompt ?? ""))),
        };
        if (leaf) {
            const { entries, facets } = graphEntries(graph, leaf);
            record.sub = digestEntries(entries);
            record.facets = facetDigests(facets);
        } else {
            // Unknown packer: nothing to walk, the slot's own values are all we have.
            record.sub = digest32(`${record.slot}|${record.dur ?? ""}|${record.prompt}`);
            record.facets = facetDigests({});
        }
        return record;
    });
}

/**
 * Witness for one Director node, or null when no group input is connected.
 * `graph`/`timeline`/`facets.*` are short digests; they are not meaningful on
 * their own.
 */
export function buildExternalGroupsWitness(node) {
    const connected = connectedGroupPort(node);
    if (!connected) return null;
    const graph = graphOf(node);
    if (!graph) return null;
    const record = linkRecord(graph, connected.link);
    const source = record ? findNode(graph, record.originId) : null;
    const { entries, facets } = graphEntries(graph, source);
    if (!entries.length) return null;
    const witness = {
        v: WITNESS_VERSION,
        port: connected.port,
        source: source ? `${source.id}:${source.comfyClass || source.type || ""}` : "",
        nodes: entries.length,
        graph: digestEntries(entries),
        facets: facetDigests(facets),
    };
    const groups = groupRecords(graph, node);
    if (groups?.length) witness.groups = groups;
    return witness;
}

/**
 * Run selection carried by the timeline payload. Under external groups
 * 「选择运行」 selects *groups*, so the panel mirrors it to keep the same
 * selected/total split the run will use.
 */
function timelineSelection(timeline) {
    if (!timeline || typeof timeline !== "object") return null;
    const on = Boolean(timeline.runSelectEnabled ?? timeline.run_select_enabled);
    if (!on) return null;
    const raw = timeline.runSelection ?? timeline.run_selection;
    if (!Array.isArray(raw)) return null;
    const idx = raw
        .slice(0, MAX_ARRAY)
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value >= 0);
    return idx.length ? { on: true, idx } : null;
}

/**
 * Attach or refresh the witness on a timeline payload object (mutates).
 * A disconnected group input drops any stale witness, so the backend never
 * mistakes a UI-timeline run for an external-group run.
 */
export function attachExternalGroupsWitness(node, timeline) {
    if (!timeline || typeof timeline !== "object") return timeline;
    delete timeline[EXTERNAL_WITNESS_KEY];
    const witness = buildExternalGroupsWitness(node);
    if (!witness) return timeline;
    witness.timeline = digest32(stableStringify(timelineProjection(timeline)));
    if (witness.facets) witness.facets.timeline = witness.timeline;
    // Segment lengths are derived from the group durations at the same fps the
    // run will read out of this very payload.
    const fps = Number(timeline.frameRate);
    if (Number.isFinite(fps) && fps > 0) witness.fps = fps;
    const selection = timelineSelection(timeline);
    if (selection) witness.sel = selection;
    timeline[EXTERNAL_WITNESS_KEY] = witness;
    return timeline;
}

function sameWitness(a, b) {
    if (!a || !b) return a === b;
    return a.graph === b.graph
        && a.timeline === b.timeline
        && a.port === b.port
        && a.nodes === b.nodes
        && a.source === b.source
        && a.fps === b.fps
        && stableStringify(a.facets) === stableStringify(b.facets)
        && stableStringify(a.groups) === stableStringify(b.groups)
        && stableStringify(a.sel) === stableStringify(b.sel);
}

/**
 * Same, for a serialized `timeline_data` string — used by the cache-status
 * payload and by the widget serializer so both paths send the same witness.
 * Returns the input unchanged when the value is not a JSON object.
 */
export function injectExternalGroupsWitness(node, timelineJson) {
    if (typeof timelineJson !== "string" || !timelineJson.trim()) return timelineJson;
    if (!connectedGroupPort(node) && !timelineJson.includes(EXTERNAL_WITNESS_KEY)) {
        return timelineJson;
    }
    let parsed = null;
    try {
        parsed = JSON.parse(timelineJson);
    } catch {
        return timelineJson;
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return timelineJson;
    const before = parsed[EXTERNAL_WITNESS_KEY];
    attachExternalGroupsWitness(node, parsed);
    if (sameWitness(before, parsed[EXTERNAL_WITNESS_KEY])) return timelineJson;
    try {
        return JSON.stringify(parsed);
    } catch {
        return timelineJson;
    }
}
