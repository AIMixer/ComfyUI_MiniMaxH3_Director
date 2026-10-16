// [MMX-MCP-MERGE] Pure three-way JSON merge. Missing keys differ from explicit null.
const MISSING = Symbol('missing');
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
const child = (value, key) => record(value) && own(value, key) ? value[key] : MISSING;
const copy = value => value === MISSING ? MISSING : JSON.parse(JSON.stringify(value));
const field = (path, key) => path ? `${path}.${key}` : key;
const entryPath = (path, id) => `${path}[id=${JSON.stringify(id)}]`;

export function directorJsonEqual(a, b) {
    if (a === b) return true;
    if (a === MISSING || b === MISSING) return false;
    if (Array.isArray(a) && Array.isArray(b)) {
        return a.length === b.length && a.every((value, i) => directorJsonEqual(value, b[i]));
    }
    if (!record(a) || !record(b)) return false;
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length
        && keys.every(key => own(b, key) && directorJsonEqual(a[key], b[key]));
}

function keyed(values) {
    return values.every(value => Array.isArray(value)
        && value.every(item => record(item) && (typeof item.id === 'string' || typeof item.id === 'number'))
        && new Set(value.map(item => String(item.id))).size === value.length);
}

function difference(path, left, right, details) {
    if (details) {
        const entry = {
            path,
            expectedValue: left === MISSING ? null : copy(left),
            actualValue: right === MISSING ? null : copy(right),
            ...(left === MISSING ? { expectedMissing: true } : {}),
            ...(right === MISSING ? { actualMissing: true } : {}),
        };
        if (typeof left === "number" && typeof right === "number"
            && Number.isFinite(left) && Number.isFinite(right)) {
            entry.delta = Number((left - right).toPrecision(12));
        }
        details.push(entry);
    }
    return [path];
}

export function directorDifferencePaths(left, right, path = '', details = null) {
    if (directorJsonEqual(left, right)) return [];
    if (record(left) && record(right)) {
        return [...new Set([...Object.keys(left), ...Object.keys(right)])]
            .flatMap(key => directorDifferencePaths(child(left, key), child(right, key), field(path, key), details));
    }
    if (keyed([left, right])) {
        const maps = [left, right].map(items => new Map(items.map(item => [String(item.id), item])));
        const result = [...new Set([...maps[0].keys(), ...maps[1].keys()])]
            .flatMap(id => directorDifferencePaths(...maps.map(map => map.has(id) ? map.get(id) : MISSING), entryPath(path, id), details));
        const leftOrder = left.map(item => String(item.id)), rightOrder = right.map(item => String(item.id));
        if (!directorJsonEqual(leftOrder, rightOrder)) result.push(...difference(`${path}.$order`, leftOrder, rightOrder, details));
        return result;
    }
    if (Array.isArray(left) && Array.isArray(right) && left.length === right.length) {
        return left.flatMap((value, i) => directorDifferencePaths(value, right[i], `${path}[${i}]`, details));
    }
    return difference(path, left, right, details);
}

// Reuse the same path traversal as readback, including stable IDs and deletions.
// Numeric delta is expected minus computed; preserve strings/types in diagnostics.
export function formatDirectorMismatches(expected, actual, paths, kind = "agent") {
    const details = [];
    directorDifferencePaths(expected, actual, '', details);
    const matches = details.filter(d => paths.some(p => !p || d.path === p
        || d.path.startsWith(`${p}.`) || d.path.startsWith(`${p}[`)
        || p.startsWith(`${d.path}.`) || p.startsWith(`${d.path}[`)));
    const format = (value, missing) => {
        if (missing) return "[missing]";
        const text = JSON.stringify(value);
        return text.length > 300 ? `${text.slice(0, 300)}…` : text;
    };
    const rows = matches.slice(0, 24).map(d => {
        const labels = kind === "frontend" ? ["frontend before", "frontend after"] : ["agent wrote", "frontend computed"];
        return `${d.path} (${labels[0]} ${format(d.expectedValue, d.expectedMissing)}, `
            + `${labels[1]} ${format(d.actualValue, d.actualMissing)}`
            + (d.delta === undefined ? "" : `, delta ${d.delta}`) + ")";
    });
    if (matches.length > 24) rows.push(`… ${matches.length - 24} more mismatches`);
    return rows.join(", ") || "none";
}

export function mergeDirectorValues(baseline, submitted, frontend) {
    const conflicts = [], changedPaths = [], preservedPaths = [];
    function conflict(path, agentValue, frontendValue) {
        conflicts.push({ path, agentValue: agentValue === MISSING ? null : copy(agentValue),
            frontendValue: frontendValue === MISSING ? null : copy(frontendValue), resolution: 'agent',
            ...(agentValue === MISSING ? { agentDeleted: true } : {}),
            ...(frontendValue === MISSING ? { frontendDeleted: true } : {}) });
    }
    function merge(base, agent, front, path) {
        if (directorJsonEqual(base, agent)) {
            preservedPaths.push(...directorDifferencePaths(base, front, path));
            return copy(front);
        }
        if (directorJsonEqual(base, front) || directorJsonEqual(agent, front)) {
            changedPaths.push(...directorDifferencePaths(base, agent, path));
            return copy(agent);
        }
        if (record(base) && record(agent) && record(front)) {
            const out = {};
            for (const key of new Set([...Object.keys(base), ...Object.keys(agent), ...Object.keys(front)])) {
                const value = merge(child(base, key), child(agent, key), child(front, key), field(path, key));
                // defineProperty also safely round-trips user keys named __proto__.
                if (value !== MISSING) Object.defineProperty(out, key, { value, enumerable: true, writable: true, configurable: true });
            }
            return out;
        }
        if (keyed([base, agent, front])) {
            const maps = [base, agent, front].map(items => new Map(items.map(item => [String(item.id), item])));
            const orders = [base, agent, front].map(items => items.map(item => String(item.id)));
            // Membership/order remains explicit intent even when both sides add
            // independent rows and the final array combines those insertions.
            if (!directorJsonEqual(orders[0], orders[1])) changedPaths.push(path + '.$order');
            else if (!directorJsonEqual(orders[0], orders[2])) preservedPaths.push(path + '.$order');
            const values = new Map();
            for (const id of new Set(orders.flat())) {
                const value = merge(...maps.map(map => map.has(id) ? map.get(id) : MISSING), entryPath(path, id));
                if (value !== MISSING) values.set(id, value);
            }
            // Reordering existing rows is separate from inserting/deleting rows.
            // Stable IDs keep a frontend insertion from shifting the Agent's target.
            const shared = orders[0].filter(id => orders[1].includes(id) && orders[2].includes(id));
            const order = orders.map(ids => ids.filter(id => shared.includes(id)));
            const agentReordered = !directorJsonEqual(order[0], order[1]);
            const frontendReordered = !directorJsonEqual(order[0], order[2]);
            if (agentReordered && frontendReordered && !directorJsonEqual(order[1], order[2])) {
                conflict(`${path}.$order`, order[1], order[2]);
            }
            const preferred = agentReordered ? orders[1] : orders[2];
            const combined = [...preferred];
            // Preserve independent insertions close to their surviving predecessor.
            for (const ids of [orders[1], orders[2]]) {
                for (let i = 0; i < ids.length; i++) {
                    const id = ids[i];
                    if (!values.has(id) || combined.includes(id)) continue;
                    const previous = ids.slice(0, i).reverse().find(candidate => combined.includes(candidate));
                    const at = previous === undefined ? 0 : combined.indexOf(previous) + 1;
                    combined.splice(at, 0, id);
                }
            }
            return combined.filter(id => values.has(id)).map(id => values.get(id));
        }
        if ([base, agent, front].every(Array.isArray)
            && base.length === agent.length && base.length === front.length) {
            return base.map((value, i) => merge(value, agent[i], front[i], `${path}[${i}]`));
        }
        conflict(path, agent, front);
        changedPaths.push(path);
        return copy(agent);
    }
    return { value: merge(baseline, submitted, frontend, ''), conflicts, changedPaths, preservedPaths };
}

// MCP timeline submissions are PATCHes: omitted object/row fields inherit the
// read baseline, so three-way merge still preserves concurrent frontend edits.
// An explicitly supplied array owns its membership/order; retained IDs inherit
// omitted row fields. null, false, 0 and [] remain explicit values.
export function completeDirectorPatch(baseline, patch) {
    if (record(baseline) && record(patch)) {
        const out = copy(baseline);
        for (const key of Object.keys(patch)) {
            Object.defineProperty(out, key, {
                value: completeDirectorPatch(child(baseline, key), patch[key]),
                enumerable: true, writable: true, configurable: true,
            });
        }
        return out;
    }
    if (keyed([baseline, patch])) {
        const rows = new Map(baseline.map(row => [String(row.id), row]));
        return patch.map(row => completeDirectorPatch(rows.get(String(row.id)) ?? MISSING, row));
    }
    return copy(patch);
}

function generatedDirectorTimeline(state) {
    return ['fl2v', 'prompt_batch', 'image_batch', 'gen_blank', 'gen_image'].includes(state.timelineMode);
}

function usesDirectorShots(state) {
    return Array.isArray(state.shots) && (state.timelineMode === 'fl2v'
        || /^fl2v(?:\s|—|$)/.test(String(state.global?.taskType || '')));
}

// Workspace containers are persisted backups plus active UI mirrors. Ignore
// incoming containers, preserve current inactive drafts, and rebuild only the
// active mirror from merged authoritative fields before the normal importer.
export function restoreDirectorWorkspaces(value, current, taskKey) {
    for (const key of ['batchWorkspaces', 'videoWorkspaces']) {
        if (record(current[key])) value[key] = copy(current[key]);
    }
    const container = value.timelineMode === 'video' ? 'videoWorkspaces'
        : ['prompt_batch', 'image_batch'].includes(value.timelineMode) ? 'batchWorkspaces' : null;
    const workspace = container && value[container]?.[taskKey];
    if (record(workspace)) {
        for (const key of ['segments', 'editMode', 'runSelectEnabled', 'runSelection']) {
            if (own(value, key)) workspace[key] = copy(value[key]);
        }
        if (container === 'videoWorkspaces') {
            for (const key of ['video', 'videoClips', 'frameRate']) {
                if (own(value, key)) workspace[key] = copy(value[key]);
            }
        }
        if (record(workspace.globalCommon) && record(value.global)) {
            for (const key of ['commonEnabled', 'commonCollapsed', 'prompt', 'refs', 'refAudios', 'refVideos', 't2vCommon']) {
                if (own(value.global, key)) workspace.globalCommon[key] = copy(value.global[key]);
            }
        }
    }
    // In generated timelines this is a synthetic canvas count, not source-video
    // metadata. Keep the current seed for the importer but never trust the Agent.
    if (generatedDirectorTimeline(value) && record(value.video)
        && own(current.video || {}, 'sourceFrameCount')) {
        value.video.sourceFrameCount = current.video.sourceFrameCount;
    }
    return value;
}

// Canonical output dimensions have legacy top-level mirrors. Treat an edit to
// either address as one edit, rather than letting a stale mirror undo it on parse.
export function canonicalDirectorMergeState(value) {
    const state = copy(value);
    if (usesDirectorShots(state)) state.timelineMode = 'fl2v';
    // Root timing totals are editor-derived. They are not edit intent in ANY
    // merge leg or readback. shot/segment durationSec remains authoritative;
    // true source-video metadata is retained outside generated timeline modes.
    delete state.totalFrames;
    delete state.durationSec;
    delete state.batchWorkspaces;
    delete state.videoWorkspaces;
    if (generatedDirectorTimeline(state) && record(state.video)) {
        delete state.video.sourceFrameCount;
    }
    if (state.output?.mode === 'fixed' && state.timelineMode !== 'video') {
        // These size mirrors are recomputed from the merged canvas dimensions.
        // Otherwise independent width/height edits can conflict through longEdge.
        delete state.width;
        delete state.height;
        delete state.refMaxSize;
        delete state.output.longEdge;
    }
    if (usesDirectorShots(state)) {
        state.timelineMode = 'fl2v';
        delete state.segments;
        delete state.keyframes;
    }
    return state;
}

export function prepareDirectorSubmission(baseline, submitted, taskKey) {
    const agent = completeDirectorPatch(baseline, submitted);
    if (usesDirectorShots(agent)) {
        agent.timelineMode = 'fl2v';
        // FL2V is authored through shots. Compatibility mirrors, including
        // contradictory or structurally stale mirrors, are not write intent.
        delete agent.segments;
        delete agent.keyframes;
    } else if (['prompt_batch', 'image_batch'].includes(agent.timelineMode)
        && own(submitted, 'shots') && !directorJsonEqual(baseline.shots, submitted.shots)) {
        throw new Error('shots author FL2V; the current prompt_batch task uses segments. Read global.taskType and edit segments, or explicitly switch to a valid FL2V task with its shots.');
    }
    const output = agent.output;
    if (!record(output) || !record(baseline.output)) return agent;
    let resized = false;
    for (const key of ['width', 'height']) {
        const outputChanged = !directorJsonEqual(output[key], baseline.output[key]);
        const rootChanged = own(agent, key) && !directorJsonEqual(agent[key], baseline[key]);
        if (outputChanged && rootChanged && output[key] !== agent[key]) {
            throw new Error(`Conflicting submitted dimension mirrors: output.${key} vs ${key}`);
        }
        if (rootChanged && !outputChanged) output[key] = agent[key];
        if (outputChanged || rootChanged) resized = true;
    }
    if (resized && Number.isFinite(output.width) && Number.isFinite(output.height)) {
        agent.width = output.width;
        agent.height = output.height;
        // Prompt/gen/fl2v editors derive the fixed canvas and its size mirrors.
        if (output.mode === 'fixed' || agent.timelineMode === 'prompt_batch' || agent.timelineMode === 'fl2v') {
            output.mode = 'fixed';
            output.aspectRatio = '自定义';
            output.longEdge = Math.max(output.width, output.height);
            agent.refMaxSize = output.longEdge;
        }
    }
    return agent;
}

// Paths present in the expected editable state. Added importer defaults are
// not failed writes, but ID order/membership and explicit scalar/null values are.
function directorExpectedPaths(value, path = '') {
    if (record(value)) {
        return Object.keys(value).flatMap(key => directorExpectedPaths(value[key], field(path, key)));
    }
    if (keyed([value])) {
        return [path + '.$order', ...value.flatMap(row => directorExpectedPaths(row, entryPath(path, String(row.id))))];
    }
    if (Array.isArray(value) && value.length) {
        return [path, ...value.flatMap((item, i) => directorExpectedPaths(item, path + '[' + i + ']'))];
    }
    return [path];
}

// Verify only changed/preserved editable intent. Missing PATCH fields were
// filled from the read baseline; frontend-added defaults have no expected leaf.
export function lostDirectorFrontendEdits(expected, actual, preservedPaths) {
    const intent = directorExpectedPaths(expected);
    return directorDifferencePaths(expected, actual).filter(path =>
        preservedPaths.some(p => !p || path === p || path.startsWith(p + '.')
            || path.startsWith(p + '[') || p.startsWith(path + '.') || p.startsWith(path + '['))
        && intent.some(p => path === p || p.startsWith(path + '.') || p.startsWith(path + '[')));
}
