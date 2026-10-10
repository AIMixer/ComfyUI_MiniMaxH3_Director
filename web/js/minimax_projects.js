// 搅拌机版导演台：项目管理纯函数（新增 / 复制 / 重命名 / 删除 / 切换）。
// 项目数据存于节点 properties.mmh3_projects_v1，随工作流一起保存。
// 后端按 timeline_data 顶层 projectId 对 minimax_seg_cache/<projectId>/ 做缓存隔离。
export const MMH3_PROJECT_STORE_VERSION = 1;

export function cloneMMH3Payload(value) {
    return value === undefined || value === null
        ? value
        : JSON.parse(JSON.stringify(value));
}

/** 生成合法的 projectId：仅保留 [0-9A-Za-z_-]，最长 80，且不留首尾下划线。 */
function sanitizeProjectId(value) {
    return String(value || "").trim()
        .replace(/[^0-9A-Za-z_-]+/g, "_")
        .slice(0, 80)
        .replace(/^_+|_+$/g, "");
}

/** 生成 projectId 基名；字符集与后端 resolve_project_id 的清洗规则一致。 */
export function newMMH3ProjectId() {
    return `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

/** 生成不会与 store 中既有 id 冲突的 id。 */
function nextProjectId(store, makeId) {
    const used = new Set((store.projects || []).map((p) => p.id));
    const base = sanitizeProjectId(typeof makeId === "function" ? makeId() : "project") || "project";
    if (base && !used.has(base)) return base;
    for (let attempt = 1; attempt < 200; attempt++) {
        const id = sanitizeProjectId(`${base}_${attempt}`);
        if (id && !used.has(id)) return id;
    }
    const stamp = Date.now().toString(36);
    let id = `project_${stamp}`;
    let k = 0;
    while (used.has(id)) id = `project_${stamp}_${k++}`;
    return id;
}

function nextUntitledName(store) {
    const used = new Set((store.projects || []).map((p) => String(p.name || "")));
    let index = 1;
    while (used.has(`未命名项目${index}`)) index++;
    return `未命名项目${index}`;
}

function makeProject(id, name, payload, now) {
    return {
        id,
        name: String(name || "").trim() || "未命名项目",
        payload: cloneMMH3Payload(payload) || {},
        created_at: Number(now) || Date.now(),
        updated_at: Number(now) || Date.now(),
    };
}

export function getMMH3Project(store, id) {
    if (!store || !Array.isArray(store.projects)) return null;
    return store.projects.find((p) => p && p.id === id) || null;
}

/**
 * 规范化 / 迁移项目 store。
 * - raw 为既有 properties.mmh3_projects_v1；缺失或为空时用 legacyPayload 兜底生成一个「未命名项目」。
 * - requestedActiveId 优先生效，其次取第一个项目。
 */
export function createMMH3ProjectStore(raw, requestedActiveId, legacyPayload, makeId, now = Date.now()) {
    const store = { version: MMH3_PROJECT_STORE_VERSION, projects: [] };
    const used = new Set();
    const rawProjects = raw && Array.isArray(raw.projects) ? raw.projects : [];
    for (const source of rawProjects) {
        if (!source || typeof source !== "object") continue;
        let id = sanitizeProjectId(source.id);
        if (!id || used.has(id)) id = nextProjectId(store, makeId);
        used.add(id);
        const baseName = String(source.name || "").trim();
        const project = makeProject(
            id,
            baseName || `未命名项目${store.projects.length + 1}`,
            source.payload,
            Number(source.created_at) || now,
        );
        project.updated_at = Number(source.updated_at) || project.created_at;
        store.projects.push(project);
    }
    if (store.projects.length === 0) {
        const id = sanitizeProjectId(requestedActiveId) || nextProjectId(store, makeId);
        store.projects.push(makeProject(id, "未命名项目", legacyPayload, now));
    }
    const requested = sanitizeProjectId(requestedActiveId);
    const active = getMMH3Project(store, requested) || store.projects[0];
    store.activeId = active.id;
    return store;
}

/** 把当前 timeline 快照写回指定项目。 */
export function saveMMH3Project(store, id, payload, now = Date.now()) {
    const project = getMMH3Project(store, id);
    if (!project) return store;
    project.payload = cloneMMH3Payload(payload) ?? {};
    project.updated_at = Number(now) || Date.now();
    return store;
}

export function addMMH3Project(store, name, payload, makeId, now = Date.now()) {
    const id = nextProjectId(store, makeId);
    const project = makeProject(id, String(name || "").trim() || nextUntitledName(store), payload, now);
    store.projects.push(project);
    return { store, project };
}

export function duplicateMMH3Project(store, id, makeId, now = Date.now()) {
    const source = getMMH3Project(store, id);
    if (!source) return { store, project: null };
    const project = makeProject(
        nextProjectId(store, makeId),
        (source.name || "未命名项目") + " 副本",
        source.payload,
        now,
    );
    store.projects.push(project);
    return { store, project };
}

export function renameMMH3Project(store, id, name) {
    const project = getMMH3Project(store, id);
    if (!project) return store;
    project.name = String(name || "").trim() || "未命名项目";
    project.updated_at = Date.now();
    return store;
}

/**
 * 删除项目。至少保留一个；删除后返回相邻项目 id（activeId）。
 */
export function deleteMMH3Project(store, id, now = Date.now()) {
    if (!store || !Array.isArray(store.projects) || store.projects.length <= 1) {
        return { store, activeId: (store && store.projects && store.projects[0] && store.projects[0].id) || "", deleted: false };
    }
    const index = store.projects.findIndex((p) => p.id === id);
    if (index < 0) return { store, activeId: store.projects[0].id, deleted: false };
    store.projects.splice(index, 1);
    const next = store.projects[Math.min(index, store.projects.length - 1)];
    return { store, activeId: next.id, deleted: true };
}