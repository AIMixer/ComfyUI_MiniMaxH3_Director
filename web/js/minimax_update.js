/** User-triggered, fast-forward-only Director updates. */

import { api } from "../../scripts/api.js";
import { t } from "./minimax_i18n.js";


async function postJson(path, body = {}) {
    const response = await api.fetchApi(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    let data = {};
    try {
        data = await response.json();
    } catch {
        data = {};
    }
    if (!response.ok) {
        const error = new Error(data.message || `HTTP ${response.status}`);
        error.reason = data.reason || "update_failed";
        error.data = data;
        throw error;
    }
    return data;
}


function shortHash(value) {
    return String(value || "").slice(0, 7) || "-";
}


function reasonText(reason, data = {}) {
    const known = new Set([
        "git_not_found",
        "git_timeout",
        "not_git_checkout",
        "detached_head",
        "no_upstream",
        "unsupported_upstream",
        "fetch_failed",
        "dirty_worktree",
        "diverged_history",
        "local_commits",
        "queue_busy",
        "state_changed",
        "remote_changed",
        "invalid_request",
        "update_failed",
    ]);
    const key = known.has(reason) ? `update.reason.${reason}` : "update.reason.update_failed";
    let message = t(key, {
        running: data.queueRunning ?? 0,
        pending: data.queuePending ?? 0,
    });
    if (data.message && ["fetch_failed", "update_failed"].includes(reason)) {
        message += `\n\n${data.message}`;
    }
    return message;
}


async function showDialog(editor, options) {
    if (typeof editor?.showBdDialog === "function") {
        return editor.showBdDialog(options);
    }
    return window.confirm(options.message || options.title || "");
}


async function showMessage(editor, message) {
    if (typeof editor?.showBdMessage === "function") {
        return editor.showBdMessage(t("update.title"), message);
    }
    window.alert(message);
}


function availableMessage(status) {
    const commitLines = (status.commits || [])
        .map((item) => `${item.hash}  ${item.subject}`)
        .join("\n");
    let message = t("update.available", {
        count: status.behind,
        local: shortHash(status.localHead),
        remote: shortHash(status.remoteHead),
        commits: commitLines || "-",
    });
    if (status.dependencyFiles?.length) {
        message += `\n\n${t("update.dependenciesWarning", { files: status.dependencyFiles.join(", ") })}`;
    }
    return message;
}


function setBusy(button, key) {
    button.disabled = Boolean(key);
    button.textContent = key ? t(key) : t("toolbar.checkUpdate");
    button.title = t("tooltip.checkUpdate");
}


export function bindUpdateAction(editor) {
    const button = editor.root?.querySelector('[data-a="check-update"]');
    if (!button) return;
    button.onclick = async (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (button.disabled) return;

        try {
            setBusy(button, "update.checking");
            const status = await postJson("/minimax/director/update/status");
            if (!status.supported || !status.canUpdate) {
                setBusy(button, "");
                await showMessage(editor, reasonText(status.reason, status));
                return;
            }
            if (status.queueRunning || status.queuePending) {
                setBusy(button, "");
                await showMessage(editor, reasonText("queue_busy", status));
                return;
            }
            if (!status.updateAvailable) {
                setBusy(button, "");
                await showMessage(editor, t("update.current", { head: shortHash(status.localHead) }));
                return;
            }

            const confirmed = await showDialog(editor, {
                title: t("update.availableTitle"),
                message: availableMessage(status),
                confirmText: t("update.installNow"),
                cancelText: t("dialog.cancel"),
            });
            if (!confirmed) return;

            setBusy(button, "update.updating");
            const result = await postJson("/minimax/director/update/apply", {
                expectedHead: status.localHead,
                expectedRemoteHead: status.remoteHead,
            });
            let message = t("update.done", {
                from: shortHash(result.from),
                to: shortHash(result.to),
                count: result.commits,
            });
            if (result.dependencyFiles?.length) {
                message += `\n\n${t("update.dependenciesAfter", { files: result.dependencyFiles.join(", ") })}`;
            }
            setBusy(button, "");
            await showMessage(editor, message);
        } catch (error) {
            console.error("[MiniMax H3 Director] update:", error);
            setBusy(button, "");
            await showMessage(editor, reasonText(error.reason, error.data || { message: error.message }));
        } finally {
            setBusy(button, "");
        }
    };
}
