"""User-triggered, fast-forward-only updates for this Director checkout."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from aiohttp import web
from server import PromptServer


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
_GIT_TIMEOUT_SEC = 60
_UPDATE_LOCK = asyncio.Lock()


class GitUpdateError(RuntimeError):
    pass


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


async def _git(*args: str) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "core.quotepath=false",
            "-c",
            "i18n.logOutputEncoding=utf-8",
            "-C",
            str(PLUGIN_ROOT),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except FileNotFoundError as exc:
        raise GitUpdateError("git_not_found") from exc

    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=_GIT_TIMEOUT_SEC)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise GitUpdateError("git_timeout") from exc

    text = output.decode("utf-8", errors="replace").strip()
    if process.returncode:
        detail = text[-2000:] if text else f"git exited with code {process.returncode}"
        raise GitUpdateError(detail)
    return text


def _queue_counts() -> tuple[int, int]:
    server = getattr(PromptServer, "instance", None)
    if server is None or server.prompt_queue is None:
        return 0, 0
    running, pending = server.prompt_queue.get_current_queue_volatile()
    return len(running), len(pending)


def _unsupported(reason: str, message: str = "") -> dict:
    running, pending = _queue_counts()
    return {
        "supported": False,
        "canUpdate": False,
        "updateAvailable": False,
        "reason": reason,
        "message": message,
        "queueRunning": running,
        "queuePending": pending,
    }


async def inspect_update(*, fetch: bool) -> dict:
    try:
        checkout = Path(await _git("rev-parse", "--show-toplevel"))
    except GitUpdateError as exc:
        reason = str(exc)
        if reason != "git_not_found":
            reason = "not_git_checkout"
        return _unsupported(reason, str(exc))

    if not _same_path(checkout, PLUGIN_ROOT):
        return _unsupported("not_git_checkout")

    try:
        branch = await _git("symbolic-ref", "--quiet", "--short", "HEAD")
    except GitUpdateError:
        return _unsupported("detached_head")

    try:
        upstream = await _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    except GitUpdateError:
        return _unsupported("no_upstream")

    remote, separator, remote_branch = upstream.partition("/")
    if (
        not separator
        or remote_branch != branch
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", remote)
    ):
        return _unsupported("unsupported_upstream", upstream)

    try:
        remote_url = await _git("remote", "get-url", remote)
        clean = not bool(await _git("status", "--porcelain"))
        if fetch:
            await _git("fetch", "--quiet", "--no-tags", "--prune", remote)
        local_head = await _git("rev-parse", "HEAD")
        remote_head = await _git("rev-parse", upstream)
        counts = await _git("rev-list", "--left-right", "--count", f"HEAD...{upstream}")
        ahead_text, behind_text = counts.split()
        ahead = int(ahead_text)
        behind = int(behind_text)
    except (GitUpdateError, ValueError) as exc:
        return _unsupported("fetch_failed", str(exc))

    changed_files: list[str] = []
    commits: list[dict[str, str]] = []
    if behind and not ahead:
        changed = await _git("diff", "--name-only", f"HEAD..{upstream}")
        changed_files = [line for line in changed.splitlines() if line]
        log_output = await _git(
            "log",
            "--max-count=5",
            "--format=%h%x09%s",
            f"HEAD..{upstream}",
        )
        for line in log_output.splitlines():
            short_hash, _, subject = line.partition("\t")
            commits.append({"hash": short_hash, "subject": subject})

    running, pending = _queue_counts()
    reason = "current"
    can_update = clean and ahead == 0
    update_available = can_update and behind > 0
    if not clean:
        reason = "dirty_worktree"
    elif ahead and behind:
        reason = "diverged_history"
    elif ahead:
        reason = "local_commits"
    elif behind:
        reason = "update_available"

    dependency_files = [
        name for name in changed_files
        if name in {"requirements.txt", "pyproject.toml"}
    ]
    return {
        "supported": True,
        "canUpdate": can_update,
        "updateAvailable": update_available,
        "reason": reason,
        "branch": branch,
        "remote": remote,
        "upstream": upstream,
        "remoteUrl": remote_url,
        "clean": clean,
        "localHead": local_head,
        "remoteHead": remote_head,
        "ahead": ahead,
        "behind": behind,
        "commits": commits,
        "changedFiles": changed_files,
        "dependencyFiles": dependency_files,
        "queueRunning": running,
        "queuePending": pending,
    }


async def minimax_check_update(request):
    del request
    status = await inspect_update(fetch=True)
    return web.json_response(status)


async def minimax_apply_update(request):
    try:
        body = await request.json()
    except Exception as exc:
        return web.json_response(
            {"reason": "invalid_request", "message": str(exc)},
            status=400,
        )

    expected_head = str(body.get("expectedHead") or "").strip()
    expected_remote_head = str(body.get("expectedRemoteHead") or "").strip()
    if not expected_head or not expected_remote_head:
        return web.json_response({"reason": "invalid_request"}, status=400)

    async with _UPDATE_LOCK:
        running, pending = _queue_counts()
        if running or pending:
            return web.json_response(
                {
                    "reason": "queue_busy",
                    "queueRunning": running,
                    "queuePending": pending,
                },
                status=409,
            )

        status = await inspect_update(fetch=True)
        if not status.get("supported") or not status.get("canUpdate"):
            return web.json_response(status, status=409)
        if status.get("localHead") != expected_head:
            return web.json_response({"reason": "state_changed"}, status=409)
        if status.get("remoteHead") != expected_remote_head:
            return web.json_response({"reason": "remote_changed"}, status=409)
        if not status.get("updateAvailable"):
            return web.json_response(status, status=409)

        running, pending = _queue_counts()
        if running or pending:
            return web.json_response(
                {
                    "reason": "queue_busy",
                    "queueRunning": running,
                    "queuePending": pending,
                },
                status=409,
            )

        before = status["localHead"]
        try:
            await _git("merge", "--ff-only", status["upstream"])
            after = await _git("rev-parse", "HEAD")
        except GitUpdateError as exc:
            return web.json_response(
                {"reason": "update_failed", "message": str(exc)},
                status=409,
            )

        return web.json_response(
            {
                "updated": True,
                "from": before,
                "to": after,
                "commits": status["behind"],
                "dependencyFiles": status["dependencyFiles"],
                "restartRequired": True,
            }
        )
