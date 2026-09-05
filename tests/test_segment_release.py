"""Regression tests for the 「分段导出」(segments export) mode.

Background
----------
In「分段导出」the node's ``images`` output is a list of per-segment IMAGE
batches (``segment_outputs``). Downstream ``VHS_VideoCombine`` reads the
first batch and encodes it to mp4 — a 1-frame poster there produced a
0-second clip, which is the bug this guard prevents.

Fix (revised for memory)
------------------------
``_release_segment_pixels`` gained a ``keep_segment_outputs`` flag. During
the run, a finished segment's IMAGE slot drops back to a 1-frame
placeholder **only when its disk cache is durable**
(``segment_frames_cache_exists``) — so peak RAM during sampling stays at
the rolling continuity working set instead of the whole timeline. Cache-
less segments (single-shot, no refine/continuity) keep full frames
resident via ``keep_segment_outputs=True`` because there is nothing to
rehydrate from.

At finalization, the ``if export_segments_mode:`` block rehydrates every
placeholder slot from ``load_segment_cache`` (pre-refine slots from
``load_first_pass_frames_stale``) before the node returns, so downstream
VHS_VideoCombine receives full-length clips while the high-RAM window is
just the final rehydrate.

These tests scan the source tree to make sure no call site regresses to
the old behaviors (unconditional poster release, unconditional keep, or a
missing finalization rehydrate). They run with stdlib + project source
alone — no ComfyUI runtime, no pytest required.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXECUTOR = ROOT / "director" / "executor_core.py"


def _read_executor() -> str:
    return EXECUTOR.read_text(encoding="utf-8")


class ReleaseSegmentPixelsSignature(unittest.TestCase):
    """The function must accept the flag with the right default."""

    def test_function_signature_has_keep_segment_outputs_param(self) -> None:
        src = _read_executor()
        # Match the function signature line(s); allow multi-line defs.
        m = re.search(
            r"def\s+_release_segment_pixels\s*\((?P<body>.*?)\)\s*->\s*bool\s*:",
            src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            m, "_release_segment_pixels definition not found in executor_core.py"
        )
        body = m.group("body")
        self.assertRegex(
            body,
            r"keep_segment_outputs\s*:\s*bool\s*=\s*False",
            "_release_segment_pixels must accept "
            "'keep_segment_outputs: bool = False' so callers can opt in",
        )

    def test_default_is_false(self) -> None:
        """The default must remain False so existing callers are unchanged.

        「全部导出」/ streaming concat still rely on the old behavior of
        replacing the IMAGE-list slot with a 1-frame poster after the
        segment's pixels have been merged.
        """
        src = _read_executor()
        m = re.search(
            r"def\s+_release_segment_pixels\s*\((?P<body>.*?)\)\s*->\s*bool\s*:",
            src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(m)
        self.assertIn(
            "= False",
            m.group("body"),
            "Default of keep_segment_outputs must be False to preserve "
            "「全部导出」 behavior",
        )


class ExportSegmentsModeReleaseIsCacheGuarded(unittest.TestCase):
    """Every `_release_segment_pixels` call inside an `export_segments_mode`
    branch MUST pass ``keep_segment_outputs=`` explicitly AND gate the
    poster release on ``segment_frames_cache_exists`` — releasing to a
    placeholder without a disk cache would regress the 0-second mp4 bug,
    while keeping full frames unconditionally would regress the run-time
    RAM footprint.

    We scan the file linearly and track whether we're inside such a branch
    by looking at the enclosing ``if`` lines, so a single regression in any
    of the call sites fails this test immediately.
    """

    @staticmethod
    def _call_sites_with_branch_context(src: str):
        """Yield (line_no, window_text, in_export_segments_branch).

        ``window_text`` spans 12 lines before to 14 lines after the call
        so the guard (``can_rehydrate = ... segment_frames_cache_exists``)
        computed just above the call is visible.
        """
        # Track the most recent `if export_segments_mode` (or `elif
        # export_segments_mode`) we entered without leaving. We treat the
        # block as "in export_segments_mode" until indentation drops back
        # to the same level as the `if`.
        lines = src.splitlines()
        in_seg = False
        if_indent: int | None = None
        for line_no, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            opens_seg = bool(
                re.search(r"^if\s+export_segments_mode\b", stripped)
                or re.search(r"^elif\s+export_segments_mode\b", stripped)
            )
            if opens_seg:
                in_seg = True
                if_indent = indent
                continue
            # Any non-blank, non-comment line at indent <= the opening `if`
            # closes the branch (next sibling/outer block).
            if (
                in_seg
                and if_indent is not None
                and stripped
                and not stripped.startswith("#")
                and indent <= if_indent
                and not opens_seg
            ):
                in_seg = False
                if_indent = None

            if "_release_segment_pixels(" in stripped:
                start = max(0, line_no - 13)
                end = min(line_no + 14, len(lines))
                yield line_no, "\n".join(lines[start:end]), in_seg

    def test_every_export_segments_call_is_cache_guarded(self) -> None:
        src = _read_executor()
        offenders: list[str] = []
        for line_no, window, in_seg in self._call_sites_with_branch_context(src):
            if not in_seg:
                continue
            # Checks run on code only — full-line comments legitimately
            # explain the `keep_segment_outputs=True` fallback semantics.
            code = "\n".join(
                l for l in window.splitlines() if not l.lstrip().startswith("#")
            )
            problems = []
            if "keep_segment_outputs=" not in code:
                problems.append("missing explicit `keep_segment_outputs=`")
            if "segment_frames_cache_exists" not in code:
                problems.append(
                    "no `segment_frames_cache_exists` guard — placeholder "
                    "release must be conditional on a durable disk cache"
                )
            if "keep_segment_outputs=True" in code:
                problems.append(
                    "unconditional `keep_segment_outputs=True` — keeps the "
                    "whole timeline in RAM for the entire run"
                )
            if problems:
                ctx = "\n".join(
                    f"{i + 1:>4}: {l}" for i, l in enumerate(window.splitlines())
                )
                offenders.append(
                    f"  Line {line_no}: _release_segment_pixels inside "
                    f"`if export_segments_mode:` — "
                    + "; ".join(problems)
                    + f".\n  {ctx}"
                )
        self.assertFalse(
            offenders,
            "Found _release_segment_pixels call(s) that would regress the "
            "「分段导出」 fix (0-second mp4 or run-time RAM spike):\n\n"
            + "\n\n".join(offenders),
        )


class ExportSegmentsModeRehydratesAtFinalization(unittest.TestCase):
    """The finalization block must rehydrate placeholder slots from the
    disk cache before the node returns — otherwise the cache-guarded
    release above would still ship 1-frame placeholders downstream."""

    @staticmethod
    def _finalization_block(src: str) -> str:
        """Find the *finalization* `if export_segments_mode:` block — the
        one near the bottom of the main loop, just before `elif
        stream_all_mode:`. Earlier blocks (inside the per-segment loop)
        legitimately deal with per-call placeholder semantics.
        """
        # Anchor on the unique "Export mode: segments" report text written
        # only by the finalization block, then walk backwards to the start
        # of its enclosing `if export_segments_mode:` line.
        anchor = src.find('"Export mode: segments —')
        self_assert_msg = (
            "Could not locate the export_segments_mode finalization "
            "report (expected near the bottom of execute_director_plan_core)."
        )
        assert anchor != -1, self_assert_msg  # type: ignore[unreachable]
        # Walk back to the `if export_segments_mode:` that opens this block.
        prefix = src[:anchor]
        # The finalization block is indented with 4 spaces; the inner
        # loop's `if export_segments_mode:` is indented with 8 spaces.
        # Search for the OUTERMOST matching `if export_segments_mode:`
        # before the anchor (no other enclosing `if export_segments_mode:`).
        marker = "\n    if export_segments_mode:\n"
        idx = prefix.rfind(marker)
        assert idx != -1, (  # type: ignore[unreachable]
            "Could not walk back to finalization 'if export_segments_mode:'"
        )
        # The block ends at the next `elif stream_all_mode:` or `else:` at
        # the same indent (4 spaces).
        start = idx + 1  # skip leading newline
        rest = src[start:]
        # Find the next sibling at 4-space indent.
        end_match = re.search(
            r"\n    elif stream_all_mode:\n|\n    else:\n",
            rest,
        )
        if end_match is None:
            body = rest
        else:
            body = rest[: end_match.start() + 1]
        return body

    def test_finalization_rehydrates_from_disk_cache(self) -> None:
        src = _read_executor()
        body = self._finalization_block(src)
        self.assertIn(
            "load_segment_cache(",
            body,
            "export_segments_mode finalization no longer rehydrates "
            "placeholder IMAGE slots from the disk cache — downstream "
            "VHS_VideoCombine would encode 0-second clips again.",
        )
        self.assertIn(
            "segment_outputs[pos] = loaded",
            body,
            "rehydrated frames must be written back into segment_outputs "
            "(the node's per-segment IMAGE output).",
        )

    def test_finalization_rehydrates_pre_refine_slots(self) -> None:
        src = _read_executor()
        body = self._finalization_block(src)
        self.assertIn(
            "load_first_pass_frames_stale(",
            body,
            "pre-refine IMAGE slots must also rehydrate (first-pass cache, "
            "falling back to the final frames).",
        )

    def test_report_text_does_not_claim_1_frame_poster(self) -> None:
        src = _read_executor()
        body = self._finalization_block(src)
        self.assertNotIn(
            "1-frame poster",
            body,
            "Finalization report text still mentions '1-frame poster' "
            "under export_segments_mode — downstream VHS_VideoCombine "
            "would encode a 0-second clip again.",
        )

    def test_report_text_mentions_full_frames(self) -> None:
        src = _read_executor()
        body = self._finalization_block(src)
        self.assertRegex(
            body,
            r"keeps?\b.*\bframes?\b|full-length",
            "export_segments_mode finalization report must explain that "
            "IMAGE list keeps every segment's full frames so downstream "
            "consumers get full-length clips.",
        )


class DirectorCommonReportText(unittest.TestCase):
    """The finalize report must match the new semantics."""

    def test_finalize_report_does_not_say_1_frame_poster(self) -> None:
        common = (ROOT / "nodes" / "director_common.py").read_text(encoding="utf-8")
        # Only check the segments branch of the report block.
        m = re.search(
            r'if export_segments and len\(segment_outputs\) > 1:\s*\n(?P<body>.*?)(?=        if plan\.run_indices|\Z)',
            common,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(m, "finalize segments report block not found")
        body = m.group("body")
        self.assertNotIn(
            "1-frame poster",
            body,
            "finalize_director_outputs still tells the user that "
            "released clips are 1-frame posters — contradicts the fix.",
        )
        self.assertRegex(
            body,
            r"full frames|full-length",
            "finalize_director_outputs must explain per-segment IMAGE "
            "batches keep full frames.",
        )


if __name__ == "__main__":
    # Allow running without pytest: `python tests/test_segment_release.py`.
    sys.exit(unittest.main(verbosity=2).exitstatus)
