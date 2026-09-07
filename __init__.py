"""ComfyUI MiniMax H3 Director — timeline plugin for MiniMax-H3 AV generation.

Based on ComfyUI official MiniMax H3 support (PR #15224 / #15228).
Licensed under the Apache License, Version 2.0. See LICENSE.
"""

from .nodes.conditioning import (
    MiniMaxH3DirectorConditioning,
    MiniMaxH3DirectorPlannerConditioning,
)
from .nodes.director import MiniMaxH3Director
from .nodes.director_refine import MiniMaxH3DirectorRefine
from .nodes.director_groups import (
    MiniMaxH3DirectorGroupImageToVideo,
    MiniMaxH3DirectorGroupReferenceToVideo,
    MiniMaxH3DirectorGroupsCombine,
)
from .nodes.grade_node import MiniMaxH3Grade
from .nodes.video_saver import MiniMaxH3VideoSaver

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Director": MiniMaxH3Director,
    "MiniMaxH3DirectorRefine": MiniMaxH3DirectorRefine,
    # Legacy type id kept so older workflows still load.
    "ComfyMiniMaxH3Director": MiniMaxH3Director,
    "MiniMaxH3DirectorConditioning": MiniMaxH3DirectorConditioning,
    "MiniMaxH3DirectorPlannerConditioning": MiniMaxH3DirectorPlannerConditioning,
    "MiniMaxH3DirectorGroupImageToVideo": MiniMaxH3DirectorGroupImageToVideo,
    "MiniMaxH3DirectorGroupReferenceToVideo": MiniMaxH3DirectorGroupReferenceToVideo,
    # Must stay in NODE_CLASS_MAPPINGS: ComfyUI skips comfy_entrypoint when
    # NODE_CLASS_MAPPINGS is present (if/elif in load_custom_node).
    "MiniMaxH3DirectorGroupsCombine": MiniMaxH3DirectorGroupsCombine,
    "MiniMaxH3Grade": MiniMaxH3Grade,
    "MiniMaxH3VideoSaver": MiniMaxH3VideoSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Director": "MiniMaxH3Director",
    "MiniMaxH3DirectorRefine": "MiniMax H3 Director Refine",
    "ComfyMiniMaxH3Director": "MiniMaxH3Director",
    "MiniMaxH3DirectorConditioning": "MiniMax H3 Director Conditioning",
    "MiniMaxH3DirectorPlannerConditioning": "MiniMax H3 Director Planner Conditioning",
    "MiniMaxH3DirectorGroupImageToVideo": "MiniMax H3 Director Group (Image to Video)",
    "MiniMaxH3DirectorGroupReferenceToVideo": "MiniMax H3 Director Group (Reference to Video)",
    "MiniMaxH3DirectorGroupsCombine": "MiniMax H3 Director Groups Combine",
    "MiniMaxH3Grade": "MiniMax H3 Grade (调色台)",
    "MiniMaxH3VideoSaver": "MiniMax H3 Video Saver (流式落盘)",
}

WEB_DIRECTORY = "./web/js"

import logging

_log = logging.getLogger("ComfyUI-MiniMaxH3-Director")

try:
    from .director.h3_motion_context import CONTINUITY_PIPELINE_ID, DIRECTOR_RELEASE

    _log.info(
        "MiniMax H3 Director loaded — release %s (pipeline %s)",
        DIRECTOR_RELEASE,
        CONTINUITY_PIPELINE_ID,
    )
except Exception as _release_exc:
    _log.warning("MiniMax H3 Director release info unavailable: %s", _release_exc)

try:
    from .director.http_routes import register_routes as _register_director_routes

    if not _register_director_routes():
        _log.warning(
            "MiniMax H3 Director HTTP routes deferred (PromptServer not ready). "
            "Restart ComfyUI if /minimax/director/* returns 404."
        )
except Exception as _director_routes_exc:
    _log.warning("MiniMax H3 Director HTTP routes failed to load: %s", _director_routes_exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
