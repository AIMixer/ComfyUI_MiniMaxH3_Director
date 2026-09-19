"""Graph packer: Semantic Bridge for MiniMax H3 Director.semantic_bridge."""

from __future__ import annotations

from ..director.semantic_bridge import (
    DEFAULT_ALPHA,
    MMX_DIR_SEMANTIC_BRIDGE,
    list_semantic_bridge_adapters,
    pack_semantic_bridge,
)

_CATEGORY = "MiniMaxH3"


class MiniMaxH3DirectorSemanticBridge:
    """Pack Semantic Bridge settings. Connect ``semantic_bridge`` to Director.

    Unconnected Director is unchanged. When wired, each segment's official
    CONDITIONING token tensor is rewritten with the published student MLP
    (RMS-norm → 5120→512→512→5120 → magnitude match → residual mix).

    Distilled on FL2VA (t2v / i2v / fl2v). r2v / v2v / rv2v are forced-compat
    with a report warning — the original author did not distill Ref2VA.
    Does not sample by itself. Adapter weights are user-provided.
    """

    @classmethod
    def INPUT_TYPES(cls):
        adapters = list_semantic_bridge_adapters()
        default_adapter = adapters[0] if adapters else ""
        return {
            "required": {
                "adapter": (
                    adapters,
                    {
                        "default": default_adapter,
                        "tooltip": (
                            "Student MLP 权重（约 11MB）。放到 "
                            "ComfyUI/models/semantic_bridge/。"
                            "原版蒸馏于 FL2VA；不随本插件分发。"
                        ),
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": DEFAULT_ALPHA,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "残差混合强度。C = H + alpha*(S'−H)。"
                            "原版工作流默认 0.15。"
                        ),
                    },
                ),
                "magnitude_match": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "把 student 输出的向量模长对齐到原 hidden。"
                            "原版默认开。"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (MMX_DIR_SEMANTIC_BRIDGE,)
    RETURN_NAMES = ("semantic_bridge",)
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Connect to Director.semantic_bridge (above selflift). "
        "Rewrites official cond tokens with the Semantic Bridge student. "
        "Unconnected Director is identical. Ref2VA is forced-compat — use with care."
    )

    def pack(self, adapter, alpha=DEFAULT_ALPHA, magnitude_match=True, **kwargs):
        del kwargs
        return (
            pack_semantic_bridge(
                adapter=adapter,
                alpha=alpha,
                magnitude_match=magnitude_match,
            ),
        )
