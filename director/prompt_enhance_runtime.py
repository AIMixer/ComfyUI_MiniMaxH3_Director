"""Runtime prompt enhancement hooks for MiniMax H3 Director executors."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from ..lib.prompt_enhance_templates import OUTPUT_LANGUAGE_EN, OUTPUT_LANGUAGE_ZH
from ..lib.prompt_enhancer import (
    DEFAULT_API_FORMAT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OPENAI_COMPAT_MODE,
    coerce_llm_model,
    coerce_llm_url,
    enhance_prompt_sync,
    infer_api_format,
    normalize_openai_compat_mode,
)
from .prompt_enhance_media import collect_segment_vision_b64

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")


@dataclass(frozen=True)
class PromptEnhanceSettings:
    auto_enhance: bool = False
    api_format: str = DEFAULT_API_FORMAT
    openai_compat_mode: str = DEFAULT_OPENAI_COMPAT_MODE
    url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_OLLAMA_MODEL
    api_key: str = ""
    unload_after: bool = False
    custom_template: str = ""
    output_language: str = OUTPUT_LANGUAGE_ZH
    character_feature_enhance: bool = False

    @classmethod
    def from_node(
        cls,
        *,
        llm_auto_enhance=False,
        llm_api_format=DEFAULT_API_FORMAT,
        llm_openai_compat_mode=DEFAULT_OPENAI_COMPAT_MODE,
        llm_url=DEFAULT_OLLAMA_URL,
        llm_api_key="",
        llm_model=DEFAULT_OLLAMA_MODEL,
        llm_unload_after=False,
        llm_custom_template="",
        llm_output_language=OUTPUT_LANGUAGE_ZH,
        llm_character_feature_enhance=False,
    ) -> PromptEnhanceSettings:
        from ..lib.prompt_enhance_templates import normalize_character_feature_enhance

        fmt = str(llm_api_format or DEFAULT_API_FORMAT)
        feature_enhance = normalize_character_feature_enhance(llm_character_feature_enhance)
        return cls(
            auto_enhance=bool(llm_auto_enhance),
            api_format=fmt,
            openai_compat_mode=normalize_openai_compat_mode(llm_openai_compat_mode),
            url=coerce_llm_url(llm_url),
            model=coerce_llm_model(llm_model),
            api_key=str(llm_api_key or "").strip(),
            unload_after=bool(llm_unload_after),
            custom_template=str(llm_custom_template or ""),
            output_language=str(llm_output_language or OUTPUT_LANGUAGE_ZH),
            character_feature_enhance=feature_enhance,
        )

    @property
    def active(self) -> bool:
        return self.auto_enhance and bool(self.model)


def maybe_enhance_segment_prompt(
    settings: PromptEnhanceSettings,
    *,
    task_type: str,
    user_prompt: str,
    source_clip: torch.Tensor | None = None,
    refs=None,
    reference_video: torch.Tensor | None = None,
    use_vision: bool = True,
) -> str:
    """Return enhanced prompt or original when enhancement is disabled / fails."""
    text = (user_prompt or "").strip()
    if not settings.active or not text:
        return user_prompt

    images_b64: list[str] | None = None
    ref_count = max(1, len(refs or []))
    ref_slots: list[int] = []
    source_count = 0
    ref_video_count = 0
    if use_vision:
        images_b64, ref_count, source_count, ref_slots = collect_segment_vision_b64(
            source_clip=source_clip,
            refs=refs,
            reference_video=reference_video,
        )
        if images_b64:
            ref_video_count = max(
                0,
                len(images_b64) - source_count - ref_count,
            )
        else:
            images_b64 = None

    api_format = infer_api_format(settings.url, settings.api_format)
    enhanced, err = enhance_prompt_sync(
        task_type=task_type,
        user_prompt=text,
        url=settings.url,
        model=settings.model,
        api_format=api_format,
        openai_compat_mode=settings.openai_compat_mode,
        api_key=settings.api_key,
        images_b64=images_b64,
        image_num=ref_count,
        custom_template=settings.custom_template,
        output_language=settings.output_language,
        character_feature_enhance=settings.character_feature_enhance,
        vision_source_count=source_count if images_b64 else None,
        ref_slots=ref_slots if ref_slots else None,
        vision_ref_video_count=ref_video_count,
        unload_after=settings.unload_after,
    )
    if enhanced:
        from ..lib.task_prompts import resolve_task_key
        from .fl2v_timeline import fl2v_prompt_body_only

        # Official ImageToVideo uses the PE body as-is; strip leftover lock wraps.
        if resolve_task_key(task_type) in {"fl2v", "i2v"}:
            enhanced = fl2v_prompt_body_only(enhanced) or enhanced
        log.info(
            "Director prompt enhanced (%s, %d chars, vision=%s)",
            task_type,
            len(enhanced),
            bool(images_b64),
        )
        return enhanced
    log.warning("Director auto-enhance failed: %s", err or "unknown error")
    return user_prompt


def notify_prompt_enhanced(
    node_id: str | None,
    *,
    text: str,
    segment_index: int | None = None,
    field: str = "global",
) -> None:
    if not node_id:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(
            "minimax_director_enhanced",
            {
                "node": node_id,
                "text": text,
                "segment_index": segment_index,
                "field": field,
            },
        )
    except Exception:
        pass
