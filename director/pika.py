"""Pika H3 Flex task adapters and released local reference limits."""

from .segment_loras import resolve_pika_task_loras


def pika_sigmas():
    import torch

    knots = torch.tensor([1000, 750, 500, 250, 100, 40, 0], dtype=torch.float32) / 1000
    return 12 * knots / (1 + 11 * knots)


def configure_pika_plan(plan, *, materialize_media=True):
    if abs(float(plan.frame_rate) - 24.0) > 1e-6:
        raise ValueError("Pika H3 Flex requires 24 fps.")
    for seg in plan.segments:
        if seg.source_video:
            continue
        if seg.task_key not in {"t2v", "i2v", "fl2v", "r2v"}:
            raise ValueError(f"Pika H3 Flex does not publish task '{seg.task_key}'. Choose text, endpoint frames or multimodal references.")
        seg.loras = resolve_pika_task_loras(seg.loras, seg.task_key)
        if not materialize_media:
            continue
        import folder_paths

        folder_paths.get_full_path_or_raise("loras", seg.loras[0]["name"])
        if seg.task_key != "r2v":
            continue
        images, videos, audios = seg.refs, seg.ref_videos, seg.ref_audios
        if not images and not videos:
            raise ValueError("Pika Ref2VA requires at least one image or video reference.")
        if len(images) > 9 or len(videos) > 3 or len(audios) > 3 or len(images) + len(videos) + len(audios) > 12:
            raise ValueError("Pika Ref2VA accepts at most 9 images, 3 videos, 3 standalone audio clips and 12 references total.")
        durations = [len(video.tensor) / float(plan.frame_rate) for video in videos]
        if any(d < 2 or d > 15 for d in durations) or sum(durations) > 15:
            raise ValueError("Pika video references must each be 2–15 seconds, with at most 15 seconds total.")
        from .plan import ref_audios_to_dict

        pcm = ref_audios_to_dict(audios, cache=getattr(plan, "audio_decode_cache", None)) or {}
        durations = [audio["waveform"].shape[-1] / float(audio["sample_rate"]) for audio in pcm.values()]
        if any(d < 2 or d > 15 for d in durations) or sum(durations) > 15:
            raise ValueError("Pika standalone audio references must each be 2–15 seconds, with at most 15 seconds total.")
