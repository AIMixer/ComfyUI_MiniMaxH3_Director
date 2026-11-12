"""Apply the Director face-refine pass to an existing IMAGE video batch."""
from comfy_api.latest import io

from ..director.face_refine.pack import MMX_DIR_FACE_REFINE, normalize_face_refine_pack
from ..director.face_refine.runtime import refine_face_video


class MiniMaxH3FaceRefineVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineVideo",
            display_name="H3 已有视频脸部精修",
            category="MiniMaxH3",
            description="复用导演台FaceRefine处理已有视频帧。原音轨通过原生视频节点接回，不重新生成。",
            inputs=[
                io.Image.Input("images"),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Custom(MMX_DIR_FACE_REFINE).Input("face_refine"),
                io.Image.Input("reference_image", optional=True),
                io.String.Input("prompt", optional=True, multiline=True, default=""),
                io.Int.Input("seed", optional=True, default=42, min=0, max=0xffffffffffffffff),
                io.Float.Input("cfg", optional=True, default=1.0, min=0.0, max=20.0),
                io.Float.Input("frame_rate", optional=True, default=24.0, min=0.01, max=240.0),
                io.Boolean.Input("clear_vram", optional=True, default=True),
            ],
            outputs=[io.Image.Output(display_name="修脸后"), io.String.Output(display_name="修复报告")],
        )

    @classmethod
    def execute(cls, images, model, clip, vae, audio_vae, face_refine,
                reference_image=None, prompt="", seed=42, cfg=1.0,
                frame_rate=24.0, clear_vram=True):
        pack = normalize_face_refine_pack(face_refine)
        if pack is None:
            raise ValueError("请连接 MiniMax H3 Director FaceRefine 设置。")
        if reference_image is not None:
            pack["reference_image"] = reference_image
        if prompt.strip():
            pack["prompt"] = prompt.strip()
        if clear_vram:
            from ..director.vram_cleanup import cleanup_segment_vram
            cleanup_segment_vram(enabled=True, unload_models=True)
        result, report = refine_face_video(
            frames=images, pack=pack, model=model, vae=vae, audio_vae=audio_vae,
            clip=clip, seed=seed, cfg=cfg,
        )
        return io.NodeOutput(result, f"{len(images)}帧 / {frame_rate:g}fps / {len(images)/frame_rate:.3f}秒; {report}")
