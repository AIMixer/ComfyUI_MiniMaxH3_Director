"""调色台（MiniMaxH3Grade）分段副载（自 v52.x 精简移植）。

v8 基线没有漂移统计系统——这里只保留调色台需要的真实分段写/读
（pin 相位对齐裁切后的每段帧数），供同轮运行直接读取。
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.drift_stats")

_GRADE_BOUNDS_FILE = "minimax_h3_grade_bounds.json"


def _grade_cache_dir() -> str:
    """与 director/grade_core 同目录：ComfyUI output\\temp（部署更新不丢失）。"""
    try:
        import folder_paths

        base = os.path.join(folder_paths.get_output_directory(), "temp")
        os.makedirs(base, exist_ok=True)
        return base
    except Exception:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bounds_path() -> str:
    return os.path.join(_grade_cache_dir(), _GRADE_BOUNDS_FILE)


def write_grade_bounds(total_frames: int, segments: list[int]) -> None:
    """导演台导出合并视频时写盘真实每段帧数（pin 相位对齐裁切后）。

    供调色台（MiniMaxH3Grade）同一轮运行直接读取，无需前端往返。
    只读/不可写安装时静默跳过（不影响渲染）。
    """
    try:
        with open(_bounds_path(), "w", encoding="utf-8") as fh:
            json.dump(
                {"frames": int(total_frames), "segments": [int(s) for s in segments]},
                fh,
                ensure_ascii=False,
            )
    except Exception as exc:
        log.info("grade-bounds 写盘失败（%s），调色台将回退时间轴分段。", exc)


def read_grade_bounds() -> dict | None:
    """读回导演台最近一次写盘的真实分段；不存在/损坏返回 None。"""
    try:
        path = _bounds_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None
