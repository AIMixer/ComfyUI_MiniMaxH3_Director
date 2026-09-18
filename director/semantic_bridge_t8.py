"""T8 语义桥（Semantic Bridge）集成桥（2026-09-18）

把 T8 minimax-h3-audio 包的语义桥接入导演台 conditioning 链。
语义桥在「官方 cond 编码完成后、采样开始前」应用（每段一次，positive のみ）——
即官方推荐接法（原生条件编码之后、采样之前）。

实现原则（与 prompt_enhance_t8_bridge 相同）：
- import 复用而非拷贝：T8 包更新后能力自然流入。
- 惰性加载：不在 import 期做模型/GPU 工作。
- 任何失败安静降级：桥不可用时导演台照常跑（原因写入报告）。

硬性前提：T8 minimax-h3-audio 包存在 + models/semantic_bridge 下已安装桥模型。
模型下载：https://huggingface.co/t8star/Semantic-Bridge-Comfy
（保留 models/semantic_bridge/t8_compat 子目录结构；不会自动下载。）
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
import threading

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

_T8_AUDIO_DIR = (
    Path(__file__).resolve().parent.parent.parent / "comfyui-minimax-h3-audio-T8-main"
)
_core_mod = None
_core_lock = threading.Lock()

# 节点下拉的「关闭」选项。
OFF = "off"


def _load_core():
    """惰性加载 T8 audio 包的桥核心实现。

    h3_t8/semantic_bridge.py 不含包内相对导入，可独立按文件加载，
    避免把整个 T8 audio 包拖进 sys.modules（其 __init__ 会注册全部节点）。
    """
    global _core_mod
    with _core_lock:
        if _core_mod is not None:
            return _core_mod
        src = _T8_AUDIO_DIR / "h3_t8" / "semantic_bridge.py"
        if not src.is_file():
            raise RuntimeError(f"T8 audio package not found: {_T8_AUDIO_DIR}")
        spec = importlib.util.spec_from_file_location("t8_semantic_bridge_core", src)
        mod = importlib.util.module_from_spec(spec)
        # Python 3.13 dataclasses 在 exec 期需要模块已注册进 sys.modules。
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        _core_mod = mod
        log.info("T8 semantic bridge core loaded from %s", src)
        return mod


def _model_roots():
    """models/semantic_bridge 的搜索根（含 extra_model_paths 注册项）。"""
    import folder_paths

    default = Path(folder_paths.models_dir) / "semantic_bridge"
    roots = [
        Path(path)
        for path in folder_paths.folder_names_and_paths.get(
            "semantic_bridge", ([], set())
        )[0]
    ]
    if default not in roots:
        roots.append(default)
    return roots


def bridge_model_options():
    """枚举已安装的桥模型相对名（供节点下拉；与 T8 官方 Config 节点同口径）。"""
    try:
        names = set()
        for root in _model_roots():
            if root.is_dir():
                for path in sorted(root.rglob("*.safetensors")):
                    if ".cache" not in path.relative_to(root).parts:
                        names.add(path.relative_to(root).as_posix())
        return sorted(names)
    except Exception as exc:  # noqa: BLE001 - 扫描失败只影响下拉，不阻塞加载
        log.warning("semantic bridge model scan failed: %s", exc)
        return []


def resolve_bridge_model(name):
    """桥模型相对名 → 绝对路径。"""
    for root in _model_roots():
        candidate = root / name
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"Semantic Bridge model not found: {name}. "
        "Install an author .safetensors under models/semantic_bridge and refresh."
    )


def bridge_active(model_name, alpha):
    """桥是否应当激活（与 T8 官方语义一致：enabled 且 alpha != 0）。"""
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(model_name, str)
        and model_name.strip() not in ("", OFF)
        and alpha != 0.0
    )


def apply_semantic_bridge(positive, *, model_name, alpha, device="auto"):
    """在导演台 cond 上应用语义桥。

    返回 (positive, report)。激活且成功时返回增强后的 cond；
    任何前置缺失（T8 包 / 桥模型未装）或运行失败都安静降级为原 cond，
    并把原因写进 report（导演台运行报告可见）。
    """
    if not bridge_active(model_name, alpha):
        return positive, {"enabled": False, "applied": False}
    try:
        core = _load_core()
    except Exception as exc:  # noqa: BLE001 - 包缺失属预期降级路径
        return positive, {
            "enabled": True,
            "applied": False,
            "error": f"T8 audio package unavailable: {exc}",
        }
    try:
        path = resolve_bridge_model(model_name)
    except Exception as exc:  # noqa: BLE001 - 模型缺失属预期降级路径
        return positive, {"enabled": True, "applied": False, "error": str(exc)}
    try:
        config = core.BridgeConfig(
            path=path,
            sha256=core.file_sha(path),
            alpha=float(alpha),
            magnitude_match="per_token",
            token_scope="all_tokens",
            device=device,
            chunk_tokens=256,
            enabled=True,
        )
        result, report = core.apply_bridge(
            positive, config, encoding_source="director"
        )
        report = dict(report or {})
        report["enabled"] = True
        report["applied"] = bool(report.get("applied", True))
        report["model"] = model_name
        report["alpha"] = float(alpha)
        return result, report
    except Exception as exc:  # noqa: BLE001 - 失败降级为原 cond，导演台照常跑
        log.warning("semantic bridge apply failed: %s", exc)
        return positive, {
            "enabled": True,
            "applied": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
