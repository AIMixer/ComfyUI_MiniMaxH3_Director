"""Exercise the executor's conditioning setup without a ComfyUI GPU host."""
import ast
import importlib.util
import logging
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('merge_semantic_bridge', ROOT / 'director/semantic_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

@pytest.mark.parametrize('task', ['addguide', 't2v', 'fl2v', 'r2v'])
@pytest.mark.parametrize('enabled', [False, True])
def test_refine_keeps_bridge_but_excludes_timed_guides(task, enabled):
    source = (ROOT / 'director/executor_core.py').read_text(encoding='utf-8')
    begin = source.index('        positive, negative, latent, task_hint = run_minimax_conditioning(')
    end = source.index('        trim_frames = 0', begin)
    block = textwrap.dedent(source[begin:end]).replace('from .semantic_bridge import apply_semantic_bridge', '')
    raw = [[torch.ones(1, 2, 5120), {'stock': True}]]
    def guides(positive, *args, **kwargs):
        return [[positive[0][0], {**positive[0][1], 'timed_guides': True}]]
    env = dict(positive=raw, negative=[], clip=None, positive_prompt='test', vae=None, audio_vae=None,
               ctx_w=64, ctx_h=64, sample_len=22, first_frame=None, last_frame=None,
               ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
               seg=SimpleNamespace(task_key=task, timed_guides=['picture'], timed_audio_guides=['audio'], frame_count=22),
               plan=SimpleNamespace(semantic_bridge={'enabled': True, 'adapter': 'test'} if enabled else None),
               run_minimax_conditioning=lambda **kwargs: (raw, [], {}, 'test'),
               resolve_ref_image_size=lambda *args: 'match', official_ref_image_size=lambda x: x,
               apply_minimax_timed_guides=guides, apply_semantic_bridge=bridge.apply_semantic_bridge,
               time=time, t_cond=time.perf_counter(), log=logging.getLogger(__name__))
    with patch.object(bridge, 'resolve_semantic_bridge_path', return_value='test'), patch.object(bridge, 'load_semantic_student', return_value=object()), patch.object(bridge, '_rewrite_hidden', side_effect=lambda hidden, *args: hidden + 1):
        exec(compile(block, str(ROOT / 'director/executor_core.py'), 'exec'), env)
    for name in ('positive', 'refine_positive'):
        hidden, meta = env[name][0]
        assert torch.all(hidden == (2 if enabled else 1)), name
        assert bool(meta.get(bridge.META_APPLIED_KEY)) == enabled, name
    assert 'timed_guides' not in env['refine_positive'][0][1]
    assert bool(env['positive'][0][1].get('timed_guides')) == (task == 'addguide')


@pytest.mark.parametrize("task", ["addguide", "t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"])
def test_refine_receives_final_upstream_conditioning_except_addguide(task):
    """A later motion-context rewrite must still reach non-AddGuide refine."""
    path = ROOT / "director/executor_core.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_segment_refine"
    )
    keyword = next(kw.value for kw in call.keywords if kw.arg == "positive")
    expression = compile(ast.Expression(body=keyword), str(path), "eval")
    before_guides = [[torch.ones(1, 2, 8), {"mmx_semantic_bridge": True}]]
    final_positive = [[before_guides[0][0], {
        **before_guides[0][1],
        "timed_guides" if task == "addguide" else "motion_context": True,
    }]]
    result = eval(expression, {
        "seg": SimpleNamespace(task_key=task),
        "refine_positive": before_guides,
        "positive": final_positive,
    })
    assert result is (before_guides if task == "addguide" else final_positive)
    assert "timed_guides" not in result[0][1]
    assert result[0][1]["mmx_semantic_bridge"]
    if task != "addguide":
        assert result[0][1]["motion_context"]
