from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
CUSTOM_NODES_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

from ComfyUI_MiniMaxH3_Director.director import plan
from ComfyUI_MiniMaxH3_Director.director import executor_core
from ComfyUI_MiniMaxH3_Director.lib import audio_io


class FullAudioCacheScopeTests(unittest.TestCase):
    def _decoded_pcm(self):
        samples = np.asarray([0.0, 0.0, 0.25, -0.25], dtype=np.float32)
        return SimpleNamespace(stdout=samples.tobytes(), stderr=b"")

    def test_no_process_wide_pcm_cache(self):
        self.assertFalse(hasattr(audio_io, "_FULL_AUDIO_CACHE"))
        with (
            patch.object(audio_io, "_ffmpeg_bin", return_value="ffmpeg"),
            patch.object(audio_io, "_probe_audio_stream", return_value=(44100, 2)),
            patch.object(audio_io.os.path, "isfile", return_value=True),
            patch.object(audio_io.subprocess, "run", return_value=self._decoded_pcm()) as run,
        ):
            first = audio_io._load_full_audio("ref.wav")
            second = audio_io._load_full_audio("ref.wav")

        self.assertIsNot(first, second)
        self.assertEqual(run.call_count, 2)

    def test_caller_owned_cache_reuses_only_within_its_scope(self):
        execution_cache = {}
        with (
            patch.object(audio_io, "_ffmpeg_bin", return_value="ffmpeg"),
            patch.object(audio_io, "_probe_audio_stream", return_value=(44100, 2)),
            patch.object(audio_io.os.path, "isfile", return_value=True),
            patch.object(audio_io.subprocess, "run", return_value=self._decoded_pcm()) as run,
        ):
            first = audio_io._load_full_audio("source.mp4", cache=execution_cache)
            second = audio_io._load_full_audio("source.mp4", cache=execution_cache)

        self.assertIs(first, second)
        self.assertEqual(run.call_count, 1)


class ReferenceAudioLazyLoadTests(unittest.TestCase):
    def test_file_reference_is_lazy_and_shared_object_decodes_once(self):
        item = {"index": 1, "audioFile": "refs/voice.wav"}
        waveform = __import__("torch").zeros(1, 2, 32)
        decoded = {"waveform": waveform, "sample_rate": 44100}

        with patch.object(plan.os.path, "isfile", return_value=True):
            refs = plan._load_ref_audios([item])

        self.assertEqual(len(refs), 1)
        self.assertIsNone(refs[0].audio)
        self.assertTrue(refs[0].audio_path.replace("\\", "/").endswith("refs/voice.wav"))

        with patch.object(plan, "load_reference_audio", return_value=decoded) as load:
            first = plan.ref_audios_to_dict(refs)
            second = plan.ref_audios_to_dict(refs)

        self.assertIs(first["ref_audio_1"], decoded)
        self.assertIs(second["ref_audio_1"], decoded)
        self.assertEqual(load.call_count, 1)

    def test_segment_release_keeps_shared_and_graph_audio(self):
        waveform = __import__("torch").zeros(1, 2, 32)
        decoded = {"waveform": waveform, "sample_rate": 44100}
        shared = plan.SegmentRefAudio(
            index=0,
            audio=decoded,
            audio_file="global.wav",
            audio_path="global.wav",
        )
        local = plan.SegmentRefAudio(
            index=1,
            audio=decoded,
            audio_file="local.wav",
            audio_path="local.wav",
        )
        graph_audio = plan.SegmentRefAudio(index=2, audio=decoded, audio_file="")
        fake_plan = SimpleNamespace(global_ref_audios=[shared])
        fake_segment = SimpleNamespace(ref_audios=[shared, local, graph_audio])

        executor_core._release_segment_file_ref_audios(fake_plan, fake_segment)

        self.assertIs(shared.audio, decoded)
        self.assertIsNone(local.audio)
        self.assertIs(graph_audio.audio, decoded)


if __name__ == "__main__":
    unittest.main()
