"""Run with Dapao/ComfyUI Python: set PYTHONPATH to ComfyUI, then run this file.
Tests real planning, output wiring, and pack round-trip without H3 weights.
"""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave
import zipfile

import comfy.options
sys.argv.append("--cpu")
comfy.options.enable_args_parsing()
import folder_paths
import torch

# Bypass plugin registration (which needs a running server) but preserve relative imports.
root = Path(__file__).resolve().parents[1]
package = types.ModuleType("minimax_master_audio_test")
package.__path__ = [str(root)]
sys.modules[package.__name__] = package

def module(name):
    return importlib.import_module(f"{package.__name__}.{name}")

master = module("director.master_audio")
common = module("nodes.director_common")
audio_export = module("director.audio_export")
pack = module("director.pack")
beats = module("director.audio_onsets")


class MasterAudioTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        old_input = folder_paths.get_input_directory()
        old_temp = folder_paths.get_temp_directory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(folder_paths.set_input_directory, old_input)
        self.addCleanup(folder_paths.set_temp_directory, old_temp)
        folder_paths.set_input_directory(self.dir.name)
        folder_paths.set_temp_directory(self.dir.name)
        self.path = Path(self.dir.name) / "song.wav"
        # Anti-phase stereo: a mean-of-channels envelope would incorrectly be silent.
        samples = torch.arange(44100 * 4, dtype=torch.int32) % 54321 - 27160
        sample_data = torch.stack((samples, -samples), dim=1).short().numpy().tobytes()
        with wave.open(str(self.path), "wb") as writer:
            writer.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
            writer.writeframes(sample_data)
        self.timeline = {
            "version": 5, "timelineMode": "prompt_batch", "editMode": "segment",
            "frameRate": 24, "totalFrames": 73,
            "global": {"taskType": "r2v"},
            "output": {"width": 64, "height": 64, "mode": "fixed", "continuityEnabled": True},
            "segments": [
                {"start": 0, "frameCount": 31, "durationSec": 5, "prompt": "Sing."},
                {"start": 31, "frameCount": 42, "durationSec": 5, "prompt": "Keep singing."},
            ],
            master.MASTER_KEY: {"audioFile": "song.wav", "label": "Test master"},
        }

    def plan(self):
        return common.prepare_director_plan(
            timeline_data=json.dumps(self.timeline), task_type="r2v", global_prompt="",
            total_frames=73, frame_rate=24, width=64, height=64, ref_max_size=64,
            unique_id=None,
        )

    def test_exact_window_conditioning_and_continuous_output(self):
        plan = self.plan()
        self.assertEqual([(s.start_frame, s.end_frame) for s in plan.segments], [(0, 31), (31, 73)])
        self.assertFalse(plan.continuity_enabled)
        self.assertEqual(audio_export.resolve_audio_mode(plan), master.MASTER_MODE)
        wave = plan.master_audio["waveform"]
        middle, end = round(31 * 44100 / 24), round(73 * 44100 / 24)
        self.assertTrue(torch.equal(plan.segments[0].ref_audios[0].audio["waveform"], wave[..., :middle]))
        self.assertTrue(torch.equal(plan.segments[1].ref_audios[0].audio["waveform"], wave[..., middle:end]))
        left, right = [torch.zeros((n, 16, 16, 3)) for n in (31, 42)]
        output = common.finalize_director_outputs(
            plan, torch.cat((left, right)), [left, right], "test",
            segment_audios=[{"waveform": torch.zeros(1, 1, 5), "sample_rate": 16000}] * 2,
            segment_frame_counts=[31, 42],
        )
        self.assertEqual(len(output[1]), 1)
        self.assertEqual(output[1][0]["sample_rate"], 44100)
        self.assertTrue(torch.equal(output[1][0]["waveform"], wave[..., :end]))
        self.assertIn("全局音轨", output[5])
        self.assertNotIn("extracted from input video", output[5])

    def test_create_video_muxes_one_audio_stream_and_preserves_picture_timing(self):
        from comfy_extras.nodes_video import CreateVideo
        import av

        plan = self.plan()
        left, right = [torch.zeros((n, 32, 32, 3)) for n in (31, 42)]
        images, audio, fps, *_ = common.finalize_director_outputs(
            plan, torch.cat((left, right)), [left, right], "test", segment_frame_counts=[31, 42]
        )
        video = CreateVideo.execute(images[0], fps, audio[0]).result[0]
        target = Path(self.dir.name) / "master.mp4"
        video.save_to(str(target))
        with av.open(str(target)) as container:
            self.assertEqual(len(container.streams.audio), 1)
            self.assertEqual(container.streams.audio[0].rate, 44100)
            self.assertEqual(sum(1 for _ in container.decode(video=0)), 73)
        with av.open(str(target)) as container:
            frames = list(container.decode(audio=0))
            self.assertTrue(frames)
            for previous, following in zip(frames, frames[1:]):
                self.assertEqual(previous.pts + previous.samples, following.pts)

    def test_segment_export_uses_absolute_offset_and_silence_past_end(self):
        plan = self.plan()
        plan.export_mode = "segments"
        plan.run_indices = frozenset({1})
        second = torch.zeros((42, 16, 16, 3))
        audio = common.finalize_director_outputs(plan, second, [second], "test", segment_frame_counts=[42])[1][0]
        mp4_audio = audio_export.prepare_segment_audio_for_file_export(plan, plan.segments[1], audio_dict=None, frame_count=42)
        expected = plan.master_audio["waveform"][..., round(31 * 44100 / 24):round(73 * 44100 / 24)]
        self.assertTrue(torch.equal(audio["waveform"], expected))
        self.assertTrue(torch.equal(mp4_audio["waveform"], expected))
        tail = master.slice_master(plan.master_audio, 95, 100, 24)["waveform"]
        self.assertGreater(tail[..., :1837].abs().max().item(), 0)
        self.assertEqual(tail[..., 1838:].count_nonzero().item(), 0)

    def test_waveform_and_pack(self):
        peaks = master.master_peaks(master.decode_master(self.path))
        self.assertEqual(peaks["duration"], 4)
        self.assertGreater(max(peaks["peaks"]), 140)
        exported = pack.build_export_pack(self.timeline)
        self.assertEqual(exported["missing"], [])
        archive = pack._pack_export_root() / exported["filename"]
        target = Path(self.dir.name) / "extract"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        imported = pack.import_extracted_pack(target)
        data = imported["timeline"][master.MASTER_KEY]
        self.assertEqual(imported["missing"], [])
        self.assertEqual(data["label"], "Test master")
        self.assertEqual(master.resolve_master_path(data["audioFile"]).read_bytes(), self.path.read_bytes())

    def test_waveform_distinguishes_a_transient_from_sustained_volume(self):
        # Two buckets with the same peak; sustained sound must show a much taller RMS.
        sr = 8000
        signal = torch.cat((torch.tensor([0.8] + [0.0] * 999), torch.full((1000,), 0.8), torch.zeros(999)))
        info = master.master_peaks({"waveform": signal.view(1, 1, -1), "sample_rate": sr}, limit=3)
        self.assertEqual(info["peaks"][0], info["peaks"][1])
        self.assertLess(info["rms"][0] * 20, info["rms"][1])
        self.assertEqual(info["rms"][-1], 0)
        self.assertEqual(info["rms"][1], info["peaks"][1])

    def test_onset_suggestions_find_percussion_but_not_silence(self):
        sr = 16000
        wave = torch.zeros(sr * 4)
        t = torch.arange(1600) / sr
        pulse = torch.sin(t * 2 * torch.pi * 105) * (-t * 27).exp()
        expected = [0.55, 1.10, 1.80, 2.60, 3.40]
        for timing, amplitude in zip(expected, (0.90, 0.42, 0.26, 0.81, 0.55)):
            start = round(timing * sr)
            wave[start:start + len(pulse)] = pulse * amplitude
        signal = torch.stack((wave, -wave))[None]
        actual = beats.suggest_cut_onsets({"waveform": signal, "sample_rate": sr})
        self.assertEqual(len(actual), len(expected), actual)
        self.assertTrue(all(abs(a - b) < 0.06 for a, b in zip(actual, expected)), actual)
        self.assertEqual(beats.suggest_cut_onsets({"waveform": torch.zeros_like(signal), "sample_rate": sr}), [])

    def test_cache_inspection_matches_execution_without_decoding(self):
        planned = self.plan()
        status = module("director.plan").build_director_plan(
            json.dumps(self.timeline), global_task_type="r2v", global_prompt="",
            total_frames=73, frame_rate=24, width=64, height=64, ref_max_size=64,
        )
        with patch.object(master, "decode_master", side_effect=AssertionError("unexpected decode")):
            master.attach_master(status, master.master_record(status.raw), decode=False)
        fingerprint = module("director.segment_cache").first_pass_cache_fingerprint
        self.assertIsNone(status.master_audio)
        self.assertEqual(fingerprint(status.segments[0], status), fingerprint(planned.segments[0], planned))

    def test_no_master_keeps_h3_time_and_blocks_unsafe_paths(self):
        self.timeline.pop(master.MASTER_KEY)
        self.assertNotEqual(self.plan().segments[0].frame_count, 31)
        for name in ["../outside.wav", "C:/outside.wav", "song.mp4"]:
            with self.assertRaises(ValueError):
                master.resolve_master_path(name)
        self.timeline[master.MASTER_KEY] = {"audioFile": "song.wav"}
        self.timeline["frameRate"] = 25
        with self.assertRaisesRegex(ValueError, "24 fps"):
            self.plan()


if __name__ == "__main__":
    sys.argv.remove("--cpu")
    # Comfy's CLI has already picked up --cpu during module import.
    unittest.main()
