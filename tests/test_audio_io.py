"""Run with ComfyUI on PYTHONPATH and ffmpeg/ffprobe on PATH."""
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from lib import audio_io


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
class AudioSampleRateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="director-audio-test-")
        self.addCleanup(self.temporary.cleanup)

    def make_tone(self, sample_rate, channels):
        path = Path(self.temporary.name) / f"tone-{sample_rate}-{channels}.wav"
        times = np.arange(sample_rate * 2) / sample_rate
        samples = np.rint(12000 * np.sin(2 * np.pi * 440 * times)).astype("<i2")
        pcm = np.repeat(samples[:, None], channels, axis=1)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())
        return str(path)

    def assert_tone(self, audio, expected_rate):
        self.assertIsNotNone(audio)
        self.assertEqual(audio["sample_rate"], expected_rate)
        waveform = audio["waveform"]
        self.assertEqual(tuple(waveform.shape), (1, 2, expected_rate * 2))
        self.assertEqual(waveform.shape[-1] / audio["sample_rate"], 2.0)
        spectrum = np.abs(np.fft.rfft(waveform[0, 0].numpy()))
        peak_hz = np.argmax(spectrum) * expected_rate / waveform.shape[-1]
        self.assertAlmostEqual(peak_hz, 440.0, delta=0.5)

    def test_native_rate_preserves_duration_and_pitch(self):
        for rate in (16000, 24000, 32000, 44100, 48000):
            for channels in (1, 2):
                with self.subTest(sample_rate=rate, channels=channels):
                    path = self.make_tone(rate, channels)
                    self.assert_tone(audio_io.load_reference_audio(path), rate)

    def test_missing_probe_resamples_to_declared_fallback_rate(self):
        path = self.make_tone(48000, 1)
        with mock.patch.object(audio_io, "ffprobe_bin", return_value=None):
            self.assert_tone(audio_io.load_reference_audio(path), 44100)


if __name__ == "__main__":
    unittest.main()
