"""
Tests for Streamlit demo audio loading, including MP3.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from harmonydagger.demo_audio import (
    DemoAudioLoadError,
    read_audio_for_demo,
    suffix_for_demo_upload,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _sine_wave(sample_rate: int = 22050, duration: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return (0.5 * np.sin(2.0 * np.pi * 440 * t)).astype(np.float64)


def _write_test_mp3(wav_path: Path, mp3_path: Path) -> bytes:
    from pydub import AudioSegment

    AudioSegment.from_wav(str(wav_path)).export(
        str(mp3_path), format="mp3", bitrate="192k"
    )
    return mp3_path.read_bytes()


class DemoAudioLoadTest(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 22050
        self.duration = 0.5
        self.test_audio = _sine_wave(self.sample_rate, self.duration)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.wav_path = self.temp_path / "test_audio.wav"
        sf.write(self.wav_path, self.test_audio, self.sample_rate)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_audio_for_demo_loads_wav(self):
        audio, sr = read_audio_for_demo(str(self.wav_path))

        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(audio.ndim, 1)
        self.assertAlmostEqual(len(audio) / sr, self.duration, delta=0.05)

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="Requires ffmpeg for MP3 support")
    def test_read_audio_for_demo_loads_mp3(self):
        mp3_path = self.temp_path / "test_audio.mp3"
        _write_test_mp3(self.wav_path, mp3_path)

        audio, sr = read_audio_for_demo(str(mp3_path))

        self.assertEqual(audio.ndim, 1)
        self.assertGreater(sr, 0)
        self.assertAlmostEqual(len(audio) / sr, self.duration, delta=0.1)

    def test_read_audio_for_demo_converts_stereo_to_mono(self):
        stereo = np.column_stack([self.test_audio, self.test_audio * 0.5])
        stereo_path = self.temp_path / "stereo.wav"
        sf.write(stereo_path, stereo, self.sample_rate)

        audio, sr = read_audio_for_demo(str(stereo_path))

        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(audio.ndim, 1)

    def test_read_audio_for_demo_raises_on_unreadable_file(self):
        bad_path = self.temp_path / "bad.mp3"
        bad_path.write_bytes(b"not an audio file")

        with self.assertRaises(DemoAudioLoadError):
            read_audio_for_demo(str(bad_path))

    def test_suffix_for_demo_upload_preserves_mp3(self):
        self.assertEqual(suffix_for_demo_upload("clip.mp3"), ".mp3")
        self.assertEqual(suffix_for_demo_upload("Clip.MP3"), ".mp3")

    def test_suffix_for_demo_upload_preserves_supported_formats(self):
        self.assertEqual(suffix_for_demo_upload("a.wav"), ".wav")
        self.assertEqual(suffix_for_demo_upload("a.flac"), ".flac")
        self.assertEqual(suffix_for_demo_upload("a.ogg"), ".ogg")

    def test_suffix_for_demo_upload_rejects_unknown_extension(self):
        self.assertEqual(suffix_for_demo_upload("payload.exe"), ".wav")
        self.assertEqual(suffix_for_demo_upload("noext"), ".wav")


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="Requires ffmpeg for MP3 support")
def test_streamlit_app_accepts_and_processes_mp3():
    """Uploading MP3 in the demo should decode and show original plus protected audio."""
    from streamlit.testing.v1 import AppTest

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "clip.wav"
        mp3_path = Path(tmp) / "clip.mp3"
        sf.write(wav_path, _sine_wave(), 22050)
        mp3_bytes = _write_test_mp3(wav_path, mp3_path)

        at = AppTest.from_file(str(APP_PATH), default_timeout=60)
        at.run()
        assert not at.exception

        allowed = [t.lower() for t in at.file_uploader[0].allowed_type]
        assert any("mp3" in t for t in allowed)

        at.file_uploader[0].set_value(("clip.mp3", mp3_bytes, "audio/mpeg"))
        at.run()

        assert not at.exception
        assert not at.error
        subheaders = [s.value for s in at.subheader]
        assert "Original Audio" in subheaders
        assert "Protected Audio" in subheaders
        assert any(btn.label == "Download Protected Audio" for btn in at.download_button)
        assert len(at.metric) >= 4
