"""
Audio I/O helpers for the Streamlit demo.

Librosa/soundfile handle WAV, FLAC, OGG, and MP3. M4A (AAC) is decoded
with pydub/ffmpeg because libsndfile cannot read that container.
"""
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from numpy.typing import NDArray
from pydub import AudioSegment

DEMO_UPLOAD_TYPES = ["wav", "flac", "ogg", "mp3", "m4a"]
_ALLOWED_SUFFIXES = {f".{ext}" for ext in DEMO_UPLOAD_TYPES}
_FFMPEG_SUFFIXES = {".mp3", ".m4a"}


class DemoAudioLoadError(Exception):
    """Raised when the demo cannot decode an uploaded audio file."""


def suffix_for_demo_upload(filename: str) -> str:
    """Return a safe temp-file suffix from an upload filename.

    Unknown or missing extensions fall back to ``.wav`` so user-controlled
    names cannot pick an arbitrary file type.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in _ALLOWED_SUFFIXES:
        return suffix
    return ".wav"


def read_audio_for_demo(path: str) -> Tuple[NDArray[np.float64], int]:
    """Load demo audio as mono float64 samples and sample rate.

    Args:
        path: Filesystem path to a wav, flac, ogg, mp3, or m4a file.

    Returns:
        Tuple of (mono audio samples, sample rate).

    Raises:
        DemoAudioLoadError: If the file cannot be decoded.
    """
    try:
        audio, sr = _load_audio(path)
    except Exception as exc:
        raise DemoAudioLoadError(_load_error_message(path)) from exc

    if audio is None or np.asarray(audio).size == 0:
        raise DemoAudioLoadError(_load_error_message(path))

    return np.asarray(audio, dtype=np.float64), int(sr)


def _load_audio(path: str) -> Tuple[NDArray[np.float64], int]:
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        return np.asarray(audio, dtype=np.float64), int(sr)
    except Exception:
        if Path(path).suffix.lower() != ".m4a":
            raise
        return _load_with_pydub(path)


def _load_with_pydub(path: str) -> Tuple[NDArray[np.float64], int]:
    segment = AudioSegment.from_file(path)
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64)
    if segment.channels > 1:
        samples = samples.reshape((-1, segment.channels)).mean(axis=1)
    peak = float(1 << (8 * segment.sample_width - 1))
    samples /= peak
    return samples, int(segment.frame_rate)


def _load_error_message(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _FFMPEG_SUFFIXES:
        fmt = suffix[1:].upper()
        return (
            f"Could not read this {fmt} file. HarmonyDagger requires ffmpeg "
            f"for {fmt} support."
        )
    return "Could not read this audio file."
