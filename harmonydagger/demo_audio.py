"""
Audio I/O helpers for the Streamlit demo.

Librosa (via ffmpeg) is used so MP3 uploads work in addition to WAV/FLAC/OGG.
"""
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from numpy.typing import NDArray

DEMO_UPLOAD_TYPES = ["wav", "flac", "ogg", "mp3"]
_ALLOWED_SUFFIXES = {f".{ext}" for ext in DEMO_UPLOAD_TYPES}


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
        path: Filesystem path to a wav, flac, ogg, or mp3 file.

    Returns:
        Tuple of (mono audio samples, sample rate).

    Raises:
        DemoAudioLoadError: If the file cannot be decoded.
    """
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
    except Exception as exc:
        raise DemoAudioLoadError(_load_error_message(path)) from exc

    if audio is None or np.asarray(audio).size == 0:
        raise DemoAudioLoadError(_load_error_message(path))

    return np.asarray(audio, dtype=np.float64), int(sr)


def _load_error_message(path: str) -> str:
    if Path(path).suffix.lower() == ".mp3":
        return (
            "Could not read this MP3 file. HarmonyDagger requires ffmpeg "
            "for MP3 support."
        )
    return "Could not read this audio file."
