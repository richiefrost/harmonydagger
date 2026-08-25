"""
Zero-shot voice-clone evaluation for HarmonyDagger.

Uses a TTS model to generate new speech from a text prompt, once with
the original clip as the speaker reference and once with the protected
clip. Comparing those two generations is how you check the actual
cloning threat — not whether the protected file still sounds human.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from .verify import compute_feature_similarity

DEFAULT_CLONE_TEXT = "Hello, this is a test of my voice."
MAX_CLONE_TEXT_CHARS = 500
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

Synthesizer = Callable[
    [NDArray[np.float64], int, str, str],
    Tuple[NDArray[np.float64], int],
]

_xtts_model = None


class CloneUnavailableError(RuntimeError):
    """Raised when Coqui TTS is not installed or the model cannot load."""


@dataclass(frozen=True)
class CloneResult:
    audio: NDArray[np.float64]
    sample_rate: int


def is_clone_available() -> bool:
    """Return True if the Coqui TTS package can be imported."""
    try:
        import TTS.api  # noqa: F401
    except ImportError:
        return False
    return True


def xtts_synthesizer(
    reference: NDArray[np.float64],
    sr: int,
    text: str,
    language: str,
) -> Tuple[NDArray[np.float64], int]:
    """Run Coqui XTTS-v2 using ``reference`` as the speaker sample."""
    tts = _load_xtts_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, _as_mono(reference), sr)
        wav = tts.tts(
            text=text,
            speaker_wav=tmp.name,
            language=language,
        )
    audio = np.asarray(wav, dtype=np.float64)
    out_sr = int(getattr(tts, "output_sample_rate", None) or 24000)
    return audio, out_sr


def synthesize_clone(
    reference: NDArray[np.float64],
    sr: int,
    text: str,
    *,
    language: str = "en",
    synthesizer: Optional[Synthesizer] = None,
) -> CloneResult:
    """Generate speech that should match the speaker in ``reference``."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text must be a non-empty prompt for the clone model")
    if len(cleaned) > MAX_CLONE_TEXT_CHARS:
        raise ValueError(
            f"text must be at most {MAX_CLONE_TEXT_CHARS} characters"
        )

    synth = xtts_synthesizer if synthesizer is None else synthesizer
    audio, out_sr = synth(reference, sr, cleaned, language)
    return CloneResult(
        audio=np.asarray(audio, dtype=np.float64),
        sample_rate=int(out_sr),
    )


def compare_reference_clones(
    original: NDArray[np.float64],
    protected: NDArray[np.float64],
    sr: int,
    text: str,
    *,
    language: str = "en",
    synthesizer: Optional[Synthesizer] = None,
) -> Dict[str, Union[NDArray[np.float64], int, float]]:
    """Clone from original and protected references and score both vs original."""
    original_clone = synthesize_clone(
        original, sr, text, language=language, synthesizer=synthesizer
    )
    protected_clone = synthesize_clone(
        protected, sr, text, language=language, synthesizer=synthesizer
    )
    original_sim = compute_feature_similarity(
        original,
        _resample_to(original_clone.audio, original_clone.sample_rate, sr),
        sr,
    )
    protected_sim = compute_feature_similarity(
        original,
        _resample_to(protected_clone.audio, protected_clone.sample_rate, sr),
        sr,
    )
    return {
        "original_clone": original_clone.audio,
        "original_clone_sr": original_clone.sample_rate,
        "protected_clone": protected_clone.audio,
        "protected_clone_sr": protected_clone.sample_rate,
        "original_clone_similarity": original_sim,
        "protected_clone_similarity": protected_sim,
    }


def _as_mono(audio: NDArray[np.float64]) -> NDArray[np.float64]:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _resample_to(
    audio: NDArray[np.float64],
    from_sr: int,
    to_sr: int,
) -> NDArray[np.float64]:
    if from_sr == to_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=from_sr, target_sr=to_sr)


def _load_xtts_model():
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model
    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise CloneUnavailableError(
            'Coqui TTS is not installed. Run: pip install -e ".[clone]" '
            "(installs coqui-tts)."
        ) from exc
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TTS(XTTS_MODEL_NAME).to(device)
    except Exception as exc:  # model download / runtime failures
        raise CloneUnavailableError(
            f"Could not load XTTS-v2: {exc}"
        ) from exc
    _xtts_model = model
    return model
