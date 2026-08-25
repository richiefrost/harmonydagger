"""
Audio-prompted music generation for HarmonyDagger evaluation.

Uses MusicGen-small to continue a clip: once from the original audio and
once from the protected audio. Comparing those two generations is how you
check whether protection disrupts music models — not whether the protected
file still sounds like music to a human.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .verify import compute_feature_similarity

DEFAULT_MUSIC_TEXT = "80s pop track with bassy drums and synth"
MAX_MUSIC_TEXT_CHARS = 500
MUSICGEN_MODEL_NAME = "facebook/musicgen-small"
MAX_MUSIC_PROMPT_SECONDS = 3
MAX_NEW_TOKENS = 128

Synthesizer = Callable[
    [NDArray[np.float64], int, str],
    Tuple[NDArray[np.float64], int],
]

_musicgen_model = None
_musicgen_processor = None


class MusicUnavailableError(RuntimeError):
    """Raised when Hugging Face MusicGen is not installed or cannot load."""


@dataclass(frozen=True)
class MusicResult:
    audio: NDArray[np.float64]
    sample_rate: int


def is_music_available() -> bool:
    """Return True if the transformers package can be imported."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def musicgen_synthesizer(
    reference: NDArray[np.float64],
    sr: int,
    text: str,
) -> Tuple[NDArray[np.float64], int]:
    """Continue ``reference`` with MusicGen-small guided by ``text``."""
    processor, model = _load_musicgen()
    target_sr = int(model.config.audio_encoder.sampling_rate)
    prompt = _clip_music_prompt(
        _resample_to(_as_mono(reference), sr, target_sr), target_sr
    )

    import torch

    inputs = processor(
        audio=prompt.astype(np.float32),
        sampling_rate=target_sr,
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = _move_inputs_to_model(inputs, model)
    with torch.inference_mode():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    wav = np.asarray(audio_values[0, 0].detach().cpu().numpy(), dtype=np.float64)
    del audio_values
    _release_accelerator_cache(device)
    return wav, target_sr


def synthesize_music(
    reference: NDArray[np.float64],
    sr: int,
    text: str,
    *,
    synthesizer: Optional[Synthesizer] = None,
) -> MusicResult:
    """Generate music continued from ``reference`` using a text prompt."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text must be a non-empty prompt for the music model")
    if len(cleaned) > MAX_MUSIC_TEXT_CHARS:
        raise ValueError(
            f"text must be at most {MAX_MUSIC_TEXT_CHARS} characters"
        )

    synth = musicgen_synthesizer if synthesizer is None else synthesizer
    audio, out_sr = synth(reference, sr, cleaned)
    return MusicResult(
        audio=np.asarray(audio, dtype=np.float64),
        sample_rate=int(out_sr),
    )


def compare_reference_music(
    original: NDArray[np.float64],
    protected: NDArray[np.float64],
    sr: int,
    text: str,
    *,
    synthesizer: Optional[Synthesizer] = None,
) -> Dict[str, Union[NDArray[np.float64], int, float]]:
    """Generate from original and protected prompts and score both vs original."""
    original_music = synthesize_music(
        original, sr, text, synthesizer=synthesizer
    )
    protected_music = synthesize_music(
        protected, sr, text, synthesizer=synthesizer
    )
    original_sim = compute_feature_similarity(
        original,
        _resample_to(original_music.audio, original_music.sample_rate, sr),
        sr,
    )
    protected_sim = compute_feature_similarity(
        original,
        _resample_to(protected_music.audio, protected_music.sample_rate, sr),
        sr,
    )
    return {
        "original_music": original_music.audio,
        "original_music_sr": original_music.sample_rate,
        "protected_music": protected_music.audio,
        "protected_music_sr": protected_music.sample_rate,
        "original_music_similarity": original_sim,
        "protected_music_similarity": protected_sim,
    }


def _clip_music_prompt(
    audio: NDArray[np.float64],
    sr: int,
    max_seconds: int = MAX_MUSIC_PROMPT_SECONDS,
) -> NDArray[np.float64]:
    max_samples = int(sr * max_seconds)
    if len(audio) > max_samples:
        return audio[:max_samples]
    return audio


def _move_inputs_to_model(inputs, model):
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _release_accelerator_cache(device) -> None:
    import torch

    if getattr(device, "type", None) == "mps" or str(device) == "mps":
        torch.mps.empty_cache()
    elif getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()


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


def _load_musicgen():
    global _musicgen_model, _musicgen_processor
    if _musicgen_model is not None and _musicgen_processor is not None:
        return _musicgen_processor, _musicgen_model
    try:
        from transformers import AutoProcessor, MusicgenConfig, MusicgenForConditionalGeneration
    except ImportError as exc:
        raise MusicUnavailableError(
            'MusicGen is not installed. Run: pip install -e ".[music]" '
            "(installs transformers)."
        ) from exc
    try:
        import torch

        from .gpu import get_device

        device = get_device() or torch.device("cpu")
        processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL_NAME)
        # transformers 4.44–4.57 maps this composite class to MusicgenDecoderConfig,
        # so from_pretrained loads a decoder-only config and then crashes on
        # config.decoder. Restore the composite config class first.
        MusicgenForConditionalGeneration.config_class = MusicgenConfig
        dtype = (
            torch.float16
            if getattr(device, "type", None) in ("cuda", "mps") or str(device) == "mps"
            else torch.float32
        )
        model = MusicgenForConditionalGeneration.from_pretrained(
            MUSICGEN_MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
    except Exception as exc:  # model download / runtime failures
        raise MusicUnavailableError(
            f"Could not load MusicGen-small: {exc}"
        ) from exc
    _musicgen_processor = processor
    _musicgen_model = model
    return processor, model
