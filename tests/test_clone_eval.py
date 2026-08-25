import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _sine(sr=22050, duration=0.25, freq=220.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float64), sr


def test_synthesize_clone_uses_injected_synthesizer():
    from harmonydagger.clone_eval import synthesize_clone

    reference, sr = _sine()
    expected = np.full(100, 0.3, dtype=np.float64)

    def fake_synthesizer(ref, sample_rate, text, language):
        assert ref is reference
        assert sample_rate == sr
        assert text == "Hello, this is a test of my voice."
        assert language == "en"
        return expected, 16000

    result = synthesize_clone(
        reference,
        sr,
        "Hello, this is a test of my voice.",
        synthesizer=fake_synthesizer,
    )

    assert np.array_equal(result.audio, expected)
    assert result.sample_rate == 16000


def test_synthesize_clone_rejects_blank_text():
    from harmonydagger.clone_eval import synthesize_clone

    reference, sr = _sine()

    def fake_synthesizer(ref, sample_rate, text, language):
        raise AssertionError("synthesizer should not be called")

    with pytest.raises(ValueError, match="text"):
        synthesize_clone(reference, sr, "   ", synthesizer=fake_synthesizer)


def test_compare_reference_clones_scores_original_and_protected():
    from harmonydagger.clone_eval import compare_reference_clones

    original, sr = _sine(freq=220.0)
    protected, _ = _sine(freq=880.0)

    def echo_synthesizer(ref, sample_rate, text, language):
        return ref.copy(), sample_rate

    result = compare_reference_clones(
        original,
        protected,
        sr,
        "Hello, this is a test of my voice.",
        synthesizer=echo_synthesizer,
    )

    assert np.allclose(result["original_clone"], original)
    assert np.allclose(result["protected_clone"], protected)
    assert result["original_clone_sr"] == sr
    assert result["protected_clone_sr"] == sr
    assert result["original_clone_similarity"] > 0.99
    assert result["protected_clone_similarity"] < result["original_clone_similarity"]


def test_xtts_synthesizer_raises_when_coqui_missing(monkeypatch):
    import builtins

    from harmonydagger.clone_eval import CloneUnavailableError, xtts_synthesizer

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "TTS.api" or name.startswith("TTS"):
            raise ImportError("No module named 'TTS'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    reference, sr = _sine()
    with pytest.raises(CloneUnavailableError, match="coqui-tts"):
        xtts_synthesizer(reference, sr, "Hello", "en")


def test_is_clone_available_false_when_coqui_missing(monkeypatch):
    import builtins

    from harmonydagger import clone_eval

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "TTS.api" or name.startswith("TTS"):
            raise ImportError("No module named 'TTS'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(clone_eval, "_xtts_model", None)

    assert clone_eval.is_clone_available() is False


def test_streamlit_app_shows_voice_clone_check_after_upload():
    from streamlit.testing.v1 import AppTest

    from harmonydagger.clone_eval import DEFAULT_CLONE_TEXT

    audio, sr = _sine(duration=0.3)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    wav_bytes = buf.getvalue()

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception

    at.file_uploader[0].set_value(("clip.wav", wav_bytes, "audio/wav"))
    at.run()

    assert not at.exception
    assert not at.error
    subheaders = [s.value for s in at.subheader]
    assert "Generation check" in subheaders
    radio_options = [opt for r in at.radio for opt in r.options]
    assert any("XTTS" in opt for opt in radio_options)
    assert any(DEFAULT_CLONE_TEXT in (ta.value or "") for ta in at.text_area)
    assert any(btn.label == "Generate clones" for btn in at.button)
