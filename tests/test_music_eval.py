import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _sine(sr=22050, duration=0.25, freq=220.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float64), sr


def test_synthesize_music_uses_injected_synthesizer():
    from harmonydagger.music_eval import synthesize_music

    reference, sr = _sine()
    expected = np.full(100, 0.3, dtype=np.float64)

    def fake_synthesizer(ref, sample_rate, text):
        assert ref is reference
        assert sample_rate == sr
        assert text == "80s pop track with bassy drums and synth"
        return expected, 32000

    result = synthesize_music(
        reference,
        sr,
        "80s pop track with bassy drums and synth",
        synthesizer=fake_synthesizer,
    )

    assert np.array_equal(result.audio, expected)
    assert result.sample_rate == 32000


def test_synthesize_music_rejects_blank_text():
    from harmonydagger.music_eval import synthesize_music

    reference, sr = _sine()

    def fake_synthesizer(ref, sample_rate, text):
        raise AssertionError("synthesizer should not be called")

    with pytest.raises(ValueError, match="text"):
        synthesize_music(reference, sr, "   ", synthesizer=fake_synthesizer)


def test_compare_reference_music_scores_original_and_protected():
    from harmonydagger.music_eval import compare_reference_music

    original, sr = _sine(freq=220.0)
    protected, _ = _sine(freq=880.0)

    def echo_synthesizer(ref, sample_rate, text):
        return ref.copy(), sample_rate

    result = compare_reference_music(
        original,
        protected,
        sr,
        "80s pop track with bassy drums and synth",
        synthesizer=echo_synthesizer,
    )

    assert np.allclose(result["original_music"], original)
    assert np.allclose(result["protected_music"], protected)
    assert result["original_music_sr"] == sr
    assert result["protected_music_sr"] == sr
    assert result["original_music_similarity"] > 0.99
    assert result["protected_music_similarity"] < result["original_music_similarity"]


def test_musicgen_synthesizer_raises_when_transformers_missing(monkeypatch):
    import builtins

    from harmonydagger.music_eval import MusicUnavailableError, musicgen_synthesizer

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers"):
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    reference, sr = _sine()
    with pytest.raises(MusicUnavailableError, match="music"):
        musicgen_synthesizer(reference, sr, "80s pop track")


def test_load_musicgen_passes_composite_config(monkeypatch):
    """transformers>=4.44 loads decoder-only config unless MusicgenConfig is passed."""
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    from harmonydagger import music_eval

    monkeypatch.setattr(music_eval, "_musicgen_model", None)
    monkeypatch.setattr(music_eval, "_musicgen_processor", None)

    class DummyParam:
        device = "cpu"

    class DummyModel:
        def __init__(self, config):
            self.config = config

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            yield DummyParam()

    class FakeCompositeConfig:
        decoder = object()
        audio_encoder = type("AE", (), {"sampling_rate": 32000})()

    def fake_processor_from_pretrained(cls, name, **kwargs):
        return object()

    def fake_model_from_pretrained(cls, name, **kwargs):
        if getattr(cls.config_class, "__name__", "") != "MusicgenConfig":
            raise AttributeError(
                "'MusicgenDecoderConfig' object has no attribute 'decoder'"
            )
        return DummyModel(FakeCompositeConfig())

    monkeypatch.setattr(
        AutoProcessor, "from_pretrained", classmethod(fake_processor_from_pretrained)
    )
    monkeypatch.setattr(
        MusicgenForConditionalGeneration,
        "from_pretrained",
        classmethod(fake_model_from_pretrained),
    )

    processor, model = music_eval._load_musicgen()

    assert processor is not None
    assert hasattr(model.config, "decoder")


def test_clip_music_prompt_limits_duration():
    from harmonydagger.music_eval import MAX_MUSIC_PROMPT_SECONDS, _clip_music_prompt

    audio, sr = _sine(duration=10)
    clipped = _clip_music_prompt(audio, sr)

    assert len(clipped) == sr * MAX_MUSIC_PROMPT_SECONDS


def test_move_inputs_to_model_casts_float_tensors_to_param_dtype():
    import torch

    from harmonydagger.music_eval import _move_inputs_to_model

    class DummyModel:
        def parameters(self):
            yield torch.zeros(1, dtype=torch.float16)

    inputs = {
        "input_values": torch.zeros(1, 1, 8, dtype=torch.float32),
        "input_ids": torch.ones(1, 4, dtype=torch.long),
    }

    moved = _move_inputs_to_model(inputs, DummyModel())

    assert moved["input_values"].dtype == torch.float16
    assert moved["input_ids"].dtype == torch.long


def test_load_musicgen_requests_half_precision_on_accelerator(monkeypatch):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    from harmonydagger import music_eval

    monkeypatch.setattr(music_eval, "_musicgen_model", None)
    monkeypatch.setattr(music_eval, "_musicgen_processor", None)

    class DummyParam:
        device = "cpu"

    class DummyModel:
        def __init__(self, config):
            self.config = config
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            return self

        def parameters(self):
            yield DummyParam()

    class FakeCompositeConfig:
        decoder = object()
        audio_encoder = type("AE", (), {"sampling_rate": 32000})()

    captured = {}

    def fake_from_pretrained(cls, name, **kwargs):
        captured.update(kwargs)
        return DummyModel(FakeCompositeConfig())

    monkeypatch.setattr(
        AutoProcessor, "from_pretrained", classmethod(lambda cls, name, **kwargs: object())
    )
    monkeypatch.setattr(
        MusicgenForConditionalGeneration,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    monkeypatch.setattr("harmonydagger.gpu.get_device", lambda: "mps")

    _, model = music_eval._load_musicgen()

    assert model.moved_to == "mps"
    assert captured.get("torch_dtype") is not None
    assert captured.get("low_cpu_mem_usage") is True


def test_is_music_available_false_when_transformers_missing(monkeypatch):
    import builtins

    from harmonydagger import music_eval

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers"):
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(music_eval, "_musicgen_model", None)
    monkeypatch.setattr(music_eval, "_musicgen_processor", None)

    assert music_eval.is_music_available() is False


def test_streamlit_app_offers_music_generation_after_upload():
    from streamlit.testing.v1 import AppTest

    from harmonydagger.music_eval import DEFAULT_MUSIC_TEXT

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

    radio_options = [opt for r in at.radio for opt in r.options]
    assert any("MusicGen" in opt for opt in radio_options)

    music_radio = next(r for r in at.radio if any("MusicGen" in opt for opt in r.options))
    music_choice = next(opt for opt in music_radio.options if "MusicGen" in opt)
    music_radio.set_value(music_choice)
    at.run()

    assert not at.exception
    assert not at.error
    assert any(DEFAULT_MUSIC_TEXT in (ta.value or "") for ta in at.text_area)
    assert any(btn.label == "Generate music" for btn in at.button)
