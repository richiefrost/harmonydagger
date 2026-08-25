"""
HarmonyDagger Web Demo - Upload audio and hear the protected version.

Run with: streamlit run streamlit_app.py
"""
from harmonydagger.numba_env import ensure_numba_cache_dir

ensure_numba_cache_dir()

import io
import tempfile

import soundfile as sf
import streamlit as st

from harmonydagger.benchmark import generate_benchmark_report
from harmonydagger.clone_eval import (
    DEFAULT_CLONE_TEXT,
    MAX_CLONE_TEXT_CHARS,
    CloneUnavailableError,
    compare_reference_clones,
    is_clone_available,
)
from harmonydagger.music_eval import (
    DEFAULT_MUSIC_TEXT,
    MAX_MUSIC_TEXT_CHARS,
    MusicUnavailableError,
    compare_reference_music,
    is_music_available,
)
from harmonydagger.core import generate_protected_audio
from harmonydagger.demo_audio import (
    DEMO_UPLOAD_TYPES,
    DemoAudioLoadError,
    read_audio_for_demo,
    suffix_for_demo_upload,
)
from harmonydagger.verify import verify_protection


def _play_generated(title, audio, sample_rate, metric_label, metric_value):
    st.markdown(f"**{title}**")
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    st.audio(buf.getvalue(), format="audio/wav")
    st.metric(metric_label, f"{metric_value:.3f}")


st.set_page_config(page_title="HarmonyDagger Demo", layout="wide")

st.title("HarmonyDagger Audio Protection Demo")
st.markdown(
    "Upload a short audio clip (up to 30 seconds) to hear it with "
    "psychoacoustic protection applied."
)

# Sidebar controls
st.sidebar.header("Protection Settings")
noise_scale = st.sidebar.slider("Noise Scale (strength)", 0.01, 0.5, 0.1, 0.01)
dry_wet = st.sidebar.slider(
    "Dry/Wet Mix", 0.0, 1.0, 1.0, 0.05,
    help="0.0 = original, 1.0 = fully protected"
)
adaptive = st.sidebar.checkbox("Adaptive Scaling", value=True)
vocal_mode = st.sidebar.checkbox("Vocal Mode (voice optimization)", value=False)
use_phase = st.sidebar.checkbox("Phase Perturbation", value=False)
use_temporal = st.sidebar.checkbox("Temporal Masking", value=False)

uploaded_file = st.file_uploader("Upload audio file", type=DEMO_UPLOAD_TYPES)

if uploaded_file is not None:
    suffix = suffix_for_demo_upload(uploaded_file.name)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        audio, sr = read_audio_for_demo(tmp_path)
    except DemoAudioLoadError as exc:
        st.error(str(exc))
        st.stop()

    # Limit to 30 seconds
    max_samples = sr * 30
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        st.warning("Audio trimmed to 30 seconds for the demo.")

    st.subheader("Original Audio")
    st.audio(tmp_path)

    with st.spinner("Applying protection..."):
        protected = generate_protected_audio(
            audio, sr,
            window_size=2048,
            hop_size=512,
            noise_scale=noise_scale,
            adaptive_scaling=adaptive,
            dry_wet=dry_wet,
            vocal_mode=vocal_mode,
            use_phase_perturbation=use_phase,
            use_temporal_masking=use_temporal,
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
        sf.write(out_tmp.name, protected, sr)
        out_path = out_tmp.name

    st.subheader("Protected Audio")
    st.audio(out_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Benchmark")
        bench = generate_benchmark_report(audio, protected, sr)
        st.metric("SNR (dB)", f"{bench['snr_db']:.1f}")
        st.metric("Perturbation Ratio", f"{bench['perturbation_ratio']:.4f}")

    with col2:
        st.subheader("Protection Verification")
        ver = verify_protection(audio, protected, sr)
        st.metric("Protection Score", f"{ver['protection_score']:.3f}")
        st.metric("MFCC Similarity", f"{ver['mfcc_similarity']:.3f}")

    buf = io.BytesIO()
    sf.write(buf, protected, sr, format="WAV")
    buf.seek(0)
    st.download_button("Download Protected Audio", buf, "protected.wav", "audio/wav")

    st.subheader("Generation check")
    mode = st.radio(
        "Model",
        ["Voice clone (XTTS)", "Music (MusicGen-small)"],
        horizontal=True,
    )

    if mode.startswith("Voice"):
        st.caption(
            "This is the actual attack: a TTS model uses your clip as a speaker "
            "reference and speaks new text. Compare the clone from the original "
            "with the clone from the protected audio."
        )
        clone_text = st.text_area(
            "Text for the model to speak",
            value=DEFAULT_CLONE_TEXT,
            max_chars=MAX_CLONE_TEXT_CHARS,
        )
        if not is_clone_available():
            st.info(
                'Voice cloning requires the optional extra: '
                '`pip install -e ".[clone]"` (installs coqui-tts). '
                "The first run downloads the XTTS-v2 model."
            )
        if st.button("Generate clones"):
            try:
                with st.spinner(
                    "Generating clones from original and protected references..."
                ):
                    clones = compare_reference_clones(
                        audio, protected, sr, clone_text
                    )
            except (CloneUnavailableError, ValueError) as exc:
                st.error(str(exc))
            else:
                orig_col, prot_col = st.columns(2)
                with orig_col:
                    _play_generated(
                        "Clone from original",
                        clones["original_clone"],
                        clones["original_clone_sr"],
                        "Similarity to original speaker",
                        clones["original_clone_similarity"],
                    )
                with prot_col:
                    _play_generated(
                        "Clone from protected",
                        clones["protected_clone"],
                        clones["protected_clone_sr"],
                        "Similarity to original speaker",
                        clones["protected_clone_similarity"],
                    )
    else:
        st.caption(
            "This is the music attack: MusicGen-small continues your clip "
            "using a text prompt. Compare the continuation from the original "
            "with the continuation from the protected audio."
        )
        music_text = st.text_area(
            "Text prompt for the music model",
            value=DEFAULT_MUSIC_TEXT,
            max_chars=MAX_MUSIC_TEXT_CHARS,
        )
        if not is_music_available():
            st.info(
                'Music generation requires the optional extra: '
                '`pip install -e ".[music]"` (installs transformers). '
                "The first run downloads facebook/musicgen-small."
            )
        if st.button("Generate music"):
            try:
                with st.spinner(
                    "Generating music from original and protected prompts..."
                ):
                    generations = compare_reference_music(
                        audio, protected, sr, music_text
                    )
            except (MusicUnavailableError, ValueError) as exc:
                st.error(str(exc))
            else:
                orig_col, prot_col = st.columns(2)
                with orig_col:
                    _play_generated(
                        "Music from original",
                        generations["original_music"],
                        generations["original_music_sr"],
                        "Similarity to original clip",
                        generations["original_music_similarity"],
                    )
                with prot_col:
                    _play_generated(
                        "Music from protected",
                        generations["protected_music"],
                        generations["protected_music_sr"],
                        "Similarity to original clip",
                        generations["protected_music_similarity"],
                    )
