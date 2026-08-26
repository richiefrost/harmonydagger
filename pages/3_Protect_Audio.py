"""
Interactive protection using the RESEARCH method (research/protection.py), as distinct
from the main page, which demos the original HarmonyDagger heuristics.

The difference matters: this optimizes a perturbation against MusicGen's own EnCodec
encoder with a psychoacoustic masking ceiling projected inside the optimization loop, and
scores it with the doc's validated masking metric rather than SNR.

Needs torch + transformers >= 5 (see research/README.md -- this conflicts with the
`clone` extra, which pins transformers < 5).
"""
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="Protect audio · research method", layout="wide")

ROOT = Path(__file__).resolve().parent.parent / "research"

st.title("Protect audio — research method")
st.caption(
    "Optimizes the perturbation against MusicGen's EnCodec encoder, with the masking "
    "ceiling applied every step. Scored with the validated masking-audibility metric "
    "(0 = fully masked), not SNR."
)


@st.cache_resource(show_spinner="Loading MusicGen encoder (first run downloads ~2.2 GB)...")
def _encoder():
    import torch
    from transformers import MusicgenForConditionalGeneration

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = (
        MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        .to(dev)
        .eval()
    )
    for p in m.parameters():
        p.requires_grad_(False)
    return m, dev


try:
    import torch  # noqa: F401
    import transformers

    _tv = int(transformers.__version__.split(".")[0])
except Exception:
    st.error(
        "This page needs `torch` and `transformers>=5`.\n\n"
        "```\npip install -e \".[music]\"\n```",
        icon=":material/error:",
    )
    st.stop()

if _tv < 5:
    st.error(
        f"transformers {transformers.__version__} is installed, but this page needs >= 5. "
        "The `clone` extra pins `transformers<5`; the two cannot coexist in one env. "
        "See research/README.md.",
        icon=":material/error:",
    )
    st.stop()

from research.protection import LEVELS, SR, audibility, protect  # noqa: E402
from research.viz import spectrogram_diff  # noqa: E402

# ---- input ----
src = st.radio("Audio source", ["Bundled sample", "Upload"], horizontal=True)
audio = None

if src == "Bundled sample":
    import json

    man = json.loads((ROOT / "audio_manifest.json").read_text())
    samples = {
        "6th Sense — Press Play (instrumental)": "inaudible_0_original",
        "6th Sense — 28 Chasing 1988 (instrumental)": "inaudible_1_original",
    }
    pick = st.selectbox("Sample", list(samples))
    key = samples[pick]
    if key in man:
        import soundfile as sf

        audio, file_sr = sf.read(str(ROOT / man[key]["file"]))
        if file_sr != SR:
            import librosa

            audio = librosa.resample(audio.astype("float64"), orig_sr=file_sr, target_sr=SR)
else:
    up = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "mp3", "m4a"])
    if up is not None:
        import tempfile

        from harmonydagger.demo_audio import read_audio_for_demo, suffix_for_demo_upload

        with tempfile.NamedTemporaryFile(
            suffix=suffix_for_demo_upload(up.name), delete=False
        ) as tmp:
            tmp.write(up.read())
            tmp_path = tmp.name
        audio, file_sr = read_audio_for_demo(tmp_path)
        if file_sr != SR:
            import librosa

            audio = librosa.resample(audio.astype("float64"), orig_sr=file_sr, target_sr=SR)

if audio is None:
    st.info("Pick a bundled sample or upload a clip to begin.", icon=":material/upload:")
    st.stop()

audio = np.asarray(audio, dtype=np.float64)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
# Match the doc's convention: 8s, peak-normalised to 0.7. All reported numbers assume it.
seconds = st.slider("Seconds to protect", 4, 16, 8,
                    help="The research used 8s excerpts; all reported numbers assume that.")
audio = audio[: int(SR * seconds)]
peak = np.abs(audio).max()
if peak > 0:
    audio = audio / peak * 0.7

level = st.select_slider(
    "Protection level",
    options=["A", "B", "C"],
    value="B",
    format_func=lambda k: f"{k}  (note_frac {LEVELS[k]})",
    help="B is the usable ceiling per the doc's human A/B listening. C is audible on some material.",
)
steps = st.slider("Optimization steps", 20, 200, 80, step=20)

if st.button("Protect", type="primary"):
    model, dev = _encoder()
    with st.spinner(f"Optimizing perturbation ({steps} steps)..."):
        prot = protect(model.audio_encoder, audio, SR, dev, steps=steps,
                       note_frac=LEVELS[level])
    delta = prot - audio
    aud = audibility(audio, delta, SR)
    snr = 10 * np.log10(np.mean(audio**2) / max(np.mean(delta**2), 1e-20))

    a, b, c = st.columns(3)
    a.metric("Audibility", f"{aud:.4f}",
             help="0 = fully masked. White noise scores 0.735 on this metric.")
    b.metric("SNR", f"{snr:.1f} dB",
             help="Reported for comparison only — the doc shows SNR disagrees with human ears.")
    clipped = float(np.mean(np.abs(prot) >= 0.999))
    c.metric("Samples clipped", f"{clipped:.2%}",
             delta="ok" if clipped < 1e-4 else "audible distortion",
             delta_color="normal" if clipped < 1e-4 else "inverse")

    if aud < 0.02:
        st.success("Inside the inaudible range.", icon=":material/check_circle:")
    elif aud < 0.10:
        st.warning("Borderline — verify by ear on this material.", icon=":material/warning:")
    else:
        st.error("Likely audible. Lower the level.", icon=":material/error:")

    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.audio(audio, sample_rate=SR)
    with right:
        st.markdown("**Protected**")
        st.audio(prot, sample_rate=SR)

    st.markdown("**The perturbation alone, ×20** — diagnostic, so you can hear where it sits")
    st.audio(np.clip(delta * 20, -1, 1), sample_rate=SR)

    st.subheader("Where the perturbation lives")
    st.caption(
        "Energy should track the loud tonal content — that is what the critical-band "
        "masking ceiling permits, and it is also why the perturbation survives MP3."
    )
    fig = spectrogram_diff(audio, prot, SR, dark=st.sidebar.toggle("Dark charts", value=False))
    if fig is not None:
        st.pyplot(fig, use_container_width=True)

    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, prot, SR, format="FLAC")
    buf.seek(0)
    st.download_button("Download protected audio (FLAC)", buf, "protected.flac",
                       "audio/flac",
                       help="FLAC, not MP3 — lossy encoding would alter the perturbation.")

st.divider()
st.caption(
    "Bi-level protection (the Mist-v2-style training-aware objective) is not exposed here: "
    "it fine-tunes a surrogate model and takes ~3.3 min per clip. See "
    "`research/protection_bilevel.py`."
)
