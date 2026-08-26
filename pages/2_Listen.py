"""
Listening page. The claims on the Findings page are objective metrics; this page is where
you check them by ear, because the project's own rule is that humans arbitrate audibility.

Audio is bundled as FLAC deliberately. MP3/OGG would alter or strip the perturbation --
which is the very thing the first section asks you to fail to hear.
"""
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Listen · HarmonyDagger research", layout="wide")

ROOT = Path(__file__).resolve().parent.parent / "research"
MAN = json.loads((ROOT / "audio_manifest.json").read_text())
F = json.loads((ROOT / "findings.json").read_text())


def play(key, caption=None):
    entry = MAN.get(key)
    if not entry:
        st.caption(f"_(missing: {key})_")
        return
    p = ROOT / entry["file"]
    if not p.exists():
        st.caption(f"_(missing file: {entry['file']})_")
        return
    st.audio(str(p))
    if caption:
        st.caption(caption)


st.title("Hear it for yourself")
st.markdown(
    "Two different questions, and they need separate listening. **Section 1** asks whether "
    "the protection is inaudible — if you can hear it, the approach fails its own "
    "non-negotiable constraint. **Section 2** asks whether protection degrades what a "
    "mimic produces."
)

st.divider()
st.header("1 · Is the protection inaudible?")
st.caption(
    f"Level B (note_frac 0.06). Catalogue mean audibility "
    f"{F['inaudibility']['catalogue_mean']:.4f} on the doc's validated masking metric, "
    "where white noise scores 0.735."
)

for i in (0, 1):
    st.subheader(f"Pair {i + 1}")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Original**")
        play(f"inaudible_{i}_original")
    with b:
        st.markdown("**Protected** — what the artist would release")
        play(f"inaudible_{i}_protected")
    with c:
        st.markdown("**The perturbation alone, ×20**")
        play(f"inaudible_{i}_pert_x20",
             "Diagnostic only. Amplified 20× so you can hear WHERE it sits — it should "
             "ride the loud tonal hits, which is what the masking ceiling permits.")

st.subheader("Objectives compared, same track")
st.caption(
    "Bi-level uses ~50% more perturbation energy than the baseline objective at comparable "
    "audibility, so this is where inaudibility is most likely to break."
)
a, b, c = st.columns(3)
with a:
    st.markdown("**Original**")
    play("bilevel_original")
with b:
    st.markdown("**Baseline objective** — audibility 0.0060")
    play("baseline_protected")
with c:
    st.markdown("**Bi-level** — audibility 0.0081")
    play("bilevel_protected")

st.divider()
st.header("2 · Does protection degrade what the mimic produces?")

st.subheader("Memorization setting — one track, 29-point token gap")
st.caption(
    "A model fine-tuned to memorize a single track, then asked to generate with no prime "
    "(so no real audio leaks in). The protected arm scored a 29-point token gap — more "
    "than triple the baseline."
)
a, b, c = st.columns(3)
with a:
    st.markdown("**The original track**")
    play("memo_original")
with b:
    st.markdown("**Clean-trained model's output**")
    play("memo_gen_clean")
with c:
    st.markdown("**Protected-trained model's output**")
    play("memo_gen_protected")
st.info(
    "Expected outcome: **no clear difference.** Log-mel distance from the original is 4.35 "
    "vs 4.41 — indistinguishable. This is the finding, not a failure of the demo: token-space "
    "damage does not translate into audible protection.",
    icon=":material/info:",
)

st.subheader("Style setting — LoRA on a 22-track catalogue")
st.caption(
    "Unprimed generation from the text prompt only. `base` is the pretrained model with no "
    "fine-tuning at all — the floor."
)
st.error(
    "**The result this arm was built to demonstrate has been RETRACTED.** The shuffle gate "
    "showed this fine-tune learned the artist's marginal token statistics (timbre and "
    "register), not their music — held-out improvement GROWS by 1.75–2.13x when the token "
    "sequence is shuffled in time, which destroys all musical structure. So neither arm "
    "here is a style mimic, and the clean-vs-protected comparison does not measure style "
    "protection. Kept on the page because what these actually sound like is the most "
    "direct evidence for that conclusion. See the Findings page.",
    icon=":material/cancel:",
)
idx = st.radio("Sample", [0, 1, 2, 3], horizontal=True,
               format_func=lambda i: f"sample {i + 1}")
a, b, c = st.columns(3)
with a:
    st.markdown("**Base** — no fine-tuning")
    play(f"style_gen_base_{idx}")
with b:
    st.markdown("**Clean-trained**")
    play(f"style_gen_clean_{idx}")
with c:
    st.markdown("**Protected-trained**")
    play(f"style_gen_protected_{idx}")

st.warning(
    "**What you are hearing.** Both arms are mediocre, and neither sounds like the artist. "
    "That is not a demo bug — it is the finding. A fine-tune that shifts output timbre "
    "without learning musical structure produces exactly this: audio in roughly the right "
    "register that is not the artist's music. The CLAP-of-generations metric flagged it "
    "early (every config scored below base) and we overrode it; your ears and CLAP were "
    "both right.",
    icon=":material/warning:",
)

st.divider()
st.caption(
    "Blind yourself if you can: have someone rename the files before you compare. Every "
    "listening result in this project so far was non-blind, which is a real limitation."
)
