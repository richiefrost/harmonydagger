"""
Findings dashboard for the music-model protection work.

Every number here was measured on an M5 Pro / MPS. Where a result is underpowered or
was later invalidated, this page says so on the same screen as the number -- the
project's own ground rule is "claim only what you measured".
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from research import viz

st.set_page_config(page_title="Findings · HarmonyDagger research", layout="wide")

ROOT = Path(__file__).resolve().parent.parent / "research"
F = json.loads((ROOT / "findings.json").read_text())

st.title("Can we stop a music model from learning an artist?")
dark = st.sidebar.toggle("Dark charts", value=False)
st.sidebar.caption(
    "Charts use 3 categorical slots from the validated palette (the all-pairs-safe cap). "
    "Every chart ships direct value labels and a table view."
)

# ---------------------------------------------------------------- headline
sp = F["style_protection"]
st.header("Headline: 20.8% style-protection efficacy")
c1, c2, c3 = st.columns(3)
c1.metric("Protection efficacy", f"{sp['efficacy_pct']:.1f}%",
          help="Share of what the mimic learned about this artist that protection removed.")
c2.metric("Held-out clips defended",
          f"{sp['paired_defence']['clips_defended']}/{sp['paired_defence']['n']}")
c3.metric("Perturbation audibility", f"{F['inaudibility']['catalogue_mean']:.4f}",
          help="0.000 = fully masked. White noise scores 0.735 on the same metric.")

st.markdown(
    f"An **inaudible** perturbation removed about a fifth of what a LoRA fine-tune "
    f"learned about this artist. Paired t = **{sp['paired_defence']['t']:.2f}**, "
    f"p < 0.001, and every one of {sp['paired_defence']['n']} held-out clips moved the "
    "same way."
)

st.subheader("Held-out loss: did the mimic learn the artist?")
st.caption(
    "More negative = the model got better at predicting UNSEEN tracks by this artist, "
    "i.e. it learned a style rather than memorizing clips. Protection should pull this "
    "toward zero."
)
arms = sp["arms"]
st.pyplot(
    viz.hbar([a["arm"] for a in arms], [a["heldout_loss_delta"] for a in arms],
             dark=dark, fmt="{:+.4f}", highlight=1,
             xlabel="change in held-out token loss (negative = learned the artist)"),
    use_container_width=True,
)

st.subheader("Why it resolves: pair every clip")
st.caption(
    "Each dot is one held-out clip: how much LESS the protected-trained model learned "
    "than the clean-trained one, on that same clip. The unpaired means "
    "(0.476 vs 0.377, sems ~0.09) would not have separated; pairing cancels "
    "between-clip difficulty."
)
st.pyplot(
    viz.dot_strip(sp["paired_defence"]["per_clip"], dark=dark,
                  xlabel="per-clip defence (positive = protection worked)"),
    use_container_width=True,
)

with st.expander("Table view · paired per-clip defence"):
    st.dataframe(
        pd.DataFrame({
            "held-out clip": range(1, len(sp["paired_defence"]["per_clip"]) + 1),
            "defence (loss units)": sp["paired_defence"]["per_clip"],
        }),
        hide_index=True, use_container_width=True,
    )

st.warning(
    "**Limits.** " + "  \n".join(f"– {x}" for x in sp["limits"]),
    icon=":material/warning:",
)

st.divider()

# ---------------------------------------------------------------- control
st.header("Control: is the style learning artist-specific?")
cac = F["cross_artist_control"]
st.caption(cac["what"])
st.markdown(f"_Why we suspected it:_ {cac['why_suspected']}")
st.markdown(f"**Design.** {cac['design']}")
ca = cac["arms"]
st.pyplot(
    viz.hbar([a["training_data"] for a in ca], [a["heldout_loss_delta"] for a in ca],
             dark=dark, fmt="{:+.4f}", highlight=1,
             xlabel="held-out loss on 6th Sense (negative = learned this artist)"),
    use_container_width=True,
)
k1, k2, k3 = st.columns(3)
k1.metric("Separation", f"{cac['separation_nats']:.2f} nats")
k2.metric("Control significance", f"{ca[1]['sigma']:.1f}\u03c3")
k3.metric("Control clips improving", f"{ca[1]['clips_improving_pct']}%")
st.success(cac["conclusion"], icon=":material/verified:")
st.caption("Caveat: " + cac["caveat"])
with st.expander("Table view \u00b7 cross-artist control"):
    st.dataframe(pd.DataFrame(ca), hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------- inaudibility
st.header("Is the protection inaudible?")
ia = F["inaudibility"]
st.caption(ia["metric"] + "  \n" + ia["note"])
rows = [f"level {l['level']} (note_frac {l['note_frac']})" for l in ia["levels"]]
rows += [r["what"] for r in ia["references"]]
vals = [l["audibility"] for l in ia["levels"]] + [r["audibility"] for r in ia["references"]]
st.pyplot(
    viz.hbar(rows, vals, dark=dark, fmt="{:.3f}", xlabel="audibility (0 = fully masked)"),
    use_container_width=True,
)
st.info(
    "Level B is the usable ceiling per the doc's human A/B listening. The reference rows "
    "are what the same metric gives for audible noise — the protection sits two orders of "
    "magnitude below them.",
    icon=":material/info:",
)

st.divider()

# ---------------------------------------------------------------- bi-level
st.header("Bi-level protection: the Mist-v2 mechanism")
sm = F["single_track_memorization"]
st.caption(sm["what"] + f"  \n_{sm['track']}_")
st.markdown(
    "The doc's objective optimizes through the **encoder only, never the LM**, while it "
    "describes HarmonyCloak as *bi-level*. Restoring that step was the single biggest "
    "improvement — and it explains why the effect previously eroded with training."
)
labels = [a["objective"] for a in sm["arms"]]
st.pyplot(
    viz.hbar(labels, [a["gap_pts"] for a in sm["arms"]], dark=dark, fmt="{:.1f}",
             highlight=3, xlabel="reproduction-accuracy gap (percentage points)"),
    use_container_width=True,
)

st.subheader("But the damage lands in the wrong codebooks")
st.caption(
    "MusicGen uses 4 codebooks. Codebook 1 is coarse and carries perceptual content; "
    "4 is fine detail. Damage should ideally be flat or coarse-weighted — it is neither."
)
SHORT = {
    "baseline (encoder-only, doc §5.2)": "baseline",
    "bi-level, error-minimizing": "bi-level (min)",
    "bi-level, error-maximizing": "bi-level (max)",
    "research doc, n=60 mean": "doc n=60",
}
sel = [a for a in sm["arms"] if a["per_codebook_gap"]][:3]
st.pyplot(
    viz.grouped_hbar(
        ["codebook 1\n(coarse)", "codebook 2", "codebook 3", "codebook 4\n(finest)"],
        [SHORT.get(a["objective"], a["objective"][:16]) for a in sel],
        [a["per_codebook_gap"] for a in sel],
        dark=dark, xlabel="accuracy gap (pts)",
    ),
    use_container_width=True,
)
with st.expander("Table view · per-codebook gap"):
    st.dataframe(
        pd.DataFrame(
            {"objective": [SHORT.get(a["objective"], a["objective"]) for a in sel]}
            | {f"cb{i+1}": [a["per_codebook_gap"][i] for a in sel] for i in range(4)}
        ),
        hide_index=True, use_container_width=True,
    )
st.markdown(
    "Every objective is **fine-weighted**: the coarse codebook that determines whether a "
    "mimic gets a recognizable copy is the least damaged. This is why a 29-point token "
    "gap produced no audible difference."
)

rep = sm["replication_n3"]
with st.expander("Replication (n=3 tracks) · bi-level is better on average but unreliable"):
    st.dataframe(
        pd.DataFrame([
            {"objective": k, "n": v["n"], "mean gap (pts)": v["mean_gap"],
             "sd": v["sd"], "range": f"{v['range'][0]}–{v['range'][1]}"}
            for k, v in rep.items() if isinstance(v, dict)
        ]),
        hide_index=True, use_container_width=True,
    )
    st.caption(rep["caveat"])

st.divider()

# ---------------------------------------------------------------- what failed
st.header("What we tried that did not work")
st.caption("Recorded because the negative results constrain the design space.")

t1, t2, t3 = st.tabs(["Token metric is meaningless", "Decoy / impersonation", "Coarse targeting"])

with t1:
    mc = F["metric_calibration"]
    st.metric("Tokens changed by shifting audio ONE SAMPLE",
              f"{mc['one_sample_shift_token_change_pct']:.1f}%")
    st.markdown(mc["note"])
    lm = mc["logmel_from_original"]
    st.pyplot(
        viz.hbar(
            ["clean-trained generation", "protected-trained generation",
             "identical (floor)", "unrelated track (chance)"],
            [lm["clean_trained_gen"], lm["protected_trained_gen"],
             lm["doc_reference"]["identical"], lm["doc_reference"]["unrelated_track"]],
            dark=dark, fmt="{:.2f}", xlabel="log-mel distance from the original",
        ),
        use_container_width=True,
    )
    st.markdown(
        "Clean- and protected-trained generations are **indistinguishable** on this "
        "measure (4.35 vs 4.41), and a listener could not tell them apart either."
    )

with t2:
    fd = F["failed_decoy"]
    st.markdown(fd["what"])
    st.pyplot(
        viz.hbar([s["constraint"] for s in fd["sweep"]],
                 [s["gap_closed_pct"] for s in fd["sweep"]], dark=dark, fmt="{:.0f}%",
                 xlabel="% of the way to impersonating a different song"),
        use_container_width=True,
    )
    a, b = st.columns(2)
    a.metric("EnCodec compression", f"{fd['encodec_compression']:.0f}×",
             help="near-invertible acoustic code")
    b.metric("Stable Diffusion VAE", f"{fd['sd_vae_compression']:.0f}×",
             help="semantically lossy — where Mist works")
    st.error(fd["conclusion"], icon=":material/block:")

with t3:
    fc = F["failed_coarse_targeting"]
    st.markdown(fc["what"])
    st.pyplot(
        viz.grouped_hbar(
            ["codebook-1 damage\nat the INPUT", "codebook-1 gap\nthe model LEARNED"],
            ["baseline", "coarse-targeted"],
            [[fc["input_token_damage_cb1"]["baseline"], fc["learned_gap_cb1"]["baseline"]],
             [fc["input_token_damage_cb1"]["coarse_targeted"],
              fc["learned_gap_cb1"]["coarse_targeted"]]],
            dark=dark, xlabel="percent / percentage points",
        ),
        use_container_width=True,
    )
    st.error(fc["conclusion"], icon=":material/block:")

st.divider()

# ---------------------------------------------------------------- retracted
st.header("A metric we retracted")
cg = F["clap_generation_metric_failure"]
st.caption(cg["what"])
sw = cg["sweep_lift_over_base"]
st.pyplot(
    viz.hbar([s["config"] for s in sw], [s["lift"] for s in sw], dark=dark,
             fmt="{:+.3f}", xlabel="CLAP style lift over the base model"),
    use_container_width=True,
)
st.markdown(
    f"**{cg['conclusion']}**  \n\nCLAP itself is fine — on *real* audio it separates these "
    f"artists cleanly (within-artist {cg['clap_discrimination_check']['within_6th_sense']:.3f}"
    f"/{cg['clap_discrimination_check']['within_biedermann']:.3f} vs across "
    f"{cg['clap_discrimination_check']['across_artists']:.3f}). It breaks when applied to a "
    "handful of generated samples."
)
with st.expander("Table view · sweep"):
    st.dataframe(pd.DataFrame(sw), hide_index=True, use_container_width=True)

st.divider()
st.header("Performance on a laptop")
p = F["performance"]
st.caption(p["machine"] + " — " + p["note"])
st.dataframe(
    pd.DataFrame(p["items"]).rename(columns={"op": "operation", "seconds": "seconds"}),
    hide_index=True, use_container_width=True,
)
