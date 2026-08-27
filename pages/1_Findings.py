"""
Findings dashboard for the music-model protection work.

Structure follows the honest arc of the week: the wall we hit, the gate that proved it,
what got retracted, then what still stands. Retracted results stay on the page rather
than being deleted -- a dashboard that only showed wins would misrepresent the work, and
the negative results are the most transferable thing here.
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
    "3 categorical slots from the validated palette (the all-pairs-safe cap). Every chart "
    "ships direct value labels and a table view."
)
st.markdown(
    "**Short answer: in the memorization setting, measurably yes but not audibly. At "
    "catalogue/style scale we could not build a baseline to protect against at all.** "
    "That second finding is the headline, and it is negative."
)

# ---------------------------------------------------------------- the wall
w = F["the_wall"]
st.header("The wall: no style-mimicry baseline exists at this scale")
st.caption(w["what"])
c1, c2 = st.columns(2)
for col, end in zip((c1, c2), w["two_ends"]):
    with col:
        st.markdown(f"**{end['regime']}**")
        st.markdown(end["outcome"])
st.error("No middle ground was found between memorizing and learning nothing but timbre.",
         icon=":material/block:")
st.markdown(f"**Plausible reason.** {w['plausible_reason']}")
st.success(w["implication"], icon=":material/lightbulb:")

st.divider()

# ---------------------------------------------------------------- shuffle gate
sg = F["shuffle_gate"]
st.header("The gate that proved it: shuffle the tokens in time")
st.caption(sg["what"])
st.markdown(f"_How to read it:_ {sg['how_to_read']}")

runs = sg["runs"]
pick = st.radio("Run", range(len(runs)), horizontal=True,
                format_func=lambda i: runs[i]["config"])
r = runs[pick]
st.pyplot(
    viz.hbar(["real token order", "shuffled in time"], [r["real"], r["shuffled"]],
             dark=dark, fmt="{:+.4f}", highlight=1,
             xlabel="held-out loss change (negative = the model 'learned')"),
    use_container_width=True,
)
k1, k2, k3 = st.columns(3)
k1.metric("shuffled / real", f"{r['ratio']:.2f}×",
          help="Above 1.0 means destroying musical structure IMPROVED the measured gain.")
k2.metric("clips improving, real", f"{r['real_improving_pct']}%")
k3.metric("clips improving, shuffled", f"{r['shuffled_improving_pct']}%")

st.error(sg["conclusion"], icon=":material/priority_high:")
st.info(sg["status"], icon=":material/gpp_maybe:")
with st.expander("Table view · shuffle gate, both runs"):
    st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------- retraction
sp = F["style_protection"]
st.header("Retracted: the 20.8% style-protection result")
rt = sp["retraction"]
st.error(f"**{rt['verdict']}**  \n{rt['reason']}", icon=":material/cancel:")

with st.expander("What the number was, and why it looked convincing"):
    st.markdown(
        f"Measured on held-out token loss. {sp['config']}. Paired "
        f"t = {sp['paired_defence']['t']:.2f}, p < 0.001, "
        f"{sp['paired_defence']['clips_defended']}/{sp['paired_defence']['n']} clips "
        "defended — every one moving the same way."
    )
    st.pyplot(
        viz.hbar([a["arm"] for a in sp["arms"]],
                 [a["heldout_loss_delta"] for a in sp["arms"]], dark=dark,
                 fmt="{:+.4f}", highlight=1, xlabel="change in held-out token loss"),
        use_container_width=True,
    )
    st.pyplot(
        viz.dot_strip(sp["paired_defence"]["per_clip"], dark=dark,
                      xlabel="per-clip defence (positive = protection reduced learning)"),
        use_container_width=True,
    )
    st.caption(
        "The statistics were never the problem. A tight, 100%-consistent effect on a "
        "quantity that turned out to be a timbre prior is still an effect on a timbre prior."
    )

st.markdown("**The retraction explains three anomalies at once:**")
for x in rt["explains"]:
    st.markdown(f"- {x}")

cac = F["cross_artist_control"]
with st.expander(f"The cross-artist control — {cac['status']}"):
    ca = cac["arms"]
    st.pyplot(
        viz.hbar([a["training_data"] for a in ca], [a["heldout_loss_delta"] for a in ca],
                 dark=dark, fmt="{:+.4f}", highlight=1,
                 xlabel="held-out loss on 6th Sense"),
        use_container_width=True,
    )
    st.markdown(
        f"Training on a different artist *degrades* held-out loss "
        f"({ca[1]['heldout_loss_delta']:+.4f}, {ca[1]['sigma']}σ, "
        f"{ca[1]['clips_improving_pct']}% of clips improving) — a "
        f"{cac['separation_nats']:.2f}-nat separation. This does rule out generic domain "
        "adaptation."
    )
    st.warning(cac["superseded_note"], icon=":material/warning:")

st.divider()

# ---------------------------------------------------------------- what stands
st.header("What still stands")
st.caption(
    "Everything below sits in the doc's own threat model — single-track memorization — or "
    "is a measurement result independent of the style question."
)

st.subheader("Bi-level protection tripled the effect")
sm = F["single_track_memorization"]
st.markdown(
    "The doc's objective optimizes through the **encoder only, never the LM**, while it "
    "describes HarmonyCloak as *bi-level*. Restoring that step was the single biggest "
    "improvement, and it explains why the effect previously eroded with training."
)
st.pyplot(
    viz.hbar([a["objective"] for a in sm["arms"]], [a["gap_pts"] for a in sm["arms"]],
             dark=dark, fmt="{:.1f}", highlight=3,
             xlabel="reproduction-accuracy gap (percentage points)"),
    use_container_width=True,
)
rep = sm.get("replication_n24") or sm["replication_n3"]
_arms = {k: v for k, v in rep.items() if isinstance(v, dict) and "mean_gap" in v}

if "bimodal" in rep:
    bm = rep["bimodal"]
    st.error(
        f"**Bi-level is bimodal, not merely noisy.** {bm['worked']['n']} runs cluster at "
        f"{bm['worked']['mean_gap']} pts and {bm['thrashed']['n']} at "
        f"{bm['thrashed']['mean_gap']} pts, with almost nothing between — and it is "
        f"seed-dependent, not track-dependent ({bm['seed_not_track_dependent']}). "
        f"{bm['cause']}",
        icon=":material/warning:",
    )
    pd_ = rep["paired_difference"]
    c1, c2, c3 = st.columns(3)
    c1.metric("paired difference", f"+{pd_['mean_pts']} pts", f"sem {pd_['sem']}")
    c2.metric("bi-level beat baseline", pd_["bilevel_beat_baseline"])
    c3.metric("bi-level gave <3 pts", pd_["bilevel_no_protection_lt3pts"])

if "magnitude_confound" in rep:
    mc = rep["magnitude_confound"]
    st.warning(
        f"**Open question — is bi-level better, or just louder?** {mc['concern']} "
        f"(gap vs perturbation magnitude: r = +{mc['corr_gap_vs_delta_rms']['bilevel']}). "
        f"{mc['test_in_flight']}",
        icon=":material/help:",
    )

if "sanity_invariant" in rep:
    si = rep["sanity_invariant"]
    st.success(f"**Sanity invariant.** {si['rule']} — {si['result']}",
               icon=":material/check_circle:")

with st.expander(f"Table view · replication (n={sum(v['n'] for v in _arms.values())} measurements)"):
    st.dataframe(
        pd.DataFrame([
            {"objective": k, "n": v["n"], "mean gap (pts)": v["mean_gap"], "sd": v["sd"],
             "range": f"{v['range'][0]}–{v['range'][1]}",
             "mean audibility": v.get("mean_audibility")}
            for k, v in _arms.items()
        ]),
        hide_index=True, use_container_width=True,
    )
    if "caveat" in rep:
        st.caption(rep["caveat"])

st.subheader("…but the damage lands in the wrong codebooks")
st.caption(
    "MusicGen uses 4 codebooks. Codebook 1 is coarse and carries perceptual content; 4 is "
    "fine detail. Every objective is fine-weighted — the codebook that decides whether a "
    "mimic gets a recognizable copy is the least damaged."
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

st.subheader("The perturbation is genuinely inaudible")
ia = F["inaudibility"]
st.caption(ia["metric"] + "  \n" + ia["note"])
rows = [f"level {l['level']} (note_frac {l['note_frac']})" for l in ia["levels"]]
rows += [x["what"] for x in ia["references"]]
vals = [l["audibility"] for l in ia["levels"]] + [x["audibility"] for x in ia["references"]]
st.pyplot(
    viz.hbar(rows, vals, dark=dark, fmt="{:.3f}", xlabel="audibility (0 = fully masked)"),
    use_container_width=True,
)
st.caption(
    "Level B is the usable ceiling per the doc's human A/B listening — two orders of "
    "magnitude below audible noise on the same metric. This part was never in doubt. The "
    "difficulty is making an inaudible perturbation *matter*."
)

st.divider()

# ---------------------------------------------------------------- negatives
st.header("Negative results that constrain the design space")
st.caption("The most transferable output of the week: things the next person need not retry.")

t1, t2, t3 = st.tabs(["Token metric is meaningless", "Decoy / impersonation",
                      "Coarse targeting"])

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

# ---------------------------------------------------------------- retracted metric
st.header("A metric we also retracted — and should not have")
cg = F["clap_generation_metric_failure"]
st.caption(cg["what"])
sw = cg["sweep_lift_over_base"]
st.pyplot(
    viz.hbar([s["config"] for s in sw], [s["lift"] for s in sw], dark=dark, fmt="{:+.3f}",
             xlabel="CLAP style lift over the base model"),
    use_container_width=True,
)
st.markdown(
    f"{cg['conclusion']}  \n\nCLAP itself is fine — on *real* audio it separates these "
    f"artists cleanly (within-artist "
    f"{cg['clap_discrimination_check']['within_6th_sense']:.3f}/"
    f"{cg['clap_discrimination_check']['within_biedermann']:.3f} vs across "
    f"{cg['clap_discrimination_check']['across_artists']:.3f})."
)
st.warning(
    "In hindsight this metric was **right** that the fine-tunes were not learning style. "
    "We overrode it because held-out loss looked cleaner and more sensitive. The shuffle "
    "gate later showed CLAP had been pointing at something real all along — a reminder "
    "that a metric disagreeing with your preferred one is a hypothesis, not noise.",
    icon=":material/warning:",
)
with st.expander("Table view · sweep"):
    st.dataframe(pd.DataFrame(sw), hide_index=True, use_container_width=True)

st.divider()
st.header("Performance on a laptop")
p = F["performance"]
st.caption(p["machine"] + " — " + p["note"])
st.dataframe(pd.DataFrame(p["items"]).rename(columns={"op": "operation"}),
             hide_index=True, use_container_width=True)
