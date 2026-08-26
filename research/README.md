# Music-model protection research (hackweek)

Can we perturb music so a generative model can't learn an artist's style, while it still
sounds identical to a human?

This directory holds the research-method implementation and a Streamlit dashboard over
everything measured. It is **separate from the `harmonydagger` package**: that implements
the original psychoacoustic heuristics, while this implements the research doc's method —
a perturbation optimized against MusicGen's own EnCodec encoder, with a psychoacoustic
masking ceiling projected **inside** the optimization loop.

## Run it

```bash
python3.11 -m venv .venv && ./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e ".[streamlit]" pandas matplotlib
./.venv/bin/streamlit run streamlit_app.py
```

Three pages appear in the sidebar:

| Page | What it does | Needs |
|---|---|---|
| **Findings** | Dashboard of every measured result, with its caveats | streamlit, pandas, matplotlib |
| **Listen** | A/B the bundled audio — inaudibility, then model outputs | streamlit |
| **Protect Audio** | Protect a clip live with the research method | `.[music]` (torch + transformers≥5) |

Findings and Listen work with no ML dependencies at all — useful for a demo on any laptop.

### ⚠️ The `music` and `clone` extras cannot coexist

`clone` (the XTTS voice-clone demo on the main page) pins `transformers>=4.57,<5`.
The research code needs `transformers>=5` — it relies on the 5.x `attn_implementation`
handling and the renamed `ClapProcessor(audio=...)` kwarg. Install one or the other:

```bash
pip install -e ".[streamlit,music]"    # music-model research pages
pip install -e ".[streamlit,clone]"    # voice-clone demo on the main page
```

## Headline result

**20.8% style-protection efficacy** (paired t = 5.79, p < 0.001, 18/18 held-out clips
defended) from an **inaudible** perturbation (audibility 0.0165, where white noise scores
0.735 on the same metric).

That is: an inaudible perturbation removed about a fifth of what a LoRA fine-tune learned
about one artist. It is **not** prevention, and it is n=1 artist / 1 seed / 1 config.

## What is in here

| File | Purpose |
|---|---|
| `protection.py` | The doc's §5.2 perturbation + §5.3 validated masking-audibility metric |
| `protection_bilevel.py` | Training-aware bi-level objective (the Mist-v2 mechanism) |
| `protection_coarse.py` | Codebook-1 targeting — **negative result**, kept for the record |
| `protection_decoy.py` | Targeted impersonation — **negative result**, kept for the record |
| `style_metric.py` | CLAP style similarity (see the caveat below) |
| `viz.py` | Charts. Validated 3-slot palette, direct labels + table views throughout |
| `findings.json` | Every measured number, with its limits |
| `assets/audio/*.flac` | Bundled audio for the Listen page |

**Audio is FLAC on purpose.** MP3 or OGG would alter or strip the perturbation — which is
exactly what the Listen page asks you to fail to hear. ~8 MB total.

## Two measurement traps this project fell into

Both are worth knowing before trusting any number in this space.

**1. Token-space damage is not protection.** Shifting audio by **one sample** (0.03 ms —
inaudible, harmless) changes ~31% of EnCodec tokens, comparable to our strongest
perturbation. We tripled the token gap (8.7 → 29 pts) and it bought *zero* audible
difference (log-mel 4.35 vs 4.41; a listener could not tell them apart).

**2. CLAP-on-generations reported the opposite of the truth.** It scored *every* fine-tune
config below the base model — including the one that provably generalizes at 5.5 sigma on
held-out loss. Generation plus a coarse embedding on a handful of samples is two noisy
indirect steps. CLAP itself is fine: on *real* audio it separates these artists cleanly
(within-artist 0.70/0.94 vs across 0.46). It breaks when applied to generated samples.

The fix in both cases was **held-out token loss, paired per clip**. Pairing mattered: the
unpaired means (0.476 vs 0.377, sems ≈ 0.09) would not have separated, while the paired
comparison is 100% consistent. And held-out *clips* beat held-out *tracks* — with 4 tracks
the paired sem was 0.28 and the deltas did not even share a sign.

## What did not work, and why it constrains the design space

- **Coarse (codebook-1) targeting.** 3× more codebook-1 damage at the input, yet the
  trained model reproduced coarse structure *better*. Input damage was anti-correlated
  with learned damage.
- **Decoy / impersonation (Mist's mechanism).** The latent will not move. Even
  unconstrained and audibly destroyed (audibility 0.93) it closes only 29% of the distance
  to another song. EnCodec's latent is a near-invertible acoustic code at **5×**
  compression; Stable Diffusion's VAE — where Mist works — is **48×** and semantically
  lossy. `latent(A) ≈ latent(B)` essentially requires `A ≈ B`.
- **Bi-level is stronger but unreliable.** 2.27× the mean gap of the baseline objective,
  but it was *worse* than baseline on one of three tracks (sd 9.6 vs 1.1). The alternating
  game oscillates rather than converging.

## Reproducing

The full experiment scripts (corpus builder, fine-tune harness, sweeps, replication)
live outside this repo in the working tree used for the runs. `findings.json` records
what each number came from. Everything ran on an M5 Pro / 64 GB / MPS with no cloud GPU:
protection 16 s per clip (198 s for bi-level), one fine-tune ~4.4 min, generation 13 s.
