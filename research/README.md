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

## Headline result — a negative one

**We could not build a style-mimicry baseline at catalogue scale.** MusicGen either
memorizes individual clips (full-decoder fine-tune, 0.93 CLAP similarity to one clip) or
learns only the artist's marginal token statistics — timbre and register — with nothing in
between.

The decisive test is `shuffle_test.py`: score held-out clips normally, then score them with
the token sequence **shuffled in time**, which destroys all musical structure while
preserving the marginal distribution exactly.

| | real token order | shuffled in time | ratio |
|---|---|---|---|
| LoRA r=32 q/v, 1200 steps | −0.4760 (78% of clips) | **−0.8337** (100%) | 1.75× |
| LoRA r=32 all targets, 3600 steps | −0.4443 (86%) | **−0.9485** (97%) | 2.13× |

The measured gain does not merely survive destroying the music — it **grows**. So the
fine-tune learned a timbre prior, not a style.

**This vindicates the research doc's §1 reframing.** Self-defense ("can they reproduce my
track?") was chosen because dataset-scale style mimicry is structurally unavailable to an
individual artist. Trying to measure style mimicry directly hit the same wall from the
other side. Doc §6's "untested scale" is untested for a substantive reason.

`shuffle_test.py` is now a required **gate**: if shuffled improvement is comparable to
real, the baseline is not learning style and no protection number from it means anything.

### ❌ Retracted: 20.8% style-protection efficacy

An earlier version of this README and dashboard led with a 20.8% efficacy result (paired
t = 5.79, p < 0.001, 18/18 clips defended). **It is retracted as a style result.** The
statistics were sound; the quantity being protected was a timbre prior, not style. It also
explains three anomalies at once — training loss barely descending while held-out improved
*more*, generations worse than base at every guidance scale (CFG ruled out: lower guidance
is worse), and the cross-artist control passing anyway (marginals differ between artists,
so that control cannot separate timbre from style).

The retracted result stays visible on the dashboard rather than being deleted. It is a
useful cautionary example, and hiding it would misrepresent the week.

### What does still stand

- **Bi-level protection tripled the effect** in the doc's own threat model (single-track
  memorization): 8.7 → 29 pts, replicated n=3 (mean 19.6, though sd 9.6 — unreliable).
- **The perturbation is genuinely inaudible** (0.0024–0.078 vs 0.735 for white noise).
- **Three negative results** that constrain the design space — see below.
- But bi-level's damage is **fine-codebook-weighted**, and a 29-pt token gap produced no
  audible difference. Stronger, not different in kind.

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

## Reproducing the measurements

The dashboard shows results; these scripts produce them. All are sequential by necessity —
one MPS device, and concurrent MusicGen fine-tunes trip `command buffer exited with error
status`, which surfaces as a bogus EnCodec tensor-shape error rather than an OOM.

### Read the gates first

Two checks decide whether any protection number is interpretable. Run them **before**
measuring protection, not after.

| Script | Question | Pass condition |
|---|---|---|
| `structure_gate.py` | Did the fine-tune learn temporal **structure**, or only the artist's marginal token distribution? | `ratio < 0.3` |
| `diagnose_memorization.py` | Is the model generalising, or reproducing memorised clips? | gen-vs-heldout must beat the realtrain-vs-heldout bar |

`structure_gate.py` re-measures held-out loss with the token sequences **shuffled in time**.
Shuffling destroys musical structure while preserving marginals exactly, so if shuffled
improvement ≈ real improvement the model learned timbre, not style. **~9 configurations have
now failed this gate** (effective batch 1→32, LoRA rank/targets, per-clip captions,
codebook-1 loss weighting, full-decoder FT at the community recipe) — every one *degrades*
codebook 1. That is why `style_protection` in `findings.json` is marked RETRACTED.

```bash
python research/structure_gate.py --accum 8 --lora-targets attn --steps 600     --save data/gate.json
```

### Measure protection on the threat model that works

Single-track memorization is the one setting where the clean baseline reliably works
(reproduction accuracy ~1.00, gate-passing every run).

```bash
# corpus: CC-licensed, ungated, ships artist/licence metadata
python research/build_corpus.py --list
python research/build_corpus.py --artist "6th Sense" --max-tracks 22     --out-dir data/catalogue/6th_sense_big

# one measurement per process (MPS state accumulates across runs)
python research/replicate_bilevel.py --track <t.wav> --objective clean    --seed 0
python research/replicate_bilevel.py --track <t.wav> --objective baseline --seed 0
python research/replicate_bilevel.py --track <t.wav> --objective bilevel  --seed 0
python research/summarize_replication.py
```

`overnight.sh` drives the whole matrix and is **resumable** — every measurement writes its
own file before the next starts, so a kill loses at most one. This matters: two multi-hour
runs were destroyed by session teardown with nothing saved, and `nohup` does not survive it.

### The sanity invariant

`compare_arms.py` enforces the rule that caught two bad results:

> A protected arm can never legitimately learn **more** than the clean arm. Protection can
> only remove information. Any such row is a broken measurement, not a finding.

It held across all 24 memorization measurements (0/22 negative gaps). The
CLAP-on-generations and averaged-held-out-loss metrics both **violated** it — which is how
the retracted 20.8% and the nonsensical −14.7% "efficacy" arose.

### Metrics that were tried and are wrong

Documented so nobody re-derives them. Each looked reasonable:

| Metric | Why it fails |
|---|---|
| EnCodec token-change rate | Shifting audio by **one sample** (0.03 ms, inaudible, harmless) changes ~31% of tokens — comparable to the strongest protection |
| CLAP similarity on generations | Reported "worse than base" for configs that held-out loss shows generalising; two noisy indirect steps over few samples |
| Averaged held-out loss | Dominated by codebooks 2–4, whose residuals are near-random by construction (cb4 base 7.889 > uniform-over-2048 7.625) |

The through-line: **only codebook 1 carries perceptual content**, so any aggregate over
codebooks measures mostly noise. Judge on codebook 1, and confirm audibility with ears
(doc §8 rule 2).

### Open question: is bi-level actually better, or just louder?

`gap` correlates with perturbation magnitude at **r = +0.88**, and bi-level reaches
`delta_rms` 0.0107 where the baseline objective at `note_frac` 0.06 tops out at 0.0055. So
bi-level's ~2× advantage may be amplitude rather than a smarter objective.

```bash
./research/matched_magnitude.sh   # baseline at note_frac 0.10 / 0.15 / 0.25
```

If the baseline reaches bi-level's gap at comparable **audibility**, bi-level adds nothing
and should be dropped. Resolve this before building on any bi-level number.
