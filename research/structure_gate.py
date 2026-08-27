#!/usr/bin/env python3
"""
THE GATE for any style-mimicry baseline: did the fine-tune learn TEMPORAL STRUCTURE, or
only the artist's marginal token distribution (timbre/register)?

Measures held-out loss change two ways -- with real token order, and with token sequences
SHUFFLED IN TIME. Shuffling destroys all musical structure while preserving marginals
exactly, so:

    ratio = shuffled_improvement / real_improvement

    ratio >= 0.7  -> MARGINALS ONLY. Not a style baseline. Any protection number measured
                     against it is protecting a timbre prior. (This is what the default
                     config does: real -0.476, shuffled -0.834, ratio 1.75.)
    ratio <  0.3  -> learned real structure. Usable style baseline.

Run this BEFORE measuring protection. It would have saved us a day.

Usage:
  python structure_gate.py --lora-targets all --steps 3600 --window-s 8 --hop-s 4
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from catalogue import _fine_tune, load_windows
from finetune_eval import DECODER_START, MODEL_ID, PROMPT, tokens_of

ATTN = ["q_proj", "v_proj"]
ALL = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

ap = argparse.ArgumentParser()
ap.add_argument("--artist-dir", default="data/catalogue/6th_sense_big")
ap.add_argument("--holdout", type=int, default=6)
ap.add_argument("--window-s", type=float, default=8.0)
ap.add_argument("--hop-s", type=float, default=None, help="< window_s gives overlap")
ap.add_argument("--max-windows", type=int, default=3)
ap.add_argument("--lora-rank", type=int, default=32)
ap.add_argument("--lora-targets", choices=["attn", "all"], default="attn")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--accum", type=int, default=1,
                help="clips averaged per optimizer update (effective batch size). "
                     "audiocraft uses 128-192; batch 1 learns marginals only.")
ap.add_argument("--no-cosine", action="store_true")
ap.add_argument("--weight-decay", type=float, default=1e-5)
ap.add_argument("--codebook-weights", help="EXPERIMENT A: comma-separated, e.g. 4,1,1,1")
ap.add_argument("--captions", choices=["single", "titles"], default="single",
                help="EXPERIMENT B: titles uses each track real title as its caption")
ap.add_argument("--save")
a = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tracks = sorted(Path(a.artist_dir).glob("*.wav"))
hold, train = tracks[: a.holdout], tracks[a.holdout :]
kw = dict(window_s=a.window_s, hop_s=a.hop_s, max_windows=a.max_windows)
train_wavs = [w for t in train for w in load_windows(str(t), **kw)]
hold_wavs = [w for t in hold for w in load_windows(str(t), **kw)]
if not train_wavs or not hold_wavs:
    raise SystemExit(
        f"empty split: {len(train_wavs)} train / {len(hold_wavs)} held-out clips. "
        f"window_s={a.window_s} is probably longer than the source tracks."
    )
targets = ALL if a.lora_targets == "all" else ATTN
print(f"{len(train_wavs)} train clips ({len(train_wavs)*a.window_s:.0f}s), "
      f"{len(hold_wavs)} held-out clips | window {a.window_s}s hop {a.hop_s or a.window_s}s")
print(f"LoRA r={a.lora_rank} targets={a.lora_targets} lr={a.lr} steps={a.steps} "
      f"accum={a.accum} (effective batch {a.accum})")

PAD = DECODER_START  # 2048; appears in the codebook delay pattern, must not be scored


def losses(model, proc, wavs, shuffle_seed=None, per_codebook=False):
    """Held-out CE. per_codebook=True returns (n_clips, n_q) instead of (n_clips,).

    WHY PER-CODEBOOK MATTERS. MusicGen uses residual VQ: codebook 1 is coarse and
    predictable, codebooks 2-4 encode residuals that are near-random BY CONSTRUCTION.
    Base held-out loss is 7.13 against 7.62 for uniform-over-2048, i.e. the average is
    dominated by codebooks that are almost unpredictable. Structure learning in codebook 1
    can be completely buried by that noise, so the averaged loss is the wrong readout.
    """
    import torch.nn.functional as F

    model.eval()
    inp = proc(text=[PROMPT], padding=True, return_tensors="pt").to(dev)
    out = []
    with torch.no_grad():
        for j, w in enumerate(wavs):
            lab = tokens_of(model, np.asarray(w, dtype=np.float32), dev).transpose(1, 2).contiguous()
            if shuffle_seed is not None:
                g = torch.Generator(device="cpu").manual_seed(shuffle_seed + j)
                lab = lab[:, torch.randperm(lab.shape[1], generator=g).to(lab.device), :]
            o = model(**inp, labels=lab)
            if not per_codebook:
                out.append(float(o.loss))
                continue
            B, T, nq = lab.shape
            L = lab.permute(0, 2, 1).reshape(B * nq, T)      # (nq, T)
            lg = o.logits                                     # (nq, T, V)
            n = min(lg.shape[1], L.shape[1])
            per = []
            for q in range(nq):
                tgt = L[q, :n]
                keep = tgt != PAD                             # skip delay-pattern padding
                if keep.sum() == 0:
                    per.append(float("nan")); continue
                per.append(float(F.cross_entropy(lg[q, :n][keep], tgt[keep])))
            out.append(per)
    return np.array(out)

base = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, attn_implementation="eager").to(dev)
base.config.decoder.decoder_start_token_id = DECODER_START
base.config.decoder.pad_token_id = DECODER_START
bproc = AutoProcessor.from_pretrained(MODEL_ID)
b_real, b_shuf = losses(base, bproc, hold_wavs), losses(base, bproc, hold_wavs, shuffle_seed=7)
b_real_q = losses(base, bproc, hold_wavs, per_codebook=True)
b_shuf_q = losses(base, bproc, hold_wavs, shuffle_seed=7, per_codebook=True)
del base
if dev.type == "mps": torch.mps.empty_cache()

cbw = [float(x) for x in a.codebook_weights.split(",")] if a.codebook_weights else None
prompts = None
if a.captions == "titles":
    import json as _json
    mf = Path(a.artist_dir) / "manifest.json"
    titles = {}
    if mf.exists():
        for e in _json.loads(mf.read_text()):
            titles[Path(e["file"]).stem] = e.get("title") or "instrumental music"
    prompts = []
    for tk in train:
        nw = len(load_windows(str(tk), **kw))
        ttl = titles.get(tk.stem, "instrumental music")
        prompts += [f"instrumental music, {ttl}"] * nw
model, proc = _fine_tune(dev, train_wavs, a.steps, a.lr, a.seed,
                         lora_rank=a.lora_rank, lora_targets=targets,
                         accum=a.accum, cosine=not a.no_cosine,
                         weight_decay=a.weight_decay,
                         codebook_weights=cbw, prompts=prompts)
t_real, t_shuf = losses(model, proc, hold_wavs), losses(model, proc, hold_wavs, shuffle_seed=7)
t_real_q = losses(model, proc, hold_wavs, per_codebook=True)
t_shuf_q = losses(model, proc, hold_wavs, shuffle_seed=7, per_codebook=True)

dr, ds = t_real - b_real, t_shuf - b_shuf
sr_ = dr.std(ddof=1)/np.sqrt(len(dr))
ratio = ds.mean()/dr.mean() if dr.mean() != 0 else float("nan")
print(f"\n  real     {dr.mean():+.4f}  sem {sr_:.4f}  improving {(dr<0).mean():.0%}")
print(f"  shuffled {ds.mean():+.4f}  sem {ds.std(ddof=1)/np.sqrt(len(ds)):.4f}")
print(f"  RATIO shuffled/real = {ratio:.2f}")
if dr.mean() > -0.05:
    v = "NO LEARNING"
elif ratio >= 0.7:
    v = "MARGINALS ONLY (timbre prior, not style)"
elif ratio < 0.3:
    v = "STRUCTURE -- usable style baseline"
else:
    v = "MIXED"
print(f"  => {v}")
print(f"\nPER-CODEBOOK (the averaged number above is dominated by the near-random ones):")
print(f"  {'cb':>3} {'base loss':>10} {'real delta':>11} {'shuf delta':>11} {'ratio':>7}  verdict")
for q in range(b_real_q.shape[1]):
    drq = t_real_q[:, q] - b_real_q[:, q]
    dsq = t_shuf_q[:, q] - b_shuf_q[:, q]
    rq = dsq.mean()/drq.mean() if abs(drq.mean()) > 1e-9 else float("nan")
    vq = ("structure" if (drq.mean() < -0.05 and rq < 0.3)
          else "marginals" if drq.mean() < -0.05 else "no learning")
    print(f"  {q+1:>3} {b_real_q[:, q].mean():>10.3f} {drq.mean():>+11.4f} "
          f"{dsq.mean():>+11.4f} {rq:>7.2f}  {vq}")
print(f"\n  (uniform-over-2048 baseline = {np.log(2048):.3f} nats)")

# THE REAL GATE: codebook 1. Averaged loss is dominated by codebooks 2-4, whose residuals
# are near-random by construction -- codebook 4's base loss (7.889) is WORSE than uniform
# (7.625), so "improvement" there is just drift toward the marginal. Only codebook 1 carries
# coarse structure and perceptual content, so only codebook 1 tells you about style.
d1 = t_real_q[:, 0] - b_real_q[:, 0]
s1 = t_shuf_q[:, 0] - b_shuf_q[:, 0]
r1 = s1.mean()/d1.mean() if abs(d1.mean()) > 1e-9 else float("nan")
sem1 = d1.std(ddof=1)/np.sqrt(len(d1))
print(f"\nCODEBOOK-1 GATE (the one that matters):")
print(f"  real delta {d1.mean():+.4f}  sem {sem1:.4f}  improving {(d1<0).mean():.0%}")
print(f"  shuffled   {s1.mean():+.4f}   ratio {r1:.2f}")
if d1.mean() > 0.05:
    print("  => FAIL: fine-tuning made coarse structure WORSE. Actively harmful.")
elif d1.mean() > -0.05:
    print("  => FAIL: no effect on coarse structure.")
elif r1 < 0.3:
    print("  => PASS: coarse structure improved and the gain needs real time order.")
else:
    print("  => FAIL: coarse gain survives shuffling, so it is still marginals.")

if a.save:
    json.dump({"args": vars(a), "real": dr.tolist(), "shuf": ds.tolist(),
               "ratio": float(ratio), "verdict": v}, open(a.save, "w"), indent=2)
