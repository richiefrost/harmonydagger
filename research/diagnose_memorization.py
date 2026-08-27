#!/usr/bin/env python3
"""
Is the catalogue model LEARNING A STYLE or just MEMORIZING training clips?

The distinction matters because only style generalization is what a mimic wants, and only
style generalization is what protection needs to block.

The tell: compare generations against the artist's HELD-OUT tracks, and compare that to
what the artist's own TRAINING tracks already score against those held-out tracks.

  gen-vs-heldout  ~=  realtrain-vs-heldout   -> NO generalization. Generations are
                                               memorized clips inheriting the artist's
                                               natural self-similarity for free.
  gen-vs-heldout  >   realtrain-vs-heldout   -> genuine style learning.

Also reports max similarity to any single training clip: high (>0.9) means the generation
is close to a copy of one clip, i.e. memorization.

Usage: python diagnose_memorization.py --artist-dir D --gen-dir catalogue_out/<name>
"""
import argparse, glob, itertools
from pathlib import Path
import numpy as np, soundfile as sf, torch
from protection import SR, load_excerpt
from style_metric import embed

ap = argparse.ArgumentParser()
ap.add_argument("--artist-dir", required=True)
ap.add_argument("--gen-dir", required=True)
ap.add_argument("--holdout", type=int, default=2)
a = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tr = sorted(Path(a.artist_dir).glob("*.wav"))
hold, train = tr[: a.holdout], tr[a.holdout :]
E_train = [embed(load_excerpt(str(t), sr=SR), SR, dev) for t in train]
E_hold = [embed(load_excerpt(str(t), sr=SR), SR, dev) for t in hold]

ref_hold = np.mean([[float(np.dot(x, y)) for y in E_hold] for x in E_train])
ref_self = np.mean([float(np.dot(x, y)) for x, y in itertools.combinations(E_train, 2)])
print(f"REFERENCES  real training vs held-out = {ref_hold:.4f}   (the bar to beat)")
print(f"            real training vs each other = {ref_self:.4f}\n")
print(f"{'arm':<12} {'vs heldout':>11} {'vs bar':>8} {'max vs a train clip':>21} {'verdict':<24}")
print("-" * 82)
for arm in ("base", "clean", "protected"):
    fs = sorted(glob.glob(f"{a.gen_dir}/gen_{arm}_*.wav"))
    if not fs:
        continue
    G = [embed(sf.read(f)[0], SR, dev) for f in fs]
    mh = np.mean([[float(np.dot(g, e)) for e in E_hold] for g in G])
    mx = np.mean([max(float(np.dot(g, e)) for e in E_train) for g in G])
    delta = mh - ref_hold
    if arm == "base":
        verdict = "floor"
    elif mx > 0.90:
        verdict = "MEMORIZING clips"
    elif delta > 0.02:
        verdict = "generalizing (style)"
    else:
        verdict = "no generalization"
    print(f"{arm:<12} {mh:>11.4f} {delta:>+8.4f} {mx:>21.4f} {verdict:<24}")
print("\nA useful protection experiment needs the clean arm to show 'generalizing (style)'.")
print("If it says MEMORIZING, add data (--windows-per-track, more tracks) before")
print("interpreting any protected-arm number.")
