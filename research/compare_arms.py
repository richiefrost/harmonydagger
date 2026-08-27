#!/usr/bin/env python3
"""
Paired comparison of clean vs protected style learning.

Both arms are evaluated on the SAME held-out clips with the SAME base model, so the
correct statistic is the per-clip difference of the two learning deltas:

    clean_delta[i]     = tuned_on_clean[i]     - base[i]     (negative = learned)
    protected_delta[i] = tuned_on_protected[i] - base[i]
    per_clip_effect[i] = protected_delta[i] - clean_delta[i] (positive = protection worked)

    efficacy = 1 - mean(protected_delta) / mean(clean_delta)

Pairing removes between-clip difficulty variance, which dominates the unpaired sds.

Usage: python compare_arms.py clean.json protected.json
"""
import json, sys
import numpy as np

c = json.load(open(sys.argv[1]))
p = json.load(open(sys.argv[2]))
b1, b2 = np.array(c["base_heldout"]), np.array(p["base_heldout"])
if not np.allclose(b1, b2, atol=1e-6):
    print(f"WARNING: base losses differ between runs (max {np.abs(b1-b2).max():.4f}) -- "
          "not a clean paired comparison")
cd = np.array(c["tuned_heldout"]) - b1
pd = np.array(p["tuned_heldout"]) - b2
eff = pd - cd
n = len(eff)
sem = eff.std(ddof=1) / np.sqrt(n)

print(f"held-out clips: n={n}")
print(f"  clean     learning delta: {cd.mean():+.4f}  (sem {cd.std(ddof=1)/np.sqrt(n):.4f})")
print(f"  protected learning delta: {pd.mean():+.4f}  (sem {pd.std(ddof=1)/np.sqrt(n):.4f})")
print(f"\npaired per-clip protection effect (positive = protection worked):")
print("  " + "  ".join(f"{v:+.3f}" for v in eff))
print(f"  mean {eff.mean():+.4f}  sd {eff.std(ddof=1):.4f}  sem {sem:.4f}")
print(f"  clips where protection helped: {int((eff>0).sum())}/{n}")
print(f"  |mean|/sem = {abs(eff.mean())/sem:.1f}   (>2 to claim an effect)")
# SANITY INVARIANT. A protected model cannot legitimately learn the artist BETTER than a
# clean one -- protection can only remove information, never add it. If protected beats
# clean, the measurement is broken (that is how the earlier CLAP numbers showed protected
# +0.7485 vs clean +0.7186, and the -14.7% "efficacy"). Treat it as a failed experiment,
# not a finding.
SANITY_TOL = 0.02  # nats; below this the two arms are indistinguishable
if pd.mean() < cd.mean() - SANITY_TOL:
    print(f"\n*** SANITY VIOLATION: protected learned MORE than clean "
          f"({pd.mean():+.4f} vs {cd.mean():+.4f}). Protection cannot add information. "
          f"The measurement is broken -- do not report an efficacy number. ***")
elif cd.mean() > -0.05:
    print(f"\n*** NO BASELINE: the clean arm barely learned ({cd.mean():+.4f}). "
          f"There is nothing for protection to block, so efficacy is meaningless. ***")

efficacy = 1 - pd.mean()/cd.mean()
print(f"\nPROTECTION EFFICACY: {efficacy:.1%}")
print("  (0% = protection did nothing, 100% = mimic learned nothing at all)")
if abs(eff.mean())/sem < 2:
    print("\n  NOT SIGNIFICANT at this n -- do not report the efficacy number.")

if c.get("base_cb1") and p.get("base_cb1"):
    b1c, b1p = np.array(c["base_cb1"]), np.array(p["base_cb1"])
    cd1 = np.array(c["tuned_cb1"]) - b1c
    pd1 = np.array(p["tuned_cb1"]) - b1p
    e1 = pd1 - cd1
    s1 = e1.std(ddof=1)/np.sqrt(len(e1))
    print(f"\nCODEBOOK-1 ONLY (carries perceptual content; the average is dominated by")
    print(f"near-random codebooks 2-4, so this is the number that matters):")
    print(f"  clean cb1 delta {cd1.mean():+.4f}   protected cb1 delta {pd1.mean():+.4f}")
    print(f"  effect {e1.mean():+.4f}  sem {s1:.4f}  |mean|/sem {abs(e1.mean())/max(s1,1e-9):.1f}")
    if cd1.mean() > -0.05:
        print("  => clean arm did not improve codebook 1; no coarse baseline to protect.")
    else:
        print(f"  => codebook-1 efficacy {1 - pd1.mean()/cd1.mean():.1%}")
