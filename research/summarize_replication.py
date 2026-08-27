#!/usr/bin/env python3
"""Aggregate data/replication.jsonl into per-objective gap statistics."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path("data/replication.jsonl")
CODEBOOKS = [f"codebook_{i}" for i in range(1, 5)]


def main():
    if not SRC.exists():
        raise SystemExit("no data/replication.jsonl -- run ./replicate_bilevel.sh first")

    recs = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    # index clean arms by (track, seed) so gaps are computed against the matching control
    clean = {
        (r["track"], r["seed"]): r for r in recs if r["objective"] == "clean"
    }

    gaps = defaultdict(list)
    auds = defaultdict(list)
    cbgaps = defaultdict(lambda: defaultdict(list))
    discarded = 0

    for r in recs:
        if r["objective"] == "clean":
            continue
        c = clean.get((r["track"], r["seed"]))
        if c is None:
            print(f"  (skipping {r['track']} {r['objective']}: no clean arm for seed {r['seed']})")
            continue
        if c.get("gated_out"):
            discarded += 1
            continue
        gaps[r["objective"]].append(100 * (c["accuracy"] - r["accuracy"]))
        auds[r["objective"]].append(r["audibility"])
        for k in CODEBOOKS:
            if k in c["per_codebook"] and k in r["per_codebook"]:
                cbgaps[r["objective"]][k].append(
                    100 * (c["per_codebook"][k] - r["per_codebook"][k])
                )

    if discarded:
        print(f"discarded {discarded} arm(s) whose clean baseline failed the 0.90 gate\n")

    print(f"{'objective':<10} {'n':>3} {'mean gap':>9} {'sd':>6} {'range':>14} {'mean audib':>11}")
    print("-" * 60)
    for obj in ("baseline", "bilevel"):
        g = gaps.get(obj, [])
        if not g:
            continue
        print(
            f"{obj:<10} {len(g):>3} {np.mean(g):>8.1f}p {np.std(g):>6.1f} "
            f"[{min(g):>5.1f},{max(g):>5.1f}] {np.mean(auds[obj]):>11.4f}"
        )

    print(f"\nper-codebook mean gap (pts)")
    print(f"{'objective':<10} " + " ".join(f"{k[-1]:>7}" for k in CODEBOOKS))
    for obj in ("baseline", "bilevel"):
        if obj in cbgaps:
            print(
                f"{obj:<10} "
                + " ".join(f"{np.mean(cbgaps[obj][k]):>7.1f}" for k in CODEBOOKS)
            )

    b, l = gaps.get("baseline", []), gaps.get("bilevel", [])
    if b and l:
        print(f"\nbi-level / baseline gap ratio: {np.mean(l)/max(np.mean(b),1e-9):.2f}x")
        print("doc's n=60 reference: mean 26.7 pts, 95% CI [24.9, 28.4], range 16.1-43.4")
        if len(l) < 10:
            print(f"\nNOTE: n={len(l)} is far below the doc's n=60. Directional only.")


if __name__ == "__main__":
    main()
