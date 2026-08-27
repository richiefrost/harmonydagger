#!/usr/bin/env python3
"""
Replicate: baseline (encoder-only) vs bi-level protection. ONE measurement per process.

WHY ONE PROCESS PER MEASUREMENT: MusicGen fine-tuning on MPS accumulates GPU state across
runs. Several in one process eventually fails with "command buffer exited with error
status", which surfaces as a BOGUS tensor-shape mismatch deep inside EnCodec's resnet
shortcut (e.g. "size of tensor a (255994) must match tensor b (255992)"). The shapes are
fine; the allocator isn't. A fresh process per measurement avoids it entirely.

Usage:
    ./replicate_bilevel.sh                      # drives the whole sweep
    python replicate_bilevel.py --track T --objective clean|baseline|bilevel
    python summarize_replication.py             # aggregate
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from finetune_eval import CLEAN_BASELINE_GATE, train_and_eval
from protection import SR, audibility, load_excerpt, protect

RESULTS = Path("data/replication.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--objective", required=True, choices=["clean", "baseline", "bilevel"])
    ap.add_argument("--note-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    clean = load_excerpt(args.track, sr=SR)
    name = Path(args.track).name

    if args.objective == "clean":
        r = train_and_eval(
            clean.astype(np.float32), clean.astype(np.float32), args.steps, dev, args.seed
        )
        gated = r["overall_accuracy"] < CLEAN_BASELINE_GATE
        rec = {
            "track": name, "objective": "clean", "seed": args.seed,
            "accuracy": r["overall_accuracy"], "per_codebook": r["per_codebook"],
            "gated_out": gated,
        }
        print(
            f"{name[:34]:34s} clean     acc={r['overall_accuracy']:.4f}"
            + ("  <-- BELOW GATE, track unusable" if gated else "")
        )
    else:
        if args.objective == "baseline":
            from transformers import MusicgenForConditionalGeneration

            probe = (
                MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
                .to(dev)
                .eval()
            )
            for p in probe.parameters():
                p.requires_grad_(False)
            prot = protect(
                probe.audio_encoder, clean, SR, dev, steps=80, note_frac=args.note_frac
            )
            del probe
        else:
            from protection_bilevel import protect_bilevel

            prot = protect_bilevel(
                clean, SR, dev, note_frac=args.note_frac, rounds=5,
                attacker_steps=100, defender_steps=20, seed=args.seed, verbose=False,
            )
        if dev.type == "mps":
            torch.mps.empty_cache()

        aud = audibility(clean, prot - clean, SR)
        r = train_and_eval(prot, clean.astype(np.float32), args.steps, dev, args.seed)
        rec = {
            "track": name, "objective": args.objective, "seed": args.seed,
            # note_frac MUST be recorded: without it, runs at different protection
            # strengths are indistinguishable in the JSONL and get pooled into one
            # statistic. That silently corrupted the n=24 baseline figure once.
            "note_frac": args.note_frac,
            "audibility": aud, "accuracy": r["overall_accuracy"],
            "per_codebook": r["per_codebook"],
            "delta_rms": float(np.sqrt(((prot - clean) ** 2).mean())),
        }
        print(
            f"{name[:34]:34s} {args.objective:9s} acc={r['overall_accuracy']:.4f} aud={aud:.4f}"
        )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
