#!/usr/bin/env python3
"""
Did the fine-tune LEARN THIS ARTIST'S DISTRIBUTION? Measured by held-out token loss.

WHY THIS AND NOT generate->CLAP. The sweep's gate goes through generation and then a CLAP
embedding: two noisy, indirect steps on only a handful of samples. Held-out cross-entropy is
the textbook generalization test and it is far more sensitive -- it asks directly whether
the model assigns higher likelihood to UNSEEN tracks by the same artist after fine-tuning.

    base held-out loss  >  tuned held-out loss   -> genuine generalization to the artist
    base ~= tuned                                -> learned nothing transferable
    tuned > base                                 -> fine-tuning DAMAGED the model

Also reports loss on the TRAINING clips, so memorization is visible in the same table:
train loss -> ~0 while held-out loss flat or rising is the memorization signature.

Usage:
    python heldout_loss.py --artist-dir data/catalogue/6th_sense_big --holdout 4 \
        --lora-rank 32 --lr 3e-4 --steps 3600 --windows-per-track 3
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from catalogue import _fine_tune, load_windows
from finetune_eval import DECODER_START, MODEL_ID, PROMPT, tokens_of
from protection import SR, load_excerpt


def per_clip_cb1(model, proc, wavs, dev):
    """Codebook-1 held-out CE per clip.

    The average over codebooks is dominated by codebooks 2-4, whose residuals are
    near-random by construction (cb4 base loss 7.889 > uniform-over-2048 7.625). Only
    codebook 1 carries coarse structure and perceptual content, so protection has to be
    judged there -- judging on the average is how the 20.8% figure got measured against a
    marginal-only baseline.
    """
    import torch.nn.functional as F

    model.eval()
    inputs = proc(text=[PROMPT], padding=True, return_tensors="pt").to(dev)
    out = []
    with torch.no_grad():
        for w in wavs:
            lab = tokens_of(model, np.asarray(w, dtype=np.float32), dev)
            lab = lab.transpose(1, 2).contiguous()          # (B, T, n_q)
            logits = model(**inputs, labels=lab).logits     # (B*n_q, T, V)
            tgt = lab[0, :, 0]                              # codebook 1 targets
            lg = logits[0][: tgt.shape[0]]
            keep = tgt != DECODER_START                     # skip delay-pattern pad
            out.append(float(F.cross_entropy(lg[keep], tgt[keep])))
    return np.array(out)


def per_clip_losses(model, proc, wavs, dev):
    """Teacher-forced cross-entropy for EACH clip (so deltas can be paired)."""
    model.eval()
    inputs = proc(text=[PROMPT], padding=True, return_tensors="pt").to(dev)
    out = []
    with torch.no_grad():
        for w in wavs:
            lab = tokens_of(model, np.asarray(w, dtype=np.float32), dev)
            lab = lab.transpose(1, 2).contiguous()
            out.append(float(model(**inputs, labels=lab).loss))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist-dir", required=True)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--windows-per-track", type=int, default=3)
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protected-dir",
                    help="train on PROTECTED copies of the training windows instead. "
                         "Held-out clips stay CLEAN -- the mimic wants the real music.")
    ap.add_argument("--accum", type=int, default=1,
                    help="clips per optimizer update. MUST match the config that passed the "
                         "structure gate -- batch 1 learns marginals only.")
    ap.add_argument("--lora-targets", choices=["attn", "all"], default="attn")
    ap.add_argument("--codebook-weights", help="e.g. 4,1,1,1 to emphasise codebook 1")
    ap.add_argument("--clip-list",
                    help="file of window basenames; ALL arms train on exactly these clips. "
                         "Use it whenever comparing protection objectives -- otherwise a "
                         "protected set missing a few windows trains on less data, which "
                         "looks like protection.")
    ap.add_argument("--save", help="write per-clip losses to this JSON for paired analysis")
    ap.add_argument("--train-dir",
                    help="CONTROL: train on THIS artist while scoring held-out clips from "
                         "--artist-dir. If a different artist improves the held-out loss as "
                         "much as the matching artist does, the effect is generic domain "
                         "adaptation, not style learning, and any protection number built "
                         "on it is meaningless.")
    ap.add_argument("--save-json", help="write per-clip losses here for a PAIRED arm comparison")
    ap.add_argument("--windows-dir",
                    help="pre-materialized windows (windowize.py). Required with "
                         "--protected-dir so clean and protected clips correspond 1:1.")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tracks = sorted(Path(args.artist_dir).glob("*.wav"))
    hold, train = tracks[: args.holdout], tracks[args.holdout :]
    if args.train_dir:
        # Cross-artist control: training pool comes from a DIFFERENT artist entirely.
        tr_tracks = sorted(Path(args.train_dir).glob("*.wav"))
        train = tr_tracks
        print(f"  CONTROL: training on {len(train)} tracks from {args.train_dir}")
    if args.clip_list:
        import soundfile as _sf
        names = [x.strip() for x in open(args.clip_list) if x.strip()]
        src = Path(args.protected_dir) if args.protected_dir else Path("data/catalogue/6th_sense_win")
        missing = [n for n in names if not (src / n).exists()]
        if missing:
            raise SystemExit(f"{len(missing)} clips from {args.clip_list} missing in {src}: {missing[:3]}")
        train_wavs = [_sf.read(str(src / n))[0].astype(np.float32) for n in names]
        print(f"  training on {len(train_wavs)} clips from {src} (fixed clip list)")
    elif args.protected_dir:
        import soundfile as _sf
        # Match protected windows to TRAINING tracks only, by filename stem prefix, so no
        # held-out audio leaks into training.
        stems = {t.stem for t in train}
        pf = sorted(
            f for f in Path(args.protected_dir).glob("*.wav")
            if f.stem.rsplit("__w", 1)[0] in stems
        )
        if not pf:
            raise SystemExit(f"no protected windows in {args.protected_dir} matching training tracks")
        train_wavs = [_sf.read(str(f))[0].astype(np.float32) for f in pf]
        print(f"  training on PROTECTED windows from {args.protected_dir}")
    else:
        train_wavs = [
            w for t in train for w in load_windows(str(t), max_windows=args.windows_per_track)
        ]
    # Window the HELD-OUT set too. With only 4 held-out tracks the paired per-track sem is
    # ~0.28, far too coarse to resolve a ~0.3 effect; windowing multiplies the paired sample
    # count at no extra training cost.
    hold_wavs = [
        w for t in hold for w in load_windows(str(t), max_windows=args.windows_per_track)
    ]
    print(f"{len(train_wavs)} training clips, {len(hold_wavs)} held-out clips "
          f"(from {len(hold)} held-out tracks)")

    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    # ---- base ----
    base = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    ).to(dev)
    base.config.decoder.decoder_start_token_id = DECODER_START
    base.config.decoder.pad_token_id = DECODER_START
    bproc = AutoProcessor.from_pretrained(MODEL_ID)
    b_tr = per_clip_losses(base, bproc, train_wavs[:8], dev)
    b_ho = per_clip_losses(base, bproc, hold_wavs, dev)
    b_cb1 = per_clip_cb1(base, bproc, hold_wavs, dev)
    del base
    if dev.type == "mps":
        torch.mps.empty_cache()

    # ---- fine-tuned ----
    ATTN = ["q_proj", "v_proj"]
    ALL = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    cw = [float(x) for x in args.codebook_weights.split(",")] if args.codebook_weights else None
    model, proc = _fine_tune(
        dev, train_wavs, args.steps, args.lr, args.seed, lora_rank=args.lora_rank,
        lora_targets=ALL if args.lora_targets == "all" else ATTN,
        accum=args.accum, codebook_weights=cw,
    )
    t_tr = per_clip_losses(model, proc, train_wavs[:8], dev)
    t_ho = per_clip_losses(model, proc, hold_wavs, dev)
    t_cb1 = per_clip_cb1(model, proc, hold_wavs, dev)

    print(f"\n{'':<12} {'train loss':>18} {'HELD-OUT loss':>18}")
    print("-" * 52)
    print(f"{'base':<12} {b_tr.mean():>10.4f} ±{b_tr.std():<6.3f} {b_ho.mean():>10.4f} ±{b_ho.std():<6.3f}")
    print(f"{'fine-tuned':<12} {t_tr.mean():>10.4f} ±{t_tr.std():<6.3f} {t_ho.mean():>10.4f} ±{t_ho.std():<6.3f}")

    # PAIRED per-track deltas: the same held-out track before and after, so between-track
    # variance cancels. This is the statistically correct comparison -- the unpaired sds
    # above are dominated by tracks simply differing in difficulty.
    pair = t_ho - b_ho
    print(f"\npaired per-held-out-clip delta (n={len(pair)}):")
    print("  " + "  ".join(f"{v:+.3f}" for v in pair[:12]) + ("  ..." if len(pair) > 12 else ""))
    sem = pair.std(ddof=1) / np.sqrt(len(pair)) if len(pair) > 1 else float("nan")
    print(f"  mean {pair.mean():+.4f}  sd {pair.std(ddof=1):.4f}  sem {sem:.4f}  n={len(pair)}")
    frac_neg = float((pair < 0).mean())
    print(f"  fraction improving: {frac_neg:.0%}   "
          f"|mean|/sem = {abs(pair.mean())/sem:.1f} (want >2 to claim an effect)")

    d_ho = float(pair.mean())
    d_tr = float((t_tr - b_tr).mean())
    print()
    if d_ho < -0.10:
        print("=> GENERALIZED to the artist: held-out loss dropped. This is a usable")
        print("   style-mimicry baseline to protect against.")
    elif d_tr < -1.0 and d_ho > -0.05:
        print("=> MEMORIZED: train loss collapsed while held-out did not improve.")
        print("   Not a style baseline. Reduce capacity or add data.")
    elif d_ho > 0.10:
        print("=> DAMAGED the model: held-out loss got WORSE. This config is harmful,")
        print("   not merely ineffective -- lower the lr or the rank.")
    else:
        print("=> NO EFFECT: neither memorized nor generalized. Increase capacity, lr,")
        print("   or steps.")

    if args.save:
        _save(args.save, args, b_ho, t_ho, b_tr, t_tr)


def _save(path, args, b_ho, t_ho, b_tr, t_tr):
    import json
    json.dump({
        "args": vars(args),
        "base_heldout": b_ho.tolist(), "tuned_heldout": t_ho.tolist(),
        "base_train": b_tr.tolist(), "tuned_train": t_tr.tolist(),
    }, open(path, "w"), indent=2)
    print(f"\nsaved per-clip losses to {path}")


    if args.save_json:
        import json
        Path(args.save_json).write_text(json.dumps({
            "arm": "protected" if args.protected_dir else "clean",
            "args": vars(args),
            "base_heldout": b_ho.tolist(), "tuned_heldout": t_ho.tolist(),
            "base_train": b_tr.tolist(), "tuned_train": t_tr.tolist(),
        }, indent=2))
        print(f"\nwrote {args.save_json}")


if __name__ == "__main__":
    main()
