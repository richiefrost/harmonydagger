#!/usr/bin/env python3
"""
Catalogue-level style-mimicry harness -- doc §6's "untested scale", the setting Mist
protects against for images.

THE QUESTION. Not "can the model regurgitate my track" (that's finetune_eval.py) but
"if a mimic fine-tunes on my CATALOGUE, can they generate new music in my style?"

THREE ARMS, giving an interpretable scale:

    base        pretrained MusicGen, no fine-tune   -> the FLOOR (style you get for free)
    clean       fine-tuned on the real catalogue     -> full mimicry, what we want to block
    protected   fine-tuned on the protected copies   -> lands somewhere between

Style similarity is measured against HELD-OUT artist tracks (never trained on), so this
measures style acquisition rather than memorization.

    protection_efficacy = (clean - protected) / (clean - base)

1.0 means protection pushed the mimic all the way back to the floor; 0.0 means no effect.
Negative means protection HELPED the mimic, which would be worth knowing.

THE GATE (doc §8 rule 4). If the clean arm doesn't clear the base floor by --gate-margin,
there is no mimicry to block and the comparison is meaningless -- the run is DISCARDED.
Doc §6 notes the one previous catalogue attempt was inconclusive for exactly this reason,
so expect this to bite. If it does, the fix is more/longer excerpts or more steps, not a
smaller margin.

Usage:
    python catalogue.py --artist-dir data/catalogue/6th_sense --steps 2400
    python catalogue.py --artist-dir data/catalogue/6th_sense --protected-dir data/catalogue/6th_sense_prot
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from finetune_eval import DECODER_START, MODEL_ID, PROMPT, tokens_of
from protection import SR, load_excerpt

WINDOW_S = 8.0  # MusicGen trains comfortably on 8s at 32kHz (400 EnCodec frames)


def load_windows(path, sr=SR, window_s=WINDOW_S, peak=0.7, max_windows=None, hop_s=None):
    """Slice a track into consecutive non-overlapping windows, peak-normalised.

    WHY: doc §5.1's single 8s excerpt was designed for the single-track MEMORIZATION
    experiment. For catalogue-level STYLE learning it is far too little data -- 6 tracks
    x 8s = 48 seconds total, enough to memorize six snippets but not to learn a style.
    Diagnostic that showed this: clean generations reached 0.93 CLAP similarity to an
    individual training track (memorization), while held-out similarity (0.719) merely
    matched what the real training tracks already score against held-out (0.722) -- i.e.
    no style generalization, just inherited artist self-similarity.

    FMA tracks are 30s, so windowing yields ~3x more training data per track at the same
    per-step cost (window length, hence sequence length, is unchanged).
    """
    import librosa
    import numpy as np

    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(sr * window_s)
    hop = int(sr * (hop_s if hop_s else window_s))  # hop < window => overlapping windows
    # FMA "30 second" previews are actually 29.977 s (959251 samples), so asking for a
    # 30 s window silently yields ZERO windows for most tracks. Tolerate a small shortfall
    # by taking the whole track rather than dropping it -- the earlier version dropped 16
    # of 22 tracks without a word and produced NaN downstream.
    if len(y) < n and len(y) >= 0.97 * n:
        pk = np.abs(y).max()
        return [(y / (pk + 1e-9) * peak).astype("float32")] if pk >= 1e-4 else []
    out = []
    for start in range(0, len(y) - n + 1, hop):
        w = y[start : start + n]
        pk = np.abs(w).max()
        if pk < 1e-4:  # skip near-silent windows
            continue
        out.append((w / (pk + 1e-9) * peak).astype("float32"))
        if max_windows and len(out) >= max_windows:
            break
    return out
from style_metric import style_similarity


def _fine_tune(dev, train_wavs, steps, lr, seed, verbose=True, lora_rank=0,
               lora_targets=None, accum=1, warmup_frac=0.1, cosine=True,
               weight_decay=1e-5, codebook_weights=None, prompts=None):
    """Fine-tune the decoder over a SET of tracks, cycling one per step.

    Cycling rather than batching keeps memory flat (MusicGen decoder is 420M params and
    MPS is already tight) and each clip is seen equally often.

    LORA (lora_rank > 0) IS THE RIGHT DEFAULT FOR STYLE, for three converging reasons:

      1. Capacity. Full 420M-param fine-tuning MEMORIZES. Measured: 6 clips/2400 steps
         gave 0.93 CLAP similarity to an individual training clip, and 18 clips/3600
         steps drove loss to 0.017 and produced generations WORSE than the base model
         (+0.5086 vs +0.5154 floor) -- 18 contradictory targets under one identical
         prompt tear the model apart. A rank-16 adapter (~3M params) cannot memorize
         144s of audio, so it must find the shared low-rank direction, which is style.
      2. Threat model. Real mimics fine-tune with LoRA, not full-decoder updates.
      3. Mist-v2 targets LoRA fine-tuning specifically, so this is the setting our
         bi-level perturbation should be attacking.
    """
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    ).to(dev)
    model.config.decoder.decoder_start_token_id = DECODER_START
    model.config.decoder.pad_token_id = DECODER_START

    for p in model.parameters():
        p.requires_grad_(False)
    dec = [
        p for n, p in model.named_parameters()
        if n.startswith("decoder.") and "audio_encoder" not in n
    ]
    if lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        # Base must already be frozen (above) or get_peft_model leaves ~170M trainable
        # instead of ~3M -- not a real LoRA, and it memorizes just like a full fine-tune.
        model.decoder = get_peft_model(
            model.decoder,
            LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                       target_modules=lora_targets or ["q_proj", "v_proj"],
                       lora_dropout=0.05),
        )
        dec = [p for p in model.parameters() if p.requires_grad]
    else:
        for p in dec:
            p.requires_grad_(True)
    if verbose:
        n_tr = sum(p.numel() for p in dec)
        print(f"      trainable params: {n_tr/1e6:.2f}M"
              + (f" (LoRA rank {lora_rank})" if lora_rank > 0 else " (full decoder)"))

    labels = [tokens_of(model, w, dev).transpose(1, 2).contiguous() for w in train_wavs]

    # EXPERIMENT B: per-clip captions. MusicGen is text-conditioned; training every clip on
    # one identical prompt gives the model no reason to learn anything beyond a marginal
    # shift for that prompt. Standard practice captions each clip.
    if prompts is not None:
        if len(prompts) != len(train_wavs):
            raise ValueError(f"{len(prompts)} prompts for {len(train_wavs)} clips")
        input_list = [proc(text=[t], padding=True, return_tensors="pt").to(dev) for t in prompts]
        inputs = input_list[0]
        if verbose:
            print(f"      per-clip captions: {len(set(prompts))} distinct")
    else:
        input_list = None
        inputs = proc(text=[PROMPT], padding=True, return_tensors="pt").to(dev)
    # weight_decay 1e-5 matches the community MusicGen recipe; AdamW's 0.01 default is
    # aggressive for a short fine-tune of a 420M-param pretrained decoder.
    opt = torch.optim.AdamW(dec, lr=lr, weight_decay=weight_decay)

    # GRADIENT ACCUMULATION + LR SCHEDULE, matching reference practice.
    # audiocraft trains MusicGen at batch_size 128-192 with a cosine schedule and 4000
    # warmup steps. Our earlier runs used batch size 1 and a constant lr, and that is the
    # likeliest reason they learned only the artist's marginal token distribution: with one
    # clip per update, gradient noise dominates and the only component that accumulates
    # coherently across steps is the consistent one (marginals), while per-clip structure
    # signal cancels. `accum` clips are averaged per optimizer update to fix that.
    import math

    sched = None
    if cosine and steps > 0:
        warm = max(1, int(steps * warmup_frac))

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            prog = (step - warm) / max(1, steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    idx = 0
    with torch.enable_grad():
        for i in range(steps):
            opt.zero_grad()
            running = 0.0
            for _ in range(accum):
                j = idx % len(labels)
                inp = input_list[j] if input_list is not None else inputs
                lab = labels[j]
                out = model(**inp, labels=lab)
                loss = _weighted_loss(out, lab, codebook_weights) if codebook_weights else out.loss
                (loss / accum).backward()
                running += float(loss) / accum
                idx += 1
                del out, loss
            torch.nn.utils.clip_grad_norm_(dec, 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            if verbose and (i + 1) % max(1, steps // 6) == 0:
                print(f"      step {i+1}/{steps} (x{accum})  loss {running:.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
    return model, proc


def _weighted_loss(out, labels, weights):
    """EXPERIMENT A: reweight cross-entropy across the 4 RVQ codebooks.

    The built-in loss averages CE uniformly over codebooks. But codebook 4 pretrained
    held-out loss (7.82) is WORSE than uniform-over-2048 (7.625), so the near-random fine
    codebooks present enormous apparent headroom while codebook 1 -- the only one carrying
    coarse structure and perceptual content -- has little. Uniform averaging therefore lets
    optimization chase the noise codebooks and sacrifice codebook 1, which is what every
    measured config does (cb1 delta +0.25 to +0.40).

    weights: per-codebook multipliers, e.g. [4,1,1,1] to emphasise codebook 1.
    """
    import torch.nn.functional as F

    B, T, nq = labels.shape
    L = labels.permute(0, 2, 1).reshape(B * nq, T)
    lg = out.logits
    n = min(lg.shape[1], L.shape[1])
    total, wsum = 0.0, 0.0
    for q in range(nq):
        w = float(weights[q]) if q < len(weights) else 1.0
        if w == 0.0:
            continue
        tgt = L[q, :n]
        keep = tgt != DECODER_START          # skip codebook delay-pattern padding
        if keep.sum() == 0:
            continue
        total = total + w * F.cross_entropy(lg[q, :n][keep], tgt[keep])
        wsum += w
    return total / max(wsum, 1e-9)


def _generate(model, proc, dev, n, seconds, seed):
    """Unprimed generation from the text prompt -- no leakage of any real track."""
    model.eval()
    frames = int(seconds * 50)  # EnCodec: 50 frames/sec
    inputs = proc(text=[PROMPT], padding=True, return_tensors="pt").to(dev)
    outs = []
    for k in range(n):
        torch.manual_seed(seed + 1000 * k)  # vary per sample
        with torch.no_grad():
            a = model.generate(
                **inputs, max_new_tokens=frames, do_sample=True, guidance_scale=3.0
            )
        outs.append(a[0, 0].cpu().float().numpy())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist-dir", required=True, help="clean catalogue (build_corpus --out-dir)")
    ap.add_argument("--protected-dir", help="protected copies of the same tracks")
    ap.add_argument("--contrast-dir", help="a DIFFERENT artist, as a sanity reference")
    ap.add_argument("--holdout", type=int, default=2, help="tracks reserved for measuring style")
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-gen", type=int, default=4)
    ap.add_argument("--gen-s", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-margin", type=float, default=0.05,
                    help="clean must exceed base by this much or the run is discarded")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="0 = full-decoder fine-tune (MEMORIZES; see _fine_tune docstring). "
                         "16 is a sensible style adapter and matches what real mimics use.")
    ap.add_argument("--windows-per-track", type=int, default=1,
                    help="slice each track into N consecutive 8s windows (1 = doc's single "
                         "excerpt; 3 uses the full 30s FMA track for ~3x more style data)")
    ap.add_argument("--out", default="catalogue_out")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    outdir = Path(args.out) / Path(args.artist_dir).name
    outdir.mkdir(parents=True, exist_ok=True)

    tracks = sorted(Path(args.artist_dir).glob("*.wav"))
    if len(tracks) < args.holdout + 2:
        raise SystemExit(f"need at least {args.holdout+2} tracks, found {len(tracks)}")
    hold, train = tracks[: args.holdout], tracks[args.holdout :]
    print(f"artist: {Path(args.artist_dir).name}")
    print(f"  train on {len(train)} tracks, measure style against {len(hold)} held-out")

    if args.windows_per_track > 1:
        train_wavs = [
            w for t in train
            for w in load_windows(str(t), max_windows=args.windows_per_track)
        ]
        print(f"  windowing: {len(train_wavs)} training clips "
              f"({len(train_wavs)*WINDOW_S:.0f}s of audio)")
    else:
        train_wavs = [load_excerpt(str(t), sr=SR).astype(np.float32) for t in train]
    hold_wavs = [load_excerpt(str(t), sr=SR) for t in hold]

    results = {}

    # ---- base floor: how much of this style does an un-fine-tuned model already have? ----
    print("\n[base] generating from pretrained model (no fine-tune)...")
    base = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    ).to(dev)
    base.config.decoder.decoder_start_token_id = DECODER_START
    base.config.decoder.pad_token_id = DECODER_START
    bproc = AutoProcessor.from_pretrained(MODEL_ID)
    gens = _generate(base, bproc, dev, args.n_gen, args.gen_s, args.seed)
    results["base"] = style_similarity(gens, hold_wavs, SR, dev)
    for i, g in enumerate(gens):
        sf.write(outdir / f"gen_base_{i}.wav", g, SR)
    print(f"      style similarity {results['base']['mean']:+.4f}")
    del base
    if dev.type == "mps":
        torch.mps.empty_cache()

    # ---- clean arm ----
    print(f"\n[clean] fine-tuning on {len(train)} real tracks ({args.steps} steps)...")
    model, proc = _fine_tune(dev, train_wavs, args.steps, args.lr, args.seed,
                             lora_rank=args.lora_rank)
    gens = _generate(model, proc, dev, args.n_gen, args.gen_s, args.seed)
    results["clean"] = style_similarity(gens, hold_wavs, SR, dev)
    for i, g in enumerate(gens):
        sf.write(outdir / f"gen_clean_{i}.wav", g, SR)
    print(f"      style similarity {results['clean']['mean']:+.4f}")
    del model
    if dev.type == "mps":
        torch.mps.empty_cache()

    # ---- protected arm ----
    if args.protected_dir:
        pt = sorted(Path(args.protected_dir).glob("*.wav"))
        # Protected windows are named "<track stem>__w<N>.wav" (windowize.py), so match on
        # the stem prefix. Matching on full filename silently found nothing.
        keep_stems = {t.stem for t in train}
        keep = {t.name for t in train}
        # Protected files are ALREADY 8s excerpts (written by protect_catalogue.py), so
        # read them directly -- load_excerpt would try to cut an 8s window at a 1s offset
        # and fail for want of 9s of source.
        pwavs = []
        for t in pt:
            if t.name not in keep and t.stem.rsplit("__w", 1)[0] not in keep_stems:
                continue
            y, fsr = sf.read(str(t))
            if fsr != SR:
                raise SystemExit(f"{t}: sample rate {fsr} != {SR}")
            pwavs.append(np.asarray(y, dtype=np.float32))
        if not pwavs:
            raise SystemExit(f"no protected clips in {args.protected_dir} match the training tracks")
        if len(pwavs) != len(train_wavs):
            print(f"      note: {len(pwavs)} protected clips vs {len(train_wavs)} clean "
                  f"training clips (windowing counts may differ)")
        print(f"\n[protected] fine-tuning on {len(pwavs)} protected tracks ({args.steps} steps)...")
        model, proc = _fine_tune(dev, pwavs, args.steps, args.lr, args.seed,
                                 lora_rank=args.lora_rank)
        gens = _generate(model, proc, dev, args.n_gen, args.gen_s, args.seed)
        results["protected"] = style_similarity(gens, hold_wavs, SR, dev)
        for i, g in enumerate(gens):
            sf.write(outdir / f"gen_protected_{i}.wav", g, SR)
        print(f"      style similarity {results['protected']['mean']:+.4f}")
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()

    # ---- contrast reference: a different artist's real tracks ----
    if args.contrast_dir:
        cw = [load_excerpt(str(t), sr=SR) for t in sorted(Path(args.contrast_dir).glob("*.wav"))[:4]]
        if cw:
            results["contrast_other_artist"] = style_similarity(cw, hold_wavs, SR, dev)
            print(f"\n[contrast] a different artist's real tracks vs held-out: "
                  f"{results['contrast_other_artist']['mean']:+.4f}")

    # ---- verdict ----
    b, c = results["base"]["mean"], results["clean"]["mean"]
    gated = (c - b) < args.gate_margin
    print("\n" + "=" * 66)
    print(f"{'arm':<26} {'style similarity':>18}")
    for k in ("base", "clean", "protected", "contrast_other_artist"):
        if k in results:
            print(f"{k:<26} {results[k]['mean']:>+18.4f}")
    print("=" * 66)

    if gated:
        print(f"\nDISCARDED: clean ({c:+.4f}) did not clear base ({b:+.4f}) by {args.gate_margin}.")
        print("The fine-tune did not acquire the artist's style, so there is no mimicry to")
        print("block and the protected arm means nothing. Try more steps, longer excerpts,")
        print("or more tracks -- do NOT lower the gate. (doc §6, §8 rule 4)")
    elif "protected" in results:
        p = results["protected"]["mean"]
        eff = (c - p) / (c - b)
        print(f"\nstyle acquired by clean fine-tune : {c-b:+.4f} above floor")
        print(f"protection efficacy               : {eff:+.1%}")
        print("  (1.0 = pushed the mimic back to the floor, 0.0 = no effect,")
        print("   negative = protection HELPED the mimic)")
    else:
        print(f"\nclean cleared the gate ({c-b:+.4f} above floor). Now run with")
        print("--protected-dir to measure whether protection reduces it.")

    (outdir / "results.json").write_text(
        json.dumps({"args": vars(args), "results": results, "gated_out": gated}, indent=2)
    )
    print(f"\nwrote {outdir}/ (generations + results.json)")


if __name__ == "__main__":
    main()
