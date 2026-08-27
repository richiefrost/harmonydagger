#!/usr/bin/env python3
"""
Fine-tune MusicGen on one track (clean or protected) and measure reproduction of the
CLEAN track. Per the research doc §3.4, §5.4.

The framing that makes this tractable (doc §1): this is SELF-DEFENSE, not dataset
poisoning. The question is "if someone trains on my track, can they reproduce my track?"
-- so the attacker trains on the protected copy but is evaluated on the real music.

Six earlier experiments returned null by measuring the wrong thing (whether a
protected-trained model got worse at music generally). Don't regress to that.
"""
import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID, DECODER_START, PROMPT = "facebook/musicgen-small", 2048, "instrumental music"

# Discard any track whose clean arm fails this. If the clean model never memorized the
# track, there is no memorization for protection to block and the comparison is
# meaningless. This single rule turned an impossible early data point (clean 50.6% <
# protected 81.8%) into a clean n=60 result. (doc §5.5, §8 rule 4)
CLEAN_BASELINE_GATE = 0.90

# Under-training produces confidently wrong results in BOTH directions -- the same
# question gave three different answers at 300/1200/3200 steps. (doc §4.1, §8 rule 3)
DEFAULT_STEPS = 800


def tokens_of(model, wav, dev):
    with torch.no_grad():
        return model.audio_encoder.encode(
            torch.tensor(np.asarray(wav, dtype=np.float32), device=dev).view(1, 1, -1)
        ).audio_codes[0]  # (B, n_q, T)


def train_and_eval(train_wav, clean_wav, steps=DEFAULT_STEPS, dev=None, seed=0):
    """Fine-tune on train_wav, then measure reproduction accuracy on clean_wav.

    Returns overall accuracy, per-codebook accuracy, and loss. The per-codebook
    breakdown is the important part -- see doc §4.2b.
    """
    dev = dev or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # model.generate() leaves autograd globally DISABLED. Training several models in one
    # process silently fails on the second with "element 0 of tensors does not require
    # grad" without this. (doc §5.6)
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    # attn_implementation="eager": transformers 5.x defaults to SDPA, which raises
    # "scaled_dot_product_attention for MPS does not support dropout" during TRAINING
    # (inference is fine, which is why generate/encode work without this).
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    ).to(dev)
    # These must be set on the DECODER sub-config, not the top-level config. (doc §5.6)
    model.config.decoder.decoder_start_token_id = DECODER_START
    model.config.decoder.pad_token_id = DECODER_START

    for p_ in model.parameters():
        p_.requires_grad_(False)
    dec = [
        p_
        for n_, p_ in model.named_parameters()
        if n_.startswith("decoder.") and "audio_encoder" not in n_
    ]
    for p_ in dec:
        p_.requires_grad_(True)

    model.train()
    # Labels must be (batch, time, n_codebooks) -- NOT (batch, n_codebooks, time).
    # The wrong shape throws a confusing reshape error. (doc §5.6)
    labels = tokens_of(model, train_wav, dev).transpose(1, 2).contiguous()
    inputs = proc(text=[PROMPT], padding=True, return_tensors="pt").to(dev)
    opt = torch.optim.AdamW(dec, lr=2e-4)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            out = model(**inputs, labels=labels)
            out.loss.backward()
            opt.step()

    # Measure reproduction of the CLEAN track: the attacker trains on the protected copy
    # but wants the real music.
    model.eval()
    lab = tokens_of(model, clean_wav, dev).transpose(1, 2).contiguous()
    with torch.no_grad():
        o = model(**inputs, labels=lab)
        pred = o.logits.argmax(-1)  # (B*n_q, T)
    B, T, nq = lab.shape
    L = lab.permute(0, 2, 1).reshape(B * nq, T)
    n = min(pred.shape[-1], L.shape[-1])
    per_codebook = {
        f"codebook_{q+1}": float((pred[q, :n] == L[q, :n]).float().mean())
        for q in range(nq)
    }
    return {
        "overall_accuracy": float((pred[..., :n] == L[..., :n]).float().mean()),
        "per_codebook": per_codebook,
        "loss": float(o.loss),
    }


def run_pair(track_path, note_frac=0.06, steps=DEFAULT_STEPS, seed=0, dev=None):
    """Full clean-vs-protected comparison for one track, with the baseline gate applied."""
    from protection import SR, audibility, load_excerpt, protect

    dev = dev or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    probe = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID).to(dev).eval()
    for p_ in probe.parameters():
        p_.requires_grad_(False)

    clean = load_excerpt(track_path, sr=SR)
    prot = protect(probe.audio_encoder, clean, SR, dev, steps=80, note_frac=note_frac)
    aud = audibility(clean, prot - clean, SR)

    c = train_and_eval(clean.astype(np.float32), clean.astype(np.float32), steps, dev, seed)
    p = train_and_eval(prot, clean.astype(np.float32), steps, dev, seed)

    gated = c["overall_accuracy"] < CLEAN_BASELINE_GATE
    return {
        "track": track_path,
        "note_frac": note_frac,
        "audibility": aud,
        "clean": c,
        "protected": p,
        "gap_pts": 100 * (c["overall_accuracy"] - p["overall_accuracy"]),
        "discarded": gated,
        "discard_reason": (
            f"clean baseline {c['overall_accuracy']:.3f} < {CLEAN_BASELINE_GATE} gate"
            if gated else None
        ),
    }


def print_result(r):
    print(f"\ntrack: {r['track']}")
    print(f"  note_frac={r['note_frac']}  audibility={r['audibility']:.3f}  (0.000 = fully masked)")
    if r["discarded"]:
        print(f"  DISCARDED: {r['discard_reason']}")
        print("  (clean model never memorized the track -- comparison is meaningless)")
        return
    print(f"  clean     overall {r['clean']['overall_accuracy']:.4f}")
    print(f"  protected overall {r['protected']['overall_accuracy']:.4f}")
    print(f"  GAP {r['gap_pts']:.1f} pts")
    print(f"  {'codebook':>12} {'clean':>8} {'protected':>10} {'gap':>7}")
    for k in r["clean"]["per_codebook"]:
        cv = r["clean"]["per_codebook"][k]
        pv = r["protected"]["per_codebook"][k]
        print(f"  {k:>12} {cv:>8.3f} {pv:>10.3f} {100*(cv-pv):>6.1f}")
    print("  (doc §4.2b: gap should RISE from codebook 1 to 4 -- damage lands in fine detail,")
    print("   which is why the audio still reproduces. Flattening this is the Tier-1 goal.)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("track", help="path to an audio file")
    ap.add_argument("--note-frac", type=float, default=0.06, help="0.02=A, 0.06=B, 0.15=C")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print_result(run_pair(args.track, args.note_frac, args.steps, args.seed))
