#!/usr/bin/env python3
"""
EXPERIMENTAL: bi-level / training-aware protection -- the Mist-v2 mechanism, for audio.

WHY THIS IS THE MOST PROMISING DIRECTION LEFT

The doc's objective (§3.1) optimizes "gradients flowing through the encoder only (never the
full LM)". But §2 describes HarmonyCloak as *bi-level* optimization. So the implementation
is HarmonyCloak minus the bi-level part -- and that omission predicts exactly what we
measured: a perturbation never optimized against training gets ERODED by training
(78 pts at 200 steps -> 8.7 pts at 800). Mist v2's whole point is to iterate the
perturbation against an actual LoRA fine-tuning process so the damage survives convergence.

THE BLOCKER, AND THE WAY AROUND IT

Mist v2 can backprop pixels <- LoRA loss because Stable Diffusion is differentiable
end-to-end. Here the audio -> token path runs through a non-differentiable argmin (doc §2),
and cross-entropy has no gradient w.r.t. integer target indices.

Relaxation: replace hard token targets with a SOFT distribution over codebook entries,

    soft_k(t) = softmax( -||latent(x+delta)_t - e_k||^2 / tau )

which is differentiable in delta. Verified: gradients reach delta with norm ~1.3.
The decoder input path still uses hard (detached) labels -- standard teacher forcing --
so only the target is relaxed.

ALGORITHM (alternating, as in Anti-DreamBooth's ASPL and Mist v2)

    delta = 0;  theta = pretrained decoder
    repeat R rounds:
        (a) ATTACKER step  -- fine-tune theta for K steps on hard labels from x+delta.
                              This tracks what the attacker actually learns.
        (b) DEFENDER step  -- update delta for M steps against the CURRENT theta,
                              projected under the masking ceiling every step.

Because theta chases delta and delta chases theta, the perturbation is selected for damage
that survives training rather than damage that training erodes.

DIRECTION: `mode="min"` (default) is error-MINIMIZING -- make the protected audio trivially
predictable to the current model, so it learns a shortcut instead of the music. This is what
HarmonyCloak and the unlearnable-examples literature use. `mode="max"` is error-maximizing
(make it hard to fit); included for comparison since the two behave differently.

COST: roughly 0.33 s per attacker step and ~0.5 s per defender step on an M5 Pro, so
R=5, K=100, M=20 lands around 4-5 min. Verify the OUTPUT with a real 800-step fine-tune --
input-space metrics have misled us twice on this project.
"""
import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from protection import HOP, NFFT, build_ceiling

MODEL_ID = "facebook/musicgen-small"
DECODER_START = 2048
PROMPT = "instrumental music"


def _hard_codes(audio_encoder, wav_tensor):
    """Hard EnCodec codes, shaped (B, T, n_q) for the LM's `labels` argument."""
    with torch.no_grad():
        codes = audio_encoder.encode(wav_tensor).audio_codes[0]  # (B, n_q, T)
    return codes.transpose(1, 2).contiguous()


def _soft_targets(audio_encoder, wav_tensor, tau=1.0):
    """Differentiable soft distributions over each RVQ codebook.

    Follows the residual chain: layer q quantizes what layers 0..q-1 left over. The
    residual is advanced with the HARD quantized vector (detached) to keep the chain
    numerically stable, while the soft distribution at each layer carries the gradient.

    Returns a list of (B, T, K) tensors, one per codebook.
    """
    lat = audio_encoder.encoder(wav_tensor)  # (B, D, T)
    residual = lat.transpose(1, 2)  # (B, T, D)
    softs = []
    for layer in audio_encoder.quantizer.layers:
        embed = layer.codebook.embed  # (K, D)
        d = (
            residual.pow(2).sum(-1, keepdim=True)
            - 2.0 * (residual @ embed.t())
            + embed.pow(2).sum(-1)[None, None, :]
        )
        softs.append(torch.softmax(-d / tau, dim=-1))
        idx = d.argmin(-1)  # (B, T)
        q = embed[idx]  # (B, T, D)
        residual = residual - q.detach()
    return softs


#: Floor on log-probabilities in the soft cross-entropy. After the attacker fine-tunes to
#: convergence its logits are near one-hot, so log_softmax returns very large negatives;
#: soft targets placing mass there blow the loss up to inf and then NaN. -30 corresponds to
#: p ~ 1e-13, far below anything meaningful, and keeps gradients finite.
LOGP_FLOOR = -30.0


def _soft_ce(logits, softs):
    """Cross-entropy of the LM's logits against the soft targets, averaged over codebooks.

    logits: (B*n_q, T, V) as MusicGen returns.  softs: list of n_q x (B, T, K).
    """
    nq = len(softs)
    total = 0.0
    for q in range(nq):
        s = softs[q][0]  # (T, K)
        lp = torch.log_softmax(logits[q], dim=-1).clamp(min=LOGP_FLOOR)  # (T, V)
        t = min(s.shape[0], lp.shape[0])
        k = min(s.shape[1], lp.shape[1])
        total = total + (-(s[:t, :k] * lp[:t, :k]).sum(-1).mean())
    return total / nq


def protect_bilevel(
    clean,
    sr,
    dev,
    note_frac=0.06,
    rounds=5,
    attacker_steps=100,
    defender_steps=20,
    attacker_lr=2e-4,
    defender_lr=5e-3,
    tau=1.0,
    mode="min",
    seed=0,
    verbose=True,
):
    """Alternating bi-level protection. Returns the protected waveform."""
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")

    torch.set_grad_enabled(True)
    torch.manual_seed(seed)

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    ).to(dev)
    model.config.decoder.decoder_start_token_id = DECODER_START
    model.config.decoder.pad_token_id = DECODER_START

    # Surrogate = the decoder, exactly what the attacker fine-tunes (doc §3.4).
    for p in model.parameters():
        p.requires_grad_(False)
    dec_params = [
        p
        for n, p in model.named_parameters()
        if n.startswith("decoder.") and "audio_encoder" not in n
    ]
    ae = model.audio_encoder

    x = torch.tensor(clean, dtype=torch.float32, device=dev).view(1, 1, -1)
    n = x.shape[-1]
    ceil, win = build_ceiling(clean, sr, dev, note_frac)
    inputs = proc(text=[PROMPT], padding=True, return_tensors="pt").to(dev)

    def project(d):
        Z = torch.stft(
            d.view(-1), n_fft=NFFT, hop_length=HOP, window=win, return_complex=True
        )
        mag = Z.abs()
        Tf = min(mag.shape[1], ceil.shape[1])
        Zc = Z.clone()
        mm, cc = mag[:, :Tf], ceil[:, :Tf]
        Zc[:, :Tf] = Z[:, :Tf] * torch.where(mm > cc, cc / (mm + 1e-12), torch.ones_like(mm))
        return torch.istft(Zc, n_fft=NFFT, hop_length=HOP, window=win, length=n).view(
            1, 1, -1
        )

    delta = torch.zeros_like(x, requires_grad=True)
    d_opt = torch.optim.Adam([delta], lr=defender_lr)
    a_opt = torch.optim.AdamW(dec_params, lr=attacker_lr)

    for r in range(rounds):
        # ---- (a) attacker: fine-tune the surrogate on the current protected audio ----
        for p in dec_params:
            p.requires_grad_(True)
        model.train()
        with torch.no_grad():
            labels = _hard_codes(ae, (x + delta).detach())
        a_loss = None
        for _ in range(attacker_steps):
            a_opt.zero_grad()
            out = model(**inputs, labels=labels)
            out.loss.backward()
            a_opt.step()
            a_loss = float(out.loss)
            del out
        # The attacker's Adam state (420M params) plus the defender's autograd graph
        # through the encoder will trip MPS ("command buffer exited with error status",
        # surfacing as a bogus shape mismatch) unless the cache is released between phases.
        if dev.type == "mps":
            torch.mps.empty_cache()

        # ---- (b) defender: update delta against the surrogate as it now stands ----
        for p in dec_params:
            p.requires_grad_(False)
        model.eval()
        d_loss = None
        for _ in range(defender_steps):
            d_opt.zero_grad()
            softs = _soft_targets(ae, x + delta, tau=tau)
            with torch.no_grad():
                hard = _hard_codes(ae, (x + delta).detach())
            logits = model(**inputs, labels=hard).logits
            ce = _soft_ce(logits, softs)
            # min => make the protected audio trivially predictable (a shortcut)
            loss = ce if mode == "min" else -ce
            loss.backward()
            # An overfit surrogate yields very peaked logits and correspondingly large
            # gradients; without clipping delta diverges to NaN within a round.
            torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
            if delta.grad is not None and not torch.isfinite(delta.grad).all():
                d_opt.zero_grad()
                if verbose:
                    print("      (skipped a defender step: non-finite gradient)")
                continue
            d_opt.step()
            with torch.no_grad():
                projected = project(delta.detach())
                if torch.isfinite(projected).all():
                    delta.data.copy_(projected)
                else:
                    if verbose:
                        print("      (projection produced non-finite values; keeping previous delta)")
            d_loss = float(ce)
            del softs, logits, ce, loss

        if dev.type == "mps":
            torch.mps.empty_cache()

        if verbose:
            print(
                f"      round {r+1}/{rounds}  attacker_loss={a_loss:.4f}  "
                f"defender_ce={d_loss:.4f}  |delta|rms={float(delta.detach().pow(2).mean().sqrt()):.6f}"
            )

    with torch.no_grad():
        return np.clip((x + delta).view(-1).cpu().numpy(), -1, 1).astype(np.float32)
