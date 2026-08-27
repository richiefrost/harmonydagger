#!/usr/bin/env python3
"""
EXPERIMENTAL: targeted "decoy" protection -- the Mist/Nightshade approach, for audio.

WHY THIS RATHER THAN THE BASELINE
The doc's objective (§3.1) is error-MINIMIZING: collapse the latent's temporal variance so
the model has "nothing to learn". Three objective families are possible:

    error-minimizing   make the data uninformative      -> nothing to learn
    error-maximizing   maximize training loss           -> learn badly
    TARGETED           make it resemble a decoy         -> learn the WRONG THING

Mist (images) and Nightshade both use the third. This module implements it for audio:
push the protected audio's EnCodec latent toward the latent of a DIFFERENT piece of music,
so a model fine-tuned on the protected track learns the decoy's characteristics instead of
the artist's.

Why expect this to beat removal: our codebook-1 targeting experiment (protection_coarse.py)
increased coarse INPUT-token damage 3x yet the trained model reproduced coarse structure
BETTER. Best explanation is that coarse tokens are redundant and recoverable from context --
so *removing* information fails. Substituting a *self-consistent alternative* leaves nothing
to recover: the wrong answer is internally coherent.

    minimize   L(delta) = mean( ( E(x+delta) - E(decoy) )^2 )

Same masking ceiling, same projection-inside-the-loop discipline as the baseline.

NOT VALIDATED against a fine-tune yet. Screen with decoy_screen.py, then confirm with
listen.py --objective decoy. Input-token metrics have already misled us once here.
"""
import numpy as np
import torch

from protection import HOP, NFFT, build_ceiling


def protect_decoy(
    audio_encoder,
    clean,
    decoy,
    sr,
    dev,
    steps=80,
    lr=5e-3,
    note_frac=0.06,
    variance_weight=0.0,
):
    """Push the latent of `clean` toward the latent of `decoy`, under the masking ceiling.

    Args:
        clean: the track to protect (1-D float array).
        decoy: a DIFFERENT piece of music to impersonate. Must be at least as long as
            `clean`; it is truncated to match. Pick something maximally unlike the artist.
        variance_weight: optional blend of the baseline temporal-variance objective.
            0.0 = pure decoy targeting.
    """
    x = torch.tensor(clean, dtype=torch.float32, device=dev).view(1, 1, -1)
    n = x.shape[-1]
    ceil, win = build_ceiling(clean, sr, dev, note_frac)

    d = np.asarray(decoy, dtype=np.float32)
    if len(d) < len(clean):
        raise ValueError(
            f"decoy is shorter than the track ({len(d)} < {len(clean)} samples); "
            "use a longer decoy or a shorter excerpt"
        )
    dt = torch.tensor(d[: len(clean)], dtype=torch.float32, device=dev).view(1, 1, -1)

    # The representation we want the model to see instead of the real music.
    with torch.no_grad():
        lat_target = audio_encoder.encoder(dt)

    def project(delta):
        Z = torch.stft(
            delta.view(-1), n_fft=NFFT, hop_length=HOP, window=win, return_complex=True
        )
        mag = Z.abs()
        Tf = min(mag.shape[1], ceil.shape[1])
        Zc = Z.clone()
        m, c = mag[:, :Tf], ceil[:, :Tf]
        Zc[:, :Tf] = Z[:, :Tf] * torch.where(m > c, c / (m + 1e-12), torch.ones_like(m))
        return torch.istft(Zc, n_fft=NFFT, hop_length=HOP, window=win, length=n).view(
            1, 1, -1
        )

    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        lat = audio_encoder.encoder(x + delta)
        T = min(lat.shape[-1], lat_target.shape[-1])
        loss = torch.mean((lat[..., :T] - lat_target[..., :T]) ** 2)
        if variance_weight > 0.0:
            loss = loss + variance_weight * torch.mean(
                (lat - lat.mean(dim=-1, keepdim=True)) ** 2
            )
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.data.copy_(project(delta.detach()))  # constraint INSIDE the loop

    with torch.no_grad():
        return np.clip((x + delta).view(-1).cpu().numpy(), -1, 1).astype(np.float32)


def token_allegiance(model, clean, protected, decoy, dev):
    """Do the protected audio's tokens agree more with CLEAN or with the DECOY?

    This is the screen that actually matters for a targeted attack -- raw "how many tokens
    changed" says nothing about whether they moved toward the decoy. Returns per-codebook
    agreement fractions.

    STILL ONLY A SCREEN. Input-token movement is necessary but not sufficient; the coarse
    experiment showed input damage can even anti-correlate with learned damage. Confirm
    with a real fine-tune.
    """
    def codes(w):
        with torch.no_grad():
            return model.audio_encoder.encode(
                torch.tensor(np.asarray(w, dtype=np.float32), device=dev).view(1, 1, -1)
            ).audio_codes[0]

    c, p = codes(clean), codes(protected)
    k = codes(np.asarray(decoy)[: len(clean)])
    n = min(c.shape[-1], p.shape[-1], k.shape[-1])
    nq = c.shape[1]
    return {
        "agree_clean": [
            float((p[0, q, :n] == c[0, q, :n]).float().mean()) for q in range(nq)
        ],
        "agree_decoy": [
            float((p[0, q, :n] == k[0, q, :n]).float().mean()) for q in range(nq)
        ],
        "baseline_clean_vs_decoy": [
            float((c[0, q, :n] == k[0, q, :n]).float().mean()) for q in range(nq)
        ],
    }
