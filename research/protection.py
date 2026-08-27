#!/usr/bin/env python3
"""
The protection algorithm and audibility metric, per the research doc (§3.2, §3.3, §5.2, §5.3).

Two things to understand before changing anything here:

1. The perceptual constraint must be applied INSIDE the optimization loop, every step.
   Optimizing for effectiveness and then constraining destroys the structure.

2. EnCodec's quantizer is non-differentiable, so you cannot backpropagate from the LM's
   token loss to the waveform. We attack `audio_encoder.encoder` -- the continuous,
   pre-quantizer output.

Protection levels (note_frac): A = 0.02, B = 0.06, C = 0.15.
Human listening put B as the usable ceiling. Default to B.
"""
import numpy as np
import torch
from scipy.signal import stft

NFFT, HOP, SR = 2048, 512, 32000

# Protection levels from the research doc. B is the usable ceiling per human A/B listening.
LEVELS = {"A": 0.02, "B": 0.06, "C": 0.15}


def hz_to_bark(f):
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def build_ceiling(clean, sr, dev, note_frac=0.06, silence_rel_db=-40.0, band_barks=2.0):
    """Per-(freq, frame) magnitude ceiling: noise rides the dominant note, silence stays silent.

    Three properties that matter:
      - Noise rides loud notes -- energy capped as a fraction of the loudest note present,
        so that note masks it.
      - Silence stays silent -- active(t)=0 in quiet passages. This was the single biggest
        audibility fix; leaking energy into quiet passages and note decays is immediately
        audible.
      - Critical-band taper -- energy stays within ~2 Bark of the masker.
    """
    x = torch.tensor(clean, dtype=torch.float32, device=dev)
    win = torch.hann_window(NFFT, device=dev)
    Z = torch.stft(x, n_fft=NFFT, hop_length=HOP, window=win, return_complex=True)
    mag = Z.abs()  # (F, T)
    F = mag.shape[0]
    bark = torch.tensor(
        hz_to_bark(np.linspace(0, sr / 2, F)), dtype=torch.float32, device=dev
    )  # (F,)

    dom_mag, dom_idx = mag.max(dim=0)  # loudest bin per frame
    rel_db = 20.0 * torch.log10(dom_mag / (dom_mag.max() + 1e-12) + 1e-12)
    active = (rel_db > silence_rel_db).float()  # silence gate

    dist = (bark[:, None] - bark[dom_idx][None, :]).abs()  # bark distance to the masker
    taper = torch.clamp(1.0 - dist / band_barks, min=0.0)  # critical-band taper

    ceil = note_frac * dom_mag[None, :] * taper * active[None, :]
    return torch.clamp(ceil, min=1e-6), win


def protect(audio_encoder, clean, sr, dev, steps=80, lr=5e-3, note_frac=0.06):
    """Error-minimizing perturbation, projected under the masking ceiling every step.

    Objective: collapse the EnCodec latent's temporal variance, so the LM sees no
    sequential structure to learn.
    """
    x = torch.tensor(clean, dtype=torch.float32, device=dev).view(1, 1, -1)
    n = x.shape[-1]
    ceil, win = build_ceiling(clean, sr, dev, note_frac)

    def project(delta):
        Z = torch.stft(
            delta.view(-1), n_fft=NFFT, hop_length=HOP, window=win, return_complex=True
        )
        mag = Z.abs()
        Tf = min(mag.shape[1], ceil.shape[1])
        Zc = Z.clone()
        m, c = mag[:, :Tf], ceil[:, :Tf]
        Zc[:, :Tf] = Z[:, :Tf] * torch.where(m > c, c / (m + 1e-12), torch.ones_like(m))
        return torch.istft(
            Zc, n_fft=NFFT, hop_length=HOP, window=win, length=n
        ).view(1, 1, -1)

    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        lat = audio_encoder.encoder(x + delta)  # continuous, pre-quantizer
        loss = torch.mean((lat - lat.mean(dim=-1, keepdim=True)) ** 2)  # collapse temporal variance
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.data.copy_(project(delta.detach()))  # constraint INSIDE the loop
    with torch.no_grad():
        return np.clip((x + delta).view(-1).cpu().numpy(), -1, 1).astype(np.float32)


def audibility(clean, noise, sr, nperseg=2048):
    """Fraction of perturbation energy above the clean signal's MASKING threshold.

    0.000 = fully masked. Validated against human ears: masked 0.000, white noise 0.735,
    high-freq noise 0.814.

    Do NOT substitute SNR or "energy above absolute hearing threshold" -- both were tried
    and both disagreed with human listeners, one badly (it rated the masked perturbation
    as MORE audible than white noise, because it ignored masking-in-context).
    """
    f, _, Zc = stft(clean, fs=sr, nperseg=nperseg)
    mag = np.abs(Zc)
    ref = mag.max() + 1e-12
    spl = 20 * np.log10(mag / ref + 1e-12) + 96.0

    bark = hz_to_bark(f)
    B = bark[:, None] - bark[None, :]
    spread = np.where(B >= 0, -27.0 * B, 15.0 * B)  # dB per bark, below/above
    thr = (
        np.stack(
            [(spl[None, :, t] + spread).max(axis=1) for t in range(spl.shape[1])], axis=1
        )
        - 6.0
    )  # 6 dB safety offset

    fk = np.maximum(f, 1.0) / 1000.0
    absthr = 3.64 * fk**-0.8 - 6.5 * np.exp(-0.6 * (fk - 3.3) ** 2) + 1e-3 * fk**4
    thr = np.maximum(thr, absthr[:, None])

    _, _, Zn = stft(noise, fs=sr, nperseg=nperseg)
    nmag = np.abs(Zn)
    n_db = 20 * np.log10(nmag / ref + 1e-12) + 96.0
    return float((nmag**2 * (n_db > thr)).sum() / (nmag**2).sum() + 1e-12)


def load_excerpt(path, sr=SR, offset_s=1.0, dur_s=8.0, peak=0.7):
    """Load an 8-second excerpt starting 1 s in, peak-normalized to 0.7.

    Matches the research doc's preprocessing exactly -- don't change these defaults
    without re-baselining, since all the reported numbers assume them.
    """
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True, duration=offset_s + dur_s + 2.0)
    start = int(sr * offset_s)
    clip = y[start : start + int(sr * dur_s)]
    if len(clip) < int(sr * dur_s):
        raise ValueError(f"{path}: too short for a {dur_s}s excerpt at {offset_s}s offset")
    return (clip / (np.abs(clip).max() + 1e-9) * peak).astype(np.float64)


def _self_test():
    """Validate the audibility metric against the doc's reference values."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 8, SR * 8, endpoint=False)
    music = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)) * 0.7

    white = rng.normal(0, 0.01, len(music))
    hi = np.sin(2 * np.pi * 12000 * t) * 0.01

    print("audibility metric self-test (expect white/high-freq to score high):")
    print(f"  white noise    : {audibility(music, white, SR):.3f}   (doc: 0.735)")
    print(f"  high-freq noise: {audibility(music, hi, SR):.3f}   (doc: 0.814)")
    print("  (a real masked perturbation should score ~0.00-0.11)")


if __name__ == "__main__":
    _self_test()
