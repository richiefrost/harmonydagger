#!/usr/bin/env python3
"""
EXPERIMENTAL: coarse-targeted protection objective. Doc §7 Tier-1 #1.

Motivation. The baseline objective (protection.py) collapses the pooled EnCodec latent's
temporal variance. §4.2b showed the resulting damage is fine-codebook-weighted (cb4 -59
pts, cb1 only -18), and coarse_screen.py showed cb1 input-token change SATURATES around
5% no matter how hard you push -- even at note_frac=1.0, far past audible. Style
(timbre, groove, harmonic vocabulary) lives in coarse structure, so a style-mimicry
defense needs codebook-1 damage.

Approach. EnCodec's quantizer is residual VQ: layer 0 quantizes the latent directly, and
later layers quantize the residual. So codebook 1's assignment is argmin over layer 0's
2048 centroids. argmin isn't differentiable, but we can make it soft:

    logits = -||latent - centroid_k||^2 / temperature
    loss   = log softmax(logits)[k_original]      <- minimize

i.e. actively push each frame's latent OFF the centroid the clean audio selected, across
a Voronoi boundary. Same masking ceiling, same projection-inside-the-loop discipline.

This distinguishes two hypotheses for the cb1 saturation:
  (a) the OBJECTIVE never pushes toward coarse boundaries  -> this should fix it
  (b) the masking CONSTRAINT (±2 Bark around tonal peaks) structurally cannot move
      coarse tokens -> cb1 stays saturated even with a coarse-targeted objective,
      and the answer to the project is more fundamental

Not yet validated against a real fine-tune. Screen with coarse_screen.py, then confirm
with listen.py / finetune_eval.py before reporting anything.
"""
import numpy as np
import torch

from protection import NFFT, HOP, SR, build_ceiling


def _sq_dists(latent, embed):
    """Squared Euclidean distance from each frame's latent to every centroid.

    latent: (B, D, T)   embed: (K, D)   ->   (B, T, K)
    """
    h = latent.transpose(1, 2)  # (B, T, D)
    # ||h||^2 - 2 h.e + ||e||^2
    h2 = h.pow(2).sum(-1, keepdim=True)  # (B, T, 1)
    e2 = embed.pow(2).sum(-1)[None, None, :]  # (1, 1, K)
    return h2 - 2.0 * (h @ embed.t()) + e2


def protect_coarse(
    audio_encoder,
    quantizer,
    clean,
    sr,
    dev,
    steps=80,
    lr=5e-3,
    note_frac=0.06,
    temperature=1.0,
    variance_weight=0.0,
):
    """Error-minimizing perturbation targeted at CODEBOOK 1 assignments.

    Args:
        variance_weight: optional blend of the baseline temporal-variance objective.
            0.0 = pure coarse targeting, 1.0 = equal mix.
    """
    x = torch.tensor(clean, dtype=torch.float32, device=dev).view(1, 1, -1)
    n = x.shape[-1]
    ceil, win = build_ceiling(clean, sr, dev, note_frac)

    embed = quantizer.layers[0].codebook.embed.detach().to(dev)  # (K, D)

    # The codebook-1 assignment the CLEAN audio selects -- the thing we want to move off.
    with torch.no_grad():
        lat0 = audio_encoder.encoder(x)
        k_orig = _sq_dists(lat0, embed).argmin(-1)  # (B, T)

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

        d = _sq_dists(lat, embed)  # (B, T, K)
        logp = torch.log_softmax(-d / temperature, dim=-1)
        # Minimize log-probability of the ORIGINAL code => push across the boundary.
        kt = k_orig[..., : logp.shape[1]]
        loss = logp.gather(-1, kt.unsqueeze(-1)).mean()

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
