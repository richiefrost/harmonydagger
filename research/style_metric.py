#!/usr/bin/env python3
"""
CLAP-based style similarity -- the readout the project has been missing.

WHY NOT TOKEN ACCURACY. Reproduction accuracy answers "can the model regurgitate this
waveform". Style mimicry asks "can it generate NEW music that sounds like this artist".
Those come apart, and token accuracy turns out to be a poor proxy: shifting audio by ONE
SAMPLE (0.03 ms, inaudible, harmless) changes ~31% of EnCodec tokens -- comparable to the
damage from our strongest protection. A metric a 1-sample shift can fake is not measuring
protection.

VALIDATED SEPARATION on the FMA catalogues (8s excerpts, level-B corpus):
    within  6th Sense (hip-hop instrumentals)  mean cos +0.699
    within  Peter Biedermann (solo guitar)     mean cos +0.940
    across  the two artists                    mean cos +0.460
    separation +0.42

So the scale is interpretable: ~0.46 means "unrelated artist", 0.70-0.94 means "same
artist". Protection works if a protected-trained model's generations move DOWN toward 0.46.

CLAP runs at 48 kHz; audio is resampled here. Embeddings are L2-normalised so cosine is
just a dot product.
"""
import numpy as np
import torch

CLAP_ID = "laion/clap-htsat-unfused"
CLAP_SR = 48000

_cache = {}


def _load(dev):
    if "m" not in _cache:
        from transformers import ClapModel, ClapProcessor

        _cache["p"] = ClapProcessor.from_pretrained(CLAP_ID)
        _cache["m"] = ClapModel.from_pretrained(CLAP_ID).to(dev).eval()
        for p in _cache["m"].parameters():
            p.requires_grad_(False)
    return _cache["p"], _cache["m"]


def embed(audio, sr, dev):
    """L2-normalised CLAP audio embedding for a 1-D waveform."""
    import librosa

    proc, model = _load(dev)
    y = np.asarray(audio, dtype=np.float32)
    if sr != CLAP_SR:
        y = librosa.resample(y.astype("float64"), orig_sr=sr, target_sr=CLAP_SR).astype(
            "float32"
        )
    inputs = proc(audio=y, sampling_rate=CLAP_SR, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    # transformers versions differ: tensor, or an output object with pooler_output
    if torch.is_tensor(out):
        e = out
    elif getattr(out, "pooler_output", None) is not None:
        e = out.pooler_output
    else:
        e = out.last_hidden_state.mean(1)
    e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy()[0]


def style_similarity(generations, reference_audios, sr, dev):
    """Mean cosine between each generation and each reference (held-out artist) track.

    Pass HELD-OUT tracks as references -- tracks the model never trained on. Using the
    training tracks would measure memorization again, which is the trap this whole metric
    exists to avoid.
    """
    ge = [embed(g, sr, dev) for g in generations]
    re = [embed(r, sr, dev) for r in reference_audios]
    sims = [float(np.dot(g, r)) for g in ge for r in re]
    return {
        "mean": float(np.mean(sims)),
        "sd": float(np.std(sims)),
        "n": len(sims),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
    }
