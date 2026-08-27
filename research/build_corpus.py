#!/usr/bin/env python3
"""
Build a single-artist test corpus from the Free Music Archive.

Why single-artist: the PoC asks whether protection stops a model from learning an
artist's STYLE. That only means something if there's a consistent style to learn, so a
grab-bag of unrelated royalty-free clips won't work -- the model needs several tracks
from one artist.

Data source: benjamin-paine/free-music-archive-small on HuggingFace (CC-licensed,
ungated, ships artist/license metadata alongside the audio).

Note on variants: use -small. The -large and -full variants have 171 and 972 parquet
shards respectively and streaming them times out; -small has 15 shards at ~458 MB each,
and one shard already contains 528 tracks across 178 artists -- plenty for a PoC.

Usage:
    python build_corpus.py --list                    # show artist options
    python build_corpus.py --artist "6th Sense"      # extract that artist
    python build_corpus.py --artist "6th Sense" --protected   # + protected copies
"""
import argparse
import collections
import io
import json
from pathlib import Path

SHARD = Path("data/fma/data/train-00000-of-00015.parquet")
REPO_ID = "benjamin-paine/free-music-archive-small"
SHARD_NAME = "data/train-00000-of-00015.parquet"

# License codes seen in the FMA metadata are opaque integers, so we gate on the
# explicit boolean permission fields instead, which is what actually matters for us.
REQUIRED_FIELDS = ("allow_derivatives",)


def ensure_shard():
    """Download the parquet shard if it isn't already local (~458 MB, ~2 min)."""
    if SHARD.exists():
        return SHARD
    from huggingface_hub import hf_hub_download

    print(f"downloading {SHARD_NAME} (~458 MB, a couple of minutes)...")
    hf_hub_download(
        repo_id=REPO_ID, filename=SHARD_NAME, repo_type="dataset", local_dir="data/fma"
    )
    return SHARD


def scan_artists():
    """Return {artist: [(title, allow_derivatives, instrumental), ...]}."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(ensure_shard())
    artists = collections.defaultdict(list)
    cols = ["artist", "title", "allow_derivatives", "instrumental"]
    for batch in pf.iter_batches(batch_size=200, columns=cols):
        d = batch.to_pydict()
        for i, artist in enumerate(d["artist"]):
            if artist:
                artists[artist].append(
                    (d["title"][i], d["allow_derivatives"][i], d["instrumental"][i])
                )
    return artists


def cmd_list():
    artists = scan_artists()
    total = sum(len(v) for v in artists.values())
    print(f"{total} tracks, {len(artists)} artists in this shard\n")
    print(f"{'tracks':>7} {'deriv_ok':>9} {'instrum':>8}  artist")
    for artist, tracks in sorted(artists.items(), key=lambda kv: -len(kv[1]))[:20]:
        derivs = sum(1 for t in tracks if t[1])
        inst = sum(1 for t in tracks if t[2])
        print(f"{len(tracks):>7} {derivs:>9} {inst:>8}  {artist}")
    print("\nWant >=8 tracks. Prefer instrumental (no vocals = cleaner style signal,")
    print("and it sidesteps the voice-cloning question entirely).")
    print("\nRecommended: '6th Sense' -- 22 tracks, all instrumental, all derivatives-OK.")


def cmd_extract(artist_name, also_protected, note_frac, max_tracks, out_dir=None):
    """Write one artist's tracks to data/corpus/clean/ as 32 kHz mono WAV.

    32 kHz mono because that's MusicGen's native audio_encoder sample rate -- doing the
    conversion once here keeps the training loop simple.
    """
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    clean_dir = Path(out_dir) if out_dir else Path("data/corpus/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(ensure_shard())
    manifest = []
    count = 0

    for batch in pf.iter_batches(batch_size=100):
        d = batch.to_pydict()
        for i, a in enumerate(d["artist"]):
            if a != artist_name or count >= max_tracks:
                continue
            if not d["allow_derivatives"][i]:
                print(f"  skipping (derivatives not permitted): {d['title'][i]}")
                continue

            raw = d["audio"][i]["bytes"]
            audio, sr = sf.read(io.BytesIO(raw))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # MusicGen operates at 32 kHz; resample once here.
            if sr != 32000:
                import librosa

                audio = librosa.resample(audio.astype("float64"), orig_sr=sr, target_sr=32000)
                sr = 32000
            # Guard against the clipping we saw in the source mp3s (peak > 1.0).
            peak = np.abs(audio).max()
            if peak > 0.999:
                audio = audio / (peak * 1.001)

            stem = f"{count:02d}_{_slug(d['title'][i])}"
            out = clean_dir / f"{stem}.wav"
            sf.write(out, audio, sr)
            manifest.append({
                "file": str(out), "title": d["title"][i], "artist": a,
                "duration_s": round(len(audio) / sr, 1), "source": "FMA (CC)",
                "allow_derivatives": bool(d["allow_derivatives"][i]),
                "instrumental": bool(d["instrumental"][i]),
                "license_code": d["license"][i], "url": d.get("url", [None])[i],
            })
            count += 1
            print(f"  [{count}] {d['title'][i]} ({len(audio)/sr:.0f}s)")

    if not manifest:
        print(f"no tracks found for artist {artist_name!r} -- run --list to see options")
        return

    (clean_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {count} clean tracks to {clean_dir}/ + manifest.json")

    if also_protected:
        _write_protected(manifest, note_frac)


def _write_protected(manifest, note_frac):
    """Apply the research doc's protection (protection.py) to every clean track.

    Reports the doc's validated masking-based audibility metric, NOT SNR -- §3.3 shows
    energy-based metrics disagree with human ears. 0.000 = fully masked.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import MusicgenForConditionalGeneration

    from protection import SR, audibility, load_excerpt, protect

    prot_dir = Path(f"data/corpus/protected_nf{note_frac}")
    prot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\napplying protection (note_frac={note_frac})...")

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = (
        MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        .to(dev)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)

    auds = []
    for entry in manifest:
        # Use the doc's excerpt convention (8 s at 1 s offset, peak 0.7) so the protected
        # copy lines up with what finetune_eval.py evaluates.
        clean = load_excerpt(entry["file"], sr=SR)
        prot = protect(model.audio_encoder, clean, SR, dev, steps=80, note_frac=note_frac)
        out = prot_dir / Path(entry["file"]).name
        sf.write(out, prot, SR)
        aud = audibility(clean, prot - clean, SR)
        auds.append(aud)
        print(f"  {Path(entry['file']).name}: audibility {aud:.4f}")

    print(f"\nwrote {len(auds)} protected tracks to {prot_dir}/")
    print(f"mean audibility {np.mean(auds):.4f} (0.000 = fully masked; doc expects ~0.00-0.11)")


def _slug(s):
    keep = [c if c.isalnum() else "_" for c in (s or "untitled").lower()]
    return "".join(keep).strip("_")[:40]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show artists in the shard")
    ap.add_argument("--artist", help="artist to extract")
    ap.add_argument("--protected", action="store_true", help="also write protected copies")
    ap.add_argument(
        "--note-frac", type=float, default=0.06,
        help="protection level: 0.02=A, 0.06=B (usable ceiling per human listening), 0.15=C",
    )
    ap.add_argument("--max-tracks", type=int, default=20)
    ap.add_argument("--out-dir", help="write here instead of data/corpus/clean (for per-artist catalogues)")
    args = ap.parse_args()

    if args.list or not args.artist:
        cmd_list()
    else:
        cmd_extract(args.artist, args.protected, args.note_frac, args.max_tracks, args.out_dir)
