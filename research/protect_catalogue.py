#!/usr/bin/env python3
"""Protect every track in a catalogue. One process per track (MPS state accumulates).

Usage: python protect_catalogue.py --artist-dir D --out-dir O [--objective bilevel|variance]
       (loop over tracks externally, or pass --track for a single one)
"""
import argparse
from pathlib import Path
import numpy as np, soundfile as sf, torch
from protection import SR, audibility, load_excerpt, protect

ap=argparse.ArgumentParser()
ap.add_argument("--track",required=True); ap.add_argument("--out-dir",required=True)
ap.add_argument("--objective",default="bilevel",choices=["bilevel","variance"])
ap.add_argument("--mode",default="max",choices=["min","max"])
ap.add_argument("--note-frac",type=float,default=0.06); ap.add_argument("--seed",type=int,default=0)
a=ap.parse_args()
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
import soundfile as _sf
_y,_sr=_sf.read(a.track)
if abs(len(_y)/_sr-8.0)<0.01 and _sr==SR:
    clean=_y.astype("float64")          # already an 8s window
else:
    clean=load_excerpt(a.track,sr=SR)   # full track -> take the doc's excerpt
if a.objective=="bilevel":
    from protection_bilevel import protect_bilevel
    prot=protect_bilevel(clean,SR,dev,note_frac=a.note_frac,rounds=5,attacker_steps=100,
                         defender_steps=20,seed=a.seed,mode=a.mode,verbose=False)
else:
    from transformers import MusicgenForConditionalGeneration
    m=MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    prot=protect(m.audio_encoder,clean,SR,dev,steps=80,note_frac=a.note_frac)
sf.write(out/Path(a.track).name, prot, SR)
print(f"  {Path(a.track).name}: audibility {audibility(clean,prot-clean,SR):.4f}")
