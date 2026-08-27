#!/usr/bin/env python3
"""Materialize each track's 8s windows as separate wav files, so they can be protected
individually and fed to the protected arm. Usage: python windowize.py IN_DIR OUT_DIR [N]"""
import sys
from pathlib import Path
import soundfile as sf
from catalogue import load_windows
from protection import SR

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
dst.mkdir(parents=True, exist_ok=True)
c = 0
for t in sorted(src.glob("*.wav")):
    for i, w in enumerate(load_windows(str(t), max_windows=n)):
        sf.write(dst / f"{t.stem}__w{i}.wav", w, SR)
        c += 1
print(f"wrote {c} windows ({c*8}s) to {dst}/")
