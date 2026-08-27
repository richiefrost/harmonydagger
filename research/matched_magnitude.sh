#!/usr/bin/env bash
# Is bi-level's advantage REAL, or just a larger perturbation?
#
# gap correlates with delta_rms at r=+0.88, and bi-level reaches rms 0.0107 where the
# baseline objective at note_frac 0.06 tops out at 0.0055. So the "3x better" result may be
# amplitude, not the training-aware objective being smarter.
#
# Test: raise the baseline objective's note_frac until its magnitude matches bi-level's
# high cluster (rms ~0.0072), then compare gap AND audibility on the same tracks/seed.
#   - baseline reaches ~22 pts at similar audibility  -> bi-level adds nothing; drop it
#   - baseline needs much higher audibility           -> bi-level really does place noise better
set -uo pipefail
cd "$(dirname "$0")"
ART=data/catalogue/6th_sense_big
for nf in 0.10 0.15 0.25; do
  for t in $(ls $ART/*.wav | head -4); do
    b=$(basename "$t" .wav)
    m="data/overnight/mm_${b:0:22}_nf${nf}.done"
    [ -f "$m" ] && { echo "[skip] nf=$nf ${b:0:20}"; continue; }
    echo "[run ] baseline nf=$nf ${b:0:20}  $(date +%H:%M)"
    python -u replicate_bilevel.py --track "$t" --objective baseline --seed 0 \
      --note-frac "$nf" >> logs/matched_magnitude.log 2>&1 && touch "$m" \
      || echo "[FAIL] nf=$nf ${b:0:20}"
  done
done
echo "=== done $(date +%H:%M) ==="
