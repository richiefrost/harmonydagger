#!/usr/bin/env bash
# Drive the baseline-vs-bilevel replication, one fresh process per measurement.
# Each fine-tune is ~4.4 min; 3 measurements per track (clean, baseline, bilevel).
#
#   ./replicate_bilevel.sh                    # all tracks, seed 0
#   ./replicate_bilevel.sh "0 1 2"            # seeds 0,1,2
set -uo pipefail
cd "$(dirname "$0")"

SEEDS="${1:-0}"
TRACKS=$(ls data/corpus/clean/*.wav 2>/dev/null)
[ -z "$TRACKS" ] && { echo "no corpus -- run build_corpus.py first"; exit 1; }

n=0
for t in $TRACKS; do for s in $SEEDS; do n=$((n+3)); done; done
echo "==> $n measurements, ~4.4 min each => roughly $(( n * 44 / 10 )) min"
echo

for seed in $SEEDS; do
  for t in $TRACKS; do
    for obj in clean baseline bilevel; do
      # A failure on one measurement must not abort the sweep.
      if ! python replicate_bilevel.py --track "$t" --objective "$obj" --seed "$seed" 2>/dev/null; then
        echo "  FAILED: $(basename "$t") $obj seed=$seed"
      fi
    done
  done
done

echo
python summarize_replication.py
