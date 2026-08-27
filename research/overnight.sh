#!/usr/bin/env bash
# Overnight runner. RESUMABLE: every measurement writes its own file before the next
# starts, so a kill loses at most the one in flight (session teardown has already destroyed
# two multi-hour runs with no output; nohup does not survive it).
#
#   ./overnight.sh              # run everything outstanding
#   ./overnight.sh --dry-run
#
# WHY THIS SHAPE. Style mimicry cannot currently be established: every config measured so
# far DEGRADES codebook 1 (batch 1 -> +0.277, batch 16 -> +0.249, community-standard full
# FT -> +0.395; all with shuffled/real ratio ~1.8, i.e. marginals not structure). Effective
# batch size was the leading suspect and has been ruled out. So rather than sweep more of a
# known-dead space, this run splits effort:
#
#   PHASE 1  two genuinely untried levers for a style baseline (codebook-1 loss weighting,
#            per-clip captions). Cheap; if either flips the cb1 sign it changes everything.
#   PHASE 2  protection on the threat model that DEMONSTRABLY WORKS -- single-track
#            memorization -- replicated properly. n=3 previously gave baseline 8.6 pts and
#            bilevel 19.6 pts (sd 9.6), too noisy to act on. This takes it to n~24.
#
# Phase 2 is where the morning's defensible number comes from.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs data/overnight

DRY="${1:-}"
ART=data/catalogue/6th_sense_big
run() { [ "$DRY" = "--dry-run" ] && { echo "    (dry-run)"; return 0; }; "$@"; }

echo "############ PHASE 1: untried style-baseline levers ($(date +%H:%M)) ############"
GATE_BASE="--artist-dir $ART --holdout 6 --lora-rank 32 --lr 3e-4 --steps 600 --accum 8"
for entry in \
  "cbw|--lora-targets attn --codebook-weights 4,1,1,1" \
  "titles|--lora-targets attn --captions titles" \
  "cbw_titles|--lora-targets attn --codebook-weights 4,1,1,1 --captions titles" \
  "all_cbw_titles|--lora-targets all --codebook-weights 4,1,1,1 --captions titles" \
; do
  tag="${entry%%|*}"; extra="${entry#*|}"
  out="data/overnight/gate_${tag}.json"
  [ -f "$out" ] && { echo "[skip] gate/$tag"; continue; }
  echo "[run ] gate/$tag  $extra  $(date +%H:%M)"
  run python -u structure_gate.py $GATE_BASE $extra --save "$out" \
      > "logs/gate_${tag}.log" 2>&1 || echo "[FAIL] gate/$tag"
done

echo
echo "############ PHASE 2: memorization protection, replicated ($(date +%H:%M)) ############"
# Single-track memorization is the one threat model where the clean baseline reliably works
# (clean reproduction accuracy ~1.00, gate-passing every time). Protection there is
# measurable; protection against style is not, yet.
TRACKS=$(ls $ART/*.wav | head -8)
for seed in 0 1 2; do
  for t in $TRACKS; do
    b=$(basename "$t" .wav)
    for obj in clean baseline bilevel; do
      marker="data/overnight/mem_${b:0:26}_${obj}_s${seed}.done"
      [ -f "$marker" ] && { echo "[skip] mem/$obj/s$seed/${b:0:20}"; continue; }
      echo "[run ] mem/$obj/s$seed/${b:0:20}  $(date +%H:%M)"
      if [ "$DRY" = "--dry-run" ]; then
        continue          # never create markers during a dry run
      fi
      if python -u replicate_bilevel.py --track "$t" --objective "$obj" --seed "$seed" \
           >> "logs/mem_s${seed}.log" 2>&1; then
        touch "$marker"
      else
        echo "[FAIL] mem/$obj/s$seed/${b:0:20}"
      fi
    done
  done
  echo "--- seed $seed done, interim summary ($(date +%H:%M)) ---"
  run python summarize_replication.py 2>/dev/null | tail -12 || true
done

echo
echo "############ SUMMARY ($(date)) ############"
run python summarize_overnight.py || true
echo
echo "===== memorization replication ====="
run python summarize_replication.py || true
