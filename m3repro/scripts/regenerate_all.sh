#!/usr/bin/env bash
# Regenerate every result from scratch, compare against the committed versions, run the harness.
set -uo pipefail
cd "$(dirname "$0")/.."
L=results/regenerate.log; : > "$L"
run() { echo "=== $* ($(date -u +%T))" | tee -a "$L"; "$@" >> "$L" 2>&1 || echo "FAILED: $*" | tee -a "$L"; }
for e in scaling nll tokenizer rollout; do run python3 -I scripts/run_audit.py $e --rollouts 32 --new 1024; done
run python3 -I scripts/run_audit.py decode_ablation
run python3 -I scripts/compare_rerun.py results/scaling.json results/nll.json results/tokenizer.json results/rollout.json results/decode_ablation.json
run python3 -I scripts/run_twap.py --model tiny  --horizon 60 --children 10 --reps 6  --qgrid 10000,40000,160000
run python3 -I scripts/run_twap.py --model small --horizon 60 --children 10 --reps 6  --qgrid 10000,40000,160000
run python3 -I scripts/run_twap.py --model small --horizon 60 --children 10 --reps 12 --qgrid 5000,12500,31250,78000,195000 --tag _q5r12
for f in results/twap_tiny_h60.json results/twap_small_h60.json results/twap_small_h60_q5r12.json; do run python3 -I scripts/twap_bootstrap.py "$f"; done
echo "=== DONE ($(date -u +%T))" | tee -a "$L"
