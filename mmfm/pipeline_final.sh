#!/usr/bin/env bash
# Final, pre-registered pipeline (EVAL.md). Run once the collection span satisfies addendum 1.
#   CUT_NS=<ns> MMFM_MIN=<minutes> B2_MIN=<minutes> CTX=<ctx> LR=<lr> BATCH=<b> bash pipeline_final.sh
set -euo pipefail
cd "$(dirname "$0")"
EV=/home/user/data/mmfm_events
L=results/final_pipeline.log; mkdir -p results; : > "$L"
log(){ echo "=== $* ($(date -u +%T))" | tee -a "$L"; }
log build; python3 -I data/build.py --out $EV --until-ns "$CUT_NS" >> "$L" 2>&1
log splits; python3 -I data/splits.py --events $EV >> "$L" 2>&1
log train-mmfm; python3 -I train.py --events $EV --out runs/final_mmfm --ctx "${CTX:-64}" --batch "${BATCH:-64}" --lr "${LR:-1e-3}" --minutes "$MMFM_MIN" --eval-every 400 >> "$L" 2>&1
log train-b2; python3 -I train.py --events $EV --out runs/final_b2 --model b2 --minutes "$B2_MIN" --eval-every 400 --tokenizer runs/final_mmfm/tokenizer.json >> "$L" 2>&1
log m1-val; python3 -I evaluate.py --events $EV --runs runs/final_mmfm runs/final_b2 --split val --T "${CTX:-64}" --out results/m1_val.json >> "$L" 2>&1
H=$(python3 -c "import json;print(json.load(open('runs/final_mmfm/selection_log.json'))['model_hash'])")
log "freeze model $H (sha256 of runs/final_mmfm/model.pt); unsealing test ONCE"
export MMFM_UNSEAL=$H
log m1-test; MMFM_PURPOSE=M1 python3 -I evaluate.py --events $EV --runs runs/final_mmfm runs/final_b2 --split test --T "${CTX:-64}" --out results/m1_test.json >> "$L" 2>&1
log m1-heldout; MMFM_PURPOSE=M1 python3 -I evaluate.py --events $EV --runs runs/final_mmfm runs/final_b2 --split heldout_test --T "${CTX:-64}" --out results/m1_heldout_test.json >> "$L" 2>&1
log m2-test; MMFM_PURPOSE=M2 python3 -I sim_eval.py --events $EV --run runs/final_mmfm --split test --pairs ethusd,xrpusd,dogeusd --starts 15 --paths 8 --out results/m2_test.json >> "$L" 2>&1
log done
