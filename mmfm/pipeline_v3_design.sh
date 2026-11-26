#!/usr/bin/env bash
# EVAL_v3 design phase (val only), strictly sequential so every model gets the CPU alone.
set -euo pipefail
cd "$(dirname "$0")"
EV=/home/user/data/mmfm_v3; L=runs/v3_design.log
log(){ echo "=== $* ($(date -u +%T))" | tee -a "$L"; }
until grep -q "model hash\|Traceback" runs/v3.log; do sleep 30; done
grep -q Traceback runs/v3.log && { log "v3 training failed"; exit 1; }
log b2; mkdir -p runs/v3_b2
python3 -I train_v2.py --events $EV --out runs/v3_b2 --model b2 --d 128 --ctx 64 --batch 64 --lr 1e-3 --minutes 60 --tokenizer runs/v3/tokenizer.json > runs/v3_b2.log 2>&1
log ss; mkdir -p runs/v3_ss
python3 -I train_v2.py --events $EV --out runs/v3_ss --d 128 --layers 4 --ctx 64 --batch 64 --lr 3e-4 --minutes 40 --eval-every 1000 \
  --tokenizer runs/v3/tokenizer.json --init runs/v3/model.pt --ss-p 0.25 > runs/v3_ss.log 2>&1
for R in v3 v3_ss; do
  log "m1 $R"; python3 -I evaluate_v2.py --events $EV --mmfm runs/$R --b2 runs/v3_b2 --split val --out results/v3_val_m1_$R.json >> "$L" 2>&1
  log "calib $R"; python3 -I calibrate_sim.py --run runs/$R --events $EV --temp 0.6,0.7,0.85 --starts 30 --recal >> "$L" 2>&1
  log "freeclock $R"; python3 -I sim_eval_v2.py --events $EV --run runs/$R --split val --clock model --gens MMFM-sim --starts 10 --paths 16 \
    --out results/v3_val_freeclock_$R.json >> "$L" 2>&1
done
log done
