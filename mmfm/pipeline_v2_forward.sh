#!/usr/bin/env bash
# EVAL_v2 forward test. Run AFTER the frozen bundle (model.pt, tokenizer.json, sim_calibration.json) is committed.
#   FREEZE_NS=<ns of the freeze commit> WINDOW_MIN=90 bash pipeline_v2_forward.sh
# Waits for the window to be fully recorded, builds it, unseals ONCE per metric, writes results/v2_forward_*.json.
set -euo pipefail
cd "$(dirname "$0")"
RUN=runs/v2; B2=runs/v2_b2; DES=/home/user/data/mmfm_v2; FW=/home/user/data/mmfm_fwd
START_NS=$((FREEZE_NS + 600*1000000000)); END_NS=$((START_NS + ${WINDOW_MIN:-90}*60*1000000000))
L=results/v2_forward.log; mkdir -p results; : > "$L"
log(){ echo "=== $* ($(date -u +%T))" | tee -a "$L"; }
log "window $(date -u -d @$((START_NS/1000000000)) +%T) - $(date -u -d @$((END_NS/1000000000)) +%T)"
while [ "$(date +%s)" -lt $((END_NS/1000000000 + 60)) ]; do sleep 30; done
log build; python3 -I data/build.py --out $FW --until-ns $END_NS --from-ns $START_NS >> "$L" 2>&1
log splits; python3 -I data/splits_forward.py $FW $((START_NS/1000)) $((END_NS/1000)) >> "$L" 2>&1
export MMFM_UNSEAL=$(sha256sum $RUN/model.pt | cut -d' ' -f1)
CAL=$(python3 -c "import json;c=json.load(open('$RUN/sim_calibration.json'))['chosen'];print(c['dt_scale'],c['temp'])")
DT=$(echo $CAL | cut -d' ' -f1); TP=$(echo $CAL | cut -d' ' -f2)
log m1; python3 -I evaluate_v2.py --events $FW --train-events $DES --mmfm $RUN --b2 $B2 --forward --out results/v2_forward_m1.json >> "$L" 2>&1
log m1-heldout; python3 -I evaluate_v2.py --events $FW --train-events $DES --mmfm $RUN --b2 $B2 --forward --heldout --out results/v2_forward_m1_heldout.json >> "$L" 2>&1
log sim; python3 -I sim_eval_v2.py --events $FW --train-events $DES --run $RUN --forward --pairs ethusd,xrpusd,dogeusd --starts 25 --paths 16 --dt-scale $DT --temp $TP --out results/v2_forward_sim.json >> "$L" 2>&1
log sim-heldout; python3 -I sim_eval_v2.py --events $FW --train-events $DES --run $RUN --forward --heldout --pairs linkusd --starts 25 --paths 16 --dt-scale $DT --temp $TP --out results/v2_forward_sim_heldout.json >> "$L" 2>&1
log done
