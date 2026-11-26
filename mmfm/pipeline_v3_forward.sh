#!/usr/bin/env bash
# EVAL_v3 forward test. Run AFTER FROZEN_v3.json is committed.
#   FREEZE_NS=<ns> RUN=runs/<chosen> WINDOW_MIN=90 bash pipeline_v3_forward.sh
set -euo pipefail
cd "$(dirname "$0")"
B2=runs/v3_b2; DES=/home/user/data/mmfm_v3; FW=/home/user/data/mmfm_fwd3
START_NS=$((FREEZE_NS + 600*1000000000)); END_NS=$((START_NS + ${WINDOW_MIN:-90}*60*1000000000))
L=results/v3_forward.log; mkdir -p results; : > "$L"
log(){ echo "=== $* ($(date -u +%T))" | tee -a "$L"; }
log "window $(date -u -d @$((START_NS/1000000000)) +%T) - $(date -u -d @$((END_NS/1000000000)) +%T) run $RUN"
while [ "$(date +%s)" -lt $((END_NS/1000000000 + 60)) ]; do sleep 30; done
log build; python3 -I data/build.py --out $FW --until-ns $END_NS --from-ns $START_NS --chunk-s 3600 >> "$L" 2>&1
log splits; python3 -I data/splits_forward.py $FW $((START_NS/1000)) $((END_NS/1000)) >> "$L" 2>&1
export MMFM_UNSEAL=$(sha256sum $RUN/model.pt | cut -d' ' -f1)
TP=$(python3 -c "import json;print(json.load(open('$RUN/sim_calibration.json'))['chosen']['temp'])")
log m1; python3 -I evaluate_v2.py --events $FW --train-events $DES --mmfm $RUN --b2 $B2 --forward --out results/v3_forward_m1.json >> "$L" 2>&1
log m1-heldout; python3 -I evaluate_v2.py --events $FW --train-events $DES --mmfm $RUN --b2 $B2 --forward --heldout --out results/v3_forward_m1_heldout.json >> "$L" 2>&1
log sim-recal; python3 -I sim_eval_v2.py --events $FW --train-events $DES --run $RUN --forward --pairs ethusd,xrpusd,dogeusd --starts 25 --paths 16 --temp $TP --clock anchored --recal --out results/v3_forward_sim.json >> "$L" 2>&1
log sim-heldout-recal; python3 -I sim_eval_v2.py --events $FW --train-events $DES --run $RUN --forward --heldout --pairs linkusd --starts 25 --paths 16 --temp $TP --clock anchored --recal --out results/v3_forward_sim_heldout.json >> "$L" 2>&1
log "sim-norecal (info)"; python3 -I sim_eval_v2.py --events $FW --train-events $DES --run $RUN --forward --pairs ethusd,xrpusd,dogeusd --starts 25 --paths 16 --temp $TP --clock anchored --gens MMFM-sim,MMFM-direct --out results/v3_forward_sim_norecal.json >> "$L" 2>&1
log "v2 bundle on v3 window (info)"
export MMFM_UNSEAL=$(sha256sum runs/v2/model.pt | cut -d' ' -f1)
python3 -I sim_eval_v2.py --events $FW --train-events /home/user/data/mmfm_v2 --run runs/v2 --forward --pairs ethusd,xrpusd,dogeusd --starts 25 --paths 16 --temp 0.7 --clock anchored --gens MMFM-sim,MMFM-direct --out results/v3_forward_sim_v2bundle.json >> "$L" 2>&1
log done
