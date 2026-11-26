#!/bin/bash
# Almagest synthetic-causal recipe, end to end, for ONE world (all steps reproducible, splits from splits.py):
#   0. certify the truth on DEV (pre-registered bands; abort if it fails)
#   1. base model: conditional heads + source-tagged interventions + own-order channel (TRAIN seeds)
#   2. DAgger v3 with the World oracle, tagged injections in closed loop (TRAIN seeds)
#   3. model soup between 1 and 2, alpha chosen on the TUNE split (training range), full objective
#   4. DEV evaluation of the chosen model (+ A8 held-out participations 5% and 20%)
#   5. (manual, once, after freeze/commit) TEST evaluation
# usage: run_recipe.sh <world.json> <tag>
set -e
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
W=$1; TAG=$2
export NUMBA_NUM_THREADS=2
echo "[0] certify $W on dev"
OMP_NUM_THREADS=1 WORLD=$W python ../experiments/e2_eval.py - cert_${TAG}.json dev 32 30 > cert_${TAG}.log 2>&1
python - "$TAG" <<'EOF'
import json, sys
t = json.load(open(f'cert_{sys.argv[1]}.json'))['truth']
ok = (0.35 <= t['delta_ci95'][0] and t['delta_ci95'][1] <= 0.65 and 0.4 <= t['Y'] <= 1.2 and
      0.45 <= t['ep100'] <= 0.85 and 0.42 <= t['H'] <= 0.58)
print('CERTIFIED' if ok else 'NOT CERTIFIED', {k: t[k] for k in ('delta', 'delta_ci95', 'Y', 'ep100', 'H')})
open(f'cert_{sys.argv[1]}.verdict', 'w').write('CERTIFIED' if ok else 'NOT CERTIFIED')
EOF
echo "[1] base"
COND_HEADS=1 TAG_CHANNEL=1 TORCH_THREADS=3 python ../experiments/e2_train.py $W modebank_sb_src r_${TAG}_base.pt 4 > r_${TAG}_base.log 2>&1
echo "[2] dagger"
BETA0=0 LR=3e-4 TORCH_THREADS=2 NUMBA_NUM_THREADS=1 python ../experiments/e2_dagger.py r_${TAG}_base.pt r_${TAG}_dag.pt 4 4000 64 > r_${TAG}_dag.log 2>&1
echo "[3] soup on tune"
INTERP_WITH=r_${TAG}_dag.pt python ../experiments/e2_post.py r_${TAG}_base.pt r_${TAG}_final.pt 0 0.25 0.5 0.75 1 > r_${TAG}_post.log 2>&1
tail -n 1 r_${TAG}_post.log
echo "[4] dev eval"
OMP_NUM_THREADS=1 python ../experiments/e2_eval.py r_${TAG}_final.pt r_${TAG}_dev.json dev 32 30 > r_${TAG}_dev.log 2>&1
OMP_NUM_THREADS=1 python ../experiments/e2_eval.py r_${TAG}_final.pt r_${TAG}_dev_eta05.json dev 32 30 0.05 > r_${TAG}_dev_eta05.log 2>&1
OMP_NUM_THREADS=1 python ../experiments/e2_eval.py r_${TAG}_final.pt r_${TAG}_dev_eta20.json dev 32 30 0.20 > r_${TAG}_dev_eta20.log 2>&1
echo "RECIPE DONE $TAG"
