#!/bin/bash
# EXPLORATORY (World D, dev). DAgger v2: beta-mixture (0.5 halving), stored h0, lr 5e-4, 5 rounds.
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
export TORCH_THREADS=2 NUMBA_NUM_THREADS=1 BETA0=0.5 LR=5e-4
python ../experiments/e2_dagger.py e2D_modebank_sb.pt e2Dg2_modebank_sb.pt 5 3000 64 > e2Dg2_train_modebank_sb.log 2>&1
echo "DAGGER2 done"; tail -3 e2Dg2_train_modebank_sb.log
while pgrep -f "e2Dg_dev_modebank_sb" >/dev/null; do sleep 30; done
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2
python ../experiments/e2_eval.py e2Dg2_modebank_sb.pt e2Dg2_dev_modebank_sb.json dev 32 30 > e2Dg2_dev_modebank_sb.log 2>&1
echo "EVAL2 modebank_sb"; grep -E "Traceback|Error" e2Dg2_dev_modebank_sb.log | head
echo ALLDONE2
