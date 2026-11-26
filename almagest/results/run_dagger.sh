#!/bin/bash
# EXPLORATORY (World D, dev). DAgger with World oracle to fix closed-loop collapse; then dev eval incl. collapse rate.
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
export TORCH_THREADS=3 NUMBA_NUM_THREADS=2
python ../experiments/e2_dagger.py e2D_modebank_sb.pt e2Dg_modebank_sb.pt 3 3000 64 > e2Dg_train_modebank_sb.log 2>&1
echo "DAGGER modebank_sb done"; tail -3 e2Dg_train_modebank_sb.log
export OMP_NUM_THREADS=1
python ../experiments/e2_eval.py e2Dg_modebank_sb.pt e2Dg_dev_modebank_sb.json dev 32 30 > e2Dg_dev_modebank_sb.log 2>&1
echo "EVAL modebank_sb"; grep -E "Traceback|Error" e2Dg_dev_modebank_sb.log | head
echo ALLDONE
