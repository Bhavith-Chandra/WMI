#!/bin/bash
# DAgger v3 on the best design (cond heads + src + own-order channel), World D, dev only.
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
BETA0=0 LR=3e-4 TORCH_THREADS=2 NUMBA_NUM_THREADS=1 python ../experiments/e2_dagger.py e2Dt_modebank_sb_src.pt e2Dg3_modebank_sb_src.pt 4 4000 64 > e2Dg3_train.log 2>&1
echo "DAGGER3 $(tail -n 2 e2Dg3_train.log | tr '\n' ' ')"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2
python ../experiments/e2_eval.py e2Dg3_modebank_sb_src.pt e2Dg3_dev_modebank_sb_src.json dev 32 30 > e2Dg3_dev.log 2>&1
echo "EVAL done"; grep -E "Traceback|Error" e2Dg3_dev.log | head -3; echo ALLDONE
