#!/bin/bash
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
COND_HEADS=1 TAG_CHANNEL=1 TORCH_THREADS=4 python ../experiments/e2_train.py world_D.json modebank_sb_src e2Dt_modebank_sb_src.pt 4 > e2Dt_train.log 2>&1
echo "TRAINED $(tail -n 1 e2Dt_train.log)"
OMP_NUM_THREADS=2 python ../experiments/e2_tagresp.py e2Dt_modebank_sb_src.pt
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=3
python ../experiments/e2_eval.py e2Dt_modebank_sb_src.pt e2Dt_dev_modebank_sb_src.json dev 32 30 > e2Dt_dev.log 2>&1
echo "EVAL done"; grep -E "Traceback|Error" e2Dt_dev.log | head -3; echo ALLDONE
