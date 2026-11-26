#!/bin/bash
# scale test: 2x data (32 streams) x 2x passes (8) for the src + own-order-channel design
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
NTR=32 COND_HEADS=1 TAG_CHANNEL=1 TORCH_THREADS=4 python ../experiments/e2_train.py world_D.json modebank_sb_src e2Dt2_modebank_sb_src.pt 8 > e2Dt2_train.log 2>&1
echo "TRAINED $(tail -n 1 e2Dt2_train.log)"
OMP_NUM_THREADS=1 python ../experiments/e2_collapse.py e2Dt2_modebank_sb_src.pt 12 8000
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=3
python ../experiments/e2_eval.py e2Dt2_modebank_sb_src.pt e2Dt2_dev_modebank_sb_src.json dev 32 30 > e2Dt2_dev.log 2>&1
echo "EVAL done"; grep -E "Traceback|Error" e2Dt2_dev.log | head -3; echo ALLDONE
