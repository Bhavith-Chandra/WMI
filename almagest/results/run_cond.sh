#!/bin/bash
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
COND_HEADS=1 TORCH_THREADS=4 python ../experiments/e2_train.py world_D.json modebank_sb e2Dc_modebank_sb.pt 4 > e2Dc_train.log 2>&1
echo "TRAINED $(tail -1 e2Dc_train.log)"
OMP_NUM_THREADS=2 python ../experiments/e2_persist.py e2Dc_modebank_sb.pt
OMP_NUM_THREADS=1 python ../experiments/e2_collapse.py e2Dc_modebank_sb.pt 12 5000
echo ALLDONE
