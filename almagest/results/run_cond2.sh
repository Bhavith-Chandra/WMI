#!/bin/bash
# EXPLORATORY (World D, dev). Conditional heads: interpretable impact for the first time (collapse fixed).
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
for arm in short modebank_sb_src; do
  COND_HEADS=1 TORCH_THREADS=3 python ../experiments/e2_train.py world_D.json $arm e2Dc_$arm.pt 4 > e2Dc_train_$arm.log 2>&1 &
done
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python ../experiments/e2_eval.py e2Dc_modebank_sb.pt e2Dc_dev_modebank_sb.json dev 32 30 > e2Dc_dev_modebank_sb.log 2>&1
echo "EVAL modebank_sb"
wait
for arm in short modebank_sb_src; do
  python ../experiments/e2_eval.py e2Dc_$arm.pt e2Dc_dev_$arm.json dev 32 30 > e2Dc_dev_$arm.log 2>&1 &
done
wait
echo "EVAL short modebank_sb_src"; echo ALLDONE
