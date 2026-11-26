#!/bin/bash
# EXPLORATORY (World D failed certification narrowly). Dev split only.
# 2x2: {short, modebank_sb} x {observational, interventional source-tagged}.
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
while pgrep -f run_e2C_eval.sh >/dev/null; do sleep 30; done
for arm in short_src modebank_sb_src short modebank_sb; do
  python ../experiments/e2_train.py world_D.json $arm e2D_$arm.pt 4 > e2D_train_$arm.log 2>&1
  echo "TRAINED $arm $(tail -1 e2D_train_$arm.log)"
done
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2
for arm in modebank_sb_src short_src modebank_sb short; do
  python ../experiments/e2_eval.py e2D_$arm.pt e2D_dev_$arm.json dev 32 30 > e2D_dev_$arm.log 2>&1
  echo "EVAL $arm"; grep -E "Traceback|Error" e2D_dev_$arm.log | head -3
done
echo ALLDONE
