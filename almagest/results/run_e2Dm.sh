#!/bin/bash
# EXPLORATORY: mirror (side-flip) augmentation to remove systematic free-run drift. World D, dev only.
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
while pgrep -f run_e2D.sh >/dev/null; do sleep 30; done
for arm in modebank_sb modebank_sb_src; do
  MIRROR=1 python ../experiments/e2_train.py world_D.json $arm e2Dm_$arm.pt 4 > e2Dm_train_$arm.log 2>&1
  echo "TRAINED $arm $(tail -1 e2Dm_train_$arm.log)"
done
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2
for arm in modebank_sb_src modebank_sb; do
  python ../experiments/e2_eval.py e2Dm_$arm.pt e2Dm_dev_$arm.json dev 32 30 > e2Dm_dev_$arm.log 2>&1
  echo "EVAL $arm"; grep -E "Traceback|Error" e2Dm_dev_$arm.log | head -3
done
echo ALLDONE
