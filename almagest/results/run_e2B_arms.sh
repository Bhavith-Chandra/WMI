#!/bin/bash
# train 4 lightweight arms on World A sequentially
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
for arm in short stateful modebank modebank_sb; do
  python ../experiments/e2_train.py world_B.json $arm e2B_$arm.pt 4 > e2B_train_$arm.log 2>&1
  echo "DONE $arm $(tail -1 e2B_train_$arm.log)"
done
