#!/bin/bash
# train 4 lightweight arms on World A sequentially
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
for arm in short stateful modebank modebank_sb; do
  python ../experiments/e2_train.py world_A.json $arm e2A_$arm.pt 6 > e2A_train_$arm.log 2>&1
  echo "DONE $arm $(tail -1 e2A_train_$arm.log)"
done
