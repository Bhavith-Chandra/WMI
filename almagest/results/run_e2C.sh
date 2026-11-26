#!/bin/bash
# EXPLORATORY: World C failed pre-registered certification (delta CI above band). Dev split only; no test claims.
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
for arm in short stateful modebank modebank_sb; do
  python ../experiments/e2_train.py world_C.json $arm e2C_$arm.pt 4 > e2C_train_$arm.log 2>&1
  echo "TRAINED $arm $(tail -1 e2C_train_$arm.log)"
done
for arm in short modebank_sb stateful modebank; do
  python ../experiments/e2_eval.py e2C_$arm.pt e2C_dev_$arm.json dev 32 30 > e2C_dev_$arm.log 2>&1
  echo "EVAL $arm"; grep -E "^model|Traceback|Error" e2C_dev_$arm.log | cut -c1-400
done
