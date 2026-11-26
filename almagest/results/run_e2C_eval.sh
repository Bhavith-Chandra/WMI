#!/bin/bash
# EXPLORATORY: World C failed pre-registered certification. Dev split only; no test claims.
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2
for arm in short modebank_sb stateful modebank; do
  rm -f e2C_dev_$arm.log
  python ../experiments/e2_eval.py e2C_$arm.pt e2C_dev_$arm.json dev 32 30 > e2C_dev_$arm.log 2>&1
  echo "EVAL $arm"; grep -E "^model|Traceback|Error" e2C_dev_$arm.log | cut -c1-500
done
