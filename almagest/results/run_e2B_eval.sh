#!/bin/bash
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
for arm in short stateful modebank modebank_sb; do
  until [ -f e2B_$arm.pt ]; do sleep 20; done
  python ../experiments/e2_eval.py e2B_$arm.pt e2B_eval_$arm.json 16 8 > e2B_eval_$arm.log 2>&1
  echo "EVALDONE $arm"
  grep -E "^model|Traceback" e2B_eval_$arm.log | cut -c1-600
done
