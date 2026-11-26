#!/bin/bash
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
export NUMBA_NUM_THREADS=2
INTERP_WITH=e2Dg3_modebank_sb_src.pt python ../experiments/e2_post.py e2Dt_modebank_sb_src.pt e2Ds_modebank_sb_src.pt 0 0.25 0.5 0.75 1 > e2Ds_post.log 2>&1
echo "POST $(tail -n 1 e2Ds_post.log)"
OMP_NUM_THREADS=1 python ../experiments/e2_eval.py e2Ds_modebank_sb_src.pt e2Ds_dev_modebank_sb_src.json dev 32 30 > e2Ds_dev.log 2>&1
echo "EVAL done"; grep -E "Traceback|Error" e2Ds_dev.log | head -3; echo ALLDONE
