#!/bin/bash
# Real-data DEV (select split) evaluation of unified_v1. Report split untouched.
cd /home/user/WMI/almagest/results
. ../../.venv/bin/activate
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=1
python ../experiments/e3_real.py b1 ../../data_tok real/split_v1.json real/unified_v1.pt real/unified_v1_b1_select.json select > real/b1_select.log 2>&1
echo "B1 $(tail -c 600 real/b1_select.log)"
python ../experiments/e3_forecast.py ../../data_tok real/split_v1.json real/unified_v1.pt real/unified_v1_b4_select.json select > real/b4_select.log 2>&1
echo "B4 $(cat real/b4_select.log | tail -3)"
python ../experiments/e3_realism.py ../../data_tok real/split_v1.json real/unified_v1.pt real/unified_v1_b25_select.json select 32 > real/b25_select.log 2>&1
echo "B25 $(tail -6 real/b25_select.log)"
echo ALLDONE
