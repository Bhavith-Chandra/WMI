"""Single source of truth for every seed range. Scripts import these; tests assert disjointness on these values.

History (do not reuse):
  5000+  dev: used to confirm/select World A and World B, and for a 1024-rollout dev E2 run.
  9000+  BURNED: truth on these days was observed before the audit fixes (2026-11-26). Never use again.
"""
CALIB_BASE = 1000        # World physics calibration
DEV_BASE = 5000          # confirmation / dev certification of truth
BURNED_BASE = 9000       # observed, retired
TEST_BASE = 13000        # final test: run once per frozen arm
TRAIN_BASE = 200000      # observational training streams
VAL_BASE = 300000        # validation streams
SPAN = 64                # max days used from any base
ROLL_OFFSET = 10_000_000 # rollout seed = ROLL_OFFSET + (day*1000 + window)*64 + replica  (window<1000, replica<64)


def roll_seed(day, window, replica=0):
    return ROLL_OFFSET + (day * 1000 + window) * 64 + replica
