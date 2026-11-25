# autoresearch: hindsight estimators

Adapted from karpathy/autoresearch (program.md / immutable prepare / editable train, fixed budget, keep-or-discard).

## Goal
Find the hindsight estimator (`estimator.py`) that minimises `SCORE` = geometric mean over DEV tasks of
(loss with hindsight) / (loss of the forward planner that ignores outcomes). 1.0 = no gain over forward.

## Rules
1. Edit ONLY `autoresearch/estimator.py`. Never edit `eval.py`, `wmi/`, data, or seeds.
2. Before each run, write the hypothesis and the predicted direction in `hypotheses.md` (pre-registration).
3. Run `python autoresearch/eval.py "<description>"`; it appends a row to `results.tsv` (append-only, never edit).
4. (rule v2) KEEP the change (git commit) only if WORST <= 1.05 AND SCORE improves by > 0.5% relative to the
   best kept v2-admissible score; otherwise
   `git checkout autoresearch/estimator.py`. Record the verdict in `hypotheses.md` either way.
5. The DEV suite never touches official TEST splits. Final numbers for papers come from a single sealed run
   of the best kept estimator on TEST splits (scripts/), never from this loop.
6. No hand-typed numbers anywhere: results.tsv and results/*.json are the only sources.
7. Per-task ratios are logged so a gain on one regime that costs another is visible; "fresh" cannot go
   below ~1.0 (Theorem 2) -- a value well below 1 there means a bug, stop and investigate.
