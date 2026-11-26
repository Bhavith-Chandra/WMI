# Evaluation and verification protocol

This repository makes empirical claims about planning with world models. Every claim in a paper must pass through
the pipeline below. Each rule names the file that enforces it.

## 1. Order of operations (pre-registration)

1. Write the hypothesis, the planners, the seeds, the sample size and the decision rule in
   `papers/<paper>/vN_preregistration.md`.
2. Write the analysis script `scripts/analyze_vN.py` that applies exactly those rules.
3. Commit both. **Only then** run anything that produces a result for a method under test.
4. Record the outcome, including every rule that failed, at the bottom of the same file.

Deviations made after a run starts (crashes, time limits, bug fixes) are appended to the file's execution log with
the reason. If a bug is found before a run, the fix and the test that would have caught it are logged as a
pre-run correction (example: v8, mass perturbation).

## 2. Gates (is the benchmark measuring what we think?)

Before the main run of a new task, an **oracle gate** compares only the forward planner with the oracle that
knows the latent. If the oracle does not beat forward planning by a pre-declared margin, the task does not test
hindsight and is not run. Gates may choose task parameters from a pre-declared ladder, using forward and oracle
results only and never a method under test (`scripts/run_hier.py gate`, `scripts/run_mujoco_v3.py gate`). Two gates
failed in this project; both failures are reported.

## 3. Data separation and leakage

| Stream | Seeds / source | Used for |
|---|---|---|
| World-model training data | `collect(rng)` with seeds 0, 1, 2; random actions, no task, no goals | training only |
| Model validation | 10% of training *episodes* (not transitions) | early stopping |
| Test schedules (latents, goals) | 10000 + seed | evaluation |
| Planner randomness | 20000 + seed | evaluation |
| Gates | 99, 99999 | task calibration (forward and oracle only) |
| Sealed deployment test (v8) | 100000/200000/300000 + {100, 101, 102} | the final test, run once |
| Real measured systems | official train/test splits of nonlinearbenchmark.org | `assert_no_test_in_train` |

Enforced by tests:
- `tests/test_planner_integrity.py::test_rng_streams_disjoint`: all seed ranges are pairwise disjoint.
- `test_no_privileged_latent`: no non-oracle planner's action changes when the true latent is replaced by garbage.
  `test_detector_catches_oracle` shows the test does detect a planner that reads it.
- `test_causality`, `test_causality_with_realism`: two worlds identical until step t* give identical trajectories
  until t* (no future information), including with sensor noise, actuation delay and compute delay; and different
  trajectories after t* (the test is not vacuous).
- `test_realism_changes_outcome`: every deployment-realism option actually changes the simulation.
- Real data: `wmi/realdata.assert_no_test_in_train`; strictly causal input windows (`wmi/realdata.windows`).
- Learned estimators chosen by the autoresearch loop are judged once on sealed test data
  (`scripts/run_sealed_v3.py`); the one that won on development data failed there, and that is what is reported.

## 4. Verification of the simulator and of ports

- `tests/test_physics.py`: the MuJoCo puck matches the closed-form dynamics (Euler with implicit damping) to
  1e-9, with perturbed mass and damping; a mass change must have an effect (it silently did not before
  `mj_setConst`).
- `demo/verify.mjs`: the browser implementation matches Python and MuJoCo: network outputs (2e-7), physics
  (7e-16), belief update (2e-16), planner rollout cost for every planner kind (1e-7 relative).
- Refactors that must not change results are checked for bit-identical output against the previous commit before
  running (example: realism options in `wmi/hier.run_hier`).

## 5. Statistics

- Paired comparisons on the same episodes (same latent draws, same goals) for every planner.
- 95% paired bootstrap intervals over episodes, 10,000 resamples; success rates compared as paired differences.
- Effects are reported as ratios to forward planning (costs) or differences in success rate, always with the
  oracle's value beside them so the headroom is visible.
- No result is dropped. Controls designed to fail (fresh latents) are reported beside the wins.

## 6. Provenance

`wmi/harness/provenance.run_record` is captured **before** a run: git commit, dirty-tree flag, config and its hash.
`save_result` writes `results/<experiment>/<hash>_seed<k>.json`. Paper numbers are macros generated from these files
only (`scripts/make_paper1.py` -> `generated/numbers.tex`); a number typed by hand is a bug. The generator prints
whether any source was run from a dirty tree.

## 7. Realistic benchmarking (v8)

Idealised simulation overstates every planner. The deployment benchmark adds, without tuning:
unmodelled physics (mass and damping the model never saw), sensor noise, one step of actuation delay, and **measured
compute delay**: each planner's single-thread p95 decision time on this machine is converted into control steps
during which the robot keeps executing its previous plan. A planner that thinks longer pays for it in the
simulation. The decision rules require the gain to survive these conditions (R6: at least half of the idealised gain).

## 8. Real time

`scripts/measure_latency.py` measures decision and belief-update latency for one robot on an idle machine and
records the hardware. The browser demo (`demo/`) runs the same planners in background threads against a 20 Hz
control clock, so compute latency acts in real time and is shown live.
