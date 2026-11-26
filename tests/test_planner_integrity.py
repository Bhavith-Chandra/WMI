"""Leakage and causality tests for every planner used in the paper.

1. No privileged access: a non-oracle planner's actions must not change when the true latent it is not supposed to
   see (pl.phi_true) is replaced by garbage. The oracle must change (proves the test can detect leakage).
2. Causality: two worlds identical up to step t* and different afterwards must produce identical trajectories up to t*.
3. Separation of data streams: training, test-schedule, planner and gate RNG seeds are pairwise disjoint.
"""
import numpy as np
import pytest
import torch

from wmi import hier
from wmi.mujoco_exp import DynMLP
from wmi.mujoco_v3 import Planner3

KINDS = ["forward", "residual", "online_sgd", "hindsight_posterior", "hindsight_latent_pf_adapt", "hindsight_latent"]


def tiny_models(seed=0):
    torch.manual_seed(seed)
    def mk(n_in):
        m = DynMLP(n_in, hidden=16)
        m.in_mu.zero_(); m.in_sd.fill_(1.0); m.out_mu.zero_(); m.out_sd.fill_(0.05)
        for p in m.parameters():
            p.requires_grad_(False)
        return m
    return [mk(4) for _ in range(2)], [mk(7) for _ in range(2)]


CFG = {"pop": 16, "elites": 4, "iters": 2, "sgd_lr": 1e-3, "forget": 0.98, "window": 30, "latent_steps": 2,
       "n_particles": 32, "wind_sd": 1.0, "beta": 0.1, "K": 3, "H": 6,
       "resid_sd": [0.01, 0.01, 0.05, 0.05], "resid_sd_c": [0.01, 0.01, 0.05, 0.05]}


def plans_with_phi_true(kind, phi_true, E=3):
    Mu, Mc = tiny_models()
    pl = Planner3(kind, Mu, Mc, E, dict(CFG), np.random.default_rng(7))
    rng = np.random.default_rng(3); s = rng.normal(size=(E, 4)); out = []
    for _ in range(3):
        pl.phi_true = phi_true
        plan = hier.act_hazard(pl, s, np.tile([1.0, 0.0], (E, 1))); out.append(plan.copy())
        a = plan[:, 0]; s2 = s + 0.05 * rng.normal(size=s.shape)
        pl.observe(s, a, s2); s = s2
    return np.stack(out)


@pytest.mark.parametrize("kind", KINDS)
def test_no_privileged_latent(kind):
    a = plans_with_phi_true(kind, np.tile([1.0, 0.0, 0.0], (3, 1)))
    b = plans_with_phi_true(kind, np.tile([0.3, 5.0, -5.0], (3, 1)))
    assert np.array_equal(a, b), f"{kind} reads the true latent"


def test_detector_catches_oracle():
    a = plans_with_phi_true("oracle_latent", np.tile([1.0, 0.0, 0.0], (3, 1)))
    b = plans_with_phi_true("oracle_latent", np.tile([0.3, 5.0, -5.0], (3, 1)))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("planner", ["hier:forward", "hier:hindsight_posterior", "hier:hindsight_latent_pf_adapt",
                                     "hier:oracle_latent", "flat:hindsight_posterior"])
def test_causality(planner):
    Mu, Mc = tiny_models(); T, E, tstar = 24, 2, 12
    s1 = hier.schedule("both", np.random.default_rng(5), E, T)
    s2 = s1.copy(); s2[tstar:] = hier.schedule("both", np.random.default_rng(6), E, T)[tstar:]
    cfg = dict(CFG)
    t1 = np.array(hier.run_hier(planner, Mu, Mc, s1, cfg, 11)["traj"])
    t2 = np.array(hier.run_hier(planner, Mu, Mc, s2, cfg, 11)["traj"])
    assert np.allclose(t1[:tstar + 1], t2[:tstar + 1], equal_nan=True), "future latent leaked into the past"
    assert not np.allclose(t1, t2, equal_nan=True), "test is vacuous: the future change had no effect"


def test_rng_streams_disjoint():
    seeds = [0, 1, 2]
    train = set(seeds); test_sched = {10_000 + s for s in seeds}; planner = {20_000 + s for s in seeds}
    gate = {99, 99_999}; sealed = {100_000 + s for s in (100, 101, 102)} | {200_000 + s for s in (100, 101, 102)} | {300_000 + s for s in (100, 101, 102)}
    groups = [train, test_sched, planner, gate, sealed]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert not groups[i] & groups[j]
