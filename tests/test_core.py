import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.data import collect, split_episodes
from wmi.env import KeyGrid, N_ACTIONS
from wmi.harness import LeakageError, assert_disjoint, check_split, run_record, set_seed
from wmi.interventions import logistic_direction, set_logit
from wmi.models import RecurrentWorldModel
from wmi.probing import fit_probe, probe_with_controls
from wmi.train import train_recurrent


def test_walls_block_and_key_pickup():
    env = KeyGrid()
    s = np.array([[2, 0, 0]])
    assert (env.step(s, np.array([3])) == s).all()          # wall at x=3,y=0
    s = np.array([[2, 3, 0]])
    assert env.step(s, np.array([3]))[0, 0] == 3            # gap at y=3
    s = np.array([[5, 2, 0]])
    assert env.step(s, np.array([0]))[0, 2] == 1            # step onto key (5,1)
    s = np.array([[0, 0, 0]])
    assert (env.step(s, np.array([0])) == s).all()          # boundary


def test_local_view_hides_key_after_pickup():
    env = KeyGrid()
    assert env.render_local(np.array([[5, 2, 0]]))[0, 9:].sum() == 1
    assert env.render_local(np.array([[5, 2, 1]]))[0, 9:].sum() == 0


def test_pomdp_full_view_only_at_t0():
    env = KeyGrid()
    s = np.array([[1, 1, 0], [1, 1, 0]])
    o = env.render_pomdp(s, np.array([0, 1]))
    assert o[0, :env.full_dim].sum() > 0 and o[1, :env.full_dim].sum() == 0


def test_shortest_path():
    env = KeyGrid()
    assert env.shortest_path_len((0, 0), (0, 0)) == 0
    assert env.shortest_path_len((2, 0), (4, 0)) == 3 + 3 + 2   # via gap at y=3


def test_split_is_disjoint_and_complete():
    env = KeyGrid()
    tr = collect(env, 100, 5, np.random.default_rng(0))
    sp = split_episodes(tr, rng=np.random.default_rng(1))
    assert sum(len(v) for v in sp.values()) == 100
    for a in sp:
        for b in sp:
            if a != b:
                assert not set(sp[a]) & set(sp[b])


def test_leakage_raises():
    with pytest.raises(LeakageError):
        assert_disjoint([1, 2, 3], [3, 4])
    with pytest.raises(LeakageError):
        check_split({"train": np.array([0, 1]), "test": np.array([2])}, n_total=4)


def test_probe_controls_behave():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.standard_normal((n, 10))
    y = (X[:, 0] > 0).astype(int)
    types = np.arange(n)[:, None]
    idx = {"train": slice(0, 2000), "val": slice(2000, 2500), "test": slice(2500, n)}
    r = probe_with_controls({k: X[v] for k, v in idx.items()}, {k: y[v] for k, v in idx.items()},
                            {k: types[v] for k, v in idx.items()}, np.arange(500), "classification", rng, n_boot=50)
    assert r["real"]["test"] > 0.95
    assert abs(r["shuffled"]["test"] - 0.5) < 0.1
    assert abs(r["control_task"]["test"] - 0.5) < 0.1      # unique types -> control unlearnable
    assert r["selectivity"] > 0.35


def test_set_logit_hits_target():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 5)) * 3 + 1
    y = (X @ np.array([1, -2, 0, 0, 0.5]) > 0).astype(int)
    r = fit_probe(X, y, X, y, X, "classification")
    w, b = logistic_direction(r["model"], r["scaler"])
    raw = r["model"].decision_function(r["scaler"].transform(X))
    assert np.allclose(X @ w + b, raw, atol=1e-6)
    h = set_logit(X, w, b, 4.0)
    assert np.allclose(h @ w + b, 4.0, atol=1e-6)


def test_training_is_deterministic():
    env = KeyGrid()

    def run():
        rng = set_seed(0)
        tr = collect(env, 40, 6, rng)
        m = RecurrentWorldModel(env.pomdp_dim, N_ACTIONS, env.local_dim, latent=8, hidden=16)
        train_recurrent(m, env, tr, tr, 10, rng, batch=8, eval_every=5)
        return torch.cat([p.flatten() for p in m.parameters()])

    assert torch.equal(run(), run())


def test_provenance_fields():
    r = run_record({"seed": 1})
    for k in ("config_hash", "git_commit", "git_dirty", "timestamp_utc", "torch"):
        assert k in r
