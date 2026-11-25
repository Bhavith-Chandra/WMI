"""Probe a recurrent world model for ground-truth state, with all controls, then intervene.

Usage: python scripts/run_probes.py --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.data import collect, split_episodes  # noqa: E402
from wmi.env import N_ACTIONS, KeyGrid  # noqa: E402
from wmi.harness import observation_overlap, run_record, save_result, set_seed  # noqa: E402
from wmi.interventions import logistic_direction  # noqa: E402
from wmi.models import RecurrentWorldModel  # noqa: E402
from wmi.probing import probe_with_controls  # noqa: E402
from wmi.train import recurrent_loss, train_recurrent  # noqa: E402

TARGETS = {"x": 0, "y": 1, "has_key": 2}


@torch.no_grad()
def activations(model, traj):
    _, cache = model(torch.from_numpy(traj.obs_pomdp), torch.from_numpy(traj.actions))
    out = {k: v[:, 1:].reshape(-1, v.shape[-1]).numpy() for k, v in cache.items()}  # t >= 1 only
    out["obs"] = traj.obs_pomdp[:, 1:].reshape(-1, traj.obs_pomdp.shape[-1])
    return out


def per_step(traj, arr):
    return arr[:, 1:].reshape(-1, *arr.shape[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--probe_train_n", type=int, default=15000, help="random subsample of train timesteps for probes")
    ap.add_argument("--probe_val_n", type=int, default=5000)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--exp", default="probes", help="results subdir; use _smoke for throwaway runs")
    args = ap.parse_args()
    cfg = {"experiment": "probes", **vars(args), "env": {"size": 7, "wall_x": 3, "gap_y": 3, "key_pos": [5, 1]}}

    rec = run_record(cfg)  # captured BEFORE running, so the commit/dirty flag describe the code that ran
    rng = set_seed(args.seed)
    env = KeyGrid()
    traj = collect(env, args.episodes, args.horizon, rng)
    split = split_episodes(traj, rng=rng)
    S = {k: traj.subset(v) for k, v in split.items()}

    leak = {"episode_split_sizes": {k: int(len(v)) for k, v in split.items()},
            "episode_ids_disjoint": True,  # asserted in split_episodes; would have raised
            "test_obs_seen_in_train_frac_t>=1": observation_overlap(
                per_step(S["train"], S["train"].obs_pomdp), per_step(S["test"], S["test"].obs_pomdp))}

    torch.manual_seed(args.seed + 10_000)
    untrained = RecurrentWorldModel(env.pomdp_dim, N_ACTIONS, env.local_dim, latent=args.latent).eval()
    torch.manual_seed(args.seed)
    model = RecurrentWorldModel(env.pomdp_dim, N_ACTIONS, env.local_dim, latent=args.latent)
    hist = train_recurrent(model, env, S["train"], S["val"], args.steps, rng)
    model.eval()
    wm_metrics = {"val_bce_best": hist["best_val_bce"], "test_bce": recurrent_loss(model, env, S["test"]),
                  "test_bce_untrained": recurrent_loss(untrained, env, S["test"])}

    acts = {"trained": {k: activations(model, S[k]) for k in S},
            "untrained": {k: activations(untrained, S[k]) for k in S}}
    states = {k: per_step(S[k], S[k].states) for k in S}
    groups_test = np.repeat(S["test"].episode_ids, args.horizon)
    sub = {"train": np.sort(rng.choice(len(states["train"]), min(args.probe_train_n, len(states["train"])), replace=False)),
           "val": np.sort(rng.choice(len(states["val"]), min(args.probe_val_n, len(states["val"])), replace=False)),
           "test": np.arange(len(states["test"]))}  # test is never subsampled

    probes, fits = {}, {}
    for mname, sites in (("trained", ("obs", "enc", "h")), ("untrained", ("enc", "h"))):
        for site in sites:
            for tname, col in TARGETS.items():
                X = {k: acts[mname][k][site][sub[k]] for k in S}
                y = {k: states[k][sub[k], col] for k in S}
                types = {k: states[k][sub[k]] for k in S}
                r = probe_with_controls(X, y, types, groups_test, "classification", rng, n_boot=args.boot)
                fits[(mname, site, tname)] = r.pop("_real_fit")
                probes[f"{mname}/{site}/{tname}"] = r
                print(f"{mname:9s} {site:4s} {tname:8s} real={r['real']['test']:.3f} "
                      f"ctrl={r['control_task']['test']:.3f} shuf={r['shuffled']['test']:.3f} "
                      f"maj={r['majority']['test']:.3f}", flush=True)

    # ---- causal intervention: has_key direction in trained h -> does predicted key visibility follow?
    fit = fits[("trained", "h", "has_key")]
    w, b = logistic_direction(fit["model"], fit["scaler"])
    h_tr = acts["trained"]["train"]["h"]
    hk_tr = states["train"][:, 2]

    te = S["test"]
    h_te = torch.from_numpy(acts["trained"]["test"]["h"])
    # h_t for t>=1 pairs with a_t for t=1..T-1; drop last step (no next obs)
    E, T = te.actions.shape
    mask_t = np.tile(np.arange(1, T + 1) < T, E)
    h_al = h_te.numpy()[mask_t]
    a_al = te.actions[:, 1:].reshape(-1)
    s_next = te.states[:, 2:].reshape(-1, 3)
    s_now = te.states[:, 1:-1].reshape(-1, 3)
    key_slot = slice(9, 18)

    def key_prob(h, a):
        with torch.no_grad():
            return torch.sigmoid(model.predict_from_h(torch.from_numpy(h).float(), torch.from_numpy(a))).numpy()

    # Directions (unit norm, raw h space). Edit size is in units of D = ||mu1 - mu0||, the actual
    # between-class distance, swept as a dose-response curve. Random control: per-row random unit
    # direction at the same norm.
    mu1, mu0 = h_tr[hk_tr == 1].mean(0), h_tr[hk_tr == 0].mean(0)
    D = float(np.linalg.norm(mu1 - mu0))
    dirs = {"probe": w / np.linalg.norm(w), "diff_means": (mu1 - mu0) / D}
    cos_probe_dm = float(dirs["probe"] @ dirs["diff_means"])

    interventions = {"D_between_class": D, "cos_probe_vs_diff_means": cos_probe_dm,
                     "alphas": list(args.alphas), "conditions": {}}
    # (A) agent lacks key, key visible next step -> push toward has_key=1: key should vanish
    vis = env.render_local(s_next)[:, key_slot]
    selA = (s_now[:, 2] == 0) & (s_next[:, 2] == 0) & (vis.sum(1) == 1)
    # (B) agent HAS key; cell where the key would be visible -> push toward has_key=0: key should appear
    s_cf = s_next.copy(); s_cf[:, 2] = 0
    vis_cf = env.render_local(s_cf)[:, key_slot]
    selB = (s_now[:, 2] == 1) & (vis_cf.sum(1) == 1)
    for name, sel, sign, vmask in (("remove_key_belief", selA, +1.0, vis), ("insert_key_belief", selB, -1.0, vis_cf)):
        h0, a0, m = h_al[sel], a_al[sel], vmask[sel].astype(bool)
        blocked_true = env.render_local(s_next[sel])[:, :9] > 0.5
        res = {"n": int(sel.sum())}
        for dname in ("probe", "diff_means", "random"):
            curve = []
            for alpha in args.alphas:
                if dname == "random":
                    r = rng.standard_normal(h0.shape)
                    r /= np.linalg.norm(r, axis=1, keepdims=True)
                    hh = h0 + alpha * D * r
                else:
                    hh = h0 + sign * alpha * D * dirs[dname][None, :]
                p = key_prob(hh.astype(np.float32), a0)
                curve.append({"alpha": alpha, "p_key_at_cell": float(p[:, key_slot][m].mean()),
                              "blocked_channel_acc": float(((p[:, :9] > 0.5) == blocked_true).mean())})
            res[dname] = curve
        interventions["conditions"][name] = res
        for dname in ("probe", "diff_means", "random"):
            print(name, dname, [round(c["p_key_at_cell"], 4) for c in res[dname]], flush=True)

    path = save_result(args.exp, rec, {"world_model": wm_metrics, "probes": probes, "interventions": interventions},
                       extra={"leakage": leak, "train_history": hist})
    print("wrote", path)


if __name__ == "__main__":
    main()
