"""Real measured benchmarks: forward vs residual-reuse vs hindsight (gain+offset posterior) vs online SGD.

Official train/test splits from nonlinearbenchmark.org (package `nonlinear-benchmarks`). Test records are
never used for training, normalisation, early stopping or any hyper-parameter (all fixed a priori below).
Usage: python scripts/run_realdata.py --seed 0
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nonlinear_benchmarks as nb  # noqa: E402

from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.realdata import Norm, assert_no_test_in_train, block_bootstrap_diff, fit_narx, stream_eval  # noqa: E402

CACHE = str(Path(__file__).resolve().parents[1] / "data_cache")
# a-priori hyper-parameters (NOT tuned on test data)
HP = {"n": 10, "k": 10, "train_steps": 4000, "s_theta": 0.5, "s_w_over_sigma": 2.0, "forget": 0.999, "sgd_lr": 1e-3,
      "val_frac": 0.2, "stride_k": 10}


def split_series(r, frac):
    T = len(r.y)
    c = int(T * (1 - frac))
    return (r.u[:c], r.y[:c]), (r.u[c:], r.y[c:])


def datasets():
    out = {}
    for name in ("Cascaded_Tanks", "EMPS", "Silverbox", "WienerHammerBenchMark"):
        tr, te = getattr(nb, name)(dir_placement=CACHE)
        te = te if isinstance(te, (list, tuple)) else [te]
        a, b = split_series(tr, HP["val_frac"])
        out[name] = {"train": [a], "val": [b], "test": [(r.u, r.y) for r in te]}
    tr, te = nb.ParWH(dir_placement=CACHE)
    lv = np.round([np.std(r.u) for r in tr], 3)
    levels = sorted(set(lv))
    test = [(r.u, r.y) for r in te]
    # iid: all amplitude levels; hold out the last 2 records of each level for validation
    tr_i, va_i, tr_s, va_s = [], [], [], []
    for L in levels:
        recs = [(tr[i].u, tr[i].y) for i in np.flatnonzero(lv == L)]
        tr_i += recs[:-2]; va_i += recs[-2:]
        if L in levels[:2]:  # shift: train only on the two LOWEST amplitude levels
            tr_s += recs[:-2]; va_s += recs[-2:]
    out["ParWH_iid"] = {"train": tr_i, "val": va_i, "test": test}
    out["ParWH_shift_low_to_all"] = {"train": tr_s, "val": va_s, "test": test}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", default="realdata_hindsight")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    rec = run_record({"experiment": "realdata_hindsight", "seed": args.seed, **HP})
    rng = set_seed(args.seed)
    torch.set_num_threads(4)

    results = {}
    for name, d in datasets().items():
        if args.only and name not in args.only:
            continue
        assert_no_test_in_train(d["train"] + d["val"], d["test"])
        norm = Norm(np.concatenate([r[0] for r in d["train"]]), np.concatenate([r[1] for r in d["train"]]))
        torch.manual_seed(args.seed)
        model, info = fit_narx(d["train"], d["val"], norm, HP["n"], HP["train_steps"], rng)
        cfg = {"s_theta": HP["s_theta"], "s_w": HP["s_w_over_sigma"] * info["val_rmse_norm"],
               "sigma": info["val_rmse_norm"], "forget": HP["forget"], "sgd_lr": HP["sgd_lr"]}
        per_rec = []
        for j, (u, y) in enumerate(d["test"]):
            se1, sek, extra = stream_eval(model, norm.nu(u), norm.ny(y), HP["n"], HP["k"], cfg, rng, stride=HP["stride_k"])
            sc = norm.sd_y  # back to original units
            r = {"record": j, "u_std": float(np.std(u)), "n_steps": int(len(se1["forward"])), **extra,
                 "rmse1": {m: float(np.sqrt(v.mean()) * sc) for m, v in se1.items()},
                 f"rmse{HP['k']}": {m: float(np.sqrt(v.mean()) * sc) for m, v in sek.items()},
                 "ci95_mse1_diff_vs_forward": {m: [x * sc ** 2 for x in block_bootstrap_diff(se1[m], se1["forward"], rng)]
                                               for m in se1 if m != "forward"},
                 f"ci95_mse{HP['k']}_diff_vs_forward": {m: [x * sc ** 2 for x in block_bootstrap_diff(sek[m], sek["forward"], rng, block=50)]
                                                       for m in sek if m != "forward"}}
            per_rec.append(r)
            print(name, j, "rmse1", {m: round(v, 5) for m, v in r["rmse1"].items()},
                  f"rmse{HP['k']}", {m: round(v, 5) for m, v in r[f'rmse{HP["k"]}'].items()}, flush=True)
        results[name] = {"train_info": info, "records": per_rec}
    path = save_result(args.exp, rec, results)
    print("wrote", path)


if __name__ == "__main__":
    main()
