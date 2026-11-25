"""ONE sealed run of a frozen autoresearch candidate on the official TEST splits (full length, no cap).

Uses the same wmi.realdata.stream_generic as autoresearch/eval_v3.py so dev and test cannot diverge.
Models: fit on the official TRAIN split (first 80%), early-stopped on its last 20% -- identical to
scripts/run_realdata.py. Usage: python scripts/run_sealed_v3.py --seed 0 --candidate autoresearch/candidates/X.py
"""
import argparse
import hashlib
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_realdata import HP, datasets  # noqa: E402
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.realdata import Norm, assert_no_test_in_train, block_bootstrap_diff, fit_narx, stream_generic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--exp", default="realdata_sealed_v3")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cand", args.candidate)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sha = hashlib.sha1(Path(args.candidate).read_bytes()).hexdigest()[:10]
    rec = run_record({"experiment": args.exp, "seed": args.seed, "candidate": args.candidate, "candidate_sha": sha, **HP})
    rng = set_seed(args.seed)
    results = {}
    for name, d in datasets().items():
        assert_no_test_in_train(d["train"] + d["val"], d["test"])
        norm = Norm(np.concatenate([r[0] for r in d["train"]]), np.concatenate([r[1] for r in d["train"]]))
        torch.manual_seed(args.seed)
        model, info = fit_narx(d["train"], d["val"], norm, HP["n"], HP["train_steps"], rng)
        sig = info["val_rmse_norm"]
        cfg = {"s_theta": HP["s_theta"], "s_w": HP["s_w_over_sigma"] * sig, "sigma": sig, "sgd_lr": HP["sgd_lr"]}
        recs = []
        for j, (u, y) in enumerate(d["test"]):
            se1, sek = stream_generic(model, norm.nu(u), norm.ny(y), HP["n"], HP["k"], mod, cfg, HP["stride_k"])
            sc = norm.sd_y
            r = {"record": j, "u_std": float(np.std(u)),
                 "rmse1": {m: float(np.sqrt(v.mean()) * sc) for m, v in se1.items()},
                 "rmse10": {m: float(np.sqrt(v.mean()) * sc) for m, v in sek.items()},
                 "ci95_mse10_cand_minus": {m: [x * sc ** 2 for x in block_bootstrap_diff(sek["candidate"], sek[m], rng, block=50)]
                                           for m in ("forward", "sgd")},
                 "ci95_mse1_cand_minus": {m: [x * sc ** 2 for x in block_bootstrap_diff(se1["candidate"], se1[m], rng)]
                                          for m in ("forward", "sgd")}}
            recs.append(r)
            print(name, j, "rmse1", {m: round(v, 6) for m, v in r["rmse1"].items()},
                  "rmse10", {m: round(v, 6) for m, v in r["rmse10"].items()}, flush=True)
        results[name] = {"train_info": info, "records": recs}
    print("wrote", save_result(args.exp, rec, results))


if __name__ == "__main__":
    main()
