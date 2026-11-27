"""Apply the EVAL_v3 selection rule mechanically (val artefacts only) and write FROZEN_v3.json.
Rule: min calibrate_sim loss over {v3, v3_ss} x temps; chosen model's val NLL must beat B2-v3 (bootstrap CI upper < 0),
else the best candidate that does."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    cands = []
    for r in ("v3", "v3_ss"):
        cal = json.loads((ROOT / f"runs/{r}/sim_calibration.json").read_text())
        m1 = json.loads((ROOT / f"results/v3_val_m1_{r}.json").read_text())
        d = m1["mmfm_minus_b2"]
        assert cal["chosen"]["recal"] and cal["split"] == "val" and m1["split"] == "val"
        cands.append(dict(run=f"runs/{r}", loss=cal["chosen"]["loss"], calib=cal["chosen"], nll_delta=d["delta"], nll_ci=d["ci95"],
                          beats_b2=d["ci95"][1] < 0, bits=m1["models"]["MMFM-v2"]["bits_per_event"]))
    ok = [c for c in cands if c["beats_b2"]]
    if not ok:
        raise SystemExit(f"no candidate beats B2-v3 on val: {cands}")
    best = min(cands, key=lambda c: c["loss"])
    ch = min(ok, key=lambda c: c["loss"])
    reason = "min val calib loss" if ch is best else f"{best['run']} had min loss but failed NLL constraint"
    run = ch["run"]
    files = [f"{run}/model.pt", f"{run}/tokenizer.json", f"{run}/sim_calibration.json", f"{run}/selection_log.json",
             "runs/v3_b2/model.pt", "runs/v3_b2/tokenizer.json", "runs/v3_b2/selection_log.json"]
    ns = time.time_ns()
    out = dict(freeze_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ns / 1e9)), freeze_ns=ns, run=run,
               files={f: sha(f) for f in files}, calib=ch["calib"], selection=dict(reason=reason, candidates=cands),
               note="EVAL_v3 forward test: window = [freeze+10 min, +90 min]; data built only after the window ends.")
    (ROOT / "FROZEN_v3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("FREEZE_NS", ns, "RUN", run)


if __name__ == "__main__":
    main()
