"""Export the seed-0 world models and planner constants for the browser demo, plus reference vectors that the
JavaScript port must reproduce (tests in demo/verify.mjs)."""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_hier import HGATE, cfg_for  # noqa: E402
from tests.test_physics import closed_form  # noqa: E402
from wmi import hier  # noqa: E402
from wmi.mujoco_exp import PuckBatch, Posterior  # noqa: E402
from wmi.mujoco_v3 import Planner3  # noqa: E402

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def member(m):
    L = [m.body[0], m.body[2], m.head]
    return {"in_mu": b64(m.in_mu), "in_sd": b64(m.in_sd), "out_mu": b64(m.out_mu), "out_sd": b64(m.out_sd),
            "layers": [{"W": b64(l.weight), "b": b64(l.bias), "shape": list(l.weight.shape)} for l in L]}


cfg, Mu, Mc = cfg_for(10, 0)
gap = json.loads(HGATE.read_text())["gap"]
out = {"Mu": [member(m) for m in Mu], "Mc": [member(m) for m in Mc],
       "cfg": {k: cfg[k] for k in ("pop", "elites", "iters", "beta", "forget", "K", "H")} |
       {"resid_sd": list(map(float, cfg["resid_sd"])), "resid_sd_c": list(map(float, cfg["resid_sd_c"]))},
       "task": {"walls": hier.WALLS, "wall_half": hier.WALL_HALF, "ylim": hier.YLIM, "gap": gap, "reach": hier.REACH,
                "goal_r": hier.GOAL_R, "start": hier.START.tolist(), "goal": hier.GOAL.tolist(),
                "subgoals": hier.subgoals().tolist(), "hz": 50.0}}
(ROOT / "demo" / "models.json").write_text(json.dumps(out))

# ---------------- reference vectors
rng = np.random.default_rng(1); ref = {}
x = rng.normal(size=(64, 4)).astype(np.float32); xc = rng.normal(size=(64, 7)).astype(np.float32)
with torch.no_grad():
    ref["mlp_u_in"] = x.tolist(); ref["mlp_u_out"] = [m(torch.tensor(x)).numpy().tolist() for m in Mu]
    ref["mlp_c_in"] = xc.tolist(); ref["mlp_c_out"] = [m(torch.tensor(xc)).numpy().tolist() for m in Mc]
E = 3; mass, damp = rng.uniform(.8, 1.25, E), rng.uniform(.7, 1.4, E)
phi = np.c_[rng.uniform(.4, 1.6, E), rng.normal(size=(E, 2))]; start = rng.uniform(-1, 1, (E, 2)); acts = rng.uniform(-1.2, 1.2, (60, E, 2))
env = PuckBatch(E); hier.apply_physics(env, mass, damp); env.reset(start); env.set_phi(phi)
sim = np.array([env.step(a) for a in acts])
assert np.abs(sim - closed_form(start, acts, phi, mass, damp)).max() < 1e-9
ref["phys"] = {"mass": mass.tolist(), "damp": damp.tolist(), "phi": phi.tolist(), "start": start.tolist(),
               "acts": acts.tolist(), "mujoco": sim.tolist()}
sd = np.array(cfg["resid_sd"]); post = Posterior(1, 0.5, 2 * sd, sd, cfg["forget"]); seq = []
for _ in range(40):
    eff, r = rng.normal(size=(1, 4)) * 0.05, rng.normal(size=(1, 4)) * 0.02
    post.update(eff, r); seq.append({"eff": eff[0].tolist(), "r": r[0].tolist(), "mean": post.mean()[0].tolist()})
ref["posterior"] = seq
# deterministic rollout cost for fixed candidate action sequences, per planner kind
s0 = np.array([[0.3, 0.1, 0.4, -0.2]]); goal = np.array([[0.65, 0.5]]); A = rng.uniform(-1, 1, (1, 8, cfg["H"], 2))
costs = {}
for kind in ("forward", "hindsight_posterior", "oracle_latent"):
    pl = Planner3(kind, Mu, Mc, 1, cfg, np.random.default_rng(0)); pl.phi_true = np.array([[0.8, 0.5, -0.3]])
    if kind == "hindsight_posterior":
        for it in seq[:10]:
            pl.post.update(np.array([it["eff"]]), np.array([it["r"]]))
    with torch.no_grad():
        At = torch.tensor(A, dtype=torch.float32); st = torch.tensor(s0, dtype=torch.float32)[:, None].expand(-1, 8, -1)
        c = torch.zeros(1, 8); prev = st[..., :2]; dead = torch.zeros(1, 8, dtype=torch.bool); g = torch.tensor(goal, dtype=torch.float32)
        for t in range(cfg["H"]):
            d, dsd = pl.predict(st[..., 2:], At[:, :, t]); st = st + d; p = st[..., :2]
            dead = dead | hier.hazard_t(p) | hier.hazard_t(0.5 * (p + prev)); prev = p
            c += ((p - g[:, None]) ** 2).sum(-1) + 0.01 * (At[:, :, t] ** 2).sum(-1) + cfg["beta"] * ((dsd / pl.rsd) ** 2).mean(-1) + 50 * dead.float()
    costs[kind] = c[0].numpy().tolist()
ref["rollout"] = {"s0": s0[0].tolist(), "goal": goal[0].tolist(), "A": A[0].tolist(), "phi_true": [0.8, 0.5, -0.3],
                  "post_updates": 10, "costs": costs}
(ROOT / "demo" / "reference.json").write_text(json.dumps(ref))
print("exported; gap", gap, "models.json", (ROOT / "demo" / "models.json").stat().st_size // 1024, "KB")
