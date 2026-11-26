// Verifies the JS port against reference vectors from scripts/export_demo.py. Exit code 1 on any failure.
import { readFileSync } from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const H = require("./hlp_core.js");
const M = H.loadModels(JSON.parse(readFileSync(new URL("./models.json", import.meta.url))));
const R = JSON.parse(readFileSync(new URL("./reference.json", import.meta.url)));
const res = {}; let ok = true;
const check = (name, err, tol) => { res[name] = { max_err: err, tol, pass: err <= tol }; if (err > tol) ok = false; };

// 1. MLP members
for (const [key, mem, nin] of [["u", M.Mu, 4], ["c", M.Mc, 7]]) {
  const X = Float32Array.from(R[`mlp_${key}_in`].flat()); let e = 0;
  mem.forEach((m, j) => { const o = H.mlp(m, X, 64); R[`mlp_${key}_out`][j].flat().forEach((v, k) => e = Math.max(e, Math.abs(v - o[k]))); });
  check(`mlp_${key}`, e, 1e-4);
}
// 2. physics vs MuJoCo (perturbed mass and damping)
{ const P = R.phys; let e = 0;
  for (let w = 0; w < P.start.length; w++) {
    let s = [P.start[w][0], P.start[w][1], 0, 0];
    P.acts.forEach((a, t) => { s = H.physicsStep(s, a[w], P.phi[w], P.mass[w], P.damp[w]);
      for (let d = 0; d < 4; d++) e = Math.max(e, Math.abs(s[d] - P.mujoco[t][w][d])); });
  }
  check("physics_vs_mujoco", e, 1e-9); }
// 3. posterior
{ const sd = M.cfg.resid_sd; const post = new H.Posterior(0.5, sd.map(x => 2 * x), sd, M.cfg.forget); let e = 0;
  for (const it of R.posterior) { post.update(it.eff, it.r); post.mean().forEach((tw, d) => { e = Math.max(e, Math.abs(tw[0] - it.mean[d][0]) / (1 + Math.abs(it.mean[d][0])), Math.abs(tw[1] - it.mean[d][1]) / (1 + Math.abs(it.mean[d][1]))); }); }
  check("posterior", e, 1e-6); }
// 4. rollout cost per planner kind (float32 network, so relative tolerance)
{ const r = R.rollout, Hh = M.cfg.H, A = Float32Array.from(r.A.flat(2));
  const sd = M.cfg.resid_sd; const post = new H.Posterior(0.5, sd.map(x => 2 * x), sd, M.cfg.forget);
  R.posterior.slice(0, r.post_updates).forEach(it => post.update(it.eff, it.r));
  for (const kind of ["forward", "hindsight_posterior", "oracle_latent"]) {
    const c = H.rolloutCost(kind, M, Float32Array.from(r.s0), r.goal, A, 8, { z: r.phi_true, postMean: post.mean() });
    let e = 0; r.costs[kind].forEach((v, i) => e = Math.max(e, Math.abs(v - c[i]) / Math.max(1, Math.abs(v))));
    check(`rollout_cost_${kind}`, e, 1e-3);
  }
}
// 5. timing of one decision (E = 1) at the paper's population
{ const R0 = H.rng(1), mu = new Float32Array(M.cfg.H * 2), post = new H.Posterior(0.5, M.cfg.resid_sd.map(x => 2 * x), M.cfg.resid_sd, M.cfg.forget);
  for (const kind of ["forward", "hindsight_posterior"]) {
    const ts = [];
    for (let i = 0; i < 6; i++) { const t0 = performance.now(); H.cemPlan(kind, M, Float32Array.from([0.2, 0.1, 0.5, -0.2]), [0.65, 0.5], mu, R0, { postMean: post.mean() }); ts.push(performance.now() - t0); }
    res[`node_ms_per_decision_${kind}_pop${M.cfg.pop}`] = Math.round(ts.slice(1).sort((a, b) => a - b)[2]);
  }
}
console.log(JSON.stringify(res, null, 1)); console.log(ok ? "ALL CHECKS PASS" : "VERIFICATION FAILED"); process.exit(ok ? 0 : 1);
