// Planning worker: runs CEM-MPC off the main thread so the simulation keeps real time while a decision is computed.
importScripts("hlp_core.js");
let M = null;
onmessage = (e) => {
  const d = e.data;
  if (d.type === "init") { M = HLP.loadModels(d.models); postMessage({ type: "ready" }); return; }
  if (d.type !== "plan" || !M) return;
  const t0 = performance.now();
  M.cfg.K = d.K; M.cfg.H = d.H;
  const ctx = { postMean: d.postMean, z: d.z };
  const out = HLP.cemPlan(d.kind, M, Float32Array.from(d.obs), d.goal, Float32Array.from(d.mu), HLP.rng(d.seed), ctx, d.pop);
  const ms = performance.now() - t0;
  // what this planner believes its plan will do: one mean rollout of its own model
  const path = []; const s = Float32Array.from(d.obs);
  for (let t = 0; t < d.H; t++) {
    const { mean } = HLP.predict(d.kind, M, Float32Array.from([s[2], s[3]]),
      Float32Array.from([out.plan[t * 2], out.plan[t * 2 + 1]]), 1, ctx);
    for (let k = 0; k < 4; k++) s[k] += mean[k];
    path.push([s[0], s[1]]);
  }
  postMessage({ type: "plan", id: d.id, ep: d.ep, plan: Array.from(out.plan), mu: Array.from(out.mu), ms, path });
};
