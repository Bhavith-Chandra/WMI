// Hindsight latent-variable planning: browser/Node core. A line-by-line port of wmi/hier.py + wmi/mujoco_v3.Planner3
// (kinds forward, hindsight_posterior, oracle_latent) and the MuJoCo puck dynamics. Verified by demo/verify.mjs
// against reference vectors produced by scripts/export_demo.py from the Python code.
(function (root) {
  "use strict";

  function decodeF32(b64) {
    const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(buf);
  }

  function loadMember(m) {
    return {
      inMu: decodeF32(m.in_mu), inSd: decodeF32(m.in_sd), outMu: decodeF32(m.out_mu), outSd: decodeF32(m.out_sd),
      layers: m.layers.map(l => ({ W: decodeF32(l.W), b: decodeF32(l.b), nout: l.shape[0], nin: l.shape[1] })),
    };
  }

  function loadModels(json) {
    return { Mu: json.Mu.map(loadMember), Mc: json.Mc.map(loadMember), cfg: json.cfg, task: json.task };
  }

  // One MLP member on a batch: X is n x nin (row-major). Returns n x 4.
  function mlp(mem, X, n) {
    const nin = mem.layers[0].nin;
    let h = new Float32Array(n * nin);
    for (let i = 0; i < n; i++) for (let j = 0; j < nin; j++) h[i * nin + j] = (X[i * nin + j] - mem.inMu[j]) / mem.inSd[j];
    for (let li = 0; li < mem.layers.length; li++) {
      const L = mem.layers[li], out = new Float32Array(n * L.nout), relu = li < mem.layers.length - 1;
      for (let i = 0; i < n; i++) {
        const hi = i * L.nin;
        for (let o = 0; o < L.nout; o++) {
          let s = L.b[o]; const wo = o * L.nin;
          for (let k = 0; k < L.nin; k++) s += L.W[wo + k] * h[hi + k];
          out[i * L.nout + o] = relu && s < 0 ? 0 : s;
        }
      }
      h = out;
    }
    for (let i = 0; i < n; i++) for (let d = 0; d < 4; d++) h[i * 4 + d] = h[i * 4 + d] * mem.outSd[d] + mem.outMu[d];
    return h;
  }

  // Bayesian regression r_d = theta_d * eff_d + w_d with forgetting (port of wmi.mujoco_exp.Posterior, E = 1).
  class Posterior {
    constructor(sTheta, sW, sigma, forget) {
      this.P0 = [], this.P = [], this.b = [], this.s2 = sigma.map(s => s * s), this.lam = forget;
      for (let d = 0; d < 4; d++) {
        this.P0.push([1 / (sTheta * sTheta), 0, 0, 1 / (sW[d] * sW[d])]);
        this.P.push(this.P0[d].slice()); this.b.push([0, 0]);
      }
    }
    mean() {   // per dim: [theta, w]
      return this.P.map((P, d) => {
        const det = P[0] * P[3] - P[1] * P[2], b = this.b[d];
        return [(P[3] * b[0] - P[1] * b[1]) / det, (-P[2] * b[0] + P[0] * b[1]) / det];
      });
    }
    update(eff, r) {
      for (let d = 0; d < 4; d++) {
        const P = this.P[d], P0 = this.P0[d], x = [eff[d], 1], s2 = this.s2[d];
        for (let k = 0; k < 4; k++) P[k] = this.lam * P[k] + (1 - this.lam) * P0[k];
        this.b[d][0] *= this.lam; this.b[d][1] *= this.lam;
        P[0] += x[0] * x[0] / s2; P[1] += x[0] * x[1] / s2; P[2] += x[1] * x[0] / s2; P[3] += x[1] * x[1] / s2;
        this.b[d][0] += x[0] * r[d] / s2; this.b[d][1] += x[1] * r[d] / s2;
      }
    }
  }

  // Ensemble prediction of the state increment for n (v, a) pairs. Returns {mean, std} (n x 4), std unbiased (torch).
  function predict(kind, M, V, A, n, ctx) {
    const members = kind === "oracle_latent" ? M.Mc : M.Mu, nin = kind === "oracle_latent" ? 7 : 4;
    const X = new Float32Array(n * nin);
    for (let i = 0; i < n; i++) {
      X[i * nin] = V[i * 2]; X[i * nin + 1] = V[i * 2 + 1]; X[i * nin + 2] = A[i * 2]; X[i * nin + 3] = A[i * 2 + 1];
      if (nin === 7) { X[i * 7 + 4] = ctx.z[0]; X[i * 7 + 5] = ctx.z[1]; X[i * 7 + 6] = ctx.z[2]; }
    }
    const outs = members.map(m => mlp(m, X, n));
    if (kind === "hindsight_posterior") {
      const X0 = X.slice(); for (let i = 0; i < n; i++) { X0[i * 4 + 2] = 0; X0[i * 4 + 3] = 0; }
      const pm = ctx.postMean;
      members.forEach((m, j) => {
        const o0 = mlp(m, X0, n), o = outs[j];
        for (let i = 0; i < n; i++) for (let d = 0; d < 4; d++) {
          const k = i * 4 + d; o[k] = o[k] + pm[d][0] * (o[k] - o0[k]) + pm[d][1];
        }
      });
    }
    const E = outs.length, mean = new Float32Array(n * 4), std = new Float32Array(n * 4);
    for (let k = 0; k < n * 4; k++) {
      let s = 0; for (let j = 0; j < E; j++) s += outs[j][k];
      const mu = s / E; let v = 0; for (let j = 0; j < E; j++) v += (outs[j][k] - mu) ** 2;
      mean[k] = mu; std[k] = Math.sqrt(v / (E - 1));
    }
    return { mean, std };
  }

  function hazard(x, y, task) {
    if (Math.abs(y) > task.ylim) return true;
    for (const [xw, yc] of task.walls) if (Math.abs(x - xw) < task.wall_half && Math.abs(y - yc) > task.gap / 2) return true;
    return false;
  }

  // Cost of P candidate action sequences A (P x H x 2) from state s0 toward goal (port of hier.act_hazard's cost).
  function rolloutCost(kind, M, s0, goal, A, P, ctx) {
    const c = M.cfg, H = c.H, task = M.task, rsd = c.resid_sd;
    const st = new Float32Array(P * 4), cost = new Float64Array(P), dead = new Uint8Array(P);
    const prev = new Float32Array(P * 2), V = new Float32Array(P * 2), Aa = new Float32Array(P * 2);
    for (let i = 0; i < P; i++) { st.set(s0, i * 4); prev[i * 2] = s0[0]; prev[i * 2 + 1] = s0[1]; }
    for (let t = 0; t < H; t++) {
      for (let i = 0; i < P; i++) {
        V[i * 2] = st[i * 4 + 2]; V[i * 2 + 1] = st[i * 4 + 3];
        Aa[i * 2] = A[(i * H + t) * 2]; Aa[i * 2 + 1] = A[(i * H + t) * 2 + 1];
      }
      const { mean, std } = predict(kind, M, V, Aa, P, ctx);
      for (let i = 0; i < P; i++) {
        for (let d = 0; d < 4; d++) st[i * 4 + d] += mean[i * 4 + d];
        const x = st[i * 4], y = st[i * 4 + 1], mx = 0.5 * (x + prev[i * 2]), my = 0.5 * (y + prev[i * 2 + 1]);
        if (!dead[i] && (hazard(x, y, task) || hazard(mx, my, task))) dead[i] = 1;
        prev[i * 2] = x; prev[i * 2 + 1] = y;
        let unc = 0; for (let d = 0; d < 4; d++) unc += (std[i * 4 + d] / rsd[d]) ** 2;
        const ax = Aa[i * 2], ay = Aa[i * 2 + 1];
        cost[i] += (x - goal[0]) ** 2 + (y - goal[1]) ** 2 + 0.01 * (ax * ax + ay * ay) + c.beta * unc / 4 + task.hz * dead[i];
      }
    }
    return cost;
  }

  function rng(seed) {   // mulberry32 + Box-Muller
    let a = seed >>> 0, spare = null;
    const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    const n = () => { if (spare !== null) { const s = spare; spare = null; return s; }
      let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r === 0 || r >= 1);
      const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f; };
    return { u, n };
  }

  // CEM over H x 2 action sequences, warm-started from mu (H x 2). Returns {plan, mu} (mu shifted by K for next time).
  function cemPlan(kind, M, s0, goal, mu, R, ctx, popOverride) {
    const c = M.cfg, H = c.H, P = popOverride || c.pop, nE = Math.min(c.elites, Math.max(2, Math.floor(P / 10)));
    let m = mu.slice(), sd = new Float32Array(H * 2).fill(0.6);
    const A = new Float32Array(P * H * 2);
    for (let it = 0; it < c.iters; it++) {
      for (let i = 0; i < P; i++) for (let k = 0; k < H * 2; k++)
        A[i * H * 2 + k] = Math.max(-1, Math.min(1, m[k] + sd[k] * R.n()));
      const cost = rolloutCost(kind, M, s0, goal, A, P, ctx);
      const idx = Array.from(cost.keys()).sort((p, q) => cost[p] - cost[q]).slice(0, nE);
      const nm = new Float32Array(H * 2), ns = new Float32Array(H * 2);
      for (const i of idx) for (let k = 0; k < H * 2; k++) nm[k] += A[i * H * 2 + k] / nE;
      for (const i of idx) for (let k = 0; k < H * 2; k++) ns[k] += (A[i * H * 2 + k] - nm[k]) ** 2 / nE;
      for (let k = 0; k < H * 2; k++) ns[k] = Math.sqrt(ns[k]) + 0.05;
      m = nm; sd = ns;
    }
    const plan = m.slice(), K = c.K, next = new Float32Array(H * 2);
    next.set(m.subarray(K * 2)); return { plan, mu: next };
  }

  // MuJoCo puck: Euler with implicit joint damping, h = 0.01, 5 substeps per 50 ms control step (tests/test_physics.py).
  function physicsStep(s, a, z, mass, damp) {
    const ax = Math.max(-1, Math.min(1, a[0])), ay = Math.max(-1, Math.min(1, a[1]));
    let [qx, qy, vx, vy] = s; const h = 0.01;
    for (let k = 0; k < 5; k++) {
      const Fx = 5 * z[0] * ax + z[1], Fy = 5 * z[0] * ay + z[2];
      vx = vx + h * (Fx - damp * vx) / (mass + h * damp); vy = vy + h * (Fy - damp * vy) / (mass + h * damp);
      qx += h * vx; qy += h * vy;
    }
    return [qx, qy, vx, vy];
  }

  // Belief update after one executed step (port of Planner3.observe for hindsight_posterior, E = 1).
  function observePosterior(M, post, s, a, s2) {
    const X = new Float32Array([s[2], s[3], a[0], a[1]]), X0 = new Float32Array([s[2], s[3], 0, 0]);
    const eff = [0, 0, 0, 0], base = [0, 0, 0, 0], E = M.Mu.length;
    for (const m of M.Mu) {
      const o = mlp(m, X, 1), o0 = mlp(m, X0, 1);
      for (let d = 0; d < 4; d++) { base[d] += o[d] / E; eff[d] += (o[d] - o0[d]) / E; }
    }
    post.update(eff, [0, 1, 2, 3].map(d => (s2[d] - s[d]) - base[d]));
  }

  const api = { decodeF32, loadModels, mlp, Posterior, predict, hazard, rolloutCost, rng, cemPlan, physicsStep, observePosterior };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.HLP = api;
})(typeof self !== "undefined" ? self : this);
