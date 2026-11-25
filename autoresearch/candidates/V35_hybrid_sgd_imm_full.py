"""THE ONLY FILE THE AUTORESEARCH AGENT MAY EDIT.  (interface: see eval.py)

V3-5 HYBRID (online-SGD base) + AR-3 interacting multiple model (IMM) bank of Kalman filters over disturbance persistence rates.
Each filter j models c_t = c_{t-1} + eta, eta ~ N(0, q_j * S0) (q relative to the prior covariance S0).
"""
import numpy as np

BASE = "sgd"

QS = np.array([0.0, 1e-5, 1e-4, 1e-3, 1e-2, np.inf])  # inf = 'fresh': prior re-drawn every step
P_STAY = 0.99


class Estimator:
    def __init__(self, M, prior_mean, prior_cov, sigma):
        J = len(QS)
        self.S0 = np.asarray(prior_cov, float)
        self.m0 = np.asarray(prior_mean, float)
        self.m = np.broadcast_to(np.asarray(prior_mean, float), (M, J, 2)).copy()
        self.C = np.broadcast_to(self.S0, (M, J, 2, 2)).copy()
        self.logw = np.full((M, J), -np.log(J))
        self.s2 = sigma ** 2
        self.Pi = np.full((J, J), (1 - P_STAY) / (J - 1)); np.fill_diagonal(self.Pi, P_STAY)

    def _predict_step(self):
        # IMM mixing
        w = np.exp(self.logw - self.logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        pred_w = w @ self.Pi                                  # (M,J)
        mix = (w[:, :, None] * self.Pi[None]) / np.maximum(pred_w[:, None, :], 1e-300)  # (M,i,j)
        m = np.einsum("mij,mid->mjd", mix, self.m)
        dm = self.m[:, :, None, :] - m[:, None, :, :]
        C = np.einsum("mij,mide->mjde", mix, self.C) + np.einsum("mij,mijd,mije->mjde", mix, dm, dm)
        fin = np.isfinite(QS)
        C[:, fin] = C[:, fin] + QS[fin][None, :, None, None] * self.S0[None, None]
        m[:, ~fin] = self.m0; C[:, ~fin] = self.S0
        return m, C, np.log(np.maximum(pred_w, 1e-300))

    def belief(self):
        m, C, logw = self._predict_step()
        w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        mean = np.einsum("mj,mjd->md", w, m)
        dm = m - mean[:, None]
        cov = np.einsum("mj,mjde->mde", w, C) + np.einsum("mj,mjd,mje->mde", w, dm, dm)
        return mean, cov

    def update(self, x, r):
        m, C, logw = self._predict_step()
        Cx = np.einsum("mjde,me->mjd", C, x)
        S = np.einsum("md,mjd->mj", x, Cx) + self.s2
        innov = r[:, None] - np.einsum("md,mjd->mj", x, m)
        K = Cx / S[..., None]
        self.m = m + K * innov[..., None]
        self.C = C - np.einsum("mjd,mje->mjde", K, Cx)
        self.logw = logw - 0.5 * (np.log(2 * np.pi * S) + innov ** 2 / S)
