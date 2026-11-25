"""V3-6 HYBRID: online-SGD base + Bayesian model averaging of two hypotheses about the disturbance:
  P: persistent (Gaussian posterior with forgetting lambda=0.995)
  F: fresh (prior re-drawn every step == forward, Thm 2)
weighted by exponentially-forgotten predictive log-evidence (lambda_e = 0.99). Unlike IMM, P's state is never
mixed with F's, so evidence for 'fresh' shrinks the correction instead of resetting the estimate.
"""
import numpy as np

BASE = "sgd"
FORGET, FORGET_EV = 0.995, 0.99


class Estimator:
    def __init__(self, M, prior_mean, prior_cov, sigma):
        self.S0 = np.asarray(prior_cov, float); self.m0 = np.asarray(prior_mean, float)
        self.P0 = np.linalg.inv(self.S0)
        self.P = np.broadcast_to(self.P0, (M, 2, 2)).copy()
        self.b = np.broadcast_to(self.P0 @ self.m0, (M, 2)).copy()
        self.s2 = sigma ** 2
        self.le = np.zeros((M, 2))  # forgotten log-evidence for [P, F]

    def _post(self):
        C = np.linalg.inv(self.P)
        return np.einsum("mij,mj->mi", C, self.b), C

    def belief(self):
        mP, CP = self._post()
        w = np.exp(self.le - self.le.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        mF = np.broadcast_to(self.m0, mP.shape); CF = np.broadcast_to(self.S0, CP.shape)
        mean = w[:, :1] * mP + w[:, 1:] * mF
        dP, dF = mP - mean, mF - mean
        cov = (w[:, 0, None, None] * (CP + np.einsum("md,me->mde", dP, dP))
               + w[:, 1, None, None] * (CF + np.einsum("md,me->mde", dF, dF)))
        return mean, cov

    def update(self, x, r):
        mP, CP = self._post()
        for j, (m, C) in enumerate(((mP, CP), (np.broadcast_to(self.m0, mP.shape), np.broadcast_to(self.S0, CP.shape)))):
            S = np.einsum("md,mde,me->m", x, C, x) + self.s2
            e = r - np.einsum("md,md->m", x, m)
            self.le[:, j] = FORGET_EV * self.le[:, j] - 0.5 * (np.log(2 * np.pi * S) + e ** 2 / S)
        self.P = FORGET * self.P + (1 - FORGET) * self.P0
        self.b = FORGET * self.b + (1 - FORGET) * (self.P0 @ self.m0)
        self.P = self.P + np.einsum("mi,mj->mij", x, x) / self.s2
        self.b = self.b + x * (r / self.s2)[:, None]
