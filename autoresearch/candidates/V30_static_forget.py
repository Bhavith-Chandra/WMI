"""V3-0: static Gaussian posterior over (gain, offset) with exponential forgetting lambda=0.999; frozen base."""
import numpy as np

BASE = "frozen"
FORGET = 0.999


class Estimator:
    def __init__(self, M, prior_mean, prior_cov, sigma):
        self.P0 = np.linalg.inv(prior_cov); self.m0 = np.asarray(prior_mean, float)
        self.P = np.broadcast_to(self.P0, (M, 2, 2)).copy()
        self.b = np.broadcast_to(self.P0 @ self.m0, (M, 2)).copy()
        self.s2 = sigma ** 2

    def belief(self):
        cov = np.linalg.inv(self.P)
        return np.einsum("mij,mj->mi", cov, self.b), cov

    def update(self, x, r):
        self.P = FORGET * self.P + (1 - FORGET) * self.P0
        self.b = FORGET * self.b + (1 - FORGET) * (self.P0 @ self.m0)
        self.P = self.P + np.einsum("mi,mj->mij", x, x) / self.s2
        self.b = self.b + x * (r / self.s2)[:, None]
