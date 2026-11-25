"""Causal interventions along probe directions, with matched random-direction controls."""
from __future__ import annotations

import numpy as np


def logistic_direction(model, scaler):
    """Binary logistic probe -> (w, b) in raw activation space: logit = x @ w + b."""
    coef = model.coef_.ravel()
    w = coef / scaler.scale_
    b = float(model.intercept_[0] - np.sum(coef * scaler.mean_ / scaler.scale_))
    return w, b


def set_logit(h: np.ndarray, w: np.ndarray, b: float, target: float) -> np.ndarray:
    """Minimal-norm edit of h so that the probe logit equals `target`."""
    cur = h @ w + b
    return h + ((target - cur) / (w @ w))[:, None] * w[None, :]


def matched_random(h: np.ndarray, h_edit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random direction per row with the same displacement norm as the probe edit."""
    d = np.linalg.norm(h_edit - h, axis=1, keepdims=True)
    r = rng.standard_normal(h.shape)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    return h + d * r
