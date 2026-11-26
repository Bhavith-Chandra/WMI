"""The simulator must match the closed-form puck dynamics (MuJoCo Euler with implicit joint damping):
v <- v + h (F - b v) / (m + h b),  q <- q + h v,  F = 5 gain a + wind,  h = 0.01, 5 substeps per control step.
This is also the reference the browser demo is verified against."""
import numpy as np

from wmi.hier import apply_physics
from wmi.mujoco_exp import FRAME_SKIP, PuckBatch


def closed_form(start, acts, phi, mass, damp):
    q = start.copy(); v = np.zeros_like(start); out = []; h = 0.01; m, b = mass[:, None], damp[:, None]
    for a in acts:
        a = np.clip(a, -1, 1)
        for _ in range(FRAME_SKIP):
            F = 5 * phi[:, [0]] * a + phi[:, 1:]; v = v + h * (F - b * v) / (m + h * b); q = q + h * v
        out.append(np.c_[q, v])
    return np.array(out)


def test_simulator_matches_closed_form_with_perturbed_physics():
    rng = np.random.default_rng(0); E = 5
    mass, damp = rng.uniform(.8, 1.25, E), rng.uniform(.7, 1.4, E)
    phi = np.c_[rng.uniform(.4, 1.6, E), rng.normal(size=(E, 2))]
    start = rng.uniform(-1, 1, (E, 2)); acts = rng.uniform(-1.2, 1.2, (100, E, 2))
    env = PuckBatch(E); apply_physics(env, mass, damp); env.reset(start); env.set_phi(phi)
    sim = np.array([env.step(a) for a in acts])
    assert np.abs(sim - closed_form(start, acts, phi, mass, damp)).max() < 1e-9


def test_mass_change_has_effect():
    E = 1; start = np.zeros((1, 2)); acts = np.ones((20, 1, 2)); phi = np.array([[1.0, 0.0, 0.0]])
    out = []
    for m in (1.0, 1.25):
        env = PuckBatch(E); apply_physics(env, np.array([m]), np.array([1.0])); env.reset(start); env.set_phi(phi)
        out.append(np.array([env.step(a) for a in acts]))
    assert np.abs(out[0] - out[1]).max() > 1e-3
