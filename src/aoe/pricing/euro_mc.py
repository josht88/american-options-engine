import numpy as np

from aoe.models.gbm import simulate_gbm_path


def price_euro_call_mc(s0, k, r, sigma, t, n_paths=50_000, n_steps=100):
    payoffs = []
    for _ in range(n_paths):
        path = simulate_gbm_path(s0, r, sigma, t, n_steps)
        payoffs.append(max(path[-1] - k, 0.0))
    return np.exp(-r * t) * np.mean(payoffs)
