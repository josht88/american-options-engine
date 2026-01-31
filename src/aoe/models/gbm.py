import numpy as np


def simulate_gbm_path(s0, mu, sigma, t, n_steps, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = t / n_steps
    increments = np.random.normal(0.0, np.sqrt(dt), size=n_steps)
    log_path = np.log(s0) + np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * increments)
    return np.exp(log_path)
