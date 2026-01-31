"""
American binomial tree with a Poisson jump branch at each step.
Up-/down factors follow CRR; jump multiplicative factor
    J = exp(mu_J + 0.5 * sigma_J^2).
"""

import math
import numpy as np


def price_american_jump_tree(
    s0: float,
    k: float,
    r: float,
    q: float,
    t: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    n_steps: int = 300,
    option_type: str = "put",
) -> float:
    dt = t / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p_up = (math.exp((r - q) * dt) - d) / (u - d)

    # jump branch
    j_factor = math.exp(mu_j + 0.5 * sigma_j**2)
    p_jump = lam * dt
    p_stay = 1 - p_jump

    disc = math.exp(-r * dt)

    # ---------- terminal layer ----------
    j = np.arange(n_steps + 1)
    s_prev = s0 * (u ** (n_steps - j)) * (d ** j)          # descending grid
    values = (
        np.maximum(k - s_prev, 0.0)
        if option_type == "put"
        else np.maximum(s_prev - k, 0.0)
    )

    # ---------- backward induction ----------
    for step in range(n_steps, 0, -1):
        m = values.size
        idx = np.arange(m - 1)
        s_curr = s0 * (u ** (step - 1 - idx)) * (d ** idx)   # length m-1

        # ascending copies for interpolation
        xp = s_prev[::-1]
        fp = values[::-1]

        cont = disc * (
            p_stay * (p_up * values[:-1] + (1 - p_up) * values[1:]) +
            p_jump * np.interp(
                s_curr * j_factor,   # target x
                xp,                  # increasing xp
                fp                   # reversed fp
            )
        )

        intrinsic = (
            np.maximum(k - s_curr, 0.0)
            if option_type == "put"
            else np.maximum(s_curr - k, 0.0)
        )

        values = np.maximum(cont, intrinsic)
        s_prev = s_curr                        # update grid reference

    return float(values[0])
