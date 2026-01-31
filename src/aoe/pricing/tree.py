"""
CRR binomial tree (vectorised NumPy) with optional early-exercise.
"""

import numpy as np

def price_option_crr(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int = 200,
    option_type: str = "call",
    exercise: str = "euro",  # "euro" or "amer"
) -> float:
    """
    Returns discounted fair value via CRR tree.

    Parameters
    ----------
    s0, k, r, sigma, t : usual inputs
    n_steps            : tree depth
    option_type        : "call" or "put"
    exercise           : "euro" or "amer"
    """
    dt   = t / n_steps
    u    = np.exp(sigma * np.sqrt(dt))       # up factor
    d    = 1 / u                             # down factor
    p    = (np.exp(r * dt) - d) / (u - d)    # risk-neutral prob
    disc = np.exp(-r * dt)

    # terminal prices S_T at each node
    j = np.arange(n_steps + 1)
    sT = s0 * (u ** (n_steps - j)) * (d ** j)

    if option_type == "call":
        values = np.maximum(sT - k, 0.0)
    else:
        values = np.maximum(k - sT, 0.0)

    # backwards induction
    for _ in range(n_steps):
        m = values.size                     # node count at this level
        # step 1: risk-neutral expectation
        values = disc * (p * values[:-1] + (1 - p) * values[1:])

        if exercise == "amer":
            # step 2: intrinsic value must match the NEW node count (m-1)
            j = np.arange(m - 1)
            s = s0 * (u ** (m - 1 - j)) * (d ** j)
            intrinsic = (
                np.maximum(s - k, 0.0)
                if option_type == "call"
                else np.maximum(k - s, 0.0)
            )
            values = np.maximum(values, intrinsic)


    return float(values[0])
