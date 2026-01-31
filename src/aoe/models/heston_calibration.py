"""
Least-squares calibration of Heston parameters to an implied-vol surface
using QuantLib’s analytic European engine.

Inputs
------
strikes      : 1-D array of strike levels K
maturities   : 1-D array of maturities T (in *years*)
iv_matrix    : shape (len(T), len(K)) – market vols
s0, r, q     : spot, risk-free rate, dividend yield
"""

from typing import Tuple

import numpy as np
import QuantLib as ql
from scipy.optimize import least_squares

from .heston import HestonModel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ql_price(
    model: HestonModel,
    s0: float,
    k: float,
    r: float,
    q: float,
    t: float,
    opt_type: str = "call",
) -> float:
    """
    Convert a Heston European price into a Black-Scholes implied volatility.

    • If the Heston price is below intrinsic (or numerically zero) we
      return a tiny vol (1e-6) to avoid QuantLib’s put–call parity error.
    • Any QuantLib failure in the inversion also falls back to 1e-6.
    """
    price = model.price_euro(s0, float(k), r, q, t, opt_type)

    option_type = ql.Option.Call if opt_type == "call" else ql.Option.Put
    fwd = s0 * np.exp((r - q) * t)
    disc = np.exp(-r * t)

    # intrinsic lower bound
    intrinsic = (
        max(0.0, disc * (fwd - k))
        if opt_type == "call"
        else max(0.0, disc * (k - fwd))
    )

    if price <= intrinsic + 1e-8:
        return 1e-6  # effectively “flat” vol in the optimiser

    try:
        stddev = ql.blackFormulaImpliedStdDev(option_type, float(k), fwd, price, disc)
        return stddev / np.sqrt(t)
    except RuntimeError:
        return 1e-6  # fallback on numerical failure


# --------------------------------------------------------------------------- #
# Public calibrator
# --------------------------------------------------------------------------- #
def calibrate_heston(
    s0: float,
    r: float,
    q: float,
    strikes,
    maturities,
    iv_matrix,
    start: Tuple[float, float, float, float, float] = (
        1.5,
        0.04,
        0.3,
        -0.5,
        0.04,
    ),
):
    """
    Calibrate (kappa, theta, sigma, rho, v0) by least squares to market IVs.

    Returns
    -------
    params : fitted tuple (kappa, theta, sigma, rho, v0)
    rms    : root-mean-square fitting error
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    iv_target = np.asarray(iv_matrix, dtype=float)

    def residuals(params):
        kappa, theta, sigma, rho, v0 = params
        model = HestonModel(kappa, theta, sigma, rho, v0)
        res = []
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                iv_mod = _ql_price(model, s0, K, r, q, T)
                if np.isnan(iv_mod):
                    continue  # skip bad points
                res.append(iv_mod - iv_target[i, j])
        return res

    bounds = (
        (0.01, 0.0001, 0.05, -0.999, 0.0001),  # lower
        (10.0, 1.0, 1.5, 0.999, 2.0),          # upper
    )

    opt = least_squares(
        residuals,
        start,
        bounds=bounds,
        xtol=1e-6,
        ftol=1e-6,
        verbose=0,
    )

    rms = np.sqrt(np.mean(np.square(opt.fun))) if opt.fun.size else np.nan
    return opt.x, rms

