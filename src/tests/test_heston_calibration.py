import numpy as np
from aoe.models.heston_calibration import calibrate_heston

def test_heston_calibration():
    # --- true parameters used to make faux market vols ---
    true = dict(kappa=2.0, theta=0.05, sigma=0.4, rho=-0.6, v0=0.04)

    s0, r, q = 100, 0.02, 0.0
    K = np.array([90, 100, 110])
    T = np.array([0.25, 0.5, 1.0])

    # quick-and-dirty approx: Black vol = ATM vol + skew*log(K/F)
    atm_vols = np.array([0.28, 0.26, 0.24])
    skew = -0.1  # negative skew
    iv = np.empty((len(T), len(K)))
    fwd = s0 * np.exp((r - q) * T[:, None])
    for i in range(len(T)):
        for j in range(len(K)):
            moneyness = np.log(K[j] / fwd[i, 0])
            iv[i, j] = atm_vols[i] + skew * moneyness

    params, rms = calibrate_heston(
        s0, r, q, K, T, iv, start=(1.0, 0.04, 0.3, -0.5, 0.05)
    )

    # basic sanity checks
    kappa, theta, sigma, rho, v0 = params
    assert 0.0 < kappa < 5.0
    assert 0.0 < theta < 0.2
    assert 0.1 < sigma < 1.0
    assert -1.0 < rho < 0.0
    assert 0.0 < v0 < 0.2
    # rms error should be small (< 0.03 ≈ 3 vol points)
    assert rms < 0.03
