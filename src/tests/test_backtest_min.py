import numpy as np
import pandas as pd

from aoe.backtest.engine import run_on_series

def _gbm_series(n=120, s0=100.0, mu=0.02, sigma=0.20, seed=7):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n)
    s = s0 * np.exp(np.cumsum(shocks))
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(s, index=idx, name="Close")

def test_backtest_minimal_runs():
    ser = _gbm_series()
    res = run_on_series(ser, dte_days=21, trade_every_n_days=5, bankroll=50_000.0, sanity=True)

    # Basic structure
    assert "trades" in res and "equity" in res and "summary" in res and "sanity_ok" in res
    assert isinstance(res["summary"]["n_trades"], int)
    assert res["summary"]["n_trades"] > 0
    assert res["equity"].shape[0] > 0

    # Sanity: first-trade MC EV was non-negative (within tolerance)
    assert res["sanity_ok"] is True
