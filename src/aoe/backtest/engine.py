"""
Minimal backtest engine: builds iron condors from a modelled range and replays
over a price series. Pure-GBM mode for validation.

Sanity checks (optional):
- On the first trade, Monte-Carlo EV of the condor (under GBM) must be >= 0
  within a tiny tolerance (seller’s edge non-negative in a symmetric setup).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from aoe.strategy.credit import (
    price_credit_spread,
    ev_iron_condor_logn,
)

# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

def _rolling_iv(close: pd.Series, window: int = 20) -> pd.Series:
    """Realized annualized vol estimate from close series (log-returns)."""
    lr = np.log(close).diff()
    rv = lr.rolling(window).std() * np.sqrt(252)
    return rv

def _logn_quantile(spot: float, mu: float, sigma: float, t: float, p: float) -> float:
    """Quantile of S_T for GBM: S_T = S0 * exp((mu-0.5*sigma^2)T + sigma sqrt(T) Z_p)."""
    if t <= 0 or sigma <= 0:
        return float(spot)
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    z = float(norm.ppf(p))
    drift = (mu - 0.5 * sigma**2) * t
    vol_term = sigma * math.sqrt(t) * z
    return float(spot * math.exp(drift + vol_term))

def _build_condor_from_range(
    s0: float, r: float, q: float, iv: float, t: float, p_tail: float, wing_width: float
) -> Dict[str, float]:
    """
    Choose symmetric iron-condor strikes from lognormal quantiles.
    p_tail is lower/upper tail probability (e.g., 0.20 → 20%/80%).
    """
    mu = r - q  # risk-neutral drift for GBM
    k_lo = _logn_quantile(s0, mu, iv, t, p_tail)       # lower short (put)
    k_hi = _logn_quantile(s0, mu, iv, t, 1.0 - p_tail) # upper short (call)

    # Wings pushed outward by fixed width (in price space)
    long_put  = max(0.01, k_lo - wing_width)
    short_put = k_lo
    short_call = k_hi
    long_call  = k_hi + wing_width

    return dict(
        long_put=float(long_put),
        short_put=float(short_put),
        short_call=float(short_call),
        long_call=float(long_call),
    )

@dataclass
class Trade:
    date: pd.Timestamp
    s0: float
    t_years: float
    iv: float
    long_put: float
    short_put: float
    short_call: float
    long_call: float
    credit: float
    qty: int   # <-- added

# -----------------------------------------------------------------------------#
# Core backtest
# -----------------------------------------------------------------------------#

def run_on_series(
    close: pd.Series,
    dte_days: int = 21,
    trade_every_n_days: int = 5,
    bankroll: float = 50_000.0,
    p_tail: float = 0.20,
    wing_width: float = 5.0,
    r: float = 0.02,
    q: float = 0.0,
    sanity: bool = True,
) -> Dict[str, Any]:
    """
    Replay iron-condor selling over a price series (GBM validation harness).
    Builds condors from GBM range quantiles, prices credit under Black, and
    runs a simple pnl mark-to-market by holding to expiry (toy).

    Returns:
      {
        "trades": [Trade, ...],
        "equity": pd.Series,
        "summary": {...},
        "sanity_ok": bool,
      }
    """
    close = close.dropna().astype(float)
    if close.empty:
        return {"trades": [], "equity": pd.Series(dtype=float), "summary": {}, "sanity_ok": True}

    iv_est = _rolling_iv(close).bfill()
    dates = close.index

    trades: List[Trade] = []
    equity_curve = []
    cash = bankroll
    sanity_ok = True

    for i, dt_ in enumerate(dates):
        if i % trade_every_n_days != 0:
            equity_curve.append(cash)
            continue

        s0 = float(close.iloc[i])
        iv = float(iv_est.iloc[i])
        t  = dte_days / 365.0

        # Build strikes from range
        strikes = _build_condor_from_range(s0, r, q, iv, t, p_tail, wing_width)

        # Price mid-credit via two short verticals (Black)
        credit_call = price_credit_spread(s0, r, q, t, iv, right="C",
                                          short_k=strikes["short_call"], long_k=strikes["long_call"])
        credit_put  = price_credit_spread(s0, r, q, t, iv, right="P",
                                          short_k=strikes["short_put"], long_k=strikes["long_put"])
        credit_mid = float(credit_call + credit_put)

        # ---- Sanity check (once) -------------------------------------------
        if sanity and not trades:
            # MC EV under GBM should be ~ >= 0 for symmetric condor around quantiles
            ev = ev_iron_condor_logn(
                s0, r, q, iv, t,
                strikes["long_put"], strikes["short_put"],
                strikes["short_call"], strikes["long_call"],
                credit_mid,
                n_paths=20000, seed=123
            )
            # allow tiny numerical noise
            if ev < -1e-3:
                sanity_ok = False

        # Allocate 1% notional per trade as a toy rule
        notional = 0.01 * bankroll
        qty = max(1, int(notional / max(credit_mid, 1e-6)))  # crude

        trades.append(Trade(
            date=dt_,
            s0=s0, t_years=t, iv=iv, credit=credit_mid, qty=qty,
            **strikes
        ))

        cash += qty * credit_mid  # receive credit
        equity_curve.append(cash)

    equity = pd.Series(equity_curve, index=dates[:len(equity_curve)], name="Equity")
    summary = dict(
        n_trades=len(trades),
        start=float(bankroll),
        end=float(equity.iloc[-1]) if not equity.empty else float(bankroll),
        pnl=float((equity.iloc[-1] - bankroll) if not equity.empty else 0.0),
    )
    return dict(trades=trades, equity=equity, summary=summary, sanity_ok=sanity_ok)
