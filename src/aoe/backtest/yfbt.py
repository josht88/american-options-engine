# (full file)
from __future__ import annotations
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import yfinance as yf

from aoe.strategy.credit import (
    price_credit_spread,
    _black_call_price,
    _black_put_price,
)

def _rolling_iv(close: pd.Series, lookback: int = 21) -> pd.Series:
    rets = close.pct_change()
    return rets.rolling(lookback).std() * math.sqrt(252)

def _logn_quantile(spot: float, mu: float, sigma: float, t: float, p: float) -> float:
    if sigma <= 0 or t <= 0:
        return spot
    from scipy.stats import norm
    z = norm.ppf(p)
    return float(spot * math.exp((mu - 0.5 * sigma * sigma) * t + sigma * math.sqrt(t) * z))

def _build_condor_from_range(s0: float, r: float, q: float, iv: float,
                             t: float, p_tail: float, wing_width: float) -> Dict[str, float]:
    mu = r - q
    k_lo = _logn_quantile(s0, mu, iv, t, p_tail)
    k_hi = _logn_quantile(s0, mu, iv, t, 1.0 - p_tail)
    return dict(
        short_put = k_lo,
        long_put  = max(1e-6, k_lo - wing_width),
        short_call= k_hi,
        long_call = k_hi + wing_width,
    )

def _realized_condor_loss_buyer(sT: float, strikes: Dict[str, float]) -> float:
    width_p = strikes["short_put"] - strikes["long_put"]
    width_c = strikes["long_call"] - strikes["short_call"]
    put_spread_payoff  = max(0.0, min(width_p, strikes["short_put"] - sT))      # buyer payoff
    call_spread_payoff = max(0.0, min(width_c, sT - strikes["short_call"]))     # buyer payoff
    return float(put_spread_payoff + call_spread_payoff)  # this is *loss* to seller

def _price_debit_spread_black(s0: float, r: float, q: float, t: float, iv: float,
                              right: Literal["C","P"], long_k: float, short_k: float) -> float:
    if right.upper().startswith("C"):
        long_leg  = _black_call_price(s0, long_k, r, q, t, iv)
        short_leg = _black_call_price(s0, short_k, r, q, t, iv)
    else:
        long_leg  = _black_put_price(s0, long_k, r, q, t, iv)
        short_leg = _black_put_price(s0, short_k, r, q, t, iv)
    return float(long_leg - short_leg)

def _realized_debit_payoff_buyer(sT: float, right: str, long_k: float, short_k: float) -> float:
    width = abs(short_k - long_k)
    if right.upper().startswith("C"):
        return float(max(0.0, min(width, sT - long_k)))
    else:
        return float(max(0.0, min(width, long_k - sT)))

@dataclass
class Trade:
    date: pd.Timestamp
    t_years: float
    s0: float
    iv: float
    credit: float = 0.0
    debit: float  = 0.0
    qty_condor: int = 0
    qty_debit:  int = 0
    short_put: float = 0.0
    long_put: float = 0.0
    short_call: float = 0.0
    long_call: float = 0.0
    d_right: str = ""
    d_long: float = 0.0
    d_short: float = 0.0
    sT: float = 0.0
    pnl: float = 0.0

def backtest_ticker(
    ticker: str,
    start: str,
    end: str,
    strategy: Literal["condor","debit","both"] = "condor",
    dte_days: int = 21,
    step_days: int = 5,
    p_tail: float = 0.20,
    wing_width: float = 5.0,
    slip_bps: float = 20.0,
    fee_per_contract: float = 0.65,
    risk_per_trade: float = 0.01,
    r: float = 0.02,
    q: float = 0.0,
) -> Dict[str, object]:
    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return dict(trades=[], equity=pd.Series(dtype=float), report=dict(
            ticker=ticker, pnl=0.0, trades=0, ret_pct=0.0, wins=0, max_dd=0.0
        ))

    close = hist["Close"].dropna().astype(float)
    iv_est = _rolling_iv(close).bfill()
    dates = close.index.to_list()

    trades: List[Trade] = []
    cash = 100_000.0
    equity = []

    condor_legs = 4
    debit_legs  = 2
    slip = lambda px: float(px) * (slip_bps / 10_000.0)

    for i, d in enumerate(dates):
        if i % step_days != 0 or i + dte_days >= len(dates):
            equity.append(cash)
            continue

        s0 = float(close.iloc[i])
        sT = float(close.iloc[i + dte_days])
        iv = float(iv_est.iloc[i])
        t  = dte_days / 365.0

        trade = Trade(date=pd.Timestamp(d), t_years=t, s0=s0, iv=iv, sT=sT)

        # ---------- Condor leg ----------
        if strategy in ("condor", "both"):
            strikes = _build_condor_from_range(s0, r, q, iv, t, p_tail, wing_width)

            credit_call = price_credit_spread(s0, r, q, t, iv, right="C",
                                              short_k=strikes["short_call"], long_k=strikes["long_call"])
            credit_put  = price_credit_spread(s0, r, q, t, iv, right="P",
                                              short_k=strikes["short_put"], long_k=strikes["long_put"])
            credit_mid  = float(credit_call + credit_put)
            credit_net  = max(0.0, credit_mid - slip(credit_mid))

            # Risk per *contract* in dollars
            risk_per_condor = max(1e-6, (wing_width - credit_net) * 100.0)
            alloc = risk_per_trade * cash
            qty_c = max(0, int(alloc / risk_per_condor))

            # Entry fees only (no double counting)
            cash -= fee_per_contract * condor_legs * qty_c

            # Realized at expiry
            loss_buyer = _realized_condor_loss_buyer(sT, strikes)          # in $ per share
            pnl_c = (credit_net * 100.0 - loss_buyer * 100.0) * qty_c      # in dollars

            trade.credit = credit_net
            trade.qty_condor = qty_c
            trade.short_put  = strikes["short_put"]
            trade.long_put   = strikes["long_put"]
            trade.short_call = strikes["short_call"]
            trade.long_call  = strikes["long_call"]
            trade.pnl += pnl_c

        # ---------- Debit leg (optional) ----------
        if strategy in ("debit", "both"):
            sma10 = close.iloc[max(0, i-10):i+1].mean()
            sma30 = close.iloc[max(0, i-30):i+1].mean()
            if sma10 >= sma30:
                right = "C"; long_k = s0; short_k = s0 + wing_width
            else:
                right = "P"; long_k = s0; short_k = max(1e-6, s0 - wing_width)

            debit_mid = _price_debit_spread_black(s0, r, q, t, iv, right, long_k, short_k)
            debit_net = debit_mid + slip(debit_mid)

            alloc = risk_per_trade * cash
            qty_d = max(0, int(alloc / max(1e-6, debit_net * 100.0)))

            # Pay entry cost (debit + fees)
            cash -= (debit_net * 100.0 + fee_per_contract * debit_legs) * qty_d

            payoff = _realized_debit_payoff_buyer(sT, right, long_k, short_k)  # $ per share
            pnl_d  = payoff * 100.0 * qty_d

            trade.debit   = debit_net
            trade.qty_debit = qty_d
            trade.d_right = right
            trade.d_long  = long_k
            trade.d_short = short_k
            trade.pnl += pnl_d

        cash += trade.pnl
        trades.append(trade)
        equity.append(cash)

    eq = pd.Series(equity, index=close.index[:len(equity)], name=ticker)
    pnl = float(cash - 100_000.0)
    wins = sum(1 for tr in trades if tr.pnl > 0)
    ret_pct = (pnl / 100_000.0) * 100.0
    cum = eq.cummax()
    dd = (eq - cum) / cum.replace(0, np.nan)
    max_dd = float(dd.min()) if len(dd) else 0.0

    return dict(
        trades=trades,
        equity=eq,
        report=dict(ticker=ticker, pnl=pnl, trades=len(trades), wins=wins, ret_pct=ret_pct, max_dd=max_dd)
    )

def backtest_universe(
    tickers: List[str],
    start: str,
    end: str,
    strategy: Literal["condor","debit","both"] = "condor",
    dte_days: int = 21,
    step_days: int = 5,
    p_tail: float = 0.20,
    wing_width: float = 5.0,
    slip_bps: float = 20.0,
    fee_per_contract: float = 0.65,
    risk_per_trade: float = 0.01,
) -> Dict[str, object]:
    rows = []
    eq_map = {}
    for tk in tickers:
        try:
            res = backtest_ticker(
                tk, start, end, strategy=strategy,
                dte_days=dte_days, step_days=step_days,
                p_tail=p_tail, wing_width=wing_width,
                slip_bps=slip_bps, fee_per_contract=fee_per_contract,
                risk_per_trade=risk_per_trade
            )
            rows.append(res["report"])
            eq_map[tk] = res["equity"]
        except Exception as e:
            rows.append(dict(ticker=tk, pnl=0.0, trades=0, wins=0, ret_pct=0.0, max_dd=0.0, error=str(e)))

    summaries = pd.DataFrame(rows)
    if eq_map:
        aligned = pd.concat(eq_map.values(), axis=1).ffill()
        start_vals = aligned.iloc[0]
        pnl_mat = aligned.subtract(start_vals, axis=1)
        portfolio = pnl_mat.sum(axis=1)
    else:
        portfolio = pd.Series(dtype=float)

    return dict(summaries=summaries, equity=portfolio, portfolio=summaries.assign(total_pnl=summaries["pnl"].sum()))
