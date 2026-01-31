"""
Universe scorer – keeps the 24 blue-chip sandbox but ranks them each day.

Score = 0.4 * ADV_USD + 0.4 * OPT_USD – 0.2 * MEDIAN_SPREAD_PCT
Scaled 0-5 per factor. Retains top N (default 10).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from . import BLUE_CHIPS


def _scale(arr, hi, lo=0):
    """Min-max to 0–5, protect div/0."""
    return np.clip(5 * (arr - lo) / (hi - lo + 1e-9), 0, 5)


def fetch_metrics(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1mo")
    adv_usd = (hist["Close"] * hist["Volume"]).mean()  # ADV $

    try:
        chain = tk.option_chain()
        dollar = (chain.calls["lastPrice"] * chain.calls["volume"]).sum()
        dollar += (chain.puts["lastPrice"] * chain.puts["volume"]).sum()
    except Exception:
        dollar = 0.0

    try:
        med_spread = (chain.calls["ask"] - chain.calls["bid"]).abs().median()
        med_mid = ((chain.calls["ask"] + chain.calls["bid"]) / 2).median()
        spread_pct = med_spread / med_mid if med_mid else 1.0
    except Exception:
        spread_pct = 1.0

    return dict(ticker=ticker, adv_usd=adv_usd, opt_usd=dollar, spread_pct=spread_pct)


def score_universe(top_n=10):
    rows = [fetch_metrics(tk) for tk in BLUE_CHIPS]
    df = pd.DataFrame(rows).fillna(0.0)

    df["adv_score"] = _scale(np.log10(df["adv_usd"] + 1), hi=10, lo=6)
    df["opt_score"] = _scale(np.log10(df["opt_usd"] + 1), hi=9, lo=5)
    df["spr_score"] = 5 - _scale(df["spread_pct"], hi=0.08, lo=0.005)

    df["score"] = 0.4 * df["adv_score"] + 0.4 * df["opt_score"] + 0.2 * df["spr_score"]

    return df.sort_values("score", ascending=False).head(top_n)
