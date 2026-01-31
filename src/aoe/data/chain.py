"""
Chain scanner – returns only very high-EV “hand” candidates:

* DTE 7-45
* |delta| 0.60-0.80  (AA/AK zone)
* bid-ask % <= 3 %
* open interest >= 500
"""

import datetime as dt
import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


def _black_delta(spot, strike, vol, t, r=0.0, q=0.0, right="C"):
    """Approximate delta (spot convention)."""
    if vol <= 0 or t <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol ** 2) * t) / (
        vol * math.sqrt(t)
    )
    delta = math.exp(-q * t) * norm.cdf(d1)
    return delta if right == "C" else delta - math.exp(-q * t)


def scan_high_ev_contracts(ticker):
    tk = yf.Ticker(ticker)
    today = dt.date.today()

    out = []
    for expiry in tk.options[:3]:  # nearest 3 expiries
        exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if not 7 <= dte <= 45:
            continue

        ch = tk.option_chain(expiry)
        for side, df in [("C", ch.calls), ("P", ch.puts)]:
            df = df.copy()
            df["mid"] = (df["bid"] + df["ask"]) / 2
            df["spread_pct"] = (df["ask"] - df["bid"]).abs() / df["mid"]
            df["delta"] = df.apply(
                lambda r: abs(
                    _black_delta(
                        tk.fast_info["lastPrice"],
                        r["strike"],
                        r["impliedVolatility"],
                        dte / 365.0,
                        right=side,
                    )
                ),
                axis=1,
            )

            sel = df[
                (df["delta"].between(0.60, 0.80))
                & (df["openInterest"] >= 500)
                & (df["spread_pct"] <= 0.03)
            ]
            for _, row in sel.iterrows():
                out.append(
                    dict(
                        ticker=ticker,
                        expiry=expiry,
                        strike=row["strike"],
                        right=side,
                        mid=row["mid"],
                        delta=row["delta"],
                        iv=row["impliedVolatility"],
                        oi=row["openInterest"],
                    )
                )

    return pd.DataFrame(out)
