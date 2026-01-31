"""
Build high-EV debit spreads (“AA/AK” range).

Inputs
------
single_df : DataFrame from scan_high_ev_contracts
model      : calibrated model with .spot

Returns    : DataFrame with spread candidates
"""

import math, datetime as dt
import pandas as pd


def _sigma_move(spot, sigma, t, side: str):
    factor = math.exp(sigma * math.sqrt(t))
    return spot * factor if side == "C" else spot / factor


def build_range_spreads(single_df: pd.DataFrame, model) -> pd.DataFrame:
    if single_df.empty:
        return pd.DataFrame()

    today = dt.date.today()
    out = []

    for (tk, exp, side), grp in single_df.groupby(["ticker", "expiry", "right"]):
        # pick long leg closest to |Δ| = 0.70
        long_row = grp.iloc[(grp["delta"] - 0.70).abs().argmin()]

        spot = model.spot
        t = (dt.datetime.strptime(exp, "%Y-%m-%d").date() - today).days / 365.0
        sigma = long_row["iv"]
        target = _sigma_move(spot, sigma, t, side)

        # candidate short strikes on proper wing
        wing = grp["strike"] > long_row["strike"] if side == "C" else grp["strike"] < long_row["strike"]
        strike_series = grp.loc[wing, "strike"].sort_values()
        if strike_series.empty:
            continue

        short_k = strike_series.iloc[(strike_series - target).abs().argmin()]
        short_row = grp.loc[grp["strike"] == short_k].iloc[0]

        width = abs(short_k - long_row["strike"])
        net_mid = long_row["mid"] - short_row["mid"]

        out.append(
            dict(
                ticker=tk,
                expiry=exp,
                right=side,
                long_k=long_row["strike"],
                short_k=short_k,
                width=width,
                net_mid=net_mid,
                max_gain=width - net_mid,
                max_loss=net_mid,
                delta=long_row["delta"] - short_row["delta"],
            )
        )

    return pd.DataFrame(out)
