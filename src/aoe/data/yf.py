"""
Light wrapper around yfinance.

get_snapshot(ticker) -> dict
    {
        "spot": 187.32,
        "iv_surface": DataFrame(index=maturity_yrs, columns=strikes, data=iv),
        "earnings":  datetime.date or None
    }
"""

import datetime as dt
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------- helpers
def _to_years(date_obj: dt.date) -> float:
    """Convert a calendar date to year-fraction from today."""
    return (date_obj - dt.date.today()).days / 365.0


# ---------------------------------------------------------------------- main API
def get_snapshot(ticker: str) -> Dict:
    tk = yf.Ticker(ticker)

    # -------- spot -------------------------------------------------------------
    spot = tk.fast_info["lastPrice"]

    # -------- earnings date ----------------------------------------------------
    cal = tk.calendar
    earnings = None

    # yfinance sometimes returns a DataFrame, sometimes a dict
    if isinstance(cal, pd.DataFrame):
        if not cal.empty and "Earnings Date" in cal.index:
            earnings = pd.to_datetime(cal.loc["Earnings Date"][0]).date()
    elif isinstance(cal, dict) and "Earnings Date" in cal:
        raw = cal["Earnings Date"][0]
        earnings = pd.to_datetime(raw).date()

    # -------- IV surface (ATM ± 10 %) -----------------------------------------
    chain = tk.option_chain()

    # yfinance ≤ 0.2.4 had an "expiration" column, ≥ 0.2.5 removed it.
    if "expiration" in chain.calls.columns:
        maturity = pd.to_datetime(chain.calls["expiration"].iloc[0]).date()
        calls = chain.calls[chain.calls["expiration"] == maturity]
    else:
        # fall back to tk.options list
        maturity = pd.to_datetime(tk.options[0]).date()
        calls = chain.calls.copy()

    # extract strikes & vols
    strikes = calls["strike"].to_numpy()
    iv = calls["impliedVolatility"].to_numpy()

    # keep ATM ±10 % band
    atm = spot
    mask = (strikes >= 0.9 * atm) & (strikes <= 1.1 * atm)
    strikes = strikes[mask]
    iv = iv[mask]

    iv_df = pd.DataFrame(
        [iv],
        index=[_to_years(maturity)],
        columns=strikes,
    )

    return dict(spot=spot, iv_surface=iv_df, earnings=earnings)
