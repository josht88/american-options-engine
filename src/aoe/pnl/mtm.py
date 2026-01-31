"""
Utility to mark-to-market the spreads sitting in data/pnl.csv
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

LEDGER = Path("data/pnl.csv")
COLS   = [
    "open_ts", "ticker", "expiry", "right", "long_k", "short_k",
    "qty", "open_px", "close_px"
]


def _mid_price(tk: yf.Ticker, expiry: str, right: str, strike: float) -> float:
    """Return mid of bid/ask for a single option leg."""
    ch  = tk.option_chain(expiry)
    df  = ch.calls if right == "C" else ch.puts
    row = df[df["strike"] == strike]
    if row.empty:
        return np.nan
    return float((row["bid"].iloc[0] + row["ask"].iloc[0]) / 2)


def snapshot_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(LEDGER, parse_dates=["open_ts", "close_ts"], na_values=[""])
    if "close_px" not in df.columns:
        df["close_px"] = np.nan
    return df


def mark_to_market(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `mtm_px` and `pnl` columns."""
    today = dt.date.today()
    rows = []
    for _, row in df.iterrows():
        tk = yf.Ticker(row.ticker)
        long_mid  = _mid_price(tk, row.expiry, row.right,  row.long_k)
        short_mid = _mid_price(tk, row.expiry, row.right, row.short_k)
        if np.isnan(long_mid) or np.isnan(short_mid):
            continue
        net_mid = long_mid - short_mid
        pnl = (net_mid - row.open_px) * row.qty * 100
        rows.append({**row, "mtm_px": net_mid, "pnl": pnl})
    return pd.DataFrame(rows)
