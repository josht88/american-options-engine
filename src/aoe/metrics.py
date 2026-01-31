# src/aoe/metrics.py
from __future__ import annotations
import numpy as np, pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    peaks = equity.cummax()
    dd = (equity/peaks - 1.0).min()
    return float(dd)

def cagr(equity: pd.Series, periods_per_year=252) -> float:
    if len(equity) < 2: return 0.0
    n_years = len(equity)/periods_per_year
    return float((equity.iloc[-1]/equity.iloc[0])**(1/n_years) - 1)

def sharpe(returns: pd.Series, rf=0.0, periods_per_year=252) -> float:
    ex = returns - rf/periods_per_year
    if ex.std(ddof=0) == 0: return 0.0
    return float(np.sqrt(periods_per_year) * ex.mean()/ex.std(ddof=0))

def sortino(returns: pd.Series, rf=0.0, periods_per_year=252) -> float:
    ex = returns - rf/periods_per_year
    downside = ex[ex < 0]
    denom = downside.std(ddof=0)
    if denom == 0 or np.isnan(denom): return 0.0
    return float(np.sqrt(periods_per_year) * ex.mean()/denom)

def trade_stats(trades: pd.DataFrame) -> dict:
    # Expect trades with columns: pnl, realised, entry_time, exit_time
    pnl = trades["pnl"] if "pnl" in trades else trades["realised"]
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    hit = len(wins) / max(1, len(pnl))
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = pnl.mean()
    return dict(hit_rate=hit, avg_win=float(avg_win or 0.0),
                avg_loss=float(avg_loss or 0.0), expectancy=float(expectancy or 0.0),
                trades=int(len(pnl)))

def summarize(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    # equity columns: date, equity
    eq = equity.set_index(equity.columns[0])[equity.columns[1]].astype(float)
    rets = eq.pct_change().dropna()
    stats = {
        "CAGR": cagr(eq),
        "Sharpe": sharpe(rets),
        "Sortino": sortino(rets),
        "MaxDD": max_drawdown(eq),
        "MAR": (cagr(eq) / abs(max_drawdown(eq)) if max_drawdown(eq) < 0 else 0.0),
        "Vol": float(rets.std(ddof=0) * np.sqrt(252)),
    }
    stats.update(trade_stats(trades))
    return stats
