from types import SimpleNamespace
import json
import os
import pandas as pd

import scripts.run_backtest_yf as runner

def test_runner_equity_only_metrics(monkeypatch, tmp_path):
    # --- stub args (no config; write into tmp outroot) ---
    ns = SimpleNamespace(
        config=None, tickers=["AAA","BBB"], start="2025-01-01", end="2025-01-10",
        strategy="both", dte=21, step=5, p_tail=0.20, wing=5.0,
        slip_bps=20.0, fee=0.65, risk_frac=0.01, save_trades=False,
        outroot=str(tmp_path)
    )
    monkeypatch.setattr(runner, "_parse_args", lambda: ns)
    monkeypatch.setattr(runner, "_merge_args_with_config", lambda a: (a, None))

    # --- stub backtest_universe to return small, deterministic objects ---
    def fake_universe(tickers, start, end, **params):
        # summaries (per-ticker)
        summaries = pd.DataFrame({
            "ticker": tickers,
            "pnl": [1000.0, 500.0],
            "trades": [2, 1],
            "wins": [2, 1],
            "ret_pct": [1.0, 0.5],
            "max_dd": [-0.01, -0.02],
        })
        # equity: DAILY P&L so the runner must cumsum + base
        pnl_series = pd.Series([0.0, 800.0, -200.0, 900.0], index=pd.date_range("2025-01-01", periods=4, freq="B"))
        pnl_series.index.name = "date"
        return {"summaries": summaries, "equity": pnl_series}

    # --- stub backtest_ticker to yield two "trade objects" (not used in metrics) ---
    def fake_ticker(tk, start, end, **params):
        t1 = SimpleNamespace(date="2025-01-02", realised=400.0, ticker=tk)
        t2 = SimpleNamespace(date="2025-01-05", realised=600.0, ticker=tk)
        return {"trades": [t1, t2]}

    monkeypatch.setattr(runner, "backtest_universe", fake_universe)
    monkeypatch.setattr(runner, "backtest_ticker", fake_ticker)

    # --- run ---
    runner.main()

    # --- assert outputs exist ---
    outdir = os.path.join(ns.outroot, f"{ns.strategy}_{ns.start}_{ns.end}")
    assert os.path.isdir(outdir)
    assert os.path.exists(os.path.join(outdir, "summaries.csv"))
    assert os.path.exists(os.path.join(outdir, "equity.csv"))
    assert os.path.exists(os.path.join(outdir, "trades.csv"))
    assert os.path.exists(os.path.join(outdir, "metrics.json"))

    # --- metrics are equity-derived only => 'trades' field should be 0 ---
    with open(os.path.join(outdir, "metrics.json")) as f:
        m = json.load(f)
    assert "CAGR" in m and "Sharpe" in m and "MaxDD" in m
    # because we passed an empty trades df to summarize()
    assert m.get("trades", 0) == 0
