import pandas as pd
from web.app import _resolve_spot, _scan_high_ev_contracts_compat

def test_spot_resolves_from_chain():
    ch = pd.DataFrame({
        "expiry":["2025-10-03"]*3,
        "right":["P","P","P"],
        "strike":[95,100,105],
        "bid":[0.8, 2.4, 5.2],
        "ask":[0.95,2.6, 5.4],
        "mid":[0.875,2.5,5.3],
        "spot":[100,100,100]  # allow inference
    })
    s = _resolve_spot("DUMMY", mdl=None, singles=ch, snap={"spot": None})
    assert s == 100

def test_scanner_fallback_runs(monkeypatch):
    # If project scanner explodes, fallback to yfinance path returns a DataFrame (may be empty in CI).
    # We don't assert non-empty (no network guarantee), but assert it returns a DataFrame.
    out = _scan_high_ev_contracts_compat("SPY", p_tail=0.20, dte_days=21)
    assert hasattr(out, "columns")
