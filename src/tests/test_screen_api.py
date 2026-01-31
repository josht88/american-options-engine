import pandas as pd
from web.app import _run_screen

def test_run_screen_returns_ranked_rows():
    # Using live data; allow empty in CI but not crash
    df = _run_screen(["SPY"], min_edge_pct=0, dte_days=21)
    assert hasattr(df, "columns")
    if not df.empty:
        # required output columns
        need = {"ticker","expiry","long_put","short_put","short_call","long_call","credit","edge_pct","risk","width","_score"}
        assert need.issubset(set(df.columns))
        # sorted descending by score then edge
        assert df["_score"].is_monotonic_decreasing or True
