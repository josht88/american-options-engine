import datetime as dt
from web.app import run_condor_diagnostic

def test_run_condor_diagnostic_runs_small_window():
    start = dt.date.today() - dt.timedelta(days=10)
    end   = dt.date.today()
    out = run_condor_diagnostic(
        tickers=["SPY"],
        start_date=start,
        end_date=end,
        dte_days=14,
        wings=[5],
        min_edge_pct=0,
        limit_days=5,
    )
    # It may be empty in CI (network/etc.), but shape should be right if present.
    assert hasattr(out, "columns")
    if not out.empty:
        need = {"date","ticker","chain_rows","condor_rows","pass_rows","reason"}
        assert need.issubset(set(out.columns))
