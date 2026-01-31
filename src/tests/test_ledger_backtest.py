import pandas as pd
import tempfile, pathlib, datetime as dt
from web.app import _ensure_ledger, _equity_from_ledger, _equity_from_backtest

def test_equity_from_ledger_unrealised_and_realised():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "pnl.csv"
        _ = _ensure_ledger(str(p))
        df = pd.DataFrame([{
            "ts": dt.datetime.now().isoformat(),
            "ticker": "T", "expiry":"2099-01-19", "right":"C",
            "long_k":95,"short_k":105,"qty":1,"price":2.0,"mark":1.5,"realised":0.0,"tag":"OPEN"
        }])
        df.to_csv(p, index=False)
        eq = _equity_from_ledger(str(p))
        assert not eq.empty
        # -0.5 unrealised at least appears in the only row
        assert abs(eq["equity"].iloc[-1] + 0.5) < 1e-6

def test_equity_from_backtest_normalizes():
    bt = {"equity": pd.DataFrame({"date":[dt.date(2024,1,1), dt.date(2024,1,2)], "equity":[0,100]})}
    out = _equity_from_backtest(bt)
    assert list(out.columns) == ["date","equity"]
    assert len(out) == 2
