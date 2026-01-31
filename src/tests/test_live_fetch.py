"""
Offline test: monkey-patch yfinance.Ticker so no web call is made.
"""

import types, datetime as dt, pandas as pd, numpy as np
import pytest, builtins

from aoe.data import yf as yf_mod


class DummyTicker:
    def __init__(self, *a, **k):
        self.fast_info = {"lastPrice": 100.0}
        self._make_options()

    def _make_options(self):
        exp = dt.date.today() + dt.timedelta(days=30)
        strikes = np.array([90, 95, 100, 105, 110])
        iv      = np.array([0.32, 0.30, 0.28, 0.29, 0.31])
        df = pd.DataFrame({
            "contractSymbol": ["A"]*5,
            "strike": strikes,
            "impliedVolatility": iv,
            "expiration": [exp]*5
        })
        self._chain = types.SimpleNamespace(calls=df, puts=df)

        # empty calendar
        self.calendar = pd.DataFrame()

    def option_chain(self):
        return self._chain


@pytest.fixture(autouse=True)
def patch_yf(monkeypatch):
    monkeypatch.setattr(yf_mod, "yf", types.SimpleNamespace(Ticker=DummyTicker))


def test_snapshot_structure():
    snap = yf_mod.get_snapshot("AAPL")
    assert "spot" in snap and "iv_surface" in snap
    df = snap["iv_surface"]
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 5)
