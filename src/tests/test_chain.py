import types, pandas as pd, datetime as dt
from aoe.data.chain import scan_high_ev_contracts
import aoe.data.chain as ch_mod


class DummyChainTicker:
    def __init__(self):
        self.fast_info = {"lastPrice": 100}
        # expiry 30 days from today ⇒ dte within 7–45 filter
        self.options = [
            (dt.date.today() + dt.timedelta(days=30)).strftime("%Y-%m-%d")
        ]

    def option_chain(self, *args, **kwargs):
        strikes = [95, 100, 105]                # near-ATM to ensure 0.6-0.8 delta
        df = pd.DataFrame(
            dict(
                strike=strikes,
                impliedVolatility=[0.28] * 3,
                bid=[4.95, 5.95, 4.95],         # ~2 % spreads
                ask=[5.05, 6.05, 5.05],
                openInterest=[800] * 3,
            )
        )
        return types.SimpleNamespace(calls=df, puts=df)


def test_chain_scanner(monkeypatch):
    monkeypatch.setattr(
        ch_mod, "yf", types.SimpleNamespace(Ticker=lambda tk: DummyChainTicker())
    )

    df = scan_high_ev_contracts("AAPL")
    assert not df.empty
    assert df["delta"].between(0.60, 0.80).all()
    assert (df["oi"] >= 500).all()
