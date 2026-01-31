import types, pandas as pd
import aoe.universe.select as sel_mod
from aoe.universe import select as sel


class DummyTicker:
    def __init__(self, adv, opt):
        self._adv, self._opt = adv, opt
        self.fast_info = {"lastPrice": 100}

    def history(self, period="1mo"):
        df = pd.DataFrame({"Close": [100], "Volume": [self._adv / 100]})
        return pd.concat([df] * 20)

    def option_chain(self):
        base = pd.DataFrame(
            {
                "strike": [90, 100, 110],
                "lastPrice": [2, 3, 4],
                "volume": [self._opt / 30] * 3,
                "bid": [1.9, 2.9, 3.9],
                "ask": [2.1, 3.1, 4.1],
                "impliedVolatility": [0.25] * 3,
                "openInterest": [1000] * 3,
            }
        )
        return types.SimpleNamespace(calls=base, puts=base)


def test_score_universe(monkeypatch):
    def fake_Ticker(tk):
        return DummyTicker(adv=1e9, opt=2e8)

    monkeypatch.setattr(sel_mod, "yf", types.SimpleNamespace(Ticker=fake_Ticker))

    df = sel.score_universe(top_n=5)
    assert df.shape[0] == 5
    assert df.iloc[0]["score"] >= df.iloc[-1]["score"]
