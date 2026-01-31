import types, pandas as pd
from scripts import screen_day as cli


def test_screen_day(monkeypatch, capsys):
    # ----- patch universe scorer to return one ticker -----
    monkeypatch.setattr(cli, "score_universe",
        lambda n: pd.DataFrame({"ticker": ["TEST"]}))
    # ----- dummy snapshot / model -------------
    snap = {"spot": 100}
    monkeypatch.setattr(cli, "ensure_model", lambda tk: snap)

    class DummyModel:
        spot = 100
        def price_euro(self, *a, **k): return 6.0
    monkeypatch.setattr(cli, "load_model", lambda tk, s: DummyModel())

    # ----- dummy chain + spreads -------------
    spread_row = pd.Series(dict(
        ticker="TEST", expiry="2099-01-19", right="C",
        long_k=95, short_k=105, width=10,
        net_mid=3.0, max_gain=7.0, max_loss=3.0
    ))
    monkeypatch.setattr(cli, "scan_high_ev_contracts",
        lambda tk: pd.DataFrame({"dummy":[]}))
    monkeypatch.setattr(cli, "build_range_spreads",
        lambda df, mdl: pd.DataFrame([spread_row]))

    # ----- stub execute_order -----
    fills = []
    monkeypatch.setattr(cli, "execute_order", lambda order: fills.append(order))

    # provide bullish fair-value so edge passes
    monkeypatch.setattr(cli, "price_debit_spread", lambda *a, **k: 3.5)

    cli.EDGE_THRES = 0.10
    cli.BANKROLL   = 50_000

    cli.screen_day()


    # one fill expected
    assert fills and fills[0]["qty"] > 0
    out = capsys.readouterr().out
    assert "Filled orders" in out
