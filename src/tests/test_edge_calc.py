from aoe.models.heston import HestonModel
from aoe.signals.eps import Shock
from aoe.screen.edge_calc import evaluate_trade


def test_edge_positive():
    """Higher implied vol → model price > market → trade."""
    model = HestonModel(2.0, 0.05, 0.4, -0.6, 0.04)
    contract = dict(s0=100, k=100, r=0.02, q=0.0, t=0.5, option_type="call")

    # shock that RAISES vol (e.g., −5 % earnings miss)
    shock = Shock(mu_shift=0, sigma_mult=1.2, jump_lambda=0, confidence=1, decay=5)

    mid_price = 5.0                       # lower mid so model > market
    res = evaluate_trade(model, contract, [shock], mid_price)

    assert res["edge_pct"] > 0
    assert res["trade"] is True



def test_edge_negative():
    """Vol crush → model price < market → no trade."""
    model = HestonModel(2.0, 0.05, 0.4, -0.6, 0.04)
    contract = dict(s0=100, k=100, r=0.02, q=0.0, t=0.5, option_type="call")

    shock = Shock(mu_shift=0, sigma_mult=0.6, jump_lambda=0, confidence=1, decay=5)

    mid_price = 10.0
    res = evaluate_trade(model, contract, [shock], mid_price)

    assert res["edge_pct"] < 0
    assert res["trade"] is False
