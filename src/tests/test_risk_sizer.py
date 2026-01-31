import numpy as np
import pandas as pd
from aoe.risk.var import rolling_var
from aoe.risk.sizer import contracts_to_buy

def test_rolling_var():
    np.random.seed(0)
    rets = pd.Series(np.random.normal(0, 0.02, 100))
    var = rolling_var(rets, window=20, alpha=0.95)

    # expected value = empirical 5-percentile of last 20 returns
    expected = np.quantile(rets[-20:], 0.05)

    assert abs(var.iloc[-1] - expected) < 1e-10
    assert var.iloc[-1] < 0                    # VAR is a loss


def test_contracts_to_buy():
    edge_pct = 0.10          # 10 % edge
    delta = 0.5
    S = 100
    opt_price = 2.0
    var_1d = 0.02            # 2 % daily move
    n = contracts_to_buy(edge_pct, delta, S, opt_price, var_1d,
                         risk_cap_usd=1_000,
                         open_interest=2_000,
                         oi_limit=0.1)
    assert n > 0             # should allocate some contracts
    assert n <= 200          # respects 10 % OI cap
