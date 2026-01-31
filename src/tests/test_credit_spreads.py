import numpy as np
import pandas as pd
from aoe.strategy.credit import (
    build_iron_condor_from_chain,
    price_credit_spread,
    ev_iron_condor_logn,
)

# Dummy model that uses Black for price_euro (keeps test offline/stable)
class BlackModel:
    def price_euro(self, s0, k, r, q, t, opt_type):
        from aoe.strategy.credit import _black_call_price, _black_put_price
        vol = 0.25
        if opt_type == "call":
            return _black_call_price(s0, k, r, q, vol, t)
        else:
            return _black_put_price(s0, k, r, q, vol, t)

def _make_chain(spot=100, vol=0.25, r=0.02, q=0.0, t=0.5):
    from aoe.strategy.credit import _black_call_price, _black_put_price
    strikes = np.arange(80, 121, 5)
    calls_mid = [_black_call_price(spot, k, r, q, vol, t) for k in strikes]
    puts_mid  = [_black_put_price (spot, k, r, q, vol, t) for k in strikes]
    df_calls = pd.DataFrame(dict(
        strike=strikes, bid=np.array(calls_mid)*0.99, ask=np.array(calls_mid)*1.01, mid=calls_mid,
        impliedVolatility=vol, openInterest=1000
    ))
    df_puts  = pd.DataFrame(dict(
        strike=strikes, bid=np.array(puts_mid)*0.99, ask=np.array(puts_mid)*1.01, mid=puts_mid,
        impliedVolatility=vol, openInterest=1000
    ))
    return df_calls, df_puts

def test_price_credit_spread_sign():
    mdl = BlackModel()
    s0, r, q, t = 100, 0.02, 0.0, 0.5
    # bear CALL credit: short 100, long 110  -> credit should be positive
    val = price_credit_spread(mdl, s0, 100, 110, r, q, t, "C")
    assert val > 0

def test_build_and_ev_condor():
    s0, r, q, t, vol = 100, 0.02, 0.0, 0.5, 0.25
    calls, puts = _make_chain(s0, vol, r, q, t)

    condor_df = build_iron_condor_from_chain(
        "TEST", s0, r, q, t, calls, puts, iv_atm=vol, p_tail=0.20, wing_width=5.0, expiry="2099-01-19"
    )
    assert not condor_df.empty
    row = condor_df.iloc[0]
    assert row["credit_mid"] > 0
    assert row["max_loss"] > 0

    ev = ev_iron_condor_logn(
        s0, r, q, vol, t,
        row["long_put"], row["short_put"],
        row["short_call"], row["long_call"],
        row["credit_mid"], n_paths=50_000, seed=42
    )
    # For a symmetric condor around 20/80% quantiles, EV to seller should be >= 0
    assert ev >= 0.0
