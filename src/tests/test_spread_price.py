import math
from aoe.pricing.spread import price_debit_spread


class DummyModel:
    def price_euro(self, s0, k, r, q, t, option_type):
        # Black-Scholes closed form, σ = 0.2 fixed
        import numpy as np, scipy.stats as st
        sigma = 0.2
        d1 = (math.log(s0 / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        if option_type == "call":
            price = s0 * np.exp(-q * t) * st.norm.cdf(d1) - k * math.exp(-r * t) * st.norm.cdf(d2)
        else:
            price = k * math.exp(-r * t) * st.norm.cdf(-d2) - s0 * np.exp(-q * t) * st.norm.cdf(-d1)
        return price

    price_amer = price_euro  # reuse dummy


def test_spread_pricer():
    mdl = DummyModel()
    px = price_debit_spread(mdl, 100, 95, 105, 0.02, 0.0, 0.5, "call", "euro")
    # long ITM call minus short OTM call must be positive and < width
    assert 0 < px < 10
