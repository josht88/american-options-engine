"""
aoe.models.heston
-----------------
Lightweight HestonModel wrapper that stores parameters and prices European
vanillas via QuantLib's AnalyticHestonEngine (FFT formulation).

Public API
----------
class HestonModel
    .price_euro(s0, k, r, q, t, option_type="call") -> float
"""

import QuantLib as ql


class HestonModel:
    def __init__(self, kappa, theta, sigma, rho, v0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0

    # --- internal helper -------------------------------------------------
    def _process(self, s0, r, q):
        today = ql.Date.todaysDate()
        dc = ql.Actual365Fixed()
        ql.Settings.instance().evaluationDate = today

        spot = ql.SimpleQuote(s0)
        r_ts = ql.FlatForward(today, r, dc)
        q_ts = ql.FlatForward(today, q, dc)

        return ql.HestonProcess(
            ql.YieldTermStructureHandle(r_ts),
            ql.YieldTermStructureHandle(q_ts),
            ql.QuoteHandle(spot),
            self.v0,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
        )

    # --- public method ---------------------------------------------------
    def price_euro(self, s0, k, r, q, t, option_type="call"):
        payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put,
        float(k),          # cast avoids NumPy int → QuantLib TypeError
    )

        expiry = ql.Date.todaysDate() + int(t * 365)
        exercise = ql.EuropeanExercise(expiry)
        option = ql.VanillaOption(payoff, exercise)

        engine = ql.AnalyticHestonEngine(ql.HestonModel(self._process(s0, r, q)))
        option.setPricingEngine(engine)
        return option.NPV()
