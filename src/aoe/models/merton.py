"""
aoe.models.merton
-----------------
Simple Merton Jump–Diffusion model wrapper:

dS/S = (r - q - λκ) dt + σ dW + d(Σ_{i=1}^{N_t}(Y_i - 1))
with Y ~ log-N(μ_J, σ_J^2), jump intensity λ, κ = E[Y - 1].

Provides analytic European pricing via QuantLib's
`Merton76VanillaEngine`.
"""

import QuantLib as ql


class MertonModel:
    def __init__(self, sigma, lam, mu_j, sigma_j):
        self.sigma = sigma
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    # ---------- analytic European price ----------
    def price_euro(self, s0, k, r, q, t, option_type="call"):
        today = ql.Date.todaysDate()
        dc = ql.Actual365Fixed()
        ql.Settings.instance().evaluationDate = today

        spot = ql.SimpleQuote(s0)
        flat_r = ql.FlatForward(today, r, dc)
        flat_q = ql.FlatForward(today, q, dc)
        flat_vol = ql.BlackConstantVol(today, ql.NullCalendar(), self.sigma, dc)

        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(spot),
            ql.YieldTermStructureHandle(flat_q),
            ql.YieldTermStructureHandle(flat_r),
            ql.BlackVolTermStructureHandle(flat_vol),
        )

        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type == "call" else ql.Option.Put, float(k)
        )
        expiry = today + int(t * 365)
        exercise = ql.EuropeanExercise(expiry)

        option = ql.VanillaOption(payoff, exercise)
        engine = ql.Merton76Engine(
            process, self.lam, self.mu_j, self.sigma_j
        )
        option.setPricingEngine(engine)
        return option.NPV()
