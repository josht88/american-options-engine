"""
Heston American pricer — delegates to QuantLib's FdHestonVanillaEngine
so we get a rock-solid benchmark while keeping the public signature
identical to the earlier NumPy prototype.
"""

import QuantLib as ql


def price_heston_fd(
    s0,
    k,
    r,
    q,
    t,
    kappa,
    theta,
    sigma,
    rho,
    v0,
    s_max_mult=3,
    v_max=1.0,
    Mx=120,
    Mv=60,
    Nt=120,
    option_type="put",
):
    day_count = ql.Actual365Fixed()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, k
    )
    expiry = today + int(t * 365)
    exercise = ql.AmericanExercise(today, expiry)

    spot = ql.SimpleQuote(s0)
    flat_r = ql.FlatForward(today, r, day_count)
    flat_q = ql.FlatForward(today, q, day_count)

    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(flat_r),
        ql.YieldTermStructureHandle(flat_q),
        ql.QuoteHandle(spot),
        v0,
        kappa,
        theta,
        sigma,
        rho,
    )

    # order: model, tGrid, xGrid, vGrid
    engine = ql.FdHestonVanillaEngine(
        ql.HestonModel(process),
        Nt,
        Mx,
        Mv,
    )


    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option.NPV()
