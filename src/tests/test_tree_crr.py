import QuantLib as ql
from aoe.pricing.tree import price_option_crr

N_STEPS = 10000

def reference_ql(s0, k, r, sigma, t, opt_type, exercise, n_steps=800):
    day_count = ql.Actual365Fixed()
    today     = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if opt_type == "call" else ql.Option.Put, k
    )
    ex_date = today + int(t * 365)
    exercise_ql = ql.AmericanExercise(today, ex_date) if exercise == "amer" else ql.EuropeanExercise(ex_date)

    spot = ql.SimpleQuote(s0)
    flat_r = ql.FlatForward(today, r, day_count)
    flat_vol = ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count)

    process = ql.BlackScholesProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(flat_r),
        ql.BlackVolTermStructureHandle(flat_vol),
    )

    if exercise == "amer":
        engine = ql.BinomialVanillaEngine(process, "crr", n_steps)
    else:
        engine = ql.AnalyticEuropeanEngine(process)

    option = ql.VanillaOption(payoff, exercise_ql)
    option.setPricingEngine(engine)
    return option.NPV()

def test_crr_american_put():
    price = price_option_crr(
        100, 95, 0.03, 0.2, 0.5,
        n_steps=N_STEPS, option_type="put", exercise="amer"
    )
    ref = reference_ql(
        100, 95, 0.03, 0.2, 0.5,
        "put", "amer", n_steps=N_STEPS
    )

    assert abs(price - ref) < 0.03

def test_crr_euro_call():
    price = price_option_crr(
        100, 100, 0.05, 0.25, 1.0,
        n_steps=N_STEPS, option_type="call", exercise="euro"
    )
    ref = reference_ql(
        100, 100, 0.05, 0.25, 1.0,
        "call", "euro", n_steps=N_STEPS
    )

    assert abs(price - ref) < 0.02
