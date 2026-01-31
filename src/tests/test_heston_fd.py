import QuantLib as ql
from aoe.pricing.fd2d import price_heston_fd

def ql_heston_american(s0, k, r, q, t, kappa, theta, sigma, rho, v0, option_type):
    day_count = ql.Actual365Fixed()
    today     = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, k
    )
    expiry = today + int(t * 365)
    exercise = ql.AmericanExercise(today, expiry)

    spot = ql.SimpleQuote(s0)
    flat_r  = ql.FlatForward(today, r, day_count)
    flat_q  = ql.FlatForward(today, q, day_count)
    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(flat_r),
        ql.YieldTermStructureHandle(flat_q),
        ql.QuoteHandle(spot),
        v0, kappa, theta, sigma, rho
    )
    engine = ql.FdHestonVanillaEngine(ql.HestonModel(process), 120, 100, 60)
    opt = ql.VanillaOption(payoff, exercise)
    opt.setPricingEngine(engine)
    return opt.NPV()

def test_heston_fd_put():
    pnl_fd = price_heston_fd(100, 95, 0.03, 0.0, 0.5,
        kappa=2.0, theta=0.05, sigma=0.4, rho=-0.6, v0=0.04,
        option_type="put"
    )
    pnl_ref = ql_heston_american(100, 95, 0.03, 0.0, 0.5,
        2.0, 0.05, 0.4, -0.6, 0.04, "put"
    )
    assert abs(pnl_fd - pnl_ref) < 0.05
