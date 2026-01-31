import datetime as dt
from aoe.signals.macro import sigma_multiplier

def test_sigma_multiplier():
    # choose a known CPI date: 2025-08-13 is 2nd Wednesday
    cpi_day  = dt.date(2025, 8, 13)
    pre_day  = cpi_day - dt.timedelta(days=1)
    post_day = cpi_day + dt.timedelta(days=1)
    normal   = dt.date(2025, 8, 20)

    assert sigma_multiplier(cpi_day)  == 1.50
    assert sigma_multiplier(pre_day)  == 1.25
    assert sigma_multiplier(post_day) == 1.25
    assert sigma_multiplier(normal)   == 1.00
