"""
Fair-value pricer for a *debit* vertical spread:
    long option – short option   (same expiry, same right)
Works for either European (Heston FFT) or American (CRR tree) by
delegating to the calibrated model’s `price_euro` / `price_amer` helpers.
"""

def price_debit_spread(
    model,
    s0: float,
    k_long: float,
    k_short: float,
    r: float,
    q: float,
    t: float,
    option_type: str = "call",   # "call" or "put"
    exercise: str = "euro",      # "euro" or "amer"
) -> float:
    """
    Returns model fair value = long_leg − short_leg  (>= 0).

    Parameters
    ----------
    model : calibrated model instance (e.g. HestonModel)
    s0,k_long,k_short,r,q,t : usual inputs
    option_type : "call" or "put"
    exercise    : style ("euro" uses model.price_euro, else price_amer)
    """

    pricer = model.price_euro if exercise == "euro" else model.price_amer

    price_long = pricer(s0, k_long, r, q, t, option_type)
    price_short = pricer(s0, k_short, r, q, t, option_type)

    return max(price_long - price_short, 0.0)
