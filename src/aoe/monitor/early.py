"""
Early-exercise checker + MTM.

• For puts: exercise when intrinsic ≥ model value + 1 ¢ buffer.
• For calls: only if no dividends (simplify) so always hold.
Returns dict with mark, exercise_flag.
"""
import math, datetime as dt
from aoe.pricing.tree import price_option_crr

BUFFER = 0.01  # 1 ¢ to avoid flip-flop

def mark_and_check(model, side, s0, k, r, q, dte, long_qty):
    """
    Parameters
    ----------
    model : HestonModel (calibrated)
    side  : "C" | "P"
    s0    : spot today
    k     : strike
    dte   : days-to-expiry (int)
    long_qty : +1 for long, −1 for short leg

    Returns
    -------
    dict(mark=float, exercise=bool, pnl_if_ex=float)
    """
    t = dte / 365
    if t <= 0:
        return dict(mark=0.0, exercise=True, pnl_if_ex=0.0)

    # model fair value
    theo = model.price_euro(s0, k, r, q, t, "call" if side == "C" else "put")

    # American CRR tree mark (fast)
    amer = price_option_crr(s0, k, r, model.sigma0, t,
                            n_steps=200, option_type="call" if side=="C" else "put",
                            exercise="amer")

    intrinsic = max(s0-k, 0) if side=="C" else max(k-s0, 0)

    if side == "P" and intrinsic >= amer - BUFFER:
        # exercise now
        pnl = long_qty * intrinsic
        return dict(mark=0.0, exercise=True, pnl_if_ex=pnl)
    # calls: skip early exercise (assume no dividend)
    return dict(mark=amer * long_qty, exercise=False, pnl_if_ex=0.0)
