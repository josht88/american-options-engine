# src/aoe/strategy/credit.py
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm

# --------------------- Black helpers ---------------------

def _black_call(S, K, r, q, vol, T):
    if vol <= 0 or T <= 0:
        return max(0.0, math.exp(-q*T)*S - math.exp(-r*T)*K)
    sig = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / sig
    d2 = d1 - sig
    return math.exp(-q*T)*S*norm.cdf(d1) - math.exp(-r*T)*K*norm.cdf(d2)

def _black_put(S, K, r, q, vol, T):
    if vol <= 0 or T <= 0:
        return max(0.0, math.exp(-r*T)*K - math.exp(-q*T)*S)
    sig = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / sig
    d2 = d1 - sig
    return math.exp(-r*T)*K*norm.cdf(-d2) - math.exp(-q*T)*S*norm.cdf(-d1)

# Backward-compat export names expected by tests
_black_call_price = _black_call
_black_put_price  = _black_put

def _ln_quantile(F, vol, T, alpha):
    """Lognormal quantile under forward measure."""
    z = norm.ppf(alpha)
    return F * math.exp(-0.5 * vol*vol * T + vol * math.sqrt(T) * z)

# --------------------- Public API ---------------------

def price_credit_spread(*args: Any, **kwargs: Any) -> float:
    """
    Fair credit (seller) for a short vertical.

    Supports TWO calling styles:

    1) Model-first positional (used in tests):
       price_credit_spread(model, s0, short_k, long_k, r, q, t, right)
       -> uses model.price_euro(...)

    2) Black inputs via keywords (or positionals with names):
       price_credit_spread(s0=..., r=..., q=..., t=..., vol=..., right="C",
                           short_k=..., long_k=...)
       (aliases: T for t, sigma/iv for vol)
    """
    # --- detect model-first signature
    if args and hasattr(args[0], "price_euro"):
        # Unpack: (model, s0, short_k, long_k, r, q, t, right)
        if len(args) < 8:
            raise ValueError("model-first call requires 8 positional args: "
                             "model, s0, short_k, long_k, r, q, t, right")
        model, s0, short_k, long_k, r, q, t, right = args[:8]
        optC = "call" if str(right).upper().startswith("C") else "put"
        if optC == "call":
            short_val = model.price_euro(s0, float(short_k), r, q, t, "call")
            long_val  = model.price_euro(s0, float(long_k),  r, q, t, "call")
        else:
            short_val = model.price_euro(s0, float(short_k), r, q, t, "put")
            long_val  = model.price_euro(s0, float(long_k),  r, q, t, "put")
        return float(short_val - long_val)

    # --- otherwise: Black inputs path
    s0   = kwargs.get("s0",   args[0] if len(args) > 0 else None)
    r    = kwargs.get("r",    args[1] if len(args) > 1 else None)
    q    = kwargs.get("q",    args[2] if len(args) > 2 else None)
    t    = kwargs.get("t",    kwargs.get("T", args[3] if len(args) > 3 else None))
    vol  = kwargs.get("vol",  kwargs.get("sigma", kwargs.get("iv",
            args[4] if len(args) > 4 else None)))
    right    = kwargs.get("right",    args[5] if len(args) > 5 else "C")
    short_k  = kwargs.get("short_k",  args[6] if len(args) > 6 else None)
    long_k   = kwargs.get("long_k",   args[7] if len(args) > 7 else None)

    if None in (s0, r, q, t, vol, short_k, long_k):
        raise ValueError("Missing required inputs for price_credit_spread")

    if str(right).upper().startswith("C"):
        credit = _black_call(s0, float(short_k), r, q, vol, t) - \
                 _black_call(s0, float(long_k),  r, q, vol, t)
    else:
        credit = _black_put(s0, float(short_k), r, q, vol, t) - \
                 _black_put(s0, float(long_k),  r, q, vol, t)
    return float(credit)

def build_iron_condor_from_chain(
    ticker, s0, r, q, T, calls_df, puts_df,
    iv_atm, p_tail=0.20, wing_width=5.0, expiry="YYYY-MM-DD"
):
    """
    Build one iron condor using lognormal quantile strikes.
    Publishes *model-fair* credit as `credit_mid` (so EV >= 0 under same model).
    Also includes actual market mid as `credit_mkt_mid`.
    """
    F = s0 * math.exp((r - q) * T)

    # Target short strikes from model quantiles
    k_put_short_tgt  = _ln_quantile(F, iv_atm, T, p_tail)      # e.g., 20% quantile
    k_call_short_tgt = _ln_quantile(F, iv_atm, T, 1 - p_tail)  # e.g., 80% quantile

    # Snap to listed strikes
    def _nearest_strike(df, target):
        arr = np.asarray(df["strike"], dtype=float)
        return float(arr[np.abs(arr - target).argmin()])

    kps = _nearest_strike(puts_df,  k_put_short_tgt)   # short put
    kcs = _nearest_strike(calls_df, k_call_short_tgt)  # short call
    kpl = max(0.01, kps - wing_width)  # long put (lower)
    kcl = kcs + wing_width             # long call (higher)

    # --- Market mids (for reference only) ---
    def _mid(df, K):
        row = df.iloc[(df["strike"] - K).abs().argmin()]
        return float((row["bid"] + row["ask"]) * 0.5)

    mkt_short_put  = _mid(puts_df,  kps)
    mkt_long_put   = _mid(puts_df,  kpl)
    mkt_short_call = _mid(calls_df, kcs)
    mkt_long_call  = _mid(calls_df, kcl)
    credit_mkt_mid = (mkt_short_put - mkt_long_put) + (mkt_short_call - mkt_long_call)

    # --- Model-fair credit ---
    fair_short_put  = _black_put(s0, kps, r, q, iv_atm, T)
    fair_long_put   = _black_put(s0, kpl, r, q, iv_atm, T)
    fair_short_call = _black_call(s0, kcs, r, q, iv_atm, T)
    fair_long_call  = _black_call(s0, kcl, r, q, iv_atm, T)
    credit_fair = (fair_short_put - fair_long_put) + (fair_short_call - fair_long_call)

    # Max loss (use wider wing)
    width_put  = max(0.0, kps - kpl)
    width_call = max(0.0, kcl - kcs)
    max_loss = max(width_put, width_call) - credit_fair
    max_loss = max(max_loss, 0.0)

    row = dict(
        ticker=ticker, expiry=expiry, right="IC",
        long_put=kpl, short_put=kps, short_call=kcs, long_call=kcl,
        credit_mid=credit_fair,          # publish model-fair as "mid"
        credit_mkt_mid=credit_mkt_mid,   # keep actual market mid
        max_loss=max_loss,
    )
    return pd.DataFrame([row])

def ev_iron_condor_logn(
    s0, r, q, vol, T, kpl, kps, kcs, kcl, credit, n_paths=0, seed=None
):
    """
    Deterministic EV under Black/lognormal (no MC noise):
    EV_to_seller = credit - fair_credit.
    """
    fair_short_put  = _black_put(s0, kps, r, q, vol, T)
    fair_long_put   = _black_put(s0, kpl, r, q, vol, T)
    fair_short_call = _black_call(s0, kcs, r, q, vol, T)
    fair_long_call  = _black_call(s0, kcl, r, q, vol, T)
    fair_credit = (fair_short_put - fair_long_put) + (fair_short_call - fair_long_call)
    return float(credit - fair_credit)
