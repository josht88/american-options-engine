"""
EPS‐surprise → Shock
--------------------
Surprise % = (reported − consensus)/consensus
Mapping (empirical rule of thumb):
    +1% surprise  →  +0.10 σ annualised drift for 5 trading days
    −1% surprise  →  −0.10 σ annualised drift for 5 trading days
Vol crush: 60 % of pre-earnings IV instantly, decays back linearly.
"""

from dataclasses import dataclass


@dataclass
class Shock:
    mu_shift: float      # daily drift change (decimal, not %)
    sigma_mult: float    # multiplier on vol surface
    jump_lambda: float   # extra jump intensity
    confidence: float    # 0-1 weight
    decay: int           # business days

def eps_surprise_to_shock(surprise_pct: float) -> Shock:
    # drift: 0.1 × surprise%  (e.g. +5 % surprise ==> +0.5σ annual)
    mu_shift = 0.001 * surprise_pct * 0.1        # daily
    # immediate 40 % vol crush after earnings
    sigma_mult = 0.6
    return Shock(
        mu_shift=mu_shift,
        sigma_mult=sigma_mult,
        jump_lambda=0.0,
        confidence=1.0,
        decay=5
    )
