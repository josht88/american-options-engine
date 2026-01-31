"""
edge_calc
---------
Given:
    * base model object (GBM, Heston, Merton)
    * contract dict  {s0, k, r, q, t, option_type}
    * list[Shock]
    * mid_price  (market)
returns:
    edge_pct, pot_odds, fair_value_bumped
"""

import math
from typing import List
from aoe.signals import Shock


def apply_shocks(model, shocks: List[Shock]):
    """
    Return *new* model with drift/vol/jump adjusted.
    For this stub we only scale sigma via the first shock's sigma_mult
    and ignore drift/jump (those come later).
    """
    sigma_mult = math.prod([s.sigma_mult for s in shocks])
    if hasattr(model, "sigma"):            # GBM or Merton
        new_sigma = model.sigma * sigma_mult
        model.sigma = new_sigma
    elif hasattr(model, "theta"):          # Heston
        model.theta *= sigma_mult
        model.v0    *= sigma_mult
    return model


def price_after_shock(model, contract):
    return model.price_euro(
        contract["s0"],
        contract["k"],
        contract["r"],
        contract["q"],
        contract["t"],
        contract["option_type"],
    )


def compute_edge(fair_value, mid_price):
    edge_pct = (fair_value - mid_price) / mid_price
    pot_odds = edge_pct / max(1 - edge_pct, 1e-9)
    return edge_pct, pot_odds


def evaluate_trade(model, contract, shocks, mid_price, edge_thresh=0.05):
    bumped_model = apply_shocks(model, shocks)
    fv = price_after_shock(bumped_model, contract)
    edge_pct, pot_odds = compute_edge(fv, mid_price)
    trade = edge_pct > edge_thresh
    return dict(edge_pct=edge_pct, pot_odds=pot_odds, fair=fv, trade=trade)
