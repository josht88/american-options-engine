import math

def kelly_fraction(edge_pct, sigma_pnl):
    """
    Continuous-Kelly fraction, capped at 1.
    """
    if sigma_pnl <= 1e-9:
        return 0.0
    f = edge_pct / (sigma_pnl**2)
    return max(0.0, min(f, 1.0))

def contracts_to_buy(
    edge_pct,
    delta,
    underlying_price,
    option_price,
    var_1d,
    risk_cap_usd=1_000,          # max $ to risk on 1-day VAR
    oi_limit=0.1,                # 10 % of open interest
    open_interest=10_000,
    lot_size=100,                # US equity options multiplier
):
    sigma_pnl = abs(delta) * underlying_price * var_1d
    f = kelly_fraction(edge_pct, sigma_pnl)
    kelly_dollars = f * risk_cap_usd / option_price

    # liquidity cap
    liq_cap = open_interest * oi_limit

    contracts = math.floor(min(kelly_dollars, liq_cap))
    return max(0, contracts)

# ------------------------------------------------------------------
# Kelly-style position size for a *fixed-loss* trade
# ------------------------------------------------------------------

def kelly_size(edge_pct: float,
               loss_per_contract: float,
               bankroll: float,
               cap: float = 0.05) -> int:
    """
    Return integer contract qty sized by fractional Kelly.

    Parameters
    ----------
    edge_pct          : (model_price – market_mid) / market_mid  (0.12 = 12 %)
    loss_per_contract : max loss in dollars (net debit or margin)
    bankroll          : total risk capital
    cap               : max bankroll fraction per trade (e.g. 0.05 = 5 %)

    Kelly f* ≈ edge / (loss/gain).  For a debit spread,
    gain = width – debit; loss = debit.
    Here we approximate with  f* = edge_pct / (1 + edge_pct).
    """
    if edge_pct <= 0 or loss_per_contract <= 0:
        return 0

    f_star = edge_pct / (1 + edge_pct)
    f_star = min(f_star, cap)                # risk cap
    dollar_risk = bankroll * f_star
    qty = int(dollar_risk // loss_per_contract)
    return max(qty, 0)
