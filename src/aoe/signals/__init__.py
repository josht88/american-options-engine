"""
signals package
---------------

Currently implemented:
    * EPS surprise  -> Shock

Future:
    * Macro CPI     -> Shock
    * M&A rumour    -> Shock
"""

from .eps import eps_surprise_to_shock, Shock

def get_active_shocks(ticker: str, surprise_pct: float):
    """
    For now we inject surprise_pct manually (tests / notebooks).
    Later you’ll query a DB or API.

    Returns
    -------
    list[Shock]
    """
    return [eps_surprise_to_shock(surprise_pct)]
