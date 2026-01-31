from aoe.signals import get_active_shocks

def test_eps_shock_positive():
    shocks = get_active_shocks("AAPL", surprise_pct=5.0)   # +5%
    shock = shocks[0]
    assert shock.mu_shift > 0
    assert shock.sigma_mult == 0.6
    assert shock.decay == 5

def test_eps_shock_negative():
    shocks = get_active_shocks("AAPL", surprise_pct=-3.0)
    shock = shocks[0]
    assert shock.mu_shift < 0
    assert shock.sigma_mult == 0.6
