from aoe.pricing.euro_mc import price_euro_call_mc

def test_gbm_euro_mc():
    price = price_euro_call_mc(
        100, 100, 0.05, 0.20, 1.0,
        n_paths=100_000,   # more paths → lower MC error
        n_steps=200
    )
    assert abs(price - 10.45) < 0.15
