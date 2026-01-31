import QuantLib as ql
from aoe.pricing.jump_tree import price_american_jump_tree


def test_merton_jump_tree_put():
    # coarse tree (400) under test
    price_coarse = price_american_jump_tree(
        100, 95, 0.03, 0.0, 0.5,
        sigma=0.25, lam=1.2, mu_j=-0.1, sigma_j=0.2,
        option_type="put", n_steps=400,
    )

    # fine tree (3000) used as 'reference'
    price_fine = price_american_jump_tree(
        100, 95, 0.03, 0.0, 0.5,
        sigma=0.25, lam=1.2, mu_j=-0.1, sigma_j=0.2,
        option_type="put", n_steps=3000,
    )

    # difference should be < 3 cents
    assert abs(price_coarse - price_fine) < 0.03

