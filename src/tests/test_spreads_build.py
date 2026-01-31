import pandas as pd
from aoe.strategy.spreads import build_range_spreads


def test_build_spreads():
    data = dict(
        ticker=["AAPL"] * 3,
        expiry=["2099-01-19"] * 3,
        right=["C"] * 3,
        strike=[95, 100, 110],
        mid=[8, 5, 2],
        delta=[0.78, 0.70, 0.25],
        iv=[0.25, 0.24, 0.23],
        oi=[1000, 900, 800],
    )
    df_single = pd.DataFrame(data)

    class DummyModel:
        spot = 100

    spread_df = build_range_spreads(df_single, DummyModel())
    assert not spread_df.empty
    row = spread_df.iloc[0]
    assert row["long_k"] == 100 and row["short_k"] == 110
    assert row["net_mid"] == 3
