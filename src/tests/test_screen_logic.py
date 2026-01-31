import math
import pandas as pd

# After conftest.py runs, this import should succeed
from web.app import build_ic_from_chain_simple

def _toy_chain_around(spot: float = 100.0):
    """
    Build a tiny, internally consistent chain:
      - puts/calls at strikes 95/100/105
      - bid/ask so that mids are sensible and credit < width
    """
    expiry = "2025-10-03"
    puts = pd.DataFrame({
        "expiry": [expiry]*3,
        "right":  ["P"]*3,
        "strike": [95.0, 100.0, 105.0],
        "bid":    [0.80, 2.40, 5.20],
        "ask":    [0.95, 2.60, 5.40],
    })
    puts["mid"] = (puts["bid"] + puts["ask"]) / 2.0

    calls = pd.DataFrame({
        "expiry": [expiry]*3,
        "right":  ["C"]*3,
        "strike": [95.0, 100.0, 105.0],
        "bid":    [5.40, 2.60, 0.95],
        "ask":    [5.60, 2.80, 1.10],
    })
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0

    return pd.concat([puts, calls], ignore_index=True)

def test_builder_produces_sane_condor():
    singles = _toy_chain_around(100.0)
    out = build_ic_from_chain_simple(
        singles=singles,
        spot=100.0,
        wing=5.0,
        dte_days=21,
        slippage_bps=20.0,
        fee_per_contract=0.65,
        width_tol=0.25,
        credit_leeway=0.10,
        use_exec_prices=True,
    )
    # Should yield exactly one row
    assert isinstance(out, pd.DataFrame) and len(out) == 1, "builder returned no row"

    r = out.iloc[0]
    width = float(r["width"])
    credit = float(r["credit"])
    risk = float(r["risk"])
    edge = float(r["edge_pct"])
    bcred = float(r["builder_credit"])
    # exec_credit may be NaN if toy quotes cause negative executable credit after fees
    # so we only sanity-check when finite
    ecred = float(r["exec_credit"]) if pd.notna(r["exec_credit"]) else float("nan")

    assert width > 0, "width must be positive"
    assert credit >= 0, "credit should not be negative"
    assert credit < width + 1e-6, "credit must be < width"
    assert risk > 0, "risk must be positive (= width - credit)"
    assert 0 <= edge <= 120.0, "edge% out of sanity range"
    assert bcred <= width + 1e-6, "builder credit cannot exceed width"
    if math.isfinite(ecred):
        assert ecred <= width + 1e-6, "exec credit cannot exceed width"

    # Strikes should be symmetric around shorts (equal width legs)
    put_w = abs(float(r["short_put"]) - float(r["long_put"]))
    call_w = abs(float(r["long_call"]) - float(r["short_call"]))
    assert abs(put_w - call_w) <= 0.25, "wings too unequal for toy chain"
