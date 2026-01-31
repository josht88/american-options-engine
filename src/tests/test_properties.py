import pandas as pd
from hypothesis import given, strategies as st
from web.app import build_ic_from_chain_simple

@given(
    mid=st.floats(min_value=0.1, max_value=20),
    skew=st.floats(min_value=-0.5, max_value=0.5),
    noise=st.floats(min_value=0.0, max_value=0.2),
)
def test_credit_never_exceeds_width_under_randomized_quotes(mid, skew, noise):
    # Build a synthetic, sane chain around spot=100 with mild skew
    strikes = [90,95,100,105,110]
    puts = []; calls = []
    for k in strikes:
        d = abs(k-100)/10
        p_mid = max(0.05, mid*(1+d+max(0,skew)*d))
        c_mid = max(0.05, mid*(1+d+max(0,-skew)*d))
        puts.append({"expiry":"2099-01-01","right":"P","strike":k,"bid":p_mid*(1-noise),"ask":p_mid*(1+noise),"mid":p_mid})
        calls.append({"expiry":"2099-01-01","right":"C","strike":k,"bid":c_mid*(1-noise),"ask":c_mid*(1+noise),"mid":c_mid})
    ch = pd.DataFrame(puts + calls)
    df = build_ic_from_chain_simple(ch, spot=100, wing=5, dte_days=21)
    if not df.empty:
        row = df.iloc[0]
        assert row.credit <= row.width + 1e-8
