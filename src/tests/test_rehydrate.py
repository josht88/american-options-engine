import pandas as pd
from web.app import _rehydrate_condors, _edge_and_checks_all

def test_rehydrate_populates_missing_columns():
    # A “compat builder” style minimal row lacking the rich columns
    singles = pd.DataFrame({
        "expiry":["2025-10-03"]*3 + ["2025-10-03"]*3,
        "right":["P","P","P","C","C","C"],
        "strike":[95,100,105,95,100,105],
        "bid":[0.8,2.4,5.2,5.4,2.6,0.95],
        "ask":[0.95,2.6,5.4,5.6,2.8,1.1],
    })
    singles["mid"] = (singles["bid"] + singles["ask"]) / 2
    base = pd.DataFrame([{
        "expiry":"2025-10-03","long_put":95,"short_put":100,"short_call":100,"long_call":105
    }])
    out = _rehydrate_condors(singles, base, mdl=None, edge_cap=120.0, wing_tol=0.10, credit_leeway=0.05)
    row = out.iloc[0]
    # must exist
    for c in ["builder_credit","exec_credit","credit","width","risk","edge_pct","be_low","be_high","rr","pass_gates"]:
        assert c in out.columns
    assert row.credit <= row.width + 1e-9
    assert row.risk > 0
    assert row.pass_gates is True
