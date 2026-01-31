import pandas as pd
import math
from web.app import (
    build_ic_from_chain_simple, build_ic_grid_search,
    _sanity_gate_row, _recompute_credit_variants, _condor_widths
)

def _toy_chain(expiry="2025-10-03"):
    puts  = pd.DataFrame({"expiry":[expiry]*5,"right":["P"]*5,"strike":[90,95,100,105,110],
                          "bid":[0.2,0.8,2.4,5.2,8.7],"ask":[0.3,0.95,2.6,5.4,9.1]})
    calls = pd.DataFrame({"expiry":[expiry]*5,"right":["C"]*5,"strike":[90,95,100,105,110],
                          "bid":[9.1,5.4,2.6,0.95,0.3],"ask":[9.4,5.6,2.8,1.1,0.4]})
    for df in (puts,calls):
        df["mid"] = (df.bid + df.ask) / 2
    return pd.concat([puts, calls], ignore_index=True)

def test_recompute_credit_variants_and_widths():
    ch = _toy_chain()
    lp, sp, sc, lc = 95, 100, 100, 105
    info = _recompute_credit_variants(ch, lp, sp, sc, lc)
    pw, cw, w = _condor_widths(lp, sp, sc, lc)

    assert math.isclose(pw, 5.0)
    assert math.isclose(cw, 5.0)
    assert math.isfinite(info["credit_ps_mid"])
    assert info["credit_ps_mid"] <= w + 1e-9  # credit ≤ width
    assert info["credit_ps_aggr"] >= info["credit_ps_cons"]  # bounds monotonicity

def test_builder_simple_respects_invariants():
    ch = _toy_chain()
    df = build_ic_from_chain_simple(ch, spot=100, wing=5, dte_days=21)
    assert not df.empty
    row = df.iloc[0]
    assert 0 <= row.credit <= row.width + 1e-9
    assert row.risk > 0
    assert 0 <= row.edge_pct <= 120

def test_grid_search_topn_and_sorting():
    ch = _toy_chain()
    out = build_ic_grid_search(ch, spot=100, dte_days=21, wings=(3,5,7), top_n=2)
    assert len(out) <= 2
    # sorted by _score (exec then builder)
    # we can at least assert non-increasing by credit proxy
    creds = out["exec_credit"].fillna(out["builder_credit"]).tolist()
    assert creds == sorted(creds, reverse=True)

def test_sanity_gate_row():
    ok, why = _sanity_gate_row(95, 100, 100, 105, credit_ps=1.5, tol=0.10, edge_cap=120)
    assert ok and not why
    ok, why = _sanity_gate_row(95, 100, 101, 106, credit_ps=6.0, tol=0.10, edge_cap=120)  # credit > width
    assert not ok and why in {"credit_gt_width","unequal_wings"}
