# src/tests/test_metrics.py
import pandas as pd
from aoe.metrics import summarize

def test_metrics_shapes():
    eq = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=10, freq="B"),
                       "equity": [100,101,102,103,102,104,105,106,108,109]})
    tr = pd.DataFrame({"realised": [10,-5,8,-2,0,5]})
    m = summarize(eq, tr)
    for k in ["CAGR","Sharpe","Sortino","MaxDD","MAR","Vol","hit_rate","avg_win","avg_loss","expectancy","trades"]:
        assert k in m
