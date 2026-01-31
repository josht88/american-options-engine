# src/tests/test_walkforward_windows.py
from scripts.walkforward import _daterange_qtr
import datetime as dt

def test_quarter_windows_simple():
    s = dt.date(2024, 1, 15)
    e = dt.date(2024, 12, 20)
    windows = list(_daterange_qtr(s, e))
    # Expect Q1,Q2,Q3,Q4 (start pinned to Q1)
    assert windows[0][0] == "2024-01-01"
    assert windows[-1][1] == "2024-12-31"
    assert len(windows) == 4
