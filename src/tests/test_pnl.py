import csv, pathlib
from aoe.pnl.ledger import book_fill, update_marks

def test_ledger_roundtrip(tmp_path, monkeypatch):
    # Make both module constant and env point to tmp ledger
    p = tmp_path / "pnl.csv"
    monkeypatch.setenv("AOE_LEDGER", str(p))
    monkeypatch.setattr("aoe.pnl.ledger.LEDGER", p)

    order = dict(ticker="T", expiry="2099-01-19", right="C",
                 long_k=95, short_k=105, qty=1, price=2.0)
    book_fill(order)

    marks = [dict(ticker="T", expiry="2099-01-19", right="C",
                  long_k=95, short_k=105, mark=1.5, realised=0.0)]
    update_marks(marks)

    assert p.exists()
    with p.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "T"
    assert abs(float(r["mark"]) - 1.5) < 1e-6
    assert abs(float(r["realised"]) - 0.0) < 1e-6
