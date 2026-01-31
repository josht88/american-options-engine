import os, csv
from aoe.pnl import ledger as L

def test_ledger_respects_env(tmp_path, monkeypatch):
    custom = tmp_path/"pnl_paper.csv"
    monkeypatch.setenv("AOE_LEDGER", str(custom))

    # Book one fill
    L.book_fill({
        "ticker":"TEST","expiry":"2025-01-17","right":"C",
        "long_k":100,"short_k":105,"qty":1,"price":2.50
    })
    assert custom.exists()

    # Update marks
    L.update_marks([{
        "ticker":"TEST","expiry":"2025-01-17","right":"C",
        "long_k":100,"short_k":105,"mark":3.20,"realised":0.0,"tag":"PAPER"
    }])

    # Read back
    with custom.open() as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["tag"] in ("OPEN","PAPER")
    assert float(rows[0]["mark"]) >= 0.0
