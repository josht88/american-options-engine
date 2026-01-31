import os
from aoe.pnl.ledger import ensure_ledger_exists, HEADERS

def test_ensure_ledger_exists_creates_file(tmp_path, monkeypatch):
    p = tmp_path/"paper.csv"
    monkeypatch.setenv("AOE_LEDGER", str(p))
    out = ensure_ledger_exists()
    assert out == p and p.exists()
    assert p.read_text().strip().splitlines()[0].split(",") == HEADERS
