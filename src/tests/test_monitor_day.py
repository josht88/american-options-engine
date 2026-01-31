import csv
import datetime as dt
import aoe.pnl.ledger as ledg_mod
import scripts.monitor_day as mon

def test_monitor_day(monkeypatch, tmp_path):
    # --- force ledger path to tmp AND clear any global env override ---
    tmp_ledger = tmp_path / "pnl.csv"
    monkeypatch.setattr(ledg_mod, "LEDGER", tmp_ledger)
    monkeypatch.setenv("AOE_LEDGER", str(tmp_ledger))  # <— ensure consistency

    # --- create one OPEN row with legacy headers (as the test expects) ---
    with tmp_ledger.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp","ticker","expiry","right","long_k","short_k",
                "qty","entry","mark","realised","tag",
            ],
        )
        w.writeheader()
        w.writerow(dict(
            timestamp="2025-07-01T20:00:00",
            ticker="TEST", expiry="2099-01-19", right="C",
            long_k="95", short_k="105",
            qty="1", entry="3.00", mark="", realised="", tag="OPEN",
        ))

    # --- patch model loader & spot getter ---
    class DummyModel: spot = 100
    monkeypatch.setattr(mon, "_load_model", lambda t, s: DummyModel())
    monkeypatch.setattr(mon, "_fetch_spot", lambda tk: 100.0)

    # --- patch per-leg marker(s) ---
    def fake_mark_and_check(*a, **k):
        return dict(mark=3.10, pnl_if_ex=0.0, exercise=False)
    monkeypatch.setattr(mon, "mark_and_check", fake_mark_and_check)

    try:
        import aoe.pnl.mtm as mtm_mod
        def fake_runner(rows, model_loader, spot_fetcher, r, q):
            out = []
            for rrow in rows:
                if rrow.get("tag") != "OPEN":
                    continue
                out.append(dict(
                    ticker=rrow["ticker"], expiry=rrow["expiry"],
                    right=rrow["right"], long_k=rrow["long_k"], short_k=rrow["short_k"],
                    mark=3.10, realised=0.0, tag="OPEN",
                ))
            return out
        monkeypatch.setattr(mtm_mod, "mark_open_legs", fake_runner)
    except Exception:
        pass

    # --- run & verify ---
    mon.run()
    rows = list(csv.DictReader(tmp_ledger.open()))
    assert len(rows) == 1
    assert abs(float(rows[0]["mark"]) - 3.10) < 1e-6
    assert rows[0]["tag"] == "OPEN"
