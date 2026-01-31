import json, pathlib
from types import SimpleNamespace

def test_screen_day_model_resolver(tmp_path, monkeypatch):
    # Arrange fake model dir
    model_dir = tmp_path / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "AAPL_20250820.json").write_text(json.dumps({
        "ticker":"AAPL","date":"2025-08-20","params":{},"rms":0.12,
        "snapshot":{"spot":190.0,"earnings":None,"iv_surface":{"index":[],"columns":[],"data":[]}}
    }))
    (model_dir / "AAPL_20250731.json").write_text(json.dumps({
        "ticker":"AAPL","date":"2025-07-31","params":{},"rms":0.23,
        "snapshot":{"spot":185.0,"earnings":None,"iv_surface":{"index":[],"columns":[],"data":[]}}
    }))

    # Patch model dir used by screen_day
    import scripts.screen_day as sd
    monkeypatch.setattr(sd, "MODEL_DIR", model_dir)

    # Patch today and suppress calibration on fresh file
    monkeypatch.setattr(sd.dt, "date", SimpleNamespace(today=lambda: __import__("datetime").date(2025,8,26)))

    m = sd.load_model("AAPL", spot=None, max_age_days=14)  # 2025-08-20 is 6 days old → fresh
    assert m["date"] == "2025-08-20"

    # Force stale to trigger calibration path; instead of actually calibrating, fake it:
    def fake_check_call(cmd):  # write a new file as if calibrator ran
        (model_dir / "AAPL_20250826.json").write_text(json.dumps({
            "ticker":"AAPL","date":"2025-08-26","params":{},"rms":0.05,
            "snapshot":{"spot":192.0,"earnings":None,"iv_surface":{"index":[],"columns":[],"data":[]}}
        }))
        return 0
    monkeypatch.setattr(sd.subprocess, "check_call", fake_check_call)
    m2 = sd.load_model("AAPL", spot=None, max_age_days=0)  # everything stale → calibrate
    assert m2["date"] == "2025-08-26"
