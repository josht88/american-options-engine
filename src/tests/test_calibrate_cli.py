import json, pathlib, types
import aoe.data.yf as yf_mod
from aoe.models.heston_calibration import calibrate_heston
import scripts.calibrate as cli


class DummySnap:
    def __init__(self):
        import numpy as np, pandas as pd, datetime as dt
        self.spot = 100
        strikes = [90, 100, 110]
        maturities = [0.5]
        iv = [[0.25, 0.24, 0.26]]
        self.iv_surface = pd.DataFrame(iv, index=maturities, columns=strikes)
        self.earnings = dt.date.today()

    def to_dict(self):
        return dict(
            spot=self.spot,
            iv_surface=self.iv_surface,
            earnings=self.earnings,
        )


def test_calibrate_cli(monkeypatch, tmp_path):
    # monkey-patch snapshot
    snap = DummySnap().to_dict()
    # monkey-patch snapshot that calibrate_ticker actually calls
    snap = DummySnap().to_dict()
    monkeypatch.setattr(cli, "get_snapshot", lambda ticker: snap)

    # redirect output dir
    out_dir = tmp_path / "models"
    monkeypatch.setattr(cli, "OUT_DIR", out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    cli.calibrate_ticker("TEST")

    json_file = next(out_dir.glob("TEST_*.json"))
    with json_file.open() as f:
        data = json.load(f)

    assert data["ticker"] == "TEST"
    assert all(k in data for k in ["kappa", "theta", "sigma", "rho", "v0"])
