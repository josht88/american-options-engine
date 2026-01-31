import os, pathlib, re

def _read(p): return pathlib.Path(p).read_text()

def test_preflop_script_has_env_and_is_executable():
    p = pathlib.Path("scripts/run_preflop.sh")
    assert p.exists()
    txt = _read(p)
    assert "AOE_LEDGER" in txt and "screen_day" in txt
    assert os.access(p, os.X_OK)

def test_postflop_script_has_env_and_is_executable():
    p = pathlib.Path("scripts/run_postflop.sh")
    assert p.exists()
    txt = _read(p)
    assert "AOE_LEDGER" in txt and "monitor_day" in txt
    assert os.access(p, os.X_OK)
