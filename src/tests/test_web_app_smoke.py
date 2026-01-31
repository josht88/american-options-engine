# src/tests/test_web_app_smoke.py
import importlib
def test_imports_and_helpers():
    m = importlib.import_module("web.app")
    for fn in ["_ensure_ledger","_run_screen","_equity_from_ledger","_equity_from_backtest"]:
        assert hasattr(m, fn)
