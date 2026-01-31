def test_import_web_app():
    # just make sure module import works in test env
    import web.app as app
    # core helpers exist
    for fn in ["_ensure_ledger","_equity_from_ledger","_equity_from_backtest","_run_screen"]:
        assert hasattr(app, fn)
