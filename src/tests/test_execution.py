import os, csv, pathlib, math
from aoe.execution.broker import Order, execute_order

def test_execute_order_and_blotter(tmp_path):
    blotter = tmp_path / "blotter.csv"

    order = Order(
        ticker="AAPL",
        expiry="2025-09-19",
        strike=100,
        right="C",
        qty=10,
        limit=2.50,
        side="BUY"
    )

    fill = execute_order(order)       # default path is data/blotter.csv
    assert math.isclose(fill.fill_px, order.limit, abs_tol=0.05)

    # ensure a row was written
    with open("data/blotter.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["ticker"] == "AAPL"
    assert float(rows[-1]["fill_px"]) == fill.fill_px
