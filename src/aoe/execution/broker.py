import pathlib
from dataclasses import dataclass
import random

BLOTTER_PATH = pathlib.Path("data") / "blotter.csv"
BLOTTER_PATH.parent.mkdir(exist_ok=True)


@dataclass
class Order:
    ticker: str
    expiry: str    # YYYY-MM-DD
    strike: float
    right: str     # "C" or "P"
    qty: int
    limit: float
    side: str      # "BUY" only for now


@dataclass
class Fill:
    order: Order
    fill_px: float


def simulated_slippage(limit):
    """±2 ¢ random slip."""
    return limit + random.choice([-0.02, 0, 0.02])


def execute_order(order: Order) -> Fill:
    # import here to avoid circular dependency
    from aoe.execution.blotter import record_fill

    fill_px = simulated_slippage(order.limit)
    fill = Fill(order, fill_px)
    record_fill(fill, BLOTTER_PATH)
    return fill

