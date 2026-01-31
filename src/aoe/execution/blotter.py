from __future__ import annotations

import csv
import pathlib
from datetime import datetime, UTC

from aoe.execution.broker import Fill


HEADERS = [
    "timestamp", "ticker", "expiry", "strike", "right",
    "qty", "side", "limit", "fill_px"
]

def record_fill(fill: Fill, path: pathlib.Path):
    line = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticker": fill.order.ticker,
        "expiry": fill.order.expiry,
        "strike": fill.order.strike,
        "right": fill.order.right,
        "qty": fill.order.qty,
        "side": fill.order.side,
        "limit": fill.order.limit,
        "fill_px": fill.fill_px,
    }
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(line)
