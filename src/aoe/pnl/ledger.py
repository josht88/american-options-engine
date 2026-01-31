"""
Simple CSV-backed trade & PnL ledger.

Columns: ts, ticker, expiry, right, long_k, short_k, qty, price, mark, realised, tag

Notes:
- Backward compatible: exports `LEDGER` constant pointing to the default path.
- Env override: set AOE_LEDGER to route reads/writes elsewhere (e.g., data/pnl_paper.csv).
- New: ensure_ledger_exists() to create the file with headers proactively.
"""
from __future__ import annotations
import csv, pathlib, os
from datetime import datetime, UTC

# Default path
DEFAULT_LEDGER_PATH = pathlib.Path("data") / "pnl.csv"

# Back-compat for older code/tests
LEDGER = DEFAULT_LEDGER_PATH

HEADERS = [
    "ts","ticker","expiry","right","long_k","short_k",
    "qty","price","mark","realised","tag"
]

def _ledger_path() -> pathlib.Path:
    """Return the active ledger path (env override if set)."""
    env = os.environ.get("AOE_LEDGER")
    path = pathlib.Path(env) if env else LEDGER
    path.parent.mkdir(exist_ok=True, parents=True)
    return path

def _ensure_file(path: pathlib.Path):
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(HEADERS)

def ensure_ledger_exists() -> pathlib.Path:
    """
    Public initializer: creates the active ledger file with headers if missing.
    Returns the pathlib.Path to the active ledger.
    """
    path = _ledger_path()
    _ensure_file(path)
    return path

def book_fill(order: dict):
    """
    Append a new OPEN fill row to the ledger.
    Required keys in `order`: ticker, expiry, right, long_k, short_k, qty, price
    """
    ledger = _ledger_path()
    _ensure_file(ledger)
    row = [
        datetime.now(UTC).isoformat(timespec="seconds"),
        order["ticker"], order["expiry"], order["right"],
        order["long_k"], order["short_k"],
        order["qty"], order["price"],
        0.0, 0.0, "OPEN"
    ]
    with ledger.open("a", newline="") as f:
        csv.writer(f).writerow(row)

def update_marks(marks: list[dict]):
    """Update mark/realised/tag for matching OPEN rows in the ledger."""
    ledger = _ledger_path()
    _ensure_file(ledger)

    with ledger.open() as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        for m in marks:
            if (r["ticker"], r["expiry"], r["right"], r["long_k"], r["short_k"]) == (
                m["ticker"], m["expiry"], m["right"], str(m["long_k"]), str(m["short_k"])
            ):
                prev_real = float(r.get("realised") or 0.0)
                add_real  = float(m.get("realised", 0.0) or 0.0)
                r.update(
                    mark=f'{float(m["mark"]):.2f}',
                    realised=f'{prev_real + add_real:.2f}',
                    tag=m.get("tag", r["tag"]),
                )

    cleaned = [{h: r.get(h, "") for h in HEADERS} for r in rows]
    with ledger.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        w.writerows(cleaned)
