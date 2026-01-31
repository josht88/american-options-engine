#!/usr/bin/env python
"""
Simple macro-event calendar that returns a σ-multiplier for a given date.

Hard-coded events
-----------------
* US CPI (second Wednesday each month)
* FOMC statement (eight scheduled meetings per year)

Multiplier
----------
1.50  on event day
1.25  the day before / after
1.00  all other days
"""

import calendar
import datetime as dt
from functools import lru_cache

# ---------------------------------------------------------------------------#
# helpers
# ---------------------------------------------------------------------------#


def _nth_weekday(year: int, month: int, n: int, weekday: int) -> dt.date:
    """Return the *n*-th `weekday` (0=Mon) of a month."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday() + 7) % 7
    day = 1 + offset + 7 * (n - 1)
    return dt.date(year, month, day)


@lru_cache(None)
def _build_calendar(start_year: int = 2024, years_ahead: int = 5) -> set[dt.date]:
    """Pre-compute all CPI & FOMC dates into a set for O(1) look-ups."""
    cal: set[dt.date] = set()

    for y in range(start_year, start_year + years_ahead):
        # ---- CPI: 2nd Wednesday each month --------------------------------
        for m in range(1, 13):
            cal.add(_nth_weekday(y, m, 2, 2))  # 2 = Wednesday

        # ---- FOMC: Mar, May, Jun, Jul, Sep, Nov, Dec ----------------------
        for m in (3, 5, 6, 7, 9, 11, 12):
            # last Wednesday of the month (robust: works for 4- or 5-Wed months)
            last_dom = calendar.monthrange(y, m)[1]              # last day num
            last_day = dt.date(y, m, last_dom)
            back = (last_day.weekday() - 2) % 7                  # to Wednesday
            cal.add(last_day - dt.timedelta(days=back))

    return cal


# ---------------------------------------------------------------------------#
# public
# ---------------------------------------------------------------------------#


def sigma_multiplier(day: dt.date) -> float:
    """Return volatility multiplier for the given calendar *day*."""
    cal = _build_calendar()
    if day in cal:
        return 1.50
    if (day - dt.timedelta(days=1) in cal) or (day + dt.timedelta(days=1) in cal):
        return 1.25
    return 1.00
