#!/usr/bin/env python3
from __future__ import annotations

# ---------------- PATH BOOTSTRAP ----------------
from pathlib import Path
import sys as _sys

ROOT = Path(__file__).resolve().parent          # .../American_Options_Pricer_Model/web
PROJECT_ROOT = ROOT.parent                       # .../American_Options_Pricer_Model
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(SRC_DIR))

# ---------------- stdlib / 3rd-party ----------------
import os
import json
import pathlib
import datetime as dt
from typing import Iterable, Tuple, Dict, Any, List, Optional, Sequence

import pandas as pd
import streamlit as st

# Optional deps
try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

try:
    import plotly.express as px
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

# ---------------- project imports ----------------
# (imports from src/aoe/* now work thanks to the PATH bootstrap above)
from aoe.edge import compute_edge, EdgeResult  # noqa: F401  (used by tests and helpers)

from aoe.backtest.yfbt import backtest_universe  # type: ignore
from aoe.universe.select import score_universe  # type: ignore  # noqa: F401
from aoe.data.chain import scan_high_ev_contracts  # type: ignore
from aoe.strategy.spreads import build_range_spreads  # type: ignore  # noqa: F401
from aoe.pricing.spread import price_debit_spread  # type: ignore
from aoe.risk.sizer import kelly_size  # type: ignore  # noqa: F401
from aoe.execution.broker import execute_order  # type: ignore

# Optional optimizer
try:
    from scripts.optimize_ic import run_optimize as _run_optimize_ic  # type: ignore  # noqa: F401
except Exception:
    _run_optimize_ic = None  # type: ignore

# Calibrator / model
try:
    from scripts.calibrate import ensure_model  # type: ignore
except Exception:
    def ensure_model(_tk: str) -> Dict[str, Any]:
        return {"spot": None}

try:
    from scripts.screen_day import load_model  # type: ignore
except Exception:
    load_model = None  # type: ignore

try:
    from scripts import monitor_day as mon  # type: ignore
except Exception:
    mon = None  # type: ignore

from types import SimpleNamespace

__all__ = [
    "_ensure_ledger", "_equity_from_ledger", "_equity_from_backtest",
    "build_ic_from_chain_simple", "build_ic_grid_search",
    "_rehydrate_condors", "run_condor_diagnostic",
]

# ---------------- constants & small utils ----------------
DATE_FMT = "%Y-%m-%d"
DEFAULT_LEDGER = os.environ.get("AOE_LEDGER", str(PROJECT_ROOT / "data" / "pnl_paper.csv"))
DEFAULT_BANKROLL = 50_000.0
DEFAULT_KELLY_CAP = 0.05     # 5% of bankroll per trade
SHOW_DEBUG_DEFAULT = False

def is_streamlit_cloud() -> bool:
    # Streamlit Community Cloud sets these in many deployments
    return (
        os.getenv("STREAMLIT_SHARING") == "true"
        or os.getenv("STREAMLIT_CLOUD") == "true"
        or "streamlit.app" in os.getenv("HOSTNAME", "")
    )

DEMO_MODE_DEFAULT = is_streamlit_cloud()
def _is_running_under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ

def _parse_universe(text: str) -> List[str]:
    return [t.strip().upper() for t in text.split(",") if t.strip()]

def _horizon_to_dates(horizon: str) -> Tuple[str, str]:
    years = {"1y": 1, "3y": 3, "5y": 5, "10y": 10}.get(horizon, 1)
    end = dt.date.today()
    start = end.replace(year=end.year - years)
    return (start.strftime(DATE_FMT), end.strftime(DATE_FMT))

def _ensure_equity_df(equity_obj: Any) -> pd.DataFrame:
    """
    Return a DataFrame with ['date','equity'] (both present), numeric equity, datetime date, sorted.
    Accepts many shapes (Series, DF, dicts), falls back to a single zero row for today on error.
    """
    out = pd.DataFrame({"date": [], "equity": []})
    try:
        if isinstance(equity_obj, pd.Series):
            ser = equity_obj.copy()
            name = ser.name if isinstance(ser.name, str) and ser.name.strip() else "equity"
            if ser.index.name is None:
                ser.index.name = "date"
            df = ser.rename(name).reset_index()
        elif isinstance(equity_obj, pd.DataFrame):
            df = equity_obj.copy()
            if "equity" not in df.columns:
                if df.shape[1] == 1:
                    df.columns = ["equity"]
                else:
                    for cand in ["pnl", "PnL", "equity_curve", "value", "nav", "NAV"]:
                        if cand in df.columns:
                            df = df.rename(columns={cand: "equity"})
                            break
            if "date" not in df.columns:
                idx_name = df.index.name or "index"
                df = df.reset_index().rename(columns={idx_name: "date"})
                if "date" not in df.columns:
                    df.insert(0, "date", pd.RangeIndex(start=0, stop=len(df), step=1))
        else:
            df = out.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df = pd.DataFrame({"date": [pd.Timestamp.today().normalize()], "equity": [0.0]})

        if "equity" in df.columns:
            df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
        else:
            df["equity"] = 0.0

        df = df.loc[pd.notna(df["date"])].sort_values("date")
        return df[["date", "equity"]]
    except Exception:
        return pd.DataFrame({"date": [pd.Timestamp.today().normalize()], "equity": [0.0]})

@st.cache_data(show_spinner=False)
def _read_ledger_csv(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        return pd.DataFrame(
            columns=["ts", "ticker", "expiry", "right", "long_k", "short_k", "long_k2", "short_k2", "qty", "price", "mark", "realised", "tag"]
        )
    df = pd.read_csv(p)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for col in ("long_k", "short_k", "long_k2", "short_k2", "qty", "price", "mark", "realised"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _ensure_ledger(path: str) -> str:
    cols = [
        "ts", "ticker", "expiry", "right",
        "long_k", "short_k", "long_k2", "short_k2",
        "qty", "price", "mark", "realised", "tag"
    ]
    p = pathlib.Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        pd.DataFrame(columns=cols).to_csv(p, index=False)
    return str(p)

def _equity_from_ledger(path: str) -> pd.DataFrame:
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        return pd.DataFrame({"date": [dt.date.today()], "equity": [0.0]})

    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame({"date": [dt.date.today()], "equity": [0.0]})

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        df["ts"] = pd.Timestamp(dt.date.today())
    for c in ("qty","price","mark","realised"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    if "tag" not in df.columns:
        df["tag"] = ""

    closed = df[df["tag"].str.upper() == "CLOSED"].copy()
    real_by_day = closed.groupby(closed["ts"].dt.normalize())["realised"].sum().cumsum() if not closed.empty else pd.Series(dtype=float)

    open_legs = df[df["tag"].str.upper() == "OPEN"].copy()
    unreal_now = 0.0
    if not open_legs.empty:
        open_legs["unrl"] = (open_legs["mark"] - open_legs["price"]) * open_legs["qty"]
        unreal_now = float(open_legs["unrl"].sum())

    if not real_by_day.empty:
        eq_series = real_by_day.copy()
        eq_series.iloc[-1] = eq_series.iloc[-1] + unreal_now
        out = eq_series.reset_index()
        out.columns = ["date", "equity"]
    else:
        out = pd.DataFrame({"date": [dt.date.today()], "equity": [unreal_now]})

    out["date"] = pd.to_datetime(out["date"])
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce").fillna(0.0)
    return out[["date","equity"]]

def _equity_from_backtest(bt_res: dict | None) -> pd.DataFrame:
    if not isinstance(bt_res, dict):
        return pd.DataFrame({"date": [], "equity": []})
    return _ensure_equity_df(bt_res.get("equity", pd.DataFrame()))

def _mark_to_market(ledger_path: str) -> Tuple[bool, str]:
    # ensure file exists
    try:
        _ensure_ledger(ledger_path)
    except Exception as e:
        return False, f"Failed to initialize ledger: {e}"

    if mon is not None:
        try:
            os.environ["AOE_LEDGER"] = ledger_path
            mon.run()
            return True, "Mark-to-market complete (monitor_day)."
        except Exception:
            pass  # fall back below

    # local conservative MTM
    p = pathlib.Path(ledger_path).expanduser().resolve()
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return False, f"Failed to read ledger: {e}"

    if df.empty:
        return True, "Ledger empty (no positions to mark)."

    for c in ("tag", "price", "mark", "realised"):
        if c not in df.columns:
            df[c] = 0.0 if c in ("price", "mark", "realised") else ""
    df["tag"] = df["tag"].astype(str).str.upper()
    for c in ("price", "mark", "realised"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    open_mask = df["tag"] == "OPEN"
    df.loc[open_mask, "mark"] = df.loc[open_mask, "price"]

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df.loc[df["ts"].isna(), "ts"] = pd.Timestamp.utcnow()
    else:
        df.insert(0, "ts", pd.Timestamp.utcnow())

    try:
        df.to_csv(p, index=False)
        return True, "Mark-to-market complete (local fallback)."
    except Exception as e:
        return False, f"Failed to write ledger: {e}"

# ---------------- spot & chain helpers ----------------
def _resolve_spot(ticker: str, mdl: Any, singles: Any, snap: dict | None) -> float | None:
    if isinstance(snap, dict):
        v = snap.get("spot")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    if mdl is not None:
        v = getattr(mdl, "spot", None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        snap2 = getattr(mdl, "snapshot", None) or getattr(mdl, "__dict__", {}).get("snapshot")
        if isinstance(snap2, dict):
            v2 = snap2.get("spot")
            if isinstance(v2, (int, float)) and v2 > 0:
                return float(v2)
    try:
        if isinstance(singles, pd.DataFrame):
            for c in ("s0", "spot", "underlying"):
                if c in singles.columns and pd.api.types.is_numeric_dtype(singles[c]):
                    s = float(pd.to_numeric(singles[c], errors="coerce").median())
                    if s > 0:
                        return s
    except Exception:
        pass
    try:
        import yfinance as yf  # type: ignore
        info = yf.Ticker(ticker).history(period="5d")
        if not info.empty:
            last = float(info["Close"].iloc[-1])
            if last > 0:
                return last
    except Exception:
        pass
    return None

def _yf_fetch_chain_loose(ticker: str, target_dte_days: int) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return pd.DataFrame(columns=["expiry","right","strike","bid","ask","mid"])
    tk = yf.Ticker(ticker)
    expiries = tk.options or []
    if not expiries:
        return pd.DataFrame(columns=["expiry","right","strike","bid","ask","mid"])
    today = dt.date.today()

    def _dte_days(e):
        try:
            ed = dt.datetime.strptime(e, "%Y-%m-%d").date()
            return abs((ed - today).days - target_dte_days)
        except Exception:
            return 10_000

    chosen = min(expiries, key=_dte_days)
    try:
        ch = tk.option_chain(chosen)
    except Exception:
        return pd.DataFrame(columns=["expiry","right","strike","bid","ask","mid"])

    def _prep(df, right):
        if df is None or df.empty:
            return pd.DataFrame(columns=["expiry","right","strike","bid","ask","mid"])
        out = df[["strike","bid","ask"]].copy()
        out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
        out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
        out["mid"] = (out["bid"].fillna(0) + out["ask"].fillna(0)) / 2.0
        out["right"] = right
        out["expiry"] = chosen
        return out.reset_index(drop=True)

    puts  = _prep(ch.puts,  "P")
    calls = _prep(ch.calls, "C")
    chain = pd.concat([puts, calls], ignore_index=True)
    chain = chain.dropna(subset=["strike"]).copy()
    chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
    chain = chain.dropna(subset=["strike"])
    return chain[["expiry","right","strike","bid","ask","mid"]]

def _scan_high_ev_contracts_compat(ticker: str, p_tail: float | None = None, dte_days: int | None = None) -> pd.DataFrame:
    df = pd.DataFrame()
    try:
        if p_tail is not None:
            df = scan_high_ev_contracts(ticker, p_tail=p_tail)  # type: ignore[arg-type]
        else:
            df = scan_high_ev_contracts(ticker)
    except TypeError:
        try:
            df = scan_high_ev_contracts(ticker)
        except Exception:
            df = pd.DataFrame()
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = df.copy()
        for c in ["bid","ask","mid","mid_px"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        if "mid" not in out.columns and "mid_px" in out.columns:
            out["mid"] = out["mid_px"]
        return out
    loose = _yf_fetch_chain_loose(ticker, int(dte_days or 21))
    if loose.empty:
        return pd.DataFrame()
    return loose

# ---------- Iron condor helpers ----------
def _nearest_row_by_strike(df: pd.DataFrame, target_k: float) -> pd.Series:
    s = pd.to_numeric(df["strike"], errors="coerce")
    idx = (s - float(target_k)).abs().argsort().iloc[0]
    return df.iloc[idx]

def _leg_price(row: pd.Series, side: str, use_exec: bool) -> float:
    bid = pd.to_numeric(row.get("bid"), errors="coerce")
    ask = pd.to_numeric(row.get("ask"), errors="coerce")
    mid = pd.to_numeric(row.get("mid"), errors="coerce")
    if use_exec:
        if side == "sell":
            return float(bid if pd.notna(bid) else mid)
        else:
            return float(ask if pd.notna(ask) else mid)
    return float(mid if pd.notna(mid) else (bid + ask) / 2.0)

def _compute_ic_one(singles: pd.DataFrame, spot: float, wing: float, slippage_bps: float,
                    fee_per_contract: float, width_tol: float, credit_leeway: float,
                    use_exec_prices: bool) -> tuple[dict, list[str]]:
    flags: list[str] = []
    df = singles.copy()
    if "mid" not in df.columns:
        df["mid"] = df.get("mid_px")
    df = df.dropna(subset=["expiry","right","strike"]).copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["mid"]    = pd.to_numeric(df["mid"], errors="coerce")
    df = df.dropna(subset=["strike"])
    if df.empty:
        return {}, ["empty_chain"]

    expiry = df.groupby("expiry")["strike"].count().sort_values(ascending=False).index[0]
    sub = df[df["expiry"] == expiry].copy()
    puts  = sub[sub["right"] == "P"].sort_values("strike").reset_index(drop=True)
    calls = sub[sub["right"] == "C"].sort_values("strike").reset_index(drop=True)
    if puts.empty or calls.empty:
        return {}, ["no_puts_or_calls"]

    sp_put  = float(_nearest_row_by_strike(puts,  spot)["strike"])
    sp_call = float(_nearest_row_by_strike(calls, spot)["strike"])

    lp_row = _nearest_row_by_strike(puts,  sp_put  - wing)
    lc_row = _nearest_row_by_strike(calls, sp_call + wing)
    lp = float(lp_row["strike"])
    lc = float(lc_row["strike"])

    sp_row = _nearest_row_by_strike(puts,  sp_put)
    sc_row = _nearest_row_by_strike(calls, sp_call)

    sp_mid = _leg_price(sp_row, "sell", use_exec=False)
    sc_mid = _leg_price(sc_row, "sell", use_exec=False)
    lp_mid = _leg_price(lp_row, "buy",  use_exec=False)
    lc_mid = _leg_price(lc_row, "buy",  use_exec=False)
    builder_credit = max(0.0, (sp_mid + sc_mid) - (lp_mid + lc_mid))

    sp_exe = _leg_price(sp_row, "sell", use_exec=True)
    sc_exe = _leg_price(sc_row, "sell", use_exec=True)
    lp_exe = _leg_price(lp_row, "buy",  use_exec=True)
    lc_exe = _leg_price(lc_row, "buy",  use_exec=True)
    exec_credit_raw = (sp_exe + sc_exe) - (lp_exe + lc_exe)

    if pd.notna(exec_credit_raw):
        exec_credit_slip = float(exec_credit_raw) * (1.0 - float(slippage_bps) / 10_000.0)
    else:
        exec_credit_slip = float("nan")
    exec_credit = (exec_credit_slip - 4.0 * float(fee_per_contract)) if pd.notna(exec_credit_slip) else float("nan")
    exec_credit = max(0.0, float(exec_credit)) if pd.notna(exec_credit) else float("nan")

    width_put  = abs(sp_put - lp)
    width_call = abs(lc - sp_call)
    width = min(width_put, width_call)
    if abs(width_put - width_call) > float(width_tol) + 1e-9:
        flags.append(f"wing_asym|put={width_put:.2f}|call={width_call:.2f}")

    if pd.notna(builder_credit) and builder_credit > width * (1.0 + float(credit_leeway)):
        flags.append("builder_credit>width")
    if pd.notna(exec_credit) and exec_credit > width * (1.0 + float(credit_leeway)):
        flags.append("exec_credit>width")

    use_credit = exec_credit if (pd.notna(exec_credit) and exec_credit >= 0) else builder_credit
    risk = max(0.0, width - use_credit)
    if risk <= 0:
        flags.append("non_positive_risk")

    edge_pct = (use_credit / width) * 100.0 if width > 0 else float("nan")

    row = dict(
        expiry=str(expiry),
        long_put=float(lp), short_put=float(sp_put), short_call=float(sp_call), long_call=float(lc),
        builder_credit=float(round(builder_credit, 4)),
        exec_credit=float(round(exec_credit, 4)) if pd.notna(exec_credit) else float("nan"),
        credit=float(round(use_credit, 4)),
        width_put=float(round(width_put, 4)),
        width_call=float(round(width_call, 4)),
        width=float(round(width, 4)),
        risk=float(round(risk, 4)),
        edge_pct=float(round(edge_pct, 2)),
        sp_put_mid=float(round(sp_mid, 4)), sp_call_mid=float(round(sc_mid, 4)),
        lp_mid=float(round(lp_mid, 4)),     lc_mid=float(round(lc_mid, 4)),
        sp_put_exe=float(round(sp_exe, 4)), sc_call_exe=float(round(sc_exe, 4)),
        lp_exe=float(round(lp_exe, 4)),     lc_exe=float(round(lc_exe, 4)),
    )
    return row, flags

def _get_leg_quotes(singles: pd.DataFrame, right: str, strike: float) -> Tuple[float, float, float]:
    sub = singles[singles["right"] == right]
    row = _nearest_row_by_strike(sub, strike)
    if row.empty:
        return float("nan"), float("nan"), float("nan")
    bid = float(pd.to_numeric(pd.Series([row.get("bid", float("nan"))]), errors="coerce").iloc[0])
    ask = float(pd.to_numeric(pd.Series([row.get("ask", float("nan"))]), errors="coerce").iloc[0])
    mid = float(pd.to_numeric(pd.Series([row.get("mid", row.get("mid_px", float("nan")))]), errors="coerce").iloc[0])
    return bid, ask, mid

def _recompute_credit_variants(singles: pd.DataFrame,
                               long_put: float, short_put: float,
                               short_call: float, long_call: float) -> Dict[str, float]:
    sp_bid, sp_ask, sp_mid = _get_leg_quotes(singles, "P", short_put)
    lp_bid, lp_ask, lp_mid = _get_leg_quotes(singles, "P", long_put)
    sc_bid, sc_ask, sc_mid = _get_leg_quotes(singles, "C", short_call)
    lc_bid, lc_ask, lc_mid = _get_leg_quotes(singles, "C", long_call)

    credit_mid = sp_mid + sc_mid - lp_mid - lc_mid
    credit_cons = sp_bid + sc_bid - lp_ask - lc_ask
    credit_aggr = sp_ask + sc_ask - lp_bid - lc_bid

    return dict(
        credit_ps_mid=float(credit_mid),
        credit_ps_cons=float(credit_cons),
        credit_ps_aggr=float(credit_aggr),
        legs_bidaskmid=dict(
            sp=(sp_bid, sp_ask, sp_mid),
            lp=(lp_bid, lp_ask, lp_mid),
            sc=(sc_bid, sc_ask, sc_mid),
            lc=(lc_bid, lc_ask, lc_mid),
        ),
    )

def _condor_widths(long_put: float, short_put: float, short_call: float, long_call: float) -> Tuple[float, float, float]:
    pw = abs(float(short_put) - float(long_put))
    cw = abs(float(long_call) - float(short_call))
    return pw, cw, max(pw, cw)

def _sanity_gate_row(long_put: float, short_put: float, short_call: float, long_call: float,
                     credit_ps: float, *, tol: float = 0.10, edge_cap: Optional[float] = 120.0) -> Tuple[bool, str]:
    pw, cw, w = _condor_widths(long_put, short_put, short_call, long_call)
    if w <= 0:
        return False, "zero_or_negative_width"
    if abs(pw - cw) > tol:
        return False, "unequal_wings"
    if credit_ps > w + 1e-6:
        return False, "credit_gt_width"
    if edge_cap is not None and credit_ps / w * 100.0 > edge_cap:
        return False, "edge_too_large"
    return True, ""

def _per_contract(x: float) -> float:
    return float(x) * 100.0

def _breakevens(short_put: float, short_call: float, width: float, credit_ps_mid: float) -> Tuple[float, float]:
    lower = float(short_put) - float(width - credit_ps_mid)
    upper = float(short_call) + float(width - credit_ps_mid)
    return lower, upper

def _rr_ratio(credit_ps: float, max_loss_ps: float) -> float:
    if max_loss_ps <= 0:
        return float("inf")
    return float(credit_ps / max_loss_ps)

def _theo_credit_from_model(mdl: Any,
                            long_put: float, short_put: float,
                            short_call: float, long_call: float) -> float:
    try:
        Dp = float(price_debit_spread(model=mdl, right="P", long_k=float(short_put), short_k=float(long_put)))  # type: ignore
        put_width = abs(float(short_put) - float(long_put))
        theo_put_credit = put_width - Dp
    except Exception:
        theo_put_credit = float("nan")
    try:
        Dc = float(price_debit_spread(model=mdl, right="C", long_k=float(long_call), short_k=float(short_call)))  # type: ignore
        call_width = abs(float(long_call) - float(short_call))
        theo_call_credit = call_width - Dc
    except Exception:
        theo_call_credit = float("nan")

    if HAVE_NP:
        if not np.isfinite(theo_put_credit): theo_put_credit = float("nan")
        if not np.isfinite(theo_call_credit): theo_call_credit = float("nan")

    return float(theo_put_credit + theo_call_credit)

# -------- payoff & PNL viz helpers --------
def _ic_payoff_at_expiry(S: "np.ndarray", lp: float, sp: float, sc: float, lc: float, credit_ps: float) -> "np.ndarray":
    """
    Expiry P&L per share for an iron condor (sell 2 shorts, buy wings),
    positive = profit. Vectorized over price grid S.
    """
    if not HAVE_NP:
        raise RuntimeError("NumPy required for payoff chart.")
    S = np.asarray(S, dtype=float)
    # Short put spread loss when S < sp (bounded by put width)
    put_loss = np.clip(sp - S, 0, sp - lp)
    # Short call spread loss when S > sc (bounded by call width)
    call_loss = np.clip(S - sc, 0, lc - sc)
    loss = put_loss + call_loss
    return credit_ps - loss  # per share

def _ic_mark_value(model: Any, S: float, ttm_years: float,
                   lp: float, sp: float, sc: float, lc: float) -> float:
    try:
        put_w  = abs(sp - lp)
        call_w = abs(lc - sc)
        Dp = float(price_debit_spread(model=model, right="P", long_k=float(sp), short_k=float(lp)))
        Dc = float(price_debit_spread(model=model, right="C", long_k=float(lc), short_k=float(sc)))
        return float((put_w - Dp) + (call_w - Dc))
    except Exception:
        return float("nan")

def _render_pnl_over_time(row: pd.Series, mdl: Any, fees_per_contract: float) -> None:
    if not (HAVE_PLOTLY and HAVE_NP):
        st.info("Install `plotly` and `numpy` to see interactive PNL charts.")
        return
    import plotly.express as px

    lp, sp, sc, lc = float(row.long_put), float(row.short_put), float(row.short_call), float(row.long_call)
    credit_ps = float(row.credit)
    width = min(sp - lp, lc - sc)

    days = st.slider("Days to Expiry (projection)", min_value=0, max_value=60, value=15, step=1, key="pnl_days")
    qty  = st.number_input("Qty (contracts)", 1, 2000, int(row.get("qty", 1)), key="pnl_qty")
    ttm_years = max(0.0, days) / 365.0

    center = (sp + sc) / 2.0
    S_grid = np.linspace(center - 15*width, center + 15*width, 401)

    pnl_exp_ps = _ic_payoff_at_expiry(S_grid, lp, sp, sc, lc, credit_ps)
    pnl_exp = (pnl_exp_ps*100.0 - fees_per_contract*4.0) * int(qty)

    curves = [{"S": S_grid, "PNL": pnl_exp, "when": "At Expiry"}]
    v_now = _ic_mark_value(mdl, S=S_grid[len(S_grid)//2], ttm_years=ttm_years, lp=lp, sp=sp, sc=sc, lc=lc)
    if pd.notna(v_now):
        pnl_now = np.repeat((v_now - credit_ps)*100.0 - fees_per_contract*4.0, len(S_grid)) * int(qty)
        curves.append({"S": S_grid, "PNL": pnl_now, "when": f"TTE ≈ {days}d"})

    dfp = pd.concat([pd.DataFrame({"S": c["S"], "PNL": c["PNL"], "when": c["when"]}) for c in curves], ignore_index=True)
    fig = px.line(dfp, x="S", y="PNL", color="when", title="PNL Projection")
    fig.add_vline(x=sp, line_dash="dot"); fig.add_vline(x=sc, line_dash="dot")
    be_low = sp - (width - credit_ps); be_high = sc + (width - credit_ps)
    fig.add_vline(x=be_low, line_dash="dash"); fig.add_vline(x=be_high, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Projection uses expiry payoff (solid). Mid-time curve is illustrative when a model is available.")

# ---------------- compute edge/rehydrate ----------------
def _edge_and_checks_all(row: pd.Series, singles: pd.DataFrame,
                         mdl: Any,
                         edge_cap: float, wing_tol: float, credit_leeway: float) -> Dict[str, Any]:
    lp, sp, sc, lc = float(row["long_put"]), float(row["short_put"]), float(row["short_call"]), float(row["long_call"])

    cr = _recompute_credit_variants(singles, lp, sp, sc, lc)
    credit_ps_mid = float(cr["credit_ps_mid"])
    credit_ps_cons = float(cr["credit_ps_cons"])
    credit_ps_aggr = float(cr["credit_ps_aggr"])
    legs_q = cr["legs_bidaskmid"]

    pw, cw, w = _condor_widths(lp, sp, sc, lc)
    ok, reason = _sanity_gate_row(lp, sp, sc, lc, credit_ps_mid, tol=wing_tol, edge_cap=edge_cap)
    if ok is False and reason == "credit_gt_width" and (credit_ps_mid <= w + credit_leeway):
        ok, reason = True, ""

    edge_pct_mid = (100.0 * credit_ps_mid / w) if w > 0 else float("nan")
    edge_pct_cons = (100.0 * credit_ps_cons / w) if w > 0 else float("nan")
    edge_pct_aggr = (100.0 * credit_ps_aggr / w) if w > 0 else float("nan")

    credit_ps_theo = float("nan")
    try:
        if mdl is not None:
            credit_ps_theo = _theo_credit_from_model(mdl, lp, sp, sc, lc)
    except Exception:
        pass
    theo_edge_pct = (100.0 * credit_ps_theo / w) if (w > 0 and pd.notna(credit_ps_theo)) else float("nan")

    max_loss_ps = max(0.0, w - credit_ps_mid)
    rr = _rr_ratio(credit_ps_mid, max_loss_ps)
    be_low, be_high = _breakevens(sp, sc, w, credit_ps_mid)

    return dict(
        credit_ps_mid=credit_ps_mid,
        credit_ps_cons=credit_ps_cons,
        credit_ps_aggr=credit_ps_aggr,
        credit_ps_theo=credit_ps_theo,
        edge_pct_mid=edge_pct_mid,
        edge_pct_cons=edge_pct_cons,
        edge_pct_aggr=edge_pct_aggr,
        theo_edge_pct=theo_edge_pct,
        put_width=pw, call_width=cw, width=w,
        max_loss_ps=max_loss_ps, rr=rr,
        be_low=be_low, be_high=be_high,
        pass_gates=bool(ok),
        fail_reason=str(reason),
        legs_q=legs_q,
    )

def _rehydrate_condors(singles: pd.DataFrame, base: pd.DataFrame, mdl: Any,
                       edge_cap: float, wing_tol: float, credit_leeway: float) -> pd.DataFrame:
    if base is None or base.empty:
        return pd.DataFrame()
    rows = []
    for _, r in base.iterrows():
        info = _edge_and_checks_all(r, singles, mdl, edge_cap=edge_cap, wing_tol=wing_tol, credit_leeway=credit_leeway)
        row = dict(r)
        row.update(dict(
            builder_credit=float(info["credit_ps_mid"]),
            exec_credit=float(info["credit_ps_cons"]),  # conservative proxy
            credit=float(info["credit_ps_mid"]),
            width=float(info["width"]),
            risk=float(info["max_loss_ps"]),
            edge_pct=float(info["edge_pct_mid"]),
            be_low=float(info["be_low"]),
            be_high=float(info["be_high"]),
            rr=float(info["rr"]),
            pass_gates=bool(info["pass_gates"]),
            fail_reason=str(info["fail_reason"]),
        ))
        rows.append(row)
    df = pd.DataFrame(rows)
    if "pass_gates" in df.columns:
        df["pass_gates"] = df["pass_gates"].map(lambda x: True if bool(x) else False).astype("object")
    return df

# ---------------- programmatic screen API (used in tests) ----------------
def build_ic_from_chain_simple(
    singles: pd.DataFrame,
    spot: float | None,
    wing: float,
    dte_days: int,
    *,
    slippage_bps: float = 20.0,
    fee_per_contract: float = 0.65,
    width_tol: float = 0.10,
    credit_leeway: float = 0.05,
    use_exec_prices: bool = True,
) -> pd.DataFrame:
    if not isinstance(spot, (int, float)) or spot <= 0:
        try:
            allk = pd.to_numeric(singles["strike"], errors="coerce").dropna().sort_values()
            if allk.empty:
                return pd.DataFrame(columns=["expiry","long_put","short_put","short_call","long_call","credit","max_gain","max_loss"])
            spot = float(allk.iloc[len(allk)//2])
        except Exception:
            return pd.DataFrame(columns=["expiry","long_put","short_put","short_call","long_call","credit","max_gain","max_loss"])

    row, flags = _compute_ic_one(
        singles, float(spot), float(wing), float(slippage_bps),
        float(fee_per_contract), float(width_tol), float(credit_leeway),
        bool(use_exec_prices),
    )
    if not row:
        return pd.DataFrame(columns=["expiry","long_put","short_put","short_call","long_call","credit","max_gain","max_loss"])

    return pd.DataFrame([{
        "expiry": row["expiry"],
        "long_put": row["long_put"], "short_put": row["short_put"],
        "short_call": row["short_call"], "long_call": row["long_call"],
        "credit": row["credit"],  # per-share
        "max_gain": row["credit"],
        "max_loss": max(0.0, row["width"] - row["credit"]),
        "builder_credit": row["builder_credit"],
        "exec_credit": row["exec_credit"],
        "width_put": row["width_put"], "width_call": row["width_call"], "width": row["width"],
        "risk": row["risk"], "edge_pct": row["edge_pct"],
        "sp_put_mid": row["sp_put_mid"], "sp_call_mid": row["sp_call_mid"],
        "lp_mid": row["lp_mid"], "lc_mid": row["lc_mid"],
        "sp_put_exe": row["sp_put_exe"], "sc_call_exe": row["sc_call_exe"],
        "lp_exe": row["lp_exe"], "lc_exe": row["lc_exe"],
        "flags": "|".join(flags) if flags else "",
    }])

def build_ic_grid_search(
    singles: pd.DataFrame,
    spot: float | None,
    dte_days: int,
    wings: Iterable[float] = (2,3,5,10),
    top_n: int = 10,
    *,
    slippage_bps: float = 20.0,
    fee_per_contract: float = 0.65,
    width_tol: float = 0.10,
    credit_leeway: float = 0.05,
    use_exec_prices: bool = True,
) -> pd.DataFrame:
    if not isinstance(spot, (int, float)) or spot <= 0:
        try:
            allk = pd.to_numeric(singles["strike"], errors="coerce").dropna().sort_values()
            if allk.empty:
                return pd.DataFrame(columns=["expiry","long_put","short_put","short_call","long_call","credit","max_gain","max_loss"])
            spot = float(allk.iloc[len(allk)//2])
        except Exception:
            return pd.DataFrame(columns=["expiry","long_put","short_put","short_call","long_call","credit","max_gain","max_loss"])

    rows: list[dict] = []
    for w in wings:
        row, flags = _compute_ic_one(
            singles, float(spot), float(w), float(slippage_bps),
            float(fee_per_contract), float(width_tol), float(credit_leeway),
            bool(use_exec_prices),
        )
        if not row:
            continue
        rows.append({
            "expiry": row["expiry"],
            "long_put": row["long_put"], "short_put": row["short_put"],
            "short_call": row["short_call"], "long_call": row["long_call"],
            "credit": row["credit"],
            "max_gain": row["credit"],
            "max_loss": max(0.0, row["width"] - row["credit"]),
            "builder_credit": row["builder_credit"],
            "exec_credit": row["exec_credit"],
            "width_put": row["width_put"], "width_call": row["width_call"], "width": row["width"],
            "risk": row["risk"], "edge_pct": row["edge_pct"], "wing": float(w),
            "sp_put_mid": row["sp_put_mid"], "sp_call_mid": row["sp_call_mid"],
            "lp_mid": row["lp_mid"], "lc_mid": row["lc_mid"],
            "sp_put_exe": row["sp_put_exe"], "sc_call_exe": row["sc_call_exe"],
            "lp_exe": row["lp_exe"], "lc_exe": row["lc_exe"],
            "flags": "|".join(flags) if flags else "",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    score = out["exec_credit"].where(out["exec_credit"] > 0, out["builder_credit"])
    return out.assign(_score=score).sort_values("_score", ascending=False).drop(columns=["_score"]).head(int(top_n))

def _build_range_spreads_compat(singles, model, wing: float | int | None = None):
    try:
        return build_range_spreads(singles, model, wing=wing)
    except TypeError:
        try:
            return build_range_spreads(singles, model, width=wing)
        except TypeError:
            return build_range_spreads(singles, model)

# ---------------- diagnostics adapter (used in tests) ----------------
def run_condor_diagnostic(
    *,
    tickers: Sequence[str],
    start_date,
    end_date,
    dte_days: int = 14,
    wings: Sequence[float] = (5.0,),
    min_edge_pct: float = 0.0,
    limit_days: int = 5,
    slippage_bps: float = 20.0,
    fee_per_contract: float = 0.65,
    wing_tol: float = 0.10,
    credit_leeway: float = 0.05,
    **kwargs,
) -> pd.DataFrame:
    try:
        wing = float(wings[0]) if len(wings) else 5.0
    except Exception:
        wing = 5.0

    rows: list[dict] = []
    today = dt.date.today().isoformat()

    for tk in [str(t).upper() for t in tickers]:
        try:
            singles = _scan_high_ev_contracts_compat(tk, p_tail=0.20, dte_days=int(dte_days))
        except Exception:
            singles = pd.DataFrame()
        chain_rows = int(len(singles)) if isinstance(singles, pd.DataFrame) else 0

        try:
            snap = ensure_model(tk)
        except Exception:
            snap = {"spot": None}
        try:
            mdl = load_model(tk, spot=snap.get("spot") if isinstance(snap, dict) else None, max_age_days=5) if load_model else None
        except Exception:
            mdl = None

        spot_val = _resolve_spot(tk, mdl, singles, snap)

        condors = build_ic_from_chain_simple(
            singles, spot=spot_val, wing=float(wing), dte_days=int(dte_days),
            slippage_bps=float(slippage_bps), fee_per_contract=float(fee_per_contract),
            width_tol=float(wing_tol), credit_leeway=float(credit_leeway), use_exec_prices=True
        )
        if condors.empty:
            condors = build_ic_grid_search(
                singles, spot=spot_val, dte_days=int(dte_days),
                wings=(max(1.0, wing-2), wing, wing+2),
                top_n=5,
                slippage_bps=float(slippage_bps), fee_per_contract=float(fee_per_contract),
                width_tol=float(wing_tol), credit_leeway=float(credit_leeway), use_exec_prices=True
            )

        pass_rows, reason = 0, ""
        if isinstance(condors, pd.DataFrame) and not condors.empty:
            c2 = condors.copy()
            for c in ("edge_pct","credit","width","risk"):
                c2[c] = pd.to_numeric(c2.get(c), errors="coerce")
            c2 = c2[(c2["credit"] >= 0) & (c2["risk"] > 0) & (c2["edge_pct"] >= float(min_edge_pct))]
            pass_rows = int(len(c2))
            if pass_rows == 0:
                reason = "no_candidates"
        else:
            reason = "empty_chain"

        condor_rows = int(len(condors)) if isinstance(condors, pd.DataFrame) else 0

        rows.append({
            "date": today,
            "ticker": tk,
            "chain_rows": int(chain_rows),
            "condor_rows": int(condor_rows),
            "pass_rows": int(pass_rows),
            "reason": reason,
        })

    return pd.DataFrame(rows, columns=["date","ticker","chain_rows","condor_rows","pass_rows","reason"])

# ---------------- UI: sidebar ----------------
def sidebar_global() -> tuple[str, float, float, float, float, bool]:
    st.sidebar.header("Global")
    ledger_path = st.sidebar.text_input(
        "Ledger CSV (AOE_LEDGER)",
        value=os.environ.get("AOE_LEDGER", DEFAULT_LEDGER),
        key="glob_ledger_path",
    )
    os.environ["AOE_LEDGER"] = ledger_path

    st.sidebar.subheader("Execution & Costs")
    slippage_bps = st.sidebar.number_input("Slippage (bps, applied to credit)", 0.0, 200.0, 20.0, step=1.0, key="glob_slip")
    fee_per_contract = st.sidebar.number_input("Commission per contract ($)", 0.0, 5.0, 0.65, step=0.01, key="glob_fee")

    st.sidebar.subheader("Debug / Safety")
    edge_cap = st.sidebar.number_input("Edge% cap (sanity)", 10.0, 300.0, 120.0, step=5.0, key="glob_edgecap")
    wing_tol = st.sidebar.number_input("Wing equality tolerance ($)", 0.0, 2.0, 0.10, step=0.05, key="glob_wingtol")
    credit_leeway = st.sidebar.number_input("Credit ≤ width leeway", 0.0, 0.50, 0.05, step=0.01, key="glob_crleeway")
    demo_mode = st.sidebar.checkbox(
    "Demo mode (disable live Yahoo/yfinance)",
    value=DEMO_MODE_DEFAULT,
    help="Recommended on Streamlit Cloud. Run locally for live data.",
    )

    if st.sidebar.button("Mark-to-Market Now", key="glob_mtm_btn"):
        ok, msg = _mark_to_market(ledger_path)
        (st.sidebar.success if ok else st.sidebar.error)(msg)
    st.sidebar.caption(f"Ledger file: {ledger_path}")
    st.sidebar.divider()

    return ledger_path, float(slippage_bps), float(fee_per_contract), float(wing_tol), float(credit_leeway), bool(demo_mode)

# ---------------- UI: Screen ----------------
def _build_ic_order_namespace(row) -> Any:
    return SimpleNamespace(
        ticker=str(row.ticker),
        expiry=str(row.expiry),
        right="CONDOR",
        long_k=float(row.long_put),
        short_k=float(row.short_put),
        long_k2=float(row.long_call),
        short_k2=float(row.short_call),
        qty=int(row.qty),
        limit=float(row.credit),
        price=float(row.credit),
        time_in_force="DAY",
        account="PAPER",
        strike=None,
        side="SELL",
        tag="OPEN",
    )

def _paper_append_order(order_obj: Any, ledger_path: str) -> None:
    _ensure_ledger(ledger_path)
    p = pathlib.Path(ledger_path).expanduser().resolve()

    def _get(name, default=None):
        if hasattr(order_obj, name):
            return getattr(order_obj, name)
        if isinstance(order_obj, dict):
            return order_obj.get(name, default)
        return default

    now = pd.Timestamp.utcnow()
    row = {
        "ts": now,
        "ticker": str(_get("ticker", "")),
        "expiry": str(_get("expiry", "")),
        "right": "IC",
        "long_k": float(_get("long_k", 0.0)),
        "short_k": float(_get("short_k", 0.0)),
        "long_k2": float(_get("long_k2", 0.0)),
        "short_k2": float(_get("short_k2", 0.0)),
        "qty": int(_get("qty", 0)),
        "price": float(_get("price", _get("limit", 0.0))),
        "mark": float("nan"),
        "realised": 0.0,
        "tag": "OPEN",
    }

    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(p, index=False)

def _demo_candidates(tickers: list[str], dte_days: int, wing_width: float, fee_per_contract: float) -> pd.DataFrame:
    # Simple, deterministic rows so the UI always shows "working" output
    base = []
    tickers = tickers or ["SPY", "QQQ", "AAPL"]
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = tickers[:6]  # keep small

    for tk in tickers:
        # Fake strikes centered around a dummy spot of 100 for consistency
        spot = 100.0
        sp = 100.0
        lp = sp - float(wing_width)
        sc = 100.0
        lc = sc + float(wing_width)

        credit = round(max(0.10, 0.18 * float(wing_width)), 2)  # looks reasonable
        max_loss = round(max(0.01, float(wing_width) - credit), 2)

        base.append(dict(
            select=False,
            ticker=tk,
            expiry=(dt.date.today() + dt.timedelta(days=int(dte_days))).isoformat(),
            right="CONDOR",
            short_put=sp,
            long_put=lp,
            short_call=sc,
            long_call=lc,
            credit=float(credit),
            max_loss=float(max_loss),
            fee_per_contract=float(fee_per_contract),
            portfolio_pct=int(DEFAULT_KELLY_CAP * 100),
            qty=1,
        ))

    return pd.DataFrame(base)


def tab_screen(slippage_bps: float, fee_per_contract: float, wing_tol: float, credit_leeway: float, demo_mode: bool) -> None:
    st.subheader("Daily Screening — Iron Condors")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        uni_raw = st.text_input("Universe (comma-sep)", "AAPL,MSFT,SPY,QQQ,NVDA", key="scr_universe")
    with c2:
        dte_days = st.number_input("DTE (days)", 7, 60, 15, key="scr_dte")
    with c3:
        wing_width = st.number_input("Wing width ($)", 1.0, 25.0, 5.0, step=0.5, key="scr_wing")

    p_tail = st.slider("p_tail (quantile for shorts)", 0.05, 0.35, 0.20, key="scr_ptail")
    min_edge_pct = st.slider("Min edge % (credit/width, based on MID)", 0.0, 50.0, 5.0, key="scr_edge_min")

    run_btn = st.button("Run Screening", key="scr_run_btn")
    if demo_mode:
        st.caption("Demo mode is ON (no live Yahoo/yfinance calls). Run locally and disable demo mode for live chains.")
    else:
        st.caption("Models are used if available; otherwise we fall back to raw chains via yfinance.")
    
        
    if run_btn:
        tickers = _parse_universe(uni_raw)
        if not tickers:
            st.warning("Please enter at least one ticker.")
            st.session_state["scr_candidates_df"] = pd.DataFrame()
        else:
            # --- DEMO MODE: never touch yfinance / scan_high_ev_contracts ---
            if demo_mode:
                st.info("Demo mode: showing sample candidates (no live data calls).")
                st.session_state["scr_candidates_df"] = _demo_candidates(
                    tickers=tickers,
                    dte_days=int(dte_days),
                    wing_width=float(wing_width),
                    fee_per_contract=float(fee_per_contract),
                )
            else:
                # --- LIVE MODE (best effort) ---
                rows: list[dict] = []
                for tk in tickers:
                    try:
                        snap = ensure_model(tk)
                    except Exception:
                        snap = {"spot": None}
    
                    try:
                        mdl = load_model(tk, spot=snap.get("spot") if isinstance(snap, dict) else None, max_age_days=5) if load_model else None
                    except Exception:
                        mdl = None
    
                    try:
                        singles = _scan_high_ev_contracts_compat(tk, p_tail=p_tail, dte_days=int(dte_days))
                    except Exception as e:
                        singles = pd.DataFrame()
                        # avoid log spam: show one-line warning in UI per ticker
                        st.warning(f"{tk}: live chain fetch failed (likely Yahoo/yfinance blocked on cloud). Error: {e}")
    
                    spot_val = _resolve_spot(tk, mdl, singles, snap)
    
                    try:
                        condors = build_ic_from_chain_simple(
                            singles, spot=spot_val, wing=wing_width, dte_days=int(dte_days),
                            slippage_bps=slippage_bps, fee_per_contract=fee_per_contract,
                            width_tol=wing_tol, credit_leeway=credit_leeway, use_exec_prices=True
                        )
                        if condors.empty:
                            condors = build_ic_grid_search(
                                singles, spot=spot_val, dte_days=int(dte_days),
                                wings=(max(1.0, wing_width-2), wing_width, wing_width+2),
                                top_n=5,
                                slippage_bps=slippage_bps, fee_per_contract=fee_per_contract,
                                width_tol=wing_tol, credit_leeway=credit_leeway, use_exec_prices=True
                            )
                    except Exception as e:
                        st.warning(f"{tk}: model/spread construction failed. Error: {e}")
                        continue
    
                    if condors.empty:
                        continue
    
                    c2df = condors.copy()
                    for c in ("edge_pct","credit","width","risk"):
                        c2df[c] = pd.to_numeric(c2df.get(c), errors="coerce")
                    c2df = c2df[(c2df["credit"] >= 0) & (c2df["risk"] > 0) & (c2df["edge_pct"] >= float(min_edge_pct))]
                    if c2df.empty:
                        continue
    
                    score = pd.to_numeric(c2df.get("exec_credit"), errors="coerce").fillna(0)
                    score = score.where(score > 0, pd.to_numeric(c2df.get("builder_credit"), errors="coerce").fillna(0))
                    c2df = c2df.assign(_score=score).sort_values("_score", ascending=False)
    
                    top = c2df.iloc[0]
                    per_contract_risk = float(top["risk"])
                    qty = int(max(0, (DEFAULT_BANKROLL * DEFAULT_KELLY_CAP) // per_contract_risk)) if per_contract_risk > 0 else 0
    
                    rows.append(dict(
                        select=False,
                        ticker=tk,
                        expiry=str(top["expiry"]),
                        right="CONDOR",
                        short_put=float(top["short_put"]),
                        long_put=float(top["long_put"]),
                        short_call=float(top["short_call"]),
                        long_call=float(top["long_call"]),
                        credit=float(round(top["credit"], 2)),
                        max_loss=float(round(per_contract_risk, 2)),
                        fee_per_contract=float(fee_per_contract),
                        portfolio_pct=int(DEFAULT_KELLY_CAP * 100),
                        qty=int(qty),
                    ))
    
                st.session_state["scr_candidates_df"] = pd.DataFrame(rows)


    df = st.session_state.get("scr_candidates_df", pd.DataFrame())
    if df.empty:
        st.info("No candidates yet. Click **Run Screening**.")
        return

    st.write("**Candidates** (edit qty / select to place):")
    edited = st.data_editor(
        df,
        key="scr_table",
        hide_index=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "portfolio_pct": st.column_config.NumberColumn("% of portfolio", format="%d%%", disabled=True),
            "fee_per_contract": st.column_config.NumberColumn("Contract Fee ($)", format="%.2f", disabled=True),
            "credit": st.column_config.NumberColumn("Credit ($)", format="%.2f", disabled=True),
            "max_loss": st.column_config.NumberColumn("Max Loss ($)", format="%.2f", disabled=True),
            "qty": st.column_config.NumberColumn("Qty", min_value=0, step=1),
        },
        use_container_width=True,
    )

    if st.button("Place Selected", key="scr_place_btn"):
        placed = 0
        paper_filled = 0
        errors: list[str] = []

        ledger_path = os.environ.get("AOE_LEDGER", DEFAULT_LEDGER)
        _ensure_ledger(ledger_path)

        for _, r in edited[edited["select"]].iterrows():
            try:
                order_obj = _build_ic_order_namespace(r)
                try:
                    execute_order(order_obj)         # attribute-style
                except AttributeError:
                    execute_order(vars(order_obj))   # dict-style
                placed += 1
            except Exception as e:
                try:
                    _paper_append_order(order_obj if "order_obj" in locals() else r.to_dict(), ledger_path)
                    paper_filled += 1
                except Exception as e2:
                    errors.append(f"{r.ticker}: {e} (and paper append failed: {e2})")
                else:
                    errors.append(f"{r.ticker}: {e} (recorded to paper ledger)")

        if placed:
            st.success(f"Placed {placed} orders.")
        if paper_filled and not placed:
            st.info(f"Recorded {paper_filled} paper fills (broker not wired).")
        elif paper_filled:
            st.info(f"{paper_filled} orders also recorded to paper ledger.")

        if placed or paper_filled:
            edited["select"] = False
            st.session_state["scr_candidates_df"] = edited

        for msg in errors:
            st.error(f"Failed to place {msg}")

    # Optional: PNL visualization for the first selected row
    sel = edited[edited["select"]]
    if not sel.empty:
        st.markdown("#### PNL Projection")
        first = sel.iloc[0]
        # Try load model snapshot again for richer PNL curve (ok if None)
        try:
            snap = ensure_model(first.ticker)
            mdl = load_model(first.ticker, spot=snap.get("spot") if isinstance(snap, dict) else None, max_age_days=5) if load_model else None
        except Exception:
            mdl = None
        _render_pnl_over_time(first, mdl, fees_per_contract=float(first.fee_per_contract))

# ---------------- Cached backtest runner (compat shim) ----------------
@st.cache_data(show_spinner=False)
def _run_backtest_cached(tickers: List[str], start: str, end: str,
                         dte_days: int, wing: float, p_tail: float,
                         slip_bps: float, fee_per_contract: float) -> Dict[str, Any]:
    params = dict(
        strategy="both",
        dte_days=int(dte_days),
        step_days=5,
        p_tail=float(p_tail),
        wing_width=float(wing),
        slip_bps=float(slip_bps),
        fee_per_contract=float(fee_per_contract),
        risk_per_trade=0.01,
    )
    return backtest_universe(tickers, start, end, **params)

# ---------------- UI: Results ----------------
def tab_results(ledger_path: str, slippage_bps: float, fee_per_contract: float, demo_mode: bool) -> None:
    st.subheader("Results")

    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("##### Paper Ledger")
        df = _read_ledger_csv(ledger_path)
        if df.empty:
            st.info("No ledger entries yet.")
        else:
            st.dataframe(df.sort_values("ts", na_position="last"), use_container_width=True)
    with colB:
        st.markdown("##### Equity (Paper)")
        eq = _equity_from_ledger(ledger_path)
        if HAVE_PLOTLY and not eq.empty:
            fig = px.line(eq, x="date", y="equity", title="Paper Equity")
            st.plotly_chart(fig, use_container_width=True)
        elif not eq.empty:
            st.line_chart(eq.set_index("date")["equity"])

    st.divider()
    st.markdown("### Backtests (auto-run for current universe)")

    if demo_mode:
        st.info("Demo mode: backtests are disabled on Streamlit Cloud. Run locally for full backtests.")
        return

    st.markdown("### Backtests (auto-run for current universe)")
    uni_raw = st.session_state.get("scr_universe", "SPY")
    dte_days = int(st.session_state.get("scr_dte", 21))
    wing = float(st.session_state.get("scr_wing", 5.0))
    p_tail = float(st.session_state.get("scr_ptail", 0.20))

    tickers = _parse_universe(uni_raw)
    horizons = [("10y", *_horizon_to_dates("10y")),
                ("5y",  *_horizon_to_dates("5y")),
                ("1y",  *_horizon_to_dates("1y"))]

    for label, start, end in horizons:
        try:
            res = _run_backtest_cached(tickers, start, end, dte_days, wing, p_tail, slippage_bps, fee_per_contract)
        except Exception as e:
            st.warning(f"{label} backtest failed: {e}")
            continue

        equity = _ensure_equity_df(res.get("equity", pd.DataFrame()))
        summaries = res.get("summaries", pd.DataFrame())

        st.markdown(f"**{label.upper()}**  ({start} → {end})")
        cols = st.columns([2, 1])
        with cols[0]:
            if not isinstance(equity, pd.DataFrame):
                equity = _ensure_equity_df(equity)
            if "date" not in equity.columns or "equity" not in equity.columns:
                equity = _ensure_equity_df(equity)
            equity = equity.dropna(subset=["date"])

            if HAVE_PLOTLY and not equity.empty:
                try:
                    st.plotly_chart(px.line(equity, x="date", y="equity"), use_container_width=True)
                except Exception:
                    st.line_chart(equity.set_index("date")["equity"])
            elif not equity.empty:
                try:
                    st.line_chart(equity.set_index("date")["equity"])
                except Exception:
                    st.dataframe(equity, use_container_width=True)
            else:
                st.caption("No equity series available.")
        with cols[1]:
            if isinstance(summaries, pd.DataFrame) and not summaries.empty:
                s = summaries.sum(numeric_only=True) if "CAGR" not in summaries.columns else summaries
                cagr = float(s.get("CAGR", s.get("cagr", float("nan"))))
                sharpe = float(s.get("Sharpe", s.get("sharpe", float("nan"))))
                mdd = float(s.get("MaxDD", s.get("maxdd", float("nan"))))
                st.metric("CAGR", f"{cagr:.2%}" if pd.notna(cagr) else "—")
                st.metric("Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "—")
                st.metric("Max Drawdown", f"{mdd:.2%}" if pd.notna(mdd) else "—")
            else:
                st.caption("No summary stats available.")
        st.divider()

# ---------------- UI: Model Stats (precomputed JSONs) ----------------
def tab_model_stats_static() -> None:
    st.subheader("Model Stats — Precomputed (5y / 10y)")

    paths = [
        ("10y", PROJECT_ROOT / "docs" / "backtests" / "10y.json"),
        ("5y",  PROJECT_ROOT / "docs" / "backtests" / "5y.json"),
    ]
    for label, p in paths:
        if not p.exists():
            st.warning(f"{label}: stats file missing at {p}")
            continue
        try:
            payload = json.loads(p.read_text())
        except Exception as e:
            st.error(f"Failed to load {label}: {e}")
            continue

        meta = payload.get("meta", {})
        eq_list = payload.get("equity", [])
        eq = pd.DataFrame(eq_list)
        if "date" in eq.columns:
            eq["date"] = pd.to_datetime(eq["date"], errors="coerce")

        st.markdown(f"**{label.upper()}**  ({meta.get('start','?')} → {meta.get('end','?')})")
        cols = st.columns([2,1])
        with cols[0]:
            if HAVE_PLOTLY and not eq.empty:
                try:
                    st.plotly_chart(px.line(eq, x="date", y="equity", title="Equity"), use_container_width=True)
                except Exception:
                    st.line_chart(eq.set_index("date")["equity"])
            elif not eq.empty:
                st.line_chart(eq.set_index("date")["equity"])
            else:
                st.caption("No equity series.")

        with cols[1]:
            sm = payload.get("summaries", {})
            if isinstance(sm, dict) and sm:
                def _get(*keys):
                    for k in keys:
                        if k in sm: return sm[k]
                    return float("nan")
                cagr   = float(_get("CAGR","cagr"))
                sharpe = float(_get("Sharpe","sharpe"))
                mdd    = float(_get("MaxDD","maxdd"))
                st.metric("CAGR", f"{cagr:.2%}" if pd.notna(cagr) else "—")
                st.metric("Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "—")
                st.metric("Max Drawdown", f"{mdd:.2%}" if pd.notna(mdd) else "—")
            else:
                st.caption("No summary stats available.")
        with st.expander("Run configuration"):
            st.json(meta)
        st.divider()

# ---------------- UI: Research page ----------------
def tab_research() -> None:
    st.subheader("Research — Why Delta-Neutral Iron Condors?")
    st.markdown(
        """
**Thesis.** Options aren’t just for directional bets; they’re a way to sell *risk* you understand and can bound.  
An **iron condor** sells a put spread and a call spread simultaneously (delta-neutral around the short strikes), then **buys wings** to cap risk.  
You collect net **credit** up front; your profit is maximized when price stays between the short strikes, and losses are bounded by the wing width.

**Why it’s robust.**
- *Delta-neutral*: limited directional exposure; you harvest premium from time decay and implied vol.
- *Defined risk*: long wings bound tail risk; worst-case loss ≤ width − credit.
- *Universal*: works in bullish, bearish, or choppy regimes; your edge is selecting a sensible range + price.

**How this console helps.**
- Scans chains and proposes ATM-anchored candidates with sanity checks (credit ≤ width, wing symmetry, etc.).
- Lets you tweak slippage/fees/tolerances and view an interactive PNL diagram.
- Paper-trade locally (CSV ledger), mark-to-market with a safe fallback, and visualize equity.

**Notes on “edge.”**
- We display **edge% = credit / width**. Positive edge with defined risk is the core of the seller’s P&L.
- Where a pricing model is available, we compare builder vs. executable credits and (optionally) show a model-theoretical curve.

*Reading list (general, accessible intros)*:
- CBOE: Iron Condor strategy overview  
- Hull, **Options, Futures, and Other Derivatives** (delta-neutral hedging & spread payoffs)  
- Taleb, **Dynamic Hedging** (risk bounding, volatility selling caveats)  
*(Add your preferred links in README.)*
        """
    )

# ---------------- App entry ----------------
def run() -> None:
    st.set_page_config(page_title="AOE Options — Research Console", layout="wide")
    st.title("AOE Options — Research Console")

    tabs = st.tabs(["🔎 Screen", "📈 Results", "📊 Model Stats", "📚 Research"])
    ledger_path, slippage_bps, fee_per_contract, wing_tol, credit_leeway, demo_mode = sidebar_global()
    
    with tabs[0]:
        tab_screen(slippage_bps, fee_per_contract, wing_tol, credit_leeway, demo_mode)

    with tabs[1]:
        tab_results(ledger_path, slippage_bps, fee_per_contract, demo_mode)

    with tabs[2]:
        tab_model_stats_static()
    with tabs[3]:
        tab_research()

if __name__ == "__main__" and not _is_running_under_pytest():
    run()
