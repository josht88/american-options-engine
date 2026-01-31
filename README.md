# AOE Options — Research Console

A local, privacy-first Streamlit app to **screen, sanity-check, and paper-trade** delta-neutral **iron condors** with **defined risk**.  
It also includes a page to display **precomputed 5y/10y backtest** snapshots and a write-up of the research rationale.

> **No accounts, no servers.** Everything runs locally; data stays on your machine. The optional paper ledger is a CSV file.

---

## Features

- **Daily Screening (UI):**
  - Universe input (e.g. `AAPL,MSFT,SPY,QQQ,NVDA`)
  - DTE and wing width controls
  - Sanity gates: credit ≤ width, wing symmetry tolerance, edge% filter
  - Executable vs. builder credit handling with slippage/fees
  - Click **Place Selected** to log paper fills (and try a broker adapter if present)
  - **Interactive PNL diagram** for selected candidate(s) (expiry payoff; optional mid-time curve when a model is available)

- **Results:**
  - Paper **ledger** (CSV) and a simple **equity** curve from that ledger
  - Auto backtests (1y/5y/10y) using the current screen settings (best-effort via cached runner)

- **Model Stats (Static):**
  - Load **precomputed** 5y/10y backtests from `docs/backtests/5y.json` and `10y.json`
  - Show equity lines and summary metrics (CAGR, Sharpe, Max DD)

- **Research Page:**
  - Rationale for delta-neutral, defined-risk premium selling
  - Notes on edge, risk, and model usage

---

## Install

```bash
git clone <your-repo-url> American_Options_Pricer_Model
cd American_Options_Pricer_Model
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
## Run the app (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Launch Streamlit UI
streamlit run web/app.py
```

### Notes
- The paper ledger defaults to `data/pnl.csv`. Override with `AOE_LEDGER=/path/to/file.csv`.
- This is a research / education project, not investment advice.

## Repo hygiene (recommended)
- Do **not** commit `.venv/`, `__pycache__/`, `.pytest_cache/`, `.hypothesis/`, or your live `data/pnl.csv`.
