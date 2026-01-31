# AOE Options — Research Console

A **local-first Streamlit research app** to screen, sanity-check, and paper-trade
**delta-neutral iron condors** with **defined risk**.

The project also includes:
• a paper ledger & equity view  
• optional local backtests  
• static summaries of long-horizon results (5y / 10y)

> This is a research / education project. Not investment advice.

---

## Features

### 🔎 Daily Screening
- Comma-separated universe input (e.g. `AAPL,MSFT,SPY,QQQ,NVDA`)
- DTE and wing-width controls
- Sanity gates:
  - credit ≤ width
  - wing symmetry tolerance
  - minimum edge %
- Builder vs executable credit with slippage & fees
- Paper trade logging (CSV ledger)
- Interactive **PNL diagrams** (expiry payoff; optional model-based curve)

### 📈 Results
- Paper ledger viewer
- Equity curve derived from the ledger
- Optional local backtests (1y / 5y / 10y when run off-cloud)

### 📊 Model Stats (Static)
- Displays **precomputed placeholders** for 5y / 10y backtests
- Full datasets intentionally excluded due to size limits
- Files live in `docs/backtests/`

### 📚 Research
- Rationale for delta-neutral premium selling
- Defined-risk framing
- Notes on execution realism and volatility regimes

---

## Running Locally

```bash
git clone <your-repo-url> American_Options_Pricer_Model
cd American_Options_Pricer_Model

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run web/app.py
