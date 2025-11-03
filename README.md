# Trading Strategy Optimization Framework

## Overview
This repository lets you optimize and evaluate trading strategies on historical candlestick data. Strategies live under `functions/`, configs are YAML, and scripts under `scripts/` provide CLI entry points for data generation, optimization, evaluation, and plotting.

## Features
- Modular strategy interface (`BaseStrategy`) and example (`MovingAverageCross`).
- JSON OHLCV data loader with resampling helpers.
- Simple backtesting engine with SL/TP, spread, commission, and risk-based position sizing.
- Grid/random parameter search with CSV outputs.
- Evaluation runner that saves metrics and equity curve to JSON.
- Plot utility to visualize equity and key metrics.

## Repository Structure
- `data/` — Candlestick JSON files
- `functions/` — Strategy base classes and templates
  - `strategies/` — Individual strategy implementations
  - `configs/` — Strategy-specific YAML configs
- `configs/` — Global settings (account, broker, general)
- `scripts/` — CLI: fetch_data, optimize, evaluate, plot
- `results/` — Strategy-specific results (organized by strategy name)

## Data Retrieval

### Setting up HistData source files
Place your downloaded HistData zip files in the `data/raw/histdata/` directory. The system supports:
- Monthly files: `EURUSD_M1_202001.zip` or `EURUSD_M1_2020_01.zip`
- Annual packs: `HISTDATA_COM_MT_EURUSD_M12020.zip`
- Any zip containing CSV/TXT files with M1 OHLC data

### Fetching data
The `fetch_data.py` script extracts data from your HistData source files:

```bash
# Fetch 5 days of 1m data back from today
python scripts/fetch_data.py --symbol EURUSD --timeframe 1m --days 5

# Fetch specific date range with 5m timeframe  
python scripts/fetch_data.py --symbol EURUSD --timeframe 5m --startdate 2020-01-01 --enddate 2020-01-03

# Fetch 1h data (resampled from 1m source data)
python scripts/fetch_data.py --symbol EURUSD --timeframe 1h --startdate 2020-01-01 --enddate 2020-01-02
```

**Output format**: `data/candle_data_list_frxSYMBOL_SECONDSs.json` (e.g., `candle_data_list_frxEURUSD_3600s.json` for 1h data)

**Supported timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d (all resampled from 1m source data)

**Data availability**: The script will tell you exactly which months are missing if data is not available for your requested date range.

## Quickstart

### 1) Install dependencies (Python 3.10+)
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 2) Fetch historical data
```bash
# Place HistData zip files in data/raw/histdata/ first, then:
python scripts/fetch_data.py --symbol EURUSD --timeframe 5m --startdate 2020-01-01 --enddate 2020-01-10
```
This creates `data/candle_data_list_frxEURUSD_300s.json`.

### 3) Optimize parameters
```bash
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/example_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method grid
```
Outputs CSV at `results/optimizations/optimization_results.csv` and best params JSON beside it.

### 4) Evaluate best parameters
```bash
python scripts/evaluate_strategy.py \
  --strategy_config functions/configs/example_strategy.yaml \
  --main_config configs/main_config.yaml \
  --optimization_csv results/optimizations/optimization_results.csv
```
Writes JSON to `results/evaluations/eval_results.json`.

### 5) Plot results
```bash
python scripts/plot_results.py --input results/evaluations/eval_results.json
```

### 6) Visualize candlestick data
Open `scripts/candle_viz.ipynb` in VS Code or Jupyter to interactively plot any candlestick data from the `data/` folder with volume charts and statistics.

## Data Format
Output JSON files (`data/candle_data_list_frxSYMBOL_SECONDSs.json`) contain lists of records like:
```json
[
  {"Date": "2020-01-01 17:00:00+0000", "Open": 1.1212, "High": 1.12121, "Low": 1.12117, "Close": 1.1212, "Volume": 0.0}
]
```
- **Date**: UTC timezone-aware strings ending with `+0000`
- **OHLC**: Numeric price values from HistData source
- **Volume**: Usually 0.0 for forex data (HistData doesn't provide tick volume for most forex pairs)

## Strategy Interface
See `functions/base_strategy.py`. A strategy must implement `generate_signals()` and return `(action, stop_loss_pips, take_profit_pips)` where action ∈ {`HOLD`,`BUY`,`SELL`}.

## Notes
- **Volume data**: HistData forex files typically contain zero volume for all bars (this is normal - most retail forex data doesn't include tick volume). The volume column is preserved for compatibility with other data sources.
- **Forex pip rules**: Simplified for demonstration. Position sizing is abstracted (units sized by stop distance). Adjust for your broker if needed.
- **PyTorch**: Listed as optional and not required for core functionality.
- **Custom strategies**: Add your own under `functions/` with matching YAML configs under `functions/configs/`.
- **Interactive plotting**: Use `scripts/candle_viz.ipynb` for detailed candlestick analysis and visualization.

## Next Steps
- Add transaction cost models per-symbol.
- Add walk-forward evaluation and cross-validation.
- Integrate a real broker API (e.g., Exness demo) for live paper trading.
