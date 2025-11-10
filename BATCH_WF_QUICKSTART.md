# Walk-Forward Batch Optimization - Quick Start

## Updated Script: `optimize_bayesian_wf_batch.sh`

The batch script has been updated to use the new walk-forward optimization approach.

## Current Configuration

**Data Settings** (from `configs/main_config.yaml`):
- Symbol: `frxEURUSD`
- Timeframe: `15m` ✓ (Good choice!)
- This gives you 4x more data points than 1h candles

## Usage

### Basic Run (Your Command):
```bash
./optimize_bayesian_wf_batch.sh \
  --n_trials 100 \
  --n_jobs 4 \
  --n_folds 8 \
  --validation_ratio 0.15 \
  --reward balanced
```

This will run Bayesian walk-forward optimization on ALL strategies that have a `parameters_bayesian` section.

### Other Examples:

**Quick Test (20 trials, 5 folds):**
```bash
./optimize_bayesian_wf_batch.sh --n_trials 20 --n_jobs 4 --n_folds 5
```

**Overnight Run (200 trials, 10 folds, 8 cores):**
```bash
nohup ./optimize_bayesian_wf_batch.sh \
  --n_trials 200 \
  --n_jobs 8 \
  --n_folds 10 \
  --validation_ratio 0.15 \
  > logs/wf_batch_$(date +%Y%m%d).log 2>&1 &
```

**Use Different Reward Metric:**
```bash
# For strategies emphasizing consistency over raw returns
./optimize_bayesian_wf_batch.sh --n_trials 100 --reward consistency

# For high Sharpe ratio strategies
./optimize_bayesian_wf_batch.sh --n_trials 100 --reward sharpe
```

## What It Does

For each strategy:
1. **Splits data**: 85% training, 15% hold-out validation
2. **Walk-forward on training**: Tests parameters across 8 time slices
3. **Bayesian optimization**: 100 trials to find best parameters
4. **Validation test**: Tests best parameters on unseen 15% hold-out data
5. **Saves results**: 
   - `bayesian_wf_optimization_results.csv` (all trials)
   - `bayesian_wf_optimization_results_best.json` (best params + metrics)
   - `bayesian_wf_optimization_results_validation.json` (validation details + equity curve)

## After Running

### 1. Compare All Strategies:
```bash
python scripts/compare_bayesian_results.py
```

This generates a comprehensive report showing:
- Top strategies by validation Sharpe
- Overfitting analysis (train vs validation)
- Robustness scores
- Recommended strategy

### 2. Review Specific Strategy:
```bash
# View validation metrics
cat results/RSIStrategy/optimizations/bayesian_wf_optimization_results_validation.json | python -m json.tool

# Plot equity curve
python scripts/plot_results.py \
  --input results/RSIStrategy/optimizations/bayesian_wf_optimization_results_validation.json \
  --show
```

### 3. Full Backtest on Winner:
```bash
# After identifying best strategy from comparison
./run_full_backtest.sh <BestStrategyName>
```

## Important Notes

### About Your RSIStrategy Test Results

From your 20-trial test run, the results showed:
- ⚠️ **Average train Sharpe: -0.06** (negative!)
- ⚠️ **Average train return: -0.69%** (losing money)
- ⚠️ **High variance across folds** (Sharpe std: 0.21)

**This means:**
- RSIStrategy with current parameter ranges doesn't have an edge on 15m data
- The positive validation result (+7%) was **luck** on that specific slice
- Walk-forward correctly identified lack of consistency

### Before Running Full Batch:

1. **Expand parameter ranges** for RSI strategy:
```yaml
# In functions/configs/rsi_strategy.yaml
parameters_bayesian:
  rsi_period: [5, 50]      # Was [10, 50], try lower periods for 15m
  oversold: [15, 40]       # Was [20, 40], try more extreme levels
  overbought: [60, 85]     # Was [60, 80], try more extreme levels
  take_profit_pips: [10, 100]  # Was [10, 50], try wider range
  stop_loss_pips: [5, 50]      # Was [10, 30], try tighter stops
```

2. **Focus on promising strategies first**:
```bash
# Test a few strategies manually before running full batch
python scripts/optimize_bayesian_walkforward.py --strategy ADXTrend --n_trials 20 --n_jobs 4 --n_folds 8
python scripts/optimize_bayesian_walkforward.py --strategy MACDMomentum --n_trials 20 --n_jobs 4 --n_folds 8
python scripts/optimize_bayesian_walkforward.py --strategy BollingerBreakout --n_trials 20 --n_jobs 4 --n_folds 8
```

3. **Check results before overnight run**:
- Look for strategies with positive train Sharpe (>0.3)
- Check that train and validation are consistent (<30% difference)
- Verify sufficient trades (>50 per fold)

## Estimated Runtime

With 15-minute data and your settings:
- **Per strategy**: ~15-30 minutes (100 trials, 4 cores, 8 folds)
- **24 strategies**: ~6-12 hours total
- **Recommendation**: Run overnight with `nohup`

## Expected Good Results

You'll know a strategy works when:
- ✓ Train Sharpe > 0.3
- ✓ Validation Sharpe > 0.2
- ✓ Overfitting < 30% (train vs validation)
- ✓ Sharpe std across folds < 0.15
- ✓ Consistent positive returns across folds
- ✓ Sufficient trades (>30 per fold)

Strategies not meeting these criteria should be:
- Re-optimized with expanded parameter ranges
- Enhanced with additional filters/conditions
- Replaced with alternative strategies
