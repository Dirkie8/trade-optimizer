# Batch Strategy Optimization

## Overview

This document explains how to use the batch optimization system to test multiple trading strategies.

## 🚀 Quick Start - Which Batch Script Should I Use?

You have **THREE** batch optimization scripts:

| Script | Time | Quality | Best For |
|--------|------|---------|----------|
| `./optimize_all_strategies.sh` | ~1 hour | Medium ⚠️ | Quick tests |
| `./optimize_with_walk_forward.sh` | ~2 hours | High ✓ | Reliable params |
| `./optimize_hybrid_batch.sh` | ~3 hours | **Excellent ✓✓** | **Production use** |

**RECOMMENDED:** For best results, use `./optimize_hybrid_batch.sh` - it combines random exploration + gradient-based refinement + walk-forward validation!

## What Was Created

### 5 New Trading Strategies:

1. **RSIStrategy** (`functions/strategies/rsi_strategy.py`)
   - Simple RSI mean reversion strategy
   - Buy when RSI crosses below oversold, sell when crosses above overbought
   - Parameters: rsi_period, oversold, overbought, TP, SL

2. **BollingerBreakout** (`functions/strategies/bollinger_breakout.py`)
   - Breakout strategy using Bollinger Bands
   - Buy on upper band breakout, sell on lower band breakdown
   - Parameters: bb_period, bb_std_dev, TP, SL

3. **MACDMomentum** (`functions/strategies/macd_momentum.py`)
   - MACD momentum strategy
   - Buy when MACD crosses above signal, sell when crosses below
   - Parameters: macd_fast, macd_slow, macd_signal, TP, SL

4. **StochasticOscillator** (`functions/strategies/stochastic_oscillator.py`)
   - Stochastic oscillator strategy
   - Signals when %K crosses %D in extreme zones
   - Parameters: stoch_k_period, stoch_d_period, oversold, overbought, TP, SL

5. **MultiIndicatorConfluence** (`functions/strategies/multi_indicator_confluence.py`)
   - Complex multi-indicator strategy
   - Combines RSI, MACD, and Bollinger Bands
   - Requires multiple indicators to agree (confluence)
   - Parameters: RSI params, MACD params, BB params, confluence_required, TP, SL

### Config Files:

Each strategy has a YAML config in `functions/configs/`:
- `rsi_strategy.yaml`
- `bollinger_breakout.yaml`
- `macd_momentum.yaml`
- `stochastic_oscillator.yaml`
- `multi_indicator_confluence.yaml`

All configs have broad parameter ranges for exploration.

## Usage

### Quick Start (Random Search - Recommended)

Run all strategies with random search (faster, explores parameter space efficiently):

```bash
./optimize_all_strategies.sh --method random --n_random 50 --n_jobs 4
```

### Grid Search (Exhaustive but Slower)

Run all strategies with grid search (tests all combinations):

```bash
./optimize_all_strategies.sh --method grid --n_jobs 4
```

### Custom Settings

```bash
./optimize_all_strategies.sh \
    --method random \
    --n_random 100 \
    --n_jobs 8 \
    --main_config configs/main_config.yaml
```

## Parameter Counts

Approximate number of parameter combinations per strategy:

- **RSIStrategy**: 4×4×4×5×5 = 1,600 combinations
- **BollingerBreakout**: 4×4×5×5 = 400 combinations
- **MACDMomentum**: 3×3×3×5×5 = 675 combinations
- **StochasticOscillator**: 4×3×3×5×5 = 900 combinations
- **MultiIndicatorConfluence**: 2×2×2×2×2×2×2×2×2×4×4 = 32,768 combinations

**Total combinations across all strategies**: ~36,000+

### Recommendations:

- **Random search with n_random=50-100**: Fast, explores diverse parameter space
- **Random search with n_random=500-1000**: More thorough exploration
- **Grid search**: Only if you have lots of time (hours to days depending on data size)

## Results

Results are saved to strategy-specific directories:

```
results/
  RSIStrategy/
    optimizations/
      optimization_results.csv
      optimization_results_best.json
    evaluations/
      evaluation_results.csv
      evaluation_results_best.json
      eval_results.json
  BollingerBreakout/
    optimizations/...
    evaluations/...
  ...
```

## Analyzing Results

### View Best Parameters

```bash
cat results/RSIStrategy/optimizations/optimization_results_best.json
```

### Plot Results

```bash
python scripts/plot_results.py --input results/RSIStrategy/evaluations/eval_results.json
```

### Compare Strategies

```bash
# View best evaluation results for each strategy
for strategy in RSIStrategy BollingerBreakout MACDMomentum StochasticOscillator MultiIndicatorConfluence; do
    echo "=== $strategy ==="
    cat results/$strategy/evaluations/evaluation_results_best.json
    echo ""
done
```

## Script Output

The batch script provides:
- ✅ Colored progress indicators
- ⏱️ Timing for each strategy
- 📊 Summary statistics
- ✅/❌ Success/failure tracking
- 📁 Results directory listing

## Stopping the Script

Press `Ctrl+C` to stop the batch optimization. Completed strategies will have their results saved.

## Tips

1. **Start small**: Test with `--n_random 10` first to verify everything works
2. **Use parallel jobs**: Set `--n_jobs` to your CPU core count for speed
3. **Monitor progress**: The script shows which strategy is running and progress bars
4. **Check results incrementally**: Results are saved after each strategy completes
5. **Compare strategies**: Look at Sharpe ratio, total return, and max drawdown across strategies

## Example Workflow

```bash
# 1. Quick test (10 random samples per strategy)
./optimize_all_strategies.sh --method random --n_random 10 --n_jobs 4

# 2. If successful, run thorough optimization
./optimize_all_strategies.sh --method random --n_random 100 --n_jobs 8

# 3. Analyze results
python scripts/plot_results.py --input results/RSIStrategy/evaluations/eval_results.json
python scripts/plot_results.py --input results/BollingerBreakout/evaluations/eval_results.json
# ... etc for other strategies

# 4. Compare best parameters
grep -r "sharpe_ratio" results/*/evaluations/evaluation_results_best.json
```

## Troubleshooting

If a strategy fails:
1. Check the config file exists
2. Test the strategy individually:
   ```bash
   python scripts/optimize_strategy.py \
       --strategy_config functions/configs/rsi_strategy.yaml \
       --main_config configs/main_config.yaml \
       --method random \
       --n_random 5 \
       --n_jobs 1
   ```
3. Check for Python errors in the output

## Next Steps

After optimization:
1. Review best parameters for each strategy
2. Plot equity curves
3. Compare performance metrics
4. Select best-performing strategies
5. Consider ensemble approaches combining multiple strategies
