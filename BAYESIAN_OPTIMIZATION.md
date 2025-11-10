# Bayesian Optimization Guide

## Overview
The new Bayesian optimization method uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to intelligently explore the parameter space. Unlike grid or random search, Bayesian optimization learns from previous trials to suggest better parameters.

## Key Features
- **Smart parameter exploration**: Uses past results to guide future trials
- **Custom reward metric**: Emphasizes consistency and stability over raw returns
- **Automatic validation**: Holds out 20% of data for final validation
- **Parallel execution**: Supports multi-core optimization
- **Continuous parameter spaces**: Explores full ranges, not just discrete values

## Reward Metric
The optimizer uses a custom reward function that balances:
- **Sharpe consistency**: Mean Sharpe ratio across 5 data chunks
- **Stability**: Low variance in Sharpe across chunks (1 - std_sharpe)
- **Drawdown penalty**: Penalizes excessive drawdowns (0.1 * max_drawdown)

```
reward = mean_sharpe × stability - 0.1 × max_drawdown
```

This encourages strategies that:
- Perform consistently across different market conditions
- Have stable risk-adjusted returns
- Avoid catastrophic drawdowns

## Configuration

### Strategy Config (YAML)
Each strategy config must have a `parameters_bayesian` section defining parameter ranges:

```yaml
parameters_bayesian:
  rsi_period: [9, 28]           # Search between 9 and 28
  oversold: [20, 35]            # Search between 20 and 35
  overbought: [65, 80]          # Search between 65 and 80
  take_profit_pips: [15, 50]
  stop_loss_pips: [10, 30]
```

**Important**: All parameters are integers. For float values (e.g., std_dev multipliers), use integers and divide by 10 in your strategy code:
```yaml
parameters_bayesian:
  bb_std_dev: [15, 30]  # Represents 1.5 to 3.0 when divided by 10
```

### Main Config
The `configs/main_config.yaml` has a new `validation_ratio` setting:

```yaml
general:
  validation_ratio: 0.2  # Hold out 20% for validation
```

## Usage

### Basic Usage
```bash
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --n_jobs 1
```

### With Parallel Execution
```bash
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 200 \
  --n_jobs 4
```

### More Trials for Complex Strategies
```bash
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/multi_indicator_confluence.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 500 \
  --n_jobs 8
```

### Custom Sort Metric
By default, results are sorted by `reward_metric`. You can change this:
```bash
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 100 \
  --sort-by sharpe
```

Available sort metrics:
- `reward_metric` (default)
- `sharpe`
- `sortino`
- `calmar`
- `consistency_score`
- `total_return_pct`

## Output Files

### CSV Results
All trials saved to: `results/<strategy>/optimizations/bayesian_optimization_results.csv`

Contains:
- All parameter combinations tested
- Full metrics for each trial (return, Sharpe, drawdown, trades, etc.)
- Custom reward metric for each trial

### Best Parameters JSON
Best parameters saved to: `results/<strategy>/optimizations/bayesian_optimization_results_best.json`

Format:
```json
{
  "method": "bayesian",
  "n_trials": 100,
  "sort_by": "reward_metric",
  "params": {
    "rsi_period": 14,
    "oversold": 28,
    "overbought": 72,
    "take_profit_pips": 35,
    "stop_loss_pips": 18
  },
  "metrics": {
    "reward_metric": 0.0523,
    "total_return_pct": 45.2,
    "sharpe": 1.85,
    "max_drawdown_pct": 12.3,
    "trades": 156
  }
}
```

### Validation Results
Validation on held-out data: `results/<strategy>/optimizations/bayesian_optimization_results_validation.json`

Format:
```json
{
  "params": { ... },
  "train_metrics": { ... },
  "validation_metrics": {
    "reward_metric": 0.0498,
    "total_return_pct": 42.1,
    "sharpe": 1.72,
    "trades": 38
  },
  "equity_curve": [...]
}
```

## Workflow Comparison

### Grid Search
- Exhaustive: tests all combinations
- Slow: exponential growth with parameters
- No learning: treats all trials independently
- Best for: small parameter spaces, final verification

### Random Search
- Faster than grid for large spaces
- No learning: treats all trials independently
- Good baseline: often beats grid in practice
- Best for: initial exploration

### Bayesian Optimization (New)
- Smart: learns from past trials
- Efficient: finds good parameters with fewer trials
- Adaptive: focuses on promising regions
- Best for: complex strategies, limited compute budget

## Tips

### Number of Trials
- Simple strategies (2-4 parameters): 50-100 trials
- Medium strategies (5-7 parameters): 100-300 trials
- Complex strategies (8+ parameters): 300-1000 trials

### Parallel Jobs
- Set to number of CPU cores minus 1 for best performance
- Use `--n_jobs 1` for debugging (easier to read output)
- Use `--n_jobs -1` to auto-detect optimal workers

### Parameter Ranges
- Start with wide ranges to explore
- Narrow ranges after initial runs if patterns emerge
- Ensure ranges include known good values from other methods

## Example Workflows

### Initial Exploration
```bash
# 1. Run Bayesian optimization
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 200 \
  --n_jobs 4

# 2. Check validation results
cat results/RSIStrategy/optimizations/bayesian_optimization_results_validation.json

# 3. If validation looks good, run on full dataset
python scripts/evaluate_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml
```

### Comparison with Other Methods
```bash
# 1. Random search baseline
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method random \
  --n_random 200

# 2. Bayesian optimization
python scripts/optimize_bayesian.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --n_trials 200

# 3. Compare results
python scripts/compare_strategies.py
```

## Troubleshooting

### "No parameters_bayesian section found"
Add `parameters_bayesian` to your strategy config YAML:
```yaml
parameters_bayesian:
  param1: [min, max]
  param2: [min, max]
```

### Validation performance much worse than training
- Likely overfitting
- Try fewer trials or wider parameter ranges
- Check if training period is too short
- Consider walk-forward validation (use original optimizer with `--walk-forward`)

### All trials have negative reward
- Strategy may not be profitable on this data
- Try different parameter ranges
- Check strategy implementation for bugs
- Review reward metric to ensure it aligns with goals

### Optimization too slow
- Reduce `--n_trials`
- Increase `--n_jobs` for parallel execution
- Use shorter data period in `main_config.yaml`
- Profile strategy code for bottlenecks
